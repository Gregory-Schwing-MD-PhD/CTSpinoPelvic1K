"""measure_pelvic_width.py -- the demo study: is the female pelvis relatively wider?

Run from the repository root, pointing at the dataset folder (the one holding labels/ and
manifest.json):

    python examples/pelvic_width_dimorphism/measure_pelvic_width.py \
        --data /path/to/CTSpinoPelvic1K --out examples/pelvic_width_dimorphism/results

On the grid, through SLURM (see templates/slurm_job.sh):

    mkdir -p logs
    sbatch templates/slurm_job.sh examples/pelvic_width_dimorphism/measure_pelvic_width.py \
        --data ~/data/CTSpinoPelvic1K --out examples/pelvic_width_dimorphism/results --workers 8

Start with --limit 10 while developing.

TWO MEASUREMENTS, AND WHY THERE ARE TWO. The first thing anyone tries is the width of the
pelvis itself, the left-right extent of the two hip bones. It is larger in males, because
males are larger, and dividing it by the sacrum does not rescue it. The measurement the
anatomy literature actually describes is different: how far apart the hip joints sit on a
skeleton of a given size. So the script measures both and lets you see the contrast.

  bi-iliac width         left-right extent of hip labels 30 + 31, mm (the naive try)
  femoral head distance  distance between the centres of the two femoral heads, mm; each
                         head centre is the centroid of the medial-most 28 mm of the top
                         30 mm of the femur label (32 left, 33 right), which is the ball of
                         the joint and not the greater trochanter beside it
  vertebral body width   left-right extent of the anterior 18 mm of the L4 label
                         (identifier 23), which is the vertebral body and not the transverse
                         processes or facets behind it, mm; L5 (24) when L4 is absent
  outer width            the same line extended outward through each femur mask to its
                         last voxel: the lateral edge of each joint on the bicoxofemoral axis
  relative separation    femoral head distance / vertebral body width, dimensionless

Axes are found from the affine's axis codes, never assumed, so prone and supine records are
measured identically. Sex comes from manifest.json. Records with surgical hardware are
excluded from the comparison (eight carry hip prostheses).

WHAT IT WRITES: results/pelvic_width.csv (one row per record), results/report.txt (the
numbers, with Welch t tests and Cohen's d) and results/pelvic_width.png. Every number in
the report comes from the CSV, and the figure is drawn from the same rows.
"""
from __future__ import annotations

import argparse
import csv
import json
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

HIPS = (30, 31)          # left_hip, right_hip
FEMUR_L, FEMUR_R = 32, 33
L4, L5 = 23, 24
HEAD_SLAB_MM = 30.0      # the top of the femur that holds the head and neck
HEAD_MEDIAL_MM = 28.0    # of that slab, the medial-most part: the head, not the trochanter
BODY_SLAB_MM = 18.0      # the anterior part of the vertebra: body only, no processes


def axes(img):
    """Axis indices and directions, read from the affine, never assumed.
    Returns lr, si, ap and the signs right (+1 if a larger lr index is the patient's right),
    up (+1 if a larger si index is superior) and ant (+1 if a larger ap index is anterior)."""
    codes = nib.aff2axcodes(img.affine)
    lr = [i for i, c in enumerate(codes) if c in "LR"][0]
    si = [i for i, c in enumerate(codes) if c in "SI"][0]
    ap = [i for i, c in enumerate(codes) if c in "AP"][0]
    return (lr, si, ap, 1 if codes[lr] == "R" else -1,
            1 if codes[si] == "S" else -1, 1 if codes[ap] == "A" else -1)


def extent_mm(mask, axis, mm):
    others = tuple(k for k in range(3) if k != axis)
    idx = np.nonzero(mask.any(axis=others))[0]
    return float((idx.max() - idx.min() + 1) * mm) if idx.size else float("nan")


def keep_extreme(m, axis, sign, depth_mm, zoom):
    """The part of mask m within depth_mm of its extreme along axis, in direction sign."""
    idx = np.nonzero(m.any(axis=tuple(k for k in range(3) if k != axis)))[0]
    if not idx.size:
        return m
    n = int(round(depth_mm / zoom))
    keep = np.zeros(m.shape[axis], bool)
    if sign > 0:
        keep[max(0, idx.max() - n + 1):idx.max() + 1] = True
    else:
        keep[idx.min():idx.min() + n] = True
    m2 = np.moveaxis(m, axis, 0).copy()
    m2[~keep] = False
    return np.moveaxis(m2, 0, axis)


def head_centre(lab, fid, lr, si, right, up, zooms, affine):
    """World-space centroid of the femoral head. The greater trochanter sits at the same
    height as the head but lateral to it, so the top slab is restricted to its medial part."""
    m = lab == fid
    if not m.any():
        return None
    m = keep_extreme(m, si, up, HEAD_SLAB_MM, zooms[si])
    medial = right if fid == FEMUR_L else -right      # the left femur's medial side is toward the patient's right
    m = keep_extreme(m, lr, medial, HEAD_MEDIAL_MM, zooms[lr])
    ijk = np.array(np.nonzero(m), float).mean(axis=1)
    return (affine @ np.append(ijk, 1.0))[:3]


def body_width(lab, lr, ap, ant, zooms):
    """Left-right extent of the anterior BODY_SLAB_MM of L4 (else L5). The body is the most
    anterior part of a vertebra; the transverse processes and facets lie behind it, so a
    depth restriction excludes them and lordotic tilt does not matter."""
    for vid in (L4, L5):
        m = lab == vid
        if not m.any():
            continue
        m = keep_extreme(m, ap, ant, BODY_SLAB_MM, zooms[ap])
        return extent_mm(m, lr, zooms[lr]), vid
    return float("nan"), None


def outer_point(lab, fid, start, direction, affine, step_mm=0.5, max_mm=120.0):
    """Walk from a head centre along the bicoxofemoral axis, outward, and return the last
    world point still inside that femur's mask: the lateral edge of the joint on that line."""
    inv = np.linalg.inv(affine)
    last = start
    for k in range(1, int(max_mm / step_mm)):
        pt = start + direction * (k * step_mm)
        ijk = np.rint(inv @ np.append(pt, 1.0))[:3].astype(int)
        if np.any(ijk < 0) or np.any(ijk >= np.array(lab.shape)):
            break
        if lab[tuple(ijk)] == fid:
            last = pt
        elif k * step_mm > 8.0:          # allow a small gap near the centre, then stop at exit
            break
    return last


def measure(path: str):
    img = nib.load(path)
    lab = np.asanyarray(img.dataobj)                  # never reorient
    zooms = img.header.get_zooms()[:3]
    lr, si, ap, right, up, ant = axes(img)
    case = Path(path).name.split("_")[0]
    biiliac = extent_mm(np.isin(lab, HIPS), lr, zooms[lr])
    cl = head_centre(lab, FEMUR_L, lr, si, right, up, zooms, img.affine)
    cr = head_centre(lab, FEMUR_R, lr, si, right, up, zooms, img.affine)
    if cl is not None and cr is not None:
        heads = float(np.linalg.norm(cl - cr))
        u = (cr - cl) / heads                          # the bicoxofemoral axis, left to right
        # extend that line outward through each femur to the edge of its mask
        outer = float(np.linalg.norm(outer_point(lab, FEMUR_R, cr, u, img.affine) - outer_point(lab, FEMUR_L, cl, -u, img.affine)))
    else:
        heads = outer = float("nan")
    width, vid = body_width(lab, lr, ap, ant, zooms)
    return case, biiliac, heads, outer, width, vid


def main() -> int:
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--data", required=True, help="folder holding labels/ and manifest.json")
    ap_.add_argument("--out", default="examples/pelvic_width_dimorphism/results")
    ap_.add_argument("--workers", type=int, default=8)
    ap_.add_argument("--limit", type=int, default=0, help="measure only the first N records (development)")
    ap_.add_argument("--from-csv", action="store_true",
                     help="skip the measurement and rebuild the report and figure from results/pelvic_width.csv")
    a = ap_.parse_args()
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    man = {r["label_file"].split("/")[-1].split("_")[0]: r
           for r in json.load(open(data / "manifest.json", encoding="utf-8"))}
    files = sorted(str(p) for p in (data / "labels").glob("*_label.nii.gz"))
    if a.limit:
        files = files[:a.limit]
    print(f"{len(files)} label volumes")

    rows = []
    if a.from_csv:
        # the measurement is the slow part; the comparison and the figure are seconds
        with open(out / "pelvic_width.csv", newline="") as fh:
            for r in csv.DictReader(fh):
                for k in ("biiliac_mm", "femoral_head_distance_mm", "bicoxofemoral_outer_width_mm",
                          "body_width_mm", "relative_separation"):
                    r[k] = float(r[k]) if r[k] not in ("", "nan") else float("nan")
                r["hardware"] = r["hardware"] == "True"
                rows.append(r)
        print(f"{len(rows)} rows read from {out / 'pelvic_width.csv'}")
    with Pool(a.workers if not a.from_csv else 1) as pool:
        if a.from_csv:
            files = []
        for k, (case, biiliac, heads, outer, width, vid) in enumerate(pool.imap_unordered(measure, files), 1):
            r = man.get(case, {})
            rel = heads / width if width == width and width > 0 else float("nan")
            rows.append({"case": case, "sex": r.get("sex") or "missing", "age": r.get("age"),
                         "position": r.get("position"), "hardware": bool(r.get("hardware_labelled")),
                         "biiliac_mm": round(biiliac, 1), "femoral_head_distance_mm": round(heads, 1),
                         "bicoxofemoral_outer_width_mm": round(outer, 1),
                         "body_width_mm": round(width, 1), "body_level": {L4: "L4", L5: "L5"}.get(vid, ""),
                         "relative_separation": round(rel, 3)})
            if k % 100 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    rows.sort(key=lambda r: r["case"])
    with open(out / "pelvic_width.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- the comparison --------------------------------------------------------------
    from scipy import stats
    keep = [r for r in rows if not r["hardware"] and r["relative_separation"] == r["relative_separation"]]
    grp = {s: [r for r in keep if r["sex"] == s] for s in ("female", "male")}

    def arr(key, s):
        return np.array([r[key] for r in grp[s] if r[key] == r[key]])

    def line(key, label, unit):
        f, m = arr(key, "female"), arr(key, "male")
        t, p = stats.ttest_ind(f, m, equal_var=False)
        d = (f.mean() - m.mean()) / np.sqrt((f.var(ddof=1) + m.var(ddof=1)) / 2)
        return (f"{label:28s} female {f.mean():7.2f} +/- {f.std(ddof=1):5.2f} (median {np.median(f):6.2f})   "
                f"male {m.mean():7.2f} +/- {m.std(ddof=1):5.2f} (median {np.median(m):6.2f}) {unit}   "
                f"Welch t = {t:6.2f}, p = {p:.1e}, d = {d:+.2f}")

    lines = [f"records measured: {len(rows)}; analysed (no hardware, both femoral heads and a body width present): "
             f"{len(keep)}; female {len(grp['female'])}, male {len(grp['male'])}",
             line("biiliac_mm", "bi-iliac width", "mm"),
             line("femoral_head_distance_mm", "femoral head distance", "mm"),
             line("bicoxofemoral_outer_width_mm", "outer width on that axis", "mm"),
             line("body_width_mm", "vertebral body width (L4)", "mm"),
             line("relative_separation", "head distance / body width", "")]
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- the figure --------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes_ = plt.subplots(1, 3, figsize=(9.6, 3.0))
    rng = np.random.default_rng(0)
    panels = [("biiliac_mm", "bi-iliac width (mm)", "(a) Pelvis width, absolute"),
              ("femoral_head_distance_mm", "femoral head distance (mm)", "(b) Hip joint separation, absolute"),
              ("relative_separation", "head distance / L4 body width", "(c) Relative to skeletal size")]
    for ax, (key, ylabel, title) in zip(axes_, panels):
        data_ = [list(arr(key, s)) for s in ("female", "male")]
        ax.boxplot(data_, tick_labels=[f"female (n={len(data_[0])})", f"male (n={len(data_[1])})"],
                   widths=0.5, showfliers=False)
        for i, dd in enumerate(data_, 1):
            ax.scatter(rng.normal(i, 0.06, len(dd)), dd, s=4, alpha=0.25, color="#1c6b73")
        ax.set_ylabel(ylabel); ax.set_title(title, loc="left", fontsize=9)
        ax.grid(axis="y", color="#CCCCCC", lw=0.5); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(out / "pelvic_width.png", dpi=200)
    print("wrote", out / "pelvic_width.csv", out / "report.txt", out / "pelvic_width.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
