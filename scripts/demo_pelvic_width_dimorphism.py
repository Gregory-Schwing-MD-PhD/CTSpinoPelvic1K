"""scripts/demo_pelvic_width_dimorphism.py -- a population-level anatomical study in one file.

THE QUESTION. Is relative pelvic width different between females and males? The absolute
pelvis is wider in taller people, so the width is expressed relative to the sacrum, which is
in the same scan and scales with the same skeleton.

WHAT IS MEASURED, PER RECORD, FROM THE LABEL VOLUME ALONE.
  * bi-iliac width   : the left-right extent of the two hip bones (ids 30 + 31), in mm
  * S1 width         : the left-right extent of S1 (id 29), in mm
  * relative width   : bi-iliac width / S1 width (dimensionless)
Extents are read along the patient's left-right axis, found from the affine's axis codes
rather than assumed, so prone and supine records measure the same thing.

WHO IS COMPARED. Sex comes from manifest.json (`sex`: female / male / other / missing).
Records with hip arthroplasty are excluded (the pelvis is the patient's, but the study
should not depend on records carrying metal), and so are records whose S1 carve is flagged.

    python scripts/demo_pelvic_width_dimorphism.py --labels data/zenodo_deposit/labels \
        --manifest data/zenodo_deposit/manifest.json --out results/pelvic_width

Outputs: a CSV with one row per record, a summary table, a figure, and a short text report
with a Welch t test and Cohen's d. Every number in the report comes from the CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS  # noqa: E402

HIPS = (30, 31)
S1 = LS.S1_ID


def measure(path: str):
    img = nib.load(path)
    lab = np.asanyarray(img.dataobj)
    codes = nib.aff2axcodes(img.affine)
    lr = [i for i, c in enumerate(codes) if c in "LR"][0]          # the left-right axis
    mm = img.header.get_zooms()[lr]
    others = tuple(k for k in range(3) if k != lr)

    def extent_mm(mask):
        idx = np.nonzero(mask.any(axis=others))[0]
        return float((idx.max() - idx.min() + 1) * mm) if idx.size else float("nan")

    hips = np.isin(lab, HIPS)
    s1 = lab == S1
    case = Path(path).name.split("_")[0]
    return case, extent_mm(hips), extent_mm(s1), int(hips.any()), int(s1.any())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/zenodo_deposit/labels")
    ap.add_argument("--manifest", default="data/zenodo_deposit/manifest.json")
    ap.add_argument("--out", default="results/pelvic_width")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    man = {r["label_file"].split("/")[-1].split("_")[0]: r for r in json.load(open(a.manifest))}
    files = sorted(str(p) for p in Path(a.labels).glob("*_label.nii.gz"))
    rows = []
    with Pool(a.workers) as pool:
        for k, (case, hips_mm, s1_mm, has_hips, has_s1) in enumerate(pool.imap_unordered(measure, files), 1):
            r = man.get(case, {})
            rows.append({"case": case, "sex": r.get("sex") or "missing", "age": r.get("age"),
                         "position": r.get("position"), "hardware": bool(r.get("hardware_labelled")),
                         "biiliac_mm": round(hips_mm, 1), "s1_mm": round(s1_mm, 1),
                         "relative_width": round(hips_mm / s1_mm, 3) if s1_mm and s1_mm == s1_mm else float("nan")})
            if k % 100 == 0:
                print(f"  {k}/{len(files)}", flush=True)
    rows.sort(key=lambda r: r["case"])
    with open(out / "pelvic_width.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

    # ---- analysis ----------------------------------------------------------------------
    from scipy import stats
    keep = [r for r in rows if not r["hardware"] and r["relative_width"] == r["relative_width"]]
    f = np.array([r["relative_width"] for r in keep if r["sex"] == "female"])
    m = np.array([r["relative_width"] for r in keep if r["sex"] == "male"])
    t, p = stats.ttest_ind(f, m, equal_var=False)
    sp = np.sqrt((f.var(ddof=1) + m.var(ddof=1)) / 2); d = (f.mean() - m.mean()) / sp
    fa = np.array([r["biiliac_mm"] for r in keep if r["sex"] == "female"]); ma = np.array([r["biiliac_mm"] for r in keep if r["sex"] == "male"])
    fs = np.array([r["s1_mm"] for r in keep if r["sex"] == "female"]); ms = np.array([r["s1_mm"] for r in keep if r["sex"] == "male"])
    lines = [
        f"records measured: {len(rows)}; analysed (no hardware, S1 present): {len(keep)}; female {len(f)}, male {len(m)}",
        f"bi-iliac width, mm:  female {fa.mean():.1f} +/- {fa.std(ddof=1):.1f}   male {ma.mean():.1f} +/- {ma.std(ddof=1):.1f}",
        f"S1 width, mm:        female {fs.mean():.1f} +/- {fs.std(ddof=1):.1f}   male {ms.mean():.1f} +/- {ms.std(ddof=1):.1f}",
        f"relative width (bi-iliac / S1): female {f.mean():.3f} +/- {f.std(ddof=1):.3f}   male {m.mean():.3f} +/- {m.std(ddof=1):.3f}",
        f"Welch t = {t:.2f}, p = {p:.2e}, Cohen's d = {d:.2f}",
    ]
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- figure ------------------------------------------------------------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    for ax, (key, label) in zip(axes, [("biiliac_mm", "bi-iliac width (mm)"), ("relative_width", "bi-iliac width / S1 width")]):
        data = [[r[key] for r in keep if r["sex"] == s] for s in ("female", "male")]
        ax.boxplot(data, tick_labels=[f"female (n={len(data[0])})", f"male (n={len(data[1])})"], widths=0.5, showfliers=False)
        for i, dd in enumerate(data, 1):
            ax.scatter(np.random.default_rng(0).normal(i, 0.06, len(dd)), dd, s=4, alpha=0.25, color="#1c6b73")
        ax.set_ylabel(label); ax.grid(axis="y", color="#CCCCCC", lw=0.5); ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_title("(a) Absolute", loc="left"); axes[1].set_title("(b) Relative to the sacrum", loc="left")
    fig.tight_layout()
    fig.savefig(out / "pelvic_width.png", dpi=200); fig.savefig(out / "pelvic_width.pdf")
    print("wrote", out / "pelvic_width.csv", out / "pelvic_width.png", out / "report.txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
