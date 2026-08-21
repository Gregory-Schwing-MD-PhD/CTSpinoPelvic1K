"""scripts/extract_surgical_morphometrics.py — the measures that drive operative decisions.

Chosen from what the surgical literature actually decides on, and restricted to what BONE
labels alone can support on a NON-CONTRAST supine CT. Vessels are excluded deliberately:
the iliac confluence position would be the ideal LSTV planning variable, but CT
colonography is unenhanced and the veins have almost no contrast against retroperitoneal
fat.

WHAT SUPINE ACQUISITION PERMITS, AND WHAT IT DOES NOT

  PELVIC INCIDENCE is position-INDEPENDENT -- identical standing, seated and supine,
  because it is a morphological property of the pelvis rather than a posture. So PI from a
  supine CT needs no caveat, and PI is the parameter that dictates how much lordosis a
  given spine requires.

  LUMBAR LORDOSIS is position-dependent, but supine sits only ~4.6 deg below standing
  (50.2 vs 54.8 in published series); it is SEATED that collapses, to ~16. So supine LL is
  reported explicitly AS supine, with the offset stated, rather than dropped or silently
  passed off as standing.

  PELVIC TILT and SACRAL SLOPE are postural and are reported supine for the same reason.

THE MEASURES

  pelvic_incidence_deg   angle at the S1 endplate midpoint between its normal and the line
                         to the bicoxofemoral axis. The decision-maker: PI-LL mismatch
                         beyond ~10 deg predicts residual pain after fusion, including
                         short-segment.
  ll_supine_deg          topmost-lumbar superior to S1 superior endplate angle. Reported
                         with ll_top_vertebra and ll_complete, because these are spine-
                         limited scans: when L1 is clipped the arc starts lower and the
                         angle is smaller than the patient's.
  pi_ll_mismatch_deg     the two together, supine -- computed ONLY where the arc reaches
                         L1, since a truncated lordosis inflates the mismatch.

  crest_height_mm        how far the iliac crest rises ABOVE the L4-5 disc. The lateral
                         corridor: a high crest blocks an L4-5 lateral approach outright,
                         and a published cutoff of 12mm marks rising subsidence risk after
                         oblique lateral fusion at that level.
  rib12_to_crest_mm      the other boundary of the same corridor. This is where the rib
                         work becomes operative: a hypoplastic or absent twelfth rib moves
                         the upper limit of the working window.

  pedicle_width_mm       narrowest transverse width, per level -- it selects screw
                         diameter, and is itself a phenotype: wider at every lumbar level
                         in degenerative stenosis.
  canal_ap_mm            sagittal canal diameter, the decisive measure in degenerative
                         stenosis, with the Torg ratio against body depth.
  wedge_ratio            anterior over posterior body height; compression-fracture
                         screening, opportunistic.

    python scripts/extract_surgical_morphometrics.py --labels data/v5_final --workers 16
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

THORACIC_BASE = 7
LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}
SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R = 26, 29, 30, 31, 32, 33
RIB_L, RIB_R = 33, 45
MIN_VOX = 3000


def _com(mask, sp):
    idx = np.argwhere(mask)
    return (idx.mean(0) * sp) if len(idx) else None


def _endplate(mask, sp, superior=True):
    """Centroid and normal of a vertebral endplate.

    The top (or bottom) decile of the body by height, fitted with a plane. A decile rather
    than a single slice because one slice of a curved endplate is noise, and rather than the
    whole body because the body's own principal axis is not the endplate normal.
    """
    idx = np.argwhere(mask)
    if len(idx) < 50:
        return None, None
    z = idx[:, 2]
    cut = np.percentile(z, 90 if superior else 10)
    sel = idx[z >= cut] if superior else idx[z <= cut]
    if len(sel) < 20:
        return None, None
    pts = sel * sp
    c = pts.mean(0)
    q = pts - c
    # smallest singular direction is the plate normal
    n = np.linalg.svd(q, full_matrices=False)[2][-1]
    if n[2] < 0:
        n = -n
    return c, n


def _angle(u, v):
    cu = u / (np.linalg.norm(u) + 1e-9)
    cv = v / (np.linalg.norm(v) + 1e-9)
    return float(np.degrees(np.arccos(np.clip(cu @ cv, -1, 1))))


def one(path: str) -> dict:
    stem = Path(path).name.replace("_label.nii.gz", "")
    r = {"case": stem}
    try:
        img = nib.as_closest_canonical(nib.load(path))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        sp = np.array(img.header.get_zooms()[:3], float)
    except Exception as exc:                                        # noqa: BLE001
        return {"case": stem, "error": type(exc).__name__}

    have = {v: (lab == v) for v in list(LUMBAR) + [SACRUM, S1, HIP_L, HIP_R, FEM_L, FEM_R]
            if (lab == v).sum() >= MIN_VOX}

    # ---- pelvic incidence (position independent) ---------------------------------
    fem = None
    if FEM_L in have and FEM_R in have:
        cl, cr = _com(have[FEM_L], sp), _com(have[FEM_R], sp)
        fem = (cl + cr) / 2
    elif HIP_L in have and HIP_R in have:
        # fall back to the acetabular centroids; less exact than femoral heads but the
        # bicoxofemoral axis is a midpoint either way
        cl, cr = _com(have[HIP_L], sp), _com(have[HIP_R], sp)
        fem = (cl + cr) / 2
    s1c, s1n = (_endplate(have[S1], sp, True) if S1 in have
                else (None, None))
    if s1c is None and SACRUM in have:
        s1c, s1n = _endplate(have[SACRUM], sp, True)
    if fem is not None and s1c is not None and s1n is not None:
        v = fem - s1c
        # PI is measured in the sagittal plane: drop the left-right component
        vs = np.array([0.0, v[1], v[2]])
        ns = np.array([0.0, s1n[1], s1n[2]])
        if np.linalg.norm(vs) > 1 and np.linalg.norm(ns) > 1e-6:
            r["pelvic_incidence_deg"] = round(_angle(ns, vs), 1)
            # sacral slope: S1 endplate against the axial plane
            r["sacral_slope_deg"] = round(90.0 - _angle(ns, np.array([0.0, 0.0, 1.0])), 1)
            if "pelvic_incidence_deg" in r:
                r["pelvic_tilt_deg"] = round(r["pelvic_incidence_deg"]
                                             - r["sacral_slope_deg"], 1)

    # ---- lumbar lordosis, SUPINE -------------------------------------------------
    # WHAT THE TOP OF THE ARC ACTUALLY IS. Lordosis is conventionally L1 superior to S1
    # superior, but lums[0] is the topmost lumbar IN THE FIELD OF VIEW, and these are
    # spine-limited scans. If L1 is clipped the arc starts at L2 and the angle comes back
    # smaller than the patient's -- which then propagates straight into PI-LL mismatch,
    # the one number here a surgeon would act on. So record where the arc really started
    # and how many bodies it spanned, and let the reader gate on it.
    lums = sorted(v for v in have if v in LUMBAR)
    if lums and s1n is not None:
        c1, n1 = _endplate(have[lums[0]], sp, True)
        if n1 is not None:
            r["ll_supine_deg"] = round(_angle(n1, s1n), 1)
            r["ll_top_vertebra"] = LUMBAR.get(lums[0])
            r["ll_levels_spanned"] = len(lums)
            # A six-lumbar spine spans six and still starts at L1, so the test is where
            # the arc STARTS, not how many bodies are under it.
            r["ll_complete"] = int(LUMBAR.get(lums[0]) == "L1")
            if "pelvic_incidence_deg" in r and r["ll_complete"]:
                r["pi_ll_mismatch_deg"] = round(r["pelvic_incidence_deg"]
                                                - r["ll_supine_deg"], 1)

    # ---- the lateral corridor ----------------------------------------------------
    crest = None
    if HIP_L in have or HIP_R in have:
        hips = np.zeros_like(lab, bool)
        for h in (HIP_L, HIP_R):
            if h in have:
                hips |= have[h]
        crest = float(np.nonzero(hips.any(axis=(0, 1)))[0].max()) * sp[2]
        r["crest_z_mm"] = round(crest, 1)
    # L4-5 disc level = midpoint between the two bodies
    l4 = next((v for v in have if LUMBAR.get(v) == "L4"), None)
    l5 = next((v for v in have if LUMBAR.get(v) == "L5"), None)
    if crest is not None and l4 and l5:
        z4 = np.nonzero(have[l4].any(axis=(0, 1)))[0]
        z5 = np.nonzero(have[l5].any(axis=(0, 1)))[0]
        disc = (z4.min() + z5.max()) / 2 * sp[2]
        # POSITIVE means the crest rises above the L4-5 disc and obstructs a lateral
        # approach; the published subsidence cutoff is +12mm
        r["crest_above_l45_mm"] = round(crest - disc, 1)
        r["crest_blocks_l45"] = int((crest - disc) >= 12.0)
    # the other boundary: lowest rib to crest
    ribs = np.zeros_like(lab, bool)
    lowest = None
    for base in (RIB_L, RIB_R):
        for n in range(1, 13):
            m = lab == base + n
            if m.sum() > 200:
                ribs |= m
                z = np.nonzero(m.any(axis=(0, 1)))[0].min() * sp[2]
                lowest = z if lowest is None else min(lowest, z)
    if crest is not None and lowest is not None:
        r["rib12_to_crest_mm"] = round(lowest - crest, 1)

    # ---- per-level pedicle width, canal, wedging ---------------------------------
    ped, canal, torg, wedge = {}, {}, {}, {}
    for vid in sorted(have):
        if vid not in LUMBAR:
            continue
        m = have[vid]
        idx = np.argwhere(m)
        zc = int(np.median(idx[:, 2]))
        sl = m[:, :, zc]
        if sl.sum() < 60:
            continue
        # canal = the enclosed hole in the ring at mid-body
        filled = ndimage.binary_fill_holes(sl)
        hole = filled & ~sl
        if hole.any():
            cc, ncc = ndimage.label(hole)
            sizes = ndimage.sum(hole, cc, range(1, ncc + 1))
            big = cc == (int(np.argmax(sizes)) + 1)
            hy = np.nonzero(big.any(axis=0))[0]
            canal[LUMBAR[vid]] = round(float(len(hy)) * sp[1], 1)
            by = np.nonzero(sl.any(axis=0))[0]
            depth = float(len(by)) * sp[1]
            if depth > 0:
                torg[LUMBAR[vid]] = round(canal[LUMBAR[vid]] / depth, 3)
            # pedicle isthmus: narrowest left-right run of bone between body and canal
            rows = []
            for y in hy:
                col = np.nonzero(sl[:, y])[0]
                if len(col) > 3:
                    rows.append(float(col.max() - col.min()) * sp[0])
            if rows:
                ped[LUMBAR[vid]] = round(min(rows) / 2.0, 1)
        # wedging: anterior vs posterior body height
        ys = np.nonzero(m.any(axis=(0, 2)))[0]
        if len(ys) > 6:
            ant = m[:, ys[-max(2, len(ys) // 5):], :]
            post = m[:, ys[:max(2, len(ys) // 5)], :]
            za = np.nonzero(ant.any(axis=(0, 1)))[0]
            zp = np.nonzero(post.any(axis=(0, 1)))[0]
            if len(za) and len(zp) and len(zp) > 0:
                wedge[LUMBAR[vid]] = round(len(za) / max(1, len(zp)), 3)

    for nm, d in (("pedicle_mm", ped), ("canal_ap_mm", canal),
                  ("torg", torg), ("wedge", wedge)):
        for k, v in d.items():
            r[f"{nm}_{k}"] = v
    if ped:
        r["pedicle_min_mm"] = round(min(ped.values()), 1)
    if torg:
        r["torg_min"] = round(min(torg.values()), 3)
    if wedge:
        r["wedge_min"] = round(min(wedge.values()), 3)
    return r


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--manifest", default="data/hf_export_v4/manifest.json")
    ap.add_argument("--cases", default="")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="morphometrics")
    a = ap.parse_args()

    labdir = Path(a.labels)
    stems = ([c.strip() for c in a.cases.split(",") if c.strip()]
             or sorted(p.name.replace("_label.nii.gz", "")
                       for p in labdir.glob("*_label.nii.gz")))
    files = [str(labdir / f"{s}_label.nii.gz") for s in stems
             if (labdir / f"{s}_label.nii.gz").exists()]
    print(f"{len(files)} case(s)\n", flush=True)

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, x in enumerate(ex.map(one, files, chunksize=2), 1):
            res.append(x)
            if i % 50 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    lut = {}
    mp = Path(a.manifest)
    if mp.exists():
        recs = json.load(open(mp))
        recs = recs if isinstance(recs, list) else recs.get("records", [])
        for rec in recs:
            s = str(rec.get("label_file", "")).split("/")[-1].replace("_label.nii.gz", "")
            if s:
                lut[s] = {k: rec.get(k) for k in ("age", "sex", "lstv_label", "config")}
    for x in res:
        x.update(lut.get(x["case"], {}))

    ok = [x for x in res if "error" not in x]
    cols = sorted({k for x in ok for k in x}, key=lambda k: (k != "case", k))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    p = out / "surgical_morphometrics.csv"
    with open(p, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader(); w.writerows(ok)

    def summarise(key, unit=""):
        v = np.array([x[key] for x in ok if isinstance(x.get(key), (int, float))], float)
        if v.size < 5:
            print(f"  {key:24s} n={v.size} (too few)")
            return
        print(f"  {key:24s} n={v.size:4d}  median {np.median(v):7.1f}{unit}  "
              f"IQR {np.percentile(v,25):.1f}-{np.percentile(v,75):.1f}")

    print(f"\n  {len(ok)} of {len(res)} cases measured\n")
    for k, u in (("pelvic_incidence_deg", "°"), ("sacral_slope_deg", "°"),
                 ("pelvic_tilt_deg", "°"), ("ll_supine_deg", "°"),
                 ("pi_ll_mismatch_deg", "°"), ("crest_above_l45_mm", "mm"),
                 ("rib12_to_crest_mm", "mm"), ("pedicle_min_mm", "mm"),
                 ("torg_min", "")):
        summarise(k, u)
    n_ll = sum(1 for x in ok if x.get("ll_supine_deg") is not None)
    n_full = sum(1 for x in ok if x.get("ll_complete"))
    if n_ll:
        print(f"\n  lordosis arc reaches L1 in {n_full}/{n_ll} measured cases; "
              f"PI-LL mismatch is reported for those only")
    blocked = sum(1 for x in ok if x.get("crest_blocks_l45"))
    n_cr = sum(1 for x in ok if x.get("crest_blocks_l45") is not None)
    if n_cr:
        print(f"\n  crest >= 12mm above the L4-5 disc (raised subsidence risk after "
              f"lateral fusion): {blocked}/{n_cr} = {100*blocked/n_cr:.1f}%")
    print(f"\n  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
