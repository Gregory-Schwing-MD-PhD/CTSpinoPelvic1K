"""Which ribs in the release are in two or more substantial pieces, and why.

`results/rib_qc_v4_vs_release` counts 13 such ribs in 10 release records but names none of them. A count cannot
say whether a split is a defect or an exception, so this lists every one with what would decide it:

  pieces          voxel count of the largest and second piece, and the gap between them in mm
  fov_edge        the smaller piece lies within 2 voxels of the reconstruction boundary (CT <= -1000 HU outside the
                  circle, or a volume face), i.e. the rib left the field and came back: an acquisition artefact
  spine_side      the smaller piece is the medial (spine-side) end: a rib whose head was segmented separately
  same_axis       the two pieces are roughly collinear (gap under 25 mm along the rib): one bone with a cut
  other_rib_touch the smaller piece touches a differently numbered rib: a mislabelled fragment

Definition follows rib_qc_stages.fragments: a second connected component of at least 50 voxels and at least 15%
of the largest. Writes rib_fragments_release.csv and a coronal maximum-intensity PNG per flagged rib with the two
pieces coloured, so the call can be checked by eye.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS  # noqa: E402

ST = ndimage.generate_binary_structure(3, 3)


def rib_name(rid):
    if LS.RIB_LEFT_OFFSET < rid <= LS.RIB_LEFT_OFFSET + 13:
        return f"left {rid - LS.RIB_LEFT_OFFSET}"
    if LS.RIB_RIGHT_OFFSET < rid <= LS.RIB_RIGHT_OFFSET + 13:
        return f"right {rid - LS.RIB_RIGHT_OFFSET}"
    return str(rid)


def one(args):
    lab_p, ct_p, png_dir = args
    out = []
    try:
        im = nib.load(str(lab_p)); lab = np.asanyarray(im.dataobj).astype(np.int16)
        sp = np.asarray(im.header.get_zooms()[:3], np.float32)
        ct = None
        if ct_p and Path(ct_p).exists():
            ct = np.asanyarray(nib.load(str(ct_p)).dataobj).astype(np.float32)
        inside = None
        if ct is not None and ct.shape == lab.shape:
            inside = ct > -1000.0                       # reconstruction circle + patient/table, air pad excluded
            inside = ndimage.binary_erosion(inside, iterations=2)
        vert = (lab >= 1) & (lab <= 28)
        for rid in range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13 + 1):
            m = lab == rid
            n = int(m.sum())
            if n < 50:
                continue
            cc, k = ndimage.label(m, structure=ST)
            if k < 2:
                continue
            sizes = np.bincount(cc.ravel())[1:]
            order = np.argsort(sizes)[::-1]
            big, second = int(sizes[order[0]]), int(sizes[order[1]])
            if second < 50 or second < 0.15 * big:
                continue
            A = cc == (order[0] + 1); B = cc == (order[1] + 1)
            ia = np.argwhere(A); ib = np.argwhere(B)
            ca, cb = ia.mean(0), ib.mean(0)
            # gap: nearest-point distance between the pieces
            da = ndimage.distance_transform_edt(~A, sampling=sp)
            gap_mm = float(da[B].min())
            # fov edge: any voxel of the smaller piece dilated by 2 falls outside the reconstruction, or on a face
            Bd = ndimage.binary_dilation(B, iterations=2)
            face = bool(ib[:, 0].min() <= 1 or ib[:, 1].min() <= 1 or ib[:, 2].min() <= 1 or
                        ib[:, 0].max() >= lab.shape[0] - 2 or ib[:, 1].max() >= lab.shape[1] - 2 or ib[:, 2].max() >= lab.shape[2] - 2)
            fov = bool(face or (inside is not None and (Bd & ~inside).any()))
            # spine side: the smaller piece is nearer the vertebral column than the larger
            dv = ndimage.distance_transform_edt(~vert, sampling=sp) if vert.any() else None
            spine_side = bool(dv is not None and float(dv[B].min()) < float(dv[A].min()))
            touch = sorted({int(x) for x in np.unique(lab[Bd]) if x != rid and x != 0 and (LS.RIB_LEFT_OFFSET < x <= LS.RIB_RIGHT_OFFSET + 13)})
            out.append({"case": lab_p.name.split("_")[0], "rib": rib_name(rid), "rid": rid, "pieces": int(k), "big": big, "second": second,
                        "gap_mm": round(gap_mm, 1), "fov_edge": fov, "on_face": face, "spine_side": spine_side,
                        "other_rib_touch": ",".join(rib_name(x) for x in touch), "centroid_sep_mm": round(float(np.linalg.norm((ca - cb) * sp)), 1)})
            if png_dir:
                try:
                    from PIL import Image
                    # coronal MIP (collapse the anterior-posterior axis, assumed axis 1 for this dataset's P,I,R volumes)
                    ax = 0
                    base = (np.clip((ct if ct is not None else lab.astype(np.float32)) , -200, 1200) + 200) / 1400.0 if ct is not None else (lab > 0).astype(np.float32)
                    mip = base.max(axis=ax); rgb = np.stack([mip] * 3, -1)
                    ma = A.max(axis=ax); mb = B.max(axis=ax)
                    rgb[ma] = [1.0, 0.25, 0.25]; rgb[mb] = [0.2, 0.9, 1.0]
                    img = (rgb * 255).astype(np.uint8)
                    Image.fromarray(np.rot90(img)).save(Path(png_dir) / f"{lab_p.name.split('_')[0]}_{rid}.png")
                except Exception:  # noqa: BLE001
                    pass
    except Exception as e:  # noqa: BLE001
        out.append({"case": lab_p.name, "error": repr(e)[:120]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="~/data/CTSpinoPelvic1K/labels")
    ap.add_argument("--ct", default="~/data/CTSpinoPelvic1K/ct")
    ap.add_argument("--out", default="~/rib_fragments_release")
    ap.add_argument("--workers", type=int, default=16); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    L = Path(a.labels).expanduser(); C = Path(a.ct).expanduser(); O = Path(a.out).expanduser(); (O / "png").mkdir(parents=True, exist_ok=True)
    labs = sorted(L.glob("*_label.nii.gz"))
    if a.limit:
        labs = labs[: a.limit]
    jobs = [(p, C / p.name.replace("_label", "_ct"), str(O / "png")) for p in labs]
    rows = []
    with ProcessPoolExecutor(a.workers) as ex:
        for i, r in enumerate(ex.map(one, jobs, chunksize=4)):
            rows += r
            if i % 100 == 0:
                print(f"  {i}/{len(jobs)} labels, {len(rows)} flagged ribs", flush=True)
    good = [r for r in rows if "error" not in r]
    with open(O / "rib_fragments_release.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "rib", "rid", "pieces", "big", "second", "gap_mm", "fov_edge", "on_face", "spine_side", "other_rib_touch", "centroid_sep_mm"])
        w.writeheader(); w.writerows(good)
    summary = {"labels": len(labs), "flagged_ribs": len(good), "records": len({r["case"] for r in good}),
               "fov_edge": sum(1 for r in good if r["fov_edge"]), "spine_side": sum(1 for r in good if r["spine_side"]),
               "touching_other_rib": sum(1 for r in good if r["other_rib_touch"]), "gap_under_25mm": sum(1 for r in good if r["gap_mm"] < 25),
               "errors": [r for r in rows if "error" in r][:5]}
    json.dump(summary, open(O / "summary.json", "w"), indent=1)
    print(json.dumps(summary, indent=1))
    for r in good:
        print(f"  {r['case']} {r['rib']:9s} pieces {r['pieces']} big {r['big']:6d} second {r['second']:5d} gap {r['gap_mm']:5.1f} mm  fov_edge={r['fov_edge']} spine_side={r['spine_side']} touches={r['other_rib_touch'] or '-'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
