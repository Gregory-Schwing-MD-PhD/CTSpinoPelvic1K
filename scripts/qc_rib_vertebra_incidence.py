"""scripts/qc_rib_vertebra_incidence.py — does each rib sit on the vertebra its number claims?

Rib N should be incident on T-N. That is the classic anchor: it checks the numbering is tied
to anatomy rather than merely internally consecutive, so it catches an off-by-one run that a
"ribs are consecutive" check sails straight past.

WHY THIS IS NOT JUST review_anatomy_qc.rib_anchor
-------------------------------------------------
rib_anchor searches THORACIC vertebrae only. A rib whose nearest vertebra is LUMBAR therefore
comes back as "not incident on T-N", which reads as a numbering error -- and it is usually the
opposite: a rib articulating with L1 is the lumbar-rib phenotype, a real finding this dataset
exists to capture, not a mistake to correct. Conflating the two would let a QC pass "fix" away
the anomalies.

So every rib is resolved against thoracic AND lumbar vertebrae and lands in one bucket:

    match       nearest labelled vertebra is T-N, within ANCHOR_MM        -> correct
    offset      nearest is T-M within ANCHOR_MM, M != N                   -> NUMBERING ERROR
    lumbar      nearest is a lumbar vertebra, within ANCHOR_MM            -> PHENOTYPE
    no_contact  nothing within ANCHOR_MM                                  -> segmentation gap
    skipped     T-N not in the FOV and nothing else in reach              -> not evaluable

Only `offset` is a defect. `lumbar` is a result. `no_contact` is usually a rib entering the
scan antero-laterally with its vertebra out of view, which is why it is separated from both.

A case's ribs are called MISNUMBERED only on a consistent run of `offset` at the same delta:
one stray rib is a segmentation artefact, a whole side shifted by one is a renumber.

    python scripts/qc_rib_vertebra_incidence.py --labels DIR [--workers 14] [--out DIR]
    python scripts/qc_rib_vertebra_incidence.py --hf-rev v4          # straight from the Hub
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "review"))

import label_scheme as LS                                   # noqa: E402
from review_anatomy_qc import ANCHOR_MM, MIN_VERT_VOX       # noqa: E402

THORACIC_BASE = 7            # T-n has id 7 + n, so T1=8 .. T12=19 (VerSe-native v4 scheme)

# A rib touching a face of the volume was CUT BY THE SCAN, not by anatomy. This matters
# more than it sounds: on v4, 228 of 236 ribs that made a case look left/right asymmetric
# were touching the top slice. The patient is not square to the gantry, so the superior
# FOV edge clips one side a level before the other and "asymmetry" is really "where the
# scan stopped". 32% of those present-side ribs were under 1000 voxels -- partial-volume
# slivers, not ribs.
#
# Truncation is recorded per rib rather than silently dropped, because whether to exclude
# a truncated rib depends on the question: it still counts for "does this rib reach its
# vertebra" if its head is in the field, but it must never count for "how many ribs does
# this person have".
LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}
BUCKETS = ("match", "offset", "lumbar", "no_contact", "skipped")


def _rib_id(side: str, n: int) -> int:
    return (LS.RIB_LEFT_OFFSET if side == "left" else LS.RIB_RIGHT_OFFSET) + n


def _pts(mask, cap=250):
    """Subsampled point cloud. A full EDT per rib-vertebra pair explodes when a mislabelled
    rib is far from its vertebra; the anchor decision only needs a min distance."""
    p = np.argwhere(mask)
    return p[:: max(1, len(p) // cap)] if len(p) else p


def _touches_face(mask) -> bool:
    """Does the structure reach any face of the volume, i.e. was it cut by the scan?"""
    return bool(mask[0].any() or mask[-1].any()
                or mask[:, 0].any() or mask[:, -1].any()
                or mask[:, :, 0].any() or mask[:, :, -1].any())


def _mindist(a, b, spacing):
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = (a[:, None, :] - b[None, :, :]) * spacing
    return float(np.sqrt((d ** 2).sum(-1)).min())


def incidence(lab: np.ndarray, affine) -> list:
    """One row per rib present: which vertebra it actually touches, and how that compares
    with the vertebra its number claims."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    # A vertebra anchors a rib only if it is really there. A sliver at the edge of the FOV
    # would otherwise sit "nearest" to a rib whose true vertebra is out of view entirely.
    verts = {}
    for n in range(1, 13):
        m = lab == THORACIC_BASE + n
        if m.sum() >= MIN_VERT_VOX:
            verts[f"T{n}"] = _pts(m)
    for vid, name in LUMBAR.items():
        m = lab == vid
        if m.sum() >= MIN_VERT_VOX:
            verts[name] = _pts(m)

    rows = []
    for side in ("left", "right"):
        for n in range(1, 13):
            rm = lab == _rib_id(side, n)
            if not rm.any():
                continue
            rp = _pts(rm)
            trunc = _touches_face(rm)
            vox = int(rm.sum())
            own = f"T{n}"
            gaps = {v: _mindist(rp, p, spacing) for v, p in verts.items()}
            near, gap = (min(gaps.items(), key=lambda kv: kv[1])
                         if gaps else ("", float("inf")))
            gap_own = gaps.get(own, float("inf"))

            if gap_own <= ANCHOR_MM:
                bucket, delta = "match", 0
            elif gap <= ANCHOR_MM and near.startswith("L"):
                bucket, delta = "lumbar", None
            elif gap <= ANCHOR_MM and near.startswith("T"):
                bucket, delta = "offset", int(near[1:]) - n
            elif own not in verts:
                bucket, delta = "skipped", None
            elif trunc:
                # cut by the scan before it could reach anything -- not a defect
                bucket, delta = "skipped", None
            else:
                bucket, delta = "no_contact", None
            rows.append({"side": side, "rib": n, "expect": own,
                         "nearest": near, "gap_mm": round(gap, 1),
                         "gap_own_mm": (round(gap_own, 1)
                                        if np.isfinite(gap_own) else ""),
                         "bucket": bucket, "delta": "" if delta is None else delta,
                         "truncated": int(trunc), "voxels": vox})
    return rows


def verdict(rows: list) -> tuple:
    """Case-level call. A consistent run of the same offset on >=3 ribs is a renumber; one or
    two stray offsets are far more likely a segmentation artefact than a counting mistake."""
    off = [r for r in rows if r["bucket"] == "offset"]
    by_delta = Counter(r["delta"] for r in off)
    misnumbered = any(c >= 3 for c in by_delta.values())
    note = ""
    if misnumbered:
        d, c = by_delta.most_common(1)[0]
        note = f"{c} ribs consistently off by {d:+d} -> renumber"
    elif off:
        note = f"{len(off)} isolated offset rib(s) -> check segmentation, not numbering"
    return misnumbered, note


def one(path: str) -> dict:
    try:
        img = nib.load(path)
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        rows = incidence(lab, img.affine)
    except Exception as exc:                                  # noqa: BLE001
        return {"case": Path(path).name, "error": f"{type(exc).__name__}: {exc}"}
    mis, note = verdict(rows)
    c = Counter(r["bucket"] for r in rows)
    return {"case": Path(path).name, "rows": rows, "misnumbered": mis, "note": note,
            **{b: c.get(b, 0) for b in BUCKETS}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", help="directory of *.nii.gz label volumes")
    ap.add_argument("--hf-repo", default=os.environ.get(
        "DS_REPO", "anonymous-mlhc/CTSpinoPelvic1K"))
    ap.add_argument("--hf-rev", help="pull labels from this dataset revision instead")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=14)
    ap.add_argument("--out", default="qc_rib_incidence")
    a = ap.parse_args()

    if a.hf_rev:
        from huggingface_hub import snapshot_download
        root = snapshot_download(a.hf_repo, repo_type="dataset", revision=a.hf_rev,
                                 allow_patterns="labels/*", max_workers=16,
                                 token=os.environ.get("HF_TOKEN"))
        files = sorted(str(p) for p in Path(root, "labels").glob("*.nii.gz"))
        src = f"{a.hf_repo}@{a.hf_rev}"
    else:
        files = sorted(str(p) for p in Path(a.labels).glob("*.nii.gz"))
        src = a.labels
    if a.limit:
        files = files[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} labels from {src}\n")

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, files, chunksize=1), 1):
            res.append(r)
            if i % 50 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    bad = [r for r in res if r.get("error")]
    ok = [r for r in res if not r.get("error")]

    with open(out / "rib_incidence.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "side", "rib", "expect", "nearest", "gap_mm",
                    "gap_own_mm", "bucket", "delta", "truncated", "voxels"])
        for r in ok:
            for x in r["rows"]:
                w.writerow([r["case"], x["side"], x["rib"], x["expect"], x["nearest"],
                            x["gap_mm"], x["gap_own_mm"], x["bucket"], x["delta"],
                            x["truncated"], x["voxels"]])

    mis = [r for r in ok if r["misnumbered"]]
    with open(out / "rib_misnumbered.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "note", "offset", "lumbar", "no_contact", "match"])
        for r in sorted(mis, key=lambda r: -r["offset"]):
            w.writerow([r["case"], r["note"], r["offset"], r["lumbar"],
                        r["no_contact"], r["match"]])

    lum = [r for r in ok if r["lumbar"]]
    with open(out / "lumbar_rib_phenotype.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "n_lumbar_ribs", "detail"])
        for r in sorted(lum, key=lambda r: -r["lumbar"]):
            det = "; ".join(f"{x['side'][0]}{x['rib']}->{x['nearest']}"
                            for x in r["rows"] if x["bucket"] == "lumbar")
            w.writerow([r["case"], r["lumbar"], det])

    tot = Counter()
    for r in ok:
        for b in BUCKETS:
            tot[b] += r[b]
    n_ribs = sum(tot.values()) or 1
    print(f"\n  cases            {len(ok)}  ({len(bad)} unreadable)")
    print(f"  ribs evaluated   {n_ribs}")
    for b in BUCKETS:
        print(f"    {b:11s} {tot[b]:6d}  {100*tot[b]/n_ribs:5.1f}%")
    print(f"\n  MISNUMBERED cases (>=3 ribs at one offset) : {len(mis)}")
    print(f"  cases with a LUMBAR rib (phenotype, keep)  : {len(lum)}")
    deltas = Counter(x["delta"] for r in ok for x in r["rows"] if x["bucket"] == "offset")
    if deltas:
        print(f"  offset deltas: {dict(sorted(deltas.items()))}")
    json.dump({"source": src, "cases": len(ok), "unreadable": len(bad),
               "buckets": dict(tot), "misnumbered_cases": len(mis),
               "lumbar_rib_cases": len(lum),
               "offset_deltas": {str(k): v for k, v in deltas.items()}},
              open(out / "summary.json", "w"), indent=2)
    print(f"\n  wrote {out}/rib_incidence.csv, rib_misnumbered.csv, "
          f"lumbar_rib_phenotype.csv, summary.json")
    if bad:
        print(f"  unreadable: {[r['case'] for r in bad][:5]}")


if __name__ == "__main__":
    main()
