"""scripts/qc_vertebra_grouping.py — group vertebrae by whether they bear a rib, and count.

The naming argument (is that T13, or L1 with a lumbar rib, or T12 with a hypoplastic one?)
is a dispute about where counting starts. It disappears if you stop asserting names and
record what the image shows: this vertebra bears a rib, that one does not.

The quantity that falls out is the one clinicians already use:

    NON-RIB-BEARING COUNT = labelled vertebrae strictly between the lowest rib-bearing
                            vertebra and the sacrum.

Normally 5 (L1-L5). Four suggests sacralization, six lumbarization or a sixth lumbar-type
segment -- and the count is invariant to where you start numbering, so two readers who
disagree about every vertebra's NAME can still agree on it. `Lumbosac.csv` records exactly
this under "Non-rib bearing vertebra #", which makes the radiologist's column a direct
check on the automated one rather than a loose analogue.

DETERMINABILITY IS ENFORCED, NOT ASSUMED. The count is only meaningful when the chain from
the lowest rib-bearing vertebra down to the sacrum is complete: sacrum present, a
rib-bearing vertebra present, and every id in between actually labelled. A gap in that
chain (FOV, annotation) would silently shorten the count and manufacture a sacralization.
Cases failing any of those are reported as `indeterminate` and never counted -- an
undercount of transitional anatomy is a fabricated finding, not a conservative one.

    python scripts/qc_vertebra_grouping.py --hf-rev v4 [--workers 24] [--out DIR]
    python scripts/qc_vertebra_grouping.py --labels data/v5_final --out qc_group_v5
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "review"))

import label_scheme as LS                                   # noqa: E402
from review_anatomy_qc import ANCHOR_MM, MIN_VERT_VOX       # noqa: E402

THOR = {7 + n: f"T{n}" for n in range(1, 13)}       # T1=8 .. T12=19
LUMB = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}
SACRUM, S1 = 26, 29
SPINE = {**THOR, **LUMB}                            # ordered by id = cranio-caudal


def _pts(mask, cap=250):
    p = np.argwhere(mask)
    return p[:: max(1, len(p) // cap)] if len(p) else p


def _mind(a, b, spacing):
    if len(a) == 0 or len(b) == 0:
        return float("inf")
    d = (a[:, None, :] - b[None, :, :]) * spacing
    return float(np.sqrt((d ** 2).sum(-1)).min())


def group(lab: np.ndarray, affine) -> dict:
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))

    present, pts = {}, {}
    for vid, name in SPINE.items():
        m = lab == vid
        if m.sum() >= MIN_VERT_VOX:
            present[vid] = name
            pts[vid] = _pts(m)
    ribs = [_pts(lab == (off + n))
            for off in (LS.RIB_LEFT_OFFSET, LS.RIB_RIGHT_OFFSET)
            for n in range(1, 13) if (lab == (off + n)).any()]

    # A vertebra bears a rib when it is that rib's NEAREST vertebra -- not merely when a
    # rib passes within ANCHOR_MM of it. Proximity is not articulation: a 12th rib head
    # sitting at the T12/L1 junction comes within 10 mm of BOTH, and scoring by proximity
    # marked L1 rib-bearing in 61 cases where the strict incidence QC found only 14
    # lumbar ribs in the whole cohort. Assigning each rib to its single closest vertebra
    # reconciles the two and stops a normal 12th rib from manufacturing a phenotype.
    bearing = {vid: False for vid in present}
    for rp in ribs:
        d = {vid: _mind(pts[vid], rp, spacing) for vid in present}
        if not d:
            continue
        vid, gap = min(d.items(), key=lambda kv: kv[1])
        if gap <= ANCHOR_MM:
            bearing[vid] = True

    has_sac = (lab == SACRUM).sum() >= MIN_VERT_VOX or (lab == S1).sum() >= MIN_VERT_VOX
    rb = sorted([v for v, b in bearing.items() if b])
    nrb = sorted([v for v, b in bearing.items() if not b])

    out = {
        "n_vert_labelled": len(present),
        "n_rib_bearing": len(rb),
        "n_non_rib_bearing": len(nrb),
        "lowest_rib_bearing": SPINE[rb[-1]] if rb else "",
        "has_sacrum": int(has_sac),
        "rib_bearing": " ".join(SPINE[v] for v in rb),
        "non_rib_bearing": " ".join(SPINE[v] for v in nrb),
    }

    # The count between the anchors -- only where the chain is provably complete.
    reason = ""
    if not rb:
        reason = "no rib-bearing vertebra in the FOV"
    elif not has_sac:
        reason = "no sacrum in the FOV"
    else:
        lo = rb[-1]                                   # lowest rib-bearing vertebra
        chain = list(range(lo + 1, max(LUMB) + 1))
        below = [v for v in chain if v in present]
        # every id from just under the last rib-bearing vertebra to the last labelled
        # lumbar one must be present, or the count is missing a level
        if below and below != list(range(lo + 1, below[-1] + 1)):
            reason = "gap in the chain between the last rib and the sacrum"
        else:
            out["n_between"] = len(below)
            out["between"] = " ".join(SPINE[v] for v in below)
    out["indeterminate"] = reason
    return out


def one(path: str) -> dict:
    try:
        img = nib.load(path)
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        g = group(lab, img.affine)
    except Exception as exc:                                  # noqa: BLE001
        return {"case": Path(path).name, "error": f"{type(exc).__name__}: {exc}"}
    return {"case": Path(path).name, **g}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels")
    ap.add_argument("--hf-repo", default="anonymous-mlhc/CTSpinoPelvic1K")
    ap.add_argument("--hf-rev")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default="qc_vertebra_grouping")
    a = ap.parse_args()

    if a.hf_rev:
        from huggingface_hub import snapshot_download
        root = snapshot_download(a.hf_repo, repo_type="dataset", revision=a.hf_rev,
                                 allow_patterns="labels/*", max_workers=16,
                                 token=os.environ.get("HF_TOKEN"))
        a.labels = str(Path(root, "labels"))
        src = f"{a.hf_repo}@{a.hf_rev}"
    else:
        src = a.labels
    files = sorted(str(p) for p in Path(a.labels).glob("*.nii.gz"))
    if a.limit:
        files = files[:a.limit]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    print(f"{len(files)} labels from {src}\n")

    res = []
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for i, r in enumerate(ex.map(one, files, chunksize=1), 1):
            res.append(r)
            if i % 100 == 0:
                print(f"  {i}/{len(files)}", flush=True)

    ok = [r for r in res if not r.get("error")]
    cols = ["case", "n_vert_labelled", "n_rib_bearing", "n_non_rib_bearing",
            "lowest_rib_bearing", "has_sacrum", "n_between", "between",
            "rib_bearing", "non_rib_bearing", "indeterminate"]
    with open(out / "vertebra_grouping.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in ok:
            w.writerow(r)

    det = [r for r in ok if not r["indeterminate"]]
    nb = Counter(r["n_between"] for r in det)
    print(f"\n  cases                {len(ok)}  ({len(res)-len(ok)} unreadable)")
    print(f"  determinate count    {len(det)}  "
          f"({100*len(det)/max(1,len(ok)):.0f}%)")
    print(f"  indeterminate        {len(ok)-len(det)}")
    for why, c in Counter(r["indeterminate"] for r in ok if r["indeterminate"]).most_common():
        print(f"      {why:52s} {c}")
    print(f"\n  NON-RIB-BEARING between last rib and sacrum:")
    for k in sorted(nb):
        flag = "  <- normal" if k == 5 else ("  <- sacralization?" if k < 5
                                             else "  <- lumbarization / 6th?")
        print(f"      {k}: {nb[k]:4d}  ({100*nb[k]/max(1,len(det)):4.1f}%){flag}")
    json.dump({"source": src, "cases": len(ok), "determinate": len(det),
               "n_between": {str(k): v for k, v in sorted(nb.items())}},
              open(out / "summary.json", "w"), indent=2)
    print(f"\n  wrote {out}/vertebra_grouping.csv, summary.json")


if __name__ == "__main__":
    main()
