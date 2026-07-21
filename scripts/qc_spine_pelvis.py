"""
scripts/qc_spine_pelvis.py — screen v4 for CLASS-MIXING in the spine/pelvis (the thing the missing-
vertebra scan does NOT catch): a solid bone split into pieces, or one connected bone carrying two
labels. Two checks, over all 802:
  * structure_integrity  -> a vertebra/sacrum/hip/femur (ids 1-33) split so no single piece dominates
  * _vertebra_label_mixing -> one connected vertebral bone carries TWO vertebra labels (half-L3/half-L4)
FOV-truncated bones are exempted inside structure_integrity (advisory, not counted).

Splits are classified SPINE (vertebra ids 1-25, T13) vs PELVIS (sacrum/S1/hip/femur) because the fix
differs: spine mixing is radiologist GT (yours); pelvis mixing is pseudolabel.

    HF_TOKEN=... HF_HUB_OFFLINE=1 python scripts/qc_spine_pelvis.py [--workers 4]
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np, nibabel as nib
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import review_anatomy_qc as RA          # noqa: E402
from huggingface_hub import hf_hub_download   # noqa: E402

DS = os.environ.get("V2_REPO", "anonymous-mlhc/CTSpinoPelvic1K")
REV = os.environ.get("V4_REV", "v4")
PELVIS_WORDS = ("hip", "femur", "sacrum", "S1", "coccyx")


def _work(args):
    t, p, rev = args
    tok = os.environ["HF_TOKEN"]
    try:
        img = nib.load(hf_hub_download(DS, p, repo_type="dataset", token=tok, revision=REV))
        lab = np.asanyarray(img.dataobj); aff = img.affine
        ok, msgs = RA.structure_integrity(lab, aff)
        splits = [m[2:] for m in msgs if m.startswith("X")]       # "X <name> is split..." -> drop "X "
        vmix = RA._vertebra_label_mixing(lab)
        return (t, rev, splits, bool(vmix))
    except Exception:
        return None


def main(argv=None) -> int:
    from concurrent.futures import ProcessPoolExecutor
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv); tok = os.environ["HF_TOKEN"]
    recs = json.load(open(hf_hub_download(DS, "manifest.json", repo_type="dataset", token=tok, revision=REV)))
    recs = recs if isinstance(recs, list) else recs.get("records", [])
    reviewed = set()
    try:
        wl = json.load(open(hf_hub_download(DS, "rib_worklist.json", repo_type="dataset", token=tok, revision=REV)))
        reviewed = {str(x) for x in (wl.get("tokens") if isinstance(wl, dict) else wl)}
    except Exception:
        pass
    items = [(str(r.get("token")), r.get("pseudo_label_file") or r.get("label_file"),
              str(r.get("token")) in reviewed)
             for r in recs if (r.get("pseudo_label_file") or r.get("label_file"))]
    if a.limit:
        items = items[:: max(1, len(items)//a.limit)][:a.limit]
    print(f"scanning {len(items)} v4 cases for spine/pelvis class-mixing [{a.workers} procs]\n", flush=True)

    rows = []; n = spine_c = pelvis_c = vmix_c = 0; done = 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(_work, items, chunksize=4):
            done += 1
            if done % 50 == 0: print(f"  ...{done}/{len(items)}", flush=True)
            if not r: continue
            t, rev, splits, vmix = r; n += 1
            if not splits and not vmix: continue
            is_pelvis = any(any(w in s for w in PELVIS_WORDS) for s in splits)
            is_spine = any(not any(w in s for w in PELVIS_WORDS) for s in splits) or vmix
            if is_spine: spine_c += 1
            if is_pelvis: pelvis_c += 1
            if vmix: vmix_c += 1
            rows.append({"token": t, "reviewed": rev,
                         "region": ("spine+pelvis" if (is_spine and is_pelvis) else "pelvis" if is_pelvis else "spine"),
                         "vertebra_label_mixing": vmix,
                         "splits": " | ".join(splits)})
    rows.sort(key=lambda r: (r["region"], r["token"]))
    with open("spine_pelvis_mixing.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "reviewed", "region", "vertebra_label_mixing", "splits"])
        w.writeheader(); w.writerows(rows)
    print(f"\n===== SPINE/PELVIS CLASS-MIXING ({n} cases) -> spine_pelvis_mixing.csv =====")
    print(f"   cases with a split/mixed bone: {len(rows)}")
    print(f"   involving SPINE (vertebra GT — your fix):  {spine_c}")
    print(f"   involving PELVIS (sacrum/S1/hip/femur):    {pelvis_c}")
    print(f"   one bone carrying TWO vertebra labels:     {vmix_c}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
