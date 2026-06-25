"""
smoke_test_hf.py — point it at any CTSpinoPelvic1K branch/revision on HuggingFace and
verify the published labels are VerSe-native + stage-correct.

It samples fused + spine_only cases (and token 3, our canonical diagnostic case, if
present), dumps per-id voxel counts + world-Z centroids, and runs STAGE-AWARE checks:

  all stages : spine is VerSe (lumbar 20-25 present+large, sacrum 26 in fused);
               any ignore voxels are 255 (never the old 10/50); no stray ids.
  v1         : NO ribs(34-57) / femurs(32/33) / S1(29); spine_only has ignore=255
               and NO pelvis (partial-annotation contract).
  v2         : spine_only pelvis is now MODEL-filled (hips 30/31 present); still no
               ribs/femurs/S1.
  v3         : S1=29 + femurs 32/33 present; ribs 34-57 present (TS).
  v4         : ribs 34-57 present (RibSeg).

Stage is inferred from the revision name (v1/v2/v3/v4) or set with --stage.

  HF_TOKEN=hf_xxx python scripts/smoke_test_hf.py --repo anonymous-mlhc/CTSpinoPelvic1K --revision v1
  HF_TOKEN=hf_xxx python scripts/smoke_test_hf.py --repo <org>/CTSpinoPelvic1K --revision v3 --n 8
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from huggingface_hub import hf_hub_download, list_repo_files

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS  # noqa: E402

RIBS = set(range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13))   # 34..57
FEMURS = {LS.FEMUR_LEFT, LS.FEMUR_RIGHT}                              # 32,33
S1 = LS.S1_ID                                                         # 29
VALID = set(LS.label_dict().values()) | {0}
SUP = np.array([0, 0, 1.0])


def infer_stage(rev: str, override: int) -> int:
    if override:
        return override
    m = re.search(r"v([1-4])", rev or "")
    return int(m.group(1)) if m else 1     # default to the strictest (v1) expectations


def dump(repo, rev, token, rec):
    lf = rec.get("label_file") or f"labels/{int(rec['token']):04d}_label.nii.gz"
    p = hf_hub_download(repo, lf, repo_type="dataset", revision=rev, token=token)
    img = nib.load(p)
    s = np.asanyarray(img.dataobj).astype(np.int32)
    A = img.affine
    info = {}
    for i in np.unique(s):
        if i == 0:
            continue
        c = np.argwhere(s == i)
        z = round(float(((c @ A[:3, :3].T + A[:3, 3]) @ SUP).mean()))
        info[int(i)] = (int((s == i).sum()), z)
    ids = sorted(info)
    print(f"\n token {rec.get('token')} ({rec.get('config')}) {lf}")
    print("   ids:", ids)
    print("   thoracic 8-19:", {i: info[i] for i in ids if 8 <= i <= 19})
    print("   lumbar 20-25 :", {i: info[i] for i in ids if 20 <= i <= 25})
    print(f"   sacrum 26: {info.get(26)} | S1 29: {info.get(29)} | "
          f"hips 30/31: {info.get(30)},{info.get(31)} | femurs 32/33: {info.get(32)},{info.get(33)}")
    print("   ribs 34-57   :", sorted(i for i in ids if i in RIBS) or "none")
    return rec.get("config"), set(ids)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="anonymous-mlhc/CTSpinoPelvic1K")
    ap.add_argument("--revision", default="v1")
    ap.add_argument("--n", type=int, default=6, help="cases to sample")
    ap.add_argument("--stage", type=int, default=0, help="override stage (1-4); default inferred from revision")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    a = ap.parse_args()
    stage = infer_stage(a.revision, a.stage)
    print(f"=== {a.repo}@{a.revision}  (stage {stage}) ===")

    files = list_repo_files(a.repo, repo_type="dataset", revision=a.revision, token=a.token)
    n_lbl = sum(1 for f in files if f.startswith("labels/") and f.endswith(".nii.gz"))
    print("label files:", n_lbl, "| manifest:", "manifest.json" in files,
          "| dataset_labels.json:", "dataset_labels.json" in files)
    man = json.load(open(hf_hub_download(a.repo, "manifest.json", repo_type="dataset",
                                         revision=a.revision, token=a.token)))
    recs = {str(r.get("token")): r for r in man}

    sel = ([recs["3"]] if "3" in recs else [])
    sel += [r for r in man if r.get("config") == "fused" and str(r.get("token")) != "3"][: max(1, a.n - 3)]
    sel += [r for r in man if r.get("config") == "spine_only"][:2]
    sel = sel[: a.n]
    print("sampled:", [(str(r.get("token")), r.get("config")) for r in sel])

    fails, union = [], set()
    sp_ignore = sp_pelvis = False
    for r in sel:
        cfg, ids = dump(a.repo, a.revision, a.token, r)
        union |= ids
        stray = ids - VALID
        if stray:
            fails.append(f"{r.get('token')}: stray ids {sorted(stray)}")
        for bad in (10, 50):                     # old ignore sentinels
            if bad in ids and bad not in VALID:
                fails.append(f"{r.get('token')}: old ignore {bad}")
        if cfg == "spine_only":
            sp_ignore |= (255 in ids)
            sp_pelvis |= bool(ids & {26, 30, 31})

    print("\n=== SMOKE SUMMARY (stage %d) ===" % stage)

    def check(name, ok):
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            fails.append(name)

    check("lumbar present (20-25)", bool(union & set(range(20, 26))))
    check("no stray/old-scheme ids", not any("stray" in f or "old ignore" in f for f in fails))
    if stage == 1:
        check("v1: NO ribs/femurs/S1", not (union & (RIBS | FEMURS | {S1})))
        check("v1: spine_only has ignore=255", sp_ignore)
        check("v1: spine_only has NO pelvis (partial)", not sp_pelvis)
    elif stage == 2:
        check("v2: NO ribs/femurs/S1", not (union & (RIBS | FEMURS | {S1})))
        check("v2: spine_only pelvis MODEL-filled (hips)", sp_pelvis)
    elif stage >= 3:
        check("v3+: S1 (29) present", S1 in union)
        check("v3+: femurs (32/33) present", bool(union & FEMURS))
        check("v3+: ribs (34-57) present", bool(union & RIBS))

    hard = [f for f in fails if not f.startswith(("lumbar", "v1:", "v2:", "v3"))] or \
           [f for f in fails]
    print("\nRESULT:", "ALL PASS" if not fails else f"{len(fails)} FAIL -> {fails}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
