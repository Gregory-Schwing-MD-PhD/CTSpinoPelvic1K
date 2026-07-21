"""
scripts/spine_fix.py — the RADIOLOGIST corrects the spine directly. No HF Space: you are the single
source of truth for the vertebra GT, so there is no review / IRR / adjudication and no force-restore.

    pull   -> stage each flagged case (CT + editable label) in spine_fix_work/<case>/
    (edit)  -> open <case>/label.nii.gz over ct.nii.gz in ITK-SNAP; fix the spine only
    push   -> upload changed labels back to the dataset @v4 + write spine_fixes.csv (change log)

Flagged cases = union of:
  * missing_vertebra.csv   (interior gaps + upper-thoracic-in-FOV)      <- from qc_v4_ribs_and_verts
  * student_vertebra_work.csv rows with kind == "L6/transitional"       <- radiologist read

    HF_TOKEN=... python scripts/spine_fix.py pull   [--only 7,233,...] [--dry-run]
    HF_TOKEN=... python scripts/spine_fix.py push   [--dry-run]

Runs on the Windows box, but if the socket pool wedges use WSL (see the network note).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from huggingface_hub import HfApi, hf_hub_download

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "review"))
import label_scheme as LS            # noqa: E402
import review_anatomy_qc as RA       # noqa: E402
import schema                        # noqa: E402  (scripts/review/schema.py: case_id)

DS = os.environ.get("V2_REPO", "anonymous-mlhc/CTSpinoPelvic1K")
REV = "v4"
WORK = Path("spine_fix_work")


def _flagged_tokens(only=None):
    """{token: reason} across the two flag sources (or just --only)."""
    if only:
        return {t: "manual" for t in only}
    reasons = {}
    if Path("missing_vertebra.csv").exists():
        for r in csv.DictReader(open("missing_vertebra.csv")):
            why = []
            if int(r.get("n_gaps", 0) or 0) > 0:
                why.append(f"interior-gap:{r['interior_gaps']}")
            if r.get("ribs_above_spine"):
                why.append(f"rib-above-spine:{r['ribs_above_spine']}")
            reasons.setdefault(r["token"], "; ".join(why) or "missing-vertebra")
    if Path("student_vertebra_work.csv").exists():
        for r in csv.DictReader(open("student_vertebra_work.csv")):
            if r.get("kind") == "L6/transitional":
                t = r["case"].rsplit("__", 1)[0]
                prev = reasons.get(t, "")
                reasons[t] = (prev + "; " if prev else "") + f"L6/transitional (student added {r['added_verts']})"
    return reasons


def _manifest():
    mp = hf_hub_download(DS, "manifest.json", repo_type="dataset", revision=REV,
                         token=os.environ.get("HF_TOKEN"))
    recs = json.load(open(mp))
    recs = recs if isinstance(recs, list) else recs.get("records", [])
    return {str(r.get("token")): r for r in recs}


def cmd_pull(a) -> int:
    reasons = _flagged_tokens(a.only)
    by = _manifest()
    WORK.mkdir(exist_ok=True)
    todo = []
    n = skip = miss = 0
    for t, why in sorted(reasons.items()):
        r = by.get(t)
        if not r or not r.get("ct_file") or not r.get("label_file"):
            miss += 1; print(f"  ! {t}: no ct/label in manifest — skipped"); continue
        cid = schema.case_id(t, r.get("config"))
        d = WORK / cid; d.mkdir(parents=True, exist_ok=True)
        todo.append({"case": cid, "token": t, "reason": why,
                     "label_file": r["label_file"], "ct_file": r["ct_file"]})
        if a.dry_run:
            print(f"  [dry] {cid:26s} <- {why[:70]}"); continue
        for kind, rel in (("ct", r["ct_file"]), ("label", r["label_file"])):
            dst = d / f"{kind}.nii.gz"
            if dst.exists():
                skip += 1; continue
            p = hf_hub_download(DS, rel, repo_type="dataset", revision=REV,
                                token=os.environ.get("HF_TOKEN"))
            dst.write_bytes(Path(p).read_bytes())
        n += 1
        print(f"  staged {cid:26s} {why[:60]}", flush=True)
    (WORK / "TODO.csv").write_text(
        "case,token,reason,label_file\n" +
        "\n".join(f'{r["case"]},{r["token"]},"{r["reason"]}",{r["label_file"]}' for r in todo))
    print(f"\n{'[dry] ' if a.dry_run else ''}{len(todo)} cases flagged  (downloaded {n}, cached {skip}, missing {miss})")
    print(f"edit each spine_fix_work/<case>/label.nii.gz over ct.nii.gz in ITK-SNAP, then: spine_fix.py push")
    return 0


def cmd_push(a) -> int:
    by = _manifest()
    todo = list(csv.DictReader(open(WORK / "TODO.csv")))
    api = HfApi(token=os.environ.get("HF_TOKEN"))
    names = RA._id2name()
    ops = []; log = []
    from huggingface_hub import CommitOperationAdd
    for row in todo:
        cid = row["case"]; edited = WORK / cid / "label.nii.gz"
        if not edited.exists():
            continue
        # compare to the current v4 label; only push if the SPINE actually changed
        cur = np.asanyarray(nib.load(hf_hub_download(DS, row["label_file"], repo_type="dataset",
                                                     revision=REV, token=os.environ.get("HF_TOKEN"))).dataobj)
        new = np.asanyarray(nib.load(str(edited)).dataobj)
        if cur.shape != new.shape:
            print(f"  ! {cid}: shape mismatch — skipped"); continue
        spine = ((cur >= 1) & (cur <= 33)) | ((new >= 1) & (new <= 33))
        changed = int(((cur != new) & spine).sum())
        if changed == 0:
            continue                                  # not edited (or ribs-only) -> leave it
        before = {int(v) for v in np.unique(cur[(cur >= 1) & (cur <= 25)])}
        after = {int(v) for v in np.unique(new[(new >= 1) & (new <= 25)])}
        added = sorted(names.get(v, v) for v in (after - before))
        removed = sorted(names.get(v, v) for v in (before - after))
        log.append({"case": cid, "voxels_changed": changed,
                    "vertebrae_added": " ".join(map(str, added)),
                    "vertebrae_removed": " ".join(map(str, removed))})
        if not a.dry_run:
            ops.append(CommitOperationAdd(path_in_repo=row["label_file"],
                                          path_or_fileobj=str(edited)))
        print(f"  {cid:26s} +{changed:>7} vox  added={added} removed={removed}")
    if not log:
        print("no spine changes detected — nothing to push"); return 0
    with open("spine_fixes.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "voxels_changed", "vertebrae_added", "vertebrae_removed"])
        w.writeheader(); w.writerows(log)
    if a.dry_run:
        print(f"\n[dry] would push {len(ops)} corrected label(s) to {DS}@{REV}; log -> spine_fixes.csv")
        return 0
    api.create_commit(repo_id=DS, repo_type="dataset", revision=REV, operations=ops,
                      commit_message=f"spine GT: radiologist corrected {len(ops)} vertebra label(s)")
    print(f"\npushed {len(ops)} corrected spine label(s) -> {DS}@{REV}; log -> spine_fixes.csv")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pull"); p.add_argument("--only", type=lambda s: s.split(",")); \
        p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_pull)
    p = sub.add_parser("push"); p.add_argument("--dry-run", action="store_true"); p.set_defaults(fn=cmd_push)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
