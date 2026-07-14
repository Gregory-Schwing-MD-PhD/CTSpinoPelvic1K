"""
scripts/restore_spine.py — keep the students' RIBS, restore the radiologist's SPINE.

THE PROBLEM. 58 of 191 rib submissions altered the spine: 14 renumbered it (inserting an L6 or
deleting a level), 6 re-labelled it wholesale, the rest moved vertebra voxels. Rib reviewers were
told not to touch the vertebrae -- the radiologist's labels are the ground truth of this dataset --
and a renumbered spine cascades: shift the vertebrae and every rib number shifts with them.

THE FIX -- and it costs the students nothing. Their RIB work is what we asked for and it is good;
only the spine edits are invalid. So instead of kicking 58 people back to redo work, we simply take
their ribs (ids 34-57) and superimpose them onto the pristine v4 label, which already carries the
radiologist's spine:

    out = v4_label            (radiologist spine + pelvis, untouched)
    out[ribs] = 0             (drop the v4 auto-ribs)
    out[student_ribs] = ...   (paste the student's ribs -- ONLY onto background)

The SPINE WINS every conflict: a student rib voxel is never written over a vertebra. The result is
the radiologist's spine, by construction, plus the annotator's rib corrections.

WHAT THIS DOES NOT DO. It does not touch the rib NUMBERING. A student who inserted an L6 probably
shifted their rib numbers to match it, so their ribs may now be off-by-one against the restored
spine. That is a genuine disagreement about rib numbering and it still goes to ADJUDICATION -- which
is exactly where a human belongs.

THE RAW SUBMISSION IS PRESERVED. The original is archived to <slot>_label_raw.nii.gz before the
normalised label replaces it, and every change is written to a CSV manifest (case, slot, annotator,
spine voxels restored, rib voxels kept, git commit). Nothing is silently rewritten.

  python scripts/restore_spine.py            # DRY RUN: report what would change
  python scripts/restore_spine.py --apply
"""
from __future__ import annotations

import argparse
import csv
import io
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent / "review_service"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import store as store_mod            # noqa: E402
import label_scheme as LS            # noqa: E402
from huggingface_hub import hf_hub_download   # noqa: E402

REPO = os.environ.get("REVIEW_REPO", "anonymous-mlhc/CTSpinoPelvic1K-reviews-ribs")
DS = os.environ.get("V2_REPO", "anonymous-mlhc/CTSpinoPelvic1K")
LO, HI = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
TOL = 5000                            # ignore trivial drift; a real edit is orders of magnitude more


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(_HERE.parent), text=True).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def restore(given: np.ndarray, sub: np.ndarray):
    """Radiologist spine (from `given`) + annotator ribs (from `sub`). Spine always wins."""
    out = given.copy()
    out[(out >= LO) & (out <= HI)] = 0                   # drop the v4 auto-ribs
    student = (sub >= LO) & (sub <= HI)
    place = student & (out == 0)                         # never overwrite a vertebra/pelvis voxel
    out[place] = sub[place]
    nonrib = (given >= 1) & (given <= 33)
    spine_fixed = int((((given != sub) & (nonrib | ((sub >= 1) & (sub <= 33)))).sum()))
    return out, {"spine_voxels_restored": spine_fixed,
                 "rib_voxels_kept": int(place.sum()),
                 "rib_voxels_dropped_over_spine": int((student & ~place).sum())}


def _to_bytes(arr, ref):
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "l.nii.gz"
        nib.save(nib.Nifti1Image(arr.astype(np.asanyarray(ref.dataobj).dtype),
                                 ref.affine, ref.header), str(p))
        return p.read_bytes()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write (default: dry run)")
    ap.add_argument("--manifest", default="spine_restored_manifest.csv")
    a = ap.parse_args(argv)

    tok = os.environ["HF_TOKEN"]
    commit = _commit()
    store = store_mod.ReviewStore(store_mod.HFBackend(repo_id=REPO, token=tok))
    rows, files = [], {}
    n_fixed = 0

    cases = [c for c in store.list_cases() if c.get("region_to_review") == "ribs"]
    print(f"scanning {len(cases)} rib cases (commit {commit})\n", flush=True)
    for i, case in enumerate(cases):
        cid = case["case_id"]
        try:
            gimg = nib.load(hf_hub_download(DS, case["pseudo_label_file"], repo_type="dataset",
                                            token=tok, revision="v4"))
            given = np.asanyarray(gimg.dataobj)
        except Exception:                                # noqa: BLE001
            continue
        # The FINAL label matters most: it is the dataset's OUTPUT, and it is derived from a student
        # submission -- so if the annotator whose label was chosen had tampered with the spine, that
        # tampering is baked into the finalized case. Fix the finals as well as the submissions.
        targets = [(s, f"reviews/{cid}/{s}_label.nii.gz") for s in ("1", "2")
                   if case.get("slots", {}).get(s)]
        fin_rel = (case.get("final") or {}).get("label_rel")
        if fin_rel:
            targets.append(("final", fin_rel))

        for slot, rel in targets:
            sl = case.get("slots", {}).get(slot) or {}
            try:
                simg = nib.load(hf_hub_download(REPO, rel, repo_type="dataset", token=tok))
                sub = np.asanyarray(simg.dataobj)
            except Exception:                            # noqa: BLE001
                continue
            if sub.shape != given.shape:
                continue
            nonrib = ((given >= 1) & (given <= 33)) | ((sub >= 1) & (sub <= 33))
            changed = int(((given != sub) & nonrib).sum())
            if changed <= TOL:
                continue                                  # spine untouched -> nothing to do
            out, st = restore(given, sub)
            n_fixed += 1
            rev = sl.get("reviewer") or (case.get("final") or {}).get("chosen_reviewer") or "FINAL"
            rows.append({"case": cid, "slot": slot, "reviewer": rev,
                         "spine_voxels_restored": st["spine_voxels_restored"],
                         "rib_voxels_kept": st["rib_voxels_kept"],
                         "rib_voxels_dropped_over_spine": st["rib_voxels_dropped_over_spine"],
                         "commit": commit})
            print(f"  RESTORE {cid}/{slot} {rev}: spine {st['spine_voxels_restored']:,} vox "
                  f"restored, {st['rib_voxels_kept']:,} rib vox kept", flush=True)
            if a.apply:
                stem = rel[:-len(".nii.gz")]
                files[f"{stem}_raw.nii.gz"] = _to_bytes(sub, simg)   # archive the original
                files[rel] = _to_bytes(out, simg)                                        # normalised
                if len(files) >= 12:
                    store.b.write_many(files, commit_message=f"restore radiologist spine ({commit})")
                    files = {}
        if i % 20 == 0:
            print(f"    ...{i}/{len(cases)}", flush=True)

    if a.apply and files:
        store.b.write_many(files, commit_message=f"restore radiologist spine ({commit})")
    if rows:
        with open(a.manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
    print(f"\n{'APPLIED' if a.apply else 'DRY RUN'}: {n_fixed} submissions had the spine restored "
          f"(ribs kept). manifest -> {a.manifest}")
    if not a.apply:
        print("  nothing written. re-run with --apply.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
