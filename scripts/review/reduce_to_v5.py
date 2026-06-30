"""
scripts/review/reduce_to_v5.py — fuse anatomy-pure rib + spine reviews into a v5 tree.

v4 is the base. Two SEPARATE review cohorts each fix ONE anatomy:
  * rib cohort  (TASK=rib_fix, the 165 dup cases)  -> corrects ribs   (ids 34-57)
  * spine cohort (the pelvic_native pseudo-spine)   -> corrects spine  (ids 1-29)
Because the id ranges are DISJOINT (spine 1-29, hips/femurs 30-33, ribs 34-57), the fuse is a
conflict-free per-voxel overlay:

    v5 = v4
    if a rib  review exists for the case:  replace ids 34-57 with the rib-corrected ribs
    if a spine review exists for the case: replace ids 1-29  with the spine-corrected spine
    (hips/femurs 30-33 always stay v4 — they are pelvic GT and reviewed by neither cohort)

Anatomy purity is ENFORCED here, not trusted: we only ever copy a cohort's OWN id range, so a
reviewer who strayed outside their anatomy can't leak into v5.

REBASE caveat: a spine correction must be VerSe-native (same scheme as v4: lumbar 20-25,
thoracic 8-19, sacrum 26, S1 29). If the spine cohort were run on the legacy v2 pseudo (ids
1-9), its ids would mean different bones — so we GUARD: a spine label whose spine ids are all
<= 9 is treated as v2-scheme and SKIPPED (with a warning) rather than corrupting v5. Run the
spine cohort on v4 (SOURCE_REVISION=v4) and this never trips.

The fuse itself (fuse_label) is a pure function and unit-tested; main() is the file I/O.

finals index (per cohort, same shape as reduce_to_v3), one entry per finalized case_id:
  { "<token>__<config>": { "decision": "corrected"|"accept"|"reject",
                           "label_rel": "reviews/<...>/final_label.nii.gz" }, ... }
only `corrected` (with label_rel) overlays; `accept`/`reject`/missing leave that anatomy as v4.
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/ for label_scheme
import label_scheme as LS          # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.review.reduce_v5")

RIB_IDS = list(range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13))   # 34..57
SPINE_IDS = list(range(1, LS.S1_ID + 1))                                  # 1..29 (vert+sacrum+S1)
#                                                          hips/femurs 30-33 are deliberately NOT
#                                                          in either range -> always kept from v4.


def is_verse_spine(arr: np.ndarray) -> bool:
    """True if `arr`'s spine ids look VerSe-native (v4), not the legacy v2 1-9 scheme.
    A lumbosacral spine in v4 carries lumbar ids 20-25 / thoracic 8-19; a v2 label tops out at
    9. So: any spine id >= 10 -> VerSe. No spine ids at all -> treat as VerSe (nothing to add)."""
    present = set(int(v) for v in np.unique(arr) if 1 <= int(v) <= LS.S1_ID)
    if not present:
        return True
    return max(present) >= 10


def fuse_label(v4: np.ndarray, rib: Optional[np.ndarray] = None,
               spine: Optional[np.ndarray] = None) -> np.ndarray:
    """v5 = v4 with ribs (34-57) taken from `rib` and spine (1-29) from `spine` where given.
    Disjoint id ranges -> no conflict. hips/femurs (30-33) always stay v4."""
    out = v4.copy()
    if rib is not None:
        out[np.isin(out, RIB_IDS)] = 0                     # clear v4 ribs
        m = np.isin(rib, RIB_IDS)
        out[m] = rib[m]                                    # write the rib-cohort ribs
    if spine is not None:
        out[np.isin(out, SPINE_IDS)] = 0                   # clear v4 spine
        m = np.isin(spine, SPINE_IDS)
        out[m] = spine[m]                                  # write the spine-cohort spine
    return out


def _load_manifest(p: Path):
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def _corrected_path(finals: dict, cid: str, root: Optional[Path]) -> Optional[Path]:
    """Resolve a cohort's corrected label for a case, or None (accept/reject/missing)."""
    f = finals.get(cid)
    if not f or f.get("decision") != "corrected" or not f.get("label_rel"):
        return None
    return (root / f["label_rel"]) if root else Path(f["label_rel"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v4", required=True, type=Path, help="v4 export tree (ct/ + labels/ + manifest.json)")
    ap.add_argument("--rib_finals", type=Path, help="rib cohort finalized-reviews index JSON")
    ap.add_argument("--rib_root", type=Path, help="local root for rib label_rel (pulled rib ledger)")
    ap.add_argument("--spine_finals", type=Path, help="spine cohort finalized-reviews index JSON")
    ap.add_argument("--spine_root", type=Path, help="local root for spine label_rel (pulled spine ledger)")
    ap.add_argument("--out", required=True, type=Path, help="v5 tree to create")
    ap.add_argument("--labels_only", action="store_true", help="skip copying CT volumes (labels + manifest only)")
    a = ap.parse_args()

    records = _load_manifest(a.v4 / "manifest.json")
    rib_finals = json.loads(a.rib_finals.read_text()) if a.rib_finals else {}
    spine_finals = json.loads(a.spine_finals.read_text()) if a.spine_finals else {}
    rib_root = a.rib_root or (a.rib_finals.parent if a.rib_finals else None)
    spine_root = a.spine_root or (a.spine_finals.parent if a.spine_finals else None)

    for sub in ("ct", "labels"):
        (a.out / sub).mkdir(parents=True, exist_ok=True)
    for extra in ("manifest.json", "splits_5fold.json", "splits_summary.json",
                  "dataset_interface.py", "README.md"):
        if (a.v4 / extra).exists():
            shutil.copy2(str(a.v4 / extra), str(a.out / extra))

    n = n_rib = n_spine = n_skip_v2 = n_ct = 0
    for rec in records:
        cid = LS_case_id(rec)
        lf = rec.get("label_file")
        if not lf or not (a.v4 / lf).exists():
            continue
        v4_img = nib.load(str(a.v4 / lf))
        v4 = np.asanyarray(v4_img.dataobj)

        rib = None
        rp = _corrected_path(rib_finals, cid, rib_root)
        if rp and rp.exists():
            rib = np.asanyarray(nib.load(str(rp)).dataobj); n_rib += 1
        elif rp:
            log.warning("%s: rib correction missing at %s — keeping v4 ribs", cid, rp)

        spine = None
        sp = _corrected_path(spine_finals, cid, spine_root)
        if sp and sp.exists():
            s = np.asanyarray(nib.load(str(sp)).dataobj)
            if is_verse_spine(s):
                spine = s; n_spine += 1
            else:
                n_skip_v2 += 1
                log.warning("%s: spine correction is NOT VerSe-native (v2 ids?) — SKIPPING "
                            "spine overlay; run the spine cohort on v4", cid)
        elif sp:
            log.warning("%s: spine correction missing at %s — keeping v4 spine", cid, sp)

        out = fuse_label(v4, rib, spine)
        outp = a.out / lf
        outp.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(out.astype(np.int16), v4_img.affine, v4_img.header), str(outp))
        n += 1

        if not a.labels_only:
            cf = rec.get("ct_file")
            if cf and (a.v4 / cf).exists() and not (a.out / cf).exists():
                (a.out / cf).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(a.v4 / cf), str(a.out / cf)); n_ct += 1

    log.info("=" * 60)
    log.info("v5 tree -> %s", a.out)
    log.info("  labels written : %d  (rib-overlaid=%d, spine-overlaid=%d)", n, n_rib, n_spine)
    if n_skip_v2:
        log.info("  spine SKIPPED  : %d (non-VerSe / v2-scheme corrections)", n_skip_v2)
    log.info("  ct copied      : %d", n_ct)
    log.info("=" * 60)
    log.info("Publish:  HF_TOKEN=... HF_REPO_ID=org/Name HF_REVISION=v5 \\")
    log.info("            HF_EXPORT_DIR=%s make hf-push", a.out)
    return 0


def LS_case_id(rec: dict) -> str:
    """case_id used by the review ledgers (token__config), via review.schema."""
    from review import schema
    return schema.case_id(rec.get("token"), rec.get("config"))


if __name__ == "__main__":
    raise SystemExit(main())
