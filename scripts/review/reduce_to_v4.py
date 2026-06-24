"""
scripts/review/reduce_to_v4.py — merge finalized v4 overlay reviews into a v4 tree.

v4 = v3 (bone) + the human overlay tasks (ribs, LS nerves, iliolumbar), each
reviewed in its OWN Space/ledger. Unlike reduce_to_v3 (which SWAPS a whole
corrected label), this MERGES each task's overlay classes onto the v3 base:

  * ribs        -> ids 34-57 (rib_left_1..12 / rib_right_1..12)  [VerSe-native dataset ids]
  * iliolumbar  -> ids 58/59
  * ls_nerve    -> ids 60-65
  * ignore      -> 255 (already 255 in v3; no relocation)

Only the OVERLAY-class voxels are taken from each task's final, and only where the
merged label is still background — so overlays are purely additive and the v3 base
structures (VerSe-native spine + pelvis) are preserved verbatim from the v3 tree (a student's incidental
tidy-ups to base classes are NOT folded here). A case with no overlay finals passes
through as its v3 label with ignore relocated, so v4 is a strict superset of v3.

Per-task finals index (same shape as reduce_to_v3), one entry per finalized case:
  { "<token>__<config>": { "decision": "corrected"|"accept"|"reject",
        "label_rel": "reviews/<...>/final_label.nii.gz", "final_review_id": ... } }
A task contributes its overlay iff decision != "reject" and label_rel exists.

Usage (one --finals / --labels-root per task ledger you pulled):
  python scripts/review/reduce_to_v4.py \
      --v3 data/hf_export_v3 \
      --finals ribs=pull/ribs/finals.json \
      --finals ls_nerve=pull/nerve/finals.json \
      --finals iliolumbar=pull/ili/finals.json \
      --labels-root ribs=pull/ribs --labels-root ls_nerve=pull/nerve \
      --labels-root iliolumbar=pull/ili \
      --out data/hf_export_v4
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # scripts/ for label_scheme
import schema  # noqa: E402
import label_scheme as LS  # noqa: E402  (THE single source of truth for label ids)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.review.reduce_v4")

V3_IGNORE, V4_IGNORE = LS.IGNORE_LABEL, LS.IGNORE_LABEL   # both 255 (no relocation needed)
# overlay tasks folded into v4 (rib_anchor is a separate minimal LSTV pass, not a
# v4 dataset structure, so it is intentionally NOT merged here).
V4_OVERLAY_TASKS = ("ribs", "iliolumbar", "ls_nerve")
TASK_PROV_KEY = {t: f"prov_{t}" for t in V4_OVERLAY_TASKS}


def v4_label_dict() -> Dict[str, int]:
    """The full v4 {name: id} map for dataset.json — VerSe-native (label_scheme).

    v4 has the same legend as v3 (the overlay structures already have reserved ids in
    label_scheme); the overlays just go from reserved-but-empty to populated.
    """
    return LS.label_dict()


def apply_overlays_to_records(records: List[dict],
                              task_finals: Dict[str, Dict[str, dict]]
                              ) -> List[dict]:
    """Pure: stamp per-task provenance ('manual') on each record that received an
    overlay. Records pass through 1:1 (v4 is a superset of v3 — nothing dropped)."""
    out: List[dict] = []
    for rec in records:
        cid = schema.case_id(rec.get("token"), rec.get("config"))
        r = dict(rec)
        for task in V4_OVERLAY_TASKS:
            f = task_finals.get(task, {}).get(cid)
            if f and f.get("decision") != "reject" and f.get("label_rel"):
                r[TASK_PROV_KEY[task]] = "manual"
        out.append(r)
    return out


def _kv(pairs: List[str], what: str) -> Dict[str, Path]:
    """Parse repeated 'task=path' CLI args into {task: Path}, validating tasks."""
    d: Dict[str, Path] = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--{what} expects task=path, got {p!r}")
        task, path = p.split("=", 1)
        if task not in V4_OVERLAY_TASKS:
            raise SystemExit(f"--{what} unknown task {task!r}; "
                             f"expected one of {V4_OVERLAY_TASKS}")
        d[task] = Path(path)
    return d


def _load_manifest(p: Path) -> List[dict]:
    data = json.loads(p.read_text())
    if isinstance(data, dict):
        data = data.get("records", data.get("cases", []))
    return [r for r in data if isinstance(r, dict)]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v3", required=True, type=Path,
                    help="v3 export tree (data/hf_export_v3).")
    ap.add_argument("--finals", action="append", default=[], metavar="TASK=PATH",
                    help="finalized-reviews index per task (repeatable).")
    ap.add_argument("--labels-root", action="append", default=[],
                    metavar="TASK=PATH",
                    help="root each task index's label_rel resolves against "
                         "(repeatable; default: that --finals file's parent).")
    ap.add_argument("--out", required=True, type=Path,
                    help="v4 tree to create (data/hf_export_v4).")
    ap.add_argument("--labels_only", action="store_true",
                    help="skip copying CT volumes — labels + manifest only.")
    args = ap.parse_args()

    import numpy as np
    import nibabel as nib

    v3, out = args.v3, args.out
    finals_paths = _kv(args.finals, "finals")
    roots = _kv(args.labels_root, "labels-root")
    task_finals = {t: json.loads(p.read_text()) for t, p in finals_paths.items()}
    labels_root = {t: roots.get(t, finals_paths[t].parent) for t in finals_paths}

    records = _load_manifest(v3 / "manifest.json")
    new_records = apply_overlays_to_records(records, task_finals)
    got = {t: sum(1 for r in records
                  if task_finals[t].get(schema.case_id(r.get("token"),
                                                        r.get("config")), {})
                  .get("decision") != "reject"
                  and task_finals[t].get(schema.case_id(r.get("token"),
                                                        r.get("config")), {})
                  .get("label_rel"))
           for t in task_finals}
    log.info("reduce v4: %d records; overlays folded: %s", len(records), got)

    for sub in ("ct", "labels"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    for extra in ("splits_5fold.json", "splits_summary.json",
                  "dataset_interface.py", "README.md"):
        if (v3 / extra).exists():
            shutil.copy2(str(v3 / extra), str(out / extra))

    n_ct = n_lbl = 0
    merged_counts = {t: 0 for t in task_finals}
    for rec in new_records:
        lbl_rel = rec.get("label_file")
        ct_rel = rec.get("ct_file")
        if not lbl_rel or not (v3 / lbl_rel).exists():
            continue
        cid = schema.case_id(rec.get("token"), rec.get("config"))

        img = nib.load(str(v3 / lbl_rel))
        base = np.asarray(img.dataobj).astype(np.int32)
        merged = base.copy()
        merged[merged == V3_IGNORE] = V4_IGNORE          # relocate ignore

        for task in V4_OVERLAY_TASKS:
            f = task_finals.get(task, {}).get(cid)
            if not f or f.get("decision") == "reject" or not f.get("label_rel"):
                continue
            src = labels_root[task] / f["label_rel"]
            if not src.exists():
                log.warning("%s: overlay missing %s — skipping", cid, src)
                continue
            ov = np.asarray(nib.load(str(src)).dataobj).astype(np.int32)
            if ov.shape != base.shape:
                log.warning("%s: %s overlay shape %s != base %s — skipping",
                            cid, task, ov.shape, base.shape)
                continue
            for oid in schema.OVERLAY_CLASSES[task]:     # additive, bg only
                m = (ov == oid) & (merged == 0)
                if m.any():
                    merged[m] = oid
            merged_counts[task] += 1

        out_lbl = out / lbl_rel
        out_lbl.parent.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(merged.astype(np.uint16), img.affine, img.header),
                 str(out_lbl))
        n_lbl += 1
        if not args.labels_only and ct_rel and (v3 / ct_rel).exists() \
                and not (out / ct_rel).exists():
            (out / ct_rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(str(v3 / ct_rel), str(out / ct_rel))
            n_ct += 1

    (out / "dataset_labels.json").write_text(json.dumps(v4_label_dict(), indent=2))

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))      # scripts/
    from export_hf import write_manifest
    write_manifest([{**r, "ok": True} for r in new_records], out)

    log.info("=" * 60)
    log.info("v4 tree -> %s", out)
    log.info("  records      : %d", len(new_records))
    log.info("  labels/ct    : %d / %d written", n_lbl, n_ct)
    log.info("  overlays      : %s cases merged per task", merged_counts)
    log.info("  ignore        : relocated %d -> %d", V3_IGNORE, V4_IGNORE)
    log.info("=" * 60)
    log.info("Publish:  HF_TOKEN=... HF_REPO_ID=org/Name HF_REVISION=v4 \\")
    log.info("            HF_EXPORT_DIR=%s make hf-push", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
