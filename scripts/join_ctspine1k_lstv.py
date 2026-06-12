"""
join_ctspine1k_lstv.py — map CTSpine1K's published LSTV cohort onto our tokens.

CTSpine1K open-sources the IDs of the transitional cases its radiologists
adjudicated (the [check] in their Table 1). The COLONOG entries are canonical
patient UIDs that map straight onto our tokens (patient_db.canonical_uid /
patient_token), so we can:

  * pull the radiologist-adjudicated LSTV cohort joined to our tokens, and
  * CROSS-CHECK it against our own lstv labels — every disagreement is either a
    bug in our derivation or a case worth a second look. Since we've decided to
    trust CTSpine1K full stop, their call wins; this lists where ours differs.

"Sacral Lumbarization" = lumbarization = a real L6 (extra mobile vertebra).
"Lumbar Sacralization"  = sacralization = the last lumbar fused into the sacrum.

VerSe / MSD-liver entries are listed too but NOT auto-joined (different token
scheme); they're reported so you can map them by hand if those sources are in
your tree.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from patient_db import canonical_uid, patient_token  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
log = logging.getLogger("join_ctspine1k_lstv")

# ── CTSpine1K published LSTV lists (open-source release) ──────────────────────
# value = filename stem of the seg; we classify the source from its shape.
LUMBARIZATION = [          # "Sacral Lumbarization" — a true L6
    "liver_106_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0004_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0067_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0149_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0167_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0175_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0189_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0215_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0261_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0267_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0344_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0401_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0587_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0666_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0672_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0699_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0737_seg.nii.gz",
    "verse506_CT-iso_seg.nii.gz", "verse519_CT-iso_seg.nii.gz", "verse532_seg.nii.gz",
    "verse539_CT-iso_seg.nii.gz", "verse542_CT-iso_seg.nii.gz", "verse565_CT-iso_seg.nii.gz",
    "verse586_CT-iso_seg.nii.gz", "verse619_CT-iso_seg.nii.gz",
]
SACRALIZATION = [          # "Lumbar Sacralization" — last lumbar into the sacrum
    "liver_83_seg.nii.gz", "liver_93_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0064_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0104_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0107_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0110_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0537_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0554_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0555_seg.nii.gz", "1.3.6.1.4.1.9328.50.4.0615_seg.nii.gz",
    "1.3.6.1.4.1.9328.50.4.0721_seg.nii.gz",
    "verse100_seg.nii.gz", "verse584_seg.nii.gz", "verse594_seg.nii.gz",
]


def _classify(stem: str) -> Tuple[str, Optional[str]]:
    """(source, token) for a CTSpine1K seg filename. token only for COLONOG."""
    base = stem.replace("_seg.nii.gz", "")
    if base.startswith("1.3.6.1.4.1.9328.50."):
        return "colonog", patient_token(canonical_uid(base))
    if base.startswith("verse"):
        return "verse", None
    if base.startswith("liver"):
        return "msd_liver", None
    return "other", None


def _load_manifest(path: Path) -> Dict[str, dict]:
    data = json.loads(path.read_text())
    recs = data if isinstance(data, list) else data.get("records", [])
    out: Dict[str, dict] = {}
    for r in recs:
        t = r.get("token")
        if t is not None:
            out.setdefault(str(t), r)           # first record per token is fine
    return out


def _our_call(rec: Optional[dict]) -> str:
    """Best-effort current LSTV label from a manifest record."""
    if rec is None:
        return ""
    for k in ("lstv_label", "lstv_vertebral", "lstv"):
        v = rec.get(k)
        if v:
            return str(v).lower()
    if rec.get("has_l6") is True:
        return "lumbarization(has_l6)"
    return "normal?" if rec else ""


def _agrees(ctspine: str, ours: str) -> str:
    o = ours.lower()
    if not o:
        return "no-label"
    if ctspine == "lumbarization":
        return "OK" if ("lumbar" in o or "l6" in o) else "MISMATCH"
    if ctspine == "sacralization":
        return "OK" if "sacral" in o else "MISMATCH"
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--manifest", type=Path, default=None,
                    help="manifest.json to join + cross-check (token->record).")
    ap.add_argument("--out_csv", type=Path, default=None,
                    help="write the joined cohort here.")
    a = ap.parse_args()

    manifest = _load_manifest(a.manifest) if a.manifest else {}
    rows: List[dict] = []
    for kind, names in (("lumbarization", LUMBARIZATION),
                        ("sacralization", SACRALIZATION)):
        for stem in names:
            source, token = _classify(stem)
            rec = manifest.get(token) if (token and manifest) else None
            in_ds = token is not None and token in manifest
            ours = _our_call(rec)
            rows.append({
                "ctspine1k_id": stem.replace("_seg.nii.gz", ""),
                "source": source, "ctspine1k_lstv": kind,
                "token": token or "", "in_dataset": in_ds,
                "config": (rec or {}).get("config", ""),
                "our_lstv": ours,
                "agreement": _agrees(kind, ours) if in_ds else "",
            })

    # ── summary ──────────────────────────────────────────────────────────────
    from collections import Counter
    by_src = Counter((r["source"], r["ctspine1k_lstv"]) for r in rows)
    colonog = [r for r in rows if r["source"] == "colonog"]
    in_ds = [r for r in colonog if r["in_dataset"]]
    mism = [r for r in in_ds if r["agreement"] == "MISMATCH"]
    nolab = [r for r in in_ds if r["agreement"] == "no-label"]

    log.info("=" * 70)
    log.info("CTSpine1K LSTV cohort  (%d lumbarization + %d sacralization)",
             len(LUMBARIZATION), len(SACRALIZATION))
    for (src, kind), n in sorted(by_src.items()):
        log.info("  %-10s %-14s : %d", src, kind, n)
    log.info("-" * 70)
    if manifest:
        log.info("COLONOG entries                : %d", len(colonog))
        log.info("  └─ present in this dataset   : %d", len(in_ds))
        log.info("  └─ agree with our LSTV label : %d",
                 sum(1 for r in in_ds if r["agreement"] == "OK"))
        log.info("  └─ MISMATCH (ours != theirs) : %d  %s",
                 len(mism), [r["token"] for r in mism])
        log.info("  └─ we have NO lstv label     : %d  %s",
                 len(nolab), [r["token"] for r in nolab])
        log.info("  trusting CTSpine1K: the MISMATCH/no-label tokens are the "
                 "ones to correct to their call.")
    else:
        log.info("(no --manifest given — source breakdown only; pass a manifest "
                 "to join to tokens + cross-check)")

    if a.out_csv:
        a.out_csv.parent.mkdir(parents=True, exist_ok=True)
        with open(a.out_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        log.info("wrote %d rows -> %s", len(rows), a.out_csv)
    log.info("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
