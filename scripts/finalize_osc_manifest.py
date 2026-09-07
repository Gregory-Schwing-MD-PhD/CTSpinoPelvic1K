"""finalize_osc_manifest.py -- make the shipped manifest match what the card claims.

Three defects in the v6 manifest, all of them things a reader checks first:

  The card says all transitional cases carry an expert Castellvi grade. The manifest
  declares `castellvi_type` / `castellvi_second_read` / `castellvi_agreement` and leaves
  all three None in every one of the 802 records, and the grading sheet is not shipped at
  all. The grades exist -- in _lstv_phenotypes.csv -- so this fills the declared fields
  from it rather than deleting the claim.

  `postwrite_hip_bone_pct` is None in all 802 records, yet the card spends a paragraph on
  it as a QC diagnostic with a stated failure threshold. A field that is never populated
  cannot be a diagnostic; it is dropped here and the paragraph goes with it.

  `qc_file` points at `qc/<token>_qc.png`. No such file exists in the tree or on the
  cluster, so every record carries a path that resolves to nothing. Dropped.

Nothing clinical is reinterpreted. In particular `lstv_class` is left exactly as it is,
including the two cases a radiologist graded semi-sacralization that it calls normal --
that disagreement is reported in the card and queued for re-read, not silently patched.

    python finalize_osc_manifest.py --tree data/hf_export_v6 \
        --phenotypes _lstv_phenotypes.csv --out-dir data/hf_export_v6
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

# Populated from the grading sheet; declared in the schema since v1, never filled.
CASTELLVI_FIELDS = ("castellvi_type", "castellvi_second_read", "castellvi_agreement")
# Declared, never populated in any record, and documented as if it were live.
DEAD_FIELDS = ("postwrite_hip_bone_pct", "qc_file")


def norm(tok: str) -> str:
    """Manifest tokens are unpadded ('5'); the sheet's are too. Normalise both."""
    return str(tok).strip().lstrip("0") or "0"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tree", required=True)
    ap.add_argument("--phenotypes", required=True)
    ap.add_argument("--out-dir", required=True)
    a = ap.parse_args()

    tree, out = Path(a.tree), Path(a.out_dir)
    records = json.loads((tree / "manifest.json").read_text())

    grades = {}
    for row in csv.DictReader(open(a.phenotypes, newline="", encoding="utf-8")):
        grades[norm(row["token"])] = row

    n_graded = n_second = n_agree = 0
    dropped = {f: 0 for f in DEAD_FIELDS}
    disagreements = []

    for rec in records:
        for f in DEAD_FIELDS:
            if f in rec:
                dropped[f] += 1
                del rec[f]

        g = grades.get(norm(rec["token"]))
        if not g:
            continue

        first = (g.get("castellvi_type") or "").strip() or None
        second = (g.get("castellvi_second_read") or "").strip() or None
        rec["castellvi_type"] = first
        rec["castellvi_second_read"] = second
        rec["castellvi_agreement"] = (first == second) if second else None
        rec["castellvi_notes"] = (g.get("notes") or "").strip() or None
        rec["lstv_phenotype"] = (g.get("category") or "").strip() or None
        rec["non_rib_bearing_vertebrae"] = (
            int(g["non_rib_bearing_vertebrae"])
            if (g.get("non_rib_bearing_vertebrae") or "").strip()
            else None
        )

        n_graded += 1
        if second:
            n_second += 1
            n_agree += first == second
        # graded transitional by a radiologist, but lstv_class calls it normal
        if rec.get("lstv_class") == 0:
            disagreements.append((rec["token"], rec["lstv_phenotype"], first))

    (out / "manifest.json").write_text(json.dumps(records, indent=1))

    # the card has referenced manifest.csv since v1; it has never shipped
    cols: list[str] = []
    for rec in records:
        for k in rec:
            if k not in cols:
                cols.append(k)
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for rec in records:
            w.writerow(rec)

    print(f"  records            : {len(records)}")
    print(f"  castellvi populated: {n_graded}")
    print(f"  double-read        : {n_second} (exact agreement {n_agree})")
    for f, n in dropped.items():
        print(f"  dropped {f:24s}: {n} records")
    print(f"  columns in csv     : {len(cols)}")
    print(f"  graded-but-lstv_class-0: {disagreements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
