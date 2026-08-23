"""scripts/populate_castellvi_manifest.py -- write the Castellvi grades into the manifest.

The released v5 manifest declares `castellvi_type`, `castellvi_second_read` and
`castellvi_agreement` and leaves all three null in every one of the 802 records, while the
grades themselves sit in `_lstv_phenotypes.csv`, joined to nothing. A schema that promises
a field it never fills is worse than one that omits it: it reads as an annotation layer to
anybody who lists the columns, and only somebody who checks the values finds out.

This fills them from `morphometrics/castellvi_grades.csv` and changes nothing else.

WHAT STAYS NULL AND WHY. A grade is written only for the 33 records that carry one. The
other 769 keep null, and null here means UNGRADED, not "no transitional vertebra" -- these
are colonography scans read for polyps, and a transitional vertebra is easy to pass over.
Writing "none" into those 769 would manufacture 769 negatives nobody established, which is
the same mistake `screen_missed_castellvi.py` exists to avoid.

`castellvi_agreement` stays blank where there is no second read, so that a single read is
never mistaken for a consensus. Five second reads is enough to say the grades were checked
and nowhere near enough to quote a kappa; two of the five disagree.

    python scripts/populate_castellvi_manifest.py --check     # report, write nothing
    python scripts/populate_castellvi_manifest.py
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

FIELDS = ("castellvi_type", "castellvi_second_read", "castellvi_agreement")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="data/hf_export_v5/manifest.json")
    ap.add_argument("--grades", default="morphometrics/castellvi_grades.csv")
    ap.add_argument("--out", default=None, help="default: in place")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    a = ap.parse_args()

    mp = Path(a.manifest)
    if not mp.exists():
        print(f"  ! {mp} not found. It is not in the repo; pull it from the release.")
        return 1

    doc = json.loads(mp.read_text(encoding="utf-8"))
    recs = doc if isinstance(doc, list) else doc.get("records", list(doc.values()))

    by_tok = {}
    for r in csv.DictReader(open(a.grades)):
        if (r.get("castellvi_type") or "").strip():
            by_tok[str(r["token"]).strip()] = r

    before = sum(1 for r in recs if (r.get("castellvi_type") or "").strip())
    hit, changed = 0, 0
    for r in recs:
        g = by_tok.get(str(r.get("token", "")).strip())
        if not g:
            continue
        hit += 1
        for f in FIELDS:
            v = (g.get(f) or "").strip() or None
            if r.get(f) != v:
                r[f] = v
                changed += 1

    missed = sorted(set(by_tok) - {str(r.get("token", "")).strip() for r in recs})
    print(f"  {len(recs)} record(s); {len(by_tok)} graded token(s)")
    print(f"  matched {hit}, wrote {changed} field value(s), was {before} non-null")
    if missed:
        print(f"  ! {len(missed)} graded token(s) absent from the manifest: {missed}")

    after = [r for r in recs if (r.get("castellvi_type") or "").strip()]
    print(f"  grades now in manifest: {dict(Counter(r['castellvi_type'] for r in after))}")
    print(f"  {len(recs) - len(after)} record(s) stay null, meaning UNGRADED, not negative")

    if a.check:
        print("\n  --check: nothing written")
        return 0
    if missed:
        print("\n  ! refusing to write while a graded case is unmatched; fix the join first")
        return 1

    out = Path(a.out or a.manifest)
    out.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    print(f"\n  wrote {out}")
    print("  The release itself is unchanged until this manifest is uploaded.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
