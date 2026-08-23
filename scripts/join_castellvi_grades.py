"""scripts/join_castellvi_grades.py — attach the radiologist Castellvi grades to the release.

THE GAP THIS CLOSES. All 33 LSTV cases carry a radiologist Castellvi grade, five of them
with a second independent read, and they live in `_lstv_phenotypes.csv` at the repository
root -- untracked, joined to nothing. The released v5 manifest declares
`castellvi_type`, `castellvi_second_read` and `castellvi_agreement` and leaves all three
null in every one of the 802 records. So the grades exist and are not released, which is
the same failure mode as a declared-but-unpopulated label class: the schema promises
something the data does not contain, and only someone who checks finds out.

It also blocks work. `screen_missed_castellvi.py` cannot be validated as a Castellvi screen
without Castellvi grades, and has been scored against LSTV labels -- which describe a COUNT
where Castellvi describes a MORPHOLOGY -- with a header saying so.

THE JOIN IS BY TOKEN, NOT BY CASE ID, AND THERE IS NO FALLBACK. The phenotype file is
keyed by patient token; the morphometrics and label files are keyed by record id, and the
two are unrelated -- token 149 is record 0208. Zero-padding the token reproduces the record
id for exactly ONE record in 802, so a padded join is not an approximation of the right
answer, it is 32 grades landing on the wrong 32 patients.

An earlier version fell back to padding when the manifest was absent, printed a warning,
and wrote the file anyway. That is worse than failing: the output is the right shape, the
right length and silently wrong, and re-running it without the manifest present quietly
replaced a correct file with a corrupt one. The fallback is gone. Without a manifest this
script exits non-zero and writes nothing.

    python scripts/join_castellvi_grades.py --phenotypes _lstv_phenotypes.csv \\
        --manifest data/hf_export_v5/manifest.json --out morphometrics/castellvi_grades.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

VALID = {"Ia", "Ib", "IIa", "IIb", "IIIa", "IIIb", "IV"}


def load_manifest(path):
    """-> token -> [record ids]. Accepts the list or dict shapes the manifest has had."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    recs = d if isinstance(d, list) else d.get("records", list(d.values()))
    tok2rec = defaultdict(list)
    for r in recs:
        if not isinstance(r, dict):
            continue
        tok = str(r.get("token", "")).strip()
        lf = r.get("label_file") or r.get("ct_file") or ""
        rec = Path(str(lf)).name.split("_")[0]
        if tok and rec:
            tok2rec[tok].append(rec)
    return tok2rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phenotypes", default="_lstv_phenotypes.csv")
    ap.add_argument("--manifest", default="data/hf_export_v5/manifest.json")
    ap.add_argument("--out", default="morphometrics/castellvi_grades.csv")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.phenotypes)))
    print(f"  {len(rows)} phenotype row(s) in {a.phenotypes}")

    graded = [r for r in rows if (r.get("castellvi_type") or "").strip()]
    bad = [r for r in graded if r["castellvi_type"].strip() not in VALID]
    if bad:
        print(f"  ! {len(bad)} grade(s) outside Castellvi I-IV a/b: "
              f"{sorted({r['castellvi_type'] for r in bad})}")
    print(f"  {len(graded)} carry a grade; "
          f"{sum(1 for r in graded if (r.get('castellvi_second_read') or '').strip())} "
          f"carry a second read")

    if not Path(a.manifest).exists():
        print(f"  ! {a.manifest} not found, and there is no fallback.")
        print("  Zero-padding the token is right for 1 record in 802 -- it would put 32 of")
        print("  the 33 grades on the wrong patients, in a file of the correct shape.")
        print("  Fetch the manifest from the release and re-run.")
        return 1
    tok2rec = load_manifest(a.manifest)
    print(f"  manifest maps {len(tok2rec)} token(s) to records")

    out, unmatched = [], []
    for r in rows:
        tok = str(r.get("token", "")).strip()
        recs = tok2rec.get(tok) or tok2rec.get(tok.lstrip("0")) or []
        if not recs:
            unmatched.append(tok)
            continue
        first = (r.get("castellvi_type") or "").strip()
        second = (r.get("castellvi_second_read") or "").strip()
        for rec in recs:
            out.append({
                "case": rec,
                "token": tok,
                "phenotype_category": (r.get("category") or "").strip(),
                "n_non_rib_bearing_read": (r.get("non_rib_bearing_vertebrae") or "").strip(),
                "castellvi_type": first,
                "castellvi_second_read": second,
                # agreement is only meaningful where a second read exists; recording it as
                # blank rather than True keeps a single read from looking like a consensus
                "castellvi_agreement": ("" if not second
                                        else ("agree" if second == first else "disagree")),
                "notes": (r.get("notes") or "").strip(),
            })

    if unmatched:
        print(f"  ! {len(unmatched)} token(s) not in the manifest: {sorted(unmatched)}")
        print("  A grade with no record is a grade that cannot be released. Refusing.")
        return 1

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(out[0]))
        w.writeheader(); w.writerows(out)

    print(f"\n  {len(out)} record-level row(s) written to {a.out}")
    print(f"  grades: {dict(Counter(r['castellvi_type'] for r in out if r['castellvi_type']))}")
    second = [r for r in out if r["castellvi_second_read"]]
    if second:
        ag = Counter(r["castellvi_agreement"] for r in second)
        print(f"  second reads: {len(second)} record-level, {dict(ag)}")
        print("  NOTE: five second reads is not an inter-rater study. It is enough to say "
              "the grades were checked and not enough to quote a kappa.")
    print("\n  The released manifest declares castellvi_type and leaves it null in all 802")
    print("  records. Populating it from this file is what makes the paper's claim of a")
    print("  released annotation layer true.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
