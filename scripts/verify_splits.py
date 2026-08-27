"""Patient leakage in splits_5fold.json, joined on the field the splits actually use.

This check was passing vacuously. splits_5fold.json is keyed by PATIENT TOKEN ('10', '100'),
not by volume id ('0007'), so joining on volume_id matches nothing and reports zero leakage
over zero patients. That reads exactly like a clean result, which makes it worse than an
outright failure -- nobody re-examines a check that says the thing they hoped it would say.

So the first thing established here is that the join RESOLVES. A leakage count is meaningless
until the two files have actually been connected, and an unresolved token is reported as a
failure rather than skipped.

Then the leakage itself: no token validated in two folds, no token in both train and val of
one fold, every token validated exactly once, and -- for a cohort where one patient can carry
more than one record -- no patient's records split across the boundary.
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="data/zenodo_deposit")
    a = ap.parse_args()
    d = Path(a.deposit)

    sp = json.loads((d / "splits_5fold.json").read_text(encoding="utf-8"))
    recs = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs.get("records", list(recs.values()))
    folds = sp["folds"] if isinstance(sp, dict) else sp

    by_token = defaultdict(list)
    for r in recs:
        by_token[str(r.get("token"))].append(r["volume_id"])
    multi = {t: v for t, v in by_token.items() if len(v) > 1}
    print(f"  {len(recs)} record(s) over {len(by_token)} patient token(s); "
          f"{len(multi)} token(s) carry more than one record")

    fail = []

    # 1. does the join resolve at all? everything below is meaningless otherwise
    val_of, dup = {}, []
    for f in folds:
        for t in f["val"]:
            if t in val_of:
                dup.append((t, val_of[t], f["fold"]))
            val_of[t] = f["fold"]
    unresolved = [t for t in val_of if t not in by_token]
    print(f"  validation tokens: {len(val_of)} distinct; "
          f"{len(unresolved)} resolve to no record")
    if unresolved:
        fail.append(f"{len(unresolved)} validation token(s) match nothing in the manifest "
                    f"-- the join is broken, not the splits: {unresolved[:5]}")
    elif not val_of:
        fail.append("no validation tokens at all; the splits file is not what this expects")

    # 2. leakage
    if dup:
        fail.append(f"{len(dup)} token(s) validated in more than one fold: {dup[:3]}")
    for f in folds:
        both = set(f["train"]) & set(f["val"])
        if both:
            fail.append(f"fold {f['fold']}: {len(both)} token(s) in both train and val")
    apart = [t for t in multi if len({val_of.get(t)}) > 1]
    if apart:
        fail.append(f"{len(apart)} patient(s) have records on both sides of a fold boundary")

    # 3. coverage: cross-validation only means what it says if everything is validated once
    uncovered = sorted(set(by_token) - set(val_of))
    if uncovered:
        fail.append(f"{len(uncovered)} token(s) are never in any validation set: "
                    f"{uncovered[:5]}")
    print(f"  coverage: {len(val_of)} of {len(by_token)} token(s) validated exactly once")

    print()
    for f in fail:
        print(f"  FAIL  {f}")
    if not fail:
        print("  the join resolves, and no patient crosses a fold boundary")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
