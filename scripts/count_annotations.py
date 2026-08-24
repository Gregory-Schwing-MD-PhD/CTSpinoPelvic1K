"""scripts/count_annotations.py — how many cases each reviewer actually completed.

Authorship order on this dataset article is by contribution, and contribution here is a
countable thing: the review ledgers record, per case, which slot was filled and by whom.
This counts completed slots per reviewer across every review campaign so the order rests on
the ledgers rather than on recollection.

A CASE IS COUNTED ONCE PER REVIEWER PER CAMPAIGN. A reviewer who filled two slots on the
same case in the same campaign did one case of work, not two; a reviewer who reviewed the
same case in the rib campaign and again in the spine campaign did two. Counting raw slots
would reward whichever campaign happened to use more slots per case.

Only slots that were actually completed count. A slot with an assigned reviewer and no
submission is an assignment, not a contribution, and the ledgers contain both.

    python scripts/count_annotations.py --out morphometrics/annotation_counts.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

REPOS = [
    "CTSpinoPelvic1K-reviews-ribs",
    "CTSpinoPelvic1K-reviews-spine",
    "CTSpinoPelvic1K-reviews-nerve",
    "CTSpinoPelvic1K-reviews-ili",
    "CTSpinoPelvic1K-reviews-classfix",
    "CTSpinoPelvic1K-reviews-triaged",
]

DONE_KEYS = ("submitted", "completed", "done", "finished", "committed")

# Not people. auto:* are pipeline actions the ledger records the same way it records a
# reviewer -- pick-better chooses between two submitted reviews, qc-pass-passthrough accepts
# an unchanged label -- and anonymous-mlhc is the org service account. Counting them would
# put a cron job in the author list.
NOT_PEOPLE = ("auto:", "anonymous-mlhc", "bot:", "system")


def is_person(who):
    w = who.lower()
    return not any(w.startswith(x) or w == x.rstrip(":") for x in NOT_PEOPLE)


def slot_done(slot):
    """A slot counts only if it was actually submitted, not merely assigned."""
    if not isinstance(slot, dict):
        return False
    for k in DONE_KEYS:
        v = slot.get(k)
        if v is True:
            return True
        if isinstance(v, str) and v.strip():
            return True
    st = str(slot.get("status", "")).lower()
    return st in ("done", "submitted", "complete", "completed", "accepted")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--org", default="anonymous-mlhc")
    ap.add_argument("--out", default="morphometrics/annotation_counts.csv")
    ap.add_argument("--limit", type=int, default=0, help="files per repo, 0 = all")
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download
    tok = os.environ.get("HF_TOKEN")
    api = HfApi(token=tok)

    # reviewer -> campaign -> set of cases
    seen = defaultdict(lambda: defaultdict(set))
    assigned_only = defaultdict(int)

    for repo in REPOS:
        rid = f"{a.org}/{repo}"
        try:
            files = [f for f in api.list_repo_files(rid, repo_type="dataset", token=tok)
                     if f.startswith("cases/") and f.endswith(".json")]
        except Exception as e:                                        # noqa: BLE001
            print(f"  ! {repo}: {type(e).__name__} {str(e)[:70]}")
            continue
        if a.limit:
            files = files[: a.limit]
        print(f"  {repo}: {len(files)} case ledger(s)", flush=True)
        for i, f in enumerate(files, 1):
            try:
                p = hf_hub_download(rid, f, repo_type="dataset", token=tok)
                d = json.loads(Path(p).read_text(encoding="utf-8"))
            except Exception:                                         # noqa: BLE001
                continue
            case = Path(f).name.split("__")[0]
            # slots is a DICT keyed by slot number ("1", "2"), not a list. Iterating it
            # directly yields the keys, and slot.get would then fail on a string.
            raw = d.get("slots") or {}
            slots = list(raw.values()) if isinstance(raw, dict) else list(raw)
            for slot in slots:
                who = (slot.get("reviewer") or "").strip()
                if not who or not is_person(who):
                    continue
                if slot_done(slot):
                    seen[who][repo].add(case)
                else:
                    assigned_only[who] += 1
            if i % 200 == 0:
                print(f"    {i}/{len(files)}", flush=True)

    rows = []
    for who, per in seen.items():
        total = sum(len(v) for v in per.values())
        rows.append({
            "reviewer": who,
            "cases_total": total,
            "assigned_not_completed": assigned_only.get(who, 0),
            **{r.replace("CTSpinoPelvic1K-reviews-", ""): len(per.get(r, set()))
               for r in REPOS},
        })
    rows.sort(key=lambda r: -r["cases_total"])

    if not rows:
        print("  ! no completed slots found; check the ledger schema")
        return 1

    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print(f"\n  {len(rows)} reviewer(s), by cases completed:")
    for r in rows:
        print(f"    {r['cases_total']:>5}  {r['reviewer']}")
    print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
