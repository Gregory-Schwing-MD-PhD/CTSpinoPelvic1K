"""
review_service/admin.py — maintenance ops that need the dataset WRITE token.

These are intentionally NOT exposed over the API (a reviewer must not be able
to reset their own work). Run locally by the project owner, who holds the
review repo's HF write token.

  python3 -m review_service.admin reset-slot CASE_ID [CASE_ID ...] \
      --reviewer USERNAME [--slot 1|2|adj] [--delete-files] [--apply]

Default is a DRY RUN — it prints the plan and writes nothing. Add --apply to
commit. Repo/token come from --repo/--token or REVIEW_REPO/HF_TOKEN env.

`reset-slot` removes a reviewer's claimed/submitted slot from each case so the
case returns to 'unassigned' and can be claimed + reviewed afresh. Use it to
undo a bogus submission — e.g. an 'accept' recorded by `resume` when ITK-SNAP
never actually opened the case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import store as store_mod          # noqa: E402  (sibling)
from review import schema          # noqa: E402


def plan_reset(case: dict, reviewer: Optional[str], slot: Optional[str]):
    """Pure: compute (updated_case, removed) for a slot reset.

    A slot is removed when it matches the filter: if both `reviewer` and
    `slot` are given, BOTH must match; otherwise whichever is given matches.
    Returns a NEW case dict (the input is not mutated) and a list of
    {"slot","reviewer","decision","label_path","review_id"} describing what was
    removed. Clears case-level agree/irr/final when the basis for them is gone
    (fewer than N_PRIMARY primaries done and no completed adjudication)."""
    case = json.loads(json.dumps(case))          # deep copy
    slots = case.get("slots", {})

    def _matches(k: str, s: dict) -> bool:
        rev_ok = reviewer is not None and \
            str(s.get("reviewer", "")).lower() == reviewer.lower()
        slot_ok = slot is not None and k == slot
        if reviewer is not None and slot is not None:
            return rev_ok and slot_ok
        return rev_ok or slot_ok

    removed = []
    for k in list(slots.keys()):
        s = slots[k]
        if _matches(k, s):
            removed.append({"slot": k, "reviewer": s.get("reviewer"),
                            "decision": s.get("decision"),
                            "label_path": s.get("label_path"),
                            "review_id": s.get("review_id")})
            del slots[k]

    adj_done = slots.get(schema.ADJ_SLOT, {}).get("done")
    if len(schema.primary_done(case)) < schema.N_PRIMARY and not adj_done:
        for k in ("agree", "irr", "final"):
            case.pop(k, None)
    return case, removed


def _orphan_files(case_id: str, removed: list) -> list:
    """Repo paths of the now-unreferenced review record + label for a removal."""
    out = []
    for r in removed:
        if r.get("label_path"):
            out.append(r["label_path"])
        if r.get("review_id"):
            out.append(f"reviews/{case_id}/{r['review_id']}.json")
    return out


def cmd_reset_slot(a) -> int:
    repo = a.repo or os.environ.get("REVIEW_REPO")
    token = a.token or os.environ.get("HF_TOKEN")
    if not repo or not token:
        sys.exit("need --repo/REVIEW_REPO and --token/HF_TOKEN "
                 "(the review repo's WRITE token).")
    if not a.reviewer and not a.slot:
        sys.exit("specify --reviewer and/or --slot to select which slot to reset.")

    backend = store_mod.HFBackend(repo_id=repo, token=token)
    store = store_mod.ReviewStore(backend)
    from huggingface_hub import CommitOperationAdd, CommitOperationDelete

    any_planned = False
    for cid in a.cases:
        case = store.get_case(cid)
        if case is None:
            print(f"[skip] {cid}: no such case in {repo}")
            continue
        new_case, removed = plan_reset(case, a.reviewer, a.slot)
        if not removed:
            print(f"[skip] {cid}: no matching slot (already clean?)")
            continue
        any_planned = True
        desc = ", ".join(f"slot {r['slot']} (reviewer={r['reviewer']}, "
                         f"decision={r['decision']})" for r in removed)
        print(f"{cid}: remove {desc}  "
              f"[{schema.derive_status(case)} -> {schema.derive_status(new_case)}]")

        ops = [CommitOperationAdd(
            path_in_repo=store.case_path(cid),
            path_or_fileobj=json.dumps(new_case, indent=2).encode("utf-8"))]
        if a.delete_files:
            for f in _orphan_files(cid, removed):
                if backend.exists(f):
                    print(f"    delete {f}")
                    ops.append(CommitOperationDelete(path_in_repo=f))
                else:
                    print(f"    (missing, skip delete) {f}")

        if not a.apply:
            continue
        backend.api.create_commit(
            repo_id=repo, repo_type="dataset", token=token, operations=ops,
            commit_message=f"admin: reset slot(s) on {cid}")
        print(f"  committed reset for {cid}")

    if not a.apply and any_planned:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="review_service.admin", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("reset-slot",
                       help="remove a reviewer's slot from cases (-> unassigned)")
    p.add_argument("cases", nargs="+", help="case ids, e.g. 101__spine_only")
    p.add_argument("--reviewer", default=None, help="reviewer username to reset")
    p.add_argument("--slot", default=None, choices=["1", "2", "adj"],
                   help="specific slot to reset (combined with --reviewer = AND)")
    p.add_argument("--repo", default=None, help="review repo (or REVIEW_REPO env)")
    p.add_argument("--token", default=None, help="HF write token (or HF_TOKEN env)")
    p.add_argument("--delete-files", action="store_true",
                   help="also delete the orphaned review record + label blobs")
    p.add_argument("--apply", action="store_true",
                   help="actually commit (default is a dry run)")
    p.set_defaults(fn=cmd_reset_slot)
    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
