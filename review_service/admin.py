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

    removed_adj = any(r["slot"] == schema.ADJ_SLOT for r in removed)
    adj_done = slots.get(schema.ADJ_SLOT, {}).get("done")
    # Removing the adjudicator slot invalidates the `final` it produced — drop it
    # so the case reverts to needs_adjudication (the primaries' agree/irr stand).
    if removed_adj:
        case.pop("final", None)
    # Removing a primary below the double-review floor (with no adjudication)
    # invalidates the agreement basis too.
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


def cmd_queue(a) -> int:
    """Everything not yet finalized, and why. Read-only.

    The ledger records state but nothing ever printed it, so 'what is still
    outstanding' meant opening case JSONs by hand. Distinguishes an EXPIRED
    claim (reclaimable by anyone — `claimable_primary_slot` frees it
    automatically, no admin action needed) from a HELD one (still the
    reviewer's), because those need opposite responses."""
    repo = a.repo or os.environ.get("REVIEW_REPO")
    token = a.token or os.environ.get("HF_TOKEN")
    if not repo or not token:
        sys.exit("need --repo/REVIEW_REPO and --token/HF_TOKEN.")
    store = store_mod.ReviewStore(store_mod.HFBackend(repo_id=repo, token=token))
    now = schema.utcnow()

    cases = store.list_cases()
    by_status, open_n = {}, 0
    print(f"repo {repo} | {len(cases)} cases | now {now[:19]}\n")
    for case in sorted(cases, key=lambda c: c["case_id"]):
        st = schema.derive_status(case)
        by_status[st] = by_status.get(st, 0) + 1
        if st == "finalized" and not a.all:
            continue
        open_n += 1
        tags = []
        if case.get("needs_expert"):
            tags.append("NEEDS_EXPERT")
        if case.get("agree") is False:
            tags.append("disagree")
        if case.get("deferred_by"):
            tags.append("deferred_by=" + ",".join(case["deferred_by"]))
        print(f"{case['case_id']:22s} {st:18s} {'  '.join(tags)}")
        for slot, s in sorted((case.get("slots") or {}).items()):
            exp = str(s.get("expires_at") or "")[:19]
            if s.get("done"):
                state = "done"
            elif s.get("amend"):
                state = "AMEND"          # theirs to fix; never auto-reclaimed
            elif exp and exp < now:
                state = "EXPIRED"        # free for anyone to claim, no action needed
            else:
                state = "held"
            print(f"    slot {slot:3s} {str(s.get('reviewer')):20s} {state:8s} "
                  f"exp={exp}  decision={s.get('decision')}")
        if not (case.get("slots") or {}):
            print("    (no slots — unassigned)")
    print(f"\nstatus counts: {by_status}")
    print(f"open: {open_n}")
    return 0


def plan_expert_take(case: dict, expert: str, roster) -> dict:
    """Pure: hand a flagged case to the EXPERT, without opening it to students.

    A student flags a transitional level they must not decide; `flag()` sets
    needs_expert, which `claim()` skips — correct, but it means the case is
    then served to NOBODY. The docstring on flag() says 'Greg reads the flags
    via the expert queue'; that queue was never built, so flagged cases just
    accumulate. This is that queue, expressed in mechanics that already exist
    rather than a new trust path in the service:

      needs_expert cleared      -> claim() will consider it again
      every OTHER reviewer into deferred_by
                                -> claimable_primary_slot() returns None for
                                   them, so clearing the flag cannot leak the
                                   case back to a student
      priority raised           -> the expert is served it before anything else

    The reread history is kept: the reason the student flagged it is what the
    expert needs to read. Returns a NEW case dict.
    """
    case = json.loads(json.dumps(case))
    case.pop("needs_expert", None)
    dby = case.setdefault("deferred_by", [])
    for r in sorted(roster):
        if r and r.lower() != expert.lower() and r not in dby:
            dby.append(r)
    # the expert must NOT be deferred, or the case is claimable by no one at all
    case["deferred_by"] = [r for r in dby if r.lower() != expert.lower()]
    case["priority"] = max(int(case.get("priority", 0)), 1_000_000)
    return case


def cmd_expert_take(a) -> int:
    repo = a.repo or os.environ.get("REVIEW_REPO")
    token = a.token or os.environ.get("HF_TOKEN")
    if not repo or not token:
        sys.exit("need --repo/REVIEW_REPO and --token/HF_TOKEN (the WRITE token).")
    store = store_mod.ReviewStore(store_mod.HFBackend(repo_id=repo, token=token))

    # Roster from the LEDGER, not a hardcoded list: a reviewer who never appears
    # here cannot be blocked, and one that is invented would block nothing.
    roster = {s.get("reviewer") for c in store.list_cases()
              for s in (c.get("slots") or {}).values() if s.get("reviewer")}
    print(f"roster from ledger: {sorted(roster)}\n")

    now = schema.utcnow()
    planned = False
    for cid in a.cases:
        case = store.get_case(cid)
        if case is None:
            print(f"[skip] {cid}: no such case in {repo}")
            continue
        new_case = plan_expert_take(case, a.expert, roster)
        slot = schema.claimable_primary_slot(new_case, a.expert, now=now)
        print(f"{cid}: needs_expert {bool(case.get('needs_expert'))} -> False | "
              f"priority {case.get('priority', 0)} -> {new_case['priority']} | "
              f"status {schema.derive_status(case)}")
        for r in (case.get("reread") or []):
            print(f"    flagged by {r.get('by')}: {r.get('reason')}")
        if slot is None:
            # Almost always: the expert already holds a slot on this case, and
            # double-review distinctness forbids a second. Say so plainly instead
            # of writing a change that silently achieves nothing.
            why = ("expert already holds a slot here"
                   if any(s.get("reviewer") == a.expert
                          for s in (new_case.get("slots") or {}).values())
                   else "no primary slot is open (both done?)")
            print(f"    !! {a.expert} still could NOT claim it: {why}")
        else:
            print(f"    -> {a.expert} will be served slot {slot}")
        planned = True
        if a.apply:
            store.put_case(new_case)
            print("    committed")
    if not a.apply and planned:
        print("\nDRY RUN — nothing written. Re-run with --apply to commit.")
    return 0


def cmd_set_priority(a) -> int:
    """Set a case's claim priority. The server serves UNASSIGNED cases highest-
    priority-first, so a big number makes `reviewtool next` hand out that case
    next — handy to force a specific case for a demo/tutorial."""
    repo = a.repo or os.environ.get("REVIEW_REPO")
    token = a.token or os.environ.get("HF_TOKEN")
    if not repo or not token:
        sys.exit("need --repo/REVIEW_REPO and --token/HF_TOKEN (the WRITE token).")
    backend = store_mod.HFBackend(repo_id=repo, token=token)
    store = store_mod.ReviewStore(backend)
    for cid in a.cases:
        case = store.get_case(cid)
        if case is None:
            print(f"[skip] {cid}: no such case in {repo}")
            continue
        old = case.get("priority", 0)
        print(f"{cid}: priority {old} -> {a.priority}  "
              f"[status now: {schema.derive_status(case)}]")
        if a.apply:
            case["priority"] = a.priority
            store.put_case(case)
            print(f"  committed (it will be served next while UNASSIGNED)")
    if not a.apply:
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

    p = sub.add_parser("queue", help="print every case that is not finalized, and why")
    p.add_argument("--all", action="store_true", help="include finalized cases too")
    p.add_argument("--repo", default=None, help="review repo (or REVIEW_REPO env)")
    p.add_argument("--token", default=None, help="HF token (or HF_TOKEN env)")
    p.set_defaults(fn=cmd_queue)

    p = sub.add_parser("expert-take",
                       help="hand a needs_expert case to the expert WITHOUT "
                            "reopening it to students")
    p.add_argument("cases", nargs="+", help="case ids, e.g. 155__fused")
    p.add_argument("--expert", required=True, help="HF username of the radiologist")
    p.add_argument("--repo", default=None, help="review repo (or REVIEW_REPO env)")
    p.add_argument("--token", default=None, help="HF write token (or HF_TOKEN env)")
    p.add_argument("--apply", action="store_true",
                   help="actually commit (default is a dry run)")
    p.set_defaults(fn=cmd_expert_take)

    p = sub.add_parser("set-priority",
                       help="set a case's claim priority (higher = served sooner; "
                            "force a specific case for a demo)")
    p.add_argument("cases", nargs="+", help="case ids, e.g. 103__fused")
    p.add_argument("--priority", type=int, default=1_000_000,
                   help="priority value (default 1000000 — served first)")
    p.add_argument("--repo", default=None, help="review repo (or REVIEW_REPO env)")
    p.add_argument("--token", default=None, help="HF write token (or HF_TOKEN env)")
    p.add_argument("--apply", action="store_true",
                   help="actually commit (default is a dry run)")
    p.set_defaults(fn=cmd_set_priority)

    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
