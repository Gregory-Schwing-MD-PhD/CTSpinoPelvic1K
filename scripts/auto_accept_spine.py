"""
scripts/auto_accept_spine.py — auto-finalize the SPINE-review cases that do NOT need a human.

THE PROBLEM. The SPINE-EXTENSION queue (TASK=spine_extend, CHECK=spine_extend) serves a student the
v4 label so they can ADD any thoracic vertebra that is in the field of view but not yet labelled. Many
enqueued cases carry NOTHING for a student to do: the served label already passes the live gate
(`spine_extend_qc`) and its only caveat is a vertebra CLIPPED BY THE TOP/BOTTOM OF THE SCAN — FOV
truncation, which is un-fixable and now emits an ADVISORY "note ... (FOV-truncated)" rather than a
block (see scripts/review_anatomy_qc.py: structure_integrity / spine_extend_qc). A student would open
such a case, see it pass, and submit an empty accept. This pass does that trivial accept for them.

THE RULE (safety-critical — be conservative). For each pending spine case, load the served label +
affine and run the EXACT live gate the spine-extension Space uses:

    RA.check_label("spine_extend", lab, affine, given=lab)   ==   spine_extend_qc(lab, affine, given)

AUTO-ACCEPT the case iff BOTH:
  1. the QC returns ok == True (nothing blocking), AND
  2. every message is either an "OK ..." line or an advisory "note ..." line that is specifically
     about FOV truncation ("FOV-truncated" / "clipped by the top/bottom of the scan" / "clipped by
     the scan edge").
ANY "X " blocking line, OR any note that is NOT the FOV-truncation advisory (anything implying real
work is still needed), leaves the case for a person. We only skip cases that pass and whose sole
caveat is the un-fixable scan edge; everything else still goes to a human.

The served (base) label is passed THROUGH to the final unchanged — hence `by: auto:fov-truncation-
passthrough` — plus the git commit, so nothing is silently passed off as a human review. Raw
student/base submissions under reviews/<case>/<slot>_label.nii.gz are NEVER mutated.

  python scripts/auto_accept_spine.py            # DRY RUN: report the split, write nothing
  python scripts/auto_accept_spine.py --apply    # finalize the passthrough cases + write the lists
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent / "review_service"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import store as store_mod            # noqa: E402
import review_anatomy_qc as RA       # noqa: E402
from review import schema            # noqa: E402

# The spine-extension Space (review_service/deploy_v4_spaces.py: TASK=spine_extend, CHECK=spine_extend)
# runs the SAME gate; mirror it exactly so this pass never accepts something the live server would block.
SPINE_CHECK = "spine_extend"

# The review ledger (cases + finals) for the spine-extension task. One Space/ledger per task, so this
# is NOT the rib ledger; must be provided (no silent cross-task default).
REPO = os.environ.get("REVIEW_REPO")
# The dataset repo + revision that hold the SERVED base label (case["pseudo_label_file"]); the
# spine-extension Space serves v4 (SOURCE_REVISION=v4). Mirrors review_service/app.py + service._given_label.
V2_REPO = os.environ.get("V2_REPO", "org/CTSpinoPelvic1K")
SOURCE_REVISION = os.environ.get("SOURCE_REVISION", "v4")

# A message is a benign FOV-truncation advisory iff it is a "note" AND names the scan edge. Every
# FOV note the QC emits contains "(FOV-truncated)"; the other two phrasings are kept for robustness
# and to match the reviewer-facing wording exactly.
FOV_MARKERS = (
    "fov-truncated",
    "clipped by the top/bottom of the scan",
    "clipped by the scan edge",
    "clipped by the edge of the scan",
)


# ── the core rule (pure; unit-tested; no numpy/HF needed) ─────────────────────

def _is_fov_note(msg: str) -> bool:
    """True iff `msg` is an advisory note whose ONLY subject is FOV truncation (the un-fixable scan
    edge). Requires both the 'note' prefix AND an explicit scan-edge phrase, so a substantive note
    (e.g. an LSTV / ambiguity flag) is never mistaken for the benign one."""
    low = msg.strip().lower()
    if not low.startswith("note"):                       # "note " / "note:"
        return False
    return any(mk in low for mk in FOV_MARKERS)


def is_auto_acceptable(ok: bool, msgs) -> bool:
    """Decide passthrough vs human for one case's QC result. THE core rule (see module docstring).

    Accept iff the gate passed AND every line is either an "OK ..." summary or an FOV-truncation
    advisory. Any "X " blocker, or any other note/informational line (which implies real work), is a
    REJECT -> the case stays for a person. Conservative by construction: the default is REJECT."""
    ok_flag, _ = classify(ok, msgs)
    return ok_flag


def classify(ok: bool, msgs):
    """(auto_acceptable, reason). Split out so the dry run can print WHY a case fell to a human."""
    if not ok:
        return False, "blocking: gate ok == False"
    for m in msgs or ():
        s = (m or "").strip()
        if not s:
            continue
        if s.startswith("X"):                            # a blocker (defensive; ok==True implies none)
            return False, f"blocking finding: {s}"
        if s.lower().startswith("ok"):                   # "OK ..." summary line
            continue
        if _is_fov_note(s):                              # benign scan-edge advisory
            continue
        return False, f"non-FOV note -> needs a human: {s}"   # anything else = real work
    return True, "passes; sole caveat is FOV truncation (or no findings beyond OK)"


# ── queue enumeration + QC (the network-touching part, kept separate) ─────────

def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(_HERE.parent), text=True).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


_TASK_GATE = {"spine": "spine_extend", "class_mixing": "class_mixing"}


def pending_cases(store) -> list:
    """Both spine-region queues in this ledger, still-pending: the SPINE-EXTENSION task
    (task='spine', gate spine_extend_qc) and the CLASS-MIXING FIX task (task='class_mixing', gate
    class_mixing_qc). Both stamp region_to_review='spine'. The FOV order-exemption lives in
    spine_sanity, which BOTH gates use, so an FOV-only block clears in either. Idempotent:
    already-finalized / excluded cases are skipped so a re-run never re-finalizes."""
    out = []
    for c in store.list_cases():
        if c.get("task") not in _TASK_GATE or c.get("region_to_review") != "spine":
            continue
        if schema.derive_status(c) in ("finalized", "excluded"):
            continue
        out.append(c)
    return out


def _served_label(token: str, tok: str):
    """Fetch the base label the Space serves for this case (nib image + array).

    Uses a TIMEOUT-BOUNDED streaming download (connect 10s, read 30s/chunk, 3 tries) instead of
    hf_hub_download, whose lack of a read timeout hangs forever on a stalled WSL/HF socket. On
    exhausting retries it raises -> the case is marked 'load failed' and left for a human, so one
    bad connection can never freeze the whole run."""
    import tempfile
    import requests
    url = f"https://huggingface.co/datasets/{V2_REPO}/resolve/{SOURCE_REVISION}/{token}"
    headers = {"Authorization": f"Bearer {tok}"} if tok else {}
    last = None
    for _ in range(3):
        try:
            r = requests.get(url, headers=headers, stream=True, timeout=(10, 30))
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
                for chunk in r.iter_content(chunk_size=1 << 20):
                    if chunk:
                        f.write(chunk)
                p = f.name
            img = nib.load(p)
            return img, np.asanyarray(img.dataobj)
        except Exception as e:                           # noqa: BLE001
            last = e
            continue
    raise last if last else RuntimeError("download failed")


def evaluate_case(case: dict, tok: str):
    """(auto_acceptable, reason, msgs, img, lab) for one case, or (None, err, [], None, None) on load
    failure. Runs the case's OWN live gate on the served base label (given == lab: no student edits) --
    spine_extend for the spine task, class_mixing for the class-mixing task."""
    gate = _TASK_GATE.get(case.get("task"))
    if gate is None:
        return None, f"unknown task {case.get('task')!r}", [], None, None
    rel = case.get("pseudo_label_file")
    if not rel:
        return None, "no pseudo_label_file on case", [], None, None
    try:
        img, lab = _served_label(rel, tok)
    except Exception as e:                               # noqa: BLE001
        return None, f"load failed: {str(e)[:60]}", [], None, None
    # given=None: we screen the SERVED BASE itself (no student edit), so the no-renumber guard
    # (given vs lab, an O(33 x full-volume ndimage.label) cost) is a semantic no-op here and is
    # skipped. The gate still runs the one-piece / order (FOV-exempt) / no-fused-duplicate checks,
    # which is exactly "is the base already clean?" -- and it's ~100x faster per case.
    ok, msgs = RA.check_label(gate, lab, img.affine, given=None)
    accept, reason = classify(ok, msgs)
    return accept, reason, msgs, img, lab


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write finals (default: dry run)")
    ap.add_argument("--manifest", default="auto_accept_spine_manifest.csv",
                    help="CSV of the auto-accepted (passthrough) cases")
    ap.add_argument("--escalate", default="spine_needs_human.csv",
                    help="CSV of the cases left for a human + the deciding QC line")
    ap.add_argument("--sample", type=int, default=6, help="cases to show per side in the dry run")
    a = ap.parse_args(argv)

    if not REPO:
        ap.error("set REVIEW_REPO to the spine-extension review ledger (one Space/ledger per task; "
                 "this is NOT the rib ledger)")
    tok = os.environ["HF_TOKEN"]
    commit = _commit()
    store = store_mod.ReviewStore(store_mod.HFBackend(repo_id=REPO, token=tok))

    cases = pending_cases(store)
    _bt = Counter(c.get("task") for c in cases)
    print(f"{len(cases)} pending cases  (commit {commit}; by task: {dict(_bt)})\n", flush=True)

    acc_rows, esc_rows = [], []
    pending: dict = {}
    for i, case in enumerate(cases):
        cid = case["case_id"]
        accept, reason, msgs, img, lab = evaluate_case(case, tok)
        if accept is None:                               # couldn't load -> leave for a human
            print(f"  [skip] {cid}: {reason}", flush=True)
            esc_rows.append({"case": cid, "reason": reason, "commit": commit})
            continue
        if accept:
            acc_rows.append({"case": cid, "reason": reason, "commit": commit})
            if a.apply:
                pending[cid] = (case, lab, img)
                if len(pending) >= 8:
                    _flush(store, pending, commit)
        else:
            esc_rows.append({"case": cid, "reason": reason, "commit": commit})
        if i % 25 == 0:
            print(f"    ...{i}/{len(cases)}", flush=True)

    if a.apply and pending:
        _flush(store, pending, commit)

    # dry-run sample: a few of each side + the line that decided them
    print(f"\n  AUTO-ACCEPT (passthrough) sample:")
    for r in acc_rows[:a.sample]:
        print(f"    + {r['case']}: {r['reason']}")
    print(f"\n  NEEDS A HUMAN sample:")
    for r in esc_rows[:a.sample]:
        print(f"    - {r['case']}: {r['reason']}")

    if a.apply:
        for rows, path in ((acc_rows, a.manifest), (esc_rows, a.escalate)):
            if rows:
                with open(path, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader(); w.writerows(rows)

    n = len(acc_rows) + len(esc_rows)
    print(f"\n{'APPLIED' if a.apply else 'DRY RUN'} (commit {commit})")
    print(f"  AUTO-ACCEPTED (passthrough): {len(acc_rows)}/{n}"
          + (f"   -> {a.manifest}" if a.apply else ""))
    print(f"  LEFT FOR A HUMAN           : {len(esc_rows)}/{n}"
          + (f"   -> {a.escalate}" if a.apply else ""))
    if not a.apply:
        print("\n  nothing written. re-run with --apply to finalize the passthrough cases.")
    return 0


def _flush(store, pending, commit):
    """Write a batch of passthrough finals in ONE commit (final label + case json per case).

    The final label is the SERVED base label, unchanged (a passthrough accept). Only the case json and
    a NEW reviews/<case>/final_label.nii.gz are written — the raw <slot>_label submissions are never
    touched. Mirrors scripts/auto_finalize._flush."""
    import tempfile
    files = {}
    for cid, (case, lab, img) in pending.items():
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "final_label.nii.gz"
            nib.save(nib.Nifti1Image(lab.astype(np.asanyarray(img.dataobj).dtype),
                                     img.affine, img.header), str(tp))
            data = tp.read_bytes()
        rel = f"reviews/{cid}/final_label.nii.gz"
        files[rel] = data
        prov_after = schema.provenance_after(case.get("prov_before") or {}, "spine", "accept")
        case["final"] = {
            "decision": "accept",                        # passthrough: the served label is unchanged
            "label_rel": rel,
            "prov_after": prov_after,
            "by": "auto:qc-pass-passthrough",
            "at": schema.utcnow(),
            "rule": "task gate ok (with FOV-truncation exemption); every message an OK line or an "
                    "FOV advisory -> served base label passed through unchanged, nothing to fix",
            "commit": commit,
        }
        case.setdefault("slots", {})[schema.ADJ_SLOT] = {
            "reviewer": "auto:qc-pass-passthrough", "done": True,
            "submitted_at": schema.utcnow()}
        files[store.case_path(cid)] = json.dumps(case, indent=2)
    store.b.write_many(files,
                       commit_message=f"auto-accept (QC-pass passthrough) {len(pending)} cases ({commit})")
    pending.clear()


if __name__ == "__main__":
    raise SystemExit(main())
