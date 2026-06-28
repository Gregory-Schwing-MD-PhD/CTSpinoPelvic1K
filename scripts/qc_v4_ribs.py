"""
qc_v4_ribs.py — aggregate the per-case rib QC that build_v4_ribs writes to
<v4_dir>/_v4ribs_done/*.json (instancer report + rib_invariants verdict), and:

  1. print a CONTRACT summary (how many cases pass the anatomical contract, and the
     breakdown of HARD violations: order / coherence; soft: gap / asymmetry);
  2. emit the QUARANTINE WORKLIST — the only cases a human need touch (a HARD invariant
     violation, an anchor offset conflict, or rib bone past rib 12) — plus the ready
     `reviewtool review-cases ... --check ribs` command;
  3. optionally act as a BUILD GATE: `--gate FRAC` exits non-zero if the quarantine
     fraction exceeds FRAC, so a bad rib build can't be pushed (wired into ship_v4).

Every rib is kept + masked regardless (good for DRR); this only gates the *numbering*.

  python scripts/qc_v4_ribs.py --v4_dir data/hf_export_v4 [--csv rib_review.csv] [--gate 0.02]
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def _load_manifest_cid2token(v4_dir: Path) -> dict:
    """Map a case's file base ('0004') -> manifest token ('3') for the review command."""
    man = v4_dir / "manifest.json"
    if not man.exists():
        return {}
    payload = json.loads(man.read_text())
    if isinstance(payload, dict):
        payload = payload.get("records") or payload.get("cases") or []
    out = {}
    for r in payload:
        lf = (r or {}).get("label_file", "") or ""
        if lf.endswith("_label.nii.gz"):
            out[Path(lf).name[: -len("_label.nii.gz")]] = str(r.get("token"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--v4_dir", type=Path, required=True,
                    help="v4 tree containing _v4ribs_done/*.json (and ideally manifest.json)")
    ap.add_argument("--csv", type=Path, default=None, help="write the quarantine worklist to this CSV")
    ap.add_argument("--repo", default="anonymous-mlhc/CTSpinoPelvic1K",
                    help="dataset repo for the emitted review command")
    ap.add_argument("--revision", default="v4", help="branch to review (default v4)")
    ap.add_argument("--gate", type=float, default=None,
                    help="BUILD GATE: exit 1 if quarantine fraction exceeds this (e.g. 0.02)")
    ap.add_argument("--show", type=int, default=40, help="max rows to print per section")
    a = ap.parse_args()

    done = sorted((a.v4_dir / "_v4ribs_done").glob("*.json"))
    if not done:
        raise SystemExit(f"no QC files in {a.v4_dir}/_v4ribs_done/ — run build_v4_ribs first")
    recs = []
    for f in done:
        try:
            recs.append(json.loads(f.read_text()))
        except Exception:                                    # noqa: BLE001
            pass
    n = len(recs)
    cid2tok = _load_manifest_cid2token(a.v4_dir)

    quarantined = [r for r in recs if r.get("quarantine")]
    hard_fail = [r for r in recs if not r.get("hard_ok", True)]
    vtypes: Counter = Counter()
    for r in recs:
        for v in r.get("violations", []):
            vtypes[v.get("type")] += 1
    n_conflict = sum(1 for r in recs if r.get("conflicts"))
    n_overflow = sum(1 for r in recs if r.get("out_of_range"))
    qfrac = len(quarantined) / n if n else 0.0

    print(f"=== v4 rib contract QC  ({n} cases) ===")
    print(f"  PASS contract (no hard violation)   : {n - len(hard_fail)}/{n}")
    print(f"  QUARANTINE (hard / conflict / overflow): {len(quarantined)}  ({100 * qfrac:.1f}%)")
    print(f"    - hard violations  : {len(hard_fail)}  "
          f"(order x{vtypes.get('order', 0)}, coherence x{vtypes.get('coherence', 0)})")
    print(f"    - anchor conflicts : {n_conflict}")
    print(f"    - past rib 12      : {n_overflow}")
    print(f"  soft flags: gap x{vtypes.get('gap', 0)}, asymmetry x{vtypes.get('asymmetry', 0)} "
          f"(not quarantined — likely real anatomy / partial FOV)")

    if quarantined:
        print(f"\n=== QUARANTINE WORKLIST  ({len(quarantined)} cases) ===")
        rows = []
        for r in quarantined:
            cid = r.get("ct")
            why = r.get("viol_summary", "")
            extra = []
            if r.get("conflicts"):
                extra.append("anchor-conflict")
            if r.get("out_of_range"):
                extra.append("past-12")
            tag = ", ".join([why] + extra) if extra else why
            rows.append((cid, cid2tok.get(cid), tag))
        for cid, tok, tag in rows[: a.show]:
            print(f"    {cid}  token={tok}: {tag}")
        if len(rows) > a.show:
            print(f"    ... and {len(rows) - a.show} more")
        toks = sorted({t for _, t, _ in rows if t})
        print("\n  review as its OWN cohort (separate from pelvic_native):")
        if toks:
            print(f"    python -m reviewtool review-cases --repo {a.repo} --revision {a.revision} \\")
            print(f"        --tokens {','.join(toks)} --check ribs --out ./rib_review")
        else:
            print("    (no manifest.json token map in --v4_dir to emit --tokens)")
        if a.csv:
            import csv
            with open(a.csv, "w", newline="") as fh:
                w = csv.writer(fh)
                w.writerow(["ct", "token", "reason", "left_ribs", "right_ribs"])
                for r in quarantined:
                    cid = r.get("ct")
                    w.writerow([cid, cid2tok.get(cid), r.get("viol_summary"),
                                r.get("left_ribs"), r.get("right_ribs")])
            print(f"\n  wrote quarantine CSV -> {a.csv}")
    else:
        print("\nNo quarantined cases — every case satisfies the rib contract.")

    if a.gate is not None:
        ok = qfrac <= a.gate
        print(f"\nGATE: quarantine {100 * qfrac:.1f}% {'<=' if ok else '>'} "
              f"threshold {100 * a.gate:.1f}% -> {'PASS' if ok else 'FAIL'}")
        return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
