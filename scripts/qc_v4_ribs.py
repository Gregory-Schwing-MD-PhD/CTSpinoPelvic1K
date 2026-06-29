"""
qc_v4_ribs.py — aggregate the per-case rib-connection QC that build_v4_ribs writes
to <v4_dir>/_v4ribs_done/*.json, and emit:

  1. a CONNECTION-QUALITY summary (how much rib bone connected vs dropped, plus
     numbering-gap / duplicate-id cases — the adjacent-rib bridging suspects), and
  2. a MANUAL-CORRECTION WORKLIST: whole ribs with a small gap to the spine that
     Möller MISSED (so only TS saw them) — exactly the ribs a medical student
     should connect/verify. This is shipped as its OWN review cohort, separate from
     the pelvic-native pseudo-spine review.

Partial-FOV fragments (anterior tips, far-from-spine blobs) are dropped at build
time and never reach the worklist — nothing to correct there.

  python scripts/qc_v4_ribs.py --v4_dir data/hf_export_v4 [--csv rib_review.csv]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load_manifest_cid2token(v4_dir: Path) -> dict:
    """Map a case's file base (e.g. '0004') -> manifest token (e.g. '3'), so the
    worklist can emit a `reviewtool review-cases --tokens ...` command. Empty if no
    manifest.json is present in the tree."""
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
    ap.add_argument("--csv", type=Path, default=None, help="write the worklist to this CSV")
    ap.add_argument("--repo", default="anonymous-mlhc/CTSpinoPelvic1K",
                    help="dataset repo for the emitted review command")
    ap.add_argument("--revision", default="v4", help="branch to review (default v4)")
    ap.add_argument("--gap_min", type=float, default=0.0,
                    help="only worklist ribs whose gap >= this many mm (default 0 = all)")
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

    drops = sorted(float(r.get("drop_frac", 0.0)) for r in recs)

    def pct(p):
        return drops[min(len(drops) - 1, int(p * len(drops)))] if drops else 0.0

    tot_overlap = sum(int(r.get("n_overlap", 0)) for r in recs)
    tot_tsoff = sum(int(r.get("n_tsoff", 0)) for r in recs)
    tot_extrap = sum(int(r.get("n_extrap", 0)) for r in recs)
    tot_fp = sum(int(r.get("n_dropped_fp", 0)) for r in recs)
    tot_fp_vox = sum(int(r.get("dropped_fp_vox", 0)) for r in recs)
    fpdrop = sorted((r for r in recs if int(r.get("n_dropped_fp", 0)) > 0),
                    key=lambda r: -int(r.get("dropped_fp_vox", 0)))
    gappy = [r for r in recs if r.get("left_gaps") or r.get("right_gaps")]
    dupy = [r for r in recs if r.get("duplicate_rib_ids")]
    review = [r for r in recs if r.get("review_ribs")]

    print(f"=== v4 rib numbering QC  ({n} cases) ===")
    print(f"  ribs numbered by overlap vote (anchored)  : {tot_overlap}")
    print(f"  ribs from TS numbering (offset-corrected) : {tot_tsoff}")
    print(f"  ribs numbered by counting (Möller-only)   : {tot_extrap}")
    print(f"  Möller off-anatomy blobs filtered out (FP): {tot_fp} comps / {tot_fp_vox} vox "
          f"in {len(fpdrop)} cases")
    print(f"  drop_frac (union bone unnumbered = noise) : median {pct(0.5):.3f}  "
          f"p90 {pct(0.9):.3f}  max {drops[-1] if drops else 0:.3f}")
    print(f"  numbering-gap cases (bridge suspects)     : {len(gappy)}")
    print(f"  duplicate-id cases  (merge suspects)      : {len(dupy)}")
    print(f"  cases with extrapolated ribs to verify    : {len(review)}")

    if fpdrop:
        print("\n  -- false-positive filter: most rib bone removed (Möller bowel/calcification; "
              "glance to confirm it was NOT a real rib) --")
        for r in fpdrop[: a.show]:
            print(f"    {r.get('ct')}: {r.get('n_dropped_fp')} comp  {r.get('dropped_fp_vox')} vox")
        if len(fpdrop) > a.show:
            print(f"    ... and {len(fpdrop) - a.show} more")

    # ---- consolidated REVIEW WORKLIST: the few cases that need a manual rib check ----
    # The strict filter already auto-dropped bowel/strays (no review). Only genuinely
    # ambiguous cases are flagged: a numbering GAP (possible miscount), a DUPLICATE id (a
    # rib split in two, or a near-spine stray sharing a number), or a pure-guess
    # EXTRAPOLATED rib. Students review ONLY these.
    flagged = []
    for r in recs:
        ct = r.get("ct")
        why = []
        if r.get("left_gaps") or r.get("right_gaps"):
            why.append(f"gap L{r.get('left_gaps')} R{r.get('right_gaps')}")
        if r.get("duplicate_rib_ids"):
            why.append(f"dup {r['duplicate_rib_ids']}")
        if r.get("review_ribs"):
            why.append(f"extrapolated x{len(r['review_ribs'])}")
        if why:
            flagged.append((ct, cid2tok.get(ct), "; ".join(why)))

    print(f"\n=== RIB REVIEW WORKLIST  ({len(flagged)}/{n} cases need a manual check) ===")
    if flagged:
        for ct, tok, why in flagged[: a.show]:
            print(f"    {ct}  token={tok}: {why}")
        if len(flagged) > a.show:
            print(f"    ... and {len(flagged) - a.show} more")
        toks = sorted({t for _, t, _ in flagged if t})
        if toks:
            print("\n  ship exactly these for review (its own cohort, separate from pelvic_native):")
            print(f"    python -m reviewtool review-cases --repo {a.repo} --revision {a.revision} \\")
            print(f"        --tokens {','.join(toks)} --check ribs --out ./rib_review")
        else:
            print("    (no manifest.json token map in --v4_dir — copy manifest in to emit --tokens)")
    else:
        print("  none — every case is clean (no gaps, dups, or extrapolated ribs).")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ct", "token", "reason"])
            for ct, tok, why in flagged:
                w.writerow([ct, tok, why])
        print(f"\nwrote review worklist CSV ({len(flagged)} cases) -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
