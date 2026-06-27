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
    tot_fallback = sum(int(r.get("n_fallback", 0)) for r in recs)
    tot_frag = sum(int(r.get("n_fragment", 0)) for r in recs)
    gappy = [r for r in recs if r.get("left_gaps") or r.get("right_gaps")]
    dupy = [r for r in recs if r.get("duplicate_rib_ids")]
    review = [r for r in recs if r.get("review_ribs")]

    print(f"=== v4 rib connection QC  ({n} cases) ===")
    print(f"  ribs numbered by overlap vote   : {tot_overlap}")
    print(f"  ribs recovered by nearest-vertebra fallback (small gap): {tot_fallback}")
    print(f"  fragments dropped (partial FOV / far) : {tot_frag}")
    print(f"  drop_frac (union bone not numbered)   : median {pct(0.5):.3f}  "
          f"p90 {pct(0.9):.3f}  max {drops[-1] if drops else 0:.3f}")
    print(f"  numbering-gap cases (bridge suspects) : {len(gappy)}")
    print(f"  duplicate-id cases  (merge suspects)  : {len(dupy)}")
    print(f"  cases needing MANUAL rib correction   : {len(review)}")

    if gappy:
        print("\n  -- numbering-gap / merge suspects (a rib may be bridged into its neighbour) --")
        for r in gappy[: a.show]:
            print(f"    {r.get('ct')}: L gaps {r.get('left_gaps')}  R gaps {r.get('right_gaps')}"
                  f"  dup {r.get('duplicate_rib_ids')}")
        if len(gappy) > a.show:
            print(f"    ... and {len(gappy) - a.show} more")

    # ---- manual-correction worklist -----------------------------------------
    work = []
    for r in review:
        cid = r.get("ct")
        ribs = [rr for rr in r["review_ribs"] if float(rr.get("gap_mm", 0)) >= a.gap_min]
        if ribs:
            work.append((cid, cid2tok.get(cid), ribs))

    if work:
        print(f"\n=== RIB MANUAL-CORRECTION WORKLIST  ({len(work)} cases) ===")
        print("  (whole rib, small gap to the spine, MISSED by Möller — connect/verify it)")
        for cid, tok, ribs in work[: a.show]:
            desc = ", ".join(f"{rr['side']} rib {rr['number']} (gap {rr['gap_mm']}mm)" for rr in ribs)
            print(f"    {cid}  token={tok}: {desc}")
        if len(work) > a.show:
            print(f"    ... and {len(work) - a.show} more")
        toks = sorted({t for _, t, _ in work if t})
        print("\n  ship as its OWN review cohort (separate from pelvic_native):")
        if toks:
            print(f"    python -m reviewtool review-cases --repo {a.repo} --revision {a.revision} \\")
            print(f"        --tokens {','.join(toks)} --check ribs --out ./rib_review")
        else:
            print("    (no manifest.json token map in --v4_dir — copy manifest into the tree to "
                  "emit --tokens, or review by case id)")
    else:
        print("\nNo ribs need manual correction (no Möller-missing small-gap ribs).")

    if a.csv:
        import csv
        with open(a.csv, "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["ct", "token", "side", "number", "rib_id", "gap_mm", "size_vox", "moller_frac"])
            for cid, tok, ribs in work:
                for rr in ribs:
                    w.writerow([cid, tok, rr["side"], rr["number"], rr["rib_id"],
                                rr["gap_mm"], rr["size_vox"], rr["moller_frac"]])
        print(f"\nwrote worklist CSV -> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
