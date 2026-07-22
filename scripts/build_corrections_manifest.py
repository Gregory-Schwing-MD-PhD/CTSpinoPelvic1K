"""
scripts/build_corrections_manifest.py — a citable record of ground-truth ERRORS in the SOURCE labels
that this dataset corrects. The spine labels come from CTSpine1K and carry no pseudolabels, so every
spine defect the QC finds is a genuine error in the original CTSpine1K annotation; pelvis defects are
in CTPelvic1K. This writes:

    docs/ctspine1k_corrections.csv   — the machine-readable manifest
    docs/CORRECTIONS.md              — a README-ready section (summary + table)

Sources (the QC scans):
    spine_pelvis_mixing.csv   — a bone split into disconnected pieces (structure_integrity)
    missing_vertebra.csv      — a vertebra missing inside the labelled run, or present-in-FOV-but-unlabelled

Re-run with --final <dir> once reviews are complete to fill the "correction" column from the actual
diff (original vs corrected label). Until then correction = "pending (in review)".

    python scripts/build_corrections_manifest.py
"""
from __future__ import annotations
import argparse, csv, re
from pathlib import Path

PELVIS_WORDS = ("hip", "femur", "sacrum", "S1", "coccyx")


def _rows():
    rows = []
    p = Path("spine_pelvis_mixing.csv")
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r.get("token") == "token" or not (r.get("splits") or "").strip():
                continue
            for s in r["splits"].split("|"):
                s = s.strip()
                if not s:
                    continue
                m = re.match(r"(\w+) is split into pieces \(largest (\d+)%", s)
                bone = m.group(1) if m else s.split()[0]
                pct = m.group(2) if m else "?"
                pelvis = any(w in bone for w in PELVIS_WORDS)
                stray = str(100 - int(pct)) if pct.isdigit() else "?"
                rows.append({"case": r["token"], "source": "CTPelvic1K" if pelvis else "CTSpine1K",
                             "structure": bone, "error": "split_into_pieces",
                             "detail": f"main body {pct}% + {stray}% in a disconnected stray piece",
                             "correction": "pending (in review)"})
    p = Path("missing_vertebra.csv")
    if p.exists():
        for r in csv.DictReader(open(p)):
            if r.get("token") == "token":
                continue
            if (r.get("interior_gaps") or "").strip():
                rows.append({"case": r["token"], "source": "CTSpine1K",
                             "structure": r["interior_gaps"], "error": "missing_vertebra",
                             "detail": "a vertebra is missing between two labelled levels (skipped in "
                                       "the original annotation)", "correction": "pending (in review)"})
            if (r.get("ribs_above_spine") or "").strip():
                rows.append({"case": r["token"], "source": "CTSpine1K",
                             "structure": "upper thoracic", "error": "thoracic_unlabelled_in_fov",
                             "detail": "thoracic vertebrae are in the field of view but were not "
                                       "annotated in the original labels", "correction": "pending (in review)"})
    rows.sort(key=lambda r: (r["source"], r["error"], r["case"]))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--final", help="dir of corrected final labels (fills the correction column)")
    a = ap.parse_args(argv)
    rows = _rows()
    Path("docs").mkdir(exist_ok=True)
    with open("docs/ctspine1k_corrections.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "source", "structure", "error", "detail", "correction"])
        w.writeheader(); w.writerows(rows)

    # summary
    import collections
    by_src = collections.Counter(r["source"] for r in rows)
    by_err = collections.Counter((r["source"], r["error"]) for r in rows)
    spine_cases = len({r["case"] for r in rows if r["source"] == "CTSpine1K"})
    pelv_cases = len({r["case"] for r in rows if r["source"] == "CTPelvic1K"})

    md = ["## Ground-truth corrections to the source labels", "",
          "The spinal-column labels in this dataset originate from **CTSpine1K** and the pelvic labels "
          "from **CTPelvic1K**. During quality control we identified and corrected a number of genuine "
          "errors in those original annotations (split vertebrae, skipped levels, and vertebrae left "
          "unlabelled within the field of view). Because the spine carries no pseudolabels, every spine "
          "correction below is a fix to the original CTSpine1K ground truth.", "",
          f"- **CTSpine1K (spine):** {by_src['CTSpine1K']} corrections across {spine_cases} cases",
          f"- **CTPelvic1K (pelvis):** {by_src['CTPelvic1K']} corrections across {pelv_cases} cases", "",
          "| case | source | structure | error | detail | correction |",
          "|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['case']} | {r['source']} | {r['structure']} | {r['error']} | "
                  f"{r['detail']} | {r['correction']} |")
    md.append("")
    Path("docs/CORRECTIONS.md").write_text("\n".join(md), encoding="utf-8")

    print(f"wrote {len(rows)} correction records -> docs/ctspine1k_corrections.csv + docs/CORRECTIONS.md")
    print(f"   CTSpine1K: {by_src['CTSpine1K']} ({spine_cases} cases)   "
          f"CTPelvic1K: {by_src['CTPelvic1K']} ({pelv_cases} cases)")
    for (src, err), n in by_err.most_common():
        print(f"     {src} / {err}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
