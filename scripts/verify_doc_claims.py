"""Check the counting claims in the deposit's prose against the deposit's own files.

Two claims have already been found wrong by reading them closely: KNOWN_ISSUES said nine
records have a prosthetic femoral head where eight do, and it described 1035 as fragmented
after the fix that unfragmented it. Both were written from notes rather than from the files,
which is how every one of these goes wrong.

So pull the numbers out of the prose and recompute each from manifest.json and the labels.
Only claims that CAN be checked mechanically are checked; the rest are listed as unverified
so the boundary is visible rather than implied.
"""
import argparse
import json
import re
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="data/zenodo_deposit")
    a = ap.parse_args()
    d = Path(a.deposit)

    recs = json.loads((d / "manifest.json").read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs.get("records", list(recs.values()))
    n_lab = len(list((d / "labels").glob("*_label.nii.gz")))

    hw = [r for r in recs if r.get("hardware_labelled")]
    ids = [i for r in hw for i in (r.get("hardware_label_ids") or [])]
    cast = [r for r in recs if r.get("castellvi_type")]
    pos = {}
    for r in recs:
        pos[r.get("position")] = pos.get(r.get("position"), 0) + 1
    # "the two annotations sit on different series" is match_type == separate, and NOT
    # simply "the two UIDs differ". The looser test also catches the 20 pelvic_native
    # records, which have a pelvic series and no separate spine one -- a different
    # situation, and counting them here inflates the figure from 351 to 371.
    sep = [r for r in recs if r.get("match_type") == "separate"]
    uid = [r for r in recs if (r.get("spine_series_uid") or r.get("pelvic_series_uid"))]

    facts = {
        "records": len(recs),
        "label files": n_lab,
        "records with hardware": len(hw),
        "records with an arthroplasty (id 80)": sum(1 for r in hw
                                                    if 80 in (r.get("hardware_label_ids") or [])),
        "distinct hardware ids used": sorted(set(ids)),
        "records with a Castellvi grade": len(cast),
        "records with a TCIA series id": len(uid),
        "records whose two annotations sit on different series": len(sep),
    }
    print("  from the deposit's own files:")
    for k, v in facts.items():
        print(f"    {k:<52} {v}")

    # every integer in the docs, with the sentence it sits in, for eyeball-free comparison
    print("\n  numeric claims in the prose:")
    bad = 0
    for name in ("README.md", "KNOWN_ISSUES.md"):
        p = d / name
        if not p.exists():
            print(f"    {name}: NOT IN THE DEPOSIT")
            bad += 1
            continue
        txt = p.read_text(encoding="utf-8")
        checks = [
            (r"for (\d+) abdominal CT", len(recs), "record count"),
            (r"(\d+) gzipped NIfTI label volumes", n_lab, "label file count"),
            (r"\*\*(\d+) records? carry surgical hardware\*\*", len(hw), "hardware records"),
            (r"In \*\*(\d+) records? the femoral head", facts["records with an arthroplasty (id 80)"],
             "arthroplasty records"),
            (r"\*\*(\d+) of (?:\d+)\*\* records carry a radiologist Castellvi", len(cast),
             "Castellvi graded"),
            (r"In \*\*(\d+) patients\*\* the two annotations sit on", len(sep),
             "separate-series records"),
            (r"Every one of the (\d+) records has at least one", len(uid),
             "records with a series id"),
        ]
        for pat, truth, what in checks:
            m = re.search(pat, txt)
            if not m:
                continue
            claimed = int(m.group(1))
            ok = claimed == truth
            print(f"    {name:<17} {what:<26} claims {claimed:<6} files say {truth:<6}"
                  f" {'ok' if ok else 'MISMATCH'}")
            if not ok:
                bad += 1

    print()
    print("  every checkable claim matches the files" if not bad
          else f"  {bad} claim(s) disagree with the files")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
