"""Does CTPelvic1K's filename naming pick the same series the bone criterion did?

    python scripts/compare_ctpelvic1k_naming.py --db data/patient_db.json \
        --manifest data/zenodo_deposit/manifest.json

WHAT CTPELVIC1K SHIPPED. Labels only, for the collections it did not itself acquire,
named for the TCIA patient identifier, the DICOM SeriesNumber and a slice count --
`dataset2_1.3.6.1.4.1.9328.50.4.0001_3_325_mask_4label.nii.gz`, in a directory called
`CTPelvic1K_dataset2_mask_mappingback`. Every field is a real DICOM-side quantity, so the
intent was clearly that a user resolve the mask to its series by name.

THE TEST. Take the SeriesNumber out of each filename, resolve it within that patient's
TCIA series, and compare the result with the series the bone-coverage criterion actually
chose. Three outcomes matter and are counted separately:

  the naming resolves to one series, and it is the chosen one
  the naming resolves to one series, and it is a DIFFERENT acquisition
  the SeriesNumber is not unique within the patient, so the name resolves to nothing

The slice count is checked too, because it is the one field that could confirm a pairing
independently, and on every case inspected by hand it disagreed with the series its own
filename names.
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    db = json.loads(Path(a.db).read_text())
    pats = db[[k for k in db if k != "metadata"][0]]
    man = json.loads(Path(a.manifest).read_text())
    chosen = {str(r.get("token")): r for r in man if r.get("pelvic_series_uid")}
    print(f"patients in db: {len(pats)}   records with a resolved pelvic series: "
          f"{len(chosen)}")

    rows, tally = [], Counter()
    nz_state = Counter()
    for p in pats.values():
        pm = p.get("pelvic_masks") or []
        tok = str(p.get("patient_token"))
        if not pm:
            continue
        if tok not in chosen:
            tally["mask exists but no record in the release"] += 1
            continue
        rec = chosen[tok]
        m = pm[0]
        sn = m.get("series_number_from_fname")
        nz_fname = m.get("nz_from_fname")

        series = p.get("tcia_series") or []
        by_uid = {s.get("series_uid"): s for s in series}
        sel = by_uid.get(rec["pelvic_series_uid"])

        cand = [s for s in series if s.get("series_number") == sn]
        if sn is None:
            verdict = "filename carries no series number"
        elif len(cand) == 0:
            verdict = "series number matches no series in this patient"
        elif len(cand) > 1:
            verdict = "series number is not unique within the patient"
        else:
            named = cand[0]
            if sel is None:
                verdict = "chosen series absent from the catalogue"
            elif named.get("series_uid") == sel.get("series_uid"):
                verdict = "naming resolves to the chosen series"
            elif (named.get("patient_position") or "").upper() == \
                 (sel.get("patient_position") or "").upper():
                verdict = "different series, same acquisition"
            else:
                verdict = "DIFFERENT ACQUISITION"
        tally[verdict] += 1

        if sel is not None and nz_fname:
            n_dcm = sel.get("n_dcm")
            if n_dcm:
                r = n_dcm / nz_fname
                nz_state["exact" if n_dcm == nz_fname else
                         ("about 2x" if 1.9 <= r <= 2.1 else "other ratio")] += 1

        rows.append({
            "token": tok, "volume_id": rec.get("volume_id"),
            "fname_series_number": sn, "fname_nz": nz_fname,
            "chosen_uid": rec["pelvic_series_uid"],
            "chosen_position": rec.get("position"),
            "chosen_n_dcm": (sel or {}).get("n_dcm"),
            "named_position": (cand[0].get("patient_position") if len(cand) == 1 else None),
            "verdict": verdict,
        })

    print(f"\n  {'':<48} {'n':>5}")
    for k, v in tally.most_common():
        print(f"  {k:<48} {v:>5}")

    ok = tally["naming resolves to the chosen series"]
    diff = tally["DIFFERENT ACQUISITION"]
    same = tally["different series, same acquisition"]
    resolved = ok + diff + same
    if resolved:
        print(f"\n  of {resolved} masks whose series number resolves uniquely, "
              f"{ok} name the chosen series, {same} name a sibling reconstruction of the "
              f"same acquisition, and {diff} name a different acquisition")

    print(f"\n  filename slice count against the named series' slice count:")
    for k, v in nz_state.most_common():
        print(f"    {k:<14} {v:>5}")

    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
