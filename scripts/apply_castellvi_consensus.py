"""scripts/apply_castellvi_consensus.py -- the two readers' consensus Castellvi grades, applied.

SOURCE. Lumbosac.csv (repo root, 2026-09-04): both radiology residents (NA, MI) read every
graded case independently; where they differed, the comment column records the grade they
agreed on. Names are "Token N" / "token N" / "token_N" and casing of the grades is mixed
(IIb / IIB); both are normalised here. Token 10 is a normal reference read (NL).

WHAT IS WRITTEN. For each graded record, in every manifest this repo stages:
    castellvi_type        the CONSENSUS grade (Ia..IV, or "0" for the normal reference)
    castellvi_read_1      reader NA's independent call
    castellvi_read_2      reader MI's independent call
    castellvi_agreement   True where the two independent calls matched
    castellvi_consensus   True (the grade above is a consensus, not a single read)
    castellvi_second_read is dropped: it recorded five second reads that the consensus
                          supersedes.
Also morphometrics/transition_morphometrics.csv (castellvi_type column) and
docs/castellvi_consensus.csv, the flat table the paper's counts are derived from.

    python scripts/apply_castellvi_consensus.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "Lumbosac.csv"
MANIFESTS = [ROOT / "data/zenodo_deposit/manifest.json", ROOT / "data/hf_export_v5/manifest.json"]


def norm_grade(g):
    if g is None or (isinstance(g, float) and pd.isna(g)):
        return None
    g = str(g).strip()
    if not g:
        return None
    if g.upper() == "NL":
        return "0"
    m = re.fullmatch(r"(I{1,3}|IV)\s*([abAB])?", g)
    if not m:
        raise ValueError(f"unreadable grade {g!r}")
    return m.group(1) + (m.group(2).lower() if m.group(2) else "")


def consensus_from_comment(comment):
    if not isinstance(comment, str):
        return None
    m = re.search(r"agreed on\s*(I{1,3}|IV)\s*([abAB])?", comment, re.I)
    if not m:
        return None
    roman = m.group(1).upper()
    return roman + (m.group(2).lower() if m.group(2) else "")


def read_source():
    df = pd.read_csv(SRC)
    rows = []
    for _, r in df.iterrows():
        name = r["Name"]
        if not isinstance(name, str) or not re.search(r"\d", name):
            continue
        token = int(re.sub(r"\D", "", name))
        r1 = norm_grade(r["1st Reviewer NA"]); r2 = norm_grade(r["2nd reviewer MI"])
        comment = r["Describe or comments (optional)"]
        agreed = consensus_from_comment(comment)
        if r1 == "0" and r2 is None:
            cons = "0"
        elif r1 == r2:
            cons = r1
        elif agreed:
            cons = agreed
        else:
            raise ValueError(f"token {token}: {r1} vs {r2} with no agreed grade")
        rows.append({"token": token, "read_1_NA": r1, "read_2_MI": r2 or "",
                     "agreement": (r1 == r2), "consensus": cons,
                     "comment": comment if isinstance(comment, str) else ""})
    # duplicate rows (token 737 appears twice) must agree
    out = {}
    for row in rows:
        if row["token"] in out:
            assert out[row["token"]]["consensus"] == row["consensus"], row
        out[row["token"]] = row
    return out


def main() -> int:
    cons = read_source()
    print(f"  {len(cons)} tokens read from {SRC.name}")
    dist = {}
    for r in cons.values():
        dist[r["consensus"]] = dist.get(r["consensus"], 0) + 1
    print("  consensus grades:", dict(sorted(dist.items())))
    print("  independent disagreements:", sorted(t for t, r in cons.items() if not r["agreement"]))

    tok2vol = {}
    for mp in MANIFESTS:
        recs = json.load(open(mp, encoding="utf-8"))
        changed = 0
        for rec in recs:
            tok = rec.get("token")
            try:
                tok = int(str(tok).replace("Token", "").replace("token", "").strip("_ "))
            except (TypeError, ValueError):
                continue
            if tok not in cons:
                if rec.get("castellvi_type") not in (None, ""):
                    print(f"  ! {mp.name}: {rec['volume_id']} has a grade but is not in the consensus file")
                continue
            c = cons[tok]
            tok2vol[tok] = rec["volume_id"]
            before = rec.get("castellvi_type")
            rec["castellvi_type"] = c["consensus"]
            rec["castellvi_read_1"] = c["read_1_NA"]
            rec["castellvi_read_2"] = c["read_2_MI"]
            rec["castellvi_agreement"] = bool(c["agreement"])
            rec["castellvi_consensus"] = True
            rec.pop("castellvi_second_read", None)
            if before != c["consensus"]:
                changed += 1
                print(f"    {mp.name}: {rec['volume_id']} (token {tok}) {before} -> {c['consensus']}")
        # records never graded lose the retired field too
        for rec in recs:
            rec.pop("castellvi_second_read", None)
        json.dump(recs, open(mp, "w", encoding="utf-8"), indent=1)
        print(f"  {mp}: {changed} grades changed, {len(tok2vol)} records carry a consensus grade")

    # morphometrics
    tp = ROOT / "morphometrics/transition_morphometrics.csv"
    t = pd.read_csv(tp, dtype={"volume_id": str})
    idcol = "volume_id" if "volume_id" in t.columns else t.columns[0]
    vol2grade = {v: cons[k]["consensus"] for k, v in tok2vol.items()}
    n_before = t["castellvi_type"].notna().sum()
    t["castellvi_type"] = t[idcol].astype(str).str.zfill(4).map(vol2grade)
    t.to_csv(tp, index=False)
    print(f"  {tp.name}: castellvi_type populated on {t['castellvi_type'].notna().sum()} rows (was {n_before})")

    # the flat table
    out = ROOT / "docs/castellvi_consensus.csv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["volume_id", "token", "read_1_NA", "read_2_MI", "agreement", "consensus", "comment"])
        for tok in sorted(cons):
            c = cons[tok]
            w.writerow([tok2vol.get(tok, ""), tok, c["read_1_NA"], c["read_2_MI"], c["agreement"],
                        c["consensus"], c["comment"]])
    print(f"  wrote {out}")
    graded = [c for t_, c in cons.items() if c["consensus"] != "0"]
    fused = [c for c in graded if c["consensus"].startswith(("III", "IV"))]
    print(f"  graded transitional cases: {len(graded)}; grade III/IV (fused): {len(fused)}; "
          f"normal reference reads: {sum(c['consensus'] == '0' for c in cons.values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
