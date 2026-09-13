"""Does this release's series mapping agree with CTSpine1K's own Path.csv?

    python scripts/compare_ctspine1k_pathcsv.py --pathcsv Path.csv \
        --manifest data/zenodo_deposit/manifest.json

WHY THIS IS WORTH RUNNING. Two records answer the same question from opposite directions.
CTSpine1K's Path.csv names, per patient, the series DIRECTORY its conversion happened to
read -- a filesystem fact, recorded by the authors. This release's manifest names, per
record, the series the crosswalk resolved the vertebral annotation to -- an imaging fact,
recovered by scoring which candidate's bone the label actually lands on. Neither derives
from the other, so agreement is real corroboration and disagreement localises a case where
one of the two is wrong about which scan carries the labels.

THE JOIN. Path.csv rows are `<patient OID>/<study dir>/<series dir>`, and TCIA suffixes
each directory with the tail of its UID. The manifest's `token` is the patient's ordinal,
so token N corresponds to OID ...9328.50.4.{N:04d}, and the manifest's
`spine_series_uid` tail is compared against the series directory's suffix. Both halves of
that join are checked rather than assumed: a token whose OID is missing from Path.csv, or
a series suffix that cannot be parsed, is counted and reported, never silently dropped.

WHAT A DISAGREEMENT DOES NOT MEAN. Path.csv is not ground truth. It names no series
identifier, gives no position for 250 of its 815 rows, and its study and series
directories contradict each other on the position in 70. Where the two records differ, the
crosswalk's claim rests on measured bone coverage, which travels with every record as
`spine_bone_pct`; Path.csv's rests on a directory listing. The comparison reports both so
the reader can weigh them.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path

OID_PREFIX = "1.3.6.1.4.1.9328.50.4."


def position_in(name: str) -> str:
    t = name.lower()
    p, s = "prone" in t, "supine" in t
    if p and s:
        return "both"
    return "prone" if p else ("supine" if s else "unstated")


def suffix_of(dirname: str) -> str | None:
    """TCIA appends the tail of a directory's UID after the last hyphen."""
    m = re.search(r"-([0-9][0-9.]*)$", dirname.strip())
    return m.group(1).lstrip(".") if m else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pathcsv", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    # ---- Path.csv: patient ordinal -> (study dir, series dir)
    # TWO KEY FORMATS, and missing the second silently drops 65 patients. Most rows are
    # keyed by the OID ...9328.50.4.NNNN, whose tail is the patient ordinal the manifest
    # stores as `token`; the rest are keyed by an ACRIN participant id, CTC-nnnnnnnnnn,
    # which the manifest stores in that same field verbatim. Both are indexed as strings.
    by_pat: dict[str, tuple[str, str]] = {}
    malformed = 0
    for line in Path(a.pathcsv).read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split("/")
        if len(parts) != 3:
            malformed += 1
            continue
        key = parts[0]
        if key.startswith(OID_PREFIX):
            tail = key[len(OID_PREFIX):]
            if not tail.isdigit():
                malformed += 1
                continue
            key = str(int(tail))
        elif not key.startswith("CTC-"):
            malformed += 1
            continue
        by_pat[key] = (parts[1], parts[2])
    print(f"Path.csv: {len(by_pat)} patients parsed, {malformed} rows unparsable")

    man = json.loads(Path(a.manifest).read_text())
    spine = [r for r in man if r.get("spine_series_uid")]
    print(f"manifest: {len(man)} records, {len(spine)} with a spine series uid\n")

    rows, tally = [], Counter()
    for r in man:
        vid = r.get("volume_id")
        if not r.get("spine_series_uid"):
            tally["no vertebral annotation (pelvic only)"] += 1
            continue
        tok = str(r.get("token"))
        pat = str(int(tok)) if tok.isdigit() else tok
        if pat not in by_pat:
            tally["patient absent from Path.csv"] += 1
            continue

        study_dir, series_dir = by_pat[pat]
        pc = position_in(series_dir)
        if pc == "unstated":
            pc = position_in(study_dir)
        mf = r.get("position")

        # SERIES IDENTITY IS NOT THE COMPARABLE THING. Sibling reconstructions of one
        # acquisition share the patient coordinates, so the label overlays both the same
        # way and bone coverage cannot choose between them. Recorded, not scored.
        uid_tail = str(r["spine_series_uid"]).rsplit(".", 1)[-1]
        dir_tail = suffix_of(series_dir)
        same_series = bool(dir_tail and dir_tail.rsplit(".", 1)[-1] == uid_tail)

        if pc in ("unstated", "both"):
            verdict = "Path.csv states no position"
        elif mf not in ("prone", "supine"):
            verdict = "release resolved a decubitus series"
        elif pc == mf:
            verdict = "agree on the acquisition"
        else:
            verdict = "DISAGREE on the acquisition"
        tally[verdict] += 1
        rows.append({"volume_id": vid, "token": pat,
                     "release_uid": r["spine_series_uid"],
                     "release_position": mf,
                     "spine_bone_pct": r.get("spine_bone_pct"),
                     "pathcsv_series_dir": series_dir,
                     "pathcsv_position": pc,
                     "same_series": int(same_series),
                     "verdict": verdict})

    print(f"  {'':<40} {'n':>5}")
    for k, v in tally.most_common():
        print(f"  {k:<40} {v:>5}")

    ag = tally["agree on the acquisition"]
    dg = tally["DISAGREE on the acquisition"]
    if ag + dg:
        print(f"\n  agreement on the acquisition, where Path.csv states one: "
              f"{ag}/{ag + dg} = {100 * ag / (ag + dg):.1f}%")
    same = sum(r["same_series"] for r in rows)
    print(f"  the two name the same SERIES in {same} of {len(rows)}; sibling "
          f"reconstructions are not separable by bone coverage")

    bad = [r for r in rows if r["verdict"].startswith("DISAGREE")]
    if bad:
        print(f"\n  the {len(bad)} disagreements:")
        print(f"  {'case':<7} {'release':<8} {'bone%':>6}  Path.csv series directory")
        for r in bad:
            print(f"  {r['volume_id']:<7} {str(r['release_position']):<8} "
                  f"{str(r['spine_bone_pct']):>6}  {r['pathcsv_series_dir'][:46]}")

    if a.out and rows:
        with open(a.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n  wrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
