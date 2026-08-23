"""scripts/merge_tp_height.py — put the corrected heights back into the morphometrics table.

The full extractor produces every transitional measure in one table, and re-running it over
802 volumes takes hours. Only the transverse-process height columns were wrong: the slab
extent was taken instead of its largest connected component, so a detached speckle became
the height. Nothing else in that table -- the gaps, the spans, the disc ratios, the foramina
counts -- goes through the affected code.

So the corrected columns are spliced in rather than the whole table recomputed. The old
values are preserved as `*_prefix_mm` instead of being overwritten silently, because a
screen built on the old numbers is still in the repository and someone will want to see
exactly which cases moved and by how much.

    python scripts/merge_tp_height.py \\
        --into morphometrics/transition_morphometrics.csv \\
        --from morphometrics/tp_height.csv
"""
from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

COLS = ["tp_height_left_mm", "tp_height_right_mm"]
EXTRA = ["tp_height_slab_left_mm", "tp_height_slab_right_mm",
         "tp_tip_components_left", "tp_tip_components_right"]


def f(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--into", default="morphometrics/transition_morphometrics.csv")
    ap.add_argument("--src", default="morphometrics/tp_height.csv")
    ap.add_argument("--out", default=None, help="default: in place, with a .bak copy")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.into)))
    fix = {r["case"]: r for r in csv.DictReader(open(a.src)) if not r.get("error")}
    print(f"  {len(rows)} row(s) to update, {len(fix)} corrected measurement(s)")

    fields = list(rows[0])
    for c in COLS:
        p = c.replace("_mm", "_prefix_mm")
        if p not in fields:
            fields.insert(fields.index(c) + 1, p)
    for c in EXTRA + ["tp_height_max_prefix_mm"]:
        if c not in fields:
            fields.append(c)

    moved, missing, big = 0, 0, []
    for r in rows:
        g = fix.get(r["case"])
        if not g:
            missing += 1
            continue
        r["tp_height_max_prefix_mm"] = r.get("tp_height_max_mm", "")
        for c in COLS:
            old, new = f(r.get(c)), f(g.get(c))
            r[c.replace("_mm", "_prefix_mm")] = r.get(c, "")
            if new is None:
                continue
            r[c] = g[c]
            if old is not None and abs(old - new) > 0.05:
                moved += 1
                if old - new > 5.0:
                    big.append((r["case"], c, old, new))
        for c in EXTRA:
            r[c] = g.get(c, "")
        hl, hr = f(r.get("tp_height_left_mm")), f(r.get("tp_height_right_mm"))
        if hl is not None and hr is not None:
            r["tp_height_max_mm"] = max(hl, hr)
            r["tp_height_asym_mm"] = round(abs(hl - hr), 1)

    print(f"  {moved} column value(s) changed; {missing} row(s) had no corrected match")
    big.sort(key=lambda t: t[2] - t[3], reverse=True)
    print(f"  {len(big)} value(s) fell by more than 5 mm -- these are the speckled tips:")
    for case, c, old, new in big[:12]:
        print(f"    {case}  {c:<22} {old:>6.1f} -> {new:>5.1f}   ({old - new:.1f} mm)")

    out = Path(a.out or a.into)
    if out == Path(a.into):
        shutil.copy(a.into, str(a.into) + ".bak")
        print(f"  kept {a.into}.bak")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
