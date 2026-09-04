"""scripts/make_census_table.py -- the per-identifier census as a supplementary table.

One row per populated identifier in the release: id, name, records carrying it, percent
of 802. Vertebrae, sacrum, S1, hips, femora, ribs by level, lumbar ribs, hardware. The
figure in the main text shows the vertebral profile only; this is the full count behind
it, read from morphometrics/label_census_v7.csv (itself read from the released voxels).

Writes a LaTeX fragment the supplement \\input{}s, as a three-column-pair longtable-free
layout: 60-odd rows fit one page as two side-by-side tabulars.

    python scripts/make_census_table.py --census morphometrics/label_census_v7.csv \\
        --out paper/mpda/census_table.tex
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--census", default="morphometrics/label_census_v7.csv")
    ap.add_argument("--out", default="paper/mpda/census_table.tex")
    ap.add_argument("--total", type=int, default=802)
    a = ap.parse_args()

    rows = []
    with open(a.census, newline="") as fh:
        for r in csv.DictReader(fh):
            i = int(r["id"])
            if i == 0 or int(r["records"]) == 0:
                continue
            nm = r["name"].replace("_", r"\_")
            rows.append((i, nm, int(r["records"]), 100.0 * int(r["records"]) / a.total))

    def tabular(rs):
        out = [r"\begin{tabular}{@{}rlrr@{}}", r"\toprule",
               r"id & class & records & \% \\", r"\midrule"]
        for i, nm, n, pct in rs:
            out.append(f"{i} & {nm} & {n} & {pct:.1f} \\\\")
        out += [r"\bottomrule", r"\end{tabular}"]
        return "\n".join(out)

    half = (len(rows) + 1) // 2
    body = "\n".join([
        r"\begin{table*}[p]",
        r"\caption{\label{tab:census}Every identifier populated in the release and the",
        r"number of the 802 records carrying it, counted from the released label volumes.",
        r"Identifiers 58--73 are unassigned and occur nowhere.}",
        r"\centering\small",
        tabular(rows[:half]),
        r"\hspace{2em}",
        tabular(rows[half:]),
        r"\end{table*}",
    ])
    p = Path(a.out); p.write_text(body + "\n", encoding="utf-8")
    print(f"wrote {p}: {len(rows)} identifiers")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
