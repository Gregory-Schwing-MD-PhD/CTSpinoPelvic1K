"""scripts/make_fig_anchors.py -- regenerate Figure 2 end to end, from the release.

WHY THIS EXISTS. The figure was once rendered from four cases whose identifiers were never
written down, so it could not be regenerated and a reader could not check it against the
released labels. Recovering them was impossible -- they were not in git history, the
scripts, or any data file. This script makes that failure impossible to repeat: the cases
are CHOSEN BY RULE from the release, the choice is written to a JSON file that ships with
the morphometrics, and the render is invoked with the arguments the published figure used.

    python scripts/make_fig_anchors.py                    # select, render, install
    python scripts/make_fig_anchors.py --no-install       # leave paper/mpda alone
    python scripts/make_fig_anchors.py --scale_mode mm    # true millimetre scale

WHAT THE PUBLISHED FIGURE USES, and why those values and not the defaults:

  --scale_mode image   Each panel is rendered at true scale and then the smaller ones are
                       enlarged to match the largest. Nothing inside a panel is distorted
                       relative to anything else in it, and nothing is shrunk. This DISCARDS
                       absolute patient size, which is the right trade for a figure whose
                       subject is how many vertebrae lie between two anchors -- but it does
                       mean the four spines shown are not the same size in life.

  --legend_cols 3      One row of keys under the strip. The default of 1 stacks them and
                       turns a 2-inch-tall figure into a 5-inch one.

  --panel_in 2.05      Four panels across the 180 mm double-column width, at the aspect the
  --height_in 2.35     reprint sets them in. Authoring at final size matters: Medical
                       Physics does not rescale figures, so type authored large and shrunk
                       would print at the wrong size.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_LABELS = ROOT / "data" / "zenodo_deposit" / "labels"
CASES_JSON = ROOT / "morphometrics" / "anchor_cases.json"
FIG_DIR = ROOT / "paper" / "mpda" / "figures"

# The panel letters are the figure's, not the selector's: the selector names a phenotype,
# and the caption refers to (a)..(d). Kept here so the two cannot drift.
LETTERS = ("a", "b", "c", "d")


def run(args: list[str]) -> int:
    print("  $", " ".join(str(a) for a in args))
    return subprocess.run(args, cwd=ROOT).returncode


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default=str(DEFAULT_LABELS))
    ap.add_argument("--scale_mode", default="image",
                    choices=("image", "span", "vertebra", "mm"))
    ap.add_argument("--panel_in", default="2.05")
    ap.add_argument("--height_in", default="2.35")
    ap.add_argument("--no-install", action="store_true",
                    help="render into a scratch directory instead of paper/mpda/figures")
    ap.add_argument("--reselect", action="store_true",
                    help="re-run the case selection even if anchor_cases.json exists")
    a = ap.parse_args()

    labels = Path(a.labels)
    if not labels.exists():
        print(f"! labels not found: {labels}")
        print("  point --labels at the released label directory (data/zenodo_deposit/labels)")
        return 2

    # ---- 1. the cases, by rule
    if a.reselect or not CASES_JSON.exists():
        print("selecting the four cases by rule")
        rc = run([sys.executable, "scripts/select_anchor_cases.py",
                  "--labels", str(labels), "--write", str(CASES_JSON)])
        if rc != 0:
            return rc
    else:
        print(f"using the recorded selection in {CASES_JSON.relative_to(ROOT)}"
              f"  (--reselect to redo it)")

    chosen = json.loads(CASES_JSON.read_text())
    if len(chosen) != len(LETTERS):
        print(f"! expected {len(LETTERS)} cases, found {len(chosen)}")
        return 3

    spec = ";".join(f"{c['case']}:({L}) {c['title']}"
                    for L, c in zip(LETTERS, chosen))
    for L, c in zip(LETTERS, chosen):
        print(f"  ({L}) {c['case']}  {c['title']}")

    # ---- 2. render
    out_dir = FIG_DIR if not a.no_install else (ROOT / "scratchpad")
    out_dir.mkdir(parents=True, exist_ok=True)
    rc = run([sys.executable, "scripts/render_anchors.py",
              "--labels", str(labels),
              "--cases", spec,
              "--scale_mode", a.scale_mode,
              "--legend_cols", "3",
              "--panel_in", a.panel_in,
              "--height_in", a.height_in,
              "--name", "fig_anchors",
              "--out", str(out_dir)])
    if rc != 0:
        return rc

    pdf = out_dir / "fig_anchors.pdf"
    print(f"\nwrote {pdf.relative_to(ROOT)}")
    if not a.no_install:
        print("installed into the manuscript; rebuild with paper/mpda/build.sh --reprint")
    return 0


if __name__ == "__main__":
    sys.exit(main())
