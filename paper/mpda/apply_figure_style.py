"""apply_figure_style.py -- bring make_figures.py to the Medical Physics figure guidelines.

Written as a file rather than a shell heredoc because the replacement text contains LaTeX
backslashes; run through a heredoc, "\\resizebox" became a carriage return and silently
corrupted the source.

WHAT THE GUIDELINES REQUIRE, and what was wrong:

  SANS SERIF ("Arial, Helvetica, or Calibri, or similar"). The file specified a serif face.

  FIXED COLUMN WIDTHS: "A single column is 80 mm wide, and a double column is 180 mm wide...
  figure size cannot be reduced in the typeset version." Figures were authored at an
  arbitrary 7 inches.

  TYPE SIZE: "a font size of 20 points or larger is appropriate as the figure will typically
  be half a page wide". That assumes a figure drawn large and then shrunk. Authoring at the
  final width means no shrinking occurs, so the equivalent requirement is type that stays
  legible at 80-180 mm; the file's 8.5 pt on a 7-inch canvas rendered at roughly 3.8 pt once
  fitted to a single column, which is what the guideline exists to prevent.

  AXES BOLD BLACK, GRIDLINES GREY, major ticks outside and minor ticks inside.

  RESOLUTION: 600 dpi for line art and charts. The file used 300, which is the figure for
  photographic images.

Also makes the data path independent of the working directory: the script previously had to
be run from the repository root or every input silently loaded as empty, which produced a
crash inside numpy rather than a legible error.
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).parent
p = HERE / "make_figures.py"
s = p.read_text(encoding="utf-8")

OLD_STYLE = '''plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman"],
    "font.size": 8.5,
    "axes.linewidth": 0.7,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK, "ytick.color": INK,
    "xtick.major.width": 0.7, "ytick.major.width": 0.7,
    "legend.frameon": False,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})'''

NEW_STYLE = '''# --------------------------------------------------------------------------
# FIGURE STYLE -- Medical Physics author guidelines.
#
#   SANS SERIF: the guidelines name Arial, Helvetica or Calibri.
#
#   COLUMN WIDTHS ARE FIXED at 80 mm (single) and 180 mm (double), and "figure size
#   cannot be reduced in the typeset version". Figures are therefore authored AT the
#   final width and included at natural size, never rescaled afterwards -- rescaling
#   changes the printed type size and defeats any font choice made here.
#
#   TYPE SIZE: the guidelines ask for 20 pt or larger on the assumption that a figure
#   is drawn large and then shrunk to half a page. Authoring at the final width means
#   no shrinking occurs, so the equivalent is type that stays legible at 80-180 mm.
#
#   AXES bold black, GRIDLINES grey, MAJOR ticks outside and MINOR ticks inside.
#
#   RESOLUTION: 600 dpi for line art and charts; 300 dpi is for photographs only.
# --------------------------------------------------------------------------
MM = 1.0 / 25.4
COL1, COL2 = 80 * MM, 180 * MM          # single and double column, in inches
BLACK, GRIDGREY = "#000000", "#B0B0B0"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Calibri", "DejaVu Sans"],
    "font.size": 8.0,
    "axes.titlesize": 8.0,
    "axes.labelsize": 8.0,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "axes.linewidth": 1.1,
    "axes.edgecolor": BLACK,
    "axes.labelcolor": BLACK,
    "text.color": BLACK,
    "axes.titleweight": "normal",
    "grid.color": GRIDGREY,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": BLACK, "ytick.color": BLACK,
    "xtick.direction": "out", "ytick.direction": "out",
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "xtick.minor.width": 0.7, "ytick.minor.width": 0.7,
    "xtick.major.size": 3.5, "ytick.major.size": 3.5,
    "xtick.minor.size": 2.0, "ytick.minor.size": 2.0,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "legend.frameon": False,
    "figure.dpi": 600,
    "savefig.dpi": 600,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def mp_ticks(ax):
    """Major ticks outward, minor ticks inward, grey grid behind the data."""
    ax.tick_params(which="major", direction="out")
    ax.tick_params(which="minor", direction="in")
    ax.set_axisbelow(True)
    return ax'''

assert OLD_STYLE in s, "style block anchor not found"
s = s.replace(OLD_STYLE, NEW_STYLE)

# authored at true column widths rather than an arbitrary 7 inches
n_w = s.count("figsize=(7.0,")
s = s.replace("figsize=(7.0,", "figsize=(COL2,")

# the data path was relative to the working directory: run from anywhere but the
# repository root and every input loaded as empty, crashing inside numpy on an
# empty boolean index rather than saying the file was missing.
OLD_M = 'M = "morphometrics"'
NEW_M = ('# Resolved against the repository root, not the working directory: a relative\n'
         '# path here made every input load as empty when the script was run from its own\n'
         '# directory, and the failure surfaced as a numpy dtype error rather than a\n'
         '# missing-file message.\n'
         'M = str((Path(__file__).resolve().parents[2] / "morphometrics"))')
assert OLD_M in s, "data path anchor not found"
s = s.replace(OLD_M, NEW_M)

p.write_text(s, encoding="utf-8")
print(f"patched: style block, {n_w} figure widths -> COL2, data path made absolute")
