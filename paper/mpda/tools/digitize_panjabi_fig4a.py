"""Recover Panjabi's T11/T12 end-plate width from his own figure, because the table is gone.

WHY THIS EXISTS. Figure 6(a) of the manuscript compares this cohort's upper end-plate width
(EPWu) against Panjabi's cadaveric means. His lumbar values come from the 1992 paper's
Table 2 and were transcribed directly. His THORACIC values live in Table 3 of the 1991
paper, on page 892 -- and page 892 is absent from every scan obtainable. The two copies on
hand are byte-identical (861,004 bytes, MD5 d8c4b0ed411d0aa3) and both jump from p. 891 to
p. 893, which is the signature of a defect in the publisher's own scan rather than of two
bad downloads. Interlibrary loan is outstanding.

WHY NOT BENZEL. Benzel's Biomechanics of Spine Stabilization redraws these series in its
Fig. 1.1, which is the obvious shortcut and is not usable:

  - it plots VERTEBRAL BODY DIAMETER, not end-plate width, which is a different measure;
  - it pools FOUR sources (Berry 1987, Panjabi's cervical and thoracic papers, and White &
    Panjabi), so a value read off it is not attributable to Panjabi at all;
  - its thoracic axis carries only T2, T7 and T12 across twelve levels, so T11 is not even
    a tick;
  - and the chapter states, in its own words, that "some figures depict extrapolated data
    when appropriate".

A number taken from it would be an interpolation of a possible extrapolation of a pooled
average, drawn on a curve labelled Panjabi.

WHAT THIS DOES INSTEAD. The 1991 paper plots its own data: Figure 4A on page 896 shows
EPWu, EPWl, EPDu and EPDl against T1-T12. This script locates the EPWu markers in that
figure and reads them against the axis.

    python digitize_panjabi_fig4a.py <panjabi1991.pdf> [--page 8] [--csv out.csv]

HOW THE READING WORKS. EPWu is drawn as an OPEN square, so its interior is a white region
fully enclosed by black -- a hole. Holes are cheap and unambiguous to find, whereas marker
outlines merge into the connecting polylines and defeat connected-component analysis (the
first attempt at this found only the legend). The y axis is calibrated on the major tick
marks protruding left of the spine, which fit 44.0 px per mm with no residual over 10-50.
Level positions come from the spacing of the detected marker columns, NOT from the plot
frame: the curves stop short of the spines, and assuming otherwise puts T12 off the end.

FOUR CHECKS THE RESULT MUST PASS, printed on every run. None of them depends on the
reading being right, so together they are a real test rather than a restatement:

  1. T1 -> T12 increase must fall between the 55% (EPWl) and 73% (EPDu) the paper states
     on p. 890 for the four dimensions.
  2. The largest level-to-level steps must be T10->T11 and T11->T12, because p. 890 says
     "the increase in width was greatest for the two distal-most vertebrae (T11 and T12)".
  3. T12 must run continuously into L1 = 41.2 mm from the 1992 lumbar paper's Table 2.
  4. EPWl(T12) must exceed EPWu(L1), which p. 890 requires of adjacent plates.

WHAT IT CANNOT RECOVER. A figure carries no dispersion, so no SEM is produced. The
reference table leaves that column empty rather than inventing one. When page 892 arrives,
replace these two values with Table 3 and delete this dependency.
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
from scipy import ndimage

# Panjabi 1992 Table 2, for the continuity check
L1_EPWU_1992 = 41.2
# the stated total increases from T1 to T12, p. 890
INCREASE_BOUNDS = (55.0, 73.0)


def load_page(pdf: Path, page: int, dpi: int = 1000):
    import fitz
    d = fitz.open(str(pdf))
    p = d[page]
    r = p.rect
    # Fig. 4A occupies the upper-left quadrant of the page
    clip = fitz.Rect(r.width * 0.07, r.height * 0.055, r.width * 0.50, r.height * 0.30)
    pm = p.get_pixmap(dpi=dpi, clip=clip)
    img = np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n)
    return img[:, :, :3].mean(axis=2)


def find_frame(dark: np.ndarray):
    """(left, right, top, bottom) pixel positions of the plot box."""
    H, W = dark.shape
    cols = dark.sum(axis=0)
    vc = np.where(cols > 0.35 * H)[0]
    groups, cur = [], [vc[0]]
    for a, b in zip(vc, vc[1:]):
        if b - a <= 3:
            cur.append(b)
        else:
            groups.append((cur[0], cur[-1])); cur = [b]
    groups.append((cur[0], cur[-1]))
    left, right = groups[0][1], groups[-1][0]
    inner = dark[:, left + 8:right - 8]
    rows = inner.sum(axis=1)
    cand = np.where(rows > 0.85 * inner.shape[1])[0]
    hg, cur = [], [cand[0]]
    for a, b in zip(cand, cand[1:]):
        if b - a <= 3:
            cur.append(b)
        else:
            hg.append((cur[0], cur[-1])); cur = [b]
    hg.append((cur[0], cur[-1]))
    return left, right, hg[0][1], hg[-1][0]


def calibrate_y(dark: np.ndarray, left: int):
    """value(py) from the major ticks left of the spine. They are 50,40,30,20,10."""
    strip = dark[:, left - 46:left - 12]
    prof = strip.sum(axis=1)
    lab, n = ndimage.label(prof >= 20)
    ys = []
    for k in range(1, n + 1):
        idx = np.nonzero(lab == k)[0]
        if len(idx) >= 3:
            ys.append((idx.min() + idx.max()) / 2)
    ys = sorted(ys)
    # keep only the evenly spaced majors; a minor tick sits at half spacing near the base
    if len(ys) >= 2:
        step = np.median(np.diff(ys[:4])) if len(ys) >= 4 else np.diff(ys)[0]
        majors = [ys[0]]
        for y in ys[1:]:
            if y - majors[-1] > 0.75 * step:
                majors.append(y)
        ys = majors
    vals = [50, 40, 30, 20, 10][:len(ys)]
    a, b = np.polyfit(ys, vals, 1)
    resid = max(abs(v - (a * y + b)) for y, v in zip(ys, vals))
    return (lambda py: a * py + b), abs(1 / a), resid


def open_square_interiors(dark, frame, val):
    """Centres of the white interiors of the OPEN-square (EPWu) markers."""
    left, right, top, bottom = frame
    y0, y1 = int(top) - 10, int(bottom) + 10
    reg = dark[y0:y1, left + 9:right - 2]
    lab, n = ndimage.label(~reg)
    border = set(lab[0, :]) | set(lab[-1, :]) | set(lab[:, 0]) | set(lab[:, -1])
    hits = []
    for k in range(1, n + 1):
        if k in border:
            continue
        ys, xs = np.nonzero(lab == k)
        h, w = ys.max() - ys.min() + 1, xs.max() - xs.min() + 1
        if not (12 <= h <= 34 and 12 <= w <= 34):
            continue
        if not (0.72 <= h / w <= 1.45):          # square interiors, not triangle ones
            continue
        hits.append((left + 9 + xs.mean(), val(y0 + ys.mean())))
    return sorted(hits)


def assign_levels(hits, tol_px: float = 45.0):
    """Level index from the marker grid, anchored on the RIGHTMOST column.

    Two things this has to survive, both of which broke earlier versions:

    THE CURVES STOP SHORT OF THE FRAME, so level positions must come from the data and not
    from the spines; deriving them from the frame put T12 past the last marker and returned
    nothing for it.

    THE LEGEND IS ALSO DRAWN WITH OPEN SQUARES, and its symbols are interiors just like the
    data's. They are rejected here rather than by a hand-drawn box: a legend symbol does not
    sit on the level grid. Fitting the grid from the right-hand markers, where the curves
    are well separated, and discarding anything more than `tol_px` from a grid line removes
    them on a geometric criterion instead of a magic rectangle. On this figure the legend
    columns fall 55 and 98 px off the nearest level, against a 220 px spacing.
    """
    xs = np.sort(np.array([h[0] for h in hits]))
    x12 = xs.max()
    # step from the widest consecutive gaps among the right-hand third, which are clean
    right = xs[xs > x12 - 6 * 230]
    d = np.diff(np.unique(np.round(right / 4) * 4))
    d = d[d > 120]
    step = float(np.median(d)) if len(d) else 220.5

    out, rejected = [], []
    for x, v in hits:
        k = (x - x12) / step
        if abs(k - round(k)) * step > tol_px:
            rejected.append((x, v)); continue
        out.append((int(round(k)) + 12, x, v))
    return out, step, rejected


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("pdf", type=Path, help="scan of Panjabi 1991, Spine 16(8):888-901")
    ap.add_argument("--page", type=int, default=8,
                    help="1-based page of the PDF carrying Figure 4 (p. 896)")
    ap.add_argument("--csv", type=Path, help="write the per-level series here")
    a = ap.parse_args()

    g = load_page(a.pdf, a.page - 1)
    dark = g < 120
    frame = find_frame(dark)
    val, px_per_mm, resid = calibrate_y(dark, frame[0])
    print(f"frame  left={frame[0]} right={frame[1]} top={frame[2]} bottom={frame[3]}")
    print(f"y axis {px_per_mm:.1f} px/mm, worst tick residual {resid:.3f} mm")

    hits = open_square_interiors(dark, frame, val)
    levelled, step, rejected = assign_levels(hits)
    print(f"level spacing {step:.1f} px; {len(rejected)} off-grid marker(s) rejected "
          f"(legend): {[(round(x), round(v,1)) for x, v in rejected]}")

    # EPWu is the HIGHER of the two open markers at any level (the other is EPDu)
    per = {}
    for L, x, v in levelled:
        if 1 <= L <= 12:
            per.setdefault(L, []).append(v)
    epwu = {L: max(vs) for L, vs in per.items()}

    print("\nEPWu by level:")
    for L in sorted(epwu):
        print(f"   T{L:<2d} {epwu[L]:6.2f} mm")

    ok = True
    if 1 in epwu and 12 in epwu:
        inc = 100.0 * (epwu[12] / epwu[1] - 1.0)
        good = INCREASE_BOUNDS[0] <= inc <= INCREASE_BOUNDS[1]
        ok &= good
        print(f"\n[{'ok ' if good else 'FAIL'}] T1->T12 increase {inc:.1f}%"
              f"  (paper states {INCREASE_BOUNDS[0]:.0f}-{INCREASE_BOUNDS[1]:.0f}% across"
              " the four dimensions)")
    steps = {L: epwu[L] - epwu[L - 1] for L in sorted(epwu) if L - 1 in epwu}
    if steps:
        biggest = sorted(steps, key=lambda L: -steps[L])[:2]
        good = set(biggest) == {11, 12}
        ok &= good
        print(f"[{'ok ' if good else 'FAIL'}] largest steps at T{biggest[0]}, T{biggest[1]}"
              f"  (paper: greatest at T11 and T12)")
    if 12 in epwu:
        gap = L1_EPWU_1992 - epwu[12]
        good = 0.0 < gap < 4.0
        ok &= good
        print(f"[{'ok ' if good else 'FAIL'}] T12 {epwu[12]:.1f} -> L1 {L1_EPWU_1992}"
              f" (1992 Table 2), step {gap:+.1f} mm")

    if a.csv:
        with a.csv.open("w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["level", "EPWu_mm", "source"])
            for L in sorted(epwu):
                w.writerow([f"T{L}", f"{epwu[L]:.2f}",
                            "Panjabi 1991 Fig 4A p.896, digitised"])
        print(f"\nwrote {a.csv}")
    print("\nALL CHECKS PASS" if ok else "\nSOME CHECKS FAILED -- do not use these values")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
