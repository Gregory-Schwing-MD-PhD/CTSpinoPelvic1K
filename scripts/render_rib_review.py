"""scripts/render_rib_review.py — one sheet per case for the rib-numbering disputes.

fix_rib_offsets renumbers a cage that is uniformly shifted with nothing already correct.
What is left is the hard residue: cases where some ribs sit right and one or two do not,
or where the offsets disagree with each other. Those cannot be shifted mechanically -- a
lone rib 12 moved down onto an already-correct rib 11 corrupts a case that was 90% right.

The QC's own reading of that residue is that a couple of stray offsets are "far more
likely a segmentation artefact than a counting mistake", and the only way to tell is to
look. So this draws, per case, what the reviewer actually needs to decide:

  * every rib, numbered, coloured by side, over a coronal MIP of the spine
  * every thoracic and lumbar body, numbered, at its own centroid
  * the DISPUTED ribs picked out in red with the vertebra they actually touch

A coronal MIP rather than a slice, for the same reason as the asymmetry sheets: a rib is
oblique and leaves any single plane, so a slice manufactures absences the projection does
not.

The decision is then one of three, written into the CSV this emits:

    shift      the cage really is miscounted -- renumber by delta
    keep       the label is right and the proximity metric was fooled (a floating 12th
               rib's head sits near T11; a fragment is nearest the wrong body)
    flag       genuinely ambiguous, send to the review tool

    python scripts/render_rib_review.py --qc qc_rib_incidence_v5 --labels data/v5_final \
        --out rib_review_sheets
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path
import sys

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS                                          # noqa: E402

BLUE, ORANGE, RED, GREY = "#2a78d6", "#eb6834", "#d81f26", "#b9b8b2"
INK, SURFACE = "#0b0b0b", "#fcfcfb"
THORACIC_BASE = 7
LUMBAR = {20: "L1", 21: "L2", 22: "L3", 23: "L4", 24: "L5", 25: "L6"}

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": "#101014",
    "savefig.facecolor": SURFACE, "text.color": INK,
    "font.size": 8, "axes.titlesize": 9,
})


def rib_id(side, n):
    return (LS.RIB_LEFT_OFFSET if side == "left" else LS.RIB_RIGHT_OFFSET) + n


def load_disputes(qc: Path):
    """Cases with >=1 offset rib that fix_rib_offsets did NOT resolve."""
    rows = list(csv.DictReader(open(qc / "rib_incidence.csv")))
    buckets = collections.defaultdict(collections.Counter)
    offs = collections.defaultdict(list)
    for r in rows:
        buckets[r["case"]][r["bucket"]] += 1
        if r["bucket"] == "offset":
            offs[r["case"]].append(r)

    fixed = set()
    plan = qc / "rib_shift_plan.json"
    if plan.exists():
        fixed = {c["case"] for c in json.loads(plan.read_text()).get("changed", [])}

    out = []
    for case, ribs in sorted(offs.items()):
        if case in fixed:
            continue
        deltas = {int(r["delta"]) for r in ribs}
        why = ("mixed offsets %s" % sorted(deltas) if len(deltas) > 1
               else "%d rib(s) already correct" % buckets[case]["match"]
               if buckets[case]["match"] else "renumber would leave 1..12")
        out.append({"case": case, "ribs": ribs, "why": why,
                    "match": buckets[case]["match"],
                    "delta": sorted(deltas)[0] if len(deltas) == 1 else None})
    return out


def panel(ax, path: Path, rec: dict):
    img = nib.as_closest_canonical(nib.load(str(path)))     # RAS+: x=R, y=A, z=S
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    zoom = np.array(img.header.get_zooms()[:3], float)
    disputed = {(r["side"], int(r["rib"])): r for r in rec["ribs"]}

    spine_ids = list(range(THORACIC_BASE + 1, THORACIC_BASE + 13)) + list(LUMBAR) + [26, 29]
    ax.imshow(np.isin(lab, spine_ids).max(axis=1).T, origin="lower", cmap="Greys",
              vmin=0, vmax=1.6, aspect=zoom[2] / zoom[0], interpolation="nearest")

    # every vertebra numbered at its own centroid -- the reviewer is comparing a rib
    # number against a vertebra number, so both have to be legible in the same picture
    for vid, name in list(LUMBAR.items()) + [(THORACIC_BASE + n, f"T{n}")
                                             for n in range(1, 13)]:
        m = (lab == vid)
        if not m.any():
            continue
        mip = m.max(axis=1)
        ys, xs = np.nonzero(mip.T)
        ax.text(xs.mean(), ys.mean(), name, color="#e8e8e6", fontsize=5,
                ha="center", va="center", fontweight="bold", zorder=5)

    for side, colour in (("left", BLUE), ("right", ORANGE)):
        for n in range(1, 13):
            m = (lab == rib_id(side, n))
            if not m.any():
                continue
            hot = (side, n) in disputed
            mip = m.max(axis=1)
            ax.contour(mip.T, levels=[.5], colors=[RED if hot else colour],
                       linewidths=1.5 if hot else .7)
            ys, xs = np.nonzero(mip.T)
            k = np.argmin(xs) if side == "left" else np.argmax(xs)
            txt = f"{n}→{disputed[(side, n)]['nearest']}" if hot else str(n)
            ax.text(xs[k], ys[k], txt, color=RED if hot else colour,
                    fontsize=6.5 if hot else 5.5, fontweight="bold",
                    ha="center", va="center", zorder=6)

    det = "  ".join(f"{r['side'][0].upper()}{r['rib']}→{r['nearest']}"
                    f"({r['gap_mm']}mm)" for r in rec["ribs"][:6])
    ax.set_title(f"{rec['case'].replace('_label.nii.gz', '')}   "
                 f"{len(rec['ribs'])} disputed, {rec['match']} correct\n"
                 f"{rec['why']}\n{det}", loc="left", color=INK, fontsize=7)
    ax.set_xticks([])
    ax.set_yticks([])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--out", default="rib_review_sheets")
    a = ap.parse_args()

    qc, labels, out = Path(a.qc), Path(a.labels), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    disputes = load_disputes(qc)
    print(f"  {len(disputes)} cases left to review\n")

    for rec in disputes:
        fp = labels / rec["case"]
        if not fp.exists():
            print(f"  ! missing {rec['case']}")
            continue
        fig, ax = plt.subplots(figsize=(4.2, 6.4), dpi=170)
        panel(ax, fp, rec)
        fig.tight_layout()
        stem = rec["case"].replace("_label.nii.gz", "")
        fig.savefig(out / f"{stem}.png", bbox_inches="tight")
        plt.close(fig)
        print(f"  {stem}  {len(rec['ribs'])} disputed, {rec['match']} correct   {rec['why']}")

    # contact sheet, so all sixteen can be triaged in one look before opening any single one
    n = len(disputes)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 6.4 * rows), dpi=110)
    for ax, rec in zip(np.ravel(axes), disputes):
        fp = labels / rec["case"]
        if fp.exists():
            panel(ax, fp, rec)
    for ax in np.ravel(axes)[n:]:
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(out / "_contact_sheet.png", bbox_inches="tight")
    plt.close(fig)

    dec = out / "decisions.csv"
    with open(dec, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "disputed", "already_correct", "suggested_delta",
                    "why", "detail", "decision", "note"])
        for r in disputes:
            w.writerow([r["case"], len(r["ribs"]), r["match"],
                        r["delta"] if r["delta"] is not None else "",
                        r["why"],
                        " ".join(f"{x['side'][0].upper()}{x['rib']}->{x['nearest']}"
                                 for x in r["ribs"]),
                        "", ""])
    print(f"\n  wrote {out}/_contact_sheet.png, {n} case sheets, and {dec.name}")
    print("  fill the `decision` column with: shift | keep | flag")
    return 0


if __name__ == "__main__":
    sys.exit(main())
