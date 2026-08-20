"""scripts/render_rib_asymmetry.py — contact sheets of the cases whose rib counts differ L vs R.

The incidence QC says 229 of 801 v4 cases have an unequal number of left and right ribs.
That is far above any plausible rate of true anatomical asymmetry, so the population is
almost certainly dominated by one side's rib being missed or falling under the size gate
-- but "almost certainly" is not a finding, and the only way to separate a missed rib from
a genuinely hypoplastic one is to look.

So: a coronal maximum-intensity projection per case, ribs coloured by side and each one
labelled with its own number, nine to a sheet. The missing number is then readable at a
glance -- "L 1-12, R 1-11, R12 absent" -- without opening a volume viewer.

Sorted worst-first (|L-R| = 2 before 1), because if the two-rib cases are all genuine
misses there is no point reviewing 224 one-rib cases by hand.

MIP, not a slice: a rib is oblique and leaves any single coronal plane, so a slice would
manufacture absences that the projection does not.

    python scripts/render_rib_asymmetry.py --qc qc_rib_incidence_v4 --labels DIR --out DIR
"""
from __future__ import annotations

import argparse
import csv
import collections
from pathlib import Path

import numpy as np
import nibabel as nib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                   # noqa: E402
from matplotlib.patches import Patch                              # noqa: E402

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import label_scheme as LS                                          # noqa: E402

BLUE, ORANGE, GREY = "#2a78d6", "#eb6834", "#b9b8b2"
INK, INK2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
THORACIC_BASE = 7

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": "#101014",
    "savefig.facecolor": SURFACE, "text.color": INK,
    "font.size": 8, "axes.titlesize": 8.5,
})


def rib_id(side, n):
    return (LS.RIB_LEFT_OFFSET if side == "left" else LS.RIB_RIGHT_OFFSET) + n


def asymmetric(qc: Path):
    per = collections.defaultdict(lambda: {"left": set(), "right": set()})
    for r in csv.DictReader(open(qc / "rib_incidence.csv")):
        per[r["case"]][r["side"]].add(int(r["rib"]))
    out = []
    for case, s in per.items():
        d = len(s["left"]) - len(s["right"])
        if d:
            out.append({"case": case, "left": sorted(s["left"]),
                        "right": sorted(s["right"]), "diff": d})
    # worst first: if the two-rib cases are all real misses, the 224 one-rib cases can wait
    out.sort(key=lambda r: (-abs(r["diff"]), r["case"]))
    return out


def panel(ax, path: Path, rec: dict):
    img = nib.as_closest_canonical(nib.load(str(path)))     # RAS+: x=R, y=A, z=S
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    zoom = np.array(img.header.get_zooms()[:3], float)

    # coronal MIP: collapse the anterior-posterior axis
    vert = np.isin(lab, list(range(THORACIC_BASE + 1, THORACIC_BASE + 13)) +
                        list(range(20, 27)) + [29]).max(axis=1)
    ax.imshow(vert.T, origin="lower", cmap="Greys", vmin=0, vmax=1.6,
              aspect=zoom[2] / zoom[0], interpolation="nearest")

    for side, colour in (("left", BLUE), ("right", ORANGE)):
        for n in range(1, 13):
            m = (lab == rib_id(side, n))
            if not m.any():
                continue
            mip = m.max(axis=1)
            ax.contour(mip.T, levels=[.5], colors=[colour], linewidths=.8)
            ys, xs = np.nonzero(mip.T)
            # label at the medial end (nearest the spine) so numbers do not pile up
            k = np.argmin(xs) if side == "left" else np.argmax(xs)
            ax.text(xs[k], ys[k], str(n), color=colour, fontsize=5.5,
                    fontweight="bold", ha="center", va="center")

    miss_r = sorted(set(rec["left"]) - set(rec["right"]))
    miss_l = sorted(set(rec["right"]) - set(rec["left"]))
    note = []
    if miss_r:
        note.append("R missing " + ",".join(map(str, miss_r)))
    if miss_l:
        note.append("L missing " + ",".join(map(str, miss_l)))
    ax.set_title(f"{rec['case'].replace('_label.nii.gz','')}\n"
                 f"L{len(rec['left'])} R{len(rec['right'])}   {' · '.join(note)}",
                 loc="left", color=INK, fontsize=7.5)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", required=True)
    ap.add_argument("--labels", help="dir holding the label volumes")
    # Resolve the snapshot through the Hub rather than globbing the cache: the cache
    # holds several revisions and picking one with `ls | head -1` silently hands you
    # main's filenames while the QC was run on v4, so every case reads as "not found".
    ap.add_argument("--hf-rev", help="resolve labels from this dataset revision instead")
    ap.add_argument("--hf-repo", default="anonymous-mlhc/CTSpinoPelvic1K")
    ap.add_argument("--out", default="rib_asymmetry_sheets")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    if a.hf_rev:
        from huggingface_hub import snapshot_download
        import os
        root = snapshot_download(a.hf_repo, repo_type="dataset", revision=a.hf_rev,
                                 allow_patterns="labels/*", max_workers=8,
                                 token=os.environ.get("HF_TOKEN"))
        a.labels = str(Path(root, "labels"))
    if not a.labels:
        ap.error("need --labels DIR or --hf-rev REV")
    qc, labels, out = Path(a.qc), Path(a.labels), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    recs = asymmetric(qc)
    have = {p.name for p in labels.glob("*.nii.gz")}
    missing = [r["case"] for r in recs if r["case"] not in have]
    if missing:
        raise SystemExit(f"{len(missing)} of {len(recs)} cases are not in {labels} "
                         f"(e.g. {missing[:3]}) -- wrong revision?")
    if a.limit:
        recs = recs[:a.limit]
    print(f"{len(recs)} asymmetric cases "
          f"({sum(1 for r in recs if abs(r['diff']) == 2)} differ by 2)")

    with open(out / "asymmetry_worklist.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["sheet", "case", "n_left", "n_right", "diff",
                    "left_ribs", "right_ribs", "verdict", "note"])
        for i, r in enumerate(recs):
            w.writerow([f"sheet_{i//9:03d}.png", r["case"], len(r["left"]),
                        len(r["right"]), r["diff"],
                        " ".join(map(str, r["left"])), " ".join(map(str, r["right"])),
                        "", ""])

    made = 0
    for s in range(0, len(recs), 9):
        chunk = recs[s:s + 9]
        fig, axes = plt.subplots(3, 3, figsize=(11.5, 12.5))
        for ax in axes.ravel():
            ax.axis("off")
        for ax, rec in zip(axes.ravel(), chunk):
            ax.axis("on")
            p = labels / rec["case"]
            if not p.exists():
                ax.set_title(f"{rec['case']} (not found)", fontsize=7); continue
            try:
                panel(ax, p, rec)
            except Exception as exc:                              # noqa: BLE001
                ax.set_title(f"{rec['case']}\n{type(exc).__name__}", fontsize=7)
        fig.legend(handles=[Patch(color=BLUE, label="left ribs"),
                            Patch(color=ORANGE, label="right ribs"),
                            Patch(color=GREY, label="vertebrae")],
                   loc="lower center", ncol=3, frameon=False, fontsize=8.5)
        fig.suptitle(f"Rib count asymmetry · sheet {s//9 + 1} of "
                     f"{(len(recs)+8)//9} · coronal MIP",
                     x=.01, ha="left", fontsize=11, fontweight="bold")
        fig.tight_layout(rect=(0, .028, 1, .965))
        fig.savefig(out / f"sheet_{s//9:03d}.png", dpi=115)
        plt.close(fig)
        made += 1
        if made % 5 == 0:
            print(f"  {made} sheets", flush=True)
    print(f"\nwrote {made} sheets + asymmetry_worklist.csv -> {out}")


if __name__ == "__main__":
    main()
