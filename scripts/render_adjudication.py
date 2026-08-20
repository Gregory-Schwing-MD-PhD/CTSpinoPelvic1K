"""
scripts/render_adjudication.py — 3D side-by-side renders so an adjudicator can decide WITHOUT
opening ITK-SNAP.

Each rib is coloured BY ITS NUMBER. When two annotators disagree by an off-by-one renumbering the
COLOURS SHIFT between the two panels, so the dispute is visible at a glance instead of being
reconstructed by eye in a 3D editor. Conflicting ribs are listed under each panel.

  python scripts/render_adjudication.py            # every case in adjudication_list.csv
  python scripts/render_adjudication.py --case 276__pelvic_native
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import numpy as np, nibabel as nib
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

_H = Path(__file__).resolve().parent
sys.path.insert(0, str(_H)); sys.path.insert(0, str(_H.parent / "review_service"))
import label_scheme as LS, review_anatomy_qc as RA
from auto_finalize import halo_ids, zero_overlap_classes, completeness, LO, HI
from huggingface_hub import hf_hub_download

REPO = "anonymous-mlhc/CTSpinoPelvic1K-reviews-ribs"
TOK = os.environ["HF_TOKEN"]
NAMES = RA._id2name()
CMAP = plt.get_cmap("tab20")


def _panel(ax, lab, aff, title, bad, step=3):
    si = int(np.argmax(np.abs(aff[:3, :3][2, :])))
    lr = int(np.argmax(np.abs(aff[:3, :3][0, :])))
    ap = ({0, 1, 2} - {si, lr}).pop()
    sub = lab[::step, ::step, ::step]
    # spine, faint grey, for reference
    sp = np.argwhere((sub >= 1) & (sub <= LS.S1_ID))
    if sp.size:
        ax.scatter(sp[:, lr], sp[:, ap], -sp[:, si] if aff[2, si] >= 0 else sp[:, si],
                   c="0.75", s=1, alpha=.10, linewidths=0)
    for rid in range(LO, HI + 1):
        pts = np.argwhere(sub == rid)
        if not pts.size:
            continue
        off = LS.RIB_LEFT_OFFSET if rid <= LS.RIB_LEFT_OFFSET + 12 else LS.RIB_RIGHT_OFFSET
        n = rid - off                                  # rib NUMBER drives the colour
        col = CMAP((n - 1) % 20)
        z = -pts[:, si] if aff[2, si] >= 0 else pts[:, si]
        ax.scatter(pts[:, lr], pts[:, ap], z, color=col, s=3,
                   alpha=(1.0 if rid in bad else .45), linewidths=0)
    # label each rib with its NUMBER right on the bone -> you can read the numbering directly
    for rid in range(LO, HI + 1):
        pts = np.argwhere(sub == rid)
        if not pts.size:
            continue
        off = LS.RIB_LEFT_OFFSET if rid <= LS.RIB_LEFT_OFFSET + 12 else LS.RIB_RIGHT_OFFSET
        n = rid - off
        c = pts.mean(0)
        z = -c[si] if aff[2, si] >= 0 else c[si]
        ax.text(c[lr], c[ap], z, str(n), fontsize=8, weight="bold",
                color=("red" if rid in bad else "0.25"))
    ax.set_title(title, fontsize=10)
    ax.set_axis_off()
    ax.view_init(elev=14, azim=-70)
    # tight framing on the ribs (kill the dead space)
    rp = np.argwhere((sub >= LO) & (sub <= HI))
    if rp.size:
        zz = -rp[:, si] if aff[2, si] >= 0 else rp[:, si]
        ax.set_xlim(rp[:, lr].min(), rp[:, lr].max())
        ax.set_ylim(rp[:, ap].min(), rp[:, ap].max())
        ax.set_zlim(zz.min(), zz.max())
        ax.set_box_aspect((np.ptp(rp[:, lr]), np.ptp(rp[:, ap]), np.ptp(zz)))


def render(cid: str, out: Path):
    ia = nib.load(hf_hub_download(REPO, f"reviews/{cid}/1_label.nii.gz", repo_type="dataset", token=TOK))
    ib = nib.load(hf_hub_download(REPO, f"reviews/{cid}/2_label.nii.gz", repo_type="dataset", token=TOK))
    A, B = np.asanyarray(ia.dataobj), np.asanyarray(ib.dataobj)
    bad = set(zero_overlap_classes(A, B, drop=tuple(set(halo_ids(A)) | set(halo_ids(B)))))
    cA, cB = completeness(A), completeness(B)
    fig = plt.figure(figsize=(15, 8))
    for k, (lab, aff, who, comp) in enumerate(
            [(A, ia.affine, "reviewer 1", cA), (B, ib.affine, "reviewer 2", cB)]):
        ax = fig.add_subplot(1, 2, k + 1, projection="3d")
        _panel(ax, lab, aff, f"{who}   ({comp[0]} structures, {comp[1]:,} vox)", bad)
    names = ", ".join(sorted(NAMES.get(c, str(c)) for c in bad)) or "none"
    fig.suptitle(f"{cid}   —   CONFLICTING RIBS: {names}\n"
                 f"each rib coloured BY NUMBER — if the colours are SHIFTED between panels, "
                 f"the annotators renumbered", fontsize=10)
    fig.text(.5, .02, "numbers printed ON each rib (RED = conflicting)  |  faint grey = spine  |  "
             "colour = rib number, so a SHIFT between panels = a renumbering",
             ha="center", fontsize=9)
    fig.savefig(out, dpi=105, bbox_inches="tight")
    plt.close(fig)
    return sorted(NAMES.get(c, str(c)) for c in bad), cA, cB


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default=None)
    ap.add_argument("--outdir", default="adjudication_renders")
    a = ap.parse_args(argv)
    od = Path(a.outdir); od.mkdir(parents=True, exist_ok=True)
    cases = ([a.case] if a.case
             else [r["case"] for r in csv.DictReader(open("adjudication_list.csv"))])
    for i, cid in enumerate(cases, 1):
        try:
            bad, cA, cB = render(cid, od / f"{cid}.png")
            more = "r1" if cA >= cB else "r2"
            print(f"[{i}/{len(cases)}] {cid}: conflicts={len(bad)}  more-complete={more} "
                  f"({cA[0]} vs {cB[0]} structs)  -> {od/f'{cid}.png'}", flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f"[{i}/{len(cases)}] {cid}: FAILED {str(e)[:70]}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
