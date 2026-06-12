"""
qc_dashboard.py — aggregate the propagation/completion QC CSVs into summary figures
for the whole dataset (distributions, before/after, per-bone), so the registration
placement quality and the GT-vs-model story can be read at a glance.

Inputs (whichever exist):
  propagate_qc.csv              (propagate_pelvis): placement — bone-HU overlap
                                before/after, drop vs native, per-bone fit, accept.
  propagated_completion_qc.csv  (pseudolabel GT-first union): per-bone voxel
                                Dice(GT, model), completeness (how complete the GT
                                was), added_vox (model completion = incompleteness).

Outputs: PNG figures + a printed text summary. Missing CSVs are skipped gracefully.

Usage
-----
  python scripts/qc_dashboard.py \
      --propagate_qc  data/placed/pelvic_propagated/propagate_qc.csv \
      --completion_qc data/hf_export_v2/propagated_completion_qc.csv \
      --out_dir       data/qc_dashboard
"""
from __future__ import annotations

import argparse
import csv
import logging
from pathlib import Path
from typing import List, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("qc_dashboard")

BONES = ("sacrum", "left_hip", "right_hip")


def _read(path: Optional[Path]) -> List[dict]:
    if not path or not Path(path).exists():
        return []
    return list(csv.DictReader(open(path)))


def _floats(rows, key):
    out = []
    for r in rows:
        v = r.get(key, "")
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            pass
    return out


def _stats(v):
    if not v:
        return "n=0"
    s = sorted(v)
    mean = sum(s) / len(s)
    return f"n={len(s)} mean={mean:.2f} median={s[len(s)//2]:.2f} " \
           f"min={s[0]:.2f} max={s[-1]:.2f}"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--propagate_qc", type=Path, default=None)
    ap.add_argument("--completion_qc", type=Path, default=None)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--drop_target", type=float, default=1.0)
    ap.add_argument("--fail_drop", type=float, default=8.0)
    args = ap.parse_args()

    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    args.out_dir.mkdir(parents=True, exist_ok=True)
    placement = _read(args.propagate_qc)
    completion = _read(args.completion_qc)
    placement = [r for r in placement if r.get("status") == "ok"]
    log.info("placement rows=%d  completion rows=%d", len(placement), len(completion))

    # ---- Figure 1: placement (bone-HU overlap before/after, drop) ------------
    if placement:
        drops = _floats(placement, "bone_pct_drop")
        before = _floats(placement, "src_bone_pct")
        after = _floats(placement, "prop_bone_pct")
        accept = sum(1 for r in placement if r.get("accept") in ("1", 1))

        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        ax[0].hist(drops, bins=20, color="#3b6", edgecolor="k", alpha=0.8)
        ax[0].axvline(args.drop_target, color="green", ls="--",
                      label=f"ideal {args.drop_target}pp")
        ax[0].axvline(args.fail_drop, color="red", ls="--",
                      label=f"fail {args.fail_drop}pp")
        ax[0].set_xlabel("bone-HU overlap drop vs native (pp)")
        ax[0].set_ylabel("cases"); ax[0].set_title("Placement degradation")
        ax[0].legend(fontsize=8)

        if before and after:
            lim = max(before + after) * 1.05
            ax[1].scatter(before, after, s=18, alpha=0.6, color="#36b")
            ax[1].plot([0, lim], [0, lim], "k--", lw=1)
            ax[1].set_xlim(0, lim); ax[1].set_ylim(0, lim)
            ax[1].set_xlabel("native overlap (%)")
            ax[1].set_ylabel("propagated overlap (%)")
            ax[1].set_title("Before vs after (on y=x = no loss)")

        perbone = [(b, _floats(placement, f"{b}_bonepct")) for b in BONES]
        perbone = [(b, p) for b, p in perbone if p]
        if perbone:
            ax[2].boxplot([p for _, p in perbone])
            ax[2].set_xticklabels([b for b, _ in perbone])
        ax[2].set_ylabel("bone-HU overlap (%)")
        ax[2].set_title("Per-bone overlap (propagated)")
        fig.suptitle(f"Propagation placement  |  {len(placement)} cases, "
                     f"{accept} accepted  |  drop {_stats(drops)}", fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p1 = args.out_dir / "placement_overview.png"
        fig.savefig(str(p1), dpi=120); plt.close(fig)
        log.info("wrote %s", p1)
        log.info("  drop(pp): %s", _stats(drops))

    # ---- Figure 2: GT-first union (Dice, completeness, incompleteness) -------
    if completion:
        fig, ax = plt.subplots(1, 3, figsize=(16, 4.5))
        for b, col in zip(BONES, ("#c33", "#e80", "#3a3")):
            d = _floats(completion, f"{b}_dice")
            c = _floats(completion, f"{b}_completeness")
            if d:
                ax[0].hist(d, bins=15, alpha=0.5, label=b, color=col)
            if c:
                ax[1].hist(c, bins=15, alpha=0.5, label=b, color=col)
        ax[0].set_xlabel("Dice(propagated GT, model)")
        ax[0].set_ylabel("cases"); ax[0].set_title("GT-vs-model voxel overlap")
        ax[0].legend(fontsize=8)
        ax[1].set_xlabel("GT completeness  (1 = GT covered the whole bone)")
        ax[1].set_title("How complete the GT masks were"); ax[1].legend(fontsize=8)
        added = [(b, _floats(completion, f"{b}_added_vox")) for b in BONES]
        added = [(b, a) for b, a in added if a]
        if added:
            ax[2].boxplot([a for _, a in added])
            ax[2].set_xticklabels([b for b, _ in added])
        ax[2].set_ylabel("voxels completed by model")
        ax[2].set_title("Model completion (= GT incompleteness)")
        fig.suptitle(f"GT-first union  |  {len(completion)} propagated cases",
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))
        p2 = args.out_dir / "completion_overview.png"
        fig.savefig(str(p2), dpi=120); plt.close(fig)
        log.info("wrote %s", p2)
        for b in BONES:
            log.info("  %-10s Dice %s | completeness %s", b,
                     _stats(_floats(completion, f"{b}_dice")),
                     _stats(_floats(completion, f"{b}_completeness")))

    if not placement and not completion:
        log.warning("no QC CSVs found — nothing to plot.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
