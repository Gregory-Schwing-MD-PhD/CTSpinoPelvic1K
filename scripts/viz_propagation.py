"""
viz_propagation.py — eyeball the propagated pelves: overlay each registration-placed
pelvis on its SPINE CT (the scan it was carried onto) so you can see whether the
sacrum + ilia actually sit on the target bone.

Reads propagate_qc.csv (from propagate_pelvis.py) and, per case, renders 3 orthogonal
slices (coronal / axial / sagittal) through the pelvis centroid: the spine CT in
grayscale with the propagated pelvis overlaid (sacrum=7, left_hip=8, right_hip=9 in
the standard QC colors), titled with accept / bone-HU drop / overlap. One PNG/case.

Usage
-----
  python scripts/viz_propagation.py \
      --qc_csv     data/placed/pelvic_propagated/propagate_qc.csv \
      --nifti_dir  data/tcia_nifti \
      --out_dir    data/placed/pelvic_propagated/qc
  # optional: --tokens 4,7,1,8   to render only a few
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("viz_propagation")

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

# Display orientation (matches visualize_qc.py): volumes are PIR (0=P,1=I,2=R).
#   coronal  dim 0 -> (I, R)  head up
#   axial    dim 1 -> (P, R)  anterior up
#   sagittal dim 2 -> (P, I)  transpose so head is up
_VIEWS = (("coronal", 0), ("axial", 1), ("sagittal", 2))


def _disp(arr, dim):
    """Sagittal (dim 2) is transposed so the head is up. For an RGB image
    (H, W, 3), transpose ONLY the two spatial axes — a plain .T would move the
    colour channel to the front and break imshow."""
    if dim != 2:
        return arr
    import numpy as np
    return np.swapaxes(arr, 0, 1) if arr.ndim == 3 else arr.T


def _render_case(spine_ct: Path, mask_path: Path, out_png: Path, title: str):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from viz_pelvic_dimensions import _load_pir, _hu_window
    from export_hf import _overlay, _center_slice

    ct, _ = _load_pir(spine_ct)
    lbl, _ = _load_pir(mask_path)
    lbl = np.rint(lbl).astype(np.int16)
    if ct.shape[:3] != lbl.shape[:3]:
        log.warning("shape mismatch %s vs %s — skipping", ct.shape, lbl.shape)
        return False
    bg = _hu_window(ct)
    ci, cj, ck = _center_slice(ct, lbl)
    centers = (ci, cj, ck)

    fig, axes = plt.subplots(1, 3, figsize=(13, 5))
    for ax, (name, dim) in zip(axes, _VIEWS):
        idx = centers[dim]
        sl = np.take(bg, idx, axis=dim)               # (H,W) grayscale [0,1]
        ls = np.take(lbl, idx, axis=dim)              # (H,W) labels
        # _overlay already blends the CT background with the coloured labels.
        ax.imshow(_disp(_overlay(sl, ls), dim), interpolation="nearest",
                  aspect="auto")
        ax.set_title(f"{name} @ {idx}", fontsize=9)
        ax.axis("off")
    fig.suptitle(title, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_png), dpi=110)
    plt.close(fig)
    return True


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--qc_csv", required=True, type=Path)
    ap.add_argument("--nifti_dir", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--propagated_dir", type=Path, default=None,
                    help="resolve out_file relative to here if its path is absent.")
    ap.add_argument("--tokens", default="", help="comma-separated subset to render.")
    args = ap.parse_args()

    import re
    want = {t for t in re.split(r"[,:;\s]+", args.tokens.strip()) if t} or None
    rows = list(csv.DictReader(open(args.qc_csv)))
    rows = [r for r in rows if r.get("status") == "ok"
            and (want is None or r.get("token") in want)]
    log.info("rendering %d case(s) -> %s", len(rows), args.out_dir)

    n_ok = 0
    for i, r in enumerate(rows, 1):
        tok, suid = r.get("token", "?"), r.get("spine_uid", "")
        mask = Path(r.get("out_file", ""))
        if not mask.exists() and args.propagated_dir and suid:
            mask = args.propagated_dir / f"{suid}_pelvic_propagated.nii.gz"
        spine_ct = args.nifti_dir / f"{suid}.nii.gz"
        if not mask.exists() or not spine_ct.exists():
            log.warning("[%d/%d] token=%s missing CT/mask — skip", i, len(rows), tok)
            continue
        acc = r.get("accept", "?")
        title = (f"token {tok}   accept={acc}   "
                 f"drop={r.get('bone_pct_drop','?')}pp   "
                 f"overlap before={r.get('src_bone_pct','?')} "
                 f"after={r.get('prop_bone_pct','?')}%   "
                 f"reasons: {r.get('reasons','') or '-'}")
        out_png = args.out_dir / f"{tok}_{suid}_propagation.png"
        try:
            if _render_case(spine_ct, mask, out_png, title):
                n_ok += 1
                log.info("[%d/%d] token=%s -> %s", i, len(rows), tok, out_png.name)
        except Exception as exc:                                # noqa: BLE001
            log.warning("[%d/%d] token=%s render failed: %s", i, len(rows), tok, exc)

    log.info("done: %d/%d PNG(s) written to %s", n_ok, len(rows), args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
