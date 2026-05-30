"""
refine_review.py — visualize what a refinement changed, for human review.

Given the BEFORE tree (pseudo) and the AFTER tree (refined), per case it
classifies every voxel whose label changed into:

    removed    label -> background   (over-seg trimmed off non-bone)
    added      background -> label   (marrow fill / bounded grow)
    relabeled  class A -> class B     (cross-disc bleed fix; the risky one)

and emits, per changed case:
  * a 3D change-map NIfTI (1=removed 2=added 3=relabeled) to overlay on the CT
    in ITK-SNAP — the EXHAUSTIVE view (scroll every slice);
  * 2D PNG overlays (CT + colour-coded change) for the slices that changed —
    a fast gallery for triage;
  * a row in summary.csv and a link in index.html.

Cases are sorted relabeled-first (then flagged), so the changes most worth a
human's eyes are at the top. If --flags points at compete's review_flags.json,
the touching/fused components it could NOT resolve are marked in the report.

Usage
-----
  python scripts/refine_review.py \
      --before data/hf_export_v2 \
      --after  data/hf_export_v2_refined \
      --out    data/refine_review \
      [--flags data/hf_export_v2_refined/review_flags.json] \
      [--axis 2] [--max_slices 12] [--limit N] [--token TOKEN]
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
from intensity_refine import SCOPE, _load_manifest  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%Y-%m-%d %H:%M:%S")
log = logging.getLogger("ctspinopelvic1k.refine_review")

# RGBA overlay colours per change category.
_COLORS = {1: (1.0, 0.0, 0.0),     # removed  -> red
           2: (0.0, 1.0, 0.0),     # added    -> green
           3: (0.0, 0.55, 1.0)}    # relabeled-> blue


def classify_change(before, after):
    """Per-voxel change category: 0 none, 1 removed, 2 added, 3 relabeled."""
    import numpy as np
    before = np.asarray(before); after = np.asarray(after)
    cat = np.zeros(before.shape, dtype=np.uint8)
    bg_b, bg_a = before == 0, after == 0
    cat[(~bg_b) & bg_a] = 1                                   # removed
    cat[bg_b & (~bg_a)] = 2                                   # added
    cat[(~bg_b) & (~bg_a) & (before != after)] = 3           # relabeled
    return cat


def _slice_changes(cat, axis: int):
    """Indices along `axis` that contain any change, busiest first."""
    import numpy as np
    other = tuple(a for a in range(cat.ndim) if a != axis)
    counts = (cat > 0).sum(axis=other)
    idx = np.nonzero(counts)[0]
    return idx[np.argsort(counts[idx])[::-1]]


def _render_slice(ct2d, cat2d, png_path: Path, *, wl: float, ww: float) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    g = np.clip((np.asarray(ct2d, np.float32) - lo) / max(hi - lo, 1e-6), 0, 1)
    rgb = np.stack([g, g, g], axis=-1)
    overlay = np.zeros(rgb.shape, np.float32)
    alpha = np.zeros(cat2d.shape, np.float32)
    for k, col in _COLORS.items():
        m = cat2d == k
        if m.any():
            overlay[m] = col
            alpha[m] = 0.55
    a = alpha[..., None]
    blended = rgb * (1 - a) + overlay * a
    fig, ax = plt.subplots(figsize=(4, 4), dpi=110)
    ax.imshow(np.rot90(blended))
    ax.set_axis_off()
    fig.subplots_adjust(0, 0, 1, 1)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(png_path), bbox_inches="tight", pad_inches=0)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--before", required=True, type=Path,
                    help="pseudo tree (pre-refine)")
    ap.add_argument("--after", required=True, type=Path,
                    help="refined tree (post-refine; also holds the CTs)")
    ap.add_argument("--manual_from", type=Path, default=None,
                    help="optional CT source if not present in --after")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--flags", type=Path, default=None,
                    help="compete's review_flags.json (marks fused components)")
    ap.add_argument("--axis", type=int, default=2,
                    help="slice axis for the 2D overlays (default 2)")
    ap.add_argument("--max_slices", type=int, default=12,
                    help="max changed slices to render per case (busiest first)")
    ap.add_argument("--wl", type=float, default=400.0, help="CT window level")
    ap.add_argument("--ww", type=float, default=1800.0, help="CT window width")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--token", default=None, help="only this case token")
    ap.add_argument("--no_pngs", action="store_true",
                    help="write only the change-map NIfTI + csv (skip PNGs)")
    args = ap.parse_args()

    import numpy as np
    import nibabel as nib

    man = args.after / "manifest.json"
    if not man.exists():
        log.error("no manifest.json in %s", args.after)
        return 1
    records = [r for r in _load_manifest(man)
               if r.get("config") in SCOPE and r.get("label_file")]
    if args.token:
        records = [r for r in records if str(r.get("token")) == str(args.token)]
    if args.limit:
        records = records[:args.limit]

    flagged_tokens = set()
    flags_by_token: dict = {}
    if args.flags and args.flags.exists():
        for f in json.loads(args.flags.read_text()):
            t = str(f.get("token"))
            flagged_tokens.add(t)
            flags_by_token.setdefault(t, []).append(f)

    args.out.mkdir(parents=True, exist_ok=True)
    rows: List[dict] = []
    for i, rec in enumerate(records, 1):
        tok = str(rec.get("token"))
        lbl_rel = rec["label_file"]
        before_p, after_p = args.before / lbl_rel, args.after / lbl_rel
        if not before_p.exists() or not after_p.exists():
            continue
        ref = nib.load(str(after_p))
        before = np.asarray(nib.load(str(before_p)).dataobj).astype(np.int16)
        after = np.asarray(ref.dataobj).astype(np.int16)
        if before.shape != after.shape:
            log.warning("token=%s shape mismatch; skip", tok)
            continue
        cat = classify_change(before, after)
        n_rm = int((cat == 1).sum()); n_ad = int((cat == 2).sum())
        n_rl = int((cat == 3).sum())
        if n_rm == n_ad == n_rl == 0:
            continue                                          # nothing changed

        case_dir = args.out / tok
        case_dir.mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(cat, ref.affine, ref.header),
                 str(case_dir / "change_map.nii.gz"))

        pngs: List[str] = []
        if not args.no_pngs:
            ct_p = (args.after / rec["ct_file"]) if (args.after / rec["ct_file"]).exists() \
                else (args.manual_from / rec["ct_file"]) if args.manual_from else None
            if ct_p and Path(ct_p).exists():
                ct = np.asarray(nib.load(str(ct_p)).dataobj).astype(np.float32)
                ax = args.axis if 0 <= args.axis < cat.ndim else cat.ndim - 1
                for z in _slice_changes(cat, ax)[:args.max_slices]:
                    ct2d = np.take(ct, z, axis=ax)
                    cat2d = np.take(cat, z, axis=ax)
                    name = f"slice_{int(z):04d}.png"
                    _render_slice(ct2d, cat2d, case_dir / name,
                                  wl=args.wl, ww=args.ww)
                    pngs.append(f"{tok}/{name}")
            else:
                log.warning("token=%s: no CT found; change-map only", tok)

        rows.append({"token": tok, "config": rec.get("config"),
                     "removed": n_rm, "added": n_ad, "relabeled": n_rl,
                     "flagged": tok in flagged_tokens,
                     "n_flag_components": len(flags_by_token.get(tok, [])),
                     "pngs": pngs})
        if i % 25 == 0 or i == len(records):
            log.info("  [%d/%d] processed (%d changed so far)",
                     i, len(records), len(rows))

    # sort: relabeled desc, then flagged, then total change
    rows.sort(key=lambda r: (r["relabeled"], r["flagged"],
                             r["removed"] + r["added"]), reverse=True)

    with open(args.out / "summary.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["token", "config", "removed", "added", "relabeled",
                    "flagged", "n_flag_components"])
        for r in rows:
            w.writerow([r["token"], r["config"], r["removed"], r["added"],
                        r["relabeled"], r["flagged"], r["n_flag_components"]])

    _write_index(args.out, rows, flags_by_token)
    log.info("=" * 60)
    log.info("%d changed case(s) -> %s", len(rows), args.out)
    log.info("  open %s, or overlay <token>/change_map.nii.gz on the CT in ITK-SNAP",
             args.out / "index.html")
    log.info("=" * 60)
    return 0


def _write_index(out: Path, rows: List[dict], flags_by_token: dict) -> None:
    parts = ["<!doctype html><meta charset=utf-8><title>refine review</title>",
             "<style>body{font:14px system-ui;margin:1.5rem}"
             ".c{margin:1rem 0;border-top:1px solid #ccc;padding-top:.6rem}"
             "img{height:200px;margin:2px;border:1px solid #ddd}"
             ".k{color:#c00}.flag{background:#fee;padding:.1rem .4rem;border-radius:3px}"
             "</style>",
             "<h1>Refinement review</h1>",
             "<p>red = removed · green = added · "
             "<b style='color:#08f'>blue = relabeled</b> (cross-structure). "
             "Sorted relabeled-first.</p>"]
    for r in rows:
        flag = (f" <span class=flag>FUSED×{r['n_flag_components']} — keeps model "
                f"boundary, needs review</span>") if r["flagged"] else ""
        parts.append(f"<div class=c><b>{r['token']}</b> ({r['config']}) — "
                     f"removed {r['removed']}, added {r['added']}, "
                     f"<span class=k>relabeled {r['relabeled']}</span>{flag}<br>")
        for p in r["pngs"]:
            parts.append(f"<img src='{p}' loading=lazy>")
        parts.append("</div>")
    (out / "index.html").write_text("\n".join(parts))


if __name__ == "__main__":
    raise SystemExit(main())
