"""
validate_qc_downsample.py -- NON-NEGOTIABLE verdict-preservation check for the live-QC downsample
front-end added to review_anatomy_qc.py.

For every cached CTSpinoPelvic1K label volume it runs BOTH heavy spine gates twice --
full resolution (SPINESURG_QC_DOWNSAMPLE=1, the ground truth) and downsampled (=3) -- and asserts
the boolean PASS/FAIL verdict is IDENTICAL. The downsample is only acceptable if EVERY verdict
matches; if any mismatch, the factor or a threshold scaling is wrong.

Usage:
    python scripts/validate_qc_downsample.py [--factor 3] [--n 16]

    --factor  downsample factor to validate against full-res (default 3; drop to 2 if 3 mismatches)
    --n       max distinct volumes to test (default 16; env QC_VALIDATE_N overrides)
"""
from __future__ import annotations

import argparse
import hashlib
import os
import sys
import time
from pathlib import Path

import numpy as np
import nibabel as nib

sys.path.insert(0, str(Path(__file__).resolve().parent))
import review_anatomy_qc as RA


def _cache_root() -> Path:
    return (Path.home() / ".cache" / "huggingface" / "hub"
            / "datasets--anonymous-mlhc--CTSpinoPelvic1K")


def _find_label_files(root: Path):
    """All cached label .nii.gz volumes: the named files under */labels/ plus any content-addressed
    */blobs/* files. Deduplicated by content (size + hash of head/tail) so an identical volume that
    appears in several snapshots is tested once."""
    cands = []
    cands += list(root.glob("**/labels/*.nii.gz"))
    cands += [p for p in root.glob("**/blobs/*") if p.is_file()]
    seen, out = set(), []
    for p in sorted(cands):
        try:
            sz = p.stat().st_size
        except OSError:
            continue
        h = hashlib.md5()
        h.update(str(sz).encode())
        with open(p, "rb") as f:
            h.update(f.read(262144))
            if sz > 262144:
                f.seek(-131072, os.SEEK_END)
                h.update(f.read(131072))
        key = h.hexdigest()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def _is_label_volume(lab: np.ndarray) -> bool:
    """A label volume (ids 0..57), not a CT: small integer range, has some spine label (1..26)."""
    if lab.size == 0:
        return False
    mx = int(np.asarray(lab).max())
    if mx > 57 or mx < 1:
        return False
    # spot-check it carries spine ids, so we exercise the gates
    sample = lab
    return bool(((sample >= 1) & (sample <= 26)).any())


def _run(check: str, lab, affine, factor: int):
    t0 = time.perf_counter()
    ok, _ = RA.check_label(check, lab, affine, given=None, downsample=factor)
    return bool(ok), time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factor", type=int, default=3)
    ap.add_argument("--n", type=int, default=int(os.environ.get("QC_VALIDATE_N", "16")))
    args = ap.parse_args()

    root = _cache_root()
    if not root.exists():
        print(f"cache not found: {root}")
        raise SystemExit(2)
    files = _find_label_files(root)
    print(f"discovered {len(files)} distinct cached volumes under {root}")

    rows = []                       # (case, gate, ok_full, ok_ds, match, t_full, t_ds)
    tested = 0
    for p in files:
        if tested >= args.n:
            break
        try:
            img = nib.load(str(p))
            lab = np.asanyarray(img.dataobj)
        except Exception as exc:
            print(f"  skip {p.name}: load error {exc}")
            continue
        if not _is_label_volume(lab):
            continue                # a CT blob, not a label -> skip
        affine = img.affine
        case = p.name if "label" in p.name else f"blob:{p.name[:10]}"
        tested += 1
        for gate in ("class_mixing", "spine_extend"):
            okf, tf = _run(gate, lab, affine, 1)
            okd, td = _run(gate, lab, affine, args.factor)
            match = (okf == okd)
            rows.append((case, gate, okf, okd, match, tf, td))
            flag = "MATCH" if match else "**MISMATCH**"
            print(f"  {case:<26} {gate:<13} full={str(okf):<5} ds={str(okd):<5} {flag:<12} "
                  f"full={tf:6.1f}s ds={td:5.2f}s")

    print("\n" + "=" * 78)
    if not rows:
        print("NO volumes tested (nothing cached / all skipped).")
        raise SystemExit(2)
    n_cases = len({r[0] for r in rows})
    all_match = all(r[4] for r in rows)
    mism = [r for r in rows if not r[4]]
    mean_full = float(np.mean([r[5] for r in rows]))
    mean_ds = float(np.mean([r[6] for r in rows]))
    speedup = mean_full / mean_ds if mean_ds > 0 else float("inf")
    print(f"factor              : {args.factor}")
    print(f"cases tested        : {n_cases}  ({len(rows)} gate-runs)")
    print(f"ALL VERDICTS MATCH  : {'YES' if all_match else 'NO'}")
    if mism:
        print("MISMATCHES:")
        for r in mism:
            print(f"    {r[0]} {r[1]}: full={r[2]} ds={r[3]}")
    print(f"mean full-res time  : {mean_full:6.2f}s")
    print(f"mean downsampled    : {mean_ds:6.2f}s")
    print(f"speedup             : {speedup:5.1f}x")
    raise SystemExit(0 if all_match else 1)


if __name__ == "__main__":
    main()
