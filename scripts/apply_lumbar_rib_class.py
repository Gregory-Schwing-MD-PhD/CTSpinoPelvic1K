"""
scripts/apply_lumbar_rib_class.py — integrate the LUMBAR-RIB class into the FINAL labels.

For each label, per side independently:
  1. Find ribs that reach a vertebra (head within ANCHOR_MM) -> reliable rib->vertebra articulation.
  2. A rib on a LUMBAR vertebra (L1..L6) is a lumbar rib -> reassign to rib_left_lumbar(74)/right(75).
  3. From the THORACIC-anchored ribs compute the numbering OFFSET (target = the vertebra's own number,
     e.g. a rib on T12 should be rib 12). If the offset is CONSISTENT across the anchored ribs, apply
     it to every thoracic rib on that side (this is the "+1 shift" a lumbar rib introduces, derived
     from the anatomy rather than assumed).
  4. VERIFY: after the remap every anchored thoracic rib must sit on its own vertebra (rib N <-> T-N).
     A case that does not verify is REJECTED (left unchanged) and flagged -- a bad remap can't ship.

Only cases with a lumbar rib AND a thoracic anchor are touched; FOV-limited (no thoracic) cases are
skipped (the offset is unknowable there). Dry-run by DEFAULT: prints before/after, writes nothing.

    python scripts/apply_lumbar_rib_class.py --labels FINAL_LABELS_DIR            # dry-run
    python scripts/apply_lumbar_rib_class.py --labels DIR --out OUT_DIR --apply   # write remapped
"""
from __future__ import annotations
import argparse, csv, sys
from pathlib import Path
import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS            # noqa: E402

LO, HI = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12       # 34..57
LUM_LO, LUM_HI = 20, 25                                         # L1..L6
ANCHOR_MM = 12.0                                                # head must reach this close to count
LUML, LUMR = LS.LUMBAR_RIB_LEFT, LS.LUMBAR_RIB_RIGHT           # 74, 75


def _articulations(lab, affine):
    """rib_id -> (vertebra_id, gap_mm) from the medial-most (head) rib voxel, incl lumbar (8..25)."""
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    vmask = (lab >= 8) & (lab <= LUM_HI)
    ribs = (lab >= LO) & (lab <= HI)
    if not vmask.any() or not ribs.any():
        return {}
    idx = np.argwhere(vmask | ribs); lo = idx.min(0); hi = idx.max(0) + 1
    sl = tuple(slice(int(lo[i]), int(hi[i])) for i in range(3)); f = 2
    sub = lab[sl][::f, ::f, ::f]
    vv = (sub >= 8) & (sub <= LUM_HI)
    if not vv.any():
        return {}
    d, ind = ndimage.distance_transform_edt(~vv, sampling=spacing * f, return_indices=True)
    out = {}
    for rid in range(LO, HI + 1):
        m = (sub == rid)
        if not m.any():
            continue
        dd = np.where(m, d, np.inf)
        p = np.unravel_index(np.argmin(dd), dd.shape); gap = float(dd[p])
        if not np.isfinite(gap):
            continue
        out[rid] = (int(sub[ind[0][p], ind[1][p], ind[2][p]]), gap)
    return out


def _side(rid):
    return "L" if rid <= LS.RIB_LEFT_OFFSET + 12 else "R"


def _off(rid):
    return LS.RIB_LEFT_OFFSET if rid <= LS.RIB_LEFT_OFFSET + 12 else LS.RIB_RIGHT_OFFSET


def plan_remap(lab, affine):
    """Return (mapping old_id->new_id, notes). mapping empty => no change. Never raises."""
    arts = _articulations(lab, affine)
    anchored = {rid: v for rid, (v, g) in arts.items() if g <= ANCHOR_MM}
    mapping, notes = {}, []
    for side, off, lumid in (("L", LS.RIB_LEFT_OFFSET, LUML), ("R", LS.RIB_RIGHT_OFFSET, LUMR)):
        sids = [r for r in range(LO, HI + 1) if (lab == r).any() and _off(r) == off]
        if not sids:
            continue
        lumbar = [r for r in sids if anchored.get(r, 0) in range(LUM_LO, LUM_HI + 1)]
        if not lumbar:
            continue                                            # no lumbar rib on this side -> leave it
        thor = {r: anchored[r] for r in sids if 8 <= anchored.get(r, 0) <= 19}
        if not thor:
            notes.append(f"{side}: lumbar rib but NO thoracic anchor -> skip (offset unknowable)")
            continue
        offsets = {(v - 7) - (r - off) for r, v in thor.items()}   # target_num - current_num
        if len(offsets) != 1:
            notes.append(f"{side}: inconsistent offsets {sorted(offsets)} -> skip (manual)")
            continue
        delta = offsets.pop()
        for r in lumbar:
            mapping[r] = lumid                                  # lumbar rib -> its own class
        for r in sids:
            if r in lumbar:
                continue
            n = (r - off) + delta
            if not (1 <= n <= 12):
                notes.append(f"{side}: shift would push rib {r-off} to {n} (out of 1..12) -> skip side")
                for rr in lumbar:
                    mapping.pop(rr, None)                       # abort this side's remap
                break
            if delta:
                mapping[r] = off + n
        notes.append(f"{side}: lumbar={[r-off for r in lumbar]} offset={delta:+d}")
    return mapping, notes


def _apply(lab, mapping):
    out = lab.copy()
    for old, new in mapping.items():
        out[lab == old] = new                                  # read original -> no collision
    return out


def _verify(out, affine):
    """After remap, every anchored thoracic rib must sit on its own vertebra (rib N <-> T-N)."""
    arts = _articulations(out, affine)
    for rid, (v, g) in arts.items():
        if g <= ANCHOR_MM and 8 <= v <= 19:
            if (rid - _off(rid)) != (v - 7):
                return False, f"{_side(rid)}{rid-_off(rid)} still on {v-7 and 'T'+str(v-7)} after remap"
    return True, "rib N <-> T-N holds"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels", required=True, help="dir of FINAL label .nii.gz")
    ap.add_argument("--out", help="output dir (required with --apply)")
    ap.add_argument("--apply", action="store_true", help="write remapped labels (default: dry-run)")
    a = ap.parse_args(argv)
    src = Path(a.labels); files = sorted(src.glob("*.nii.gz"))
    if a.apply and not a.out:
        ap.error("--apply requires --out")
    out_dir = Path(a.out) if a.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
    names = {v: k for k, v in LS.label_dict().items()}
    rows = []; remapped = skipped = clean = failed = 0
    for fp in files:
        img = nib.load(str(fp)); lab = np.asanyarray(img.dataobj)
        mapping, notes = plan_remap(lab, img.affine)
        if not mapping:
            clean += 1
            if a.apply and out_dir:
                (out_dir / fp.name).write_bytes(fp.read_bytes())
            continue
        out = _apply(lab, mapping)
        ok, why = _verify(out, img.affine)
        moved = {names.get(o, o): names.get(n, n) for o, n in sorted(mapping.items())}
        if not ok:
            failed += 1
            rows.append({"file": fp.name, "action": "REJECTED", "note": why + " | " + "; ".join(notes),
                         "mapping": str(moved)})
            print(f"  REJECTED {fp.name}: {why}")
            if a.apply and out_dir:
                (out_dir / fp.name).write_bytes(fp.read_bytes())   # ship unchanged
            continue
        remapped += 1
        rows.append({"file": fp.name, "action": "remap" if a.apply else "would-remap",
                     "note": "; ".join(notes), "mapping": str(moved)})
        print(f"  {'REMAP' if a.apply else 'would remap'} {fp.name}: {'; '.join(notes)}")
        for o, n in sorted(mapping.items()):
            print(f"        {names.get(o,o):16s} -> {names.get(n,n)}")
        if a.apply and out_dir:
            nib.save(nib.Nifti1Image(out.astype(lab.dtype), img.affine, img.header),
                     str(out_dir / fp.name))
    with open("lumbar_rib_remap_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "action", "note", "mapping"])
        w.writeheader(); w.writerows(rows)
    print(f"\n{'APPLIED' if a.apply else 'DRY-RUN'}: {remapped} remapped, {failed} rejected(unchanged), "
          f"{clean} no-lumbar-rib, over {len(files)} labels  -> lumbar_rib_remap_report.csv")
    if not a.apply:
        print("re-run with --out DIR --apply to write.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
