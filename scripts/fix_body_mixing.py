"""fix_body_mixing.py — two vertebra labels sharing one vertebral body.

WHAT THE DEFECT IS. Adjacent vertebra labels always overlap in HEIGHT: a spinous process
runs caudally past the next vertebra's superior endplate, and that is anatomy, not error. In
the vertebral BODY they cannot overlap, because the disc separates them. So a body slice
carrying two different vertebra labels is class mixing and nothing else. On 0068 the L3-L4
pair shared 52 body slices -- 41.6 mm -- split 23/77, which reads as "half L3, half L4" to
anyone scrolling through it.

WHERE IT COMES FROM. On 0068, from the pseudolabel: `prov_spine: pseudo`, never reviewed.
The boundaries that are clean on that case are the ones added afterwards. This repairs a
model's output; it does not overrule an annotator.

THE APPROACH THAT DOES NOT WORK, AND WHY IT IS WORTH SAYING. Seeding each level from the
slices it holds ALONE and letting ambiguous voxels join the nearest seed sounds right and
fails badly: when two labels interleave heavily a level may hold no slice alone, so it gets
no seed and its voxels are handed to a neighbour. Tried on 0068 it moved 130,643 voxels,
gave three quarters of L2 to L3, and left an "L1-L3 adjacency" -- L2 had effectively been
deleted. A method that can silently delete a level is not usable near a release.

WHAT WORKS IS THE DISC. Down the body column the cross-sectional area is a run of plateaus
separated by deep troughs, and the troughs ARE the discs -- on 0068 the profile falls from
about 2,200 voxels a slice to 36, and to 0 at the next level. Cutting the column at the
troughs gives one span per vertebral body, from the image rather than from the labels. Each
span then takes the label that already holds most of it, so the naming still comes from the
existing annotation and only the BOUNDARIES are redrawn.

Three checks decide whether the result is trustworthy, and it refuses rather than guessing:
the spans must come out in the same order as the levels, no level may win two spans, and no
level present in the body column may lose its span altogether.

POSTERIOR ELEMENTS ARE NOT TOUCHED, and may still be mixed afterwards. Laminae, pedicles and
spinous processes legitimately overlap their neighbours and there is no disc to arbitrate
them; guessing there is what the failed method did. The report says what is left.

    python scripts/fix_body_mixing.py --label in.nii.gz --out out.nii.gz
    python scripts/fix_body_mixing.py --label in.nii.gz --report-only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

THOR = list(range(8, 20))
LUMBAR = list(range(20, 26))
LEVELS = THOR + LUMBAR
NAME = {**{v: f"T{v - 7}" for v in THOR}, **{v: f"L{v - 19}" for v in LUMBAR}}
BODY_FRAC = 0.55          # anterior share of the run that counts as vertebral body
TROUGH = 0.35             # a slice below this share of the median plateau is disc


def zc_of(lab, body, v, ax, others):
    """Mean slice index of a level's body, for ordering."""
    z = np.where(((lab == v) & body).any(axis=others))[0]
    return float(z.mean()) if len(z) else 0.0


def shared_pairs(lab, body, present, ax, others, zmm, min_mm):
    """[(a, b, mm, minority %)] for levels sharing body slices."""
    have = [v for v in present if ((lab == v) & body).any()]
    z = {v: np.where(((lab == v) & body).any(axis=others))[0] for v in have}
    order = sorted(have, key=lambda v: z[v].mean())
    out = []
    for a, b in zip(order, order[1:]):
        sh = np.intersect1d(z[a], z[b])
        if not len(sh):
            continue
        na = int(((lab == a) & body).take(sh, axis=ax).sum())
        nb = int(((lab == b) & body).take(sh, axis=ax).sum())
        mm = len(sh) * zmm
        if mm >= min_mm:
            out.append((a, b, mm, 100.0 * min(na, nb) / max(1, na + nb)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--label", required=True)
    ap.add_argument("--out")
    ap.add_argument("--report-only", action="store_true")
    ap.add_argument("--min-shared-mm", type=float, default=4.0)
    a = ap.parse_args()

    img = nib.load(a.label)
    lab = np.asanyarray(img.dataobj).astype(np.int16)
    aff = img.affine
    col = aff[2, :3]
    ax = int(np.argmax(np.abs(col)))
    zmm = float(np.linalg.norm(aff[:3, ax]))
    others = tuple(i for i in range(3) if i != ax)

    present = [v for v in LEVELS if (lab == v).any()]
    if len(present) < 2:
        print("  fewer than two vertebra labels; nothing to arbitrate")
        return 0

    idx = np.argwhere(np.isin(lab, present))
    w = (aff @ np.c_[idx, np.ones(len(idx))].T).T[:, :3]
    front = w[:, 1].max() - BODY_FRAC * (w[:, 1].max() - w[:, 1].min())
    body = np.zeros_like(lab, bool)
    body[tuple(idx[w[:, 1] > front].T)] = True
    print(f"  body column: anterior of y={front:.0f} mm, "
          f"{int(body.sum()):,} of {len(idx):,} labelled voxels")

    before = shared_pairs(lab, body, present, ax, others, zmm, a.min_shared_mm)
    print(f"\n  {'pair':<11} {'shared mm':>10} {'minority':>9}")
    print("  " + "-" * 33)
    for x, y, mm, minor in before:
        print(f"  {NAME[x]}-{NAME[y]:<7} {mm:>10.1f} {minor:>8.0f}%")
    if not before:
        print("  no pair shares a body; nothing to fix")
        return 0
    if a.report_only:
        return 0

    # ---- cut the column at the discs ------------------------------------------------
    prof = body.sum(axis=others)
    nz = prof[prof > 0]
    thr = TROUGH * float(np.median(nz))
    above = prof > thr
    spans = []
    i = 0
    while i < len(above):
        if above[i]:
            j = i
            while j + 1 < len(above) and above[j + 1]:
                j += 1
            spans.append((i, j))
            i = j + 1
        else:
            i += 1
    spans = [s for s in spans if (s[1] - s[0] + 1) * zmm >= 8.0]     # a body is not 5 mm

    # A FUSED INTERSPACE HAS NO TROUGH. Metal or bony bridging fills the disc, the
    # cross-section never drops, and two vertebrae come back as one span. Such a span is
    # skipped rather than arbitrated: there is no image boundary inside it, so the existing
    # division is somebody's inference and a majority vote would delete the smaller level.
    hw = np.isin(lab, (76, 77, 78, 79))
    heights = [(e - b + 1) * zmm for b, e in spans]
    typical = float(np.median(heights))
    fused = []
    for k, (b, e) in enumerate(spans):
        sl = [slice(None)] * 3
        sl[ax] = slice(b, e + 1)
        has_metal = bool(hw[tuple(sl)].any())
        if heights[k] > 1.5 * typical or (has_metal and heights[k] > 1.2 * typical):
            fused.append((b, e, has_metal))
    spans = [sp for sp in spans if not any(sp[0] == b and sp[1] == e for b, e, _ in fused)]
    print(f"\n  disc threshold {thr:.0f} voxels/slice; {len(spans)} body span(s) with a "
          f"disc at each end, {len(fused)} fused")
    for b, e, metal in fused:
        sl = [slice(None)] * 3
        sl[ax] = slice(b, e + 1)
        who = sorted({int(v) for v in np.unique(lab[tuple(sl)][body[tuple(sl)]])
                      if int(v) in present}, key=lambda v: zc_of(lab, body, v, ax, others))
        print(f"    span {b:>4}-{e:<4} ({(e - b + 1) * zmm:>5.1f} mm, "
              f"{'metal in it' if metal else 'no metal'}) holds "
              f"{', '.join(NAME[v] for v in who)} -- FUSED, left as it is")

    # ---- each span keeps the label that already holds most of it --------------------
    order_lo = sorted(spans, key=lambda s: s[0])
    claims = []
    for lo, hi in order_lo:
        sl = [slice(None)] * 3
        sl[ax] = slice(lo, hi + 1)
        sub = lab[tuple(sl)][body[tuple(sl)]]
        counts = {v: int((sub == v).sum()) for v in present}
        win = max(counts, key=counts.get)
        share = 100.0 * counts[win] / max(1, sum(counts.values()))
        claims.append((lo, hi, win, share))
        print(f"    span {lo:>4}-{hi:<4} ({(hi - lo + 1) * zmm:>5.1f} mm) -> "
              f"{NAME[win]:<4} with {share:.0f}% of it")

    # ---- refuse rather than guess ---------------------------------------------------
    wins = [c[2] for c in claims]
    body_levels = [v for v in present if ((lab == v) & body).any()]
    zc = {v: np.where(((lab == v) & body).any(axis=others))[0].mean() for v in body_levels}
    # a level whose body sits mostly inside a fused span was never up for arbitration
    in_fused = set()
    for b, e, _ in fused:
        for v in body_levels:
            m = (lab == v) & body
            z = np.where(m.any(axis=others))[0]
            if len(z) and (b <= z.mean() <= e):
                in_fused.add(v)
    expect = sorted([v for v in body_levels if v not in in_fused], key=lambda v: zc[v])
    if in_fused:
        print(f"  not arbitrated (inside a fused span): "
              f"{', '.join(NAME[v] for v in sorted(in_fused, key=lambda v: zc[v]))}")
    if len(set(wins)) != len(wins):
        dup = [NAME[v] for v in wins if wins.count(v) > 1]
        print(f"  ! {sorted(set(dup))} won more than one span -- refusing")
        return 3
    if len(spans) != len(expect):
        print(f"  ! {len(spans)} spans against {len(expect)} levels with a body "
              f"({', '.join(NAME[v] for v in expect)}) -- refusing")
        return 4
    if wins != sorted(wins, key=lambda v: zc[v]):
        print("  ! the spans do not come out in level order -- refusing")
        return 5

    new = lab.copy()
    moves = {}
    for lo, hi, win, _ in claims:
        sl = [slice(None)] * 3
        sl[ax] = slice(lo, hi + 1)
        reg = np.zeros_like(lab, bool)
        reg[tuple(sl)] = True
        target = reg & body & np.isin(lab, present) & (lab != win)
        for frm in present:
            n = int((target & (lab == frm)).sum())
            if n:
                moves[f"{NAME[frm]}->{NAME[win]}"] = moves.get(f"{NAME[frm]}->{NAME[win]}", 0) + n
        new[target] = win

    # THE DISC GAPS THEMSELVES. Slices below the trough threshold were claimed by no span,
    # so body voxels sitting in them -- the endplate rims -- kept whatever label they had and
    # were the whole of the sharing left after the spans were settled. A voxel in a disc gap
    # belongs to the body it is nearer to, and both neighbours are now known.
    arb = sorted([(lo, hi, wv) for lo, hi, wv, _ in claims], key=lambda t: t[0])
    for (lo1, hi1, w1), (lo2, hi2, w2) in zip(arb, arb[1:]):
        if hi1 + 1 >= lo2:
            continue
        mids = range(hi1 + 1, lo2)
        for z in mids:
            sl = [slice(None)] * 3
            sl[ax] = slice(z, z + 1)
            reg = np.zeros_like(lab, bool)
            reg[tuple(sl)] = True
            tgt = reg & body & np.isin(lab, present)
            if not tgt.any():
                continue
            win = w1 if (z - hi1) <= (lo2 - z) else w2
            for frm in present:
                n = int((tgt & (lab == frm) & (new != win)).sum())
                if n and frm != win:
                    k = f"{NAME[frm]}->{NAME[win]}"
                    moves[k] = moves.get(k, 0) + n
            new[tgt] = win

    changed = new != lab
    assert (new > 0).sum() == (lab > 0).sum(), "voxel count changed"
    assert not (changed & ~body).any(), "a voxel outside the body column was reassigned"
    for v in present:
        assert (new == v).any(), f"{NAME[v]} was deleted"
    print(f"\n  reassigned {int(changed.sum()):,} body voxels: "
          + ", ".join(f"{k} {n:,}" for k, n in sorted(moves.items(), key=lambda x: -x[1])))

    after = shared_pairs(new, body, present, ax, others, zmm, a.min_shared_mm)
    print(f"  bodies still shared: "
          + (", ".join(f"{NAME[x]}-{NAME[y]} {mm:.1f} mm" for x, y, mm, _ in after)
             or "none"))
    print("  posterior elements were not touched and may still be mixed")

    dst = Path(a.out or a.label)
    nib.save(nib.Nifti1Image(new.astype(img.get_data_dtype()), aff, img.header), str(dst))
    stem = dst.name.split(".")[0]
    (dst.parent / f"{stem}_mixing.json").write_text(json.dumps(
        {"before": [[NAME[x], NAME[y], round(mm, 1)] for x, y, mm, _ in before],
         "after": [[NAME[x], NAME[y], round(mm, 1)] for x, y, mm, _ in after],
         "spans": [[lo, hi, NAME[wv], round(sh, 1)] for lo, hi, wv, sh in claims],
         "reassigned": moves, "voxels_reassigned": int(changed.sum()),
         "posterior_elements": "not touched"}, indent=1) + "\n")
    print(f"  wrote {dst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
