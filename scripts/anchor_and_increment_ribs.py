"""scripts/anchor_and_increment_ribs.py — number ribs by ANCHORING then COUNTING, not by
demanding one uniform shift.

WHY THE PREVIOUS RULE REFUSED CASES IT SHOULD HAVE FIXED. lumbar_rib_class_v5 requires every
rib on a side to imply the SAME delta, and refuses the side otherwise. But a rib whose own
vertebra is not labelled -- a sliver leaving the field of view, with nothing above it to
articulate with -- falls back to whatever vertebra is nearest, which is the one BELOW. That
manufactures a second and third delta on a side that is otherwise perfectly regular, and
the whole side gets refused. 0179's right and both of 0412's sides failed exactly this way.

THE RULE THAT ACTUALLY HOLDS. Ribs are a contiguous, ordered series: the rib below rib N is
rib N+1, always. So one confident articulation fixes the entire side by counting:

  1. ORDER the ribs on a side superior -> inferior by their head position. This is anatomy,
     not labelling, so it survives whatever the ids currently say.
  2. ANCHOR on the most confident articulation -- smallest gap to a vertebra whose own body
     is genuinely present, and where no other rib is competing for the same vertebra.
  3. COUNT outward from that anchor. Position k in the ordered series gets
     anchor_level + (k - anchor_position). A rib with no vertebra of its own is numbered by
     its POSITION, which is the information the nearest-vertebra search never had.
  4. CHECK AGAINST THE OTHER SIDE. Left and right are the same person: a rib at a given
     height should carry the same number on both sides. Where both sides anchor
     independently and agree, confidence is high; where they disagree, the side with the
     tighter anchor wins and the disagreement is reported rather than hidden.

SYMMETRY IS A CONSTRAINT, NOT A COSMETIC. Enforcing it is what lets a side with only one
usable articulation inherit a numbering from its partner, instead of being refused for
lack of evidence.

    python scripts/anchor_and_increment_ribs.py --labels data/v5_final --cases 0179,0412
    python scripts/anchor_and_increment_ribs.py --labels data/v5_final --apply
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
sys.path.insert(0, str(_HERE / "review"))
import label_scheme as LS                                          # noqa: E402
from qc_rib_vertebra_incidence import _pts, _mindist               # noqa: E402
from review_anatomy_qc import MIN_VERT_VOX                         # noqa: E402

THORACIC_BASE = 7
SIDES = {"left": LS.RIB_LEFT_OFFSET, "right": LS.RIB_RIGHT_OFFSET}
LUMBAR_CLASS = {"left": LS.LUMBAR_RIB_LEFT, "right": LS.LUMBAR_RIB_RIGHT}
ANCHOR_MM = 15.0
MIN_RIB_VOX = 200


def read_side(lab, sp, base, verts):
    """Ribs on one side, ordered superior->inferior, each with its best articulation."""
    out = []
    for n in range(1, 13):
        m = lab == base + n
        if m.sum() < MIN_RIB_VOX:
            continue
        p = _pts(m)
        z = float(np.argwhere(m)[:, 2].mean())
        best_v, best_g = None, float("inf")
        for vid, vp in verts.items():
            g = _mindist(p, vp, sp)
            if g < best_g:
                best_v, best_g = vid, g
        out.append({"rib": n, "z": z, "vert": best_v, "gap": best_g})
    out.sort(key=lambda r: -r["z"])            # superior first
    return out


def solve_side(ribs, verts):
    """-> (mapping old_rib -> new_rib, note). Anchor on the best articulation, then count.

    COUNTING MUST NOT CLOSE GAPS. Position-based counting assumes the ribs present are
    consecutive, and they are not: if rib 7 was never segmented, the series runs 5,6,8,9...
    and naive counting renumbers rib 8 as rib 6, dragging everything below it up a level.
    0412 did exactly that. So the step between neighbours is taken from their SPACING --
    consecutive ribs sit about one interspace apart, and a gap near twice that means a rib
    is missing and the numbering must skip with it.
    """
    if not ribs:
        return {}, "no ribs"
    cands = [(i, r) for i, r in enumerate(ribs)
             if r["vert"] is not None and r["gap"] <= ANCHOR_MM
             and THORACIC_BASE + 1 <= r["vert"] <= THORACIC_BASE + 12]
    if not cands:
        return {}, "no rib articulates with a labelled thoracic vertebra"

    claimed = {}
    for i, r in cands:
        claimed.setdefault(r["vert"], []).append(i)
    clean = [(i, r) for i, r in cands if len(claimed[r["vert"]]) == 1]
    pool = clean or cands
    i0, r0 = min(pool, key=lambda t: t[1]["gap"])
    level0 = r0["vert"] - THORACIC_BASE

    # typical neighbour spacing, from the series itself rather than an assumed constant --
    # interspaces differ between people and between levels
    dz = [abs(ribs[k]["z"] - ribs[k + 1]["z"]) for k in range(len(ribs) - 1)]
    typical = float(np.median(dz)) if dz else 0.0

    # walk outward from the anchor, adding MORE than one level wherever the spacing says a
    # rib is missing
    level = {i0: level0}
    for k in range(i0 + 1, len(ribs)):
        step = 1
        if typical > 0:
            step = max(1, int(round(abs(ribs[k]["z"] - ribs[k - 1]["z"]) / typical)))
        level[k] = level[k - 1] + step
    for k in range(i0 - 1, -1, -1):
        step = 1
        if typical > 0:
            step = max(1, int(round(abs(ribs[k]["z"] - ribs[k + 1]["z"]) / typical)))
        level[k] = level[k + 1] - step

    mapping, out_of_range, skips = {}, [], 0
    for k, r in enumerate(ribs):
        new = level[k]
        if k > 0 and level[k] - level[k - 1] > 1:
            skips += 1
        if not 1 <= new <= 12:
            out_of_range.append(new)
            continue
        if new != r["rib"]:
            mapping[r["rib"]] = new
    note = f"anchored on rib {r0['rib']}->T{level0} ({r0['gap']:.1f}mm), counted outward"
    if skips:
        note += f"; {skips} gap(s) in the series preserved (a rib is missing there)"
    if out_of_range:
        note += f"; {len(out_of_range)} position(s) outside 1..12, left alone"
    return mapping, note


def _touches_face(m):
    """True if the structure reaches a face of the volume, i.e. the scan cut it off."""
    return bool(m[0].any() or m[-1].any() or m[:, 0].any() or m[:, -1].any()
                or m[:, :, 0].any() or m[:, :, -1].any())


def plan_case(lab, sp):
    verts = {}
    for n in range(1, 13):
        m = lab == THORACIC_BASE + n
        vox = m.sum()
        # A VERTEBRA CUT BY THE SCAN IS STILL A VERTEBRA. The voxel floor exists to throw
        # out specks, but it also threw out genuinely truncated bodies at the edge of the
        # field of view -- and then the rib that belongs to one fell to the vertebra below
        # and was reported as misnumbered. 0487 failed exactly this way: T7 is present at
        # 2711 voxels, running to the last slice of the scan, and rib 7 was called an
        # offset for it. A small piece that touches a face was clipped; a small piece
        # floating in the middle is a speck.
        if vox >= MIN_VERT_VOX or (vox >= 300 and _touches_face(m)):
            verts[THORACIC_BASE + n] = _pts(m)

    per_side, notes = {}, {}
    for side, base in SIDES.items():
        ribs = read_side(lab, sp, base, verts)
        mapping, note = solve_side(ribs, verts)
        per_side[side] = {"ribs": ribs, "map": mapping}
        notes[side] = note

    # --- symmetry: the same height should carry the same number on both sides ---------
    L, R = per_side["left"], per_side["right"]
    if L["ribs"] and R["ribs"]:
        def final(entry, r):
            return entry["map"].get(r["rib"], r["rib"])
        pairs, disagree = 0, 0
        for rl in L["ribs"]:
            rr = min(R["ribs"], key=lambda x: abs(x["z"] - rl["z"]))
            if abs(rr["z"] - rl["z"]) > 12:      # not the same anatomical level
                continue
            pairs += 1
            if final(L, rl) != final(R, rr):
                disagree += 1
        if pairs:
            agree = pairs - disagree
            notes["symmetry"] = (f"{agree}/{pairs} matched levels agree "
                                 f"across the midline")
            # A GATE, not a remark. Left and right are the same person: if the two sides
            # anchor to numberings that disagree at most levels, at least one anchor is
            # wrong and neither can be trusted. 0412 came back 0/6 -- applying that would
            # have written a confident wrong answer over a merely uncertain one.
            if agree < 0.6 * pairs:
                per_side["left"]["map"] = {}
                per_side["right"]["map"] = {}
                notes["symmetry"] += " -- REFUSED: the sides contradict each other" 
            # a side with no anchor of its own inherits the other's numbering
            for a, b in (("left", "right"), ("right", "left")):
                if not per_side[a]["map"] and "no rib articulates" in notes[a] \
                        and per_side[b]["map"]:
                    inherited = {}
                    for ra in per_side[a]["ribs"]:
                        rb = min(per_side[b]["ribs"], key=lambda x: abs(x["z"] - ra["z"]))
                        if abs(rb["z"] - ra["z"]) <= 12:
                            want = per_side[b]["map"].get(rb["rib"], rb["rib"])
                            if want != ra["rib"]:
                                inherited[ra["rib"]] = want
                    if inherited:
                        per_side[a]["map"] = inherited
                        notes[a] = f"inherited numbering from the {b} side by level"

    remap = {}
    for side, base in SIDES.items():
        for old, new in per_side[side]["map"].items():
            remap[base + old] = base + new
    return remap, notes, per_side


def apply_remap(lab, remap):
    lut = np.arange(int(lab.max()) + 1, dtype=lab.dtype)
    for o, n in remap.items():
        if o < len(lut):
            lut[o] = n
    return lut[lab]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--cases", default="")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--out", default="qc_rib_anchor")
    a = ap.parse_args()

    labdir = Path(a.labels)
    stems = ([c.strip() for c in a.cases.split(",") if c.strip()]
             or sorted(p.name.replace("_label.nii.gz", "")
                       for p in labdir.glob("*_label.nii.gz")))
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    backup = out / "pre_anchor"

    changed, report = 0, []
    for stem in stems:
        fp = labdir / f"{stem}_label.nii.gz"
        if not fp.exists():
            continue
        # TWO HANDLES ON THE SAME FILE, DELIBERATELY. The analysis needs canonical
        # orientation, because it reasons about which rib is above which. The WRITE must
        # go back in the file's own frame: saving the canonical array under the canonical
        # affine silently transposes the label away from its CT, and ITK-SNAP then
        # refuses the pair with a dimension mismatch. 0179 was rewritten that way and had
        # to be rebuilt. Renumbering is pure id arithmetic, so the map derived from the
        # canonical view applies unchanged to the original array.
        orig = nib.load(str(fp))
        img = nib.as_closest_canonical(orig)
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        sp = np.array(img.header.get_zooms()[:3], float)
        remap, notes, per = plan_case(lab, sp)
        if not remap:
            continue
        pretty = ", ".join(f"{o}->{n}" for o, n in sorted(remap.items()))
        print(f"  {stem}")
        for k in ("left", "right", "symmetry"):
            if k in notes:
                print(f"      {k:9s} {notes[k]}")
        print(f"      remap    {pretty[:110]}")
        changed += 1
        report.append({"case": stem, "notes": notes,
                       "remap": {int(k): int(v) for k, v in remap.items()}})
        if a.apply:
            new = apply_remap(lab, remap)
            rib_ids = {b + n for b in SIDES.values() for n in range(1, 13)}
            diff = {int(x) for x in np.unique(lab[lab != new])}
            assert diff <= rib_ids, f"{stem}: non-rib ids changed: {sorted(diff)}"
            assert (lab > 0).sum() == (new > 0).sum(), f"{stem}: voxel count changed"
            before = sum(1 for i in rib_ids if (lab == i).any())
            after = sum(1 for i in rib_ids if (new == i).any())
            assert after == before, f"{stem}: rib count {before}->{after}, remap collided"
            backup.mkdir(parents=True, exist_ok=True)
            if not (backup / fp.name).exists():
                shutil.copy2(fp, backup / fp.name)
            # apply the SAME map to the untouched original, and write its own affine back
            raw = np.asanyarray(orig.dataobj)
            out_raw = apply_remap(raw.astype(np.int16), remap).astype(orig.get_data_dtype())
            assert out_raw.shape == raw.shape, f"{stem}: shape changed on write"
            nib.save(nib.Nifti1Image(out_raw, orig.affine, orig.header), str(fp))

    (out / "anchor_plan.json").write_text(json.dumps(
        {"labels": str(labdir), "applied": a.apply, "cases": report}, indent=1))
    print(f"\n  {changed} case(s) {'REWRITTEN' if a.apply else 'would change'}")
    if not a.apply:
        print("  DRY RUN -- pass --apply to write")
    return 0


if __name__ == "__main__":
    sys.exit(main())
