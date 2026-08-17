"""scripts/lumbar_rib_class_v5.py — give the 13th rib its own class, then re-anchor the cage.

THE FINDING THIS ACTS ON. In these cases TotalSegmentator numbered the ribs from the
BOTTOM: the rib on L1 got id 12, the rib on T12 got 11, and so on up. Read against the
vertebra it actually touches, the whole side is a uniform +1. That is invisible in the QC
table because qc_rib_vertebra_incidence resolves a rib to its OWN vertebra first
(`if gap_own <= ANCHOR_MM: match`), so rib 10 sitting 3.7mm from T11 and 8.2mm from T10
is scored `match` and the shift it belongs to is broken up into scattered `offset`s.

So this reads by NEAREST vertebra throughout, which is the only reading under which a
cage is either uniformly shifted or genuinely inconsistent.

THE FIX, per side:

    1. the rib whose head is on a LUMBAR body      -> rib_{side}_lumbar (74/75)
    2. every remaining rib                          -> +delta, so the rib on T12 is rib 12

Step 1 is what makes step 2 legal: while the L1 rib is called "rib 12" the slot the
T12 rib needs is occupied, which is exactly why fix_rib_offsets and apply_rib_decisions
both refused these cases. Pull the 13th rib out of the sequence and the shift is free.

THE RULE: A RIB IS NAMED FOR THE VERTEBRA IT ARTICULATES WITH. That is the whole of it,
and it decides the awkward cases rather than leaving them to judgement.

ONE GUARD. Every rib on the side with a vertebra within reach must imply the SAME delta.
A side that disagrees with itself cannot satisfy the rule at all -- naming each rib for
its own vertebra would put two ribs on one number and leave a gap -- so no renumber fits
and the case goes to the review tool.

UNILATERAL IS REPORTED, NOT REFUSED. Where only one side carries the 13th rib (0315,
0660), that side is re-anchored and the contralateral one keeps its own numbering, so
the upper ribs end up one apart across the midline. Those upper ribs are unverifiable
either way -- the thoracic GT is FOV-limited, there are no labelled vertebrae up there --
and the rule is worth more than the symmetry. It is recorded in the note.

    python scripts/lumbar_rib_class_v5.py --labels data/v5_final --qc qc_rib_incidence_v5_fixed
    python scripts/lumbar_rib_class_v5.py ... --apply
"""
from __future__ import annotations

import argparse
import collections
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
from qc_rib_vertebra_incidence import incidence, THORACIC_BASE     # noqa: E402
from review_anatomy_qc import ANCHOR_MM                            # noqa: E402

SIDES = {"left": LS.RIB_LEFT_OFFSET, "right": LS.RIB_RIGHT_OFFSET}
LUMBAR_CLASS = {"left": LS.LUMBAR_RIB_LEFT, "right": LS.LUMBAR_RIB_RIGHT}


def read_side(rows, side):
    """(lumbar rib numbers, {rib: implied delta}, all rib numbers present) for one side.

    Deltas come from the NEAREST vertebra, not the own-vertebra-wins rule -- see the
    module docstring. `match` rows are re-read the same way as everything else.
    """
    lum, delta, have = [], {}, []
    for r in rows:
        if r["side"] != side:
            continue
        n = r["rib"]
        have.append(n)
        near, gap = r["nearest"], r["gap_mm"]
        if not near or gap > ANCHOR_MM:
            continue                                    # no vertebra in reach: no evidence
        if near.startswith("L"):
            lum.append(n)
        elif near.startswith("T"):
            delta[n] = int(near[1:]) - n
    return sorted(lum), delta, sorted(have)


def plan_case(rows):
    """-> (remap, per-side notes). remap is empty when nothing is safe to do."""
    read = {s: read_side(rows, s) for s in SIDES}
    remap, notes = {}, {}

    for side, base in SIDES.items():
        lum, delta, have = read[side]
        other = "right" if side == "left" else "left"
        o_lum, o_delta, o_have = read[other]

        # the reclass itself is never in doubt -- the rib's head is on a lumbar body
        for n in lum:
            remap[base + n] = LUMBAR_CLASS[side]
        head = f"lumbar rib {lum} -> {LUMBAR_CLASS[side]}; " if lum else ""

        # NOT gated on a lumbar rib being present. Re-anchoring is the general rule -- a
        # rib is named for the vertebra it articulates with -- and a side that has already
        # had its 13th rib reclassed has no lumbar rib left to gate on, so gating here
        # would make the second half of a two-pass fix silently do nothing.
        moving = {n: d for n, d in delta.items() if n not in lum}
        ds = set(moving.values())
        if not ds:
            notes[side] = head + "no thoracic evidence -> no renumber"
            continue
        if len(ds) > 1:
            # the one refusal that stands: naming each rib for its own vertebra would put
            # two ribs on one number and leave a gap, so no numbering satisfies the rule
            notes[side] = (head + f"REFUSED renumber: side disagrees with itself "
                                  f"(deltas {sorted(ds)})")
            continue
        d = ds.pop()
        if d == 0:
            notes[side] = head + "already anchored to its vertebrae"
            continue

        out_of_range = [n + d for n in have if n not in lum and not 1 <= n + d <= 12]
        if out_of_range:
            notes[side] = head + f"REFUSED renumber: {out_of_range} outside 1-12"
            continue

        for n in have:
            if n not in lum:
                remap[base + n] = base + n + d
        notes[side] = (head + f"ribs {have[0]}..{max(n for n in have if n not in lum)} "
                              f"{d:+d} (named for the vertebra each touches)")

        # UNILATERAL: reported, not refused. Naming a rib for its own vertebra is the rule;
        # the cost is that the contralateral side keeps its own numbering, so the upper
        # ribs -- which no labelled vertebra can adjudicate, the thoracic GT being
        # FOV-limited -- end up one apart across the midline. Worth knowing, not worth
        # overriding the rule for.
        if not o_lum and o_delta and set(o_delta.values()) == {0} \
                and set(have) & set(o_have):
            notes[side] += (f"  [NOTE: unilateral -- {other} is a clean delta-0 cage, so "
                            f"upper ribs now differ by {d:+d} across the midline]")
    return remap, notes


def apply_remap(lab, remap):
    lut = np.arange(int(lab.max()) + 1, dtype=lab.dtype)
    for old, new in remap.items():
        if old < len(lut):
            lut[old] = new
    return lut[lab]


def check_t12(lab, affine) -> str:
    """After the rewrite: is there a rib 12 on T12? The thing Greg actually asked for."""
    rows = incidence(lab, affine)
    hits = []
    for side in SIDES:
        r = next((x for x in rows if x["side"] == side and x["rib"] == 12), None)
        hits.append(f"{side[0]}12->{r['nearest'] if r else 'absent'}")
    return " ".join(hits)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--qc", default="qc_rib_incidence_v5_fixed")
    ap.add_argument("--cases", default="", help="comma-separated stems; default = every "
                                                "case the QC found a lumbar rib in")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    labels, qc = Path(a.labels), Path(a.qc)
    if a.cases:
        stems = [c.strip() for c in a.cases.split(",") if c.strip()]
    else:
        import csv as _csv
        stems = [r["case"].replace("_label.nii.gz", "")
                 for r in _csv.DictReader(open(qc / "lumbar_rib_phenotype.csv"))]
    print(f"  {len(stems)} case(s)\n")

    changed, refused, report = [], [], []
    for stem in sorted(stems):
        fp = labels / f"{stem}_label.nii.gz"
        if not fp.exists():
            print(f"  ! missing {fp.name}")
            continue
        img = nib.load(str(fp))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        rows = incidence(lab, img.affine)
        remap, notes = plan_case(rows)

        print(f"  {stem}")
        for side in ("left", "right"):
            print(f"      {side:5s}  {notes.get(side, '-')}")
        if any("REFUSED" in v for v in notes.values()):
            refused.append(stem)
        if not remap:
            report.append({"case": stem, "notes": notes, "remap": {}})
            continue

        new = apply_remap(lab, remap)
        rib_ids = {b + n for b in SIDES.values() for n in range(1, 13)}
        diff = {int(x) for x in np.unique(lab[lab != new])}
        assert diff <= rib_ids, f"{stem}: non-rib labels changed: {sorted(diff)}"
        assert (lab > 0).sum() == (new > 0).sum(), f"{stem}: voxel count changed"
        for side, base in SIDES.items():
            ids = [base + n for n in range(1, 13)]
            counts = collections.Counter(int(x) for x in np.unique(new) if x in ids)
            assert all(c == 1 for c in counts.values()), f"{stem}: duplicate rib id"
        after = check_t12(new, img.affine)
        print(f"      after   {after}")
        report.append({"case": stem, "notes": notes,
                       "remap": {int(k): int(v) for k, v in remap.items()},
                       "after": after})
        changed.append(stem)

        if a.apply:
            backup = qc / "pre_lumbar_class"
            backup.mkdir(parents=True, exist_ok=True)
            if not (backup / fp.name).exists():
                shutil.copy2(fp, backup / fp.name)
            nib.save(nib.Nifti1Image(new.astype(img.get_data_dtype()), img.affine,
                                     img.header), str(fp))

    outp = qc / "lumbar_rib_class_plan.json"
    outp.write_text(json.dumps({"labels": str(a.labels), "applied": a.apply,
                                "changed": changed, "refused": refused,
                                "cases": report}, indent=1))
    print(f"\n  cases {'REWRITTEN' if a.apply else 'that would change'}: {len(changed)}")
    print(f"  renumber refused (reclass only / flagged): {len(refused)} {refused}")
    if not a.apply:
        print("  DRY RUN -- pass --apply to write")
    print(f"  wrote {outp}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
