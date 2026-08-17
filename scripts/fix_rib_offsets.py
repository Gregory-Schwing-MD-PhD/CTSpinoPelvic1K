"""Renumber whole rib cages that are consistently off by one level.

THE VERTEBRA WINS. `docs/SPINE_REVIEW.md` anchors the spine at S1 and counts up, and ribs
never rename a vertebra -- so when a rib and the vertebra it articulates with disagree, the
rib label is what moves. qc_rib_vertebra_incidence records the disagreement as

    delta = (thoracic level the rib actually touches) - (rib number)

so the correction is `new_rib = old_rib + delta`. Getting that sign backwards produces
rib 13 and is the obvious way to silently wreck the dataset, so the range check below is
not decoration.

ONLY A WHOLE CAGE IS SAFE TO SHIFT. A case qualifies when all three hold:

    1. every offset rib in the case carries the SAME delta, and
    2. NO rib in the case is already correct (match == 0), and
    3. every renumbered rib still lands in 1..12.

Condition 2 is the one that matters and the one a naive "shift by the modal delta" pass
gets wrong. In v5, case 0359 has nine ribs sitting correctly and a single rib 12 off by
-1: shifting that one rib produces two rib 11s and no rib 12, quietly corrupting a case
that was 90% right. Eleven of the twenty-seven affected cases have a fully shifted cage
and are safe; the other sixteen are left for review, where most turn out to be a rib
fragment nearest the wrong body rather than a counting error at all.

LUMBAR RIBS ARE NEVER TOUCHED. A rib on a lumbar vertebra is bucketed `lumbar`, which is
the LSTV phenotype this dataset exists to capture -- a finding, not a defect.

    python scripts/fix_rib_offsets.py --labels data/v5_final --qc qc_rib_incidence_v5
    python scripts/fix_rib_offsets.py ... --apply
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import label_scheme as LS                                        # noqa: E402

SIDES = {"left": LS.RIB_LEFT_OFFSET, "right": LS.RIB_RIGHT_OFFSET}


def plan(qc_dir: Path) -> tuple[list, list]:
    """Split the affected cases into (auto-fixable, left for review)."""
    rows = list(csv.DictReader(open(qc_dir / "rib_incidence.csv")))
    buckets = collections.defaultdict(collections.Counter)
    offs = collections.defaultdict(list)
    for r in rows:
        buckets[r["case"]][r["bucket"]] += 1
        if r["bucket"] == "offset":
            offs[r["case"]].append((r["side"], int(r["rib"]), int(r["delta"])))

    auto, manual = [], []
    for case, ribs in sorted(offs.items()):
        deltas = {d for _, _, d in ribs}
        n_match = buckets[case]["match"]
        if len(deltas) != 1:
            manual.append((case, ribs, "mixed offsets %s" % sorted(deltas)))
            continue
        d = deltas.pop()
        if n_match:
            manual.append((case, ribs,
                           "%d rib(s) already correct -- shifting would collide" % n_match))
            continue
        # the shift moves EVERY rib of that side, not only the ones flagged: a rib whose
        # own vertebra is out of the field is bucketed no_contact, but it belongs to the
        # same miscounted cage and has to travel with it
        auto.append((case, d, ribs))
    return auto, manual


def remap_for(lab: np.ndarray, d: int) -> dict:
    """old id -> new id for every rib present, or {} if any lands outside 1..12."""
    out = {}
    for side, base in SIDES.items():
        for n in range(1, 13):
            old = base + n
            if not (lab == old).any():
                continue
            new_n = n + d
            if not 1 <= new_n <= 12:
                return {}
            out[old] = base + new_n
    return out


def apply_remap(lab: np.ndarray, remap: dict) -> np.ndarray:
    """Rewrite in one pass through a lookup table.

    Sequential assignment would overwrite: shifting -1 turns rib 8 into rib 7, and a later
    pass over rib 7 would then move the freshly written voxels again.
    """
    lut = np.arange(int(lab.max()) + 1, dtype=lab.dtype)
    for old, new in remap.items():
        if old < len(lut):
            lut[old] = new
    return lut[lab]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels", default="data/v5_final")
    ap.add_argument("--qc", default="qc_rib_incidence_v5")
    ap.add_argument("--backup", default="")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    labels, qc = Path(a.labels), Path(a.qc)
    backup = Path(a.backup) if a.backup else qc / "pre_rib_shift"
    auto, manual = plan(qc)

    print(f"  affected cases            {len(auto) + len(manual)}")
    print(f"  auto-fixable (whole cage) {len(auto)}")
    print(f"  left for review           {len(manual)}\n")

    changed, skipped, ribs_fixed = [], [], 0
    for case, d, ribs in auto:
        fp = labels / case
        if not fp.exists():
            print(f"  ! missing {case}")
            continue
        img = nib.load(str(fp))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        remap = remap_for(lab, d)
        if not remap:
            skipped.append((case, d, "renumber would leave 1..12"))
            print(f"  SKIP {case}  {d:+d}  would leave the 1..12 range")
            continue
        moved = sum(1 for _ in remap)
        ribs_fixed += len(ribs)
        pretty = ", ".join(f"{o}->{n}" for o, n in sorted(remap.items()))
        print(f"  {case:22s} {d:+d}  {moved:2d} rib labels   {pretty[:60]}")
        changed.append({"case": case, "delta": d, "remap": remap,
                        "offset_ribs": len(ribs)})
        if a.apply:
            backup.mkdir(parents=True, exist_ok=True)
            if not (backup / case).exists():
                shutil.copy2(fp, backup / case)
            new = apply_remap(lab, remap)
            # sanity: only rib ids may differ, and the voxel count must be conserved
            rib_ids = {b + n for b in SIDES.values() for n in range(1, 13)}
            diff = np.unique(lab[lab != new])
            assert set(int(x) for x in diff) <= rib_ids, \
                f"{case}: non-rib labels changed: {sorted(set(int(x) for x in diff)) }"
            assert (lab > 0).sum() == (new > 0).sum(), f"{case}: voxel count changed"
            nib.save(nib.Nifti1Image(new.astype(img.get_data_dtype()),
                                     img.affine, img.header), str(fp))

    rep = {"labels": str(labels), "qc": str(qc), "applied": a.apply,
           "cases_changed": len(changed), "offset_ribs_addressed": ribs_fixed,
           "changed": changed,
           "left_for_review": [{"case": c, "why": w} for c, _, w in manual],
           "skipped": [{"case": c, "delta": d, "why": w} for c, d, w in skipped]}
    outp = qc / "rib_shift_plan.json"
    outp.write_text(json.dumps(rep, indent=1))

    print(f"\n  cases {'REWRITTEN' if a.apply else 'that would change'}: {len(changed)}")
    print(f"  offset ribs addressed:  {ribs_fixed}")
    if a.apply:
        print(f"  originals kept in:      {backup}")
    else:
        print("  DRY RUN -- pass --apply to write")
    print(f"  wrote {outp}")
    print("\n  left for review:")
    for c, _, w in manual:
        print(f"    {c:22s} {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
