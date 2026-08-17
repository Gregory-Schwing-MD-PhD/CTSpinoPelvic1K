"""scripts/apply_rib_decisions.py — finalize the rib-numbering disputes.

Two modes.

  --suggest   fill in the `suggested` column of decisions.csv, with the reason. The
              suggestion is derived, not guessed: a case is only ever suggested `shift`
              when moving every offset rib SIMULTANEOUSLY leaves no duplicate and nothing
              outside 1..12. Everything else is suggested `keep`, and the reason names
              what the shift would have collided with -- which is usually the argument.

  --apply     read the `decision` column and act:
                 shift  renumber by the per-rib delta, simultaneously
                 keep   no change; the label stands and the proximity metric was fooled
                 flag   no change; recorded for the review tool

WHY SIMULTANEOUS. Testing each rib against its target independently calls a chain a
collision: shifting 9->10, 10->11, 11->12 looks like three collisions and is actually one
permutation. Testing the whole move at once is the only version that gets both right.
In v5 it rescues nothing extra -- the chains all collide at the top end anyway -- but the
naive test would also have refused the two cases that ARE clean.

WHY `keep` IS USUALLY RIGHT. The QC docstring says a couple of stray offsets are "far
more likely a segmentation artefact than a counting mistake", and the collision targets
say the same thing out loud: in most of these the rib the shift would land on is either
already correct, or is a rib articulating with a LUMBAR vertebra -- the LSTV phenotype
this dataset exists to record. Shifting onto it would overwrite a finding with an error.
A floating twelfth rib whose head sits up beside T11 is normal anatomy, not a mislabel.

    python scripts/apply_rib_decisions.py --qc qc_rib_incidence_v5 --suggest
    python scripts/apply_rib_decisions.py --qc qc_rib_incidence_v5 \
        --labels data/v5_final --apply
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


def read_qc(qc: Path):
    rows = list(csv.DictReader(open(qc / "rib_incidence.csv")))
    present = collections.defaultdict(dict)          # (case, side) -> {rib: bucket}
    offs = collections.defaultdict(dict)             # (case, side) -> {rib: delta}
    for r in rows:
        present[(r["case"], r["side"])][int(r["rib"])] = r["bucket"]
        if r["bucket"] == "offset":
            offs[(r["case"], r["side"])][int(r["rib"])] = int(r["delta"])
    return present, offs


def deltas_of(case: str, offs) -> set:
    return {d for side in SIDES for d in offs.get((case, side), {}).values()}


def evaluate(case: str, present, offs):
    """Can every offset rib in this case move at once? Returns (ok, remap, reason)."""
    remap, blockers = {}, []
    for side, base in SIDES.items():
        mov = offs.get((case, side), {})
        if not mov:
            continue
        have = present[(case, side)]
        stay = sorted(set(have) - set(mov))
        new = {n: n + d for n, d in mov.items()}
        final = list(new.values()) + stay
        dup = [x for x, c in collections.Counter(final).items() if c > 1]
        rng = [x for x in new.values() if not 1 <= x <= 12]
        if dup:
            for x in dup:
                who = [n for n, v in new.items() if v == x]
                hit = have.get(x, "")
                blockers.append(f"{side[0].upper()}{who[0]}->{x} hits "
                                f"{'an existing rib' if not hit else hit} rib {x}")
        if rng:
            blockers.append(f"{side[0].upper()}{sorted(mov)} -> {sorted(rng)} outside 1-12")
        if not dup and not rng:
            for n, v in new.items():
                remap[base + n] = base + v
    return (not blockers), remap, "; ".join(blockers)


def apply_remap(lab, remap):
    lut = np.arange(int(lab.max()) + 1, dtype=lab.dtype)
    for old, new in remap.items():
        if old < len(lut):
            lut[old] = new
    return lut[lab]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qc", required=True)
    ap.add_argument("--labels")
    ap.add_argument("--csv", default="")
    ap.add_argument("--suggest", action="store_true")
    ap.add_argument("--apply", action="store_true")
    a = ap.parse_args()

    qc = Path(a.qc)
    dec_path = Path(a.csv) if a.csv else Path("rib_review_sheets/decisions.csv")
    present, offs = read_qc(qc)
    rows = list(csv.DictReader(open(dec_path)))

    if a.suggest:
        for r in rows:
            ok, remap, why = evaluate(r["case"], present, offs)
            ds = deltas_of(r["case"], offs)
            if ok:
                r["suggested"], r["reason"] = "shift", "simultaneous shift is clean"
            elif len(ds) > 1:
                # the cage disagrees with ITSELF, so no single shift can be right and
                # `keep` is not an argument either -- it needs a human on the volume
                r["suggested"] = "flag"
                r["reason"] = (f"cage is internally inconsistent (deltas {sorted(ds)}); "
                               f"no shift fits, and {why}")
            else:
                r["suggested"], r["reason"] = "keep", why or "shift would collide"
        cols = list(rows[0].keys())
        for c in ("suggested", "reason"):
            if c not in cols:
                cols.append(c)
        with open(dec_path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            w.writerows(rows)
        n = sum(1 for r in rows if r["suggested"] == "shift")
        print(f"  suggested shift on {n} of {len(rows)} cases; keep on {len(rows) - n}")
        for r in rows:
            print(f"    {r['case'][:4]:6s} {r['suggested']:6s} {r['reason'][:88]}")
        print(f"\n  wrote {dec_path} -- edit the `decision` column, then --apply")
        return 0

    if not a.apply:
        print("  nothing to do: pass --suggest or --apply")
        return 0
    if not a.labels:
        print("  --apply needs --labels")
        return 2

    labels = Path(a.labels)
    backup = qc / "pre_rib_review"
    done = collections.Counter()
    log = []
    for r in rows:
        d = (r.get("decision") or "").strip().lower()
        case = r["case"]
        if not d:
            done["undecided"] += 1
            continue
        if d in ("keep", "flag"):
            done[d] += 1
            log.append({"case": case, "decision": d, "note": r.get("note", "")})
            continue
        if d != "shift":
            print(f"  ! {case}: unknown decision {d!r}")
            done["bad"] += 1
            continue
        ok, remap, why = evaluate(case, present, offs)
        if not ok:
            print(f"  REFUSED {case}: {why}")
            done["refused"] += 1
            continue
        fp = labels / case
        img = nib.load(str(fp))
        lab = np.asanyarray(img.dataobj).astype(np.int16)
        new = apply_remap(lab, remap)
        rib_ids = {b + n for b in SIDES.values() for n in range(1, 13)}
        diff = {int(x) for x in np.unique(lab[lab != new])}
        assert diff <= rib_ids, f"{case}: non-rib labels changed: {sorted(diff)}"
        assert (lab > 0).sum() == (new > 0).sum(), f"{case}: voxel count changed"
        backup.mkdir(parents=True, exist_ok=True)
        if not (backup / case).exists():
            shutil.copy2(fp, backup / case)
        nib.save(nib.Nifti1Image(new.astype(img.get_data_dtype()), img.affine,
                                 img.header), str(fp))
        print(f"  shifted {case}: {', '.join(f'{o}->{n}' for o, n in sorted(remap.items()))}")
        done["shift"] += 1
        log.append({"case": case, "decision": "shift", "remap": remap})

    out = qc / "rib_review_final.json"
    out.write_text(json.dumps({"labels": str(labels), "counts": dict(done),
                               "decisions": log}, indent=1))
    print(f"\n  {dict(done)}")
    if done["undecided"]:
        print(f"  {done['undecided']} case(s) still have an empty decision column")
    print(f"  wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
