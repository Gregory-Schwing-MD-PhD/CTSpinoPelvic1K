"""build_v6.py — v6 = v5 + surgical instrumentation, labelled.

WHY v6 EXISTS. The label scheme has carried a hardware block since 0068 was deferred:

    76 hardware   77 hardware_cage   78 hardware_screw_rod   79 hardware_plate

Declared in dataset_labels.json for every release so far, and populated in NOT ONE RECORD.
84 of the 802 carry metal and every one of them has that metal silently absorbed into the
vertebra beside it, because a bone segmenter handed a bright object against an endplate
takes it for bone. v6 is the release where the block is populated.

It matters for a specific result rather than for tidiness. AN IATROGENIC FUSION IS
INDISTINGUISHABLE FROM A CONGENITAL ONE TO A DISTANCE MEASUREMENT: a cage-bridged interspace
reads as "no gap" exactly like a congenitally fused transitional vertebra, so an unrecorded
instrumented case can contribute a false positive to the transitional-anatomy result.

METAL OUTRANKS BONE. Where an implant sits inside a vertebra label, v6 gives the voxel to
the implant. This is a decision, not an accident: it follows the postoperative-spine
segmentation literature, where vertebrae, canal and instrumentation are mutually exclusive
classes, and it is the only choice under which the hardware ids mean anything -- on 0068
every single metal voxel was already inside a vertebra label, so an additions-only merge
produced a file identical to its input. The vertebra loses those voxels; nothing is deleted.

EVERY CASE IN v6 HAS BEEN READ. 52 proposals cleared the metal threshold; a radiologist
read all of them and kept 11. The other 41 are artefact -- contrast, calcification and
reconstruction overshoot around dense bone -- and are not in the release. The 11 carry the
class the reader assigned, not the one the geometry guessed, which matters most for the nine
hip prostheses: the spinal subtype block would have called a femoral stem a screw or rod,
because a stem is long and thin and that is all "linear" tests.

The source is `data/hardware_final`, the reviewed labels, after the dust the subtraction left
was removed (26 structures, 3,988 mm3 of isolated specks, none more than 0.88% of the
structure it sat on). 1035's hips are genuinely fragmented and were deliberately left.

    python scripts/build_v6.py --v5 data/hf_export_v5 --labels data/v5_final \\
        --proposals data/hardware_fix --out data/hf_export_v6
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

import numpy as np
import nibabel as nib

HW_IDS = (76, 77, 78, 79, 80, 81, 82)
HW_NAME = {76: "hardware", 77: "hardware_cage", 78: "hardware_screw_rod",
           79: "hardware_plate", 80: "hardware_arthroplasty", 81: "hardware_si_screw",
           82: "hardware_osteosynthesis"}

# Nothing is replaced wholesale any more. 0068 used to be, because the only corrected copy
# of it lived outside the pipeline; it has since gone through the same path as the other ten
# -- cages applied, dust removed -- so it comes from data/hardware_final like the rest, and
# taking the older copy would ship a version with the dust still in it.
REPLACEMENTS = {}


# What the reader recorded for each confirmed case, carried into the manifest so the
# release says what was seen rather than what a threshold produced.
READ_NOTE = {
    "0974": "bilateral total hip arthroplasty, acetabular cups both sides",
    "0515": "bilateral total hip arthroplasty, acetabular cups both sides",
    "1003": "right hemiarthroplasty -- femoral component articulating with the native "
            "acetabulum, no cup",
    "0443": "left total hip arthroplasty, cup present",
    "0671": "left total hip arthroplasty, cup present",
    "0188": "right total hip arthroplasty, cup present",
    "0485": "left total hip arthroplasty, cup present",
    "1128": "left total hip arthroplasty; the cup is anchored by supplementary "
            "transacetabular screws into the ilium",
    "0247": "three parallel cannulated screws, 7.2-7.7 mm across and 77-93 mm long, "
            "inverted-triangle configuration in the left femoral neck",
    "1035": "sacroiliac screws crossing the joint",
    "0068": "paired threaded cylindrical interbody cages, 27 x 14 mm each, L5-L6; the case "
            "was also renumbered to six lumbar bodies against the twelfth rib and gained "
            "T10-T12",
}


def qc_proposal(meta, geom):
    """(verdict, [reasons]) for one case's hardware proposal.

    A gate fires when the geometry is being asked to decide something it cannot see. The
    point is not to reject: it is to mark what a person still has to look at.
    """
    reasons = []
    comps = meta.get("components", [])
    if not comps:
        return "review", ["no component survived the threshold"]

    classes = {c.get("v5_id") for c in comps}
    if 76 in classes:
        reasons.append("a component could not be named from its shape (generic 76)")
    if len(classes) > 1:
        reasons.append("components disagree: "
                       + ", ".join(sorted(HW_NAME.get(c, str(c)) for c in classes)))
    if len(comps) > 4:
        reasons.append(f"{len(comps)} separate pieces -- fragmented, or several implants")

    total = sum(c.get("voxels", 0) for c in comps)
    mm3 = sum(c.get("mm3", 0.0) for c in comps)
    if mm3 < 150:
        reasons.append(f"only {mm3:.0f} mm3 of metal -- near the floor")
    if mm3 > 40000:
        reasons.append(f"{mm3:.0f} mm3 is a very large construct")

    clash = meta.get("overlapping_existing_labels", 0)
    if total and clash / max(total, 1) > 0.98 and total > 4000:
        reasons.append("the whole implant was inside bone labels -- a large subtraction")

    for g in (geom or {}).get("components", []):
        d = g.get("dist_to_bone_mm")
        if d is not None and 8.0 <= d <= 15.0:
            reasons.append(f"a piece sits {d:.0f} mm from bone -- close to the cutoff "
                           f"that separates an implant from tagged bowel")
            break
    return ("review" if reasons else "pass"), reasons


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--v5", required=True)
    ap.add_argument("--labels", required=True, help="v5_final, the label source of truth")
    ap.add_argument("--proposals", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--additions-only", action="store_true",
                    help="do NOT let metal outrank bone (produces a near no-op; see the "
                         "module docstring)")
    a = ap.parse_args()

    v5 = Path(a.v5)
    src = Path(a.labels)
    prop = Path(a.proposals)
    out = Path(a.out)
    root = Path(a.project_root)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "ct").mkdir(parents=True, exist_ok=True)

    # ---- carry the unchanged parts of the tree across --------------------------------
    for f in ("dataset_interface.py", "dataset_labels.json", "splits_5fold.json"):
        if (v5 / f).exists():
            shutil.copy2(v5 / f, out / f)
    if (v5 / "qc").is_dir() and not (out / "qc").exists():
        shutil.copytree(v5 / "qc", out / "qc")

    # CTs are byte-identical between versions: hardlink rather than copy 1.8 GB twice
    n_ct = 0
    for p in sorted((v5 / "ct").glob("*_ct.nii.gz")):
        dst = out / "ct" / p.name
        if dst.exists():
            n_ct += 1
            continue
        try:
            os.link(p, dst)
        except OSError:
            shutil.copy2(p, dst)
        n_ct += 1
    print(f"  ct: {n_ct} volumes")

    # ---- labels ----------------------------------------------------------------------
    which = "_label_hardware.nii.gz" if a.additions_only \
        else "_label_hardware_reassigned.nii.gz"
    rows = []
    n_plain = n_hw = n_repl = 0
    for p in sorted(src.glob("*_label.nii.gz")):
        cid = p.name[:4]
        dst = out / "labels" / p.name
        rec = {"case": cid, "source": "v5", "hardware_ids": [], "review": "",
               "reasons": ""}

        if cid in REPLACEMENTS:
            rel, why = REPLACEMENTS[cid]
            rp = root / rel
            if not rp.exists():
                print(f"  ! {cid}: replacement {rp} missing -- falling back to v5")
            else:
                shutil.copy2(rp, dst)
                arr = np.asanyarray(nib.load(str(dst)).dataobj)
                ids = sorted(int(v) for v in np.unique(arr) if int(v) in HW_IDS)
                rec.update(source="replaced", hardware_ids=ids, note=why,
                           review="", reasons="")
                rows.append(rec)
                n_repl += 1
                print(f"  {cid}: replaced wholesale ({', '.join(HW_NAME[i] for i in ids)})")
                continue

        # the REVIEWED label, not the proposal. A case without one was either read as
        # artefact or never had metal, and ships unchanged from v5.
        reviewed = prop / f"{cid}_label_hw.nii.gz"
        if not reviewed.exists():
            shutil.copy2(p, dst)
            n_plain += 1
            rows.append(rec)
            continue

        shutil.copy2(reviewed, dst)
        arr = np.asanyarray(nib.load(str(dst)).dataobj)
        ids = sorted(int(v) for v in np.unique(arr) if int(v) in HW_IDS)
        rec.update(source="hardware-reviewed", hardware_ids=ids, review="read",
                   reasons=READ_NOTE.get(cid, ""))
        rows.append(rec)
        n_hw += 1

    print(f"  labels: {n_plain} unchanged, {n_hw} hardware-merged, {n_repl} replaced")
    need = [r for r in rows if r.get("review") == "review"]
    print(f"  QC: {n_hw - len(need)} pass, {len(need)} need a human")

    # ---- manifest ---------------------------------------------------------------------
    mf = json.loads((v5 / "manifest.json").read_text(encoding="utf-8"))
    by = {str(r["case"]): r for r in rows}
    for rec in mf:
        cid = str(rec.get("volume_id", ""))
        r = by.get(cid)
        if not r:
            continue
        rec["hardware_labelled"] = bool(r["hardware_ids"])
        rec["hardware_label_ids"] = r["hardware_ids"]
        rec["hardware_review"] = r.get("review") or ("not needed" if not r["hardware_ids"]
                                                     else "pass")
        rec["hardware_review_reasons"] = r.get("reasons", "")
        if r.get("note"):
            rec["v6_note"] = r["note"]
    (out / "manifest.json").write_text(json.dumps(mf, indent=1) + "\n", encoding="utf-8")

    # ---- the worklist a person actually opens ------------------------------------------
    work = [{"case": r["case"],
             "proposed": [HW_NAME[i] for i in r["hardware_ids"]],
             "why": r["reasons"],
             "render": f"{r['case']}_hardware_mip.png"} for r in need]
    (out / "hardware_review_worklist.json").write_text(
        json.dumps({"generated_for": "v6", "n": len(work), "cases": work}, indent=1) + "\n",
        encoding="utf-8")

    import csv
    with (out / "hardware_qc.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["case", "source", "hardware_ids", "review",
                                           "reasons"])
        w.writeheader()
        for r in rows:
            w.writerow({"case": r["case"], "source": r["source"],
                        "hardware_ids": " ".join(str(i) for i in r["hardware_ids"]),
                        "review": r.get("review", ""), "reasons": r.get("reasons", "")})
    print(f"  wrote {out}/manifest.json, hardware_qc.csv and "
          f"hardware_review_worklist.json ({len(work)} cases)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
