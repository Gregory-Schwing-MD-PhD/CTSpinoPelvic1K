"""The same rib checks on two label sets: immediately after pseudolabelling (v4) and the release.

Runs the review gates as defined in scripts/review_anatomy_qc.py -- rib_label_mixing (two
numbers on one bone), rib_spine_gap (rib detached from the spine), rib_vertebra_match (rib
offset from its vertebra, rib on a lumbar body) -- plus a per-rib fragmentation count, over
every record of one stage. Every record's result is appended to a JSONL file as soon as it is
done, so a killed job resumes where it stopped, and the stage summary is rebuilt from the
JSONL at the end. Numbering semantics never enter the mixing / gap / fragment checks, so v4 and
v10 id schemes compare directly; the vertebra-match check names ribs by the vertebra they touch.

    python rib_qc_stages.py --labels ~/tmp/v4_labels/labels --name v4_pseudolabel --out ~/tmp/rib_qc_stages
    python rib_qc_stages.py --labels ~/data/CTSpinoPelvic1K/labels --name release --out ~/tmp/rib_qc_stages
    python rib_qc_stages.py --summarise --out ~/tmp/rib_qc_stages       # -> rib_qc_stages.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import nibabel as nib
import numpy as np
from scipy import ndimage

sys.path.insert(0, os.path.expanduser("~/CTSpinoPelvic1K/scripts"))
import review_anatomy_qc as RA  # noqa: E402
import label_scheme as LS  # noqa: E402


def fragments(lab):
    """Rib ids present in two or more substantial pieces (second piece >= 15% and >= 50 voxels)."""
    st = ndimage.generate_binary_structure(3, 3)
    bad = 0
    for rid in range(LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 13 + 1):
        m = lab == rid
        if m.sum() < 50:
            continue
        cc, k = ndimage.label(m, structure=st)
        if k < 2:
            continue
        sizes = np.sort(np.bincount(cc.ravel())[1:])[::-1]
        if sizes[1] >= 50 and sizes[1] >= 0.15 * sizes[0]:
            bad += 1
    return bad


def one(p):
    t0 = time.time()
    try:
        im = nib.load(str(p)); lab = np.asanyarray(im.dataobj).astype(np.int16)
        mix_ok, mix_msg = RA.rib_label_mixing(lab, im.affine)
        gap_ok, gap_msg = RA.rib_spine_gap(lab, im.affine)
        vm_ok, vm_msg = RA.rib_vertebra_match(lab, im.affine)
        return {"case": p.name, "mix_fail": not mix_ok, "n_mix": sum(1 for m in mix_msg if m.startswith("X")),
                "gap_fail": not gap_ok, "n_gap": sum(1 for m in gap_msg if m.startswith("X")),
                "vm_fail": not vm_ok, "n_misnumbered": sum(1 for m in vm_msg if m.startswith("X")),
                "n_lumbar_rib_notes": sum(1 for m in vm_msg if m.startswith("note") and "lumbar" in m.lower()),
                "n_nearest_notes": sum(1 for m in vm_msg if m.startswith("note") and "nearest vertebra" in m),
                "n_fragmented_ribs": fragments(lab), "vm_msgs": [m for m in vm_msg if m.startswith("X")][:20],
                "seconds": round(time.time() - t0, 1)}
    except Exception as e:  # noqa: BLE001
        return {"case": p.name, "error": repr(e)[:200], "seconds": round(time.time() - t0, 1)}


def summarise(rows, d):
    ok = [r for r in rows if "error" not in r]
    return {"dir": str(d), "records": len(rows), "errors": len(rows) - len(ok),
            "records_with_two_numbers_on_one_bone": sum(r["mix_fail"] for r in ok), "bones_with_two_numbers": sum(r["n_mix"] for r in ok),
            "records_with_rib_detached_from_spine": sum(r["gap_fail"] for r in ok), "ribs_detached": sum(r["n_gap"] for r in ok),
            "records_with_fragmented_rib": sum(r["n_fragmented_ribs"] > 0 for r in ok), "fragmented_ribs": sum(r["n_fragmented_ribs"] for r in ok),
            "records_with_rib_offset_from_vertebra": sum(r["vm_fail"] for r in ok), "ribs_offset_from_vertebra": sum(r["n_misnumbered"] for r in ok),
            "records_with_rib_on_lumbar_body_note": sum(r["n_lumbar_rib_notes"] > 0 for r in ok),
            "records_passing_all_gates": sum((not r["mix_fail"]) and (not r["gap_fail"]) and (not r["vm_fail"]) and r["n_fragmented_ribs"] == 0 for r in ok),
            "mean_seconds_per_record": round(float(np.mean([r["seconds"] for r in ok])), 1) if ok else None}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels")
    ap.add_argument("--name")
    ap.add_argument("--out", required=True, help="prefix; <out>.<name>.rows.jsonl and <out>.json")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 8)))
    ap.add_argument("--summarise", action="store_true")
    a = ap.parse_args()
    if a.summarise:
        out = {}
        for p in sorted(Path(a.out).parent.glob(Path(a.out).name + ".*.rows.jsonl")):
            name = p.name.split(".")[1]
            rows = [json.loads(l) for l in open(p) if l.strip()]
            out[name] = summarise(rows, rows[0].get("dir", "") if rows else "")
        json.dump(out, open(a.out + ".json", "w"), indent=1); print(json.dumps(out, indent=1))
        return 0
    rowsp = Path(f"{a.out}.{a.name}.rows.jsonl")
    done = {json.loads(l)["case"] for l in open(rowsp) if l.strip()} if rowsp.exists() else set()
    files = [p for p in sorted(Path(a.labels).glob("*.nii.gz")) if p.name not in done]
    print(f"{a.name}: {len(files)} records to do ({len(done)} done), {a.workers} workers", flush=True)
    t0 = time.time(); n = 0
    with open(rowsp, "a") as f, Pool(a.workers) as pool:
        for r in pool.imap_unordered(one, files, chunksize=1):
            f.write(json.dumps(r) + "\n"); f.flush(); n += 1
            if n % 25 == 0:
                print(f"  {n}/{len(files)}  {time.time() - t0:.0f}s  last {r['case']} {r.get('seconds')}s", flush=True)
    rows = [json.loads(l) for l in open(rowsp) if l.strip()]
    print(a.name, json.dumps(summarise(rows, a.labels), indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
