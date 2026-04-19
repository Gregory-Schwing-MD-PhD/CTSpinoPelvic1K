#!/usr/bin/env python3
"""
generate_fillsheet.py -- Paper placeholder fill sheet from placed_manifest.json.

Reads placed_manifest.json (and optionally dcm2niix + placement JSON sidecars)
and writes two files into --out_dir:

  paper_fillsheet.txt   Human-readable fill-in sheet, one value per line,
                        ready to copy-paste into the LaTeX source.
  paper_fillsheet.json  Machine-readable dict for programmatic substitution.

Run AFTER place_fused_masks.py has completed.

Usage
-----
    python scripts/generate_fillsheet.py \\
        --manifest  data/placed/placed_manifest.json \\
        --nifti_dir data/tcia_nifti \\
        --spine_dir data/placed/spine \\
        --out_dir   data/matched
"""

import argparse
import json
import re
import statistics
import sys
from datetime import datetime
from pathlib import Path

import numpy as np


def fmt_mu_sigma(values, fmt=".1f"):
    if not values:
        return "EMPTY", "EMPTY"
    mu = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return f"{mu:{fmt}}", f"{sd:{fmt}}"


def fmt_range(values):
    return f"{min(values):.1f}" if values else "EMPTY"


def read_manifest(path):
    data = json.loads(Path(path).read_text())
    cases = data.get("cases", [])
    if isinstance(cases, dict):
        cases = list(cases.values())
    return data, cases


def nifti_header_stats(nifti_dir, workers=8):
    import json as _json
    nd = Path(nifti_dir)
    if not nd.exists():
        print(f"  [fillsheet] nifti_dir not found: {nd}", flush=True)
        return None

    json_files = sorted(nd.glob("*.json"))
    print(f"  [fillsheet] nifti_header_stats: {len(json_files)} dcm2niix JSONs ...", flush=True)
    inplane, thick, nz_list = [], [], []
    for jf in json_files:
        try:
            d = _json.loads(jf.read_text())
            ps = d.get("PixelSpacing") or d.get("pixdim", [None,None])
            if isinstance(ps, list) and len(ps) >= 1 and ps[0]:
                inplane.append(float(ps[0]))
            st = d.get("SliceThickness") or d.get("slice_thickness")
            if st:
                thick.append(float(st))
            nz = d.get("dcm_AcquisitionMatrix") or d.get("global", {}).get("slices")
            if nz and isinstance(nz, (int, float)):
                nz_list.append(int(nz))
        except Exception:
            pass

    if not inplane:
        print("  [fillsheet] no dcm2niix JSONs found — marking NEEDS_NIFTI", flush=True)
        return None

    print(f"  [fillsheet] nifti done  {len(inplane)} spacing  {len(thick)} thickness  {len(nz_list)} Nz",
          flush=True)
    return {"inplane": inplane, "thick": thick, "nz": nz_list}


def spine_label_stats(spine_dir, workers=8):
    import json as _json
    sd = Path(spine_dir)
    if not sd.exists():
        print(f"  [fillsheet] spine_dir not found: {sd}", flush=True)
        return None
    sidecars = sorted(sd.glob("*_seg_placed.json"))
    print(f"  [fillsheet] spine_label_stats: {len(sidecars)} sidecar JSONs ...", flush=True)
    n_four = n_fail = n_total = 0
    method_counts = {"affine": 0, "xcorr": 0, "nifti_anchor": 0}
    for sc in sidecars:
        try:
            d = _json.loads(sc.read_text())
            labels = d.get("labels") or []
            lumbar = [l for l in labels if l in (20,21,22,23,24,25)]
            if len(lumbar) == 4:
                n_four += 1
            if not d.get("IS_ok", True):
                n_fail += 1
            m = (d.get("method") or "").lower()
            if "xcorr" in m or "phase" in m:
                method_counts["xcorr"] += 1
            elif "anchor" in m or "nifti" in m:
                method_counts["nifti_anchor"] += 1
            else:
                method_counts["affine"] += 1
            n_total += 1
        except Exception:
            pass
    print(f"  [fillsheet] spine done  n={n_total}  4-label={n_four}  IS_fail={n_fail}", flush=True)
    return {"n_four_labels": n_four, "n_is_fail": n_fail, "n_total": n_total,
            "method_counts": method_counts}


def compute_fillsheet(manifest_path, nifti_dir=None, spine_dir=None):
    data, cases = read_manifest(manifest_path)
    vals = {}
    print(f"  [fillsheet] computing stats for {len(cases)} cases ...", flush=True)

    n_cases      = data.get("n_cases",       len(cases))
    n_fused      = data.get("n_fused",       sum(1 for c in cases if c.get("match_type") == "fused"))
    n_separate   = data.get("n_separate",    sum(1 for c in cases if c.get("match_type") == "separate"))
    n_spine_only = data.get("n_spine_only",  sum(1 for c in cases if c.get("match_type") == "spine_only"))
    n_pelv_only  = data.get("n_pelvic_only", sum(1 for c in cases if c.get("match_type") == "pelvic_only"))

    vals["n_dataset"]     = str(n_cases)
    vals["n_fused"]       = str(n_fused)
    vals["n_separate"]    = str(n_separate)
    vals["n_spine_only"]  = str(n_spine_only)
    vals["n_pelvic_only"] = str(n_pelv_only)

    spine_bps = [float(c["spine"]["bone_pct"])
                 for c in cases if c.get("spine") and c["spine"].get("bone_pct") is not None]
    pelv_bps  = [float(c["pelvic"]["bone_pct"])
                 for c in cases if c.get("pelvic") and c["pelvic"].get("bone_pct") is not None]

    if spine_bps:
        mu, sd = fmt_mu_sigma(spine_bps)
        vals["spine_bone_pct_mean"] = mu
        vals["spine_bone_pct_sd"]   = sd
        vals["spine_bone_pct_min"]  = fmt_range(spine_bps)
        vals["n_spine_placed"]      = str(len(spine_bps))
    else:
        for k in ("spine_bone_pct_mean","spine_bone_pct_sd","spine_bone_pct_min","n_spine_placed"):
            vals[k] = "NEEDS_MANIFEST"

    if pelv_bps:
        mu, sd = fmt_mu_sigma(pelv_bps)
        vals["pelv_bone_pct_mean"]  = mu
        vals["pelv_bone_pct_sd"]    = sd
        vals["pelv_bone_pct_min"]   = fmt_range(pelv_bps)
        vals["n_pelv_placed"]       = str(len(pelv_bps))
    else:
        for k in ("pelv_bone_pct_mean","pelv_bone_pct_sd","pelv_bone_pct_min","n_pelv_placed"):
            vals[k] = "NEEDS_MANIFEST"

    spine_methods = [c["spine"].get("method", "") or ""
                     for c in cases if c.get("spine")]
    affine_n = sum(1 for m in spine_methods if "world_space" in m.lower() or m == "")
    xcorr_n  = sum(1 for m in spine_methods if "xcorr" in m.lower() or "phase" in m.lower())
    anchor_n = sum(1 for m in spine_methods if "anchor" in m.lower() or "nifti" in m.lower())
    vals["n_affine_placed"] = str(affine_n) if affine_n else "NEEDS_NIFTI"
    vals["n_xcorr_placed"]  = str(xcorr_n)  if xcorr_n  else "NEEDS_NIFTI"
    vals["n_anchor_placed"] = str(anchor_n) if anchor_n else "NEEDS_NIFTI"

    lstv_labels = [str(c.get("lstv_pelvic", "") or "").lower() for c in cases]
    n_normal  = sum(1 for l in lstv_labels if l == "normal" or l == "")
    n_sacral  = sum(1 for l in lstv_labels if "sacral" in l and "semi" not in l)
    n_lumbar  = sum(1 for l in lstv_labels if "lumbar" in l)
    n_semi    = sum(1 for l in lstv_labels if "semi" in l)
    n_lstv    = n_sacral + n_lumbar + n_semi

    vals["n_lstv_total"]         = str(n_lstv)
    vals["n_sacralization"]      = str(n_sacral)
    vals["n_lumbarization"]      = str(n_lumbar)
    vals["n_semi_sacralization"] = str(n_semi)
    vals["n_normal_cases"]       = str(n_normal)

    vals["n_fused_training"]     = str(n_fused)
    vals["n_spine_native"]       = str(n_fused + n_spine_only)
    vals["n_pelvic_native"]      = str(n_fused + n_pelv_only)

    n_fused_total = n_fused
    vals["n_fused_train"] = str(round(n_fused_total*0.70))
    vals["n_fused_val"]   = str(round(n_fused_total*0.15))
    vals["n_fused_test"]  = str(round(n_fused_total*0.15))

    print("  [fillsheet] reading dcm2niix + placement sidecars ...", flush=True)
    nifti_stats   = nifti_header_stats(nifti_dir) if nifti_dir else None
    _slabs_result = spine_label_stats(spine_dir)  if spine_dir else None

    if nifti_stats and nifti_stats["inplane"]:
        mu, sd = fmt_mu_sigma(nifti_stats["inplane"], fmt=".3f")
        vals["inplane_res_mu_sigma"] = f"{mu} ± {sd}"
        mu, sd = fmt_mu_sigma(nifti_stats["thick"], fmt=".2f")
        vals["slice_thick_mu_sigma"] = f"{mu} ± {sd}"
        mu, sd = fmt_mu_sigma(nifti_stats["nz"], fmt=".0f")
        vals["spine_fov_nz_mu_sigma"] = f"{mu} ± {sd}"
    else:
        vals["inplane_res_mu_sigma"]  = "NEEDS_NIFTI"
        vals["slice_thick_mu_sigma"]  = "NEEDS_NIFTI"
        vals["spine_fov_nz_mu_sigma"] = "NEEDS_NIFTI"

    slabs = _slabs_result
    if slabs:
        vals["n_four_label_masks"]    = str(slabs["n_four_labels"])
        inversion_rate                = slabs["n_is_fail"] / max(slabs["n_total"], 1)
        vals["is_ordering_fail"]      = f"{slabs['n_is_fail']}/{slabs['n_total']}"
        vals["is_ordering_fail_pct"]  = f"{100*inversion_rate:.1f}"
    else:
        vals["n_four_label_masks"]    = "NEEDS_NIFTI"
        vals["is_ordering_fail"]      = "NEEDS_NIFTI"
        vals["is_ordering_fail_pct"]  = "NEEDS_NIFTI"

    vals["dcm2niix_version"] = "MANUAL"
    vals["n_workers"]        = "32"
    vals["wall_clock_hours"] = "MANUAL"
    vals["hardware"]         = "MANUAL"

    return vals


FILLSHEET_TEMPLATE = """\
══════════════════════════════════════════════════════════════════════════════
     CTSpinoPelvic1K  –  Paper Placeholder Fill Sheet
     Generated: {timestamp}
     Source manifest: {manifest_path}
══════════════════════════════════════════════════════════════════════════════

 DATASET SIZE
  n_dataset         : {n_dataset}
  n_fused           : {n_fused}
  n_separate        : {n_separate}
  n_spine_only      : {n_spine_only}
  n_pelvic_only     : {n_pelvic_only}

 SCAN STATS (NEEDS dcm2niix JSONs)
  inplane_res       : {inplane_res_mu_sigma}
  slice_thickness   : {slice_thick_mu_sigma}
  spine_fov_nz      : {spine_fov_nz_mu_sigma}

 PLACEMENT QUALITY (from manifest)
  n_spine_placed    : {n_spine_placed}
  n_pelv_placed     : {n_pelv_placed}
  spine bone%       : {spine_bone_pct_mean} ± {spine_bone_pct_sd}  (min={spine_bone_pct_min})
  pelvic bone%      : {pelv_bone_pct_mean} ± {pelv_bone_pct_sd}   (min={pelv_bone_pct_min})
  method: affine={n_affine_placed}  xcorr={n_xcorr_placed}  anchor={n_anchor_placed}

 SPINE LABEL STATS (from sidecars)
  n_four_label      : {n_four_label_masks}
  IS_ordering_fail  : {is_ordering_fail}  ({is_ordering_fail_pct}%)

 LSTV DISTRIBUTION
  n_normal          : {n_normal_cases}
  n_sacralization   : {n_sacralization}
  n_lumbarization   : {n_lumbarization}
  n_semi            : {n_semi_sacralization}
  n_lstv_total      : {n_lstv_total}

 TRAINING CONFIGS
  fused       : N = {n_fused_training}
  spine_native: N = {n_spine_native}
  pelvic_native: N = {n_pelvic_native}
  split 70/15/15: train/val/test = {n_fused_train} / {n_fused_val} / {n_fused_test}

 IMPLEMENTATION (MANUAL)
  dcm2niix_version  : {dcm2niix_version}
  n_workers         : {n_workers}
  wall_clock_hours  : {wall_clock_hours}
  hardware          : {hardware}

═══════════════════════════════════════════════════════════════════════════════
"""


def build_fillsheet_text(vals, manifest_path):
    ts = datetime.now().strftime("%Y-%m-%d %H:%M")
    return FILLSHEET_TEMPLATE.format(
        timestamp=ts,
        manifest_path=str(manifest_path),
        **vals,
    )


def main():
    ap = argparse.ArgumentParser(
        description="Generate paper placeholder fill sheet from placed_manifest.json"
    )
    ap.add_argument("--manifest",   required=True)
    ap.add_argument("--out_dir",    required=True)
    ap.add_argument("--nifti_dir",  default="", help="tcia_nifti/ dir (optional)")
    ap.add_argument("--spine_dir",  default="", help="placed/spine/ dir (optional)")
    args = ap.parse_args()

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        sys.exit(1)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n generate_fillsheet.py")
    print(f"   manifest : {manifest_path}")

    vals = compute_fillsheet(
        manifest_path,
        nifti_dir = args.nifti_dir or None,
        spine_dir = args.spine_dir or None,
    )

    txt_path  = out_dir / "paper_fillsheet.txt"
    json_path = out_dir / "paper_fillsheet.json"

    txt = build_fillsheet_text(vals, manifest_path)
    txt_path.write_text(txt)
    print(f"\n   Fill sheet : {txt_path}")

    json_path.write_text(json.dumps(vals, indent=2))
    print(f"   JSON dict  : {json_path}")

    needs = sum(1 for v in vals.values() if "NEEDS" in str(v) or "MANUAL" in str(v))
    ready = len(vals) - needs
    print(f"\n   {ready}/{len(vals)} values ready from manifest")
    print(f"   {needs} still need NIfTI sidecars or manual input\n")


if __name__ == "__main__":
    main()
