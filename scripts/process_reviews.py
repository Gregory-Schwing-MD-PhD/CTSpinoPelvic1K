#!/usr/bin/env python3
"""
process_reviews.py

Server-side ingestion and classification of radiologist-completed
LSTV review folders.

Inputs (per token directory):
    landmarks.mrk.json   -- 3D Slicer fiducial output (LPS, mm)
    review.txt           -- categorical confirmation + free-text notes
    spine_mask.nii.gz    -- to determine lumbar body count

Derives objectively from landmark placements:

    1. Sacral segment count   (from placed ventral foramina S1-S5)
    2. Castellvi grade        (per-side Type I / II / III, combined to
                                Ia, Ib, IIa, IIb, IIIa, IIIb, or IV)
    3. Mahato spectrum grade  (lumbarization B/C/D or sacralization
                                B/C/D/E from Fig. 1 / Fig. 2)
    4. Mahato morphometrics   (SH, IFD, IAD, BW, AW, AH bilateral,
                                S1T bilateral, S2T bilateral, S1O
                                bilateral, AS position type)
    5. Spinopelvic parameters (sacral slope, pelvic tilt, pelvic
                                incidence, L1-S1 lumbar lordosis;
                                each computed only if its required
                                landmarks are placed)

Outputs:
    reviews.json            -- consolidated per-case structured data
    reviews_summary.csv     -- flat per-case table for analysis
    validation_report.txt   -- per-case warnings and errors

Usage:
    python process_reviews.py --in_dir <path-to-lstv_review>
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

try:
    import nibabel as nib
except ImportError:
    nib = None  # spine_mask reading optional


# ===========================================================================
# Constants
# ===========================================================================
SUBTYPE_DIRS = ["lumb", "sacr_count", "semisacralization",
                "sacralization", "ambiguous"]

CASTELLVI_TP_DYSPLASIA_THRESHOLD_MM = 19.0  # Castellvi 1984 Type I criterion

# Anatomic plane normals in LPS (Slicer default)
SAGITTAL_NORMAL = np.array([1.0, 0.0, 0.0])  # plane separating L/R
CORONAL_NORMAL = np.array([0.0, 1.0, 0.0])
AXIAL_NORMAL = np.array([0.0, 0.0, 1.0])


# ===========================================================================
# I/O
# ===========================================================================
def load_landmarks(path: Path) -> dict:
    """Load Slicer markup file. Returns {label: np.array([x,y,z])}
    only for control points with positionStatus != 'preview'."""
    with open(path) as f:
        data = json.load(f)
    out = {}
    for markup in data.get("markups", []):
        coord_sys = markup.get("coordinateSystem", "LPS")
        for cp in markup.get("controlPoints", []):
            if cp.get("positionStatus") == "preview":
                continue
            pos = cp.get("position")
            label = cp.get("label")
            if not pos or len(pos) != 3 or not label:
                continue
            arr = np.array(pos, dtype=float)
            if coord_sys == "RAS":
                arr = np.array([-arr[0], -arr[1], arr[2]])
            out[label] = arr
    return out


def parse_review_txt(path: Path) -> dict:
    out = {}
    for line in path.read_text().splitlines():
        s = line.split("#", 1)[0].strip()
        if not s or ":" not in s:
            continue
        key, _, val = s.partition(":")
        if val.strip():
            out[key.strip()] = val.strip()
    return out


def count_lumbar_bodies_from_mask(mask_path: Path) -> int:
    """Count distinct lumbar labels (1-6) in a label NIfTI."""
    if nib is None or not mask_path.exists():
        return None
    img = nib.load(str(mask_path))
    arr = np.asarray(img.dataobj).astype(np.int16)
    lumbar_classes = [v for v in range(1, 7) if (arr == v).any()]
    return len(lumbar_classes)


# ===========================================================================
# Geometry helpers
# ===========================================================================
def dist(a, b):
    if a is None or b is None:
        return None
    return float(np.linalg.norm(a - b))


def midpoint(a, b):
    if a is None or b is None:
        return None
    return (a + b) / 2.0


def project_to_plane(v, normal):
    n = normal / np.linalg.norm(normal)
    return v - np.dot(v, n) * n


def angle_between_3d(v1, v2):
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return None
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)))


def angle_in_plane(v1, v2, plane_normal):
    p1 = project_to_plane(v1, plane_normal)
    p2 = project_to_plane(v2, plane_normal)
    return angle_between_3d(p1, p2)


def craniocaudal_dist(a, b):
    """Distance along the z (superior-inferior) axis only."""
    if a is None or b is None:
        return None
    return float(abs(a[2] - b[2]))


# ===========================================================================
# Sacral segment count (from foramina)
# ===========================================================================
def count_sacral_segments(L: dict) -> dict:
    """Returns dict with per-side foramen counts and segment count."""
    placed_left = sum(1 for i in range(1, 6)
                      if f"ventral_foramen_S{i}_left" in L)
    placed_right = sum(1 for i in range(1, 6)
                       if f"ventral_foramen_S{i}_right" in L)
    # Use the more conservative (smaller) count for the segment estimate
    n_pairs = min(placed_left, placed_right) if (placed_left and placed_right) \
        else max(placed_left, placed_right)
    n_segments = n_pairs + 1 if n_pairs > 0 else None
    return {
        "n_foramen_pairs_left": placed_left,
        "n_foramen_pairs_right": placed_right,
        "asymmetric_foramen_count": placed_left != placed_right,
        "n_sacral_segments": n_segments,
    }


# ===========================================================================
# Castellvi classification
# ===========================================================================
def classify_castellvi_per_side(L: dict, side: str) -> dict:
    """For one side ('left' or 'right'), determine Castellvi type."""
    fusion = f"castellvi_fusion_bridge_{side}" in L
    articulation = f"castellvi_articulation_{side}" in L

    tp_cranial = L.get(f"transitional_TP_cranial_{side}")
    tp_caudal = L.get(f"transitional_TP_caudal_{side}")
    tp_length = dist(tp_cranial, tp_caudal)

    grade = None
    if fusion:
        grade = "III"
    elif articulation:
        grade = "II"
    elif tp_length is not None and tp_length >= CASTELLVI_TP_DYSPLASIA_THRESHOLD_MM:
        grade = "I"

    return {
        "grade": grade,
        "tp_length_mm": tp_length,
        "articulation": articulation,
        "fusion": fusion,
    }


def combine_castellvi(left: dict, right: dict) -> str:
    """Combine per-side Castellvi grades into the overall classification."""
    L = left.get("grade")
    R = right.get("grade")

    # Type IV: II on one side + III on contralateral
    if {L, R} == {"II", "III"}:
        return "IV"

    # Bilateral identical
    if L is not None and L == R:
        return f"{L}b"

    # Unilateral: exactly one side typed
    if L and not R:
        return f"{L}a"
    if R and not L:
        return f"{R}a"

    # Atypical mixed (both typed but not the IV combination)
    if L and R and L != R:
        return f"atypical:{L}_left/{R}_right"

    # Neither side typed
    return None


# ===========================================================================
# Mahato spectrum grade
# ===========================================================================
def classify_mahato_spectrum(L: dict, n_lumbar: int, n_sacral_segments: int) -> dict:
    """Classify Mahato (2020) spectrum grade from observations."""
    aa_left = "castellvi_articulation_left" in L
    aa_right = "castellvi_articulation_right" in L
    fusion_left = "castellvi_fusion_bridge_left" in L
    fusion_right = "castellvi_fusion_bridge_right" in L
    any_aa = aa_left or aa_right
    bilateral_aa = aa_left and aa_right
    any_fusion = fusion_left or fusion_right
    bilateral_fusion = fusion_left and fusion_right
    unilateral_fusion = any_fusion and not bilateral_fusion

    family = "indeterminate"
    grade = None
    rationale = ""

    # Normal anatomy
    if (n_lumbar == 5 and n_sacral_segments == 5
            and not any_aa and not any_fusion):
        family = "normal"
        grade = "A"
        rationale = "5 lumbar bodies, 5-segment sacrum, no articulation, no fusion"

    # Lumbarization spectrum
    elif n_lumbar == 6:
        family = "lumbarization"
        if n_sacral_segments == 4:
            if bilateral_aa and not any_fusion:
                grade = "C"
                rationale = "6 lumbar bodies, 4-segment sacrum, bilateral L6-S1 accessory articulation"
            elif not any_aa and not any_fusion:
                grade = "D"
                rationale = "6 lumbar bodies, 4-segment sacrum, no accessory articulation"
            else:
                grade = None
                rationale = (f"6 lumbar bodies, 4-segment sacrum, but unilateral or "
                             f"mixed articulation/fusion pattern (aa_l={aa_left}, "
                             f"aa_r={aa_right}, fus_l={fusion_left}, fus_r={fusion_right})")
        elif n_sacral_segments == 5:
            grade = "B"
            rationale = "6 lumbar bodies, 5-segment sacrum (incomplete lumbarization with partial S1-S2 separation)"
        else:
            rationale = f"6 lumbar bodies but unexpected sacral segment count {n_sacral_segments}"

    # Sacralization spectrum
    elif n_lumbar == 4 or (n_lumbar == 5 and any_aa) or (n_lumbar == 4 and n_sacral_segments == 6):
        if n_sacral_segments == 6:
            family = "sacralization"
            if bilateral_fusion:
                grade = "E"
                rationale = "4 lumbar bodies, 6-segment sacrum, bilateral L5-S1 fusion (complete sacralization)"
            elif unilateral_fusion:
                grade = "D"
                rationale = "4 lumbar bodies, 6-segment sacrum, unilateral fusion (incomplete sacralization)"
            else:
                rationale = "6-segment sacrum but no fusion observed; unexpected"
        elif n_sacral_segments == 5 and n_lumbar == 5:
            family = "sacralization"
            if bilateral_aa:
                grade = "C"
                rationale = "5 lumbar bodies, 5-segment sacrum, bilateral L5-S1 accessory articulation"
            elif any_aa:
                grade = "B"
                rationale = "5 lumbar bodies, 5-segment sacrum, unilateral L5-S1 accessory articulation"
        else:
            rationale = (f"4 lumbar bodies but unexpected sacral segment count "
                         f"{n_sacral_segments}")

    else:
        rationale = (f"Combination not in standard spectrum: n_lumbar={n_lumbar}, "
                     f"n_sacral_segments={n_sacral_segments}, "
                     f"any_aa={any_aa}, any_fusion={any_fusion}")

    return {
        "family": family,
        "grade": grade,
        "rationale": rationale,
        "articulation_left": aa_left,
        "articulation_right": aa_right,
        "fusion_left": fusion_left,
        "fusion_right": fusion_right,
    }


# ===========================================================================
# Mahato morphometrics
# ===========================================================================
def compute_mahato_morphometrics(L: dict) -> dict:
    g = L.get
    promontory = g("sacral_promontory_midpoint")
    base = g("sacral_base_midpoint")
    s2_l = g("ventral_foramen_S2_left")
    s2_r = g("ventral_foramen_S2_right")

    SH = dist(promontory, base)
    IFD = dist(g("S1_facet_dorsal_left"), g("S1_facet_dorsal_right"))
    IAD = dist(g("auricular_surface_upper_left"),
               g("auricular_surface_upper_right"))
    BW = dist(g("ventral_foramen_S1_left"), g("ventral_foramen_S1_right"))
    AW = (IAD - BW) if (IAD is not None and BW is not None) else None
    AH_L = dist(g("auricular_surface_upper_left"),
                g("auricular_surface_lower_left"))
    AH_R = dist(g("auricular_surface_upper_right"),
                g("auricular_surface_lower_right"))

    # Thickness: craniocaudal (z-axis) distance between consecutive
    # ventral foramina midpoints
    S1T_L = craniocaudal_dist(g("ventral_foramen_S1_left"),
                              g("ventral_foramen_S2_left"))
    S1T_R = craniocaudal_dist(g("ventral_foramen_S1_right"),
                              g("ventral_foramen_S2_right"))
    S2T_L = craniocaudal_dist(g("ventral_foramen_S2_left"),
                              g("ventral_foramen_S3_left"))
    S2T_R = craniocaudal_dist(g("ventral_foramen_S2_right"),
                              g("ventral_foramen_S3_right"))

    # S1O: angle in coronal plane between alar vector (promontory -> AS
    # upper) and horizontal reference (left-right axis at S2 level)
    horizontal = None
    if s2_l is not None and s2_r is not None:
        horizontal = s2_r - s2_l

    def s1_obliquity(side):
        if promontory is None or horizontal is None:
            return None
        as_upper = g(f"auricular_surface_upper_{side}")
        if as_upper is None:
            return None
        v_alar = as_upper - promontory
        return angle_in_plane(v_alar, horizontal, CORONAL_NORMAL)

    S1O_L = s1_obliquity("left")
    S1O_R = s1_obliquity("right")

    # AS position type (Mahato 2010): I / II / III based on z-position
    # of AS upper border relative to S1 segment z-extent
    as_pos = None
    if (promontory is not None and s2_l is not None and s2_r is not None
            and (g("auricular_surface_upper_left") is not None
                 or g("auricular_surface_upper_right") is not None)):
        s1_top_z = promontory[2]
        s1_bottom_z = (s2_l[2] + s2_r[2]) / 2.0
        s1_mid_z = (s1_top_z + s1_bottom_z) / 2.0
        as_zs = [v[2] for v in (g("auricular_surface_upper_left"),
                                g("auricular_surface_upper_right"))
                 if v is not None]
        as_z = float(np.mean(as_zs))
        if as_z > s1_top_z:
            as_pos = "II"   # high
        elif as_z >= s1_mid_z:
            as_pos = "I"    # normal
        else:
            as_pos = "III"  # low

    return {
        "SH_mm": SH,
        "IFD_mm": IFD,
        "IAD_mm": IAD,
        "BW_mm": BW,
        "AW_mm": AW,
        "AH_left_mm": AH_L,
        "AH_right_mm": AH_R,
        "S1T_left_mm": S1T_L,
        "S1T_right_mm": S1T_R,
        "S2T_left_mm": S2T_L,
        "S2T_right_mm": S2T_R,
        "S1O_left_deg": S1O_L,
        "S1O_right_deg": S1O_R,
        "AS_position_type": as_pos,
    }


# ===========================================================================
# Spinopelvic
# ===========================================================================
def compute_spinopelvic(L: dict) -> dict:
    g = L.get
    s1_ant = g("sacral_promontory_midpoint")
    s1_post = g("S1_endplate_posterior_midline")
    fh_l = g("femoral_head_center_left")
    fh_r = g("femoral_head_center_right")
    l1_ant = g("L1_endplate_anterior_midline")
    l1_post = g("L1_endplate_posterior_midline")

    out = {
        "sacral_slope_deg": None,
        "pelvic_tilt_deg": None,
        "pelvic_incidence_deg": None,
        "lumbar_lordosis_L1S1_deg": None,
    }

    # Sacral slope: angle between S1 endplate and horizontal in
    # sagittal plane
    if s1_ant is not None and s1_post is not None:
        endplate_v = s1_post - s1_ant
        out["sacral_slope_deg"] = angle_in_plane(
            endplate_v, np.array([0.0, 1.0, 0.0]), SAGITTAL_NORMAL)

    fh_mid = midpoint(fh_l, fh_r)
    s1_mid = midpoint(s1_ant, s1_post)

    # Pelvic tilt
    if s1_mid is not None and fh_mid is not None:
        v = s1_mid - fh_mid
        out["pelvic_tilt_deg"] = angle_in_plane(
            v, np.array([0.0, 0.0, 1.0]), SAGITTAL_NORMAL)

    # Pelvic incidence
    if (s1_ant is not None and s1_post is not None and fh_mid is not None
            and s1_mid is not None):
        endplate_v = s1_post - s1_ant
        endplate_sag = project_to_plane(endplate_v, SAGITTAL_NORMAL)
        # 90-deg rotation of endplate_sag in y-z plane
        perp_sag = np.array([0.0, -endplate_sag[2], endplate_sag[1]])
        v_to_fh = s1_mid - fh_mid
        out["pelvic_incidence_deg"] = angle_in_plane(
            perp_sag, v_to_fh, SAGITTAL_NORMAL)

    # L1-S1 lumbar lordosis
    if (l1_ant is not None and l1_post is not None and s1_ant is not None
            and s1_post is not None):
        l1_v = l1_post - l1_ant
        s1_v = s1_post - s1_ant
        out["lumbar_lordosis_L1S1_deg"] = angle_in_plane(
            l1_v, s1_v, SAGITTAL_NORMAL)

    return out


# ===========================================================================
# Per-token processing
# ===========================================================================
def process_token(token_dir: Path, programmatic_subtype: str) -> dict:
    rec = {
        "token": token_dir.name.replace("token_", ""),
        "programmatic_subtype": programmatic_subtype,
        "directory": str(token_dir),
        "status": "ok",
        "issues": [],
        "review": {},
        "n_lumbar_bodies": None,
        "sacral_segments": None,
        "castellvi": None,
        "mahato_spectrum": None,
        "morphometrics": None,
        "spinopelvic": None,
        "n_landmarks_placed": 0,
    }

    # Lumbar count from segmentation label. In fused mode there's a
    # single label.nii.gz; in separate mode there's label_spine.nii.gz
    # (and label_pelvic.nii.gz). Take max across whatever's present.
    label_candidates = sorted(token_dir.glob("label*.nii.gz"))
    if label_candidates:
        max_lumbar = 0
        for lp in label_candidates:
            n = count_lumbar_bodies_from_mask(lp)
            if n is not None and n > max_lumbar:
                max_lumbar = n
        rec["n_lumbar_bodies"] = max_lumbar
    else:
        # Backward compat: old packages had separate spine_mask.nii.gz
        legacy = token_dir / "spine_mask.nii.gz"
        if legacy.exists():
            rec["n_lumbar_bodies"] = count_lumbar_bodies_from_mask(legacy)

    # review.txt
    review_path = token_dir / "review.txt"
    if review_path.exists():
        rec["review"] = parse_review_txt(review_path)
        if not rec["review"].get("Reviewer name"):
            rec["issues"].append("review.txt: Reviewer name missing")
        if not rec["review"].get("Confirmed"):
            rec["issues"].append("review.txt: Confirmed field missing")
    else:
        rec["status"] = "no_review"
        rec["issues"].append("review.txt missing")

    # landmarks
    lm_path = token_dir / "landmarks.mrk.json"
    if not lm_path.exists():
        rec["status"] = "no_landmarks" if rec["status"] == "ok" else rec["status"]
        rec["issues"].append("landmarks.mrk.json missing")
        return rec

    try:
        L = load_landmarks(lm_path)
    except Exception as e:
        rec["status"] = "landmark_parse_error"
        rec["issues"].append(f"failed to parse landmarks: {e}")
        return rec

    rec["n_landmarks_placed"] = len(L)

    # Sacral segment count
    seg = count_sacral_segments(L)
    rec["sacral_segments"] = seg
    if seg["asymmetric_foramen_count"]:
        rec["issues"].append(
            f"asymmetric foramen count: left={seg['n_foramen_pairs_left']}, "
            f"right={seg['n_foramen_pairs_right']} (using min for segment count)")

    # Castellvi
    left = classify_castellvi_per_side(L, "left")
    right = classify_castellvi_per_side(L, "right")
    combined = combine_castellvi(left, right)
    rec["castellvi"] = {
        "left": left,
        "right": right,
        "combined": combined,
    }

    # Mahato spectrum (requires lumbar count + segment count)
    if rec["n_lumbar_bodies"] is None:
        rec["issues"].append("cannot determine lumbar count (spine mask missing)")
    elif seg["n_sacral_segments"] is None:
        rec["issues"].append("cannot determine sacral segments (no foramina placed)")
    else:
        rec["mahato_spectrum"] = classify_mahato_spectrum(
            L, rec["n_lumbar_bodies"], seg["n_sacral_segments"])
        if rec["mahato_spectrum"]["grade"] is None:
            rec["issues"].append(
                f"Mahato spectrum indeterminate: {rec['mahato_spectrum']['rationale']}")

    # Morphometrics
    try:
        rec["morphometrics"] = compute_mahato_morphometrics(L)
    except Exception as e:
        rec["issues"].append(f"morphometrics compute error: {e}")

    # Spinopelvic
    try:
        rec["spinopelvic"] = compute_spinopelvic(L)
    except Exception as e:
        rec["issues"].append(f"spinopelvic compute error: {e}")

    if rec["issues"] and rec["status"] == "ok":
        rec["status"] = "ok_with_issues"

    return rec


# ===========================================================================
# Output writers
# ===========================================================================
def write_summary_csv(records, csv_path):
    fieldnames = [
        "token", "programmatic_subtype", "status",
        "reviewer", "confirmed", "proposed_subtype",
        "n_lumbar_bodies", "n_foramen_pairs_left", "n_foramen_pairs_right",
        "n_sacral_segments",
        "castellvi_left", "castellvi_right", "castellvi_combined",
        "tp_length_left_mm", "tp_length_right_mm",
        "articulation_left", "articulation_right",
        "fusion_left", "fusion_right",
        "mahato_family", "mahato_grade", "mahato_rationale",
        "SH_mm", "IFD_mm", "IAD_mm", "BW_mm", "AW_mm",
        "AH_left_mm", "AH_right_mm",
        "S1T_left_mm", "S1T_right_mm", "S2T_left_mm", "S2T_right_mm",
        "S1O_left_deg", "S1O_right_deg", "AS_position_type",
        "sacral_slope_deg", "pelvic_tilt_deg",
        "pelvic_incidence_deg", "lumbar_lordosis_L1S1_deg",
        "n_landmarks_placed", "n_issues", "issues",
    ]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for rec in records:
            review = rec.get("review", {}) or {}
            seg = rec.get("sacral_segments", {}) or {}
            cas = rec.get("castellvi", {}) or {}
            cas_l = cas.get("left", {}) or {}
            cas_r = cas.get("right", {}) or {}
            mah = rec.get("mahato_spectrum", {}) or {}
            morph = rec.get("morphometrics", {}) or {}
            sp = rec.get("spinopelvic", {}) or {}
            row = {
                "token": rec["token"],
                "programmatic_subtype": rec["programmatic_subtype"],
                "status": rec["status"],
                "reviewer": review.get("Reviewer name", ""),
                "confirmed": review.get("Confirmed", ""),
                "proposed_subtype": review.get("Proposed subtype", ""),
                "n_lumbar_bodies": rec.get("n_lumbar_bodies"),
                "n_foramen_pairs_left": seg.get("n_foramen_pairs_left"),
                "n_foramen_pairs_right": seg.get("n_foramen_pairs_right"),
                "n_sacral_segments": seg.get("n_sacral_segments"),
                "castellvi_left": cas_l.get("grade"),
                "castellvi_right": cas_r.get("grade"),
                "castellvi_combined": cas.get("combined"),
                "tp_length_left_mm": cas_l.get("tp_length_mm"),
                "tp_length_right_mm": cas_r.get("tp_length_mm"),
                "articulation_left": cas_l.get("articulation"),
                "articulation_right": cas_r.get("articulation"),
                "fusion_left": cas_l.get("fusion"),
                "fusion_right": cas_r.get("fusion"),
                "mahato_family": mah.get("family"),
                "mahato_grade": mah.get("grade"),
                "mahato_rationale": mah.get("rationale"),
                "SH_mm": morph.get("SH_mm"),
                "IFD_mm": morph.get("IFD_mm"),
                "IAD_mm": morph.get("IAD_mm"),
                "BW_mm": morph.get("BW_mm"),
                "AW_mm": morph.get("AW_mm"),
                "AH_left_mm": morph.get("AH_left_mm"),
                "AH_right_mm": morph.get("AH_right_mm"),
                "S1T_left_mm": morph.get("S1T_left_mm"),
                "S1T_right_mm": morph.get("S1T_right_mm"),
                "S2T_left_mm": morph.get("S2T_left_mm"),
                "S2T_right_mm": morph.get("S2T_right_mm"),
                "S1O_left_deg": morph.get("S1O_left_deg"),
                "S1O_right_deg": morph.get("S1O_right_deg"),
                "AS_position_type": morph.get("AS_position_type"),
                "sacral_slope_deg": sp.get("sacral_slope_deg"),
                "pelvic_tilt_deg": sp.get("pelvic_tilt_deg"),
                "pelvic_incidence_deg": sp.get("pelvic_incidence_deg"),
                "lumbar_lordosis_L1S1_deg": sp.get("lumbar_lordosis_L1S1_deg"),
                "n_landmarks_placed": rec.get("n_landmarks_placed"),
                "n_issues": len(rec.get("issues", [])),
                "issues": "; ".join(rec.get("issues", [])),
            }
            w.writerow(row)


def write_validation_report(records, path):
    n_clean = sum(1 for r in records if not r["issues"])
    n_total_issues = sum(len(r["issues"]) for r in records)
    with open(path, "w") as f:
        f.write("LSTV review validation report\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Total records:       {len(records)}\n")
        f.write(f"Clean (no issues):   {n_clean}\n")
        f.write(f"With issues:         {len(records) - n_clean}\n")
        f.write(f"Total issue count:   {n_total_issues}\n\n")

        f.write("Per-case detail:\n")
        f.write("-" * 60 + "\n")
        for rec in records:
            f.write(f"\n[{rec['programmatic_subtype']}/token_{rec['token']}] "
                    f"status={rec['status']}\n")
            f.write(f"  reviewer:  {rec.get('review', {}).get('Reviewer name', '')}\n")
            f.write(f"  confirmed: {rec.get('review', {}).get('Confirmed', '')}\n")
            cas = (rec.get("castellvi") or {}).get("combined")
            mah = (rec.get("mahato_spectrum") or {})
            f.write(f"  derived:   Castellvi={cas}, "
                    f"Mahato={mah.get('family')}-{mah.get('grade')}\n")
            if rec["issues"]:
                f.write("  issues:\n")
                for issue in rec["issues"]:
                    f.write(f"    - {issue}\n")


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True,
                    help="Path to lstv_review/ directory")
    ap.add_argument("--out_dir", default=None,
                    help="Output directory (default: <in_dir>/_processed)")
    args = ap.parse_args()

    in_dir = Path(args.in_dir).expanduser().resolve()
    if not in_dir.exists():
        sys.exit(f"ERROR: {in_dir} does not exist")
    out_dir = (Path(args.out_dir).expanduser().resolve()
               if args.out_dir else in_dir / "_processed")
    out_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for sd in SUBTYPE_DIRS:
        subtype_path = in_dir / sd
        if not subtype_path.is_dir():
            continue
        for token_dir in sorted(subtype_path.iterdir()):
            if not token_dir.is_dir() or not token_dir.name.startswith("token_"):
                continue
            rec = process_token(token_dir, programmatic_subtype=sd)
            records.append(rec)

    # JSON
    json_path = out_dir / "reviews.json"
    with open(json_path, "w") as f:
        # numpy array types serialized as lists
        def _default(o):
            if isinstance(o, np.ndarray):
                return o.tolist()
            if isinstance(o, (np.float64, np.float32)):
                return float(o)
            if isinstance(o, (np.int64, np.int32)):
                return int(o)
            if isinstance(o, np.bool_):
                return bool(o)
            return str(o)
        json.dump({"records": records}, f, indent=2, default=_default)
    print(f"Wrote {json_path} ({len(records)} records)")

    # CSV
    csv_path = out_dir / "reviews_summary.csv"
    write_summary_csv(records, csv_path)
    print(f"Wrote {csv_path}")

    # Validation report
    report_path = out_dir / "validation_report.txt"
    write_validation_report(records, report_path)
    print(f"Wrote {report_path}")

    # Console summary
    n_clean = sum(1 for r in records if not r["issues"])
    print(f"\nSummary: {n_clean}/{len(records)} clean records, "
          f"{sum(len(r['issues']) for r in records)} total issues.")
    derived = [(r['programmatic_subtype'], r['token'],
                (r.get('castellvi') or {}).get('combined'),
                (r.get('mahato_spectrum') or {}).get('family'),
                (r.get('mahato_spectrum') or {}).get('grade'))
               for r in records]
    print("\nDerived classifications:")
    print(f"  {'subtype':<22} {'token':<8} {'Castellvi':<14} {'Mahato':<14} {'grade'}")
    for subtype, tok, cas, fam, grade in derived:
        print(f"  {subtype:<22} {tok:<8} {str(cas):<14} {str(fam):<14} {str(grade)}")


if __name__ == "__main__":
    main()
