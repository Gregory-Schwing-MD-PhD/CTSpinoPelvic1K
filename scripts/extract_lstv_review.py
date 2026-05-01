#!/usr/bin/env python3
"""
extract_lstv_review.py

DELIVERY OBJECTIVE
==================
The CTSpinoPelvic1K dataset and the nnU-Net model trained from it are
intended for direct deployment as 3D Slicer's default spinal-CT
segmentation backend, replacing TotalSegmentator's spinopelvic
output. TotalSegmentator's documented LSTV failure mode (paper
accompanying this dataset) propagates from Slicer-based clinical
workflows directly to spine surgical planning, where it reproduces
the canonical mechanism of wrong-level surgery. Closing the LSTV
gap at the level of Slicer's default tool is the deliverable.

Radiologist annotations gathered through this review package
provide:
    1. Verified subtype labels for the dataset paper
    2. Castellvi/Mahato graded LSTV cases for evaluation
    3. Landmark-supervised training signal for an automated
       spinopelvic-parameter measurement model that ships
       alongside the segmentation backend in the Slicer extension

PACKAGE CONTENTS
================
Extracts LSTV-positive cases from the CTSpinoPelvic1K HF export
for radiologist annotation. Each token directory is provisioned
with the CT, segmentation masks, a 3D Slicer fiducial template,
and a brief review form.

The annotation protocol uses 31 standardized anatomical landmarks
designed to permit deterministic derivation of:

    1. Castellvi (1984) classification of the lumbosacral junction,
       including unilateral / bilateral specification (Ia, Ib, IIa,
       IIb, IIIa, IIIb, IV).
    2. Mahato (2020) lumbosacral transitional spectrum grade
       (lumbarization B/C/D from Fig. 1; sacralization B/C/D/E from
       Fig. 2).
    3. Mahato (2020) sacral morphometric measurements (linear,
       angular, and craniocaudal-thickness parameters).
    4. Standard spinopelvic parameters (sacral slope, pelvic tilt,
       pelvic incidence, L1-S1 lumbar lordosis), conditional on
       optional landmark placement.

Classification is performed downstream by process_reviews.py from the
landmark coordinates only; the radiologist does not assign categorical
grades. Their expert observation is encoded in the placement (or
deliberate non-placement) of the landmarks.
"""

import argparse
import json
import logging
import shutil
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging: line-buffered stdout, ISO-style timestamps, INFO by default.
# Configured at module load so progress lines appear in real time even when
# stdout is piped (e.g. into `tee` or a SLURM stdout file).
# ---------------------------------------------------------------------------
try:
    sys.stdout.reconfigure(line_buffering=True)
except (AttributeError, ValueError):
    pass  # older Python or non-tty stdout

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout,
)
log = logging.getLogger("extract_lstv_review")

LSTV_SUBTYPES = ["lumb", "sacr_count", "semisacralization",
                 "sacralization", "ambiguous"]
SPINE_CLASSES = list(range(1, 7))
PELVIC_CLASSES = [7, 8, 9]


# ===========================================================================
# Landmark catalogue (31 total)
# ===========================================================================
# Tuple format: (label, description, group)
#
# Groups:
#   "castellvi_observation"  -- place ONLY IF the named structure is
#                               observed; leave unplaced (preview) if
#                               absent.
#   "castellvi_measurement"  -- always place if anatomy is in field of
#                               view; used for Castellvi Type I
#                               (dysplasia) determination.
#   "sacral_foramen"         -- place every ventral foramen visible.
#                               Number of placed pairs is the algorithm's
#                               read of sacral segment count: 3 -> 4-seg,
#                               4 -> 5-seg, 5 -> 6-seg.
#   "mahato_anatomical"      -- always place if visible; required for
#                               Mahato (2020) morphometric measurements.
#   "spinopelvic_optional"   -- optional bonus landmarks for sacral
#                               slope, pelvic tilt, pelvic incidence,
#                               and L1-S1 lumbar lordosis.

LANDMARKS = [
    # ----- Castellvi observation (n=4, place IF present) -----
    ("castellvi_articulation_left",
     "[OBSERVATION] Pseudoarticulation between the LEFT transverse "
     "process of the lumbosacral transitional vertebra (L5 in "
     "sacralization, L6 in lumbarization) and the sacral ala. "
     "Place at the geometric center of the diarthrodial joint surface. "
     "Leave unplaced if no articulation is present on the left side.",
     "castellvi_observation"),
    ("castellvi_articulation_right",
     "[OBSERVATION] Same as castellvi_articulation_left, RIGHT side.",
     "castellvi_observation"),
    ("castellvi_fusion_bridge_left",
     "[OBSERVATION] Osseous fusion bridge between the LEFT transitional-"
     "vertebra transverse process and the sacral ala (Castellvi Type "
     "III). Place at the midpoint of the cortical bridge. Leave "
     "unplaced if no fusion is present on the left side.",
     "castellvi_observation"),
    ("castellvi_fusion_bridge_right",
     "[OBSERVATION] Same as castellvi_fusion_bridge_left, RIGHT side.",
     "castellvi_observation"),

    # ----- Castellvi measurement (n=4, place if anatomy in FOV) -----
    ("transitional_TP_cranial_left",
     "[MEASUREMENT] Most superior point of the LEFT transverse process "
     "of the transitional vertebra at its broadest craniocaudal "
     "dimension (typically the proximal/medial portion where it joins "
     "the vertebral body). Used to compute craniocaudal TP height for "
     "Castellvi Type I (>=19 mm dysplasia).",
     "castellvi_measurement"),
    ("transitional_TP_cranial_right",
     "[MEASUREMENT] Same as transitional_TP_cranial_left, RIGHT side.",
     "castellvi_measurement"),
    ("transitional_TP_caudal_left",
     "[MEASUREMENT] Most inferior point of the LEFT transitional-"
     "vertebra transverse process at the same axial level used for "
     "transitional_TP_cranial_left.",
     "castellvi_measurement"),
    ("transitional_TP_caudal_right",
     "[MEASUREMENT] Same as transitional_TP_caudal_left, RIGHT side.",
     "castellvi_measurement"),

    # ----- Sacral foramina (n=10, place all visible) -----
    ("ventral_foramen_S1_left",
     "[FORAMEN] Midpoint of the LEFT 1st ventral (anterior) sacral "
     "foramen, between sacral segments S1 and S2. Always present in "
     "any sacrum with >=2 segments.",
     "sacral_foramen"),
    ("ventral_foramen_S1_right",
     "[FORAMEN] Same on the RIGHT side.",
     "sacral_foramen"),
    ("ventral_foramen_S2_left",
     "[FORAMEN] Midpoint of the LEFT 2nd ventral sacral foramen, "
     "between segments S2 and S3.",
     "sacral_foramen"),
    ("ventral_foramen_S2_right",
     "[FORAMEN] Same on the RIGHT side.",
     "sacral_foramen"),
    ("ventral_foramen_S3_left",
     "[FORAMEN] Midpoint of the LEFT 3rd ventral sacral foramen, "
     "between segments S3 and S4.",
     "sacral_foramen"),
    ("ventral_foramen_S3_right",
     "[FORAMEN] Same on the RIGHT side.",
     "sacral_foramen"),
    ("ventral_foramen_S4_left",
     "[FORAMEN] Midpoint of the LEFT 4th ventral sacral foramen, "
     "between segments S4 and S5. PRESENT in 5-segment (normal) and "
     "6-segment (sacralization) sacra; ABSENT in 4-segment "
     "(lumbarization) sacra. Leave unplaced if absent.",
     "sacral_foramen"),
    ("ventral_foramen_S4_right",
     "[FORAMEN] Same on the RIGHT side.",
     "sacral_foramen"),
    ("ventral_foramen_S5_left",
     "[FORAMEN] Midpoint of the LEFT 5th ventral sacral foramen, "
     "between segments S5 and S6. PRESENT only in 6-segment sacra "
     "(complete or incomplete sacralization). Leave unplaced if "
     "absent.",
     "sacral_foramen"),
    ("ventral_foramen_S5_right",
     "[FORAMEN] Same on the RIGHT side.",
     "sacral_foramen"),

    # ----- Mahato anatomical (n=8) -----
    ("sacral_promontory_midpoint",
     "[ANATOMICAL] Midline midpoint of the superior anterior margin of "
     "the S1 vertebral body. Also serves as the anterior point of the "
     "S1 superior endplate for sacral slope and pelvic incidence.",
     "mahato_anatomical"),
    ("sacral_base_midpoint",
     "[ANATOMICAL] Midline caudal-most point of the sacrum (sacral "
     "apex; tip of the coccygeal articular surface or inferior tip if "
     "coccyx is fused).",
     "mahato_anatomical"),
    ("auricular_surface_upper_left",
     "[ANATOMICAL] Midpoint of the superior margin of the LEFT sacral "
     "auricular surface (the ear-shaped articulation with the ilium).",
     "mahato_anatomical"),
    ("auricular_surface_upper_right",
     "[ANATOMICAL] Same on the RIGHT side.",
     "mahato_anatomical"),
    ("auricular_surface_lower_left",
     "[ANATOMICAL] Most inferior extent of the LEFT auricular surface.",
     "mahato_anatomical"),
    ("auricular_surface_lower_right",
     "[ANATOMICAL] Same on the RIGHT side.",
     "mahato_anatomical"),
    ("S1_facet_dorsal_left",
     "[ANATOMICAL] Dorsal edge of the LEFT S1 superior articular facet "
     "(posterior margin of the L5-S1 zygapophyseal joint, or L6-S1 in "
     "lumbarization).",
     "mahato_anatomical"),
    ("S1_facet_dorsal_right",
     "[ANATOMICAL] Same on the RIGHT side.",
     "mahato_anatomical"),

    # ----- Spinopelvic (n=5, optional) -----
    ("S1_endplate_posterior_midline",
     "[SPINOPELVIC] Midline posterior point of the S1 superior "
     "endplate. Together with sacral_promontory_midpoint defines the "
     "S1 endplate axis used for sacral slope and pelvic incidence.",
     "spinopelvic_optional"),
    ("femoral_head_center_left",
     "[SPINOPELVIC] Geometric center of the LEFT femoral head. Use "
     "axial, sagittal, and coronal MPR to estimate the center of the "
     "spherical head.",
     "spinopelvic_optional"),
    ("femoral_head_center_right",
     "[SPINOPELVIC] Same on the RIGHT side.",
     "spinopelvic_optional"),
    ("L1_endplate_anterior_midline",
     "[SPINOPELVIC] Midline anterior point of the L1 superior "
     "endplate. With L1_endplate_posterior_midline defines the L1 "
     "endplate axis for L1-S1 lumbar lordosis. Leave unplaced if L1 "
     "is outside the CT field of view.",
     "spinopelvic_optional"),
    ("L1_endplate_posterior_midline",
     "[SPINOPELVIC] Midline posterior point of the L1 superior "
     "endplate.",
     "spinopelvic_optional"),
]


def make_landmarks_template() -> dict:
    """Return Slicer markup-schema-v1.0.0 dict with all 31 fiducials in
    'preview' state (unplaced), ready for the radiologist."""
    points = []
    for i, (label, desc, group) in enumerate(LANDMARKS, 1):
        points.append({
            "id": str(i),
            "label": label,
            "description": desc,
            "associatedNodeID": "",
            "position": [0.0, 0.0, 0.0],
            "orientation": [-1.0, 0.0, 0.0,
                            0.0, -1.0, 0.0,
                            0.0, 0.0, 1.0],
            "selected": True,
            "locked": False,
            "visibility": True,
            "positionStatus": "preview",
        })
    return {
        "@schema":
            "https://raw.githubusercontent.com/slicer/slicer/main/"
            "Modules/Loadable/Markups/Resources/Schema/"
            "markups-schema-v1.0.0.json#",
        "markups": [{
            "type": "Fiducial",
            "coordinateSystem": "LPS",
            "coordinateUnits": "mm",
            "locked": False,
            "labelFormat": "%N",
            "controlPoints": points,
            "measurements": [],
            "display": {
                "visibility": True,
                "opacity": 1.0,
                "color": [0.4, 1.0, 0.4],
                "selectedColor": [1.0, 0.5, 0.5],
                "activeColor": [0.4, 1.0, 0.0],
                "propertiesLabelVisibility": True,
                "pointLabelsVisibility": True,
                "textScale": 3.0,
                "glyphType": "Sphere3D",
                "glyphScale": 1.0,
                "glyphSize": 5.0,
            },
        }],
    }


# ===========================================================================
# README -- professional documentation for radiologist reviewers
# ===========================================================================
README_MD = r"""# CTSpinoPelvic1K -- Lumbosacral Transitional Vertebra Annotation Package

## 1. Purpose

### 1.1 Delivery objective

The CTSpinoPelvic1K dataset and the nnU-Net model trained from it
are intended for direct deployment as **3D Slicer's default spinal-
CT segmentation backend**, replacing the spinopelvic output of
TotalSegmentator. TotalSegmentator's documented LSTV failure mode
(see accompanying paper) propagates from Slicer-based clinical
workflows directly to spine surgical planning, where it reproduces
the canonical mechanism of wrong-level surgery. Closing the LSTV
gap at the level of Slicer's default tool is the deliverable of
this work, and your annotations directly enable it.

The trained model will ship as a 3D Slicer extension, complete
with an automated spinopelvic-parameter measurement module trained
from the landmark annotations gathered through this package.

### 1.2 Scope of this package

This package contains 33 lumbosacral transitional vertebra (LSTV)-
positive cases from the CTSpinoPelvic1K dataset, programmatically
identified by combining vertebral counting from CTSpine1K and
filename morphological qualifiers from CTPelvic1K. The programmatic
labels have known limitations and require expert radiologist
verification before public release.

This document describes a standardized landmark-based annotation
protocol designed to permit objective, deterministic derivation of:

1. Castellvi (1984) classification of the lumbosacral junction.
2. Mahato (2020) lumbosacral transitional spectrum grade.
3. Mahato (2020) sacral morphometric measurements.
4. Standard spinopelvic parameters (sacral slope, pelvic tilt,
   pelvic incidence, L1-S1 lumbar lordosis), where the relevant
   landmarks are within the CT field of view.

The radiologist's role is to place anatomical landmarks; categorical
classification is performed by post-processing software from the
landmark coordinates. Expert observation is encoded by the
placement (or deliberate non-placement) of each landmark.

---

## 2. Authorship contribution

Co-authorship on the dataset paper is offered to reviewers who
complete, for each of the 33 LSTV cases:

1. Confirmation or proposed reclassification of the programmatic
   subtype (in `review.txt`).
2. Placement of the 26 required landmarks (Castellvi observation,
   Castellvi measurement, sacral foramina, and Mahato anatomical
   groups) in 3D Slicer, saved as `landmarks.mrk.json` in the token
   directory.

Reviewers who additionally place the 5 optional spinopelvic
landmarks enable computation of pelvic incidence, pelvic tilt,
sacral slope, and L1-S1 lumbar lordosis -- providing the basis for
a follow-up paper on automated spinopelvic measurement, with
continuing co-authorship for participating reviewers.

Reviewers who decline the landmark placement work but provide
written subtype confirmations only will be acknowledged in the
data card.

---

## 3. Workflow in 3D Slicer

The reviewer requires only **3D Slicer 5.x** (free, available at
slicer.org). No command-line tools, no Python, and no manual JSON
editing are needed. 3D Slicer's Markups module automatically reads
and writes the `.mrk.json` files used in this protocol; the
radiologist interacts only with the standard graphical interface.

### 3.0 Recommended: install the LSTVReviewer Slicer module

A custom Slicer module, **`LSTVReviewer.py`**, is included at the
top level of this package along with a **printable install guide,
`LSTVReviewer_install_guide.pdf`**. The module collapses the
per-case workflow to three clicks (Load -> place fiducials ->
Save) and tracks completion progress across all 33 cases.
Installation takes about 2 minutes and requires no coding.

**To install:** open `LSTVReviewer_install_guide.pdf` in this
directory and follow the numbered steps. In short:

1. In Slicer: `Edit` -> `Application Settings` -> `Modules`
2. Add the directory containing `LSTVReviewer.py` (i.e., this
   directory) to "Additional module paths".
3. Restart Slicer.
4. The "LSTV Reviewer" module appears under
   `Modules` -> `Annotation`.

Once installed, you can ignore Sections 3.2-3.4 below and use
the module's three-button interface. Sections 3.2-3.4 remain as
the fallback manual procedure if you prefer the standard
Slicer workflow without the helper module.

### 3.1 One-time setup

1. Download and install 3D Slicer 5.x from
   [https://www.slicer.org](https://www.slicer.org).
2. Launch Slicer. The default modules (Volumes, Segmentations,
   Markups) are sufficient; no extensions need to be installed.
3. Optional: in **Edit > Application Settings > Markups**, increase
   the default glyph size if the default fiducial markers are too
   small to see clearly on the CT slices.

### 3.2 Per-case annotation procedure

For each token directory (e.g. `sacralization/token_167/`):

#### Step 1. Load the CT volume

- **File > Add Data > Choose File(s) to Add**
- Select `ct.nii.gz` (or, in separate-mode cases,
  `ct_separate_spine.nii.gz` -- the spine mask corresponds to this
  acquisition).
- Confirm the "Description" column shows "Volume".
- Click **OK**.

#### Step 2. Load the segmentation masks as overlays

- **File > Add Data > Choose File(s) to Add**
- Select `label.nii.gz` (or, in separate-mode cases,
  `label_spine.nii.gz` and `label_pelvic.nii.gz`)
  (Ctrl/Cmd+click to multi-select).
- For each row in the dialog, change the "Description" from
  "LabelMapVolume" to "Segmentation". This enables transparency
  control over the masks.
- Click **OK**. The masks now overlay the CT in the slice viewers
  and can be toggled or made semi-transparent in the Data module.

#### Step 3. Load the landmark template

- **File > Add Data > Choose File(s) to Add**
- Select `landmarks_template.mrk.json`.
- Confirm the "Description" reads "Markups Fiducial".
- Click **OK**. A list of 31 named fiducials appears in
  the Markups module, each in **"preview" status** (unplaced).

#### Step 4. Switch to the Markups module

- In the top-left **Module** dropdown, select **Markups**.
- The control points list shows all 31 fiducials with their
  anatomical descriptions visible when each is selected.

#### Step 5. Place the landmarks

For each fiducial:

- Click on the fiducial's row in the **Control Points** list to
  highlight it.
- Read the description in the bottom panel for the anatomical
  definition.
- Press the **Place** button (the cursor-with-target icon at the
  top of the Markups module), or use hotkey **Ctrl+Shift+A**.
- In the slice viewers (Red / Yellow / Green panes), navigate to
  the anatomical location using mouse-wheel or scrollbars.
- **Click** on the location. The fiducial is placed, its status
  changes from "preview" to "defined", and Slicer automatically
  advances to the next fiducial in the list.

#### Step 6. Skip a landmark that does not apply

For [OBSERVATION] landmarks (Castellvi articulation and fusion)
when the structure is absent on a side, OR for [FORAMEN] landmarks
S4/S5 when the sacrum has fewer segments:

- In the Control Points list, **right-click** the fiducial.
- Choose **"Unset position"**. The fiducial's status reverts to
  "preview" and is treated as "absent" by the processing software.
- If the fiducial was never placed, it is already in "preview"
  status and no action is needed -- simply skip past it.

#### Step 7. Save the landmark file

3D Slicer writes the JSON automatically based on your placements.

- Press **Ctrl+S** (or **File > Save**).
- A dialog lists all data nodes in the scene.
- Find the row for the markup node (originally named after the
  template).
- In the **File Name** column, click and edit to
  `landmarks.mrk.json` (drop the `_template` suffix; do not
  overwrite the template file).
- In the **File Directory** column, browse to the same token
  directory.
- Ensure the markup node's checkbox is ticked. The other rows
  (CT, masks) can be unchecked.
- Click **Save**.

#### Step 8. Fill in the review form

- Open `review.txt` in any plain text editor (Notepad, TextEdit,
  VS Code, BBEdit). The form is a few lines of `Field: value`
  text.
- Complete the fields and save.

#### Step 9. Move to the next case

- **File > Close Scene** (or restart Slicer) to clear loaded data.
- Repeat from Step 1 for the next token directory.

### 3.3 What gets generated automatically

Three things you do NOT type by hand:

| Output                          | Generated by                                 |
|---------------------------------|----------------------------------------------|
| `landmarks.mrk.json`            | 3D Slicer's Markups module on Save           |
| Castellvi grade (Ia/Ib/IIa/...) | Post-processing from your landmark placements |
| Mahato spectrum grade (B/C/D/E) | Post-processing from your landmark placements |
| All linear / angular morphometrics | Post-processing geometry from coordinates |

The only file you edit by hand is `review.txt`, which is plain
text.

### 3.4 Returning your work

When you have annotated all cases (or as many as you choose to
review), archive the entire `lstv_review/` folder
(`tar -czvf lstv_review.tar.gz lstv_review/` on macOS/Linux, or
right-click > "Send to > Compressed (zipped) folder" on Windows)
and return to the corresponding author.

---

## 4. Classification systems

### 4.1 Castellvi (1984) -- L5/S1 junction morphology

Castellvi describes the lumbosacral junction. It does NOT
distinguish lumbarization (4-segment sacrum) from sacralization
(6-segment sacrum); the Mahato spectrum (Section 4.2) supplies
that distinction.

| Type     | Anatomical criterion                                                                  | Subtypes      |
|----------|---------------------------------------------------------------------------------------|---------------|
| Type I   | Dysplastic transverse process(es) >=19 mm craniocaudal dimension; no articulation     | Ia / Ib       |
| Type II  | Diarthrodial pseudoarticulation between transverse process and sacral ala             | IIa / IIb     |
| Type III | Complete osseous fusion of transverse process to sacral ala                           | IIIa / IIIb   |
| Type IV  | Type II on one side AND Type III on the contralateral side                            | (no subtypes) |

Subtype suffix: `a` = unilateral, `b` = bilateral.

### 4.2 Mahato (2020) -- LSTV spectrum

The Mahato spectrum places the case on the
lumbarization-to-sacralization axis based on the number of sacral
segments and the symmetry of any articulation or fusion.

**The original Mahato (2020) paper is included in this package as
`Mahato_2020_LSTV_classification.pdf`.** Refer to Fig. 1 of the
paper for visual depictions of the lumbarization spectrum
(panels B-D), Fig. 2 for the sacralization spectrum (panels B-E),
Tables 3-4 for the morphometric measurement definitions, and the
auricular surface position classification (Section 4.3 below).

#### 4.2.1 Lumbarization spectrum (Mahato Fig. 1, panels B-D)

| Grade | Sacral segments | Description                                                          |
|-------|-----------------|----------------------------------------------------------------------|
| B     | 5               | Incomplete lumbarization: partial unilateral S1-S2 separation        |
| C     | 4               | 4-segment sacrum with bilateral L6-S1 accessory articulation         |
| D     | 4               | Complete lumbarization: 4-segment sacrum, no accessory articulation |

#### 4.2.2 Sacralization spectrum (Mahato Fig. 2, panels B-E)

| Grade | Sacral segments | Description                                                          |
|-------|-----------------|----------------------------------------------------------------------|
| B     | 5               | Unilateral L5-S1 accessory articulation                              |
| C     | 5               | Bilateral L5-S1 accessory articulation                               |
| D     | 6               | Incomplete sacralization: 6-segment sacrum, one side unfused at L5-S1 |
| E     | 6               | Complete sacralization: 6-segment sacrum, bilateral fusion           |

### 4.3 Auricular surface position (Mahato 2010)

Three-class system based on the position of the upper border of
the sacral auricular surface relative to S1:

- Type I / normal: AS upper border within the upper half of S1
- Type II / high:  AS upper border above the sacral promontory
- Type III / low:  AS upper border below the midpoint of S1

This is computed automatically from auricular surface and foramen
landmarks; the radiologist does not assign it manually.

---

## 5. How landmark placement determines classification

The downstream processing software (`process_reviews.py`) derives
classifications from landmark coordinates as follows.

### 5.1 Sacral segment count

Determined by the number of placed ventral foramen pairs:

- 3 pairs (S1, S2, S3 placed; S4 and S5 unplaced) -> 4-segment
  sacrum (lumbarization morphology)
- 4 pairs (S1-S4 placed; S5 unplaced) -> 5-segment sacrum
  (normal morphology)
- 5 pairs (S1-S5 all placed) -> 6-segment sacrum
  (sacralization morphology)

Asymmetric counts (e.g., 4 pairs left, 5 pairs right) generate a
warning in the validation report and use the more conservative
count.

### 5.2 Castellvi grade (per side, then combined)

For each side independently, the algorithm assigns:

1. Type III if `castellvi_fusion_bridge_<side>` is placed.
2. Type II if `castellvi_articulation_<side>` is placed and no
   fusion is present.
3. Type I if neither articulation nor fusion is present, but the
   craniocaudal distance between `transitional_TP_cranial_<side>`
   and `transitional_TP_caudal_<side>` is >= 19 mm.
4. None if none of the above conditions is met.

The two sides are then combined:

- Bilateral identical types -> Ib, IIb, or IIIb.
- Unilateral (one side typed, other not) -> Ia, IIa, or IIIa.
- Type II on one side + Type III on contralateral -> Type IV.
- Atypical mixed combinations (e.g., I left + II right) are
  reported with explicit per-side notation.

### 5.3 Mahato spectrum grade

Determined from the joint observation of lumbar body count
(extracted from `label.nii.gz`), sacral segment count
(Section 5.1), and articulation / fusion landmark presence:

- 5 lumbar bodies + 5-segment sacrum + no articulation, no fusion
  -> **A** (normal)
- 6 lumbar bodies + 5-segment sacrum -> Lumbarization **B**
  (incomplete; partial S1-S2 separation)
- 6 lumbar bodies + 4-segment sacrum + bilateral articulation
  -> Lumbarization **C**
- 6 lumbar bodies + 4-segment sacrum + no articulation
  -> Lumbarization **D** (complete)
- 5 lumbar bodies + 5-segment sacrum + unilateral articulation
  -> Sacralization **B**
- 5 lumbar bodies + 5-segment sacrum + bilateral articulation
  -> Sacralization **C**
- 4 lumbar bodies + 6-segment sacrum + unilateral fusion
  -> Sacralization **D** (incomplete)
- 4 lumbar bodies + 6-segment sacrum + bilateral fusion
  -> Sacralization **E** (complete)

Configurations that do not fit any grade are flagged
"indeterminate" with a description of the observed combination,
for case-by-case adjudication.

### 5.4 Mahato morphometric measurements

Computed deterministically from the placed landmark coordinates
(LPS, mm), as defined in Mahato (2020) Table 3 and Table 4:

- **SH** (sacral height): Euclidean distance from
  sacral_promontory_midpoint to sacral_base_midpoint.
- **AH** (auricular surface height, per side): distance from
  auricular_surface_upper to auricular_surface_lower.
- **IFD** (inter-facet distance): distance between left and right
  S1_facet_dorsal.
- **IAD** (inter-auricular distance): distance between left and
  right auricular_surface_upper.
- **BW** (S1 body width): distance between left and right
  ventral_foramen_S1.
- **AW** (alar width): IAD - BW.
- **S1O** (S1 transverse element obliquity, per side, coronal
  plane, degrees): angle between the vector from sacral_promontory
  to auricular_surface_upper and the horizontal reference (line
  through left and right ventral_foramen_S2).
- **S1T** (S1 corridor thickness, per side, mm): craniocaudal
  distance from ventral_foramen_S1 to ventral_foramen_S2.
- **S2T** (S2 corridor thickness, per side, mm): craniocaudal
  distance from ventral_foramen_S2 to ventral_foramen_S3.
- **AS position type**: I / II / III, derived from the z-position
  of auricular_surface_upper relative to the S1 segment z-extent
  (sacral_promontory_midpoint to mean of ventral_foramen_S2 z).

### 5.5 Standard spinopelvic measurements (optional)

Computed in the sagittal plane from the optional spinopelvic
landmarks, when placed:

- **Sacral slope**: angle between the S1 superior endplate
  (sacral_promontory_midpoint to S1_endplate_posterior_midline)
  and the horizontal.
- **Pelvic tilt**: angle between the vertical and the line from
  the S1 endplate midpoint to the bicoxofemoral axis (midpoint of
  the two femoral_head_center landmarks).
- **Pelvic incidence**: angle between the S1 endplate normal at
  its midpoint and the line to the bicoxofemoral axis.
- **L1-S1 lumbar lordosis**: angle between the L1 superior
  endplate (L1_endplate_anterior to L1_endplate_posterior) and
  the S1 superior endplate.

---

## 6. Landmark catalogue

The Slicer template `landmarks_template.mrk.json` contains 31
named fiducials, each with its anatomical description visible in
the Markups module's "Description" field.

### 6.1 Castellvi observation (n=4) -- place IF present

| Label                          | Indicates             |
|--------------------------------|-----------------------|
| castellvi_articulation_left    | Castellvi II, left    |
| castellvi_articulation_right   | Castellvi II, right   |
| castellvi_fusion_bridge_left   | Castellvi III, left   |
| castellvi_fusion_bridge_right  | Castellvi III, right  |

### 6.2 Castellvi measurement (n=4) -- TP dysplasia assessment

| Label                          | Definition                                       |
|--------------------------------|--------------------------------------------------|
| transitional_TP_cranial_left   | Most superior point of left transitional-vertebra TP |
| transitional_TP_cranial_right  | Same, right                                      |
| transitional_TP_caudal_left    | Most inferior point of left TP at the same level |
| transitional_TP_caudal_right   | Same, right                                      |

The transitional vertebra is L5 in sacralization and L6 in
lumbarization. Craniocaudal length >=19 mm without articulation or
fusion satisfies Castellvi Type I.

### 6.3 Sacral foramina (n=10) -- segment count

Place every ventral (anterior) foramen visible. Foramina S4 and
S5 may legitimately be absent depending on segment count:

| Label                       | Present in sacrum of segment count |
|-----------------------------|------------------------------------|
| ventral_foramen_S1_left/right | >=2 segments (always)             |
| ventral_foramen_S2_left/right | >=3 segments (always)             |
| ventral_foramen_S3_left/right | >=4 segments (always)             |
| ventral_foramen_S4_left/right | 5 or 6 segments                   |
| ventral_foramen_S5_left/right | 6 segments only (sacralization)   |

### 6.4 Mahato anatomical (n=8) -- always place if visible

| Label                              | Definition                                  |
|------------------------------------|---------------------------------------------|
| sacral_promontory_midpoint         | Midline midpoint of superior anterior S1    |
| sacral_base_midpoint               | Midline caudal apex of sacrum               |
| auricular_surface_upper_left/right | Midpoint of superior margin of AS, per side |
| auricular_surface_lower_left/right | Most inferior extent of AS, per side        |
| S1_facet_dorsal_left/right         | Dorsal edge of S1 superior articular facet  |

### 6.5 Spinopelvic optional (n=5) -- bonus landmarks

| Label                            | Used for                          |
|----------------------------------|-----------------------------------|
| S1_endplate_posterior_midline    | Sacral slope, pelvic incidence    |
| femoral_head_center_left/right   | Pelvic incidence, pelvic tilt     |
| L1_endplate_anterior_midline     | L1-S1 lumbar lordosis             |
| L1_endplate_posterior_midline    | L1-S1 lumbar lordosis             |

If L1 or the femoral heads are outside the CT field of view,
leave the corresponding landmark in "preview" status. The
spinopelvic measurements requiring those landmarks are then
omitted for that case; other measurements are unaffected.

---

## 7. Subtype definitions (programmatic; awaiting verification)

The directory structure groups cases by the programmatic subtype.
Your landmark placements determine the algorithmically-derived
classification, which will be compared with the programmatic
label.

- **lumb** (n=14): 6 lumbar bodies labeled in `label.nii.gz`
  (mask value 6 = L6 present).
- **sacralization** (n=6): pelvic source filename carried
  `_sacralization_` qualifier; expected Castellvi III.
- **semisacralization** (n=2, REVIEW PRIORITY): pelvic filename
  carried `_semisacralization_` qualifier; expected Castellvi
  I or II.
- **sacr_count** (n=9): 4 lumbar bodies labeled in spine mask;
  L5 inferred fused to sacrum. May represent true sacralization
  OR under-annotation; landmark placement will adjudicate.
- **ambiguous** (n=2, REVIEW PRIORITY): cross-source disagreement
  not reconcilable programmatically.

---

## 8. Reference normal case

`reference_normal/token_<id>/` contains one fused-mode normal
case for visual calibration. No annotation is requested for this
case; use it to configure your Slicer display settings (window
level, segment color, fiducial size) before beginning LSTV cases.

---

## 9. Output files per token

After your annotation:

| File                          | Source        | Purpose                            |
|-------------------------------|---------------|------------------------------------|
| ct[_<match_type>].nii.gz      | provided      | CT volume(s)                       |
| label[_<match_type>].nii.gz   | provided      | Segmentation (10 classes: L1-L6, sacrum, hips) |
| landmarks_template.mrk.json   | provided      | Template; do not modify            |
| landmarks.mrk.json            | reviewer      | Your landmark placements           |
| review.txt                    | reviewer      | Categorical confirmation + notes   |

---

## 10. References

1. Castellvi AE, Goldstein LA, Chan DPK. Lumbosacral transitional
   vertebrae and their relationship with lumbar extradural
   defects. *Spine (Phila Pa 1976)* 1984;9(5):493-495.
2. Konin GP, Walz DM. Lumbosacral transitional vertebrae:
   classification, imaging findings, and clinical relevance.
   *AJNR Am J Neuroradiol* 2010;31(10):1778-1786.
3. Mahato NK. Variable positions of the sacral auricular surface:
   classification and importance. *Neurosurg Focus* 2010;28:E12.
4. Mahato NK. Re-examining the spectrum of lumbosacral
   transitional dysmorphisms: quantifying joint asymmetries and
   evaluating the anatomy of screw fixation corridors.
   *Neurospine* 2020;17(1):294-303.
   doi:10.14245/ns.1938102.051
5. Sekharappa V, Amritanand R, Krishnan V, David KS. Lumbosacral
   transition vertebra: prevalence and its significance.
   *Asian Spine J* 2014;8(1):51-58.
6. Lumbosacral transitional vertebra. *Radiopaedia.org* (last
   revised 10 March 2026).
"""


# ===========================================================================
# review.txt -- minimal categorical form (classification is derived from
# landmarks; this is for confirmation and free-text observations only)
# ===========================================================================
REVIEW_TEMPLATE = """# CTSpinoPelvic1K -- LSTV review form
# Token: {token}
# Programmatic subtype: {subtype}
# Match type: {match_type}
#
# Categorical classifications (Castellvi grade, Mahato spectrum grade,
# auricular surface position, and morphometric measurements) are derived
# automatically from your landmark placements in landmarks.mrk.json.
# Use this form only for the items below.

# ---- Reviewer ---------------------------------------------------------
Reviewer name:
Institution:
Date (YYYY-MM-DD):

# ---- Subtype confirmation ---------------------------------------------
# Does the programmatic subtype above match what you observe?
# One of: yes / no
Confirmed:

# If "no", specify what you observe instead.
# One of: normal | lumb | sacralization | semisacralization | sacr_count | ambiguous
Proposed subtype:

Reason for reclassification (if any):

# ---- Landmark placement confirmation ----------------------------------
# Have you saved landmarks.mrk.json in this directory?
Landmarks saved:

# Note any landmark you intentionally left unplaced and why
# (e.g., "S5 foramen absent (4-segment sacrum)";
#        "no fusion bridge present on either side";
#        "L1 outside FOV").
Unplaced landmarks (with reason):

# ---- Free-text notes --------------------------------------------------
# Any clinical observations, image-quality concerns, atypical
# anatomy not captured by the standardized landmarks, etc.
Notes:


# ---- End of form ------------------------------------------------------
"""


# ===========================================================================
# Helpers
# ===========================================================================
def find_file(hf_export, subdir_candidates, basename):
    for sub in subdir_candidates:
        p = hf_export / sub / basename
        if p.exists():
            return p
    return None


def _resolve_manifest_path(hf_export: Path, rel_path: str) -> Path:
    """Resolve a relative path from the manifest against hf_export."""
    if not rel_path:
        return None
    p = hf_export / rel_path
    return p if p.exists() else None


def extract_record(record, hf_export, out_dir, ct_suffix="", label_suffix=""):
    record_id = (record.get("record_id") or record.get("id")
                 or record.get("token"))
    if not record_id:
        return {}

    # Trust manifest paths first (CTSpinoPelvic1K convention).
    ct_src = _resolve_manifest_path(hf_export, record.get("ct_file"))
    label_src = _resolve_manifest_path(hf_export, record.get("label_file"))

    # Fallback: legacy <token>.nii.gz pattern in known subdirs
    if ct_src is None:
        ct_src = find_file(hf_export, ["ct", "images", "imagesTr"],
                           f"{record_id}.nii.gz")
    if label_src is None:
        label_src = find_file(hf_export, ["label", "labels", "labelsTr"],
                              f"{record_id}.nii.gz")

    if ct_src is None:
        log.warning(f"CT not found for {record_id} "
                    f"(manifest ct_file={record.get('ct_file')!r})")
        return {}

    out = {}
    ct_dst = out_dir / f"ct{ct_suffix}.nii.gz"
    shutil.copy2(ct_src, ct_dst)
    out["ct"] = ct_dst

    if label_src is not None:
        label_dst = out_dir / f"label{label_suffix}.nii.gz"
        shutil.copy2(label_src, label_dst)
        out["label"] = label_dst
    else:
        log.warning(f"label not found for {record_id} "
                    f"(manifest label_file={record.get('label_file')!r})")
    return out


def write_review_template(token_dir, token, subtype, match_type="unknown"):
    text = REVIEW_TEMPLATE.format(token=token, subtype=subtype, match_type=match_type)
    (token_dir / "review.txt").write_text(text)


def write_landmarks_template(token_dir):
    template = make_landmarks_template()
    with open(token_dir / "landmarks_template.mrk.json", "w") as f:
        json.dump(template, f, indent=2)


# ===========================================================================
# Main
# ===========================================================================
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hf_export", default="~/CTSpinoPelvic1K/data/hf_export")
    ap.add_argument("--out_dir", default="~/CTSpinoPelvic1K/data/lstv_review")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--splits_file", default=None,
                    help="Path to splits_5fold.json (v6+) for per-token "
                         "6-way subtype assignment. Default: "
                         "<hf_export>/splits_5fold.json")
    ap.add_argument("--force", action="store_true")
    # Paths to the Slicer module and install guide PDF; default to the
    # script's own directory (assumes they sit next to extract_lstv_review.py)
    script_dir = Path(__file__).resolve().parent
    ap.add_argument("--slicer_module",
                    default=str(script_dir / "LSTVReviewer.py"),
                    help="Path to LSTVReviewer.py")
    ap.add_argument("--install_guide",
                    default=str(script_dir / "LSTVReviewer_install_guide.pdf"),
                    help="Path to the install guide PDF")
    ap.add_argument("--mahato_paper",
                    default=str(script_dir / "mahato.pdf"),
                    help="Path to the Mahato (2020) classification paper")
    ap.add_argument("--no_archive", action="store_true",
                    help="Skip creating the .tar.gz archive at the end")
    args = ap.parse_args()

    hf_export = Path(args.hf_export).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    manifest_path = Path(args.manifest).expanduser() if args.manifest \
        else hf_export / "manifest.json"

    if not hf_export.exists():
        sys.exit(f"ERROR: HF export not found: {hf_export}")
    if not manifest_path.exists():
        sys.exit(f"ERROR: manifest not found: {manifest_path}")

    if out_dir.exists():
        if args.force:
            log.info(f"Removing existing {out_dir}")
            shutil.rmtree(out_dir)
        else:
            sys.exit(f"ERROR: {out_dir} exists; pass --force to overwrite")
    out_dir.mkdir(parents=True)

    log.info(f"Loading manifest: {manifest_path}")
    with open(manifest_path) as f:
        manifest_data = json.load(f)
    records = list(manifest_data.values()) if isinstance(manifest_data, dict) else manifest_data
    log.info(f"  {len(records)} records")

    # ------------------------------------------------------------------
    # Load 6-way subtype assignment from splits_5fold.json (where the
    # 6-way taxonomy actually lives in this repo). The manifest's
    # lstv_label / lstv_class fields use a legacy 4-class scheme; the
    # 6-way taxonomy was added later and stored only in the splits.
    # ------------------------------------------------------------------
    splits_file = (Path(args.splits_file).expanduser().resolve()
                   if args.splits_file
                   else hf_export / "splits_5fold.json")

    splits_subtypes = {}  # token (str) -> subtype (str)
    if splits_file.exists():
        # First try benchmark_totalseg's loader, same as merge_benchmark_shards.py
        try:
            sys.path.insert(0, str(Path(__file__).resolve().parent))
            import benchmark_totalseg as bt
            splits_subtypes, schema = bt.load_splits_subtype_map(splits_file)
            splits_subtypes = {str(k): v for k, v in splits_subtypes.items()}
            log.info(f"Loaded {len(splits_subtypes)} subtype assignments "
                     f"from {splits_file.name} (schema v{schema}) via "
                     f"benchmark_totalseg")
        except Exception as e:
            log.warning(f"benchmark_totalseg loader failed ({e}); "
                        f"trying direct JSON read")
            try:
                with open(splits_file) as f:
                    sd = json.load(f)
                # Try several known structures
                if "lstv_subtypes" in sd and isinstance(sd["lstv_subtypes"], dict):
                    splits_subtypes = {str(k): v for k, v in sd["lstv_subtypes"].items()}
                elif "tokens" in sd and isinstance(sd["tokens"], list):
                    splits_subtypes = {str(t.get("token")): t.get("lstv_subtype")
                                       for t in sd["tokens"] if t.get("token") is not None}
                elif "records" in sd and isinstance(sd["records"], list):
                    splits_subtypes = {str(r.get("token")): r.get("lstv_subtype")
                                       for r in sd["records"] if r.get("token") is not None}
                else:
                    log.warning(f"Could not find subtype map in {splits_file.name}; "
                                f"top-level keys: {list(sd.keys())[:10]}")
                splits_subtypes = {k: v for k, v in splits_subtypes.items() if v}
                log.info(f"Loaded {len(splits_subtypes)} subtype "
                         f"assignments from {splits_file.name} (direct read)")
            except Exception as e2:
                log.warning(f"Direct JSON read also failed ({e2})")
    else:
        log.warning(f"splits_file does not exist: {splits_file}")
        log.warning(f"  -> falling back to per-record lstv_subtype/lstv_label "
                    f"fields, which may yield zero LSTV cases on this manifest")

    patient_records = {}
    for rec in records:
        token = str(rec.get("token") or rec.get("patient_id") or "")
        if token:
            patient_records.setdefault(token, []).append(rec)
    log.info(f"  {len(patient_records)} unique patients")

    lstv_patients = {st: {} for st in LSTV_SUBTYPES}
    n_assigned_from_splits = 0
    n_assigned_from_manifest = 0
    for token, recs in patient_records.items():
        # Resolution order: splits_5fold.json -> manifest lstv_subtype
        # -> manifest lstv_label -> "normal"
        st = (splits_subtypes.get(token)
              or recs[0].get("lstv_subtype")
              or recs[0].get("lstv_label")
              or "normal")
        if splits_subtypes.get(token):
            n_assigned_from_splits += 1
        else:
            n_assigned_from_manifest += 1
        if st in LSTV_SUBTYPES:
            lstv_patients[st][token] = recs

    log.info(f"Subtype source: {n_assigned_from_splits} from splits, "
             f"{n_assigned_from_manifest} from manifest fallback")

    log.info("LSTV patient counts:")
    for st in LSTV_SUBTYPES:
        log.info(f"  {st:24s}: {len(lstv_patients[st])} patients")

    if sum(len(v) for v in lstv_patients.values()) == 0:
        sys.exit("ERROR: no LSTV patients found.")

    # Pick one normal reference (prefer fused)
    normal_ref = None
    for token, recs in sorted(patient_records.items()):
        if recs[0].get("lstv_subtype", "normal") != "normal":
            continue
        for rec in recs:
            if rec.get("match_type") == "fused":
                normal_ref = (token, [rec])
                break
        if normal_ref:
            break
    if normal_ref is None:
        for token, recs in sorted(patient_records.items()):
            if recs[0].get("lstv_subtype", "normal") == "normal":
                normal_ref = (token, recs[:1])
                break

    manifest_rows = []
    total_lstv_tokens = sum(len(p) for p in lstv_patients.values())
    token_counter = 0
    extraction_start = time.time()
    for subtype in LSTV_SUBTYPES:
        patients = lstv_patients[subtype]
        if not patients:
            continue
        subtype_dir = out_dir / subtype
        subtype_dir.mkdir(exist_ok=True)
        log.info(f"=== {subtype} ({len(patients)} patients) ===")

        for token, recs in sorted(patients.items()):
            token_counter += 1
            t_token = time.time()
            token_dir = subtype_dir / f"token_{token}"
            token_dir.mkdir(exist_ok=True)
            log.info(f"[{token_counter}/{total_lstv_tokens}] {subtype}/token_{token} "
                     f"({len(recs)} record(s))")

            display_match = "+".join(sorted({r.get("match_type", "?") for r in recs}))

            for rec in recs:
                mt = rec.get("match_type", "unknown")
                suffix = f"_{mt}" if len(recs) > 1 else ""
                paths = extract_record(rec, hf_export, token_dir,
                                       ct_suffix=suffix, label_suffix=suffix)
                manifest_rows.append({
                    "subtype": subtype,
                    "token": token,
                    "record_id": rec.get("record_id"),
                    "match_type": mt,
                    "lstv_class": rec.get("lstv_class"),
                    "ct_file": paths["ct"].name if "ct" in paths else None,
                    "label_file": paths["label"].name if "label" in paths else None,
                    "directory": f"{subtype}/token_{token}",
                })

            write_review_template(token_dir, token, subtype, display_match)
            write_landmarks_template(token_dir)
            log.info(f"           done in {time.time() - t_token:.1f}s")

    log.info(f"Extraction loop complete in {time.time() - extraction_start:.1f}s "
             f"({token_counter} LSTV tokens)")

    # Reference normal
    if normal_ref:
        token, recs = normal_ref
        ref_dir = out_dir / "reference_normal" / f"token_{token}"
        ref_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"=== reference_normal ===")
        log.info(f"  token_{token}")
        for rec in recs:
            paths = extract_record(rec, hf_export, ref_dir)
            manifest_rows.append({
                "subtype": "normal_reference",
                "token": token,
                "record_id": rec.get("record_id"),
                "match_type": rec.get("match_type"),
                "lstv_class": 0,
                "ct_file": paths["ct"].name if "ct" in paths else None,
                "label_file": paths["label"].name if "label" in paths else None,
                "directory": f"reference_normal/token_{token}",
            })

    df = pd.DataFrame(manifest_rows)
    df.to_csv(out_dir / "manifest.csv", index=False)
    (out_dir / "README.md").write_text(README_MD)
    log.info(f"Done. Output: {out_dir}")
    log.info(f"  Total records: {len(df)}")
    log.info("  By subtype:")
    log.info(df["subtype"].value_counts().to_string())

    import subprocess
    try:
        size = subprocess.check_output(["du", "-sh", str(out_dir)]).decode().split()[0]
        log.info(f"  Total size: {size}")
    except Exception:
        pass

    # ---------------- Bundle Slicer module + reference PDFs ------------
    log.info("Bundling reviewer tools:")
    module_src = Path(args.slicer_module).expanduser().resolve()
    guide_src = Path(args.install_guide).expanduser().resolve()
    mahato_src = Path(args.mahato_paper).expanduser().resolve()
    bundled = []
    # Each entry: (source path, destination filename in out_dir, label)
    bundle_items = [
        (module_src, module_src.name, "Slicer module"),
        (guide_src, guide_src.name, "install guide PDF"),
        (mahato_src, "Mahato_2020_LSTV_classification.pdf",
         "Mahato (2020) classification paper"),
    ]
    for src, dst_name, label in bundle_items:
        if src.exists():
            dst = out_dir / dst_name
            shutil.copy2(src, dst)
            log.info(f"  copied {label}: {dst_name}  ({dst.stat().st_size:,} bytes)")
            bundled.append(dst_name)
        else:
            log.warning(f"{label} not found at {src}")
            log.info(f"        radiologist will receive package without it")

    # ---------------- Create distributable archive ----------------
    if not args.no_archive:
        archive_path = out_dir.parent / f"{out_dir.name}.zip"
        if archive_path.exists():
            archive_path.unlink()
        log.info(f"Creating archive: {archive_path}")
        try:
            # shutil.make_archive returns the actual path written
            written = shutil.make_archive(
                base_name=str(out_dir.parent / out_dir.name),
                format="zip",
                root_dir=str(out_dir.parent),
                base_dir=out_dir.name,
            )
            archive_path = Path(written)
            arc_size = subprocess.check_output(
                ["du", "-sh", str(archive_path)]).decode().split()[0]
            log.info(f"  {archive_path.name}  ({arc_size})")
            log.info(f"Ready to ship:")
            log.info(f"  {archive_path}")
            log.info(f"The radiologist unzips this single file. Inside they will find:")
            log.info(f"  README.md")
            for name in bundled:
                log.info(f"  {name}")
            log.info(f"  manifest.csv")
            log.info(f"  <subtype>/token_<id>/  (case directories)")
        except Exception as e:
            log.warning(f"zip creation failed ({e})")
    else:
        log.info("--no_archive set; skipping zip creation.")
        log.info(f"Output directory ready at: {out_dir}")


if __name__ == "__main__":
    main()
