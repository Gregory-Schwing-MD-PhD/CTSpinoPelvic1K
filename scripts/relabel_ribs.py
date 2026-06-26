"""
relabel_ribs.py — anatomically correct rib INSTANCE segmentation on partial-FOV CT.

TotalSegmentator labels ribs by counting from the top of the *visible* field of
view, so on a cropped/partial CT its rib numbers are systematically wrong (the
top visible rib gets called "rib 1" even when it is really rib 6). This script
fixes that by IGNORING TotalSegmentator's instance numbering entirely and instead
re-deriving each rib's number + side from GROUND-TRUTH thoracic vertebrae, which
are the only reliable anatomical ruler in a partial FOV.

Pipeline (one function per step, see the functions below):
  A. Unify all TotalSegmentator rib NIfTIs into one binary rib mask.
  B. 3D connected components -> one label per physical rib; drop tiny noise blobs.
  C. Per-vertebra morphological dilation, done LOCALLY in a padded bounding box so
     we never dilate the full 512^3 volume (that OOMs).
  D. For each rib component: vote it onto the dilated vertebra it overlaps most
     (that vertebra's number = the rib's number), then decide Left/Right from the
     rib-vs-vertebra centroid offset along the patient's L<->R world axis.
  E. Write a multi-label NIfTI in the ORIGINAL affine/header, using a configurable
     {side, number} -> integer-id scheme.

Every file is assumed already co-registered (identical shape + affine): the GT
vertebrae and the TotalSegmentator ribs share one voxel grid.

Usage
-----
  python scripts/relabel_ribs.py \
      --rib_dir   ts_output/ribs/ \
      --vertebrae gt/thoracic_vertebrae.nii.gz \
      --out       out/ribs_relabelled.nii.gz \
      [--min_voxels 500] [--dilation_radius 4] [--pad 10] [--connectivity 26]
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from nibabel.affines import apply_affine
from scipy import ndimage

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("relabel_ribs")

# ---------------------------------------------------------------------------
# Output label scheme. Left rib N -> LEFT_OFFSET + N, right rib N -> RIGHT_OFFSET
# + N. With the defaults below, left rib 9 = 109, right rib 9 = 209 — distinct,
# human-readable, and collision-free for N up to 99. Swap these (or replace
# `rib_label_id`) to emit TotalSegmentator-native ids or any custom dictionary.
# ---------------------------------------------------------------------------
LEFT_OFFSET = 100
RIGHT_OFFSET = 200


def rib_label_id(side: str, number: int) -> int:
    """Map an anatomical (side, rib-number) pair to a single integer label id.

    Parameters
    ----------
    side   : "left" or "right".
    number : the rib number (1..12) inherited from the anchoring vertebra.

    Returns
    -------
    int : LEFT_OFFSET + number for the left side, RIGHT_OFFSET + number for the right.
    """
    return (LEFT_OFFSET if side == "left" else RIGHT_OFFSET) + number


# ===========================================================================
# Step A — Unification and Binarization
# ===========================================================================
def load_and_unify_ribs(rib_dir: Path) -> Tuple[np.ndarray, np.ndarray, nib.Nifti1Header]:
    """Load every rib NIfTI in `rib_dir` and OR them into one binary rib mask.

    TotalSegmentator can emit ribs either as one multi-label file or as many
    single-rib files; this handles both by treating ANY non-zero voxel in ANY
    input file as "rib bone". The union is accumulated in place (uint8), so peak
    memory is one volume regardless of how many input files there are.

    Parameters
    ----------
    rib_dir : directory of `*.nii.gz` rib masks, all on the same voxel grid.

    Returns
    -------
    (binary_ribs, affine, header)
        binary_ribs : uint8 volume, 1 = rib bone, 0 = background.
        affine      : the shared 4x4 voxel->world affine (from the first file).
        header      : the first file's header, reused for the output.
    """
    files = sorted(rib_dir.glob("*.nii.gz"))
    if not files:
        raise FileNotFoundError(f"no rib *.nii.gz files in {rib_dir}")

    first = nib.load(str(files[0]))
    affine = first.affine
    header = first.header
    shape = first.shape[:3]
    binary = np.zeros(shape, dtype=np.uint8)

    for f in files:
        img = nib.load(str(f))
        if img.shape[:3] != shape:
            raise ValueError(f"{f.name} shape {img.shape[:3]} != {shape} (not co-registered)")
        # dataobj avoids loading a float64 copy; cast the slice-read to bool then OR in.
        binary |= (np.asarray(img.dataobj) > 0).astype(np.uint8)

    log.info("Step A: unified %d rib file(s) -> %d rib voxels", len(files), int(binary.sum()))
    return binary, affine, header


# ===========================================================================
# Step B — Connected Components & Filtering
# ===========================================================================
def label_and_filter_components(
    binary_ribs: np.ndarray, min_voxels: int = 500, connectivity: int = 26,
) -> Tuple[np.ndarray, List[int]]:
    """3D connected-component labelling of the rib mask, with small-blob removal.

    Each physically separate rib becomes one component. 26-connectivity (a full
    3x3x3 structuring element) keeps a thin, slightly-broken rib as a single
    component; ribs only rarely touch each other, and where they do the Step-D
    vote still resolves the number from the dominant vertebral overlap.

    Parameters
    ----------
    binary_ribs  : uint8/bool rib mask from Step A.
    min_voxels   : drop any component smaller than this (noise / TS speckle).
    connectivity : 6 (face) or 26 (face+edge+corner) neighbourhood.

    Returns
    -------
    (labeled, kept_labels)
        labeled     : int32 volume, each surviving component a distinct id (1..K).
        kept_labels : the component ids that passed the size filter.
    """
    structure = ndimage.generate_binary_structure(3, 1 if connectivity == 6 else 3)
    labeled, n = ndimage.label(binary_ribs, structure=structure)

    # Component sizes via a single bincount pass (index 0 = background).
    sizes = np.bincount(labeled.ravel())
    kept = [lab for lab in range(1, n + 1) if sizes[lab] >= min_voxels]
    dropped = n - len(kept)

    # Zero out the sub-threshold blobs so downstream code sees only real ribs.
    if dropped:
        keep_mask = np.isin(labeled, kept)
        labeled = np.where(keep_mask, labeled, 0).astype(np.int32)

    log.info("Step B: %d raw component(s); kept %d >= %d vox, dropped %d noise blob(s)",
             n, len(kept), min_voxels, dropped)
    return labeled, kept


# ===========================================================================
# Step C — Vertebral Dilation (memory efficient, per bounding box)
# ===========================================================================
def _ball(radius: int) -> np.ndarray:
    """Boolean 3D ball structuring element of the given voxel radius."""
    r = int(radius)
    zz, yy, xx = np.ogrid[-r:r + 1, -r:r + 1, -r:r + 1]
    return (zz * zz + yy * yy + xx * xx) <= r * r


def dilate_vertebrae_local(
    vert_data: np.ndarray, dilation_radius: int = 4, pad: int = 10,
) -> Dict[int, Tuple[Tuple[slice, slice, slice], np.ndarray]]:
    """Dilate each GT vertebra INSIDE its own padded bounding box (never globally).

    A full-volume 3D dilation of a 512^3 CT allocates several such volumes and
    OOMs. Instead, for each vertebra we crop to its bounding box, pad by `pad`
    voxels (so the dilation has room to grow without hitting the crop edge),
    dilate with a ball of radius `dilation_radius`, and remember the crop slices
    so the result can be read back in global coordinates without ever
    materialising a global dilated volume.

    Parameters
    ----------
    vert_data       : int GT label volume; voxel value N == thoracic vertebra T-N.
    dilation_radius : ball radius (vox); 3-5 reaches the costovertebral joint.
    pad             : bbox padding (vox); must exceed dilation_radius.

    Returns
    -------
    dict : {vertebra_label: (global_slices, dilated_submask_bool)}.
           global_slices indexes the original volume; dilated_submask_bool is the
           dilated vertebra cropped to those slices.
    """
    if pad <= dilation_radius:
        log.warning("pad (%d) <= dilation_radius (%d): dilation may be clipped at the "
                    "bbox edge; increase --pad", pad, dilation_radius)
    struct = _ball(dilation_radius)
    shape = vert_data.shape
    out: Dict[int, Tuple[Tuple[slice, slice, slice], np.ndarray]] = {}

    # find_objects returns, for label L, the bounding-box slice tuple at index L-1.
    labels = [int(v) for v in np.unique(vert_data) if v != 0]
    objects = ndimage.find_objects(vert_data.astype(np.int32))

    for lab in labels:
        loc = objects[lab - 1]
        if loc is None:                                   # label absent / gap in numbering
            continue
        # Pad the bbox by `pad`, clipped to the volume bounds.
        padded = tuple(
            slice(max(0, s.start - pad), min(dim, s.stop + pad))
            for s, dim in zip(loc, shape)
        )
        sub = vert_data[padded] == lab                    # this vertebra only, in the crop
        dilated = ndimage.binary_dilation(sub, structure=struct)
        out[lab] = (padded, dilated)
        log.debug("Step C: vertebra %d bbox=%s dilated to %d vox",
                  lab, tuple((s.start, s.stop) for s in padded), int(dilated.sum()))

    log.info("Step C: dilated %d GT vertebra(e) locally (radius=%d, pad=%d)",
             len(out), dilation_radius, pad)
    return out


# ===========================================================================
# Step D — Intersection, Voting, and L/R Assignment
# ===========================================================================
def _world_centroids(
    binary: np.ndarray, labeled: np.ndarray, labels: List[int], affine: np.ndarray,
) -> Dict[int, np.ndarray]:
    """World (RAS) centroid of each labelled region, in one pass.

    `ndimage.center_of_mass` returns voxel-index centroids in array-axis order
    (i, j, k); `apply_affine` maps those through the affine into nibabel's RAS+
    world frame, so the comparison in Step D is correct regardless of the file's
    stored orientation (PIR, LPS, …).
    """
    coms = ndimage.center_of_mass(binary, labels=labeled, index=labels)
    if len(labels) == 1:                                  # SciPy returns a bare tuple for one label
        coms = [coms]
    return {lab: apply_affine(affine, np.asarray(com)) for lab, com in zip(labels, coms)}


def _lr_world_axis(affine: np.ndarray) -> Tuple[int, float]:
    """Which world axis is patient Left<->Right, and which sign points Right.

    nibabel world space is RAS+, so the world X axis (index 0) runs toward the
    patient's RIGHT as it increases: a larger world-X means more to the right.
    Returning this explicitly keeps the L/R rule readable and affine-robust.
    """
    return 0, +1.0     # axis 0 = X; +X -> patient Right in RAS+


def assign_ribs(
    labeled_ribs: np.ndarray,
    kept_labels: List[int],
    vert_data: np.ndarray,
    vert_dilations: Dict[int, Tuple[Tuple[slice, slice, slice], np.ndarray]],
    affine: np.ndarray,
) -> Dict[int, Tuple[str, int]]:
    """Vote each rib component onto a vertebra (-> number) and decide its side.

    Voting (number)
        For every dilated vertebra, count how many of its voxels coincide with
        each rib component (a `bincount` over the rib labels inside the dilated
        crop). The vertebra that a component overlaps MOST gives that component
        its number. Done per-bbox, so no global dilated volume is needed.

    Left/Right
        Compare the rib component's world centroid to its anchoring vertebra's
        world centroid along the patient L<->R axis (RAS world X). A rib whose
        centroid is to the Right of the spine (greater world-X) is a right rib;
        to the Left, a left rib.

    Returns
    -------
    dict : {rib_component_label: (side, rib_number)} for every assignable rib.
           Components overlapping no vertebra are logged and omitted.
    """
    # overlap[comp] -> {vert_label: voxel_overlap}
    overlap: Dict[int, Dict[int, int]] = {c: {} for c in kept_labels}
    for vlab, (slices, dmask) in vert_dilations.items():
        sub_ribs = labeled_ribs[slices][dmask]            # rib labels under this vertebra
        if sub_ribs.size == 0:
            continue
        counts = np.bincount(sub_ribs.ravel())
        for comp in np.nonzero(counts)[0]:
            if comp == 0:
                continue
            overlap[int(comp)][vlab] = int(counts[comp])

    # Centroids (world RAS) for ribs and for the GT vertebrae, each in one pass.
    rib_centroids = _world_centroids(labeled_ribs > 0, labeled_ribs, kept_labels, affine)
    vert_labels = list(vert_dilations.keys())
    vert_centroids = _world_centroids(vert_data > 0, vert_data, vert_labels, affine)
    lr_axis, right_sign = _lr_world_axis(affine)

    assignments: Dict[int, Tuple[str, int]] = {}
    for comp in kept_labels:
        votes = overlap[comp]
        if not votes:
            log.warning("Component %d overlaps no GT vertebra (rib outside the labelled "
                        "FOV?) — left UNASSIGNED", comp)
            continue
        # Number = the most-overlapped vertebra.
        best_vert = max(votes, key=votes.get)
        # Side = which side of that vertebra the rib centroid sits on (RAS world X).
        # robust scalar extraction of the world L-R coordinate (centroids may come back
        # shaped (3,) or (1,3) depending on numpy/scipy version).
        dx = float(np.ravel(rib_centroids[comp])[lr_axis]
                   - np.ravel(vert_centroids[best_vert])[lr_axis])
        side = "right" if dx * right_sign > 0 else "left"
        assignments[comp] = (side, int(best_vert))
        log.info("Assigning Component %d to %s Rib %d  (overlap=%d vox, dx=%+.1f mm)",
                 comp, side.capitalize(), best_vert, votes[best_vert], dx)

    log.info("Step D: assigned %d/%d rib component(s)", len(assignments), len(kept_labels))
    return assignments


def assign_unassigned_by_nearest(
    labeled_ribs: np.ndarray,
    unassigned: List[int],
    vert_data: np.ndarray,
    vert_labels: List[int],
    affine: np.ndarray,
    *,
    span_tol_mm: float = 20.0,
) -> Dict[int, Tuple[str, int]]:
    """Fallback numbering for rib components the overlap vote (assign_ribs) left
    UNASSIGNED: number each by the NEAREST vertebra along the cranio-caudal (world
    Z) axis, side from world X.

    This recovers a REAL rib that doesn't reach within the dilation radius of any
    vertebra (e.g. a rib only one segmenter caught that stops short medially). It is
    CONSTRAINED to the cranio-caudal span of the labelled vertebrae (± span_tol_mm):
    a component above the topmost / below the bottommost labelled vertebra has no
    vertebra to inherit a number from, so it is left dropped rather than piling onto
    the end vertebra and creating a duplicate. Returns {comp:(side,number)} for the
    components it could number.
    """
    if not unassigned or not vert_labels:
        return {}
    rib_c = _world_centroids(labeled_ribs > 0, labeled_ribs, unassigned, affine)
    vert_c = _world_centroids(vert_data > 0, vert_data, vert_labels, affine)
    lr_axis, right_sign = _lr_world_axis(affine)
    zc = {v: float(np.ravel(vert_c[v])[2]) for v in vert_labels}
    zmin, zmax = min(zc.values()), max(zc.values())

    out: Dict[int, Tuple[str, int]] = {}
    for comp in unassigned:
        rc = np.ravel(rib_c[comp])
        if not (zmin - span_tol_mm <= rc[2] <= zmax + span_tol_mm):
            log.info("Component %d (z=%.0f mm) outside labelled vertebra span "
                     "[%.0f, %.0f] — cannot number, left dropped", comp, rc[2], zmin, zmax)
            continue
        best = min(vert_labels, key=lambda v: abs(rc[2] - zc[v]))
        dx = float(rc[lr_axis] - np.ravel(vert_c[best])[lr_axis])
        side = "right" if dx * right_sign > 0 else "left"
        out[comp] = (side, int(best))
        log.info("Fallback: Component %d -> %s Rib %d  (nearest by z, dz=%.0f mm)",
                 comp, side.capitalize(), best, abs(rc[2] - zc[best]))
    return out


# ===========================================================================
# Step E — Output Generation
# ===========================================================================
def build_output_volume(
    labeled_ribs: np.ndarray, assignments: Dict[int, Tuple[str, int]],
) -> np.ndarray:
    """Paint each assigned rib component with its final {side,number} label id.

    Unassigned components (no vertebral anchor) are dropped (stay 0). Output is
    uint16 to leave headroom for the 100+/200+ id scheme.
    """
    out = np.zeros(labeled_ribs.shape, dtype=np.uint16)
    for comp, (side, number) in assignments.items():
        out[labeled_ribs == comp] = rib_label_id(side, number)
    return out


def save_like(
    data: np.ndarray, affine: np.ndarray, header: nib.Nifti1Header, out_path: Path,
) -> None:
    """Write `data` as a NIfTI reusing the source affine + header (PHI-free)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = nib.Nifti1Image(data, affine, header)
    img.set_data_dtype(np.uint16)
    nib.save(img, str(out_path))
    log.info("Step E: wrote %s (%d labelled rib voxels)", out_path, int((data > 0).sum()))


# ===========================================================================
# Orchestration
# ===========================================================================
def relabel_ribs(
    rib_dir: Path,
    vertebrae_path: Path,
    out_path: Path,
    *,
    min_voxels: int = 500,
    dilation_radius: int = 4,
    pad: int = 10,
    connectivity: int = 26,
) -> Dict[int, Tuple[str, int]]:
    """Run the full A->E pipeline and write the relabelled rib volume."""
    binary, affine, header = load_and_unify_ribs(rib_dir)                       # A

    vert_img = nib.load(str(vertebrae_path))
    if vert_img.shape[:3] != binary.shape:
        raise ValueError(f"vertebrae shape {vert_img.shape[:3]} != ribs {binary.shape} "
                         "(inputs must be co-registered)")
    vert_data = np.asarray(vert_img.dataobj).astype(np.int32)

    labeled, kept = label_and_filter_components(binary, min_voxels, connectivity)  # B
    if not kept:
        log.warning("No rib components survived filtering — writing empty volume")
    vert_dilations = dilate_vertebrae_local(vert_data, dilation_radius, pad)       # C
    assignments = assign_ribs(labeled, kept, vert_data, vert_dilations, affine)    # D
    out = build_output_volume(labeled, assignments)                               # E
    save_like(out, affine, header, out_path)
    return assignments


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rib_dir", required=True, type=Path,
                    help="directory of TotalSegmentator rib *.nii.gz files")
    ap.add_argument("--vertebrae", required=True, type=Path,
                    help="GT thoracic vertebrae NIfTI (voxel value N = T-N)")
    ap.add_argument("--out", required=True, type=Path,
                    help="output relabelled rib NIfTI")
    ap.add_argument("--min_voxels", type=int, default=500,
                    help="drop rib components below this many voxels (default 500)")
    ap.add_argument("--dilation_radius", type=int, default=4,
                    help="vertebra dilation ball radius in voxels (default 4)")
    ap.add_argument("--pad", type=int, default=10,
                    help="bbox padding in voxels around each vertebra (default 10)")
    ap.add_argument("--connectivity", type=int, default=26, choices=[6, 26],
                    help="connected-component neighbourhood (default 26)")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = ap.parse_args()
    if args.verbose:
        log.setLevel(logging.DEBUG)

    relabel_ribs(args.rib_dir, args.vertebrae, args.out,
                 min_voxels=args.min_voxels, dilation_radius=args.dilation_radius,
                 pad=args.pad, connectivity=args.connectivity)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
