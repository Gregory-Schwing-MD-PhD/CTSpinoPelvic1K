"""
scripts/auto_adjudicate.py — auto-merge two reviewers' segmentations into a high-quality final,
git-3-way-merge style, and surface ONLY the irreconcilable conflicts for manual adjudication.

Both reviewers corrected the SAME pseudolabel P, so this is a true 3-way merge
(base = P, ours = A, theirs = B) — which auto-resolves the large majority of "disagreements",
because most are "one reviewer fixed a structure, the other left the base untouched".

Layered resolution (each layer resolves more, leaving fewer true conflicts):

  L0  3-way voxel merge
        A == B                         -> agree            -> take it
        A != P and B == P              -> only A edited     -> take A
        B != P and A == P              -> only B edited     -> take B
        A != B and A != P and B != P   -> REAL conflict     -> layers below

  L1  present-vs-absent conflicts (one says a bone class, the other says background):
        let the CT referee it via HU — solid bone -> keep the label, soft tissue -> background,
        mid-range -> leave unresolved. This also mops up rib-surface boundary jitter for free.

  L2  class-vs-class conflicts (both say a DIFFERENT non-background class on the same voxel —
        rib-numbering / bone-class mixing): the riskiest calls. Default = leave for MANUAL. Optional
        reliability tie-break (the more accurate reviewer wins) when the weight gap is large.
        [Hook: a T12/costovertebral anchor rule can resolve rib-numbering conflicts geometrically.]

  L3  conservative speck cleanup, then the anatomy QC gate (the same one students pass).

Decision: no residual conflict AND QC clean  -> AUTO-FINALIZE (no human needed).
          otherwise                          -> emit a conflict mask; a human paints ONLY those blobs.

Pure core (numpy in, numpy out) so it is testable and reusable; a CLI wraps it for nnii files.
The per-voxel disagreement margin is also a natural UNCERTAINTY MAP — usable as soft training
targets (don't force a hard label where raters genuinely disagree).
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

_HERE = Path(__file__).resolve().parent
for _p in (_HERE, _HERE / "review"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# HU thresholds for the CT referee (cortical/trabecular bone vs soft tissue)
HU_BONE = 200.0        # >= this -> clearly bone -> keep the reviewer's bone label
HU_SOFT = 100.0        # <  this -> clearly not bone -> background
SPECK_MIN_VOX = 40     # drop isolated components smaller than this in the merged label
WEIGHT_GAP = 1.5       # class-vs-class auto-resolves by reliability only if wA/wB (or inverse) >= this


def three_way_merge(P: np.ndarray, A: np.ndarray, B: np.ndarray
                    ) -> Tuple[np.ndarray, np.ndarray, dict]:
    """L0. Returns (merged, conflict_mask, stats). `merged` holds every voxel the two reviewers
    agree on or that only one of them edited; `conflict_mask` marks voxels both edited differently
    (left as base P in `merged` until a later layer resolves them)."""
    P = np.asarray(P); A = np.asarray(A); B = np.asarray(B)
    if not (P.shape == A.shape == B.shape):
        raise ValueError(f"shape mismatch: P{P.shape} A{A.shape} B{B.shape}")
    merged = P.copy()
    agree = A == B                              # both-unchanged OR both-same-edit
    only_a = (A != B) & (B == P)               # A changed, B kept base  (=> A != P)
    only_b = (A != B) & (A == P)               # B changed, A kept base
    conflict = (A != B) & (A != P) & (B != P)  # both changed, disagree
    merged[agree] = A[agree]
    merged[only_a] = A[only_a]
    merged[only_b] = B[only_b]
    stats = {"agree": int(agree.sum()), "only_a": int(only_a.sum()),
             "only_b": int(only_b.sum()), "conflict_l0": int(conflict.sum())}
    return merged, conflict, stats


def resolve_present_absent(merged: np.ndarray, conflict: np.ndarray,
                           A: np.ndarray, B: np.ndarray, ct: np.ndarray,
                           hu_bone: float = HU_BONE, hu_soft: float = HU_SOFT
                           ) -> Tuple[np.ndarray, dict]:
    """L1. Conflicts where exactly one reviewer says background: let HU decide. Returns the updated
    conflict mask (present/absent voxels that got resolved are cleared) and a stats dict. `merged`
    is edited in place."""
    A = np.asarray(A); B = np.asarray(B); ct = np.asarray(ct)
    one_bg = conflict & ((A == 0) ^ (B == 0))          # exactly one says background
    cand = np.where(A == 0, B, A)                       # the non-background candidate label
    take_label = one_bg & (ct >= hu_bone)              # clearly bone -> keep the label
    take_bg = one_bg & (ct < hu_soft)                  # clearly soft tissue -> background
    merged[take_label] = cand[take_label]
    merged[take_bg] = 0
    resolved = take_label | take_bg
    conflict = conflict & ~resolved                     # mid-HU present/absent stays unresolved
    return conflict, {"hu_kept_label": int(take_label.sum()),
                      "hu_to_background": int(take_bg.sum())}


def resolve_class_conflicts(merged: np.ndarray, conflict: np.ndarray,
                            A: np.ndarray, B: np.ndarray,
                            w_a: float = 1.0, w_b: float = 1.0,
                            weight_gap: float = WEIGHT_GAP
                            ) -> Tuple[np.ndarray, dict]:
    """L2. Conflicts where both reviewers assigned a DIFFERENT non-background class (rib numbering /
    bone-class mixing). Conservative by default: only auto-resolve when one reviewer is clearly more
    reliable (weight gap >= `weight_gap`); otherwise leave for MANUAL. `merged` edited in place."""
    both_fg = conflict & (A != 0) & (B != 0)
    n_cc = int(both_fg.sum())
    ratio = (w_a / w_b) if w_b else float("inf")
    if n_cc and (ratio >= weight_gap or (w_a and ratio <= 1.0 / weight_gap)):
        winner = A if w_a >= w_b else B
        merged[both_fg] = winner[both_fg]
        conflict = conflict & ~both_fg
        return conflict, {"class_conflict": n_cc, "class_resolved_by_weight": n_cc,
                          "winner": "A" if w_a >= w_b else "B"}
    return conflict, {"class_conflict": n_cc, "class_resolved_by_weight": 0}


def cleanup_specks(merged: np.ndarray, min_vox: int = SPECK_MIN_VOX) -> Tuple[np.ndarray, int]:
    """L3a. Drop isolated components smaller than `min_vox` (per label value) — conservative noise
    removal after the merge. Elongated ribs legitimately fragment across the FOV, so the threshold
    is deliberately small. Returns (cleaned, n_voxels_dropped)."""
    from scipy import ndimage
    out = merged.copy()
    dropped = 0
    # ONE pass: find_objects gives the bbox of every label value; label + bincount only within each
    # small crop (never re-scans the full volume per label).
    slices = ndimage.find_objects(merged)
    for i, sl in enumerate(slices):
        v = i + 1
        if sl is None:
            continue
        sub = merged[sl] == v
        lab, n = ndimage.label(sub)
        if n <= 1:
            continue
        counts = np.bincount(lab.ravel())
        small = np.nonzero(counts < min_vox)[0]
        small = small[small != 0]                    # never drop the background component
        if small.size:
            drop = np.isin(lab, small)
            out[sl][drop] = 0
            dropped += int(drop.sum())
    return out, dropped


def auto_adjudicate(P: np.ndarray, A: np.ndarray, B: np.ndarray, ct: np.ndarray,
                    affine: np.ndarray, *, w_a: float = 1.0, w_b: float = 1.0,
                    check: str = "ribs", resolve_class_by_weight: bool = False,
                    min_vox: int = SPECK_MIN_VOX) -> dict:
    """Orchestrate the layered merge and decide auto-finalize vs. needs-manual.

    Returns a dict with:
      final          - the merged label (np.ndarray)
      conflict_mask  - bool array of voxels left for MANUAL review (empty => nothing to do)
      decision       - "auto_finalize" | "needs_manual"
      qc_ok, qc_msgs - the anatomy QC gate result on `final`
      irr            - inter-rater agreement (from diff.irr)
      stats          - per-layer voxel counts (how much each layer resolved)
    """
    import review_anatomy_qc as RA
    from review import diff

    merged, conflict, stats = three_way_merge(P, A, B)
    conflict, s1 = resolve_present_absent(merged, conflict, A, B, ct)
    stats.update(s1)
    if resolve_class_by_weight:
        conflict, s2 = resolve_class_conflicts(merged, conflict, A, B, w_a, w_b)
    else:
        both_fg = conflict & (np.asarray(A) != 0) & (np.asarray(B) != 0)
        s2 = {"class_conflict": int(both_fg.sum()), "class_resolved_by_weight": 0}
    stats.update(s2)
    merged, dropped = cleanup_specks(merged, min_vox=min_vox)
    stats["speck_voxels_dropped"] = dropped
    stats["residual_conflict"] = int(conflict.sum())

    qc_ok, qc_msgs = RA.check_label(check, merged, affine, gating_only=True)
    irr = diff.irr(A, B)
    decision = "auto_finalize" if (conflict.sum() == 0 and qc_ok) else "needs_manual"
    return {"final": merged, "conflict_mask": conflict, "decision": decision,
            "qc_ok": bool(qc_ok), "qc_msgs": qc_msgs, "irr": irr, "stats": stats}


def reviewer_weight(passed: int, total: int, prior: float = 1.0) -> float:
    """A simple reliability weight from a reviewer's QC pass history (Laplace-smoothed pass-rate).
    Feed to auto_adjudicate as w_a/w_b so the more accurate reviewer breaks class-vs-class ties."""
    return prior * (passed + 1.0) / (total + 2.0)


# ── CLI ───────────────────────────────────────────────────────────────────────

def _load(p):
    import nibabel as nib
    img = nib.load(str(p))
    return np.asanyarray(img.dataobj), img.affine, img.header


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(
        description="Auto-merge two reviewers' segmentations; emit final + conflict mask.")
    ap.add_argument("pseudo", help="the shared base pseudolabel (nii/nii.gz)")
    ap.add_argument("reviewer_a", help="reviewer 1 corrected label")
    ap.add_argument("reviewer_b", help="reviewer 2 corrected label")
    ap.add_argument("ct", help="the CT image (for the HU referee)")
    ap.add_argument("--out", default="final_auto.nii.gz", help="output merged label")
    ap.add_argument("--conflict-out", default="conflict_mask.nii.gz")
    ap.add_argument("--check", default="ribs", choices=["ribs", "spine", "both", "none"])
    ap.add_argument("--wa", type=float, default=1.0, help="reviewer A reliability weight")
    ap.add_argument("--wb", type=float, default=1.0, help="reviewer B reliability weight")
    ap.add_argument("--resolve-class-by-weight", action="store_true",
                    help="auto-resolve class-vs-class conflicts by reliability (default: manual)")
    a = ap.parse_args(argv)

    import nibabel as nib
    P, aff, hdr = _load(a.pseudo)
    A, _, _ = _load(a.reviewer_a)
    B, _, _ = _load(a.reviewer_b)
    ct, _, _ = _load(a.ct)
    r = auto_adjudicate(P, A, B, ct, aff, w_a=a.wa, w_b=a.wb, check=a.check,
                        resolve_class_by_weight=a.resolve_class_by_weight)

    nib.save(nib.Nifti1Image(r["final"].astype(P.dtype), aff, hdr), a.out)
    print(f"DECISION: {r['decision'].upper()}   QC {'OK' if r['qc_ok'] else 'FAIL'}   "
          f"IRR agree={r['irr'].get('agree')}")
    s = r["stats"]
    print(f"  L0  agree={s['agree']}  onlyA={s['only_a']}  onlyB={s['only_b']}  "
          f"conflicts={s['conflict_l0']}")
    print(f"  L1  HU kept-label={s['hu_kept_label']}  HU->bg={s['hu_to_background']}")
    print(f"  L2  class-vs-class={s['class_conflict']}  by-weight={s['class_resolved_by_weight']}")
    print(f"  L3  specks dropped={s['speck_voxels_dropped']}  RESIDUAL CONFLICT={s['residual_conflict']}")
    if r["conflict_mask"].any():
        nib.save(nib.Nifti1Image(r["conflict_mask"].astype(np.uint8), aff, hdr), a.conflict_out)
        print(f"  -> wrote conflict mask to {a.conflict_out}: paint ONLY these blobs, then finalize.")
        for m in r["qc_msgs"][:8]:
            print(f"     {m}")
    else:
        print("  -> no residual conflict: this case can be AUTO-FINALIZED (no human needed).")
    print(f"  final -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
