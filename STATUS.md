# Project status / lab notebook

## 2026-06-11

### Done today

**Fused-only ablation (reviewer "fused-only vs all-cases" deliverable)**
- `spinesurg-ct-nnunet` (branch `L6_Class`): added `--include_configs` +
  `--drop_ignore_label` to `convert_hf_to_nnunet.py`; taught `spine_prep.sh` to
  pass them and accept the no-ignore scheme; parameterized `spine_eval_single.sh`.
  Runbook: `slurm/README_fused_only_ablation.md`.
- Built `Dataset804_SpineSurgCTFusedOnly` (290 train / 52 test fused, 9-key
  no-ignore). **Fold 0 training in flight** on the H200.
- Note: fused subset is LSTV-poor (7 non-normal total) — expected; the partial
  cases carried most of the LSTV signal.

**v3 has_l6 metadata fix (final manifest)**
- Found a real drift: `reduce_to_v3` / `refresh_hf_manifests` never recompute
  `has_l6`/`n_lumbar_labels` from corrected labels, and the v3 splits is copied
  from v2 → a future re-split would mislabel pseudolabel-derived L6s.
- `scripts/refresh_lstv_from_labels.py` (+ `slurm/refresh_lstv_v3.sh`):
  parallel, spine-authoritative recompute. Trusts the spine-bearing record's v3
  label (reviewer corrections included), neutralises pelvic_native pseudolabels,
  and surfaces a REVIEW QUEUE (3 pelvic-only + 7 conflicts) + 1 genuine new L6
  (token 103, student-corrected). Writes **only** the v3 tree.
- To produce the final manifest: `WRITE=1 RESPLIT=1 sbatch slurm/refresh_lstv_v3.sh`
  (add `KEEP_PELVIC=...` for confirmed pelvic-only after review).

**LSTV resident-review packet** (handed to rads resident)
- 11 flagged cases (1 student-correction, 3 pelvic-only, 7 conflicts) bundled
  with CT+label, `L6_review_form.csv` (presence) and `castellvi_addendum.csv`
  (Castellvi, in the `_lstv_phenotypes.csv` schema for clean merge-back).

**Labeling conventions decided (documented)**
- `docs/LABELING_GUIDE.md`: classes, LSTV scope, L5/L6 merge rationale, and the
  **one-vs-two-class rule** — disc present → two classes (last_lumbar + sacrum,
  even with a fused TP bridge); disc obliterated → one sacrum class. Bony
  continuity ≠ identity (ankylosis principle).
- `docs/RIB_ANCHOR_REVIEW_GUIDE.md`: segment the **last rib-bearing vertebra +
  rib** as the counting anchor (top anchor + sacrum = deterministic L5/L6 count
  in limited FOV). AI-assisted (ITK-SNAP DLS / nnInteractive).

**Synthesis repo `CTSpinoPelvic1K-Syn`** (new, local-only — no remote yet)
- Scaffolded: Castellvi taxonomy, SYNTH_ tagging + split-leak guard, Pass-1
  grading worklist, `fuse_split` (deterministic L6/sacrum boundary splitter),
  tests. **The procedural generation engine is NOT built yet** (stubs).

**Analysis tools**
- `scripts/analyze_lstv_zpos.py`: Z-position / CoordConv-feasibility analysis
  (shows the bottom vertebra's box-position can't separate L5 from L6).

### Direction (decided end of 2026-06-11)
Pivot to a **single end-to-end model** that segments AND numbers the spine from
the **last rib-bearing vertebra through the sacrum** — L1–L6 as distinct classes
(no merge), all Castellvi classes, counting anomalies (lumbar rib / T13 /
sacralization / lumbarization). The enabler is the **counting anchor**: annotate
the last rib-bearing vertebra + its rib on **all ~1000 cases** (top anchor) +
the sacrum (bottom anchor) → counting between them makes numbering deterministic.
New classes: `last_rib_vertebra` (11), `rib` (12). The anchor is *relational*
("last rib-bearing vertebra"), never an absolute number (T12/T13 unknowable in a
lumbosacral FOV). Docs written: `docs/STUDENT_ANNOTATION_PROTOCOL.md` (master),
`docs/LABELING_GUIDE.md`, `docs/RIB_ANCHOR_REVIEW_GUIDE.md`.

### Open / next
1. Wait for fused-only fold-0 → eval on fused test split (`spine_eval_single.sh`,
   `N_CLASSES=9`) + the all-cases-803-on-fused-test comparison arm.
2. Run the final v3 manifest (`refresh_lstv_v3.sh WRITE=1 RESPLIT=1`); fold in
   resident verdicts (KEEP_PELVIC / spine-label corrections) when they return.
3. **Build the `CTSpinoPelvic1K-Syn` procedural engine** — Castellvi-graded
   synthetic fused cases (the lever for the bad LSTV Dice). Port the old MONAI
   `CopyPasteLSTVTransform` + `SacralizationSynthTransform`; wire `fuse_split`
   for boundaries.
4. Train next model: real + synthetic, validate on synthetic / test on real
   (per Martinson — synthetic for val, real for train+test).
5. **Extend `review_service` for the rib-anchor segmentation task** so OSC
   members can request/claim cases through the Space (claim -> segment -> submit)
   instead of manual Drive/email hand-offs. Reuse the existing claim/slot/submit
   machinery; swap the 31-landmark template for "segment last_thoracic + rib"
   (see `docs/RIB_ANCHOR_REVIEW_GUIDE.md`). Distribution exists today only for the
   LSTV landmark/Castellvi review, not for dense-segmentation tasks. Promised as
   "upcoming" in the OSC/BIBM email.
