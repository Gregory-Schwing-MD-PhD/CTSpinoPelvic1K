# CTSpinoPelvic1K — Roadmap, Projects & Publication Deadlines

Single source of truth for project goals and submission targets. Dates verified
as of 2026-06-21; ✅ = confirmed from the venue, ⚠️ = extrapolated from the prior
year (verify on the live CFP before relying on it).

---

## Dataset versions

| Version | Contents | Status |
|---|---|---|
| **v1** | full release (all configs) | published (HF `@v1`) |
| **v2** | radiologist spine GT + pseudolabelled pelves (L1–L6, sacrum, hips, ignore) | published (HF `@v2`) |
| **v3** | v2 + **femurs (32/33) + carved S1 (29) + GT thoracic (8–19, 28)**; VerSe-native scheme; ribs reserved-but-empty | **rebuild in progress** — pipeline fixed (per-case `/tmp` cleanup + fail-fast + resume-on-ok); clean run then push to `@v3` **and** `@main` |
| **v4** (planned) | v3 + **ribs + LS nerves + iliolumbar ligament + VerSeFusion anomalies** | annotation phase (student tasks below) |

Label scheme + build logic: [`scripts/build_v3_totalseg.py`](../scripts/build_v3_totalseg.py).
Ship pipeline: [`slurm/ship_v3.sh`](../slurm/ship_v3.sh) (build → push `@v3` → promote `@main`).

---

## Projects

One project = one paper. Each paper's **aims** are concrete deliverables.

### Paper 1 — CTSpinoPelvic1K v3 (bone) dataset
*Deadline: **MLHC 2026** — in review, score 2.75/4 (conf Aug 12–14, 2026). Backups:
BIBM (no dual-submission while MLHC pending), ML4H Findings (non-archival), Scientific
Data (durable citation).*
- **Aim 1:** Build v3 — add femurs + carve S1 + GT thoracic onto v2.
- **Aim 2:** Baseline nnU-Net (pseudolabels L1–L5 + sacrum) + the reviewer-requested ablation.
- **Aim 3:** Publish v3 to HF (`@v3`, promote `@main`).

### Paper 2 — Spinopelvic parameters from CT masks (OpenSpineToolkit)
*Deadline: **BIBM 2026 — Jul 5** (full paper), or **SPIE Medical Imaging 2027 —
abstract Aug 5, 2026** (mss Jan 27, 2027). Independent of MLHC. Other homes: JOSS /
J. Imaging Informatics in Medicine / Operative Neurosurgery.*
Spec: [OpenSpineToolkit `SPEC.md`](https://github.com/Gregory-Schwing-MD-PhD/OpenSpineToolkit/blob/main/SPEC.md) (PI first — posture-invariant → valid on supine CT).
- **Aim 1:** Develop PI extraction code (femoral-head sphere-fit + S1 endplate + sagittal projection).
- **Aim 2:** Manually verify PI (MAE / ICC / Bland–Altman vs manual).
- **Aim 3:** Extend to LL + PI–LL mismatch (develop + verify).

### Paper 3 — Lumbosacral nerve segmentation & Kambin's-triangle mapping
*Deadline: **MICCAI 2027** (~Feb) / **MIDL 2027** (~Dec 2026); clinical: AJNR /
World Neurosurgery; ML4H 2027.* Precedent: Fan 2019 (SPINECT, CT, Dice 0.905);
clinical motivation: Tabarestani 2023 (MRI, percLIF).
- **Aim 1:** Annotate L4/L5/S1 nerve roots on v3 (student task + IRR).
- **Aim 2:** Train + validate a per-root nerve segmentation model (vs MRI subset).
- **Aim 3:** Derive Kambin's-triangle / foraminal geometry + LSTV neural enumeration
  (L5-nerve caliber/count at the lateral sacrum).

### Paper 4 — v4 dataset: ribs, iliolumbar ligament & LSTV/TLTV detection
*Deadline: **BIBM 2027** (~Jul) / **Scientific Data** (rolling, v4 descriptor) /
**MICCAI 2027**.* Three independent enumeration anchors: rostral bony (lowest
rib-bearing vertebra), caudal bony (S1 endplate + Castellvi morphology), neural
(L5-nerve caliber + iliolumbar ligament, which arises from the L5 TP).
- **Aim 1:** Segment ribs on v3 (student task; numbering read from GT thoracic).
- **Aim 2:** Segment the iliolumbar ligament (student task).
- **Aim 3:** LSTV/TLTV detection from the three anchors (rib-status ↔ GT-numbering
  mismatch flags candidates; GT stays authoritative) + VerSeFusion augmentation
  (~50 more LSTV/TLTV; n=33 → ~83, annotated in the v3 scheme); publish v4.

---

## Venue deadline matrix

| Venue | Deadline | Event | Type | For |
|---|---|---|---|---|
| **MLHC 2026** | ✅ Apr 17, 2026 (closed) | ✅ Aug 12–14, Baltimore | archival | P1 (in review) |
| **IEEE BIBM 2026** | ✅ **Jul 5, 2026** | ✅ Dec 1–4, Dallas | archival (IEEE Xplore) | P2; P1 fallback |
| **SPIE Medical Imaging 2027** | ✅ **Abstract 5 Aug 2026** · notif 26 Oct 2026 · mss 27 Jan 2027 | ✅ Feb 14–18, 2027, Vancouver | archival | P2; P1/P4; P3 (if early) |
| **ML4H 2026 (Findings)** | ⚠️ ~Sep 2026 (2025 = Sep 8) | ~Dec 2026 | **non-archival** | P1/P2 visibility |
| **MIDL 2027** | ⚠️ ~Dec 2026 (verify) | 2027 | archival | P3 |
| **MICCAI 2027** | ⚠️ ~Feb 2027 (verify) | 2027 | archival | P3, P4 |
| **Rolling journals** | none | — | journal | Scientific Data (P1/P4); J. Imaging Informatics in Medicine / JOSS / Operative Neurosurgery (P2); AJNR / World Neurosurgery (P3) |

**Confirmed PASSED (do not chase):** MICCAI 2026 (full mss Feb 26, 2026), MIDL 2026
(full Dec 2025 / short Apr 15, 2026).

**Near-term archival slots (two):**
1. **BIBM — Jul 5, 2026** (full paper). Best candidate = **P2 (toolbox)**: independent
   of MLHC, PI/LL pipeline is self-contained. P1 only if MLHC rejects first.
2. **SPIE Medical Imaging 2027 — abstract Aug 5, 2026** (only a **2–4 page PDF** at
   this stage; full manuscript not due until Jan 27, 2027 — a much lower near-term
   bar). Two on-target conferences: **MI102 Image Processing** (lists "open software
   for medical image processing" and "ground truth generation / validation" — fits
   P1 dataset *and* P2 toolbox) and **MI103 Computer-Aided Diagnosis** (clinical-AI /
   imaging-biomarker / musculoskeletal+neurology — fits P2's surgical-utility angle).
   Has a **student paper award** — good for the med-student contributors. Strong,
   low-risk near-term play for P2 (and a P1/P4 abstract).

---

## Student annotation tasks (v4) — see `docs/annotation/`

| Task | Difficulty | Output | Anchor role |
|---|---|---|---|
| **Ribs** | easier (TS-assisted; value-add = numbering + partials) | rib masks 34–57 | rostral enumeration |
| **Iliolumbar ligament** | moderate (CT-friendly; sometimes ossified) | ligament mask | LSTV cross-check |
| **LS nerves** | hard (per-root instance; CT contrast-limited) | per-root nerve instances | Kambin's + neural enumeration |

All tasks: **AI-assisted** annotation + **dual-assignment IRR** (per-class Dice,
adjudication on disagreement) via the `review_service/` pipeline. Reference standard
needs an **expert adjudicator**; validate nerves vs MRI where paired imaging exists.

---

## Sequencing principle

Publish the **dataset once** (v3 → MLHC/Scientific Data); model improvements are
**separate** method/benchmark papers that *cite* it (not re-publications). A new
dataset paper is justified when the data **materially expands** — i.e. **v4
(+ribs +nerves +iliolumbar +VerSeFusion)** is the next dataset paper.
