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
| **v3** | v2 + **femurs (11/12) + carved S1 (7) + GT thoracic (13–25)**; reordered scheme; ribs reserved-but-empty | **rebuild in progress** — pipeline fixed (per-case `/tmp` cleanup + fail-fast + resume-on-ok); clean run then push to `@v3` **and** `@main` |
| **v4** (planned) | v3 + **ribs + LS nerves + iliolumbar ligament + VerSeFusion anomalies** | annotation phase (student tasks below) |

Label scheme + build logic: [`scripts/build_v3_totalseg.py`](../scripts/build_v3_totalseg.py).
Ship pipeline: [`slurm/ship_v3.sh`](../slurm/ship_v3.sh) (build → push `@v3` → promote `@main`).

---

## Projects

### P1 — CTSpinoPelvic1K v3 dataset paper
v3 (femurs + S1 + thoracic) with the **baseline** (pseudolabels L1–L5 + sacrum) and
the reviewer-requested **ablation**.
- **Primary home:** MLHC 2026 — *in review, score 2.75/4, awaiting decision.*
- **Fallback (archival):** BIBM 2026 — only if MLHC rejects before its deadline;
  **no dual-submission while MLHC is pending.**
- **Visibility (non-archival, no conflict):** ML4H 2026 Findings.
- **Durable citation:** Scientific Data (rolling) — complements the conference paper.

### P2 — OpenSpineToolbox: spinopelvic parameters from masks
Automated PI / LL / PI–LL (+ Kambin/foraminal geometry) from v3 masks, validated
vs. manual. Spec: [OpenSpineToolbox `SPEC.md`](https://github.com/Gregory-Schwing-MD-PhD/OpenSpineToolbox/blob/main/SPEC.md).
Build order = clinical utility, **PI first** (posture-invariant → valid on supine CT).
- **Independent of MLHC** (no dual-submission conflict).
- **Routes:** BIBM 2026 (if PI/LL validation lands by the deadline) · SPIE Medical
  Imaging 2027 · JOSS (software) / J. Imaging Informatics in Medicine / Operative
  Neurosurgery (clinical-utility angle).

### P3 — LS-nerve segmentation (v4)
Per-root **instance** segmentation on CT (caliber + branching). Dual payload:
**Kambin's-triangle/foraminal mapping** *and* **LSTV neural enumeration** (L5-nerve
caliber/count at the lateral sacrum). Precedent: Fan 2019 (SPINECT, CT, Dice 0.905);
clinical motivation: Tabarestani 2023 (MRI, percLIF).
- **Routes:** MICCAI 2027 · MIDL 2027 · AJNR / World Neurosurgery · ML4H 2027.

### P4 — Ribs + iliolumbar ligament + LSTV/TLTV (v4 expansion)
Three independent **enumeration anchors** for transitional anatomy:
1. **Rostral bony** — lowest rib-bearing vertebra (rib mask).
2. **Caudal bony** — S1 endplate + transitional-body morphology (Castellvi).
3. **Neural** — L5-nerve caliber/count + **iliolumbar ligament** (arises from L5 TP).

Rib-status ↔ GT-numbering mismatch auto-flags candidate TLTV/LSTV (a feature/flag,
not a numbering authority — GT stays authoritative). VerSeFusion contributes ~50
additional LSTV/TLTV (current LSTV n=33 → ~83) — **must be annotated in the v3 label
scheme** to pool for training.
- **Routes:** Scientific Data (v4 descriptor) · MICCAI 2027 · BIBM 2027.

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
| **Ribs** | easier (TS-assisted; value-add = numbering + partials) | rib masks 26–49 | rostral enumeration |
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
