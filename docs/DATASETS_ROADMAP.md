# External CT Datasets — Unification Roadmap

**Goal:** assemble a single, **VerSe-native, quality-tiered** multi-dataset CT corpus covering the
full musculoskeletal target set — **vertebrae (C1→sacrum, incl. full thoracic), ribs (1–12 L/R),
sacrum, S1, hips, femurs** — for large-scale training and rigorous external testing.

**The one contract everything maps to:** `scripts/label_scheme.py` (VerSe-native). Every external
dataset gets a converter that (a) remaps its native ids → our canonical ids, and (b) sets every
structure it does **not** label to **ignore (255)** — never background — so partial-label sources
can be co-trained without poisoning classes (same contract as `export_hf` / `pseudolabel`).

> Source-of-truth principle: when datasets overlap, the **highest-quality** source wins for that
> structure; others contribute only their strong structures and `ignore` the rest. Expert GT is
> reserved for **test**; model-assisted labels are **train-only diversity**.

---

## Inventory

| Dataset | Structures | Cases | Label quality | License | Role | Status |
|---|---|---|---|---|---|---|
| **CTSpinoPelvic1K** (ours) | spine (VerSe) + S1/sacrum/hips/femurs + ribs | ~1.1k | mixed (GT spine, model pelvis) | CC-BY-NC-4.0 | **hub** | live (v1+) |
| **VerSe '19/'20** | vertebrae C1–L6 + sacrum (whole-spine) | ~374 | numbering canonical, **seg quality variable → QC** | CC-BY-SA-4.0 | **full-thoracic spine** | to integrate |
| **CTSpine1K** | vertebrae (25 types, C1–L6) | 1005 | aggregated/semi-auto | CC-BY (mixed src) | spine breadth | partial (have) |
| **CTPelvic1K** | sacrum, L/R hip, lumbar | ~1.1k (7 src) | good (pelvis) | CC-BY-NC | sacrum/hip GT | have |
| **RibSeg v2 / RibFrac** | ribs 1–24 (+binary), centerlines | 660 | **expert-corrected** | RibFrac CC-BY-NC / code Apache | **ribs (best)** | future |
| **TotalSegmentator** | 104 structures (full body) incl. vert/ribs/hip/femur | 1204 | 5 manual → iterative pseudo+review | **CC-BY-4.0** | **full-body diversity**, femur/hip/sacrum | to integrate (selective) |
| **CADS** | whole-body (aggregates TS + others) | large | derived | mixed | breadth/diversity | evaluate |
| **VertebralBodiesCT-Labels** | vertebral *bodies* (T/L + sacrum) | 1460 | derived (TS+VerSe) | mixed | bodies-only aux | optional |
| **Thoracolumbar stump-rib cohort** ([arXiv 2505.05004](https://arxiv.org/abs/2505.05004)) | ribs + **stump/transitional rib** morphology | "large" | DL + measured | weights+masks public | **LSTV / transitional-anatomy hook** | evaluate |

---

## Per-dataset notes

### CTSpinoPelvic1K (hub)
The canonical scheme lives here (`label_scheme.py`). Everything else is harmonized *to* it. FOV-limited
spinopelvic; supplies the deadline (FOV-limited rib + spine) work directly.

### VerSe — the full-thoracic spine source (with a caveat)
Whole-spine CTs with **per-vertebra numbering incl. full T1–T12** — the thing CTSpinoPelvic1K lacks
(FOV-limited). **Caveat (hands-on): VerSe segmentation quality is uneven** — mislabels / boundary
issues appear case to case, so it is *not* drop-in GT. Use it as the **numbering** authority + a
full-thoracic spine source, but gate it through a **QC pass** before trusting per-voxel masks. Pairs
with RibSeg to un-gate full-cage (vertebra+rib) work.

### CTSpine1K / CTPelvic1K (already in the pipeline)
Spine breadth + pelvis GT. Already consumed by VerseFusion / the fusion pipeline.

### RibSeg v2 / RibFrac — best ribs
Expert-corrected rib labeling (1–24) on 660 full chest-abdomen CTs. Fold in as **ribs→34–57, all else
ignore** (HU split: bone≥200 not-rib → ignore, <200 → background). No spine needed on these cases.
Note: full-cage, top-down numbering = anatomically correct *on full cages*; for OUR FOV-limited data,
number via the **T12/costovertebral anchor** instead (see `relabel_ribs`). No pretrained weights are
published — train on the RibSeg v2 dataset (H200s).

### TotalSegmentator — the full-body diversity engine (use selectively)
1204 routine CTs, 104 structures, **CC-BY-4.0** (cleanest license here). Annotation: **5 manual →
iterative nnU-Net pseudolabel + manual review** (retrain @5/20/100; all 1204 reviewed). **Correction:**
the *dataset* labels are human-reviewed GT (and community-vetted) — *usable for all structures incl.
spine/ribs*; it's the released *tool's inference* that degrades on OOD / FOV-limited scans (a separate
problem). So treat TS as legit GT, **spot-check vertebra numbering** on a sample, expect a small
residual error rate, and keep a hand-checked subset for test. Strong structures (femur/hip/sacrum) +
**domain diversity** (1204 routine CTs) are its main value. meta.csv = `image_id, age, gender,
institute, study_type, split` (no anomaly fields → derive geometrically).

### CADS / VertebralBodiesCT-Labels — derived aggregates
Whole-body / vertebral-bodies, derived from TS+VerSe. Useful for breadth but inherit source quality;
treat as train-only diversity, never test GT.

### Thoracolumbar stump-rib cohort — the transitional-anatomy hook
Stump ribs are the radiographic tell for **thoracolumbar transitional vertebrae / enumeration
anomalies** — i.e., the exact miscount mechanism behind **wrong-level surgery**, and a direct tie to
our **LSTV / Castellvi** work. Reports stump ribs articulate more posteriorly, thinner, different
angle; F1 0.84 stump-vs-regular, 98.2% length-assessment success; **publishes weights + masks**. Use
as: (a) a transitional-rib **evaluation hook** for the level-localization story, (b) extra rib
morphology supervision, (c) a citation motivating bottom-anchored (not top-down) rib numbering.

---

## Harmonization principles

1. **One scheme:** every dataset → `label_scheme.label_dict()` ids via a per-dataset converter
   (`scripts/convert_<dataset>.py`), mirroring the RibSeg→nnU-Net converter design.
2. **Partial labels = ignore, not background:** any unlabeled-but-present structure → 255.
3. **Source-of-truth per structure:** spine numbering → VerSe/CTSpine1K (QC'd); ribs → RibSeg +
   T12-anchored numbering; sacrum/hip → CTPelvic1K; femur/hip/diversity → TotalSegmentator.
4. **Quality tiers:** `expert` (VerSe-QC'd, RibSeg, CTPelvic1K) → eligible for **test/GT**;
   `model-assisted` (TS, CADS, derived) → **train-only diversity**, flagged in the manifest.
5. **Train-time contiguity:** the dataset stays VerSe-native (non-contiguous); the nnU-Net prep
   applies the reversible 0..N squeeze (already designed).
6. **QC gate:** every imported tree passes `smoke_test_hf.py`-style checks (no stray ids, structures
   at expected ids, anatomical sanity) before it joins the training pool.
7. **Licensing:** the union is **CC-BY-NC** (most restrictive member); keep a per-source license +
   attribution ledger. TS is CC-BY-4.0; RibFrac/CTPelvic1K NC; VerSe CC-BY-SA.

---

## Phased plan

**Phase 0 — deadline (FOV-limited, own data only):** CTSpinoPelvic1K spine + pelvis + T12-anchored
ribs → DRR → XR baseline. *No external data required.*

**Phase 1 — ribs + diversity:** fold **RibSeg** (ribs) and **TotalSegmentator** (femur/hip/sacrum +
diversity) via converters with ignore-the-rest. Train one nnU-Net on the partial-label union.

**Phase 2 — full thoracic spine:** QC + integrate **VerSe** whole-spine for full T1–T12 numbering;
combined with RibSeg → full-cage (vertebra+rib) labeling beyond FOV-limited.

**Phase 3 — transitional anatomy:** add the **stump-rib cohort** as a transitional/LSTV evaluation
track + rib-morphology supervision; report level-localization on transitional cases (the wrong-level
surgery story).

**Phase 4 — unified mega-corpus:** all sources harmonized to VerSe-native, quality-tiered, with a
public crosswalk + license ledger — for massive training and external testing.

---

## North star — the anomaly cohort
Build the first large, auto-phenotyped, multi-source **spinal-anomaly cohort** (LSTV, TLTV/stump,
enumeration T13/L6/≠5-lumbar, scoliosis, deformity, rib fracture) by pooling all sources (~4–5k CTs),
unifying to VerSe-native labels, and running geometric/counting **detectors** to mine each phenotype at
scale. Train on noisy auto-mined labels; **hand-verify a few-hundred-case TEST split**. Detectors mostly
exist already: Castellvi+lstv (your manifest), counting (VerSe centroid JSON), stump-rib (Möller), rib
fracture (RibFrac), Cobb/spinopelvic (ostk). Clinical thread = **miscount → wrong-level surgery**.
Caveat: pooled prevalence is selection-biased (trauma/pathology-enriched sources) — report per-source.

## Densify by provenance (not by faith)
Partial-label `ignore` is the safe default. Densify a missing structure with pseudolabels ONLY where a
trustworthy model exists, and: **tag provenance per-structure-per-case** (extend `prov_spine`/`prov_pelvis`),
weight expert > pseudo, **confidence-gate** (keep `ignore` where unsure), **iterate** (self-train), and
**never** pseudolabel the test split. Cross-pseudolabel single-structure sets (VerSe←ribs/pelvis,
RibSeg←spine) using models trained on the sets that have them. TS is already full — no densify needed.

## Verified links + anomaly-metadata crosswalk

| Dataset | Paper | Code | Data | License | Anomaly signal already present |
|---|---|---|---|---|---|
| CTSpinoPelvic1K (hub) | — | this repo | HF `anonymous-mlhc/CTSpinoPelvic1K` | CC-BY-NC | **LSTV/Castellvi per case** (manifest: `lstv_class/label/pelvic/vertebral`, `castellvi_*`, `has_l6`, `n_lumbar_labels`) |
| VerSe '19/'20 | [2001.09193](https://arxiv.org/abs/2001.09193) | [anjany/verse](https://github.com/anjany/verse) | [OSF nqjyw](https://osf.io/nqjyw/) / [t98fz](https://osf.io/t98fz/) | CC-BY-SA-4.0 | transitional n≈161, enum n≈77 (47 L6, 6 T13) → flag label 25/28 in `*_ctd.json` |
| CTSpine1K | [2105.14711](https://arxiv.org/abs/2105.14711) | [MIRACLE-Center/CTSpine1K](https://github.com/MIRACLE-Center/CTSpine1K) | HF `alexanderdann/CTSpine1K` | CC-BY (mixed) | `Pathology` col (COLONOG xlsx); counts from 25-class |
| CTPelvic1K | [2012.08721](https://arxiv.org/abs/2012.08721) | [MIRACLE-Center/CTPelvic1K](https://github.com/MIRACLE-Center/CTPelvic1K) | HF `Angelou0516/ctpelvic1k` | CC-BY-NC | metal cases in filenames (`*_CLINIC_metal_*`) |
| RibFrac | [2402.09372](https://arxiv.org/abs/2402.09372) | [FracNet](https://m3dv.github.io/FracNet/) | [grand-challenge](https://ribfrac.grand-challenge.org/) · [Zenodo 3893508](https://zenodo.org/records/3893508)/[3993380](https://zenodo.org/records/3993380) | CC-BY-NC-4.0 | **fracture 4-type labels** (~5k: buckle/nondisp/disp/segmental) |
| RibSeg v2 | [2210.09309](https://arxiv.org/abs/2210.09309) | [HINTLab/RibSeg @ribsegv2](https://github.com/HINTLab/RibSeg/tree/ribsegv2) | [Drive](https://drive.google.com/file/d/1ZZGGrhd0y1fLyOZGo_Y-wlVUP4lkHVgm/view) · [desc sheet](https://docs.google.com/spreadsheets/d/1lz9liWPy8yHybKCdO3BCA9K76QH8a54XduiZS_9fK70/edit) | Apache (CTs NC) | challenging-case sheet: scoliosis/metal/T13/incomplete/missing-floating |
| **⭐ Stump-rib (Möller)** | [**2505.05004**](https://arxiv.org/abs/2505.05004) | [**Hendrik-code/rib-segmentation**](https://github.com/Hendrik-code/rib-segmentation) | [**Zenodo 14850928**](https://doi.org/10.5281/zenodo.14850928) | Apache | **rib + stump-rib masks on VerSe+RibFrac + runnable nnU-Net weights** (stump 20.5%) |
| TotalSegmentator | [2208.05868](https://arxiv.org/abs/2208.05868) | [wasserth/TotalSegmentator](https://github.com/wasserth/TotalSegmentator) | [Zenodo 6802614](https://zenodo.org/records/6802614)·[10047292](https://zenodo.org/records/10047292) | CC-BY-4.0 | none (derive); meta.csv age/sex/institute/study_type |
| CADS | [2507.22953](https://arxiv.org/abs/2507.22953) | [murong-xu/CADS](https://github.com/murong-xu/CADS) | [HF mrmrx/CADS-dataset](https://huggingface.co/datasets/mrmrx/CADS-dataset) | mixed | none; 22k vols, fixed vert mislabel + costovertebral (Jan'26) |
| VertebralBodiesCT-Labels | under submission | — | [HF fhofmann/VertebralBodiesCT-Labels](https://huggingface.co/datasets/fhofmann/VertebralBodiesCT-Labels) | CC-BY-SA-4.0 | numbered bodies T1–T13/L1–L6/S1 (counting) |

### Already labeled vs must-derive (per anomaly)
- **LSTV** → your hub (Castellvi/lstv); derive elsewhere (VerSe L6 count, CTSpine1K, TS).
- **TLTV / stump rib** → **Möller Zenodo** (VerSe+RibFrac ready); derive on your data via T12 anchor.
- **Enumeration (T13/L6/≠5 lumbar)** → VerSe `ctd.json`, VertebralBodies slots; any numbered spine.
- **Rib fracture** → RibFrac (4-type).
- **Scoliosis / deformity** → none ship it → compute (coronal Cobb / ostk spinopelvic).
- **Metal / artifact** → CTPelvic1K filenames, RibSeg sheet.

## Open tasks
- [ ] **pull Möller Zenodo (14850928) rib + stump-rib masks** for VerSe+RibFrac → highest-leverage ready anomaly labels; eval its nnU-Net rib weights on our CTs (the rib model RibSeg never released).
- [ ] `scripts/convert_ribseg.py` — RibFrac CT + RibSeg rib-seg → ribs 34–57 + ignore (HU split, side-audit).
- [ ] `scripts/convert_totalsegmentator.py` — keep femur/hip/sacrum(/sternum/scapula); ignore rest.
- [ ] `scripts/convert_verse.py` — vertebrae → VerSe-native + **QC report** (flag mislabels).
- [ ] per-source **license/attribution ledger** + a `quality_tier` field in the unified manifest.
- [ ] extend `smoke_test_hf.py` into a generic imported-tree QC gate.
