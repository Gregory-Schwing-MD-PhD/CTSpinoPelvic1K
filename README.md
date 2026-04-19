# CTSpinoPelvic1K

**Pipeline for constructing [CTSpinoPelvic1K](https://huggingface.co/datasets/anonymous-mlhc/CTSpinoPelvic1K)** — a unified CT dataset for lumbar spine and pelvis segmentation with dedicated coverage of lumbosacral transitional vertebrae (LSTV).

Derived from [CTSpine1K](https://github.com/MIRACLE-Center/CTSpine1K), [CTPelvic1K](https://github.com/MIRACLE-Center/CTPelvic1K), and the [TCIA CT COLONOGRAPHY](https://www.cancerimagingarchive.net/collection/ct-colonography/) cohort — fused into a single 10-class label map with patient-anchored mask-to-series resolution.

---

## Quick start

```bash
# 0. Clone
git clone https://github.com/gschwing/ctspinopelvic1k.git
cd ctspinopelvic1k

# 1. Pull the container images (one-time; submits slurm/hpc_pull.sh)
make build-container          # alias for `make hpc-pull`

# 2. Stage 1 — download raw data (~3 hr for TCIA, ~1 hr for the rest)
make download-raw

# 3. Stage 2 — build the dataset (~6 hr)
make create-dataset

# 4. Stage 3 — split, export, and push to HuggingFace
HF_TOKEN=hf_xxx make export-dataset PUSH=1

# 5. Stage 4 — TotalSegmentator zero-shot benchmark (~6 hr on H200)
make benchmark-totalseg       # uses the ctspinopelvic1k-ts.sif pulled in step 1
```

Every stage is a single `sbatch` under the hood — see `slurm/` for the actual job scripts. `make help` lists every target.

---

## Pipeline

Four independent stages, each with its own SLURM script. Each stage writes a clean, self-describing artifact that the next stage reads.

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Stage 1         │ ──▶│ Stage 2         │ ──▶│ Stage 3         │ ──▶│ Stage 4         │
│ download-raw    │    │ create-dataset  │    │ export-dataset  │    │ benchmark-ts    │
│                 │    │                 │    │                 │    │                 │
│ TCIA COLONOG    │    │ build_db.py     │    │ 10-class NIfTI  │    │ TotalSegmentator│
│ CTSpine1K       │    │ → patient_db    │    │ ct/ + labels/   │    │ zero-shot eval  │
│ CTPelvic1K      │    │ place_fused     │    │ stratified      │    │ on all_fused    │
│                 │    │ → placed_man.   │    │ HF Hub push     │    │                 │
│ download_raw.sh │    │ create_dataset  │    │ export_dataset  │    │ benchmark_ts.sh │
└─────────────────┘    └─────────────────┘    └─────────────────┘    └─────────────────┘
        │                      │                      │                      │
        ▼                      ▼                      ▼                      ▼
   data/tcia/            data/patient_db.json    data/hf_export/      results/totalseg_*/
   data/ctspine1k/       data/placed/            → HF Hub              paper_tables.txt
   data/ctpelvic1k/      placed_manifest.json                          benchmark_results.json
```

Key design choice: Stage 2 uses the **patient-anchored `build_db.py`** approach — a mask is only assigned to a TCIA series when the canonical PatientID from the DICOM headers equals the patient UID embedded in the mask filename. No more ad-hoc affine matching with its duplicate-assignment failure modes, no more `colonog_matched_pairs.json` as an intermediate.

Stage 4 establishes the zero-shot TotalSegmentator baseline across the full dataset (fused split), covers LSTV subgroups, and emits the junction-analysis table showing where TS labels L5 as sacrum / sacrum as L5.

See [`docs/pipeline.md`](docs/pipeline.md) for the full technical description.

---

## Features

- **Containerized** — reproducible Singularity/Apptainer image; no `pip install` on the login node
- **SLURM-native** — three clean `sbatch` scripts, one per pipeline stage
- **Patient-anchored resolution** — mask-to-series by DICOM PatientID equality, not affine heuristics
- **Idempotent / resumable** — every stage can be re-run safely; previously-completed work is skipped
- **10-class unified labels** — L1–L6 + sacrum + bilateral hips, PIR canonical orientation, voxel-aligned CT and labels
- **LSTV stratification** — 6-stratum LSTV-aware 70/15/15 train/val/test splits guarantee fused LSTV cases appear in val and test
- **HuggingFace Hub integration** — one command to export + push; token passed via env var, never CLI arg
- **Per-case QC figures** — axial/coronal/sagittal overlays generated for every exported case

---

## Layout

```
ctspinopelvic1k/
├── Makefile                    user-facing entry points (`make help`)
├── README.md                   this file
├── LICENSE                     Apache-2.0
├── CITATION.cff                citation metadata
├── pyproject.toml              Python packaging
├── requirements.txt            host-side deps (for scripts run outside the container)
│
├── containers/                 populated by slurm/hpc_pull.sh
│   ├── README.md               explains how .sif files get there
│   ├── ctspinopelvic1k.sif     lean image (Stages 1-3, utilities)
│   └── ctspinopelvic1k-ts.sif  CUDA image (Stage 4 only)
│
├── scripts/                    all pipeline Python code
│   ├── download_tcia_colonog.py
│   ├── build_db.py             patient_db construction
│   ├── place_fused_masks.py    mask placement on winning TCIA series
│   ├── export_hf.py            10-class export + HF push
│   ├── dataset_interface.py    runtime Python API for the dataset
│   ├── docker_push.sh          workstation: build + push both Docker images
│   ├── hpc_pull.sh             HPC: pull both .sif images (invoked by slurm/hpc_pull.sh)
│   └── ...
│
├── slurm/                      five job scripts
│   ├── hpc_pull.sh             Setup: pull both container images
│   ├── download_raw.sh         Stage 1
│   ├── create_dataset.sh       Stage 2
│   ├── export_dataset.sh       Stage 3
│   └── benchmark_totalseg.sh   Stage 4 (TS zero-shot benchmark)
│
├── docker/                     Docker build definitions (built on workstation)
│   ├── Dockerfile              lean image (download / visualize / export)
│   └── Dockerfile.totalsegmentator   CUDA image (TS benchmark)
│
├── tools/                      utilities (not part of the main pipeline)
│
├── docs/
│   ├── pipeline.md             full technical walk-through
│   ├── dataset_card.md         the dataset README that ships to HF
│   └── troubleshooting.md
│
├── configs/                    pipeline parameters (thresholds, paths)
│   └── default.env
│
├── data/                       (gitignored) staging directory
└── logs/                       (gitignored) SLURM logs
```

---

## Dataset

The constructed dataset is published at:

**`https://huggingface.co/datasets/anonymous-mlhc/CTSpinoPelvic1K`**

Load it:

```python
from dataset_interface import CTSpinoPelvic1K

ds = CTSpinoPelvic1K.from_hub()
fused = ds.filter(config="fused")          # 10-class ground truth
train, val, test = ds.splits()             # LSTV-stratified 70/15/15

case = ds[0]
ct, lbl = case.load()                      # numpy arrays, identical shapes
print(case.token, case.config, case.lstv_label)
```

See [`docs/dataset_card.md`](docs/dataset_card.md) for label scheme, orientation conventions, and usage examples.

---

## Requirements

- **HPC** — SLURM cluster with Singularity ≥ 3.8 or Apptainer ≥ 1.1
- **GPU** — not required for dataset construction; only needed for downstream training
- **Disk** — ~300 GB for raw data, ~80 GB for the exported dataset
- **HF account** — required for pushing (read-only access for downloading)
- **Network** — access to `*.huggingface.co`, `*.cancerimagingarchive.net`, `*.zenodo.org`

---

## Citation

If you use this pipeline or the resulting dataset, please cite:

```bibtex
@dataset{ctspinopelvic1k_2026,
  title     = {{CTSpinoPelvic1K}: A CT-Native Benchmark for Lumbosacral
               Transitional Vertebra Segmentation via Patient-Anchored,
               Registration-Free Multi-Dataset Label Fusion},
  author    = {Schwing, Gregory and others},
  year      = {2026},
  publisher = {HuggingFace},
  url       = {https://huggingface.co/datasets/anonymous-mlhc/CTSpinoPelvic1K}
}
```

And the upstream sources:

- CTSpine1K — Liu et al., 2021
- CTPelvic1K — Liu et al., 2021
- TCIA COLONOG — Clark et al., 2013

---

## License

Code: Apache-2.0 (see `LICENSE`).
Constructed dataset: CC BY-NC 4.0 (inherited from upstream).
