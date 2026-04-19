# `data/` — staging directory for the pipeline

This directory is **gitignored**.  None of its contents should ever be
committed; the source datasets (TCIA CT COLONOG, CTSpine1K, CTPelvic1K) are
far too large and the placed / exported artefacts are fully reproducible
from the pipeline.

Expected layout after a full pipeline run:

```
data/
├── tcia/                               Stage 1 output — DICOM series by SeriesUID
│   ├── 1.3.6.1.4.1.9328.50.4.1/
│   │   └── *.dcm
│   ├── 1.3.6.1.4.1.9328.50.4.2/
│   │   └── *.dcm
│   ├── ...
│   └── .tcia_patient_index.json        cached index (built by build_db.py)
│
├── ctspine1k/                          Stage 1 output — CTSpine1K (COLONOG fold)
│   └── rawdata/
│       ├── labels/COLONOG/*_seg.nii.gz
│       └── volumes/COLONOG/*.nii.gz
│
├── ctpelvic1k/                         Stage 1 output — CTPelvic1K dataset2
│   └── masks/CTPelvic1K_dataset2_mask_mappingback/
│       └── dataset2_*.nii.gz
│
├── patient_db.json                     Stage 2 Step A — canonical patient DB
├── patient_db.pkl                      Stage 2 Step A — fast-load pickle
├── patient_db_summary.txt              Stage 2 Step A — human-readable summary
│
├── tcia_nifti/                         Stage 2 Step B — dcm2niix reference NIfTIs
│   └── {series_uid}.nii.gz
│
├── placed/                             Stage 2 Step B — placed masks + manifest
│   ├── spine/*_seg_placed.nii.gz
│   ├── spine/*_seg_placed.json         per-case sidecar (bone_pct, IS_ok, ...)
│   ├── pelvic/*_pelvic_placed.nii.gz
│   ├── pelvic/*_pelvic_placed.json
│   └── placed_manifest.json            winning series + LSTV + match_type per case
│
├── qc/                                 Stage 2 Step C — visualize_qc.py output
│   └── per_case/
│       ├── fused/*.png
│       ├── separate/*.png
│       ├── spine_only/*.png
│       ├── pelvic_only/*.png
│       └── is_fail/*.png               IS_ORDER_FAIL cases (red banner)
│
└── hf_export/                          Stage 3 — HuggingFace-ready artefacts
    ├── data/
    │   ├── train/token_*.npz
    │   ├── val/token_*.npz
    │   └── test/token_*.npz
    ├── splits/
    │   ├── train.json
    │   ├── val.json
    │   └── test.json
    ├── README.md                       dataset card (rendered on HF)
    └── splits_summary.json             class distribution per split
```

## Disk budget

| Stage                | Path             | Approx size (filtered / all) |
|----------------------|------------------|------------------------------|
| TCIA download        | `tcia/`          | 110 GB / 350 GB             |
| CTSpine1K            | `ctspine1k/`     | 12 GB                       |
| CTPelvic1K           | `ctpelvic1k/`    |  6 GB                       |
| dcm2niix NIfTIs      | `tcia_nifti/`    | 40 GB / 120 GB              |
| Placed masks         | `placed/`        |  4 GB                       |
| QC figures           | `qc/`            |  2 GB                       |
| HF export            | `hf_export/`     | 50 GB                       |

Allow ~250 GB free for `TCIA_SCOPE=filtered`, ~600 GB for `TCIA_SCOPE=all`.

## Override the root

All paths are driven by `${DATA_ROOT}` in `configs/default.env`.  Set
`DATA_ROOT=/path/to/scratch` in a sourced `env.local` (gitignored) if you need
to land the pipeline somewhere other than `./data`.
