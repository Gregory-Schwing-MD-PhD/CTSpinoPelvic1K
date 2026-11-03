# Fresh NRRD Full-Resolution Spinopelvic Colab Run

This folder frames the fresh `Testfiles-fresh/*.nrrd` scans for a full-resolution
Google Colab run using the working `spinopelvic-seg` / nnUNet checkpoint path.

## What it does

- Converts uploaded NRRD/NIfTI CTs to NIfTI without resampling.
- Runs nnUNet v2 full-resolution inference with the spinopelvic checkpoint:
  `Dataset803_SpineSurgCTFullMerged`, `3d_fullres`,
  `nnUNetResEncUNetPlans_100G`,
  `nnUNetTrainerWandB_500ep_LSTVOversample`, folds `0 1 2 3 4`.
- Writes raw masks, per-label masks, sagittal/coronal/axial CT color overlays,
  label statistics, manifests, logs, and a downloadable ZIP.
- Keeps this as research-only output requiring human review.

## Fresh local input manifest

Validated locally on 2026-09-12:

| File | Size/spacing |
|---|---|
| `0000.nrrd` | `776 x 776 x 165`, `0.2 x 0.2 x 0.2 mm` |
| `2 STANDARD 2MM.nrrd` | `512 x 512 x 150`, `0.326172 x 0.326172 x 2.0 mm` |
| `4 S-T 2mm (1).nrrd` | `512 x 512 x 59`, `0.318359 x 0.318359 x 4.241379 mm` |
| `4 S-T 2mm.nrrd` | `512 x 512 x 59`, `0.318359 x 0.318359 x 4.241379 mm` |
| `4 T-Spine  0.6  Br40  3 (1).nrrd` | `512 x 512 x 621`, `0.357422 x 0.357422 x 0.6 mm` |
| `5 BONE THIN GLOBUS.nrrd` | `512 x 512 x 282`, `0.302734 x 0.302734 x 1.0 mm` |
| `903 TSpine Sag ST 5.nrrd` | `513 x 1019 x 85`, `0.319839 x 0.319839 x 2.0 mm` |
| `cspine.nrrd` | `512 x 512 x 49`, `0.446518 x 0.446518 x 2.5 mm` |

## Browser Colab route

Open `Spinopelvic_Fullres_Fresh_GPU.ipynb` in Colab, select GPU runtime, run top
to bottom, upload the NRRDs when prompted, and enter your Hugging Face token if
the checkpoint asks for one.

## Colab CLI route

The Colab CLI exists at `colab`, but it currently needs
Google authorization. First run:

```bash
colab sessions
```

Open the URL it prints, approve, paste the authorization code, then run:

```bash
HF_TOKEN='paste_token_here_if_needed' bash ai_spine_feasibility/colab/spinopelvic_fullres_fresh/run_colab_cli.sh
```

If no token is needed, omit `HF_TOKEN=...`.

## Important limitation

This uses a public/research checkpoint and produces candidate segmentation masks.
Do not treat the result as diagnosis, autonomous LSTV classification, or final
vertebral numbering. Use the overlays and stats for human review.
