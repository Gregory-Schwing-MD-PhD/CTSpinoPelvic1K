# Reproducing the CTSpinoPelvic1K dataset article

Everything in the manuscript — every number, every figure, every reference value — is
produced by code in this repository from the released labels. This file is the order to run
it in and the things that will bite you.

The released labels are the input. They are **not** the same as `data/v5_final`, which is
what the CSVs were originally computed from; see [What changed and why](#what-changed-and-why).

---

## 0. Inputs

| what | where |
|---|---|
| labels (802 `*_label.nii.gz`) | Zenodo 10.5281/zenodo.22139642, or the grid at `~/data/CTSpinoPelvic1K/labels` |
| CT (802 `*_ct.nii.gz`) | same release; only needed for the CT-derived measures |
| manifest | `manifest.json` in the release |
| label scheme | `scripts/label_scheme.py`, mirrored in the release's `dataset_labels.json` |

The scheme is **v10** (VerSe-native): L1=20 … L5=24, sacrum=26, S1=29, hips=30/31,
femurs=32/33. Nothing in this pipeline should hard-code those; `ostk.labels.labels_for()`
detects them.

---

## 1. Measurements, from labels to CSV

Each writes one CSV into `morphometrics/`. All take `--labels`, `--manifest`, `--workers`
and `--out`; the three marked (CT) also take `--ct`.

```bash
python scripts/extract_level_gradients.py        --labels $D/labels --manifest $D/manifest.json --out morphometrics
python scripts/extract_surgical_morphometrics.py --labels $D/labels --manifest $D/manifest.json --out morphometrics
python scripts/extract_pelvic_shape.py           --labels $D/labels --manifest $D/manifest.json --out morphometrics
python scripts/extract_transition_morphometrics.py --labels $D/labels --manifest $D/manifest.json --out morphometrics
python scripts/extract_degenerative.py    --labels $D/labels --ct $D/ct --manifest $D/manifest.json --out morphometrics   # (CT)
python scripts/extract_opportunistic.py   --labels $D/labels --ct $D/ct --manifest $D/manifest.json --out morphometrics   # (CT)
python scripts/extract_proximal_femur.py  --labels $D/labels --ct $D/ct --manifest $D/manifest.json --out morphometrics   # (CT)
```

**Resources.** The label-only extractors take 15–30 min at 20 workers. The CT ones take
hours and are memory-hungry: `extract_proximal_femur.py` was killed at 110 GB and needs
~220 GB with 8 workers, because each worker holds a CT volume and a label volume.

**Exit codes lie a little.** These scripts check their own output against physiological
envelopes and exit non-zero if anything is outside. SLURM will mark the job FAILED while the
CSV is written and fine. Read the `*** N measure(s) outside the plausible range` line before
assuming a crash.

---

## 2. Reference values, from the literature

`morphometrics/make_reference_table.py` holds every published series as literals with its
citation, and writes `morphometrics/level_references.csv`. `pedicle_width_references.csv` is
maintained by hand alongside it.

```bash
python morphometrics/make_reference_table.py
```

Two values in it are **digitised from a figure rather than transcribed from a table**, and
there is a script that reproduces them:

```bash
python paper/mpda/tools/digitize_panjabi_fig4a.py MANUAL_REFS/panjabi1991.pdf
```

Panjabi's thoracic end-plate dimensions are in Table 3 on page 892, and page 892 is missing
from every scan obtainable — two byte-identical copies both jump 891 → 893. The script reads
EPWu for T11 and T12 off Figure 4A on page 896 instead, and runs four checks that the paper's
own text has to satisfy. It prints `ALL CHECKS PASS` or refuses. **When interlibrary loan
delivers page 892, replace those two values with the table and delete the dependency.**

Benzel's Fig. 1.1 is the obvious alternative and is not usable; the script's docstring says
why at length.

---

## 3. Figures

```bash
python paper/mpda/make_figures.py          # Figures 4, 5, 7  (countfree, validation, opportunistic)
python paper/mpda/make_levelatlas_fig.py   # Figure 6  (the level atlas)
python paper/mpda/make_fov_fig.py          # Figure 8  (field-of-view coverage)
```

Figures 1–3 are not regenerated from data: Figure 1 is TikZ inside `main.tex`, and Figures 2
and 3 are renders produced by `scripts/render_anchors.py` and `scripts/render_hardware.py`.

**Do not pass a relative `--out`.** `make_figures.py` resolves its default against the file,
not the working directory, because a bare relative default once wrote
`paper/mpda/paper/mpda/figures/` and the build then typeset the previous run's figures with
no error anywhere.

`make_levelatlas_fig.py` also writes `morphometrics/level_atlas.csv`, the numbers behind
Figure 6, which is released so a reader can check the plot.

---

## 4. The manuscript

```bash
bash paper/mpda/build.sh              # submission form: preprint, line numbers, caption list
bash paper/mpda/build.sh --reprint    # THE PAGE COUNT THAT MATTERS: two-column, real authors
```

The limit is **10 published pages**, and only `--reprint` measures it. The submission form
runs to ~25 because it is double-spaced with a caption list; that number means nothing
against the limit.

Read the whole output, not the tail. The script prints `=== errors ===`, overfull boxes and
undefined references before the page count, and LaTeX will happily emit a PDF and a page
count on top of a broken table — a mangled `\\` row terminator once sat in the log for
several rebuilds because only the last three lines were being read.

## 5. The packet

```bash
bash   paper/mpda/make_arxiv.sh        # tarball + a clean-tree compile that proves it builds
python paper/mpda/make_overleaf.py     # Overleaf project zip
python paper/mpda/make_submission.py   # journal packet: PDFs, figures, cover letter
```

`make_arxiv.sh` also refreshes `CTSpinoPelvic1K_arxiv_preview.pdf` from the clean-tree build,
so the preview cannot be older than the tarball it previews. `make_submission.py` shells out
to `wsl` and must be run from Windows, not from inside WSL.

---

## What changed and why

The CSVs in this repository were originally computed from `data/v5_final`. The release ships
a **different label set**, and regenerating everything against it showed the difference is
confined to one place: `level_gradients.csv` comes back bit-for-bit identical, every value
and every n, so the vertebrae and canal are untouched. **S1 is carved as its own label in the
release and was not in v5_final.** That single change moved three measurements, because three
estimators reached for the sacrum when S1 was absent.

The full account is `docs/SPINOPELVIC_ESTIMATOR.md`. The short version:

- **Pelvic incidence.** Without an S1 carve the plate was fitted to the whole sacrum, which
  the alae flatten. With it, the fit sometimes finds the near-vertical anterior face of the
  promontory instead. Neither is right. A plate whose normal lies >60° off cranial is now
  rejected and the case reported missing, gating on the geometry rather than on the value.
- **L5–S1 disc height** lost 174 records silently, because the sampling column sat at the
  median of the two bodies' pooled voxels and the carved S1 body reaches further posteriorly
  than L5. It is placed in their overlap now.
- **Pelvic inlet** was measuring neither of the three published conjugates. Corrected, it
  reads 131.4 mm in women against 125.4 in men — a 6.5 mm dimorphism against a published 6.8.

**Case ids differ between the old and new CSVs**: the old writer emitted `1`, the release
uses `0001`. Normalise with `zfill(4)` before diffing, or every row will look changed.

## Known-open

Three things are documented rather than fixed, in `docs/SPINOPELVIC_ESTIMATOR.md`:

1. **128 records** have a normal sacral plate and normal femoral-head separation yet return a
   pelvic incidence of 1–17°. Cause not identified. They are excluded and counted.
2. **ostk's pedicle width** measures L5 better than the extraction script does (15.1 mm
   against 20.7, published 16.2) but produced almost nothing in a full run and costs 30–60 s
   per level.
3. **The canal-depth low tail was artefact and is now fixed**; `docs/LEVEL_MORPHOMETRY.md`
   records what was wrong with it and with end-plate width at L5, what the measurement
   literature says, and what remains. Two things there are worth doing and are not done:
   resampling each vertebra into its own frame before slicing, which is the published
   correction for oblique-section pseudo-stenosis; and taking end-plate width from the
   SPINEPS corpus label this repository already generates, which removes the
   transverse-process contamination by construction rather than by a geometric rule.
4. **The anterior pelvic plane is not measured**, so anatomical sacral slope cannot be
   reported and the reference-frame question in `docs/SPINOPELVIC_ESTIMATOR.md` stays open.

## Diagnostics

`paper/mpda/tools/` holds the scripts written to answer specific questions, kept because the
questions will recur:

| script | question |
|---|---|
| `digitize_panjabi_fig4a.py` | what are Panjabi's T11/T12 end-plate widths, given page 892 is missing? |
| `render_pi_construction.py` | why does this case return an impossible pelvic incidence? |
| `aggregate_spinopelvic.py` | merge ostk shard output and report clean-vs-flagged rates |
