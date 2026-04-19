# Troubleshooting

## Stage 1 — downloads

**`tcia_utils` import error.**  Run `pip install tcia_utils huggingface_hub` into
the container or activate the built `.sif`.

**HF gated repo 403.**  The CTSpine1K COLONOG fold requires accepting the
dataset license on the Hugging Face Hub and a valid `HF_TOKEN` env var.

**Zenodo flaky.**  The CTPelvic1K dataset2 archive is ~6 GB; the downloader
retries on transient 5xx but will bail after 3 attempts.  Re-run the stage —
it resumes from where it left off.

**Out of disk on `$DATA_ROOT`.**  `TCIA_SCOPE=all` needs ~350 GB free; the
filtered scope needs ~110 GB.  Set `TCIA_SCOPE=filtered` in `configs/default.env`.

## Stage 2 — `build_db.py`

**`.tcia_patient_index.json` is stale.**  Pass `--rebuild_tcia_index` after
new TCIA downloads arrive — the cache is only invalidated by the flag, not
by file mtime on each series dir.

**Mask-cache mtime check says stale every run.**  This happens if the mask
archive was extracted with preserved mtimes older than the cache.  Pass
`--rebuild_mask_cache` once to rebuild; subsequent runs will use the cache.

**`FileNotFoundError: rawdata/labels/COLONOG`.**  The CTSpine1K directory
layout varies.  `parse_spine_masks()` falls back through a list of candidate
subdirs; if none match, adjust `seg_dirs` in `scripts/mask_index.py` or pass
`--spine_root` pointing one level deeper.

**Patient has no TCIA series.**  Orphan masks are logged as warnings and
excluded from placement.  This is usually a download-gap — rerun Stage 1 with
`TCIA_SCOPE=all` (the filtered scope can miss patients if the CTPelvic1K
COLONOG list is incomplete).

## Stage 2 — `place_fused_masks.py`

**dcm2niix failures (`myInstanceNumberOrderIsNotSpatial`).**  Handled
automatically — the worker retries with `-n y` (filename sort) and then
`-i y` (ignore derived).  Series that still fail are logged at the end of
Stage 2 and simply excluded from the manifest.

**All placements come back < 20 % bone_pct.**  Almost always means the TCIA
download is short on that patient (wrong kernel only, or scout instead of full
CT).  Run `visualize_qc.py --tokens <failing_token> --debug` to inspect.

**`IS_ORDER_FAIL` on many cases.**  A handful are expected (lumbar lordosis +
partial-volume noise).  If >5 % of your cases flag, something is wrong with
the CTSpine1K seg files — check whether they are in neurological vs
radiological convention.  The PCA-based check in `_spine_placement_checks`
tolerates 8 mm reversals; tighten via the `IS_ORDER_TOL_MM` constant if you
want stricter behaviour.

**`FORCE_PLACEMENT=1` leaves sidecar .json files behind.**  Known — the force
path deletes `*.nii.gz` files and the manifest but not sidecars.  They are
harmless and get overwritten on the next run.  Delete with
`rm data/placed/{spine,pelvic}/*.json` if you want a truly clean slate.

## Stage 3 — `export_hf.py`

**`HF_TOKEN env var not set` on push.**  `--push` requires the token.  Set
`HF_TOKEN=hf_...` in the shell or export it in the SLURM script.

**OOM in worker during NPZ write.**  Drop `--workers` below 8.  Each case
loads both the CT and the label volumes into RAM plus a uint8 copy for the
label remap — ~1 GB per worker for a 512³ scan.

**Stratification puts 0 lumbarization cases in test.**  Possible when a class
has <10 exemplars; the integer split rounding can starve the test bucket.
Increase the count by merging val+test (post-hoc), or rerun with `--seed`
set to a value that better distributes the rare class.

## SLURM / Singularity

**`apptainer: command not found`.**  `containers/build_container.sh` auto-detects
`apptainer` or `singularity` but needs one of them in `$PATH`.  On the Wayne
State Warrior HPC, `module load singularity` before running `make build-container`.

**`--fakeroot` denied.**  Either your cluster is configured without
user-namespace support (contact admin) or you're running on a build node
without the privilege.  As a workaround, remove `--fakeroot` from
`containers/build_container.sh` and build on a box where you have root; then
scp the `.sif` onto the cluster.

**SLURM job dies with OOM at `place_fused_masks`.**  dcm2niix + parallel
NIfTI resamples can peak at ~4 GB per worker.  In
`slurm/create_dataset.sh`, lower `#SBATCH --mem=` or
drop `DCM2NIIX_WORKERS` / `PLACE_WORKERS` in `configs/default.env`.

## Git hygiene

- Never commit files under `data/` — `.gitignore` already excludes the whole
  tree but double-check before `git push`.  A stray `.nii.gz` will balloon
  the repo size.
- Never commit `HF_TOKEN` or an `env.local` with secrets.  `configs/default.env`
  only contains non-secret defaults; put secrets in a sourced `env.local` that
  is gitignored.
