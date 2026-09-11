# CTSpinoPelvic1K — working context

A CT dataset of 802 abdominopelvic records with per-level vertebrae, ribs, sacrum, a carved
S1, hips and femora, built to ask whether local morphology can name a vertebra at the
lumbosacral junction without counting down from C2. The dataset article is in `paper/mpda/`.

## Read first

- `paper/mpda/REPRODUCE.md` — the pipeline end to end, and the traps in it.
- `docs/SPINOPELVIC_ESTIMATOR.md` — how the spinopelvic numbers are computed, four defects
  that were found in them, and three that are still open.

## Things that are true and cost time to rediscover

**There are two label schemes and mixing them does not raise.** The release is v10
(VerSe-native): L1=20 … L5=24, sacrum=26, S1=29, hips=30/31, femurs=32/33. The legacy ostk
map puts L1 at 1 and the femurs at 11/12. Resolving "S1" against the wrong one returns **C7**,
and "femur_left" returns **T4** — a finite, plausible-looking, wrong answer. Call
`ostk.labels.labels_for(volume)`; never import a fixed map. This shipped once: pelvic
incidence was computed for the whole release against the legacy map and the only symptom was
a QC flag saying S1 had too few voxels while S1 sat there with 200,000.

**The released labels are not `data/v5_final`.** They differ by the S1 carve and nothing
else — `level_gradients.csv` regenerates bit-for-bit identical. Anything touching sacrum, S1,
the lumbosacral disc or the pelvis must be recomputed from the released labels; anything
purely vertebral need not be.

**Case ids are zero-padded in the release (`0001`) and were not in the old CSVs (`1`).**
`zfill(4)` before diffing or every row looks changed.

**Volumes are canonicalised to RAS for analysis and never reoriented when written.**
`as_closest_canonical` + `nib.save` silently transposes a label away from its CT.

**The page limit is 10 PUBLISHED pages and only `build.sh --reprint` measures it.** The
submission form is ~25 pages because it is double-spaced; that number is meaningless against
the limit. Read the whole build output — LaTeX emits a PDF and a page count on top of a
broken table.

**Push to both remotes.** `origin` is the personal fork, `osc` is the OpenSpineConsortium
copy the paper cites. Same for OpenSpineToolkit (branch `main`, not `master`).

**Network goes through WSL.** Windows sockets flake; run git push, ssh and rsync via
`wsl -e bash -lc '...'`. `make_submission.py` is the exception — it shells out to `wsl` and
must run from Windows.

**Spread, not the median, is what catches a broken per-level measurement.** Canal depth
and end-plate width both had medians sitting near the published means at every level while
21% of L3 records read below any canal diameter ever reported in a living adult, and 12% of
L5 end-plate widths were walking from the published body width toward the published
transverse-process span. The gates passed; the medians looked right. What failed was
comparing the SD against the widest SD in the living-cohort literature, and comparing the
share below a threshold against the published prevalence of the condition that threshold
defines. `docs/LEVEL_MORPHOMETRY.md`.

**A maximum extent is not a diameter, and a first-slice-that-works is not a plane.** Both
defects above came from those two substitutions. Every normative series measures a
midsagittal chord at a named anatomical plane, and reproducing a published number means
reproducing its definition, not just its units.

**Render the mask.** Three analytically reasonable fixes for the L5 end-plate width were
implemented and all three were wrong, in ways the numbers alone did not reveal. One render
of the retained mask showed what was actually there. This is the second time in this
project that rendering settled something that two or three rounds of reasoning from
numbers got wrong.

## Grid

`go2432@grid.wayne.edu`. Labels at `~/data/CTSpinoPelvic1K/labels`, CT at `.../ct`.
Partition `reqp` with `--qos=requeue` for CPU work; GPU work uses `gmsap`/`gvohp`.

- `#SBATCH --output` resolves against the **submit** directory — use `sbatch -D <dir>`.
- Jobs chained with `afterok` strand permanently when the parent exits non-zero for a
  trivial reason. Prefer `afterany` and check the output yourself.
- `/tmp` is node-local **and swept**. A cleaner deleted the multiprocessing listener sockets
  of two 18-hour training runs at the same minute; put IPC under `/dev/shm`.
- Python env with nibabel/scipy: `~/mambaforge/envs/spineps/bin/python`.

## Standing preferences

- Never use the word "provenance".
- No fractional version bumps; fold into the next integer version.
- Figure captions ≤60 words; the argument belongs in the text.
- Commit directly to master, no feature branches.

## How to be useful here

The recurring failure mode in this project is a number that is wrong but plausible. Three
separate estimators produced finite, reasonable-looking values from the wrong bone, the wrong
surface, or the wrong label map, and none of them raised. Internal consistency did not catch
any of them — the PI = SS + PT identity held throughout an error that moved sacral slope by
9° and lordosis by 12°, because both shared one bad landmark and the errors cancelled.

So: check against something **outside** the computation. Published values, an independent
implementation, the distribution's shape, a stated percentage in the source paper. And when a
tidy explanation presents itself, test it before believing it — the 28% of cases carrying an
identity violation had an innocent reading (anteverted pelvis, lost sign) that was wrong;
looking at the distribution showed a second surface.

Say plainly what is not known. `docs/SPINOPELVIC_ESTIMATOR.md` records an unexplained failure
in 128 records rather than a guess at its cause, and that is the preferred form.
