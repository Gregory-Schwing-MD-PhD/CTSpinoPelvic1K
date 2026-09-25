# Camera-ready changes — NeurIPS 2026 Evaluations & Datasets Track, submission 3474

Paper: *Not All Spines Are Created Equal: How CT Segmenters Fail on Lumbosacral Transitional
Vertebrae* (OpenReview `D8yRWDAGQF`, accepted, poster). Source in `src/`; compiled PDF
`src/neurips_2026.pdf`. Build: `latexmk -pdf -interaction=nonstopmode neurips_2026.tex` in WSL
(TeX Live 2026). Line numbers below refer to `src/neurips_2026.tex` at the commit that adds this
file.

## Page budget

The NeurIPS 2026 handbook (MainTrackHandbook, applies to the E&D track's shared style file):
nine content pages at submission, **one additional content page for the camera-ready (ten)**;
references, technical appendices and the mandatory checklist do not count; all of it goes in one
PDF. Current build: **21 pages — content pages 1–10 (Acknowledgments end page 10), references
pages 11–12, checklist pages 13–21.** Zero overfull boxes; the only LaTeX warnings are the
template's `T1/ptm/m/scit` font-shape warnings (harmless). The submitted PDF ran 9 body pages;
the additions below cost about 1.5 pages and about 0.5 page of duplicated prose was cut (intro
VERIDAH/hardware/evaluation-gap paragraphs, case-study body, Interpretation, Generalization,
Implications, Related Work VERIDAH paragraph, upper-lumbar cascade).

## Metareview conditions (binding) → where addressed

The metareview's own wording is quoted in the first column; the word it uses for the origin of
the labels is not used anywhere in the paper or in this file's prose.

| Metareview / reviewer point | Where addressed | Notes |
|---|---|---|
| "state the label provenance and its limits" (metareview (e), Final Justification) | §5 paragraph *Six-way LSTV taxonomy, and where its labels come from* (lines ~360–385); §5 *Castellvi annotation layer* (lines ~405–430); §8 *Limitations*; abstract (ii); contribution (ii); checklist items 2, 12 | States plainly: no subtype was assigned from an image read; each is a function of the CTSpine1K lumbar-label count and the CTPelvic1K filename qualifier (whose origin CTPelvic1K does not document); two of six subtypes rest entirely on a filename tag. Castellvi: two PGY-2 radiology residents, each read all 33 independently, 6 disagreements resolved by consensus, no attending/neuroradiologist read, no blinded re-read, no screening of the normal cohort. |
| "scope LSTV-stratified claims to the test-split n" | Abstract (last sentence and "n=7 cases with a scorable junction"); §3 text and Table 1 (new `n_jxn` column: junction DSC is scorable on 334 normal + 7 LSTV patients; lumbarization junction 0.357 is 3 patients: 0.22, 0.23, 0.62); §7 *Headline, with its n* (the +58.9 junction gain is measured on the **one** held-out lumbarization case whose GT holds both L5 and sacrum; ± is five checkpoints on one patient); Table 6 caption (‡ marks n=1 junction strata); §7.1 (the 82.7 % residual collision is one held-out sacr_count case × 5 checkpoints); §8 Limitations; checklist items 1, 2, 7 | The submitted paper's Table 6 row "Lumbarization n=3, Jxn .946±.007" hid that only one of the three has a scorable junction (per_case_dice.csv: COLONOG_167 has junction 0.950; COLONOG_401_r1 and COLONOG_672 are spine-only). Now stated. |
| "report the subtype counts" | Table 4 (Castellvi × subtype) gained **Train / Test** columns: lumb 11/3, sacr_count 8/1, full sacralization 4/2, semi-sacralization 2/0, ambiguous 1/1, total 26/7; §6 states the 805-record training pool and 174-record held-out split; §8 Limitations lists the test-split composition and says semi-sacralization is unevaluated | Test-split membership and subtypes verified from `NeurIPS/ens/per_case_dice.csv` (Desktop) and `NeurIPS/ens/subgroup_summary.md`; totals from the paper's own Table 1. Castellvi grades of the 7 test cases (from `docs/castellvi_consensus.csv`): 167 IIIb, 401 IIa, 672 IIIb, 555 IIIb, 125 IIb, 123 IIb, 4 IV. |
| Same-data unmerged control ("still calls 84.7 to 99.4 percent of ground-truth L6 an L5, so the merge is the cause"; aoYn W3, jbUM "no same-compute unmerged baseline") | New §7 paragraph *Same-data unmerged control* (`\label{sec:unmerged}`), contribution (iv), abstract (iv), §8 Generalization, checklist items 1, 7 | **Basis and a discrepancy.** The run exists on the grid: `~/spinesurg-ct-nnunet/nnunet/results/Dataset802_SpineSurgCTFull/nnUNetTrainerWandB_500ep_LSTVOversample__nnUNetResEncUNetPlans_100G__3d_fullres/fold_{0..4}` (trained 26–31 July 2026; launcher `slurm/launch_802_unmerged.sh` on branch `L6_Class`). The paper reports what the five training logs contain: the per-epoch validation-patch fraction of GT-L6 voxels predicted as L5, **mean over the final 100 of 500 epochs = 83 / 65 / 66 / 97 % in folds 0/2/3/4** (final epoch 99.9 / 93.3 / 64.7 / 99.8 %); fold 1's confusion block reports "no lumb cases this epoch" for all 500 epochs; validation L6 pseudo-Dice mean over the final 100 epochs 0.09 / 0.10 / 0.05 / 0.13 / 0.01 (folds 0–4) vs L5 0.91–0.95. **The rebuttal's "84.7 to 99.4 %" could not be reproduced from the logs by any single rule tried** (final epoch, mean of last 50/100/250 epochs, value at the best-EMA epoch); it was probably read from smoothed W&B curves. The paper therefore quotes the log-derived numbers with the rule stated, and says the metric is per-epoch validation patches within the CV folds, not a full-volume held-out evaluation. No new experiment was run. |
| Per-subgroup uncertainty (aoYn W1/Q1; jbUM) | §3 (Wilson 95 % intervals: TS L5 DSC < 0.5 in 10/14 lumbarization cases 45–88 %, 9/9 count-style 70–100 %, 9/750 normal 0.6–2.3 %; L6 emission 0/1,152 0–0.3 %); §7 (3/3 last_lumbar emission 44–100 %; per-case ensemble last_lumbar DSC 0.645/0.666/0.725; 7/7 → 65–100 %); §6 protocol sentence; Table 6 caption; §8 Limitations; checklist 7 | Computed from the existing per-case files (`NeurIPS/bm/benchmark_per_case.csv` for TS, `NeurIPS/ens/per_case_dice.csv` for ours); no re-evaluation. |
| Restrict claims to the conspicuous regime (aoYn W1; metareview "1 Type I") | §5 *Morphology-stratified failure interpretation* (2): "Every quantitative claim … scoped to the conspicuous regime (Castellvi II–IV, 31 of 33)"; §8 Limitations | Type I count is now 2/33 (consensus: tokens 175 and 215 are Ib), not 1/33 — see the Castellvi table note below. |
| Inter-rater agreement (aoYn W2/Q2; jbUM "Castellvi labels lack inter-rater agreement"; metareview (d)) | §5 *Castellvi annotation layer*; abstract (ii); contribution (ii); §8 Limitations; checklist 2, 7 | **Cohen's κ = 0.737 on the full grade with laterality (six categories: Ib, IIa, IIb, IIIa, IIIb, IV), n = 33, 27/33 agreed, asymptotic 95 % CI 0.55–0.92 (case bootstrap 0.53–0.91); κ = 0.696 on type I–IV alone (0.47–0.92).** Computed from `docs/castellvi_consensus.csv` (columns `read_1_NA`, `read_2_MI`; the reference-normal row 0016 excluded). Disagreements: 22 (IIa/IIIa), 123 (IIb/IIIb), 125 (IIb/IIIb), 140 (IIa/IIIa), 189 (IIIb/IIb), 215 (Ib/IIb) — five cross the II/III boundary, one the I/II boundary. (The MDPA manuscript says "four of the six were III against II"; by this CSV it is five. Flagged for the MDPA owner; not changed here.) Senior adjudication: **not done**; the paper says so rather than promising it. |
| Soften the wrong-level claim to a plausible risk mechanism (jbUM Q; metareview (d)) | Abstract ("reproduces the pattern of a one-level miscount", "frames it as a plausible automated wrong-level-risk mechanism"); §1 paragraph renamed *The risk mechanism* with an explicit "does not evaluate surgical-planning software, reader behaviour or downstream workflow"; contribution (i); §4 retitled *Case Study: the Level Shift at the Junction*, closing paragraph *Why the pattern matters* states the claim is mechanism and frequency, not measured harm; §6 protocol wording; §8 Limitations and Implications; checklist 1 | "indistinguishable from a wrong-level surgical plan" removed everywhere. |
| Case-level rather than voxel-level uncertainty for the L4/last_lumbar collision (aoYn W4/Q4) | §7 *Headline* (per-case L4 DSC 0.187 ± 0.337 across folds, ensemble 0.032, one case); §7.1 final sentences ("dependent observations on one patient: a directional signal … not a population estimate"); Table 7 caption gives n per block | Not confirmed on more sacralization cases (no new data; the metareview says this is not held against the paper). |
| No semi-sacralization case in the test split (jbUM; aoYn final comment) | Table 4 Test column (0), Table 6 caption, §7 *LSTV-stratified per-subgroup view* ("unevaluated, not measured"), §8 Limitations | |
| TS lacks L6 so the comparison is structurally expected (Fbco) | §1 second paragraph (TS is what clinicians consume), §7 unmerged control (our own L6-class model makes the same error), §8 Generalization | The rebuttal also promised SPINEPS under the same protocol, a Background section and a dataset-construction figure. **Not added**: they would be new experiments/figures, the page budget is exhausted, and the metareview did not require them. Recorded as an open item. |
| Comparison to other datasets / how the benchmark is used (Fbco) | §2 Benchmarks paragraph and Table 5 tiers are unchanged; the rebuttal's dataset-comparison table is not in the paper | Not required by the metareview; page budget. Open item if Greg wants it. |
| Requires H200-class hardware (jbUM) | §1 (shortened), §6 Hardware, §8 Generalization | Unchanged in substance. |

## Other camera-ready edits

* **De-anonymised**: `\usepackage[final,eandd]{neurips_2026}`; footer reads "40th Conference on
  Neural Information Processing Systems (NeurIPS 2026). Track on Evaluations and Datasets."
  Title matches OpenReview exactly. Authors in OpenReview order with affiliations from
  `paper/mpda/title_page.tex` (Gregory: Department of Surgery, DMC and WSU; Ismoilov and
  Alnabahneh: Department of Radiology, DMC and WSU) and, for Loren Schwiebert, Department of
  Computer Science, Wayne State University (from the IRB filing in
  `spinesurg-ct-nnunet/docs/IRB/`). Corresponding-author footnote:
  `gregory.schwing@med.wayne.edu`.
* **Patrick Schwing's affiliation is a red placeholder** (`\PatrickAffil`, line ~31). It is
  recorded nowhere in the project files (title page, Zenodo metadata, CITATION.cff, IRB drafts,
  docx/pptx on the Desktop, web search) and OpenReview's profile page is behind a browser
  challenge. Must be filled before upload.
* **Acknowledgments and Disclosure of Funding** (`\begin{ack}`): "G.J.S. was supported by the
  National Institutes of Health (National Institute of General Medical Sciences, F30GM147909)
  during the first months of this work; the work received no other funding." — the sentence
  the coordinator supplied on 2026-09-25, identical to the updated MDPA title page
  (`paper/mpda/title_page.tex` line 101) and cover letter; the award (5F30GM147909, ended
  31 May 2026) is also named in the WSU invention disclosure. Plus TCIA and the source
  annotators; conflicts: none. The NeurIPS checklist has no funding item, so the disclosure
  lives only in this section, where the template puts it.
* **URLs**: anonymous HF/GitHub links replaced by Zenodo concept DOI 10.5281/zenodo.22139642
  (current v11 = 10.5281/zenodo.22797014), HF mirror
  `OpenSpineConsortium/CTSpinoPelvic1K`, checkpoints
  `huggingface.co/OpenSpineConsortium/spinopelvic-seg-checkpoints`, trainer
  `github.com/Gregory-Schwing-MD-PhD/spinesurg-ct-nnunet` at commit 50b209f, released trainer
  copy and inference code `github.com/OpenSpineConsortium/spinopelvic-seg`, pipeline
  `github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K` — exactly as `paper/mpda/main.tex`
  (line ~640) and the title page give them. **Checked 2026-09-25: all resolve (HTTP 200)
  except `Gregory-Schwing-MD-PhD/spinesurg-ct-nnunet`, which returns 404 (the repo is
  private).** The paper says "All are accessible without credentialed access", which is false
  for that one link until the repo is made public or the sentence is changed to point only at
  the spinopelvic-seg copy. Open item.
* **Castellvi table (Table 4) now shows the two-reader consensus.** The submitted table
  (Ib 1, IIa 2, IIb 4, IIIa 3, IIIb 19, IV 4) matches the first reader's preliminary sheet of
  6 May 2026 (`Desktop/NeurIPS/Lumbosac_Castelvii_Class.csv`), not the consensus released in
  the dataset manifest (Ib 2, IIa 4, IIb 6, IIIa 1, IIIb 16, IV 4; `docs/castellvi_consensus.csv`,
  identical to `zenodo/KNOWN_ISSUES.md`). Consequences in the text: "19/33 (58 %) IIIb" → 16/33
  (48 %); "only 1/33 is Type I" → 2/33; "all 9 count-style sacralization cases are IIIb" still
  holds (both readers, all nine); case-study token 215 is Ib by consensus and one of the six
  disagreements, now said in §4. Note for the coordinator: `zenodo/README.md` (v10 text) carries
  a third distribution (Ib 2, IIa 2, IIb 4, IIIa 3, IIIb 18, IV 4) and says the grades are "one
  reader's, with five cases independently read a second time" — stale relative to
  KNOWN_ISSUES and the manifest; not touched here (outside `CAMERA_READY_NEURIPS/`).
* **Figure 1 caption** cut from 107 to 52 words (standing ≤60-word rule); Table 2 caption 50 → 37.
  Table 5 (comparison) caption is 96 words and Table 6 (baseline) 74: they are table notes
  defining columns and markers, kept for correctness; Table 7 66.
* **Checklist** (`src/checklist.tex`) revised: items 1, 2, 5, 7, 12, 13, 14. Item 14 now cites
  the WSU IRB Non-Human-Participant determination HPR 2026-269 (2 Sept 2026) from the title
  page. Every answer is Yes/NA with a justification; no TODO markers remain.

## Version note (verbatim, §5 "Release used")

> Every number in this paper was computed on the April 2026 release (the HuggingFace export
> finalized 25 April 2026, later labelled v1 in the repository; training began 3 May 2026):
> 1,153 records across 802 patients, a ten-class scheme (background, L1–L6, sacrum, left and
> right hip) merged to nine classes for training, and separate spine-only and pelvic-only
> records for patients whose two source annotations lie on different acquisitions. The archive
> of record is the Zenodo concept DOI 10.5281/zenodo.22139642, whose current version is v11
> (10.5281/zenodo.22797014, 18 September 2026). v11 differs from the April release as follows:
> one record per patient (802 records), each separate-mode patient's missing pelvis completed on
> the spine's series by out-of-fold pseudolabelling; VerSe-native identifiers (L1–L6 = 20–25,
> sacrum 26, hips 30/31) with femora, per-level ribs, lumbar ribs and surgical hardware added as
> further classes; hip-laterality corrections on 22 records and hand corrections on five; both
> Castellvi reads and their consensus in the manifest; and the CT images supplied by TCIA series
> crosswalk and rebuild script, with a HuggingFace mirror carrying the volumes. The 802 patients
> and the 33 Castellvi-graded patients are the same in both. Nothing here was recomputed on v11.

Evidence: `git log` of this repo ("finalize dataset" 2026-04-25; "ship_v1/ship_v2" commits
2026-06-12), `git show 8f1ed51:README.md` (ten-class label table, fused/spine_only/pelvic_native
records), the TS benchmark files dated 2026-04-30 (`Desktop/NeurIPS/bm/`), the title page
(training from 3 May 2026, commit 50b209f), `zenodo/README.md` version history (v1–v3 joined
spine and pelvis on one series, femora and S1; v6 hardware, 22 hip-laterality corrections, five
hand-corrected records; v8 two-reader Castellvi consensus), `zenodo/KNOWN_ISSUES.md`,
`paper/mpda/main.tex` (802 records; out-of-fold pelvic pseudolabelling; TCIA crosswalk) and
`CLAUDE.md` (v11 identifiers). **Not pinned down:** the exact HuggingFace revision hash the
submission's `anonymous-neurips-ED/CTSpinoPelvic1K` pointed to, and whether the branch was
called `v1` at submission time or only from June ("later labelled v1"). The Zenodo record
begins at v6 (first published 2026-08-27); v1–v5 were HuggingFace-only, so "v1" has no DOI.
The checklist (item 5) carries the same note in short form.

## What remains for Greg

1. **Patrick Schwing's affiliation** — replace `\PatrickAffil` (line ~31) and recompile.
2. **`spinesurg-ct-nnunet` is private (404)** — make it public, or edit the Data and Code
   Availability sentence so the only trainer link is the spinopelvic-seg copy.
3. **OpenReview camera-ready form**: upload `src/neurips_2026.pdf` (21 pages: 10 content +
   references + checklist); update Dataset URL to the Zenodo concept DOI / HF mirror, Dataset
   Large URL (the anonymous `-Sample` repo has no public counterpart in the repo — decide), Code
   URL to `github.com/OpenSpineConsortium/spinopelvic-seg`; re-upload the **Croissant file**
   with the public dataset URL (regenerate with `zenodo/make_croissant.py` or use the v11
   deposit's `croissant.json`); the "Financial Support" field lists the first author — the paper
   now discloses the NIH F30 fellowship (F30GM147909) for G.J.S. and no other funding; if the
   form has a funding field, it should say the same.
4. **Decide on the rebuttal promises not delivered**: SPINEPS under the protocol, a Background
   section, a dataset-construction figure, the dataset-comparison table. None is a metareview
   condition; all would need page budget.
5. If the rebuttal's "84.7 to 99.4 %" has a source (a W&B export), the unmerged-control
   paragraph can quote it instead of the log-derived 65–97 %; otherwise the paper stands as is.
6. Optional: reconcile the Zenodo README's Castellvi paragraph and the MDPA "four of six"
   sentence with `docs/castellvi_consensus.csv` (both outside this folder).
7. Cite the paper from the MDPA article with `neurips2026.bib` (key `neurips2026lstv`).
