# Camera-ready changes — NeurIPS 2026 Evaluations & Datasets Track, submission 3474

Paper: *Not All Spines Are Created Equal: How CT Segmenters Fail on Lumbosacral Transitional
Vertebrae* (OpenReview `D8yRWDAGQF`, accepted, poster). Source in `src/`; compiled PDF
`src/neurips_2026.pdf`. Build: `latexmk -pdf -interaction=nonstopmode neurips_2026.tex` in WSL
(TeX Live 2026). Line numbers below refer to `src/neurips_2026.tex` at the commit that adds this
file.

## 2026-09-27: corrections after the adversarial audit

An adversarial audit of the 25 September camera-ready (reports A1–A5 in the session scratchpad
`neurips_audit/`, facts files `neurips_fixes/FACTS_{split,numbers,build}.md`) found the three binding
metareview conditions largely met but a set of inaccuracies. This section lists every change made
in response. Where it conflicts with the older sections below, this section is current. Every
number put into the paper was recomputed from data by `neurips_fixes/cr_apply/crverify.py`,
`crverify2.py`, `kapfce.py` and `unmerged_ranges.py` (all read only local copies of the files
named in the "Source" column). Nothing was pushed, committed, posted or published.

**Build (2026-09-27, first pass; superseded by the second-pass build below).** 24 pages: content 1–10 (the acknowledgments end on page 10 at
y ≈ 654 pt of a 722 pt text block, about six lines of slack; the references begin on page 10
after them, as the template allows), references 10–12, technical appendices A–D 13–15, checklist
16–24. 0 overfull boxes, 5 underfull (18 before), no undefined references; the only warnings are three
`h`→`ht` float notes and the template's three `T1/ptm/m/scit` font-shape notes. **Fonts: all 21
are Type 1 and embedded, no Type 3.** `tlmgr install cm-super` in the WSL TinyTeX (done
2026-09-26) replaced the bitmap `ectt1000`/`ectt0800` (F182, F77) with SFTT1000/SFTT0800; the
first rebuild after installing needs `latexmk -g` because latexmk does not track font maps.
Appendices sit after the references and before the checklist, the order of the official 2026
template (`\clearpage \appendix \input{appendix/appendix}`), and do not count as content pages;
the handbook asks that the paper stand alone without them, so every disclosure is stated in the
main text and the appendix carries only its detail. `src/appendix/appendix.tex` is a master file
with one `\input` per section; the Background, construction-figure, dataset-comparison, SPINEPS
and v11 drafts can each be added with one line.

### Changes (before → after, why, source)

| # | Where | Before | After | Why / source |
|---|---|---|---|---|
| 1 | Abstract (ii) | "Castellvi typing (I–IV) of all 33 by two radiology residents (κ=0.74)" | "… by a two-resident consensus (κ=0.74, no senior read)" | A1 item 5: abstract omitted "consensus" and the missing senior read |
| 2 | Abstract (iv) | "a same-data, same-fold unmerged control" | "a same-data unmerged control" | The control used a different record-level split (FACTS_split §5; `split/d802` logs: fold sizes 765/214 … 788/191) |
| 3 | Abstract, last sentence | "Only 7 LSTV cases fall in the held-out split" | "Only 7 held-out records carry an LSTV subtype, in a split drawn by record rather than patient" | 8 LSTV patients have a held-out record (104's pelvic record is tagged normal); split is record-level (FACTS_split §0, §2) |
| 4 | §1 ¶2 | "it is, however, the model surgeons and downstream tools consume" | "a widely used default that downstream tools can consume" | Workflow assertion the paper does not test (A4 item 4) |
| 5 | §1 Evaluation gap | "A production model therefore emits a confident, locally well-formed …"; 3-line hardware sentence (L40S/A100/H100) | "can therefore emit a locally well-formed …"; "Closing the gap needs ~100 GB of GPU memory … (§7)" | Confidence is not measured (A4); page budget (hardware is in §7 and App. D) |
| 6 | Contribution (i) | "95%+ of GT-L6 voxels labeled L5 on the case study, §4" | "95–97% of GT-L6 junction voxels labeled L5 in 2 of the 3 lumbarization patients with a scorable junction, §3" | §4 printed no such number and the benchmark kept no confusion for token 215 (A4 item 6). `bm/per_case_partial.jsonl`: 167 81,088/85,186 = 95.2 %, 699 87,054/89,667 = 97.1 %, 189 907/111,448 = 0.8 %. A TS rerun on 27 Sept gives 95.3 % for 215 (FACTS_numbers §6); not used, because it is a post-review computation — Greg's call |
| 7 | Contribution (ii) | "a two-resident read with κ=0.74" | "a two-resident read made after peer review, with κ=0.74" | Reads are from Sept 2026 (FACTS_numbers §9) |
| 8 | Contribution (iv) | "A same-data, same-fold, same-architecture control" | "A same-data, same-architecture control" | As 2 |
| 9 | §2 | "the production system downstream pipelines actually consume"; "Its ablation (Table 5)" | "a widely used production system"; "(their Table 5)" | A4 item 4; our own Table 5 is now a different table |
| 10 | §3 junction rule, Table 1 caption | "scorable only where the ground truth holds both L5 and sacrum" | "scored on fused records, whose ground truth holds spine and sacrum"; new: "In the two count-style records the window holds no GT-L5 (0.93 is sacrum alone); requiring it leaves 5 LSTV patients at 0.469, a 35.5-point drop" | A5 finding 1: tokens 107 and 615 have only GT-sacrum rows in their junction confusion (`per_case_partial.jsonl`); 107's 0.467 averages a predicted-only L5 (0) with sacrum 0.935. Recomputed: Normal 0.8242 (n=334), Any-LSTV 0.5340 (n=7, drop 29.0), strict 0.4687 (n=5, drop 35.5) (`crverify2.py`). **The facts file's suggested rule "sacrum and at least one lumbar level in the window" is refuted by the same matrices** (no GT lumbar voxel in either window), so the paper states the rule as applied. The abstract keeps 29 points (n=7) with the rule now described correctly |
| 11 | §3 new sentence | — | "In two of the three lumbarization junctions TS labels 95% and 97% of GT-L6 voxels L5; in the third it leaves them background." | Supports 6; `per_case_partial.jsonl` |
| 12 | §3 "The level-shift mechanism" | 10 lines | 7 lines, same content | Page budget |
| 13 | §4 | "establishes the mechanism at the level a surgeon would consume the output"; first sentence of "Why the pattern matters" (disc targeting); "which downstream pipelines do not perform" | "shows the mechanism on one patient's output"; sentence removed (it is in §1); "which the output itself does not flag" | A4 item 4; page budget |
| 14 | §5 composition | "Of 1,153 matched records, 342 … fused, 352 are separate-mode, 90 are spine-only, and 20 are pelvic-only" (sum 804) | "Of the 802 matched patients, 342 … fused, 351 are separate-mode …, 89 are spine-only and 20 pelvic-only: 1,153 records, 53 of them from the 33 LSTV patients" | May-1 `manifest.json` (FACTS_split §3) and the April anonymous manifest (A5): 342/351/89/20 patients, 342+702+89+20 = 1,153 records |
| 15 | Table 3 (composition) | In §5: 352, 90/20, "33 patients (44 records)" | Moved to Appendix A as Table 7: 351, 89/20, spine/pelvic-labelled records 782/713, "33 LSTV patients: 53 records", plus train/validation/test manifests (639/805, 170/174, 167/174 patients/records) and the 979-record pool | FACTS_split §1, §3. 44 is the LSTV patients' records in the pool (53 − 9 held out) |
| 16 | §5 "Release used" | "Every number in this paper was computed on the April 2026 release …"; the v11 differences listed in full | "Every segmentation number …"; adds "its train/validation/test manifests were redrawn on 1 May" and "The Castellvi grades and κ are instead the September 2026 two-reader reads (§5), released from Zenodo v8 onward"; the v11 list moved to Appendix A | A1 item 11 (no grade existed on 25 April); FACTS_split §1 (split redrawn 1 May; the 1 May 2026 HF revision, see FACTS_split, private); page budget |
| 17 | §5 taxonomy | "merging two phenotypes whose failure modes diverge (full sacralization preserves the lumbar count and near-normal per-class DSC; semisacralization confuses the L5/S1 boundary)" and "Without the corrected taxonomy … are invisible" | "merging two phenotypes the source distinguishes … Under TS (Table 1), full sacralization is near-normal in 3 of its 5 spine-labelled patients (one intermediate, one collapsed); semi-sacralization has one spine-labelled patient, near-normal, and no scorable junction." | A3 item 8; `bm/benchmark_per_case.csv` per patient: 6, 15, 32 at 0.94–0.96 on L1–L5, 123 intermediate, 125 collapsed, 140 no spine labels; semi 120 at 0.95–0.96, 22 pelvic-only |
| 18 | §5 partial annotation | "to train on all 1,153"; "784 records … 714. This is a 3.36× effective expansion." | "so that partial records train too"; "782 carry spine labels … 713 pelvic labels …, against 342 fused records"; the tag-refinement rule stated | 979 records train (FACTS_split §1); 782/713 (§3); 3.36× matches no count (1,153/342 = 3.37); rule from `convert_hf_to_nnunet.py` @ df10fea lines 104–110 |
| 19 | §5 Castellvi paragraph | "It is the surgeon's pre-operative wrong-level-risk indicator, and …"; "Two PGY-2 radiology residents each graded all 33 LSTV cases independently …"; κ 0.70 without interval; CI method not named | Removed the indicator clause; "… both co-authors, each graded all 33 LSTV cases in September 2026, after peer review …"; "The reviewed version's grades came instead from a one-column sheet of May 2026, described there as a joint two-resident consensus, so the September readers were not naive to the cases"; "κ=0.70 on type I–IV alone, 0.48–0.92; Fleiss–Cohen–Everitt standard errors"; "the consensus kept reader 1's grade in all six"; "Against the reviewed table, five grades changed (four III→II, one IIIb→Ib): Type I rose from 1 to 2 of 33, IIIb fell from 19 to 16, and three of the six full sacralizations are now Type II (Appendix B)" | A1 items 2, 3, 10; FACTS_numbers §9. Recomputed (`kapfce.py`, `docs/castellvi_consensus.csv`, row 0016 excluded): full κ 0.737 (0.553–0.920), type κ 0.696 (0.475–0.917), 27/33, consensus = reader 1 on 33/33. Grade changes from `Desktop/NeurIPS/Lumbosac_Castelvii_Class.csv` (6 May) vs consensus: 22, 140 IIIa→IIa; 123, 125 IIIb→IIb; 175 IIIb→Ib. Submitted text: "jointly annotated all 33 LSTV cases by consensus (no independent inter-rater scoring)" (`NEURIPS_CURRENT_PDF.txt` l.507) |
| 20 | Table 3 (Castellvi; was Table 4) | Caption "number of patients in the training pool … and the held-out test split"; columns Train/Test 11/8/4/2/1 = 26 and 3/1/2/0/1 = 7 | Caption dated September 2026; "Pool: patients with an LSTV-tagged record in the 979-record training pool; Test: patients with an LSTV-tagged held-out record. Token 4 is in both; five patients have records on both sides"; Pool 11/8/4/2/**2** = **27** | FACTS_split §3: the old Train column was Total − Test; token 4's spine record `COLONOG_4` is ambiguous-tagged and in the pool |
| 21 | §5 morphology (2) | "Every quantitative claim in this paper is therefore scoped to the conspicuous regime (Castellvi II–IV, 31 of 33 cases); the clinically subtler Type I fraction is uncharacterized here." | "Castellvi-stratified statements are therefore scoped to the conspicuous regime …, and Type I is uncharacterized here. The two Type I patients (tokens 175 and 215, both Ib lumbarizations; 215 is the case study) are in Table 1's lumbarization and Any-LSTV rows (without them TS's lumbarization L4/L5 DSC is 0.452/0.339, n=12) and in fold 4 of the unmerged control (§8.1); neither is held out." | A1 item 7, A4 item 3; FACTS_numbers §5; `crverify.py` (0.452/0.339, n=12; 8/12 below 0.5) |
| 22 | §6 protocol | "the clinical window for surgical planning"; four `\texttt` block names; "where a stratum holds three or fewer cases we give the per-case values" | "where the level shift lands"; "three confusion blocks on the L4/last_lumbar boundary … (Table 6)"; "single-record cells are marked, three-case strata are given per case" | A4 item 4; A5 (per-case values were not given for every stratum ≤ 3); page budget |
| 23 | §7 architecture | Encoder feature/block list and loss-weight rationale inline; "5-fold patient-stratified CV with 6-way LSTV stratification on a training pool of 805 records; 174 records (7 LSTV patients, Table 4) are held out" | Encoder detail → Appendix D; "5-fold CV, grouped by patient and stratified by LSTV subtype, over a 979-record pool (the release's 805-record train and 174-record validation manifests); the 174-record test manifest is held out" | FACTS_split §1 (converter sends train + validation manifests to imagesTr; 803 logs 781/198 … 789/190); page budget |
| 24 | §7 new paragraph | — | "The held-out split is drawn by record. … 92 of its 167 patients are separate-mode patients whose other acquisition, labelled only for the other region, is in the pool and trained four of the five fold models; among LSTV patients these are 401, 555, 123 and 4, and patient 104, whose only held-out record is pelvic and scored as Normal (as is a pelvic record of patient 672). Every junction-DSC value comes from a fused record, which has no sibling (Appendix A)." | FACTS_split §2 (`split/recon/test_straddle.csv`). The like-for-like check and the patient-disjoint sensitivity (FACTS_split §4) are **not** in the paper; they are for Greg |
| 25 | §7 two-tier emphasis, Hardware | Rationale for no patch biasing (3 lines); NVMe/NFS detail | One clause; detail → Appendix D | Page budget |
| 26 | §8 headline | "junction-DSC rises from TS = 0.357 (3 patients) to ours = 0.946 ± 0.007 …, a Δ = +58.9 ± 0.7 points. That number is measured on the one held-out lumbarization case …" | "On the one held-out lumbarization case whose ground truth holds both L5 and sacrum (token 167, a fused record), junction-DSC rises from TS = 0.221 to ours = 0.946 ± 0.007 …, a paired Δ = +72.5 ± 0.7 points (+58.9 against TS's unpaired mean of 0.357 over its three junction-scorable lumbarization patients)" | A2: 0.357 is TS over 167/189/699, not the held-out case. Paired per fold +73.8/+72.2/+72.6/+72.2/+71.8 (`bm/benchmark_per_case.csv` token 167 0.2210; `ens/cross/cross_fold_headline.json`) |
| 27 | §8 headline | "(one fold's checkpoint scores mid-range, the others near zero …)"; "§8.1 characterizes this as the empirical signature …"; sacrum "+15.2 ± 0.2 points (TS = 0.817 → 0.969 …; 113 held-out records …)" | "(one checkpoint scores 0.79, the others below 0.10 …)"; sentence removed (repeated in §8.1); "+15.2 ± 0.2 points, unpaired (TS = 0.817 over 801 patients → 0.969 ± 0.002 over the 113 held-out records with sacral GT …)" | `cross_fold_headline.json` L4 on sacr_count .016/.786/.018/.099/.014 (A2 finding 6); unpaired label per task |
| 28 | Table 4 (comparison; was 5) | "Ours: …, 5-fold ensemble on held-out test"; "Ours, Any-LSTV (n=7)† 60.1 / 46.6 / 31.1 / 15.8 / — / 65.2 / 92.0 / 97.5 / 97.6 / 97.6"; "†Case-weighted mean over the 7 LSTV test patients" | "five fold checkpoints on the held-out split"; "Ours, Any-LSTV (n=4–5)† 72.1 / 55.8 / 37.3 / 17.7 / — / 66.9 / 90.7 / 98.3 / 98.3 / 98.4"; "†Mean over the held-out LSTV-tagged records with GT for the class (5 for L1–L4, 4 otherwise)"; caption adds "Ours: mean of the five fold checkpoints; n counts records" | FACTS_numbers §1; recomputed in `crverify.py` from `ens/subgroup_summary.csv` (*_n) × `ens/cross/cross_fold_summary.csv` (*_mean). The old row weighted subgroup means by row n, counting records with no GT. All "ours" values are cross-fold means, not the ensemble (Normal row checked: 95.3 … 98.8) |
| 29 | §8 per-subgroup view | "with 7 LSTV test patients the Wilson 95% interval for 7/7 cases emitting last_lumbar is 65–100%" | "last_lumbar is emitted on all 4 held-out LSTV records with last_lumbar GT (Wilson 51–100%); on the sacr_count record emission is itself the error (Block 3), and two pelvic-only records cannot be judged." | A4 item 2; `ens/per_case_dice.csv` (125, 167, 401_r1, 672 have LL GT; 555_r1, 123_r1, 4_r1 do not) |
| 30 | Table 5 (baseline; was 6) | Caption "5-fold ensemble … cross-fold mean ± std"; ‡ only on Jxn; Δ rows unpaired: lumb −0.6/−24.2/−31.3/−17.5/N/A/+11.3/+3.2/+58.9, sacr_count +72.3/+85.9/+67.9/+18.6 | Caption "Five fold checkpoints …, mean ± std across checkpoints; n counts records (Normal: 108 with spine GT, 109 with pelvic GT) … Δ: paired, TS zero-shot on the same records … ‡One record scorable for this class"; ‡ added to lumbarization Sac and Hip_m and to full-sacralization L1–L4 and last_lumbar; Δ paired: lumb −8.0/−32.2/−32.0/−3.8/N/A/+13.6/+5.1/+72.5, sacr_count +98.8/+95.9/+73.3/+18.6 | A2 findings 5 and paired Δ; FACTS_numbers §2–3; `crverify.py` (TS record-level `bm/benchmark_per_case_record_level.csv`: 167 fused, 401/672/555 spine_only; Hip_m = mean of TS L/R) |
| 31 | §8.1 unmerged control | "the identical network (same planner, patch, loss weights, optimizer, 500 epochs, queue-level LSTV oversampling and the same five folds)"; "the fifth fold's validation split holds none"; "fluctuates (29–100% within one fold …)"; "With the same data, folds and capacity"; "(11 lumbarization patients in the training pool)" | "the same network (planner, patch, optimizer, 500 epochs, queue-level LSTV oversampling) on the same 1,153 records …, with an explicit L6 class weighted 4.0 in the cross-entropy …. Its folds come from a different record-level split with its own subtype map (count-style merged into sacralization, one more record tagged lumbarization); 149 of our 174 held-out records lie in its cross-validation pool, so it is read only inside its own folds"; "in folds 0, 2, 3 and 4 … fold 1 holds none"; "fluctuates (Appendix C)"; new: "The 84.7–99.4% quoted during review was read from folds 0 and 4 about halfway through training, before folds 1–3 had started (Appendix C)"; "With the same data and capacity"; "(12 lumbarization-tagged patients in its pool)" | A4 item 1, A5 finding 2; FACTS_split §5 (split, 149/174, subtype map lumb 15 / sacralization 14, CE weights `[0.5,1,1,1,1,1.5,4.0,2.0,1,1]`, 12 lumbarization patients in its pool, the 27 Jul 14:02–14:06 window). Per-fold figures recomputed from the copied logs (`unmerged_ranges.py`): 82.97/65.38/66.30/96.81 %, final 99.9/93.3/64.7/99.8 %, fold minima 29.0/26.0/17.9/64.3 % — the old "29–100% within one fold" described fold 0 only |
| 32 | §8.1 cascade | DSC list; "so on per-class L2–L4 TS is better"; "closes the surgically critical gap (+58.9 pts on the one scorable case)" | List removed (in Table 5); "so on per-class L1–L4 TS is better on the same records"; "closes the gap there (+72.5 points, paired, on the one scorable case)" | Paired L1 is −8.0, so L1 is also worse; A4 item 4 |
| 33 | §8.1 Interpretation | "the positional prior dominates" | "the positional prior dominates in four of five checkpoints" | A2 finding 6 (one checkpoint scores 0.79) |
| 34 | §9 | "Any spine segmenter lacking an L6 class shifts labels on lumbarization, and §8.1 shows that adding the class … does not remove it"; separate Future work paragraph | "A spine segmenter without an L6 class has no correct label for a sixth lumbar body, and §8.1 shows that adding the class … did not remove the collision"; Future work folded into the paragraph ("Generalization, hardware and future work") | A2 (TS did not shift on 4/14); page budget |
| 35 | §9 Limitations | "the Castellvi grades are a two-resident consensus (κ=0.74, n=33) with no senior read …"; "every quantitative claim is scoped to the conspicuous regime"; "The held-out test split contains 7 of the 33 LSTV patients"; the 15/5 case-fold parenthetical; "Wilson intervals for strata of 1–7 cases span 21–65% to 100%"; unmerged "within the cross-validation folds"; "the correct comparison for using TS as a production system but underselling …" | "… read after peer review by co-authors who had seen a May 2026 grading of the same cases …"; "(2 of 33, both in Table 1, neither held out), so Castellvi-stratified statements are scoped …"; "The held-out split was drawn by record, so 92 of its 167 patients have their other acquisition in training. Seven of the 33 LSTV patients have an LSTV-tagged held-out record …"; parenthetical → "(§8.1)"; "strata of 3–4 cases start at 44–51%"; "uses a different split and is read … within its cross-validation folds"; "as a production system, not as it might do if retrained …" | A1, A2, A4 items; the 7/7 interval no longer appears (item 29) |
| 36 | §9 Implications | "emits a confident, well-formed … gives no error signal and reproduces the pattern of the wrong-level miscount downstream"; "with each stratum's n" | "emits a well-formed … gives no error signal at evaluation time and, when consumed downstream, would reproduce the pattern of the wrong-level miscount"; "with n" | A4 item 4: commit 38b0cb9 had turned the conditional into an assertion |
| 37 | Data and Code Availability | "… (current version v11 …) with core and Responsible AI fields of the Croissant specification"; "trained with the released trainer and inference code at …/spinopelvic-seg (commit 50b209f)" | "…; §5 describes the release used, and the v6 version holds a Croissant file with core and Responsible AI fields"; "The five fold checkpoints, trained from commit 50b209f of a private development repository with its v20 trainer (committed there on 5 May as df10fea), are at …; that trainer, with the conversion and inference code, is at …/spinopelvic-seg (commit c105981)" | A5: v11 has no `croissant.json` (only v6 does). 50b209f is not in spinopelvic-seg (GitHub 422); its commits are fd8d565 and c105981. W&B metadata of all five Dataset803 runs (grid `wandb/run-20260503_*/files/wandb-metadata.json`) records HEAD 50b209f while the logs print trainer "(v20)"; 50b209f's trainer is v19.1 and v20 was committed as df10fea on 5 May; the spinopelvic-seg copy at c105981 differs from df10fea only in naming/anonymisation (34 diff lines) |
| 38 | Appendices (new) | — | A: release used, records and the evaluated split (the v11 list, Table 7, overlap counts, LSTV patients, Normal stratum, the evaluated split is not in the archive, the reviewer-visible HF split shares 25/174); B: Castellvi grades before and after review (Table 8: May / August / reader 1 / reader 2 / consensus; the five changes; consensus = reader 1; bootstrap 0.53–0.91; reader-sheet rib-free count disagreement 7/33); C: unmerged control (split, subtype map, loss, patch bias 0.98–1.41 %, Table 9 per fold with validation lumbarization tokens, the 84.7–99.4 % timing); D: network and hardware details moved from §7 | FACTS_split §1–5, FACTS_numbers §9, `docs/castellvi_readers_2026-09-04.csv`, `split/d802/L/f{0-4}.txt`. Normal-stratum values recomputed in `crverify2.py` (sacrum 0.9719→0.9717, hip 0.9884→0.9882, mean fg 0.9658→0.9655) |
| 39 | `bib/references.bib` | Liu2025 url `…/FracSegNet.` and doi `10.3389/FMED.2025.1511487/TEXT`; Konin2010 url = two URLs joined by a space; Meng2023 url with a personal `casa_token`; Sekuboyina2020 journal = a DOI URL and url = relative path; Akhtar2024 url `github.com/mlcommons/` | `…/FracSegNet`, doi `10.3389/fmed.2025.1511487`; one AJNR URL; token stripped; journal "Radiology: Artificial Intelligence", url `https://doi.org/10.1148/ryai.2020190074`; `github.com/mlcommons/croissant` | A5 §5. Checked via WSL curl: FracSegNet 200, croissant 200, the old `/TEXT` DOI 404 and the new DOI 200; RSNA and AJNR DOIs redirect (302) to publishers that return 403 to scripted clients |
| 40 | Checklist 1 | "Quantitative abstract values (TS per-class DSC, 0.357 junction-DSC on lumbarization, 0.817 zero-shot sacrum, κ=0.74) … (7 LSTV patients in the held-out split …)" | Values as printed in the abstract; explains the two different sevens (they share two patients); unmerged control not same-split; split drawn by record | A5 §7, A2 finding 2 |
| 41 | Checklist 2 | Conspicuous-regime and test-split wording as in the old Limitations | Matches the new Limitations (after-review reads, five grades changed, Type I patients named, record-level split, different split for the control) | Consistency |
| 42 | Checklist 4, 6 | "5-fold patient-stratified CV with 6-way LSTV stratification"; "§5 specifies the splits" | "5-fold CV grouped by patient and stratified by LSTV subtype over a 979-record pool, with a 174-record held-out split drawn by record (§7, Appendix A)"; encoder spec → Appendix D; item 4 adds that the evaluated split is not in the archive | FACTS_split §1 |
| 43 | Checklist 5 | "with core and Responsible-AI fields of the Croissant specification"; "the training repository at the cited commit … public without credentialed access" | Croissant in v6, not v11; public trainer copy at c105981, models trained from 50b209f of a private repository; "the held-out record list is therefore not yet public" | A5 bottom-line items 2–3 |
| 44 | Checklist 7 | "where a stratum holds one to three cases the per-case values are given in the text"; "Cohen's κ with an asymptotic 95% interval" | Single-record cells marked; per-case values for lumbarization; full-sacralization row explained; Δ paired / sacrum unpaired; 4/4 → 51–100 %; "Fleiss–Cohen–Everitt asymptotic 95% intervals (full grade 0.74, 0.55–0.92; type only 0.70, 0.48–0.92)"; bootstrap B = 10,000 | A5 §7 |
| 45 | Checklist 8 | "§1 documents that the 100 GB GPU-memory budget is the smallest configuration"; "(5-fold CV total ≈ 15 GPU-days)" | "§7 (Hardware) and Appendix D"; "(5 folds × 3 days ≈ 15 GPU-days)"; discloses the unmerged control's second 5-fold, 500-epoch run and one-inference-per-record TS benchmark; earlier development runs not itemized | A5 §7 |
| 46 | Checklist 10 | "produces an output indistinguishable from a wrong-level surgical plan"; "we explicitly recommend that automated outputs require radiologist review" | "whose output on lumbarization reproduces the pattern of a one-level miscount … (a plausible wrong-level-risk mechanism, not a measured harm)"; negative impact stated in the checklist itself (research release, not validated for clinical use, documented residual collision) | The phrase the earlier change log said was "removed everywhere" survived here (A4, A5); the body never recommended radiologist review |
| 47 | Checklist 12, 13, 14 | "inherits the source-data license terms in its Croissant Responsible-AI fields"; "the frozen splits … Croissant metadata with Responsible-AI fields" | "annotations are licensed CC BY-NC-SA 4.0 (the archive's LICENSE file); the underlying TCIA imaging keeps its own terms"; "a patient-level 5-fold split over the 802 patients (not the record-level split evaluated here) …; a Croissant file with Responsible-AI fields is in the v6 version, not in v11"; item 14 adds "in September 2026" | `zenodo/LICENSE`, `zenodo/README.md` §"Sources and licence"; A5 §7 |
| 48 | This file | Quoted the metareview's word for label origin (line 31); called the May sheet "the first reader's preliminary sheet" (line 80) | Paraphrased; corrected in place | Standing rule on that word; A1 (sheet author not recorded) |
| 49 | Table 4 and Table 5 captions | 98 and 71 words (the 25 Sept file excused them as table notes) | 59 and 54 words: Table 4 drops "L5 is the unmerged anatomical L5 class" and "we train per-side and report both" and compresses the rest; Table 5 drops "Semi-sacralization is absent from the test split" (stated in §8) and compresses the rest. The wording in rows 28 and 30 is the pre-trim text | ≤ 60-word caption rule; all captions now ≤ 60 (Table 1 44, Table 2 37, Figure 1 52, Table 3 47, Table 6 46, appendix Tables 7–9 14/13/33) |

**Not changed, deliberately.** No headline metric was recomputed on a new split; the reviewed
numbers stay, with pairing, n and labels corrected. The patient-disjoint sensitivity values and
the like-for-like overlap check (FACTS_split §4) are not in the paper; they are for Greg to decide
on. [Second pass: the case-bootstrap CIs on TS Dice are now in the paper, §3 and Appendix E.]
Greg's affiliation line is untouched (open question for Greg).

## 2026-09-27, second pass: residual issues from the verification of the first pass

A verification of the first-pass build found one high, seven medium and a set of low residual
issues. This section lists every change made for them; where it conflicts with anything above, it
is current. Numbers were recomputed by `neurips_fixes/cr_apply/crverify3.py` (output
`crverify3_out.txt`), which reads only `Desktop/NeurIPS/bm/benchmark_per_case.csv`,
`bm/benchmark_per_case_record_level.csv` and `ens/per_case_dice.csv`. The Castellvi dating was
checked against `git show 33596df:morphometrics/castellvi_grades.csv` (23 Aug) and
`docs/castellvi_readers_2026-09-04.csv`. Nothing was pushed, committed, posted or published.

**Build (second pass; superseded by the third-pass build below).** 25 pages: content 1–10 (the acknowledgments end on page 10 at
y ≈ 570 pt of a 722 pt text block, about 14 lines of slack; the references begin on page 10 after
them), references 10–12, technical appendices A–E 13–16, checklist 17–25. 0 overfull boxes,
3 underfull (all in the references and checklist), no undefined references or citations; the only
warnings are three `h`→`ht` float notes and the template's three `T1/ptm/m/scit` notes. Fonts: 22,
all Type 1 and embedded, no Type 3 (`cr_apply/pagecheck.py`). `\newpage` before the checklist
became `\clearpage` so that appendix floats are set before the checklist, not on its first page.
All captions ≤ 60 words (`cr_apply/capcount.py`: Table 1 44, Table 4 56, Table 5 56, Table 10 24).

| # | Issue (severity) | Where | Change | Source / recomputation |
|---|---|---|---|---|
| S1 | April release and evaluated split are not at any cited location; checklist 5 claimed faithful reproducibility (high) | §5 Release used; Availability; checklist 4, 5, 13; App. A | §5: "later labelled v1" removed (no public revision carries it); adds "Neither these records nor the evaluated split is yet at a public location this paper cites (Appendix A)". Availability ends "the April release's records and the evaluated split are not yet public (Appendix A)". App. A: new paragraph *What is public* (Zenodo v11 and the OSC HF mirror hold the 802-record v11 layout; the archive's split has no test split; the reviewer-visible anonymous HF split shares 25/174; "Until the April manifests and the held-out record list are published, the segmentation results cannot be reproduced exactly from public files"). Checklist 5 answer **Yes → No**, with a justification that lists what is open and what is not. Checklist 4 and 13 say the same | Audit evidence (OSC HF manifest 802 records, main only, history from 2026-09-01; Zenodo v11 802 records, no test split, no croissant). **Greg: if the April manifests and `splits_final` are published before upload, see item 16 below for the replacement text** |
| S2 | No interval on any Dice mean or on the 29-point drop (medium; aoYn W1) | Abstract; §3; Table 1 caption; new App. E (Table 10); §6; checklist 7 | Abstract: "drops 29 points (95% CI 10–47)". §3: "L4=0.388 [0.18, 0.60] → L5=0.291 [0.10, 0.51] over 14 patients; case-bootstrap 95% intervals, Appendix E" and "(95% CI 10–47)". App. E: per-class n and 95% case-bootstrap interval for every Table 1 cell over ≥4 patients, per-patient values otherwise; the strict-rule drop 35.5 (15.1–55.5). Checklist 7 now: "trained with 5-fold CV grouped by patient within a 979-record pool and evaluated on the 174-record held-out split, which was drawn by record", plus the interval method and examples | `crverify3.py`: patients resampled within subgroup, B = 10,000, percentile, seed 20260926. Lumb L4 .388 [.184, .602], L5 .291 [.099, .507] (n = 14); Any-LSTV Jxn .534 [.348, .722] (n = 7); drop 29.0 (9.9–47.4, two-sample); strict .469 (n = 5), drop 35.5 (15.1–55.5). These agree with FACTS_numbers §4 (scipy) to ±0.001 |
| S3 | LSTV-stratified claims generalised from one held-out record (medium) | Abstract (iv); contribution (iv); §5 morphology (1); §8 headline; §8 per-subgroup view; §8.1 Interpretation | "on the one held-out count-style sacralization case" (abstract); "(one held-out case)" (contribution iv); §5 (1): "On the one held-out count-style case (n=1), the residual … collision therefore occurs where …" and "On that case, direct semantic segmentation did not use the Castellvi correlate … an argument, from one case, for …"; §8 headline opens "On the one junction-scorable held-out lumbarization case (token 167 …), the merged-label baseline closes the junction-DSC gap"; "sacr_count cases" → "the one held-out sacr_count case"; §8.1 "on the one held-out count-style sacralization case the positional prior dominates in four of five checkpoints" | Table 5 row n = 1; `cross_fold_headline.json` |
| S4 | Type I compared with Sekharappa's 14–16 %, which is LSTV II–IV prevalence in a study that excluded Type I (medium; carried from the submission) | §5 morphology (2) | Figure and citation there removed: "Only 2/33 cases are Type I. Selection is by lumbar count and filename tag, so an LSTV with five lumbar bodies and no tag, as a Type I dysplastic transverse process can be, is not in the cohort; this share is not a prevalence." Scope sentence kept. Sekharappa stays cited in §2 | Audit: PMC3939369 ("prevalence of LSTV in groups A, B, and C was 8.1%, 14%, and 16.9%"; Type I "was not called a transitional vertebra in our study") |
| S5 | Rebuttal promised Fbco appendices (SPINEPS, Background, comparison, construction figure); none is in the PDF (medium) | OPENREVIEW_NOTE.md item 5 | Not integrated (other workflows are drafting them). The note now says they are **not** in this camera-ready and gives, in an HTML comment, the replacement bullet to use if they are integrated before upload | — |
| S6 | Castellvi reads dated "September 2026", but reader 2's grades for five cases are the 23 Aug second reads (medium) | §5 Castellvi; §5 Release used; checklist 2, 14; App. B text and Table 8 row labels; OPENREVIEW_NOTE item 2 | "each graded all 33 LSTV cases after peer review …; the reads were complete by 4 September 2026, and reader 2's grades for five cases are second reads recorded on 23 August"; "reads completed on 4 September 2026"; App. B adds "Reader 2's grades for those five cases equal the 23 August second reads, including 189 and 215, where reader 1 differs, so at least part of reader 2's grading predates September"; table rows "Reader 1/2 (by 4 September 2026)", "Consensus (4 September 2026)" | Aug sheet (33596df, 23 Aug) second reads 175 Ib, 189 IIb, 215 IIb, 261 IIb, 267 IV = Sept reader 2 (MI) on all five; reader 1 (NA) 189 IIIb, 215 Ib; consensus committed 349f061, 4 Sep 01:11 |
| S7 | OpenReview note item 4: "+58.9 … against TS's mean over three other patients" (0.357 includes 167) (medium) | OPENREVIEW_NOTE item 4 | "… over the three junction-scorable lumbarization patients (this one and two others)" | `bm`: 167 .2210, 189 .6175, 699 .2337, mean .3574 |
| S8 | TS sacrum "0.817 over 801 patients" (medium) | §8 headline | "over the 712 patients with sacral GT", plus "TS on those 113 records: 0.815" (paired) | `crverify3.py`: dice_sacrum mean .8167, n = 712 (Normal .8165, n = 683); paired on the 113 held-out records with sacral GT, TS .81546 against ours (ensemble) .9698, 113/113 matched by token (fused or pelvic_native record) |
| S9 | Wrong-level framing claims uncited or unsupported (low) | Abstract; §1 ¶1–2; §1 last ¶; §2; checklist 10 | "is its largest anatomic driver" → "is a recognized anatomic cause"; "the most widely deployed CT segmentation system and default back-end of 3D Slicer" → "a widely used CT segmentation system, available in 3D Slicer"; §1: "continues to account for the majority of reported wrong-level lumbar incidents [Carrino]" → "miscounting vertebral levels in the presence of an LSTV is a recognized cause [Konin; Carrino]"; "The most widely deployed … default segmentation back-end of 3D Slicer, with thousands of clinical and research installations" → "A widely used CT segmentation system is TS … available as a 3D Slicer extension"; "a widely used default" → "a widely used system whose output downstream tools can consume"; §2 drops "and the default 3D Slicer back-end"; checklist 10 "widely deployed" → "widely used" | Audit (Carrino 2011 abstract reports no wrong-level incident data; no deployment claim had a citation) |
| S10 | Trainer committed two days after training began (low, two entries) | Availability; checklist 5 | "The five fold checkpoints were trained on 3 May from a working tree at commit 50b209f of a private development repository with the then-uncommitted v20 trainer, committed there on 5 May as df10fea; they are at …. A copy of that trainer, changed only in naming and anonymization, with the conversion and inference code, is at …spinopelvic-seg (commit c105981)" | W&B metadata of the five Dataset803 runs (HEAD 50b209f, logs print v20); `git show 50b209f:tools/nnunet_wandb_variant.py` is v19.1; df10fea dated 5 May 01:26; df10fea vs c105981 differ only in naming/paths/anonymization |
| S11 | "Two PGY-2 radiology residents": the PGY level predates the re-dating of the reads (low) | §5 Castellvi | "PGY-2" dropped: "Two radiology residents, both co-authors". **Greg: confirm the readers' PGY level in September 2026 if you want it back** | No record of the training year found |
| S12 | Stale sections of this file (low) | This file | "Page budget", "Metareview conditions (binding) → where addressed", "Other camera-ready edits" and "Version note (verbatim, §5)" marked superseded; the first-pass build line marked superseded | — |
| S13 | Checklist 16 said LLMs were used only for writing assistance (low) | Checklist 16 | "LLM coding and writing assistants helped write analysis scripts, including those that recomputed reported numbers for this camera-ready, and helped draft and edit prose, the appendices among it; every reported number was recomputed from the data files." Answer stays N/A | Scripts in `neurips_fixes/`, commits with co-author lines |
| S14 | "No conflicts of interest" vs an invention disclosure in the same field (low) | Acknowledgments | **Not changed: Greg's judgement.** If the WSU invention disclosure concerns LSTV or level identification, name it in the competing-interests sentence (there are ~14 lines of slack on page 10) | Commit 2c0def3 (gitignored invention-disclosure package) |
| S15 | §6 "three-case strata are given per case" and checklist 7 held only for last_lumbar; Table 1 semi-sacralization L1–L5 rest on one patient under n = 2 (low, two entries) | Table 1 (cells and caption); §6; App. E; checklist 7 | Semi-sacralization L1–L5 cells marked ∗ ("∗One patient"); caption points to per-class n in App. E; §6: "single-record cells are marked, TS means over four or more patients carry case-bootstrap 95% intervals and our three-record lumbarization stratum is given per case (Appendix E)"; App. E gives the three lumbarization records' ensemble L1–L4 and last_lumbar per case | `ens/per_case_dice.csv`: 167 / 401_r1 / 672: L1 .805/1.000/.797, L2 .492/1.000/.403, L3 .175/.988/.006, L4 .0003/.675/.0003, LL .645/.725/.666; ensemble means .867/.632/.389/.225/.678 against checkpoint means .872/.611/.377/.213/.698. `bm`: semi L1–L5 n = 1 (token 120) |
| S16 | "stratified by subtype and acquisition mode" (low) | §7 held-out split | "stratified by LSTV class and acquisition mode" | HF README: "stratified by (lstv_class × match_type)" |
| S17 | "isolating the training-side L5/L6 merge as the component that removes the collision" overstates a comparison across different splits, maps and L6 help (low) | Abstract (iv); contribution (iv) title and body; checklist 1; §8.1 | "indicating that the training-side L5/L6 merge, not the added class, removes the collision"; contribution (iv) retitled "An unmerged control: the merge is necessary and insufficient", "indicating that the merge, not the added class, removes …", "pointing to the component of VERIDAH's success …"; checklist 1 "pointing to"; §8.1 "adding an L6 class did not remove" | Dataset802 logs (CE weight L6 4.0, L6 bias 0.60), `lstv_802.json` |
| S18 | App. B: the count of 6 is a single shared column, not reader 2's (low) | App. B last sentence | "The September sheet also records a count of non-rib-bearing vertebrae in a single column: the sheet's count of 6 (as in May) carries reader 1's annotation of 4 or 5 in 7 of 33 cases" | `docs/castellvi_readers_2026-09-04.csv` column "Non-rib/lumbar type bearing vertebra #" |
| S19 | OpenReview note opening claimed every item is in the camera-ready (low) | OPENREVIEW_NOTE opening | "The corrected text of items 1–4 is in the camera-ready itself; item 5 says which rebuttal promises it keeps and which it does not." Item 5 now also reports the added CIs; item 3 adds that the April records and split are not yet public (with an HTML-comment alternative if Greg publishes them first); item 4 adds the Sekharappa and 712-patient corrections | — |
| S20 | Table 5 Δ Hip_m pairs merged-hip Dice (ours) with TS per-side mean (low) | Table 5 caption | "(its Hip_m: mean of per-side DSC)" | FACTS_numbers §2 (TS has no merged-hip Dice) |
| S21 | "VerSe's 460 volumes, which label L6" (low) | §9 | "whose scheme includes L6 but which have no pelvis" | — |
| S22 | Change log cited the anonymous HF revision hash (low) | This file, rows 16 and "What remains" 9 | Hash replaced by "the 1 May 2026 HF revision (see FACTS_split, private)" | — |

**Offsetting cuts (page budget, no number or claim changed):** §1 ¶1 "(because lumbarization presents a
sixth lumbar body that has not been recognized)" → "(the sixth goes unrecognized)"; §1 ¶2 "and the
consequences when it does not" and "Modern radiology workflows increasingly route automated
segmentation as an upstream substrate …" removed; §1 risk mechanism wording compressed; §1
evaluation gap: the ~100 GB sentence removed (stated in §7 Hardware, §9 and App. D); §4 "The pelvic
classes remain at near-Normal accuracy …" compressed; §5 Release used "it keeps the 802 patients and
the 33 Castellvi-graded patients" → "the same 802 patients"; §5 partial annotation tag sentence
compressed; §8 headline "The other two held-out lumbarization cases are spine-only records, so" removed
(the sentence now opens with "the one junction-scorable … case"); Table 4 caption "Top block:
reproduced from those works' own test sets, not directly comparable" → "Top block: from those works'
own test sets" (the block header already says "not evaluated on CTSpinoPelvic1K"); §9 future work (2)
"replacing its VerSe-derived CT upstream" → "as its CT upstream"; Availability sentence order and
wording. The bookmark title of App. E uses `\texorpdfstring` (removes a hyperref warning).

## 2026-09-27, third pass: the promised appendices, two result slots, and text corrections

This pass integrates the appendices promised to reviewer Fbco (Background, construction figure,
dataset comparison), adds skeletons for SPINEPS under the benchmark protocol and for the v11 rerun,
and corrects the text findings of the rebuttal-promises and v11 workflows. Where it conflicts with
anything above, it is current. The full record (every before/after, sources, scripts) is
`…/scratchpad/neurips_integrate/INTEGRATE.md`; pre-edit copies are in `…/neurips_integrate/before/`.
Nothing was pushed, committed, posted or uploaded.

**Build (third pass, current).** 31 pages: content 1–10 (the acknowledgments end on page 10 at
y ≈ 667 pt of a 722 pt text block, about 5 lines of slack), references 10–13, appendices A–J 14–22,
checklist 23–31. 0 overfull boxes, 6 underfull (cosmetic: page-9 bottom, one reference URL, two
appendix page breaks, two checklist paragraphs), no undefined references or citations, no BibTeX
warnings. Fonts: 27, all Type 1 and embedded, no Type 3. All captions ≤ 60 words (largest: Tables 4
and 5, 56). **The PDF carries 134 red "[pending]" slots on pages 21–22 (Appendices I and J): do not
upload until the numbers pass fills them or the sections are cut.** Every grid job that feeds them
was PENDING at 02:00 on 27 September.

**Appendices added** (`src/appendix/appendix.tex`, after E; `\PENDING` is defined there):

| App. | File | Content |
|---|---|---|
| F | `background.tex` | Background: enumeration and transition, the six subtypes in those terms, Castellvi I–IV (definitions as restated by Konin & Walz), VERIDAH and SPINEPS, why a missing class shifts labels; footnote on TotalSegmentator's `total_v3` now points to Appendix I |
| G | `construction_figure.tex`, `figs/fig_app_construction.pdf` | Figure 2: how the April release was built (placement, orientation review, export, record types, subtypes, training view, Castellvi); one sentence narrowed to what was rendered (only token 480 is both rendered and hip-swapped) |
| H | `dataset_comparison.tex` | Table 11: VerSe, CTSpine1K, CTPelvic1K, TotalSegmentator, VERIDAH cohort, ours; licence line now gives ours per host; note g points to `total_v3` |
| I | `spineps.tex` (new) | SPINEPS 2.1.2 under the benchmark protocol: method paragraph (hips N/A; its sacrum includes S1); Table 12 per-level DSC by subgroup (all pending); Table 13 L5 DSC < 0.5 by subgroup with Wilson intervals, L6 emission, GT-L6 named as, junction DSC, for TS 2.4.0 (filled) / SPINEPS (pending) / TS `total_v3` (pending); the `total_v3` paragraph (`app:tsv3`) |
| J | `v11_results.tex` | Results on v11 (from the v11 workflow's draft; "of 112 cases" became a pending count because the same paragraph gives 115 admissible patients) |

Twelve bib entries from the drafts were merged into `bib/references.bib` (42 keys, none duplicated).
Table 13's TS column adds four TS numbers not in the reviewed paper, recomputed from
`Desktop/NeurIPS/bm/benchmark_per_case.csv` (`neurips_integrate/l5_subgroups.py`, which reproduces the
published 10/14, 9/9 and 9/750 exactly): L5 DSC < 0.5 in full sacralization 1/5 (Wilson 4–62 %),
semi-sacralization 0/1 (0–79 %), ambiguous 2/2 (34–100 %), any LSTV 22/31 (53–84 %).

| # | Where | Before → after | Why / source |
|---|---|---|---|
| T1–T6 | Abstract; §1 risk mechanism; §2; §3 (twice) | "TS … lacking an L6 class" and every general "TS has no L6 class" scoped to the audited **TotalSegmentator 2.4.0 `total` task**; §3 says "TS means this version and task throughout" | TotalSegmentator's `total_v3` task (L6 in place of S1) ships in PyPI 2.15.0–2.18.0 (from 1 Jul 2026), weights public as a GitHub pre-release on 7 Sep 2026 (app_background_NOTES §5); the audit ran 2.4.0 (`docker/Dockerfile.totalsegmentator` @ c53d2dc) |
| T5 | §3, new sentence | "TotalSegmentator 2.15.0 (1 July 2026) added a `total_v3` task with an L6 class, after this benchmark ran (30 April); Appendix I evaluates it." | Task. **`total_v3` has not been run** (see open item 19) |
| T7 | Limitations | "not as it might do if retrained with an L6 class on far more LSTV data" → "its later `total_v3` task (with L6) postdates the benchmark (Appendix I)" | An L6-capable TotalSegmentator task now exists |
| T8 | Contribution (ii) | "resolves a previously undocumented Y-axis coordinate-convention mismatch between CTSpine1K and CTPelvic1K" → "places each source mask on the acquisition it was drawn on, with its orientation corrections disclosed (Appendix G)" | No committed April code does a Y-axis affine reflection; what ran was a manual review of QC renders with 9 front-to-back flips (`configs/flip_list.json` @ 8f1ed51; app_construction_NOTES (a)) |
| T9–T10 | §5 fusion paragraph and its heading | "different dcm2niix invocations, producing a systematic Y-axis sign reflection …; we detect via affine-derived center sign and reverse along axis 1 with affine update" → the render review found nine patients whose CT and masks looked front-to-back inverted, each mirrored front to back; a mirror is not a rotation: the two rendered since (tokens 75, 480) show the liver on the left, so they are left–right mirror images, and 480's hip-label swap matched its labels to the mirrored image, not the patient. Heading "coordinate correction" → "orientation review" | Renders `neurips_promises/april_hipcheck/axial_liver_side*.png` on CTs sha-identical to April (control: liver on the right); hip sides `april_hipcheck/hip_sides.py` |
| T11 | §5 Release used | "the HuggingFace export finalized 25 April 2026, its … manifests redrawn on 1 May" → the export of 25 April, whose files changed after: on 27 April CTs that had skipped resampling into their mask's grid were rewritten and, after a TotalSegmentator laterality audit, six patients' hip labels were swapped left for right; on 1 May the unannotated voxels of the 811 single-annotation records became an ignore label and the manifests were redrawn | `export_hf.py` @ 32c580e (27 Apr) and @ 72e272c (1 May); `label_swaps` @ 32c580e / 9779823 (tokens 3, 460, 480, 616, 703, 704; audit "no-swap=0.000, swap=0.91-0.94"); April manifest (811 partial records with no background voxel) |
| T12 | §5 Release used | "Nothing here was recomputed on v11." → "Appendix J reruns TS and our ensemble on it." | Appendix J (numbers pending) |
| T13 | App. A, What is public | adds: the Zenodo v11 `splits_5fold.json` subtype map is not the six-way taxonomy (lumbarization 15, sacralization 15, semi 2, ambiguous 2, normal 768, like the unmerged control's), and the paper's map is the v11 manifest's `lstv_phenotype` (14/9/6/2/2, 769 empty) | Recounted from the saved Zenodo v11 `splits_5fold.json` and manifest |
| T14 | `references.bib` | VERIDAH `@article` with no journal ("… Sequences. 1 2026.") → `@misc`, arXiv:2601.14066 | arXiv v1 is the only version; no journal reference |
| T15 | Checklist 12 | licence statement → per host: annotations CC BY-NC-SA 4.0 in the archive of record (Zenodo metadata and LICENSE file), CC BY-NC 4.0 on the HuggingFace mirror's dataset card; checkpoints and code Apache-2.0; upstream CTSpine1K CC BY-NC-SA 4.0, CTPelvic1K CC-BY-NC-SA (paper) / CC BY 4.0 (Zenodo); TotalSegmentator "2.4.0"; SPINEPS 2.1.2 (Apache-2.0) as a zero-shot comparator | Checked 27 Sep: Zenodo v11 `cc-by-nc-sa-4.0`; HF card `cc-by-nc-4.0` (@ 72cd777); HF checkpoints and both GitHub repos Apache-2.0 |
| T16 | Checklist 4, 13 | "Y-axis affine correction" → the orientation review and its left–right consequences (Appendix G) | As T8 |
| C1 | Abstract; §1; §2; Implications; checklist 10 | "5–35 % prevalence" → "4–30 %" | **Reviewed number, corrected.** No checked source states 5–35 %; Konin & Walz (cited) give 4–30 %, which Appendix F uses |
| C2 | §2 Benchmarks | "VerSe (460 CT volumes) is the only public CT benchmark whose label vocabulary includes L6 …" → "VerSe (374 CT scans) and CTSpine1K both label L6 (label 25), but neither reports LSTV-stratified results and neither has pelvic GT" | **Reviewed number, corrected.** VerSe is 374 scans of 355 patients (460 = VerSe'19 series + VerSe'20 subjects); CTSpine1K labels C1–L6 as 1–25 |
| C3 | §9 future work | "VerSe's 460 volumes … 1,613-volume" → "374 scans … up to 1,527 volumes" | Follows from C2 |
| C4 | Contribution (ii); §5 | "the COLONOG cohort that underlies VerSe and CTSpine1K"; "VerSe-derived per-vertebra masks" → "annotate the COLONOG cohort"; "per-vertebra masks in the VerSe label convention" | VerSe is a TUM PACS cohort, not COLONOG (Liebl 2021); CTSpine1K masks were pre-labelled by a network trained on 41 VerSe cases, then corrected |
| C5 | §2 VERIDAH | cite `Meng2022VertebraeMLMI` (MLMI 2022) for the 77.81 % instead of the CMIG 2023 version | VERIDAH's comparator is its ref. [16], the 2022 paper |
| C6 | §2 VERIDAH | "upstream centroids (VerSe-derived models or SPINEPS)" → "vertebra locations from an upstream instance segmentation (for MRI, SPINEPS, whose CT release 2.1.2 we score in Appendix I)" | VERIDAH's CT locations were in-house instance annotations |
| C7 | §5 Castellvi | "released with the data" → "released with the data from Zenodo v8 on" | The April files carry no Castellvi field |
| C8 | §2 | adds "Appendix H [compares] CT collections" | Pointer to the promised comparison |

**Offsetting cuts (page budget):** §2 VERIDAH architecture detail (four heads, DenseNet169, terminal
double-L5) now lives in Appendix F and the main text points there; Table 2 caption drops
"zero-shot" (stated in §3). No number or claim left the paper.

## Page budget

> **Superseded on 2026-09-27.** This section describes the 25 September build (21 pages, no
> appendices). The current page map is in "2026-09-27, second pass" above: 25 pages, content 1–10,
> references 10–12, appendices A–E 13–16, checklist 17–25.

**2026-09-25 afternoon rebuild.** With Patrick Schwing's affiliation filled and the public-link sentence, the acknowledgments had slipped onto page 11 (the 10:33 PDF predated the link edit). Tightened without changing a number or claim: the availability paragraph (one spinopelvic-seg URL instead of two, the release note as a section reference), the unmerged-control and cascade paragraphs, Interpretation, Generalization, Implications, future-work item (2), the Limitations fold-pooling sentence, and the acknowledgments. Build: 21 pages, content pages 1-10 with the acknowledgments ending page 10, references 11-12, checklist 13-21, zero overfull boxes.


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

> **Superseded on 2026-09-27.** This table describes the 25 September text. Several of its
> statements no longer hold, among them "Train / Test … total 26/7" (now Pool/Test 27/7, with five
> patients on both sides), "§6 states the 805-record training pool" (the pool is 979 records),
> "each read all 33 independently" (the reads are dated by 4 September 2026, part of reader 2's
> from 23 August, and the readers had seen the May grading), "Every quantitative claim … scoped"
> (now "Castellvi-stratified statements"), "7/7 → 65–100 %" (now 4/4 with last_lumbar GT,
> 51–100 %), "same five folds" for the unmerged control (different split) and the 84.7–99.4 %
> figure. The two 2026-09-27 sections above say where each condition is now addressed.

The metareview's own wording is quoted in the first column, except that the word it uses for the
origin of the labels is paraphrased; that word is not used anywhere in the paper or in this file.

| Metareview / reviewer point | Where addressed | Notes |
|---|---|---|
| State where the labels come from, and their limits (metareview (e), Final Justification; paraphrased) | §5 paragraph *Six-way LSTV taxonomy, and where its labels come from* (lines ~360–385); §5 *Castellvi annotation layer* (lines ~405–430); §8 *Limitations*; abstract (ii); contribution (ii); checklist items 2, 12 | States plainly: no subtype was assigned from an image read; each is a function of the CTSpine1K lumbar-label count and the CTPelvic1K filename qualifier (whose origin CTPelvic1K does not document); two of six subtypes rest entirely on a filename tag. Castellvi: two PGY-2 radiology residents, each read all 33 independently, 6 disagreements resolved by consensus, no attending/neuroradiologist read, no blinded re-read, no screening of the normal cohort. |
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

> **Partly superseded on 2026-09-27.** De-anonymisation, affiliations and funding still hold. The
> URL list (trainer at spinesurg-ct-nnunet 50b209f), the caption word counts and the Castellvi
> table note describe the 25 September text; see the 2026-09-27 sections above.

* **De-anonymised**: `\usepackage[final,eandd]{neurips_2026}`; footer reads "40th Conference on
  Neural Information Processing Systems (NeurIPS 2026). Track on Evaluations and Datasets."
  Title matches OpenReview exactly. Authors in OpenReview order with affiliations from
  `paper/mpda/title_page.tex` (Gregory: Department of Surgery, DMC and WSU; Ismoilov and
  Alnabahneh: Department of Radiology, DMC and WSU) and, for Loren Schwiebert, Department of
  Computer Science, Wayne State University (from the IRB filing in
  `spinesurg-ct-nnunet/docs/IRB/`). Corresponding-author footnote:
  `gregory.schwing@med.wayne.edu`.
* **Patrick Schwing's affiliation** (`\PatrickAffil`, line ~31): School of Medicine, Louisiana
  State University Health Sciences Center, New Orleans, Louisiana, USA — supplied by the
  corresponding author on 2026-09-25 (it is recorded in no project file; the red placeholder of
  the 10:33 build is gone).
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
  (Ib 1, IIa 2, IIb 4, IIIa 3, IIIb 19, IV 4) matches a single-column sheet of
  6 May 2026 (`Desktop/NeurIPS/Lumbosac_Castelvii_Class.csv`; [corrected 2026-09-27: the files do not
  record who filled it, and it agrees with reader 2 on 30/33 and reader 1 on 28/33, so "the first
  reader's preliminary sheet" was not supported]), not the consensus released in
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

> **Superseded on 2026-09-27.** The quotation below is the 25 September text and is no longer in
> the paper. The current §5 says "Every *segmentation* number …", drops "later labelled v1" (no
> public revision carries that label), states that neither the April records nor the evaluated
> split is yet at a public location the paper cites, dates the Castellvi reads to 4 September 2026,
> and moves the v11 differences to Appendix A.

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

1. ~~Patrick Schwing's affiliation~~ — done 2026-09-25 (LSU Health Sciences Center, New Orleans, School of Medicine); recompiled.
2. ~~`spinesurg-ct-nnunet` is private (404)~~ — done 2026-09-25: the Data and Code Availability sentence cites only the
   public spinopelvic-seg copy.
3. **OpenReview camera-ready form**: upload `src/neurips_2026.pdf` ([updated 2026-09-27, second
   pass] 25 pages: 10 content + references + appendices A–E + checklist); update Dataset URL to the Zenodo concept DOI / HF mirror, Dataset
   Large URL (the anonymous `-Sample` repo has no public counterpart in the repo — decide), Code
   URL to `github.com/OpenSpineConsortium/spinopelvic-seg`; re-upload the **Croissant file**
   with the public dataset URL (regenerate with `zenodo/make_croissant.py`; [corrected
   2026-09-27: v11 has no `croissant.json`, only v6 does]); the "Financial Support" field lists the first author — the paper
   now discloses the NIH F30 fellowship (F30GM147909) for G.J.S. and no other funding; if the
   form has a funding field, it should say the same.
4. [Third pass: Background, construction figure and comparison table are now Appendices F–H;
   SPINEPS is Appendix I with its numbers pending; see items 19–20.] **Decide on the rebuttal promises not delivered**: SPINEPS under the protocol, a Background
   section, a dataset-construction figure, the dataset-comparison table. None is a metareview
   condition. [2026-09-27: drafts exist in scratch; each can be added to
   `src/appendix/appendix.tex` with one `\input` line after appendix E. If they are, switch the
   Fbco bullet of the OpenReview note to its commented alternative.]
5. ~~If the rebuttal's "84.7 to 99.4 %" has a source …~~ — settled 2026-09-27 (FACTS_split §5):
   it matches only a 27 Jul 14:02–14:06 reading of folds 0 and 4 at about epoch 256 of 500; the
   paper now says so in one clause (§8.1) and in Appendix C.
6. Optional: reconcile the Zenodo README's Castellvi paragraph and the MDPA "four of six"
   sentence with `docs/castellvi_consensus.csv` (both outside this folder).
7. Cite the paper from the MDPA article with `neurips2026.bib` (key `neurips2026lstv`).

### Added 2026-09-27

8. **Post the OpenReview note** drafted at
   `…/scratchpad/neurips_fixes/OPENREVIEW_NOTE.md` (unmerged-control figures, Castellvi re-grading
   after review, record-level split, rebuttal promises). Not posted.
9. **Publish the evaluated split.** The record lists behind every held-out number survive only in
   an HF repository history (the 1 May 2026 revision; see FACTS_split, private) that the paper does
   not and should not cite. Adding `splits_final` / the 174 held-out case ids to spinopelvic-seg or
   a Zenodo v12 would let the checklist's "not yet public" become "public". [2026-09-27 second
   pass: this is now the top open item; see item 16.]
10. **Croissant in the current version.** v11 carries no `croissant.json`; the paper now says the
    file is in v6. A v12 with a regenerated Croissant file would let the availability sentence
    point at the current version.
11. **Zenodo v11 README** still describes the August state of the Castellvi layer ("one reader's,
    with five cases independently read a second time", Ib 2 / IIa 2 / IIb 4 / IIIa 3 / IIIb 18 /
    IV 4) and names a `castellvi_second_read` field the manifest does not have.
12. **Decide** whether to report the sensitivity numbers (patient-disjoint held-out records,
    FACTS_split §4) and whether to print the 215 voxel share (95.3 %, FACTS_numbers §6, a 27 Sept
    rerun). Neither is in the paper. [Second pass: the case-bootstrap CIs on TS Dice are now in
    §3 and Appendix E.]
13. **E&D FAQ**: a dataset that "changes in a way that invalidates the original reviews" between
    submission and camera-ready may be removed. The Castellvi grades changed after review (five of
    33); the paper and the OpenReview note disclose it. Greg's judgement whether to ask the AC.
14. The public trainer copy at spinopelvic-seg c105981 still carries the anonymised citation
    ("Anonymous 2026 (VERIDAH §2.2)") and `anonymous-neurips-ED` as author; de-anonymise when
    convenient.
15. Greg's affiliation line (Department of Surgery vs General Surgery) is untouched.

### Added 2026-09-27, second pass

16. **Top item: publish the April release and the evaluated split before upload, or accept the
    checklist's "No".** The E&D call says "All accepted papers must have their code and datasets
    documented and publicly available by the camera-ready deadline." v11 is public; the April
    release's 1,153-record manifests (with the record-to-series map) and `splits_final` with the
    174 held-out ids are not at any location the paper cites. Checklist 5 now answers **No** for
    that reason. If they are published (Zenodo v12, an OSC HF tag, or files in spinopelvic-seg,
    ideally with a regenerated Croissant file), make these edits and rebuild:
    - §5 "Release used": replace "Neither these records nor the evaluated split is yet at a public
      location this paper cites (Appendix A)." with "Its manifests and the evaluated split are at
      [LOCATION] (Appendix A)."
    - Availability, last sentence: replace "the April release's records and the evaluated split are
      not yet public (Appendix A)" with "the April release's manifests and the evaluated split are
      at [LOCATION]".
    - Appendix A *What is public*: replace the first and last sentences accordingly.
    - Checklist 4, 5, 13: answer 5 back to Yes, and replace the "not yet public" clauses with the
      location.
    - OpenReview note item 3: switch to the commented alternative.
17. **Competing interests.** The acknowledgments say "No conflicts of interest". If the WSU
    invention disclosure (commit 2c0def3 gitignored its package) concerns LSTV or vertebral-level
    identification, name it there; page 10 has about 14 lines of slack [third pass: about 5].
18. **Readers' training year.** "PGY-2" was dropped from §5 because it predates the re-dating of
    the reads; restore the correct PGY level if you want one there.

### Added 2026-09-27, third pass

19. **Do not upload while any "[pending]" remains** (134 slots, Appendices I and J). Fill them from
    the SPINEPS report (`.../neurips_spineps/score_v212/REPORT.md`, jobs 40481323 → 40481314 →
    40481315; only if it shows 0 ground-truth-gate mismatches) and the v11 summaries
    (`/rs/rs_grp_oschome/go2432/neurips_v11/out/{ts,ens}/SUMMARY.md`, jobs 40480728/31/32 and
    40480730/33), or cut the sections and the sentences that point to them (INTEGRATE.md §5).
20. **TotalSegmentator `total_v3` has not been run; no job was submitted.** The main text now says
    "Appendix I evaluates it" (§3) and points there from the Limitations, the Appendix F footnote
    and Appendix H note g. Run it (the scout's plan: the 33 LSTV patients' records and the 174
    held-out records, TotalSegmentator ≥ 2.15.0 with the `v3.0.0-weights` pre-release) or cut the
    column and change those four sentences to "not evaluated here" (exact wording in INTEGRATE.md §5).
21. **Headline defects A and B (v11 workflow) are still in the main text.** The TotalSegmentator
    sacrum 0.817 (abstract "0.817 → 0.969", §8 "+15.2") counts TotalSegmentator's separate S1 class as
    a miss, and the junction comparisons subtract two window definitions. Appendix I now says the
    first in one clause and Appendix J carries the corrected-row slot, so the appendices disclose
    what the main text has not yet corrected. Decide before upload.
22. **Licence alignment.** The paper now states the licence per host because they differ: the
    HuggingFace mirror's dataset card says CC BY-NC 4.0, the Zenodo archive (and the repo's own
    `zenodo/hf_dataset_card.md`) CC BY-NC-SA 4.0. Re-uploading the card with `cc-by-nc-sa-4.0`
    would let checklist 12 give one licence.
23. **Cover letter to Medical Physics.** Appendix G's figure and Appendix H's table are captioned
    "Adapted from" the dataset article (arXiv:2609.22760); `paper/mpda/cover_letter.md` says the
    NeurIPS paper shares no text, figure or table with it. Change one of them.
24. **The mirror images.** At least tokens 75 and 480 are left–right mirror images of the patient in
    the April and v11 CTs (7 of the 9 flipped patients were never rendered), and 480's hip labels
    follow the image. The paper now discloses it; fixing v11 (rotate, re-derive laterality) and
    correcting MDPA and `zenodo/KNOWN_ISSUES.md` are separate work.
25. **OpenReview note item 5** still says the promised appendices are not in the camera-ready; switch
    it to its commented alternative (SPINEPS and v11 numbers pending).
26. Flagged by the drafts' notes and **not** changed (reviewed text): "an nnU-Net trained on 1,204
    (routine) CTs" (1,204 is the v1 dataset); "the LSTV-stratification overlay neither source
    provides"; CTSpine1K's own L6 Dice of 0 as prior evidence for the unmerged control (cited only in
    Appendices F and H); "On a private CT cohort of 1,536 volumes … 99.18 %" (365 test CTs);
    `Deng2021CTSpine1K:Tomography` year and volume against Crossref (MELBA 2025).

### Camera-ready built 2026-11-06 (upload candidate: NEURIPS_CAMERA_READY_3474.pdf)

29 pages: content to p.10, acknowledgments and disclosures p.10, references p.11, checklist p.21,
appendices A-H. Zero undefined references, zero overfull boxes, no [pending] anywhere, authors named.

**Appendices I (SPINEPS) and J (v11 results) were cut.** Their numbers were to be filled from grid runs
submitted 27 September; those runs left logs and no results, and both output directories
(`neurips_spineps/ctsp_april_v212`, `neurips_v11/out`) are empty on the volume today. Every cell would have
printed a red "[pending]". The area chair's final justification says in terms that new data, more baselines
and a reader study are not held against the paper, so the SPINEPS comparison promised to reviewer Fbco was
never a condition of acceptance. Seven pointers were retired with them, using the wording the source files
prescribe in their own comments: Section 3 and the Limitations (total_v3 "is not evaluated here"), related
work (the SPINEPS scoring clause), Section 5 ("Nothing reported here was recomputed on v11"),
`background.tex` twice, note g of `dataset_comparison.tex`, and checklist item 12 (SPINEPS is now described
as a related-work reference, not a scored comparator).

**What the cut would have taken with it, and did not.** Appendix J was the only place the paper disclosed
that the sacrum comparison is not like-for-like. That disclosure is now in the Limitations, in the main
text: TotalSegmentator emits the S1 body as a separate class, which the benchmark's sacrum score counts as a
miss, so 0.817 understates it and the +15.2-point margin is an upper bound; the level-shift findings, which
concern lumbar labels, do not depend on it.

**Three things left for a decision, none of them blocking the upload.**

1. *Defect B, the junction windows.* The note of 27 September says "the junction comparisons subtract two
   window definitions" and nothing anywhere states which two or where they differ. No disclosure of it was
   written, because a vague one is worse than none. If the junction numbers (0.824 -> 0.534, the 29-point
   drop in the TL;DR) rest on mismatched windows, that needs a sentence; the sentence cannot be written
   from what is recorded.
2. *Checklist 5 remains No.* The dataset, code and weights are public; the April release's 1,153-record
   manifests and the 174-record held-out split are not, so the segmentation results cannot be reproduced
   from public files alone. Publishing those two small files at a cited location would flip it to Yes
   (CAMERA_READY_CHANGES item 16 lists the five edits). The split could not be verified as the exact one
   evaluated, and publishing the wrong one would be worse than the honest No.
3. *Croissant.* The checklist says the file lives in Zenodo v6 and that v11 carries none; the track asks for
   an up-to-date, valid Croissant file at the camera-ready deadline. The OpenReview record has one attached.
