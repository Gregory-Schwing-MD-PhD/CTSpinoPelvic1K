# Finalising a CTSpinoPelvic1K release

What has to be true before a version is published, in the order it has to become true.
Nothing here is ceremony: each item exists because its absence has produced, or would
have produced, a wrong number that looked finished.

The governing rule is in [DATASET_PRINCIPLES.md](DATASET_PRINCIPLES.md). This document
is the operational form of it.

---

## 0. The one rule that orders everything else

**Gates run before measurements, and a failed gate stops the chain.**

Morphometrics computed on an inconsistent corpus do not look broken. They look
finished — a tidy table of medians, exit code zero. That is exactly what happened on
2026-08-20: pelvic incidence came back at 154.6° for all 802 cases, sacral slope at
82.9°, and the run reported them without complaint. `slurm/finalize_release.sh` now
runs the invariant check first and exits before computing anything if it fails.

---

## 1. Data completeness

| item | state | blocker |
|---|---|---|
| Rib numbering offsets | **closed** — 9 of 11,552 ribs resolved across 5 cases | — |
| Thoracic vertebrae on FOV-limited cases | done | — |
| Lumbar rib class (74/75) | applied release-wide | — |
| Hardware classes (76–79) | **defined but unpopulated** | `0068` needs hand annotation; see the note below |
| `pelvic_native` cases never reviewed | 6 outstanding: 0090, 0196, 0236, 0419, 0877, 1037 | reviewer time |
| — *four of those six are among the worst offenders in the physiological-envelope check (§2.0), so reviewing them is worth more than the count suggests* | | |
| Stray voxels, 0344 | **closed** — release-wide speckle sweep finds none | — |
| Vertebra-label speckle, release-wide | **closed** — 1 of 802 carried any (0412, 120 fragments / 160 voxels), stripped | — |

**On populating 76–79 automatically.** `scripts/detect_metal.py` finds every implant in
the corpus and classifies the construct, and its categories map onto the declared classes
almost directly — an elongated component spanning vertebrae is 78, a component inside a
disc space is 77. It would be easy to write those labels straight into the release, and
that would be a mistake. Metal blooms: the boundary between an implant and the bone it is
fixed to is not decidable from attenuation, which is exactly why `0068` is listed as
needing a *hand* annotation rather than a threshold. The census is a worklist for that
annotation and a stratification for templating; it is not the annotation. Anything written
from it goes to a separate directory for review, never over `data/v5_final`.

A class that is declared but carries no data must be described that way everywhere it
appears — in the label table, the dataset card, and any figure. The website gallery says
"pending annotation" on its instrumentation card for exactly this reason: a card
claiming segmented hardware would have been claiming something the release does not
contain.

**Deferred cases are listed in [DEFERRED_CASES.md](DEFERRED_CASES.md) and must be
excluded from derived measurements until resolved, not silently included.**

---

## 2. Automated gates

Run by `slurm/finalize_release.sh`, in this order.

### 2.0 Physiological envelope — `scripts/qc_physiological_envelope.py`

Every extractor already checks its **median** against a published value. That catches a
measurement wrong for everyone and cannot catch one right for 799 cases and absurd for
three, because three cases do not move a median — yet those three are what a reader sees
in the tail of a histogram and what a downstream model trains on without comment.

So this asks whether any individual value is outside what the anatomy permits: a pedicle
27 mm wide, a canal 0.7 mm across, a sacrum 22 cm tall. Bounds are deliberately generous,
so a hit is a case to open rather than a distribution to argue about.

Values are **not deleted**. The extractors record what they measured; this names what
should not be believed, so a user can exclude it knowingly. A value silently dropped is a
value nobody can audit.

### 2.1 Release invariants — `scripts/check_release_invariants.py`

**Re-run 2026-08-22 after the speckle strip modified `0412`: 802/802 pass.** Any edit to a
released label re-opens this gate; that is the point of having it run first.


Every case, four questions with exactly one right answer:

- **Geometry.** Label shape, affine and voxel spacing agree with the CT.
- **Ids.** No label id outside the published scheme.
- **Emptiness.** The label is not blank.
- **Sidedness.** Left-side ids and right-side ids fall on the correct sides of the
  midline, with the left–right axis read from `nib.aff2axcodes` rather than assumed.

The geometry check exists because a renumbering pass loaded labels through
`as_closest_canonical` and wrote the reoriented array back under the canonical affine,
transposing one label away from its CT. Renumbering is pure id arithmetic and never
needed the reorientation. Nothing in the pipeline noticed; it surfaced only when a human
opened an unrelated case and ITK-SNAP refused the pair.

The sidedness check exists because it is nearly free, and it catches a transposition
that happens to preserve the array shape — which the geometry check cannot.

> A check that fails almost everything is accusing itself. The first version of the
> sidedness test reported 801 of 802 cases as defective; it had the RAS convention
> backwards and was reading the wrong axis. Before believing a mass failure, verify the
> check against cases a human has already reviewed.

### 2.2 Rib–vertebra incidence — `scripts/qc_rib_vertebra_incidence.py`

Reports every rib whose nearest vertebra is not the one its number implies. Interpreting
the output requires knowing the three failure modes, which look identical in the CSV and
have completely different fixes:

| signature | what it is | fix |
|---|---|---|
| blank `gap_own_mm`, rib falls to the vertebra below | the rib's own vertebra is **missing or truncated** | draw the vertebra, or accept if it is cut by the FOV |
| hundreds of connected components, all within ~1 mm of a neighbouring rib | **speckle**, not a rib | engulf per component (`scripts/postprocess_halo.py`) |
| whole side consistently one level out | genuine **off-by-one** | shift that side |

### 2.3 Morphometrics plausibility — built into `extract_surgical_morphometrics.py`

The script holds published adult ranges for the measures that have them, flags any
median outside, and exits 2. Ranges must be for the **right anatomy**: the Torg–Pavlov
ratio is ~1.0 in the cervical spine but ~0.5 in the lumbar, and applying the cervical
figure condemns a correct measurement.

---

## 3. Splits

Patient-grouped and LSTV-stratified, blind to which source a label came from. **A
release that adds or changes cases needs a fresh split**, and the split file must be
frozen and shipped with the data — comparisons across papers are worthless otherwise.

---

## 4. Documentation

- `README.md` — doubles as the HF dataset card; front-matter must match the licence
- Label table — every id, including any that are declared-but-empty, marked as such
- [DEFERRED_CASES.md](DEFERRED_CASES.md) — what is excluded and why
- [CORRECTIONS.md](CORRECTIONS.md) — corrections made to the source labels
- Release notes — what changed since the previous version, and what a user must re-run
- **Limitations, stated positively.** What the release does not support is part of the
  release. Thoracic ground truth is FOV-limited (roughly T8 down, not T1); lordosis is
  supine; hardware is unpopulated; LSTV labels come from more than one source and are
  not all expert-adjudicated.

---

## 5. Deposit

Target for the data descriptor is a Medical Physics Dataset Article, which requires a
citable archive, not a model hub alone.

1. **Zenodo** — mint the DOI. This is the citable object.
2. **Hugging Face** — `anonymous-mlhc` for the anonymous review copy; the token lives at
   `~/.hf_org_token` and is read with `HF_TOKEN=$(cat ~/.hf_org_token)`, never pasted.
3. **TCIA** — for provenance back to the source imaging collection.
4. **GitHub** — code and documentation, tagged at the release commit.

Every one of them must name the same version and the same DOI.

### Licence and attribution

**Resolved 2026-08-22, and the answer was that the declaration was wrong.**

| source | licence |
|---|---|
| TCIA CT COLONOGRAPHY | CC BY 3.0 |
| CTPelvic1K (Zenodo 4588403) | CC BY 4.0 |
| CTSpine1K | **CC BY-NC-SA 4.0** |

CTSpine1K carries **ShareAlike**, and the vertebral labels in this release are an
adaptation of it. ShareAlike obliges an adaptation to carry the same terms, so the
release is `cc-by-nc-sa-4.0`. The front-matter said `cc-by-nc-4.0`, which drops the SA —
that is a violation, not a choice, and it would have been minted into an unretractable
DOI. Front-matter and both licence sections now say NC-SA and name each source with its
own terms.

The other two impose no obstacle: CC BY has no ShareAlike, so a downstream NC-SA is
permitted provided attribution is given.

**Two things still outstanding, and the first is live right now.**

1. **One of the two published HuggingFace datasets is corrected; the other is not.**
   `gregoryschwingmdphd/CTSpinoPelvic1K` now declares `cc-by-nc-sa-4.0` (one line changed,
   nothing else touched). `anonymous-mlhc/CTSpinoPelvic1K` still declares `cc-by-nc-4.0`
   and could not be changed from here: the token on this machine is scoped to the user
   namespace, and the org refuses even a pull request. **It needs an org member to edit
   `README.md` on the web UI and change that one line.**

   *Superseded note, kept for the record —* **The two published HuggingFace datasets still declare `cc-by-nc-4.0`** —
   `gregoryschwingmdphd/CTSpinoPelvic1K` (public since 2026-05-25) and
   `anonymous-mlhc/CTSpinoPelvic1K` (public since 2026-06-27). Those cards carry the
   violation described above and it is public. The repository is corrected; the published
   cards are not. This was deliberately left for Greg rather than pushed: a licence
   declaration on someone's own public dataset is his call and his exposure, and the
   anon-org token needed for the second repo no longer exists on this machine. The edit
   is one line of card front-matter in each.

2. **Institutional sign-off on this reading**, before the DOI is minted. It is a reading
   of three licence texts, not legal advice.

---

## 6. Reproducibility

- Tag the commit; every derived artefact records the commit that produced it
- Containers pinned by digest, not by tag
- Every heavy step is a batch job, never a login-node process
- Manifests, not remembered numbers: "how many cases were cleaned, and which" must be
  answerable from a CSV rather than from a conversation

---

## 7. Before you publish, the honesty pass

Read the dataset card as a sceptical reader who wants to find something overstated.

- Does every number in it come from a run that is still current, or is any of it from a
  pipeline version since fixed?
- Is any class described as populated when it is empty?
- Is any measure presented without the caveat that limits it (supine, FOV-limited,
  count-free)?
- Does any figure assert a level name that a spine-limited field of view cannot support?
- Is any null reported as a positive finding, or any positive finding reported without
  its effect size?

The last one has already caught something once: a "positive control" panel asserted that
pelvic incidence must separate by sex. Measured, the difference was 0.8° — the claim was
wrong, not the measurement, because pelvic dimorphism is strong in shape rather than in
incidence. A null stated with its effect size is a result; an unexamined adjective is
not.

## Found 2026-08-23, by checking the released volumes rather than the working copy

Everything below was established by counting identifiers in all 802 released label volumes
(`scripts/label_census.py`, now folded into `measure_tp_height.py`) or by reading the
released manifest and splits. None of it is visible from the working copy on the grid.

| item | state | what to do |
|---|---|---|
| **`0787` lumbar rib** | working copy says bilateral lumbar rib, 35.2 mm; released volume has no class 74 or 75 | one of the two is wrong. Decide which, then either re-export the case or correct the morphometrics. The paper now says 15 records, matching the release |
| **`has_l6` in the manifest** | wrong in both directions: true for one record that has no L6; false for all 17 that do | repopulate from the label volumes, or drop the field. A declared field that is wrong is worse than an absent one — the same failure `castellvi_type` had |
| **`n_lumbar_labels`** | 0 in 795 of 802 records including every LSTV case | not a count of lumbar labels. Repopulate or drop |
| **no held-out test set** | `splits_5fold.json` is five disjoint folds whose union is the cohort | either ship a test split or keep the release as-is and let the paper say so, which it now does |
| **`castellvi_type`** | **done** — populated on `v5` and `main`, 33 grades, 5 second reads | — |
| **structures not universally present** | 1 record without sacrum/hips/femora, 2 without S1, 2 without T12, 9 without L5 | expected (field of view, or the lowest segment labelled L6/absorbed). Documented in the paper's limitations; no action unless a case is genuinely defective |
| **ids 58–73** | confirmed empty in all 802; block retired 2026-09-03 (bone + hardware scheme only) | none — `label_scheme.RETIRED_IDS` keeps the gap unassigned |
| **`tp_height` in the morphometrics** | was the tip-slab extent, so speckle set it; 169 values overstated by >5 mm, several by >25 mm | **fixed** — largest connected component, spliced in by `merge_tp_height.py`, old values kept as `*_prefix_mm`. Any analysis run before 2026-08-23 that used `tp_height_*` needs re-running |

### Still blocking

- Zenodo DOI is a placeholder (`10.5281/zenodo.XXXXXXX`). Reserve before the final draft.
- IRB determination letter does not exist; the paper's Ethics section says PENDING in the
  text and must not be submitted as written.
- Co-author conflict-of-interest statements: the paper asserts none for all authors and
  only Greg has been asked.
- Twelve of sixteen references still have unverified volume/page fields; the three checked
  are named in `main.tex`.
- The `anonymous-mlhc` dataset card licence still needs the CC BY-NC-SA 4.0 sign-off.

## Closed 2026-08-23 (overnight)

| item | what happened |
|---|---|
| **`castellvi_type` null in the release** | **closed** — 33 grades populated on `v5` and `main`, 5 with a second read |
| **`has_l6` wrong in both directions** | **closed** — recomputed from the label volumes; 17 true, was 1 (and that one had no L6) |
| **`n_lumbar_labels` vestigial** | **closed** — recomputed; 776 with five, 17 with six, 9 with four |
| **`has_lumbar_rib` absent** | **closed** — added; 15 records |
| **`patient_size`, `postwrite_hip_bone_pct`** | **closed** — declared in all 802, populated in none, removed |
| **four records with transposed hips** | **closed** — 0027, 0107, 0935 and 0790 corrected and re-uploaded. The pooled check found only three; 0790 needed the per-pair check. The release invariant check tested ribs only, which is why it never saw them; it now compares each sided pair separately |
| **version-progression QC** | v2, v5pre, v5 complete. v3 running with a 24 h wall clock after the 6 h one proved far too tight — it is the slowest version because it has ribs that have not yet had speckle removed |
| **paper builds locally** | `paper/mpda/build.sh`, TinyTeX in WSL; 10 published pages, no errors, no overfull boxes |
| **Zenodo deposit** | assembled at `data/zenodo_deposit`, 1.83 GB, all readiness checks pass. `zenodo/upload.py` creates the draft and reserves a DOI but does **not** publish |

### Still open, and all of them need Greg

- **Reserve the Zenodo DOI** and replace `\datasetdoi` in `main.tex`. Run
  `python zenodo/upload.py --dir data/zenodo_deposit --reserve-doi` with `ZENODO_TOKEN` set.
  Reserving is reversible; publishing is not, and the script deliberately stops short of it.
- **Co-author list.** `\author{[co-authors]}` is still a placeholder, and the conflict-of-
  interest statement asserts none for *all* authors while only Greg has been asked.
- **IRB determination.** The Ethics section says PENDING in the text and must not be
  submitted as written.
- **Repository URL** in Data Availability.
- **Twelve of seventeen references** still have unverified volume/page fields; the five
  checked are named in `main.tex`.
- **`0787`** — the working copy calls its rib a lumbar rib, the released volume numbers it
  rib 12. One is wrong. The paper follows the release and says 15 records.
- **Neuroradiologist confirmation** of the 33 Castellvi grades.

## v8 — staged, not published (2026-09-03)

The manuscript cites v7 and must keep citing it. These changes are staged locally in
`data/zenodo_deposit/` (the labels there are the published v7, SHA-256 verified) and go out
together as v8 when Greg says so:

| change | where staged | why |
|---|---|---|
| drop the 58–73 soft-tissue names from `dataset_labels.json`; note the gap is unassigned | `data/zenodo_deposit/dataset_labels.json`, `README.md` | the scheme is bone and hardware only; v7's descriptor still lists names no record uses |
| same on the HuggingFace card | `data/hf_export_v5/dataset_labels.json` (card text on HF not yet edited) | one scheme everywhere |
| `pelvic_native` → `pelvic_only` in the manifest `config` field | not yet | one name for the 20 pelvis-only records |
| Castellvi second-read reconciliation | not yet | title page / manuscript / CSV disagree on who read what |
