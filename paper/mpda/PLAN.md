# CTSpinoPelvic1K — Medical Physics Dataset Article

Planning document. Requirements below are from the AAPM *Medical Physics* MPDA policy
([editorial guidelines](https://www.aapm.org/pubs/MPJ/policies/details.asp?id=3596&type=PUB),
and Williamson et al., *Introducing the Medical Physics Dataset Article*,
[Med Phys 2017;44:349](https://doi.org/10.1002/mp.12003)).

---

## The three that change what we do, and one of them blocks submission

### 1. The DOI must exist BEFORE the manuscript is submitted

> "Authors must place the dataset in a recognized and stable data archive prior to
> submission... The repository must provide a Digital Object Identifier (DOI) that is
> referenced in the MPDA."

Archiving is not a post-acceptance step. It gates peer review. **Zenodo first, then
write.** Zenodo can *reserve* a DOI on an unpublished draft, so the number can go into
the manuscript before the record goes public — but note that deleting the draft destroys
the reserved DOI and its files.

An institutional website is explicitly "generally not suitable". Neither GitHub nor
Hugging Face satisfies this on its own. The archive has to guarantee >10-year
persistence.

### 2. No hypothesis testing. At all.

> "Should not include hypothesis testing or data analyses supporting generalizable
> conclusions."

This rules out significance tests, and it rules out framing any finding as a result.
It does **not** rule out what we have: comprehensive descriptive analysis is *required*,
and graphical visualisation is "encouraged".

So the bimodality in the lowest-rib ratio may be **shown and described** — two modes near
0.69 and 0.33 — and must not be tested, attributed to a mechanism, or called evidence
for anything. The same applies to every gallery panel. Validation experiments *are*
allowed, which is the right home for the plausibility checks: PI−LL centring on zero is
a consistency check on the measurements, not a claim about spines.

### 3. Ten published pages

Everything below has to fit. The figures earn their place or they go.

---

## Structured abstract — required, four parts, under 300 words

| section | must contain |
|---|---|
| **Purpose** | what the dataset is, its scope, who it is for, what it is for |
| **Acquisition and Validation Methods** | the population, how data were produced, how they were validated |
| **Data Format and Usage Notes** | data types, subject count, formats, how to get it, repository link |
| **Potential Applications** | scientific and clinical uses, **and the important limitations** |

---

## Reviewers, and what they will actually do

Two, with different jobs:

**Domain expert** — judges novelty, impact, relevance to current questions, and whether
quality suffices for validation or for opening new questions.

**Data curation expert** — *downloads the dataset from the DOI*, checks the files open,
runs the tools, verifies randomly selected data structures, and confirms it is reusable
for the stated purpose.

The second reviewer is why the release checklist matters more than the prose. Anything
that fails to load, any header that disagrees with its documentation, any tool that
errors on a fresh checkout, is a finding. This is also why the invariant check exists:
`scripts/check_release_invariants.py` verifies the exact property a curation reviewer
would notice first — a label that does not match its image.

**Grounds for rejection to avoid:** obsolete imaging devices, poorly controlled
acquisition, or insufficient annotation for hypothesis-driven research.

---

## Licensing

> "Completely freely available... limited restrictions for commercial use but none for
> scientific purposes."

CC-BY-NC-4.0 — the current declaration — fits that sentence as written: unrestricted for
science, restricted for commerce. **But this is not the binding constraint.** The release
inherits terms from TCIA CT COLONOGRAPHY, CTSpine1K and CTPelvic1K, and the most
restrictive of those wins. That has to be confirmed before the DOI is minted, because a
DOI is not retractable.

Also required: complete PHI removal, and IRB documentation covering the human data.

---

## Format note

The policy prefers DICOM "if applicable". This release is NIfTI, which is the correct
format for segmentation masks and is what every downstream tool expects. Worth one
sentence in Data Format saying so rather than leaving a reviewer to wonder.

---

## Manuscript skeleton, with what already exists

| section | content | state |
|---|---|---|
| Purpose | the LSTV numbering problem; why existing collections fail on it | to write |
| Acquisition | patient-level crosswalk of three public sources; placement rule | documented in README |
| Label scheme | VerSe-native, the classes, the lumbar rib decision | documented |
| Annotation & review | dual review, server-side gating, adjudication | partly documented |
| Validation | invariants 802/802; rib QC 2/5,749 evaluable; morphometric plausibility vs published | **done, this session** |
| Descriptive analysis | count-free transition measures; spinopelvic and corridor measures | **done — 14 panels** |
| Data format & access | NIfTI, splits, dataset_interface, DOI | needs DOI |
| Limitations | FOV-limited thoracic GT, supine angles, empty hardware class, multi-source LSTV | drafted in README |

Figures, at most four given ten pages:

1. Label scheme + a worked transitional case (the count problem, shown not argued)
2. Count-free distributions — the interval count and the bimodal rib ratio
3. Spinopelvic measures against published reference values (this is the validation figure)
4. The Castellvi span/gap scatter

---

## Order of work

1. Confirm licence compatibility across all three sources ← **blocks everything**
2. Regenerate and freeze the splits
3. Finish `0068`; decide whether the hardware class ships declared-but-empty or populated
4. Build the deposit bundle; verify a clean download opens and the loader runs
5. Reserve the Zenodo DOI
6. Write against the reserved DOI
7. Publish the Zenodo record; submit
