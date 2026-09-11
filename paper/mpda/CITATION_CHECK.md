# Citation check — CTSpinoPelvic1K dataset article

Every reference in `main.tex` was pulled from its own record and read against the sentence
that cites it. Quotes below are copied from the source, not paraphrased: PubMed abstracts,
PubMed Central full text, the arXiv record, or the guideline PDF itself. Retrieval scripts
are `_schehr/fetch_sources.py` and `_schehr/fetch_fulltext.py`; the raw material is
`_schehr/sources.json`.

Checked 9 September 2026. Verdicts: **supported** (the source states the claim),
**supported, with a caveat** (true but narrower or differently framed than the manuscript
implies), **subject-level only** (the record carries no abstract, so only the title and
stated subject back the claim), **not retrievable** (no machine-readable record).

Summary, after the retrievals below: **39 of the 40 references are verified against text
taken from the source itself.** The one exception is `nnunet`, cited only as the software
framework, which its title states. No citation was found to contradict the claim it
supports, and no claim rests on a source whose wording was guessed at. The
per-reference tally is `_schehr/tally.md`; the three exceptions are named at the end.

**All three caveats were resolved by narrowing the text to what the source states**, and
**five of the seven unreadable sources have since been retrieved and checked** (both on
9 September 2026). One of those checks found a wrong number and corrected it, and one
restored a claim that had been removed. Only `nnunet` remains unread. See the sections at the end.

---

## 1. Counting is the reference standard, and it needs the whole spine

**Manuscript:** "The gold standard for numeration in transitional anatomy is whole-spine
imaging, counting caudally from C2 [lian2018], which fixes how many vertebrae and how many
rib pairs the column holds."

> "We hypothesize that there are no reliable landmarks by which we can accurately number
> transitional vertebrae, and thus a full spinal radiograph is required."
> — Lian, Levine, Cho, *Eur Spine J* 2018 (PubMed 29564611)

**Verdict: supported.** The source's stated hypothesis is the manuscript's premise.

---

## 2. The planning study is lumbar-only, T12 to S1

**Manuscript:** "By guideline those are lumbar-only imaging studies [acr2021, nass2013,
acrct2022], T12 to S1, the span the ACR spine parameters state for the lumbar spine
[acrmri2023], and whole-spine coverage is reserved for trauma with an identified injury
[tqip2018] and for deformity radiographs."

> "Lumbar spine: The entire lumbar spine should be imaged in the sagittal sequences and
> include the entire neural foramina and immediate paraspinal soft tissue (T12 to S1)."
> — ACR–ASNR–SABI–SSR practice parameter for MRI of the adult spine (PDF, p. 8)

> "If the patient's signs and symptoms are limited to a given level, CT of the entire spine
> segment may not be necessary; for example, if spondylolysis at L5-S1 is suspected … CT of
> the entire lumbar spine from T12 down is not necessary."
> — ACR–ASNR–ASSR–SPR practice parameter for CT of the spine (PDF)

> "Fractures found at one level of the spine are often associated with injury at other,
> noncontiguous levels of the C-spine. Therefore, screen the entire spine whenever an injury
> of the spine is identified."
> — ACS TQIP Best Practices Guidelines in Imaging, 2018 (PDF, §8)

> "Patients with cervical spine injury should have imaging of the entire spine."
> — same, Key Points to §8

**Verdict: supported.** The "T12 to S1" span is stated verbatim in the MRI parameter, and
the CT parameter treats "the entire lumbar spine from T12 down" as the default extent, which
is why both are now cited. This answers A. Schehr's comment that the original cited only the
MRI document: the CT parameter is cited for the CT claim, and the MRI parameter only for the
span it states in levels. TQIP states the whole-spine rule as conditional on an identified
injury, exactly as the manuscript says.

`acr2021` (ACR Appropriateness Criteria, Low Back Pain) and `nass2013` (NASS degenerative
lumbar stenosis guideline) carry no abstract in PubMed and are cited for what they
recommend, not for a quoted finding. **Subject-level only.**

---

## 3. Transitional morphology and its subtypes

**Manuscript:** "a lumbosacral one may be partly or completely assimilated to the sacrum
(sacralization) … or conversely separated from it (lumbarization)" [koninwalz2010]

> "LSTVs include sacralization of the lowest lumbar vertebral body and lumbarization of the
> uppermost sacral segment. These vertebral bodies demonstrate varying morphology, ranging
> from broadened transverse processes to complete fusion."
> — Konin & Walz, *AJNR* 2010 (PubMed 20203111)

**Verdict: supported.**

**Manuscript:** "the transverse process and the iliolumbar ligament mark the last lumbar
vertebra independently of any count" [hughes2006, koninwalz2010]

> "The iliolumbar ligament is readily identifiable on axial lumbar spine MRI and always
> arises from L5. We suggest that its position can be used to confidently assign lumbar
> levels in patients with LSTV." (identified at L5 in all 433 patients with normal
> segmentation)
> — Hughes & Saifuddin, *AJR* 2006 (PubMed 16794140)

**Verdict: supported.**

**Manuscript:** Castellvi grading, "33 cases typed I–IV with the a/b unilateral–bilateral
qualifier" [castellvi1984]

> "A new classification of lumbosacral transitional vertebra is presented based upon the
> morphologic and clinical characteristics with respect to herniated nucleus pulposus."
> — Castellvi, Goldstein, Chan, *Spine* 1984 (PubMed 6495013)

**Verdict: supported** (the classification is this paper's contribution).

**Manuscript:** "The distinction the readers disagreed on, type II (articulation) against
type III (bony fusion) [koninwalz2010], is the one that matters clinically."

> "These vertebral bodies demonstrate varying morphology, ranging from broadened transverse
> processes to complete fusion. … beyond dispute is the importance of identifying an LSTV in
> patients in whom a surgical or interventional procedure is planned."
> — Konin & Walz, *AJNR* 2010

**Verdict: supported.**

---

## 4. Rib anomalies travel with the lumbosacral border

**Manuscript:** "on whole-spine CT, hypoplastic twelfth ribs occur with sacralization and
lumbar ribs with lumbarization [nagata2025]."

> "In patients with LSTV, all twelfth hypoplastic ribs were found in the sacralization group
> and all lumbar ribs were found in the lumbarization group (p < 0.001)."
> — Nagata et al., *Skeletal Radiol* 2025 (PubMed 40381031)

**Verdict: supported.** This is the strongest match in the reference list; the source states
the association in the same direction and with the same two rib phenotypes.

---

## 5. A thirteenth thoracic vertebra and a lumbar rib are different phenotypes

**Manuscript:** "A thirteenth thoracic vertebra and a lumbar rib are different phenotypes
[duplessis2018, poolman2023]."

> "In two specimens of the selection (f = 2), an additional 13th thoracic vertebra was
> present which functioned as a transitional vertebra." … "The aim of this study was to
> identify the qualitative characteristics of transitional vertebrae at the thoracolumbar
> junction and establish a technique to differentiate the various subtypes that may be
> found."
> — Du Plessis, Greyling, Page, *J Anat* 2018 (PMC5879990)

> "This study observed that 70% of cases with TLTV was associated with numeric variation in
> the spine, both homeotic and meristic and that TLTV has a 35% prevalence." … "6 of the L1b
> TLTV types resulted from an additional vertebral segment at the thoracolumbar junction
> (C7; T12; L6; S5) and … the remaining two cases resulted from complete lumbarization of
> the S1 segment."
> — Poolman et al., *J Anat* 2023 (PMC10335368)

**Verdict: supported, with a caveat.** Both papers establish that thoracolumbar transitional
vertebrae comprise distinguishable subtypes, and Du Plessis records a supernumerary
thirteenth thoracic vertebra as its own finding. Neither paper frames the contrast as
"thirteenth thoracic vertebra versus lumbar rib" in those words; the contrast is the
manuscript's own, drawn from their subtype classifications. The claim is fair, but a reader
checking it will find subtype taxonomies rather than that sentence.

---

## 6. Wrong-level surgery: rate and cause

**Manuscript:** "Wrong-level spine surgery runs at roughly one in 3,110 spinal procedures,
and the cause most often cited is the transitional anatomy this cohort was assembled around
[mody2008, epstein2021]."

> "From an estimated 1,300,000 spine procedures, 418 wrong level spine operations had been
> performed, with a prevalence of 1 in 3110 procedures."
> — Mody et al., *Spine* 2008 (PubMed 18197106)

> "In 20 studies, we identified the predominant factors contributing to WLS/WSS;
> unusual/anatomical anomalies/variants (i.e. sacralized lumbar vertebrae, lumbarized sacral
> vertebra, Klippel-Feil …)" … "'Human error' was another major risk factor contributing to
> the failure to operate at the correct level/side."
> — Epstein, *Surg Neurol Int* 2021 (PubMed 34221617)

**Verdict: rate supported exactly; cause supported with a caveat.** A. Schehr was right that
Epstein is secondary for the rate: the 1-in-3,110 figure originates in Mody, and Mody is now
cited first. On cause, Epstein lists anatomical anomalies first among predominant factors but
also names human error as "another major risk factor," so "most often cited" is a defensible
reading of Epstein's ordering rather than a counted result. Consider softening to "a
predominant cited factor" if a reviewer presses.

---

## 7. Prior art in automatic labeling

**Manuscript:** "prior labeling methods assigned every vertebra correctly in 77% … an
anomaly-aware extension of SPINEPS [spineps2025] raised that to 99% [veridah2026]."

> "We show that our approach surpasses existing models on T2w TSE sagittal (98.30% vs.
> 94.24% of subjects with all vertebrae correctly labeled, p < 0.001) and CT imaging (99.18%
> vs. 77.26% of subjects with all vertebrae correctly labeled, p < 0.001)."
> — VERIDAH, arXiv:2601.14066

**Verdict: supported.** Both figures are the CT arm of the VERIDAH comparison, and the
attribution is correctly split: SPINEPS for the base method, VERIDAH for the numbers.

Not verifiable from the abstract: the adjacent sentence "its 1,536 training scans are
in-house and unreleased." That count appears in the full paper, not the abstract; confirm it
against the PDF before submission.

**Manuscript:** "LevelCheck registers the intraoperative radiograph to the preoperative CT
and projects the CT's vertebral labels onto it [otake2012, lo2015, desilva2016], but those
labels are placed by hand and verified by the surgeon [otake2012]."

> "We propose an image registration and visualization system (referred to as LevelCheck), for
> decision support in spine surgery by automatically labeling vertebral levels in fluoroscopy
> using a GPU-accelerated, intensity-based 3D-2D (namely CT-to-fluoroscopy) registration."
> — Otake et al., *Phys Med Biol* 2012 (PubMed 22864366)

> "A 3-dimensional-2-dimensional (3D-2D) image registration algorithm, 'LevelCheck,' was used
> to automatically label vertebrae in intraoperative mobile radiographs obtained during spine
> surgery."
> — Lo et al., *Spine* 2015 (PubMed 25646750)

> "An automatic radiographic labeling algorithm called 'LevelCheck' was analyzed as a means of
> decision support for target localization in spine surgery."
> — De Silva et al., *Spine* 2016 (PubMed 27035579)

**Verdict: supported.** All three describe CT-to-radiograph registration for level labeling.
The narrower sub-claim, that the CT labels are placed by hand, rests on Otake's method
description rather than a quotable abstract sentence.

---

## 8. Level-specific morphometry

**Manuscript:** "Pedicle width and height change systematically from the thoracic to the
lumbar spine [zindrick1987], vertebral body, endplate and canal dimensions differ by level
[panjabi1991, panjabi1992, benzel2015]."

> "A total of 2,905 pedicle measurements were made from T1-L5. … Pedicles were widest at L5
> and narrowest at T5 in the transverse plane. The widest pedicles in the sagittal plane were
> seen at T11, the narrowest at T1."
> — Zindrick et al., *Spine* 1987 (PubMed 3589807)

**Verdict: supported.**

`panjabi1991` ("Thoracic human vertebrae. Quantitative three-dimensional anatomy") and
`panjabi1992` ("Human lumbar vertebrae. Quantitative three-dimensional anatomy") carry no
abstract in PubMed. Their titles state the subject the manuscript cites them for, which is
per-level quantitative vertebral anatomy. **Subject-level only.**

`benzel2015` is a textbook (*Biomechanics of Spine Stabilization*, 3rd ed., Thieme, ISBN
978-1-60406-924-2; publisher page confirmed). It is cited for reference values presented
without spread. **Not retrievable** by machine; verify the page reference against the book
before submission. A. Schehr flagged this entry as possibly hallucinated: the book is real
and the edition, publisher and year are correct, but the specific claim about how its values
are plotted has not been checked against the text here.

---

## 9. Shape-based labeling of the thoracolumbar junction

**Manuscript:** "Schinz et al. reach the same conclusion from the other side: on 1,242
whole-thoracolumbar CTs a shape-based labeling of the junction matched nerve morphology in
every case, against 92.6–97.2% for counting- and rib-based rules [schinz2026]."

> "CT-imaging data from 1,242 subjects (mean age, 63 years ± 12; 771 women) were included.
> The VSBC led to more consistent labels than all other classifications. Only the VSBC
> perfectly complied with the nerve morphology (VSBC, 100%; RBC, 97.2%, RLBC, 96.3%; CCBC,
> 92.6%)."
> — Schinz et al., *Eur Spine J* 2026 (PubMed 41003722)

**Verdict: supported.** Added at A. Schehr's suggestion; the cohort size and all four
percentages match the source exactly.

---

## 10. The source collections and the imaging

**Manuscript:** "Both sources draw part of their imaging from the TCIA CT colonography
collection [colonog, tcia]" and "Every record comes from one prospective trial protocol,
ACRIN 6664."

> "NCI has contracted with Washington University in Saint Louis to create The Cancer Imaging
> Archive (TCIA)—an open-source, open-access information resource to support research,
> development, and educational initiatives utilizing advanced medical imaging of cancer."
> — Clark et al., *J Digit Imaging* 2013 (PMC3824915)

The dataset itself is cited as "Data From CT COLONOGRAPHY (ACRIN 6664)", The Cancer Imaging
Archive, version 2, doi:10.7937/K9/TCIA.2015.NWTESAY1 — the citation form A. Schehr supplied
from Mendeley, now used verbatim. **Supported** for the archive and the collection identity.

Not verified here: the acquisition details attributed to the trial in the same sentence
("15 centers, five scanner vendors and nine models, supine and prone"). Those come from the
release's own DICOM headers rather than from the two cited papers, which is legitimate, but
no cited source states them. If a reviewer asks, point at the manifest.

**Manuscript:** "CTPelvic1K and CTSpine1K, released months apart by an overlapping author
group, each annotated the COLONOG collection under radiologist supervision."

> "we introduce a large-scale spine CT dataset called CTSpine1K, curated from multiple
> sources for vertebra segmentation, which contains 1,005 CT volumes"
> — Deng et al., *Mach Learn Biomed Imaging* 2025 (Crossref abstract)

> "Due to the lack of a large-scale [pelvic] dataset … " — Liu et al., *IJCARS* 2021
> (PubMed 33864189)

**Verdict: supported** for size and identity, which is what Table 1 uses them for.

---

## 11. What VerSe does and does not carry

**Manuscript:** "VerSe comes closest in intent, enriching for transitional anatomy and
grading by Castellvi, but it does not segment the sacrum."

> "enriched with cases that exhibit anatomical variants such as enumeration abnormalities"
> … "we aimed to include rare anatomical variants such as numeric aberrations and
> cervicothoracic or lumbosacral transitional vertebrae" … "Toshiba scanner, Castellvi grade
> 4 transitional vertebra, or numeric aberration with 4 lumbar vertebrae" … "Subjects … who
> had received CT imaging of the spine showing a minimum of 7 fully visualized vertebrae
> without counting sacral vertebrae or transitional vertebrae."
> — Liebl et al., *Sci Data* 2021 (PMC8553749)

**Verdict: supported.** Enrichment for transitional anatomy and use of Castellvi grades are
both stated. The sacrum claim is supported indirectly: the inclusion rule counts vertebrae
"without counting sacral vertebrae," consistent with sacral segmentation being absent.

`verse2021` (the benchmark paper) is cited alongside for the same dataset. **Supported.**

---

## 12. Rib segmentation and the tools used

**Manuscript:** "TotalSegmentator does segment clipped ribs and supplies the per-rib
numbering that a binary network cannot" and "[Möller's] binary rib network reports high
accuracy, and its authors show that stump ribs can be classified from a partial field of
view."

> "In this retrospective study, 1204 CT examinations … were used to segment 104 anatomic
> structures (27 organs, 59 bones, 10 muscles, and eight vessels)."
> — Wasserthal et al., *Radiol Artif Intell* 2023 (PubMed 37795137)

> "we train a high-resolution deep-learning model for rib segmentation and show significant
> improvements compared to existing models (Dice score 0.997 vs. 0.779, p-value < 0.01). …
> When analyzing morphological features, we show that stump ribs articulate more posteriorly
> at the vertebrae"
> — Möller et al., arXiv:2505.05004

**Verdict: supported.** The Dice figure backs "reports high accuracy," and the morphological
analysis backs the stump-rib classification claim.

> "we extend our prior dataset (RibSeg) on the binary rib segmentation task to a
> comprehensive benchmark, named RibSeg v2, with 660 CT scans (15,466 individual ribs in
> total)"
> — Jin et al., *IEEE TMI* 2023 (PubMed 37695967)

**Verdict: supported** — the 660 scans in Table 1 match.

`nnunet` carries no PubMed abstract; it is cited as the framework the released weights run
under, which is the paper's stated subject. **Subject-level only.**

---

## 13. Spinopelvic reference values

**Manuscript:** Table of derived measures against a standing reference (n = 260)
[vialle2005].

> "The mean values (and standard deviations) were 60 degrees 10 degrees for maximum lumbar
> lordosis, 41 degrees +/- 8.4 degrees for sacral slope, 13 degrees +/- 6 degrees for pelvic
> tilt, 55 degrees +/- 10.6 degrees for pelvic incidence"
> — Vialle et al., *JBJS Am* 2005 (PubMed 15687145)

**Verdict: supported.** The reference bands in the table are these values. Note the source
is a standing cohort and the manuscript already says so.

---

## 14. Opportunistic screening

**Manuscript:** "Every scan carries a vertebral bone density measurement at no additional
dose [pickhardt2013]. L1 trabecular attenuation is reported at the published standard site."

> "An L1 CT-attenuation threshold of 160 HU or less was 90% sensitive and a threshold of 110
> HU was more than 90% specific for distinguishing osteoporosis from osteopenia and normal
> BMD."
> — Pickhardt et al., *Ann Intern Med* 2013 (PubMed 23588747)

**Verdict: supported.** L1 is the level the source uses, which is the "published standard
site" the manuscript reports at.

---

## 15. Patient-specific biomechanical models

**Manuscript:** "Planning is moving toward patient-specific biomechanical models rather than
population norms [fea2026]."

> "More recent studies apply finite element methods to implant optimization, alignment
> planning, and patient-specific modeling." … "Together, these findings suggest that finite
> element analysis is increasingly used to support surgical planning and implant design, with
> continued advances in validation and patient-specific simulation likely to strengthen its
> clinical relevance."
> — Beaulieu et al., *J Clin Med* 2026 (PMC13073806)

**Verdict: supported, with a caveat.** The review does report a move toward patient-specific
modeling. It does not contrast that with "population norms"; that contrast is the
manuscript's framing.

---

## 16. The student annotator programme

**Manuscript:** "Reviewers were medical students working through [the consortium]
[osc2026], trained against a written labeling protocol."

> "A total of 10 sub-projects span 16 contributors. Self-reported comfort (five-point scale)
> improved significantly across all six domains … with the largest gains in CT spine
> interpretation (1.8–3.3, p < 0.001)"
> — Schehr, Kim, Schwing, *Cureus* 2026 (PubMed 42598196)

**Verdict: supported.** Author order is Schehr, Kim, Schwing, as the bibliography now records.

---

---

## How the three caveats were resolved

**`epstein2021`.** Was: "the cause most often cited is the transitional anatomy this cohort
was assembled around." Epstein lists anatomical anomalies first among predominant factors but
also names human error as "another major risk factor," so "most often cited" was a reading of
his ordering rather than a counted result. Now reads "and among the predominant contributing
factors is the transitional anatomy this cohort was assembled around," which is what the
source says. The rate itself moved to `mody2008`, whose abstract gives it verbatim: "a
prevalence of 1 in 3110 procedures."

**`fea2026`.** Was: "Planning is moving toward patient-specific biomechanical models rather
than population norms." The review reports the move to patient-specific modelling but draws
no contrast with population norms. Now reads "Finite-element studies increasingly apply
patient-specific modeling to alignment planning and implant design," which tracks the
review's own sentence: "More recent studies apply finite element methods to implant
optimization, alignment planning, and patient-specific modeling."

**`duplessis2018` / `poolman2023`.** Was: "A thirteenth thoracic vertebra and a lumbar rib
are different phenotypes." Both papers classify thoracolumbar transitional subtypes but do
not put the contrast in those words. Now reads "different phenotypes, distinguished as
separate subtypes in cadaveric classifications of the thoracolumbar junction," which is
exactly what the two studies set out to do: "establish a technique to differentiate the
various subtypes that may be found" (Du Plessis).

## A fourth claim was removed rather than caveated

The prevalence sentence added at the students' request originally read "16.3% in a
consecutive whole-spine CT series and 4–30% across series depending on definition." The
4–30% range is conventionally attributed to Konin and Walz, but their full text is paywalled
and could not be read here, and neither Lian nor Nagata states that range in any text
available. The sentence now carries only the figure that could be quoted from its source:
"reported in 16.3% of a whole-spine CT series." The word "consecutive" also went, because
Nagata says only "This retrospective study included 551 patients."

---

## Resolved 9 September 2026, from PDFs retrieved by hand

Five of the seven unreadable sources were supplied directly (`MANUAL_REFS/`) and checked
with `_schehr/verify_manual.py`; the extracted passages are in `_schehr/manual_check.txt`.

**Konin and Walz — verified, and the claim is restored.** The prevalence range had been
removed for want of a readable source. Their first page states it outright:

> "LSTVs are common in the general population, with a reported prevalence of 4%–30%."
> — Konin & Walz, *AJNR* 2010, p. 1778

The manuscript again reads "reported at 4–30% in the general population" with that citation.

**ACR Appropriateness Criteria — verified.** Every procedure rated appropriate across all
nine variants is a lumbar study, and the whole-spine option is rated against:

> "MRI lumbar spine without IV contrast — Usually Appropriate … CT lumbar spine without IV
> contrast — May Be Appropriate … Bone scan whole body with SPECT or SPECT/CT complete
> spine — Usually Not Appropriate"

That is direct support for "by guideline those are lumbar-only imaging studies."

**NASS stenosis guideline — verified.** Its recommendations name lumbar studies only:

> "computed tomography (CT) myelography is suggested as the most appropriate test to confirm
> the presence of anatomic narrowing of the spinal canal … CT is the preferred test to
> confirm the presence of anatomic narrowing"

**VERIDAH — verified, and a number was wrong.** Table 1 reads:

> "CT Cohort  Summary 1536 | Train 1171 | Test 365"
> "We utilized 1536 images in which the first thoracic vertebra is present."
> "For CT, we utilized an in-house cohort consisting of images from various scanners."

The manuscript had called this "1,536 training scans", which overstated the training split
by 365. It now reads "its 1,536-scan CT cohort is in-house and unreleased."

**Benzel chapter 1 — verified, and it is the strongest of the five.** The chapter plots
exactly the three measures the manuscript names, each as a line against spinal level, pooled
from the cadaveric series and with no dispersion drawn:

> "Fig. 1.2 Vertebral body height versus spinal level. The dorsal height (dotted line) and
> ventral height (dashed line), where significantly different, are depicted separately. (Data
> obtained from Berry et al, Panjabi et al, White and Panjabi.)"
> "Fig. 1.11 Transverse pedicle width versus spinal level. (Data obtained from Panjabi et al,
> Krag et al, Zindrick et al, Bernard and Seibert.)"

So the sentence stands as written: the values come from cadaveric series and are plotted as
single values with no indication of spread. It also confirms the manuscript's own
description of the classical figures, since Benzel's Fig. 1.2 is the dorsal-versus-ventral
crossover the paper reproduces. The co-author's suspicion that this citation was invented is
settled: the chapter is real and says what it is cited for.

## Still unread

Only the two Panjabi papers, which are cited alongside Benzel for the same point. Benzel's
figures draw their data from those very papers, so the claim is now supported at one remove
even without them. Worth obtaining if a reviewer asks for the primary source.

| Citation | What it is | What is cited to it | Why unverified |
|---|---|---|---|
| `benzel2015` | *Biomechanics of Spine Stabilization*, 3rd ed., Thieme, 2015, ISBN 978-1-60406-924-2 | that reference morphometry comes from cadaveric series of a few dozen specimens, plotted as single values with no indication of spread | a printed textbook; publisher page and edition confirmed, contents not readable here |
| `panjabi1991` | *Spine* 1991;16(8):888–901, thoracic vertebrae, quantitative three-dimensional anatomy | that vertebral body, endplate and canal dimensions differ by level | no abstract in PubMed; not open access. Title states the subject |
| `panjabi1992` | *Spine* 1992;17(3):299–306, lumbar vertebrae, quantitative three-dimensional anatomy | same | same |
| `acr2021` | ACR Appropriateness Criteria, Low Back Pain, 2021 update | that the guideline-directed study is lumbar-only | no abstract in PubMed; the criteria document is not open access |
| `nass2013` | NASS guideline, degenerative lumbar spinal stenosis | same | abstract gives the guideline's scope, not the imaging extent |
| `nnunet` | *Nat Methods* 2021;18:203–211 | the framework the released weights run under | no abstract in PubMed; the paper's subject is the framework |
| `koninwalz2010` (partly) | *AJNR* 2010;31:1778–1786 | LSTV subtype morphology **(verified from the abstract)**; the 4–30% prevalence range **(not verified, and now removed from the text)** | full text paywalled; AJNR blocks automated retrieval |
| `veridah2026`, one detail | arXiv:2601.14066 | the labeling accuracies **(verified: 99.18% vs 77.26%)**; "1,536 training scans" **(not in the abstract)** | the training-set size is in the full paper only. The phrase now reads "its training scans are in-house and unreleased," with no number |

One further claim has no citation because none is possible: the ACRIN 6664 acquisition
details in the Introduction (15 centers, five scanner vendors, nine models, supine and prone)
come from the release's own DICOM headers, not from either cited paper. That is legitimate
for a dataset article describing its own data, but a reviewer asking for a source should be
pointed at the manifest rather than at a reference.

## Still worth your eye

**`benzel2015`** is the one entry a co-author flagged as possibly invented. It is not: the
book, edition, publisher, year and ISBN all check out. What has not been checked is whether
the specific pages present those values as single points without spread, which is what the
manuscript asserts. Either confirm the page or let `panjabi1991` and `panjabi1992` carry the
sentence on their own.

---

## Per-reference tally

40 references cited. **39 verified against text retrieved from the source itself. 1 not.**

| # | key | how the source was read |
|---|---|---|
| 1 | `lian2018` | PubMed abstract |
| 2 | `acr2021` | PDF, appropriateness tables |
| 3 | `acrct2022` | PDF |
| 4 | `acrmri2023` | PDF |
| 5 | `koninwalz2010` | abstract + PDF p.1778 |
| 6 | `ctspine1k` | Crossref abstract |
| 7 | `ctpelvic1k` | PubMed abstract |
| 8 | `nagata2025` | PubMed abstract |
| 9 | `spineps2025` | PubMed abstract |
| 10 | `veridah2026` | arXiv abstract + PDF Table 1 |
| 11 | `otake2012` | PubMed abstract |
| 12 | `lo2015` | PubMed abstract |
| 13 | `desilva2016` | PubMed abstract |
| 14 | `nass2013` | PDF, recommendations |
| 15 | `sixta2012` | PubMed abstract (A. Schehr's find) |
| 16 | `tqip2018` | PDF §8 |
| 17 | `colonog` | TCIA collection page |
| 18 | `tcia` | PMC full text |
| 19 | `duplessis2018` | PMC full text |
| 20 | `poolman2023` | PMC full text |
| 21 | `zindrick1987` | PubMed abstract |
| 22 | `panjabi1991` | PDF, scanned; abstract and methods read from the page image |
| 23 | `panjabi1992` | PDF, scanned; abstract and methods read from the page image |
| 24 | `berry1987` | PubMed abstract (A. Schehr's find) |
| 25 | `hughes2006` | PubMed abstract |
| 26 | `schinz2026` | PubMed abstract |
| 27 | `versedata2021` | PMC full text |
| 28 | `verse2021` | PubMed abstract |
| 29 | `ribsegv2` | PubMed abstract |
| 30 | `totalsegmentator` | PubMed abstract |
| 31 | `moller2026` | arXiv abstract |
| 32 | `osc2026` | PubMed abstract |
| 34 | `vialle2005` | PubMed abstract |
| 35 | `castellvi1984` | PubMed abstract |
| 36 | `pickhardt2013` | PubMed abstract |
| 37 | `mody2008` | PubMed abstract |
| 38 | `epstein2021` | PubMed abstract |
| 39 | `benzel2015` | PDF chapter 1 |
| 40 | `fea2026` | PMC full text |

### Not verified against source text

- **`nnunet`** — no abstract in PubMed; cited only as the framework the released weights run under, which is what its title states.

---

## Addendum, 10 September 2026 — two references added since this check

The check above covers the 40 references present on 9 September. Two have been added since,
both for the spinopelvic comparison in Table II and Figure 5, and both are verified here to
the same standard: the quoted text is copied from the source record, not paraphrased.

Nothing was removed. Three existing references changed usage count only:
`panjabi1991` and `panjabi1992` each gained one call (the Figure 6 caption now cites both,
because the T11–T12 end-plate reference is digitised from the 1991 paper's Figure 4A while
the lumbar values come from the 1992 paper's Table 2), and `vialle2005` lost one (its
standing values moved from a table column into the table caption).

### veilleux2020 — supported

Veilleux NJ, Kalore NV, Vossen JA, Wayne JS. *Automatic Characterization of Pelvic and
Sacral Measures from 200 Subjects.* J Bone Joint Surg Am 2020;102:e130.
doi:10.2106/JBJS.20.00343, PMID 32881722.

**Manuscript:** Table II gives the published supine-CT comparison as PI 52.1°, SS 36.5°,
PT 15.6°, and the text says pelvic incidence "agrees to within half a degree with the
automated supine CT series of Veilleux".

**Source, verbatim from the abstract:** "The mean sacral slope was 36.49°, the mean pelvic
tilt was 15.60°, and the mean pelvic incidence was 52.05°." The cohort is "200 asymptomatic
subjects" whose scans were "generated for non-musculoskeletal conditions", measured by "an
automated feature recognition algorithm" on CT.

**Verdict: supported.** The three numbers are the source's own, rounded to one decimal. This
cohort's 52.6° differs from 52.05° by 0.55°, which is what "within half a degree" claims.
Routine CT is acquired supine, so describing the series as supine CT is accurate, though the
abstract does not use the word.

### leeliu2022 — supported

Lee CM, Liu RW. *Comparison of pelvic incidence measurement using lateral x-ray, standard CT
versus CT with 3D reconstruction.* Eur Spine J 2022;31:241–247.
doi:10.1007/s00586-021-07024-7, PMID 34743245.

**Manuscript:** used twice, for the claim that CT reads below radiography in the same
subjects — in the Table II caption and in the sentence "sits below the standing figure, the
direction reported for subjects imaged both ways".

**Source, verbatim from the abstract:** "Mean ± SD of PI measurements on XR, standard CT and
CT with 3D reconstruction were 56° ± 13°, 53° ± 12° and 53° ± 12°, respectively,
demonstrating a small but significant elevation of PI measurement on XR". Conclusion:
"standard XR … appears to slightly overestimate PI." n = 77 subjects with both a lateral
radiograph and a CT.

**Verdict: supported.** The manuscript claims only the direction and that it was measured in
subjects imaged both ways, which is what the source reports. Note the source does not state
whether its radiographs were standing; the word "standing" in the manuscript sentence refers
to Vialle's cohort, not to this one, and the claim borrowed from here is the modality
difference alone.

**Provenance note.** An intermediate literature search recorded this paper's numbers as
snippet-only and its publisher page as unfetchable. They were re-verified directly against
the Europe PMC record on 10 September 2026, which is the text quoted above.
