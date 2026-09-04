---
title: "Amendment to IRB-21-10-4123: Automated fracture detection and vertebral level identification on trauma spine radiographs"
subtitle: "Improving Outcome in Orthopaedics Trauma Care — PI: Rahul Vaidya, MD — Expedited amendment, PI-originated, study open to accrual"
author: "Prepared by Gregory Schwing, MD, PhD (Other Investigator) for Andreea Geamanu"
date: "September 2026"
---

This document follows the eProtocol section order. Each section gives the text to enter, or
states that no change is needed. The amendment adds personnel and extends the approved
retrospective chart review to the imaging already obtained during care. It does not change
consent, risk, or the participant population, so it stays expedited.

# Amendment Summary (lay terms)

> This amendment adds study personnel and extends the existing retrospective chart review
> to include imaging already obtained during care. (1) Spine radiographs and spine CT scans,
> with their radiology reports, will be retrieved from DMC Radiology/PACS for patients already
> identified from the trauma registry. (2) Images will be de-identified within DMC — names,
> medical record numbers, dates, and all DICOM header identifiers removed, and any text burned
> into the image checked and removed — before being stored and analyzed on Wayne State
> University's secure research computing cluster. (3) The de-identified images will be used
> to develop and evaluate computer methods (machine learning models) for detecting fractures
> and identifying the vertebral level on trauma radiographs, with results compared against the
> CT and the radiology report. (4) The approved number of records is increased from 2,000 to
> [N]. The master list linking study IDs to identifiers remains on DMC Citrix, separate from
> study data, as currently approved. No results are returned to the medical record or to
> treating clinicians. Nothing about consent, risk, or the participant population changes.

# Summary & Purpose — append to Purpose

> A further aim is to determine whether automated image-analysis methods can detect
> fractures and identify vertebral levels on trauma spine radiographs from this population,
> using the CT and radiology reports already obtained during care as the reference standard.

# Background, Rationale, Data Analysis and Procedures

**Section A — Background (add).**

> Commercial software already detects fractures on radiographs well; GLEAMER's BoneView is
> FDA-cleared for the appendicular skeleton, ribs, and thoracolumbar spine and is in use at
> over 300 hospitals.^1^ What it does not do is name the vertebra reliably. A detector that
> reports "fracture at L1" counts vertebrae on the film, and counting fails in the patients
> where it matters: lumbosacral transitional vertebrae occur in 4–30% of people,^2^ the
> accepted method is to count down from C2 on whole-spine imaging,^3^ which trauma patients
> do not get, and wrong-level spine surgery occurs about once per 3,100 procedures with
> transitional anatomy the most common cause.^4^ A confident wrong level is worse than none,
> and it is wrong most often in the patient whose anatomy is unusual.
>
> The investigators built and published CTSpinoPelvic1K, 802 CT scans with the spine,
> sacrum, hips, femora, and ribs labeled on one coordinate frame, with explicit classes for
> L6, a thirteenth thoracic vertebra, a separate S1, and lumbar ribs.^5^ It was built to test
> whether a vertebra can be identified from its own appearance without counting from C2.
> That is the supervision a radiograph model needs and cannot obtain from radiographs.
>
> Radiographs are the target modality and are required. CT is not used to train the model;
> it serves as the reference standard for two questions the radiograph cannot answer about
> itself — whether the level was correct (CT images, for encounters with both studies) and
> whether a fracture was missed on the initial film (CT report only).

**Section A(b) — Statistical methods (replace "N/A").**

> Descriptive statistics for the cohort. For the imaging analysis: sensitivity and
> specificity of automated fracture detection against the radiology report and CT;
> agreement of automated vertebral-level identification with CT, reported separately for
> patients with and without transitional lumbosacral anatomy; agreement of automated
> spinopelvic measurements with manual measurement against published error bands.^6,7^

**Section B — Procedures (add).**

> Retrospective. No interaction with participants. Models are first developed on public
> datasets; DMC images are used to refine and evaluate them. No output is placed in the
> medical record or shown to treating clinicians.

**Section C — Data Collection (revise).**

> Data will be collected from the DMC electronic medical record (Citrix) using the data
> collection sheet, and from DMC Radiology/PACS: thoracolumbar and lumbar spine radiographs
> (AP and lateral), spine CT, and the associated radiology reports, for patients already
> identified from the trauma registry. Images are de-identified within DMC before analysis.

# Participant Population

Number of records/charts: **2,000 → [N]**. *The same number must appear on the DMC
application and in the protocol document; the 2022 DMC review flagged a mismatch.* Age
range, inclusion, and exclusion unchanged.

# Procedures to Maintain Confidentiality — revise (c), (e), (f), (i), (j)

> (c) Chart data will be coded and stored on DMC Citrix. Imaging will be de-identified within
> DMC — all DICOM header identifiers removed and burned-in text checked and removed — and the
> de-identified images stored and analyzed on the Wayne State University High Performance
> Computing cluster, an institutionally managed secure research computing environment with
> access limited to named study personnel. No imaging is stored on personal devices,
> personal cloud accounts, or third-party services.
>
> (e) PI and research study team.
>
> (f) DMC electronic medical record (Citrix) and DMC Radiology/PACS.
>
> (i) The master list (key to identifiers) and study data will be stored separately on DMC
> Citrix. De-identified imaging on the WSU cluster carries study IDs only; the key never
> leaves DMC.
>
> (j) Data will be kept until all publishable articles/abstracts have been accepted and
> published. De-identified imaging on the WSU cluster will be deleted at study closure.

# HIPAA — no change

Identifiers used remain geographic subdivisions, elements of dates, medical record number,
and other unique identifying number; disclosed: none. Imaging that leaves DMC is
de-identified. *Confirm the "Will PHI be disclosed to sponsors" answer is completed; it was
flagged blank in 2022.*

# Personnel Information — Key Personnel Additions

Role for each: Other Investigator, Student/Resident/Fellow, unless noted.

| Name | Email | Note |
|---|---|---|
| Ashley Schehr | — | Student lead |
| Annika Tekumulla | — | |
| Margret Khoushi | — | |
| Ryan Christian | — | |
| Dane Hubers | — | |
| Faris Mahjoub | — | |
| Hassan Saad | — | |
| Mia Sooch | — | |
| Sathya Siddapureddy | — | |
| Michael McLellan | — | |
| Jerick Kim | — | |
| Emi Ueda-Martinez | hv3612@wayne.edu | |
| Nishaan Makim | naahsin@gmail.com | **Not WSU-affiliated.** Every personnel entry in this protocol carries a WSU AccessID, and Obligations and COI are signed in eProtocol. He will need a sponsored/guest WSU AccessID before he can be listed; CITI must then be completed under that WSU affiliation so the protocol can pull his record. |
| Miraziz Ismoilov, MD; Nizar Alnabahneh, MD | — | Radiology. **Add only if they will adjudicate DMC images**; omit if their role stays on public data. |

Each must complete before submission: CITI *Biomedical Investigators*, *HIPS for Students
and Instructors*, *Biomedical Responsible Conduct of Research*; sign the **Obligations** page;
complete **COI**. One incomplete person holds the amendment; list whoever is done by the
submission date and add the rest on the next one.

# Study Location / DMC Application

DMC unchanged. **DMC Application, Section 5:** re-select "DMC Radiology/Imaging Services"
with a statement that images will be retrieved from PACS, not only records read. It was
deselected in 2022 at the CRO reviewer's request.

# Attachments — add, highlighted

- Revised data collection sheet ("Orthopaedic Trauma Patient Variables") with imaging fields:
  study ID, radiograph accession and views present, CT accession, report findings.
- Revised protocol document with the imaging methodology and the named storage environment.
- Revised DMC Application (Section 5).

# Open before submission

- **[N]** from a trauma-log query: encounters with a thoracolumbar or lumbar spine
  radiograph and a spine CT report in the same encounter, and the log's date range.
- Radiology residents on or off the amendment.
- A labeling tool on WSU compute (ITK-SNAP or 3D Slicer on a WSU machine, or self-hosted
  CVAT on the cluster). The Colab server used for the CT work is commercial cloud and is
  excluded by the storage text above.
- A sponsored WSU AccessID for Nishaan Makim.

# References

1. GLEAMER BoneView: FDA 510(k) clearance, March 2022; figures as reported by the
   manufacturer. https://www.gleamer.ai
2. Konin GP, Walz DM. Lumbosacral transitional vertebrae: classification, imaging findings,
   and clinical relevance. *AJNR Am J Neuroradiol.* 2010;31(10):1778–1786.
   doi:10.3174/ajnr.A2036
3. Lian J, Levine N, Cho W. A review of lumbosacral transitional vertebrae and associated
   vertebral numeration. *Eur Spine J.* 2018;27(5):995–1004. doi:10.1007/s00586-018-5554-8
4. Epstein NE. A perspective on wrong level, wrong side, and wrong site spine surgery.
   *Surg Neurol Int.* 2021;12:286. doi:10.25259/SNI_402_2021
5. Schwing G, et al. CTSpinoPelvic1K: spine, pelvis, ribs and femora in one coordinate frame,
   annotated for lumbosacral transitional anatomy. Zenodo. doi:10.5281/zenodo.22139642
   (v7 at doi:10.5281/zenodo.22242745). Dataset descriptor in preparation.
6. Abedeen I, Rahman MA, Prottyasha FZ, et al. FracAtlas: a dataset for fracture
   classification, localization and segmentation of musculoskeletal radiographs.
   *Sci Data.* 2023;10:521. doi:10.1038/s41597-023-02432-4
7. Glaser D, AlMekkawi AK, Caruso JP, et al. Deep learning for automated spinopelvic
   parameter measurement from radiographs: a meta-analysis. *Artif Intell Surg.*
   2025;5:1–15. doi:10.20517/ais.2024.36
