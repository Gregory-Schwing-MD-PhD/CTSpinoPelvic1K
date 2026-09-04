---
title: "Amendment to IRB-21-10-4123 — eProtocol text and checklist"
subtitle: "Improving Outcome in Orthopaedics Trauma Care (PI: Rahul Vaidya, MD)"
author: "Prepared by Gregory Schwing, MD, PhD, for Andreea Geamanu"
date: "September 2026"
---

# How this protocol has amended before, and what that means for this one

The event history shows 46 approved amendments since June 2022, roughly one a month, almost
all turning around in one to two weeks. The pending Amendment 47 (created 9/2/2026) is
already set up the way every prior one appears to have been: **Expedited**, originating from
the **Principal Investigator**, study **open to accrual**, with **Key Personnel Addition**
checked. That is the shape to keep. Nothing below changes the study design, adds risk, or
touches consent, so it stays expedited.

The 2022 DMC Clinical Research Office review is the other thing to draft against. It came
back with revisions on exactly the points this amendment touches: the record count had to
match across every document, the confidentiality answers had to name the secure server
(DMC Citrix) rather than "devices," the HIPAA disclosure question had to be answered, and
the DMC Application's service-area list had to match what the team would actually use.
Everything below is written so it passes that review the first time.

# Amendment form — what to select

| Field | Selection |
|---|---|
| Review type | Expedited |
| Originates from | Principal Investigator |
| Accrual status | Currently open to accrual |
| Key Personnel Modifications | Key Personnel Addition (see list below) |
| Change in PI | No |

# Amendment summary (lay terms) — paste into "Protocol Form Modifications / Amendment Summary"

> This amendment adds study personnel and extends the existing retrospective chart review
> to include the imaging already obtained during care. Specifically: (1) spine radiographs
> and spine CT scans, with their radiology reports, will be retrieved from DMC Radiology/PACS
> for patients already identified from the trauma registry; (2) those images will be
> de-identified within DMC — names, medical record numbers, dates, and all DICOM header
> identifiers removed, and any text burned into the image checked and removed — before being
> stored and analyzed on Wayne State University's secure research computing cluster; (3) the
> de-identified images will be used to develop and evaluate computer methods (machine
> learning models) for detecting fractures and identifying the vertebral level on trauma
> radiographs, with results compared against the CT and the radiology report; and (4) the
> approved number of records is increased from 2,000 to [N] to accommodate the imaging
> cohort. The master list linking study IDs to identifiers remains on DMC Citrix, separate
> from study data, as currently approved. No results are returned to the medical record or to
> treating clinicians; the work is retrospective and research-only. Nothing about consent,
> risk, or the participant population changes.

# eProtocol sections to update, with replacement text

**Personnel Information → Other Investigators (add).** Role for each: Student/Resident/
Fellow. Each must have current CITI (Biomedical Investigators; Health Information Privacy
and Security for Students and Instructors; Biomedical Responsible Conduct of Research) and
must sign the Obligations page and complete COI before submission.

- Ashley Schehr — student lead
- Annika Tekumulla
- Margret Khoushi
- Ryan Christian
- Dane Hubers
- Faris Mahjoub
- Hassan Saad
- Mia Sooch
- Sathya Siddapureddy
- Michael McLellan
- Jerick Kim
- Miraziz Ismoilov, MD, and Nizar Alnabahneh, MD (Radiology) — *only if they will
  adjudicate DMC images; omit if their role stays on public data.*

**Summary & Purpose → Purpose (append).**

> A further aim is to determine whether automated image-analysis methods can detect
> fractures and identify vertebral levels on trauma spine radiographs from this population,
> using the CT and radiology reports already obtained during care as the reference standard.

**Background, Rationale, Data Analysis and Procedures → Section C, Data Collection (revise).**

> Data will be collected from the DMC electronic medical record (Citrix) using the data
> collection sheet, and from DMC Radiology/PACS: thoracolumbar and lumbar spine radiographs
> (AP and lateral), spine CT, and the associated radiology reports, for patients already
> identified from the trauma registry. Images are de-identified within the DMC environment
> before any analysis.

**Background, Rationale, Data Analysis and Procedures → Section A(b), statistical methods
(replace "N/A").**

> Descriptive statistics for the cohort. For the imaging analysis: sensitivity and
> specificity of automated fracture detection against the radiology report and CT;
> agreement of automated vertebral-level identification with CT, reported separately for
> patients with and without transitional lumbosacral anatomy; and agreement of automated
> spinopelvic measurements with manual measurement.

**Participant Population → number of records (revise).** 2,000 → [N]. *Enter the same
number on the DMC application and in the protocol document; the 2022 DMC review flagged a
mismatch.*

**Procedures to Maintain Confidentiality → (c), (e), (f), (i), (j) (revise).**

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

**HIPAA (no change to identifiers used or disclosed).** Identifiers used remain: geographic
subdivisions, elements of dates, medical record number, other unique identifying number.
Disclosed: none. Imaging that leaves DMC is de-identified and carries none of the 18
identifiers. *Confirm the "Will PHI be disclosed to sponsors" answer is completed — the
2022 review flagged it blank.*

**Attachments (add, highlighted per amendment instructions).**

- Revised data collection sheet: "Orthopaedic Trauma Patient Variables" with imaging
  fields added (study ID, radiograph accession, views present, CT accession, report
  findings).
- Revised protocol document, with the imaging methodology and the named storage
  environment.
- Revised DMC Application: Section 5, re-select "DMC Radiology/Imaging Services" and
  state that images will be retrieved, not only records read.

# What each added person must do before Andreea can submit

1. Log in to eProtocol at https://ksprodweb.ovpr.wayne.edu/ with a WSU AccessID.
2. Confirm CITI is current: *Biomedical Investigators*, *HIPS for Students and
   Instructors*, *Biomedical Responsible Conduct of Research*. The protocol's own personnel
   pages show these are the ones checked.
3. Sign the **Obligations** page.
4. Complete the **COI disclosure**.

One incomplete person holds the whole amendment. The personnel list should be whoever is
done by the submission date, and the rest added on the next one.

# Open items before submission

- **[N]**, the new record count, from the trauma-log query (encounters with a spine
  radiograph and a CT report in the same encounter). Set it once, use it everywhere.
- Whether the two radiology residents adjudicate DMC images (add them) or only public data
  (do not).
- A labeling tool on WSU compute. The annotators have been using ITK-SNAP with a Google
  Colab server; Colab is commercial cloud and is excluded by the storage text above. ITK-SNAP
  or 3D Slicer on a WSU-managed machine, or self-hosted CVAT on the cluster, before any DMC
  image is opened.
