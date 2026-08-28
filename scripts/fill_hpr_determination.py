"""Fill the WSU Human Participant Research determination tool for OSC's public-data work.

The aim is one determination that covers every current and future Open Spine Consortium
project built on openly available imaging data, so the same form does not get re-filed per
paper. Breadth here comes from a SCOPE RULE, not from vagueness: the form states the criteria
a dataset must meet to fall under it, and states what pushes a project outside it. A
determination that just said "public data" would either be refused or, worse, be relied on
for a dataset that arrived with a data use agreement attached.

Two boundaries are written into the answers on purpose, because both are live for this group:

  RESTRICTED DATA IS OUT. Page 11 of this form says IRB approval IS required for
  restricted-use data, for datasets obtained by application, or where a data use agreement is
  involved. Those are common in medical imaging and one of them would void a blanket
  determination silently.

  THE FDA PATH IS OUT. Question 3.4 catches data about a device used on human specimens --
  de-identified ones included -- when it is submitted to the FDA for a research or marketing
  permit. Nothing here is FDA-bound today, and a future clearance filing needs its own
  determination rather than this one.

Sections A-D are the applicant's. The WSU IRB Determination on pages 11-12 is the IRB's own
and is deliberately left blank: pre-filling an official determination would be forging it.
"""
import argparse
import sys
from datetime import date

import fitz

TODO = "[COMPLETE]"

TITLE = ("Open Spine Consortium: Secondary Analysis of Publicly Available, De-Identified "
         "Imaging Datasets and the Derived Annotations, Software, and Models Produced "
         "From Them")

LOCATIONS = (
    "All activities are computational and take place at Wayne State University. No activity "
    "occurs at a clinical site, and no participant is contacted, recruited, observed, or "
    "enrolled at any location.\n\n"
    "Data are downloaded over the internet from public archives to WSU-managed computing "
    "resources, including the WSU High Performance Computing Grid and WSU-managed "
    "workstations. Analysis, annotation, model training, and manuscript preparation are "
    "performed on those resources. Derived annotations and software are published to public "
    "repositories (e.g. Zenodo, GitHub, Hugging Face) under open licences.")

DATA_OBTAINED = (
    "By direct download from public data archives. No data are collected from participants, "
    "and no records at any covered entity are accessed.\n\n"
    "This determination is intended to cover Open Spine Consortium projects that use "
    "datasets meeting ALL of the following criteria:\n"
    "(1) the dataset is already publicly available and was de-identified by the source "
    "before public release;\n"
    "(2) it can be downloaded by any member of the public without an application, request "
    "for permission, eligibility review, or data use agreement;\n"
    "(3) it is not designated restricted-use by the source;\n"
    "(4) the investigators receive no key, code, or other means of re-identification, and "
    "hold no relationship with the source enabling re-identification;\n"
    "(5) no attempt is made to re-identify any individual, and no external data are linked "
    "to the dataset for the purpose of, or with the effect of, re-identification.\n\n"
    "Examples of sources meeting these criteria include The Cancer Imaging Archive public "
    "collections, VerSe, and TotalSegmentator.\n\n"
    "If a dataset does NOT meet all five criteria -- in particular if it is restricted-use, "
    "requires a proposal or application to obtain, or is governed by a data use agreement -- "
    "it falls outside this determination and a separate IRB submission will be made before "
    "that dataset is used.")

PURPOSE = (
    "The Open Spine Consortium develops open, reproducible methods for analysing spinal and "
    "pelvic anatomy on computed tomography and radiographs. Its projects share one design: "
    "existing, publicly released, de-identified imaging is re-analysed to produce "
    "annotations, measurements, software, and machine-learning models, which are then "
    "released openly for others to use and check.\n\n"
    "The clinical problem motivating this work is that vertebral level is conventionally "
    "established by counting down from C2. An abdominal CT does not contain C2, so at the "
    "thoracolumbar junction a transitional vertebra, a rib borne by a lumbar vertebra, and a "
    "stump rib produce overlapping appearances, and the correct label depends on a count the "
    "field of view cannot support. Mislabelling a level has direct consequences for surgical "
    "planning. These datasets and tools are built to make that determination reproducible, "
    "and to state plainly where it cannot be made.\n\n"
    "The purpose of THIS determination is to establish, once, whether that pattern of work "
    "-- secondary analysis of openly available de-identified imaging, under the criteria "
    "stated in item 7 -- constitutes human participant research, rather than filing a "
    "separate determination for each resulting publication.")

OBJECTIVES = (
    "Across current and planned Open Spine Consortium projects:\n\n"
    "1. Produce and openly release expert-reviewed anatomical annotations of existing public "
    "imaging datasets (vertebrae by level, sacrum, pelvis, femurs, ribs by level, and "
    "surgical instrumentation where present).\n\n"
    "2. Characterise anatomical variation in these cohorts, particularly lumbosacral "
    "transitional anatomy and its effect on vertebral level assignment.\n\n"
    "3. Develop, train, and evaluate segmentation and measurement algorithms on those "
    "annotations, and release the software and trained models under open licences.\n\n"
    "4. Compute spinopelvic parameters (pelvic incidence, pelvic tilt, sacral slope and "
    "related measures) reproducibly from segmentations, and evaluate agreement against "
    "established methods.\n\n"
    "5. Report methods and results in peer-reviewed publications, with the underlying "
    "annotations and code archived publicly so results can be independently reproduced.\n\n"
    "No objective requires the identity of any individual, and none is served by knowing it.")

RESULTS_USE = (
    "Results are used to advance scientific and clinical understanding of spinal anatomy and "
    "its measurement, and are disseminated as:\n"
    "- peer-reviewed journal articles and conference papers;\n"
    "- openly licensed datasets of derived annotations, archived with a DOI;\n"
    "- open-source software and trained models;\n"
    "- educational and reference material for clinicians and researchers.\n\n"
    "Findings are not applied to the care of any individual whose images appear in these "
    "datasets. No individual is contacted, and no result is returned to any individual, "
    "because no individual can be identified from the data.\n\n"
    "Software released by the consortium is provided for research use only and is not "
    "offered for clinical diagnosis or treatment. Should the consortium later seek FDA "
    "clearance for any device, data supporting that application would fall under question "
    "3.4 of this tool and a separate determination would be obtained beforehand.")

PARTICIPANTS = (
    "There are no participants in this project as the term is used in 45 CFR 46.\n\n"
    "The project analyses imaging studies that were acquired previously, for clinical or "
    "research purposes unrelated to this work, and that were de-identified and released "
    "publicly by their source archives before the investigators obtained them. The "
    "investigators have no contact of any kind with the individuals whose images these were, "
    "have no means of identifying them, and receive no key or code that would permit "
    "re-identification.\n\n"
    "The information available about each imaging study is limited to de-identified "
    "technical and demographic attributes released publicly by the source, such as scanner "
    "manufacturer and model, slice thickness, reconstruction kernel, patient position, and "
    "coarse age band and sex. These do not identify an individual and are not linked to "
    "anything that does.\n\n"
    "For scale, the current flagship dataset comprises 802 abdominal CT records; other "
    "consortium projects use public radiograph and CT collections of comparable character.")

IDENTIFIERS = ("None. No individually identifiable information is obtained, used, generated, "
               "or retained. Datasets are de-identified by the source before public release, "
               "and the investigators receive no key or code permitting re-identification.")

OTHER_SOURCE = ("Public data archives that release de-identified imaging under open licences "
                "without application or data use agreement (for example The Cancer Imaging "
                "Archive public collections, VerSe, TotalSegmentator).")

NO_COVERED_ENTITY = ("N/A. No medical records at any covered entity are accessed. All "
                     "imaging is obtained from public archives that de-identified it before "
                     "public release.")

STATUS_FIELD = ("Wayne State Faculty DMC Staff WSU Graduate Student Karmanos Staff "
                "WSU Undergraduate Student J D Dingell VAMC Staff ResidentFellowTrainee "
                "OtherDivision or College")

IDENT_FIELD = ("If yes list all identifiers being collected eg name date of birth medical "
               "record number email address other codes etc")

SOURCE_FIELD = ("Medical Record Review Complete 6a Survey Interview Biobank Data Repository "
                "Other Describe")

COVERED_FIELD = ("Indicate the institutions Covered Entity you will be reviewing and "
                 "collecting medical record data from Check all that apply")

HOW_FIELD = ("Describe how the data will be obtained eg survey interview observation testing "
             "review of existing records etc")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="hpr_determination_tool_revised_11_2021.pdf")
    ap.add_argument("--out", default="OSC_HPR_Determination_public_datasets.pdf")
    ap.add_argument("--name", default="Gregory Schwing")
    ap.add_argument("--sponsor", default="Loren Schwiebert")
    a = ap.parse_args()

    doc = fitz.open(a.src)

    text_vals = {
        "Project Title": TITLE,
        "Name of person conducting the project": a.name,
        "Title": TODO,
        "Date": date.today().strftime("%m/%d/%Y"),
        "Email Address": TODO + " (WSU e-mail)",
        "Phone": TODO,
        STATUS_FIELD: TODO,
        "Department": TODO,
        "Campus Address": TODO,
        "Name": a.sponsor,
        "Title_2": "Professor, Department of Computer Science",
        "I do not have a Faculty SponsorSupervisorForm completed by": a.name,
        " Email": TODO + " (WSU e-mail)",
        "Describe the locations where activities will take place": LOCATIONS,
        IDENT_FIELD: IDENTIFIERS,
        SOURCE_FIELD: OTHER_SOURCE,
        COVERED_FIELD: NO_COVERED_ENTITY,
        HOW_FIELD: DATA_OBTAINED,
        "Describe the purpose of this project": PURPOSE,
        "Describe the objectives or aims for the project": OBJECTIVES,
        "Describe how the results will be usedapplied": RESULTS_USE,
        "Describe the participants for the project": PARTICIPANTS,
    }

    # Section D is the determination itself:
    #   1.1 and 1.2 checked   -> this IS research (generalizable knowledge)
    #   2.1 No, 2.2 No        -> it does NOT involve human participants
    #   3.1-3.4 all unchecked -> not FDA-regulated
    # Research yes + human participants no = not HPR, no IRB review required.
    checks = {
        "Behavioral social education nonmedical study": False,
        "Medical study": True,
        "Secondary or retrospective data collection": True,
        "Identifiable Protected Health Information PHI": False,
        "Prospective Data": False,
        "Secondary or retrospective collection of biospecimens": False,
        "Prospective collection of biospecimens": False,
        "Medical Record Review Complete 6a": False,
        "Survey": False,
        "Interview": False,
        "Biobank": False,
        "Data Repository": True,
        "Other_2": True,
        "Detroit Medical Center Facility": False,
        "Karmanos Cancer Institute": False,
        "JD Dingell Veterans Administration Medical Center": False,
        "Other_3": False,
        "NAProject does not involve the review of or collection of": True,
        "undefined_4": False,      # Q5 identities knowable -- Yes
        "undefined_5": True,       # Q5 -- No
        "undefined_6": True,       # 1.1 expands knowledge of a discipline
        "undefined_7": True,       # 1.1 applicable beyond the site of collection
        "undefined_8": True,       # 1.1 develops/tests theories or informs policy
        "undefined_9": True,       # 1.2 shared with a discipline, broad conclusion
        "undefined_12": False,     # 3.1 drug
        "Check Box3": False,       # 3.2 device safety/effectiveness in human subjects
        "Check Box4": False,       # 3.3 data to FDA regarding human subjects
        "undefined_15": False,     # 3.4 device on human specimens, data to FDA
    }
    radios = {
        "undefined_2": "No",       # Q3 access to identifiable info during collection
        "undefined_3": "No_2",     # Q4 data include individually identifiable info
        "undefined_10": "No_3",    # 2.1 intervention or interaction
        "undefined_11": "No_4",    # 2.2 identifiable private info or bio-specimens
    }

    filled = boxes = 0
    missing = set(text_vals) | set(checks) | set(radios)
    for page in doc:
        for w in page.widgets():
            n = w.field_name
            missing.discard(n)
            if w.field_type == fitz.PDF_WIDGET_TYPE_TEXT and n in text_vals:
                v = text_vals[n]
                w.field_value = v
                # the description boxes are most of a page each and the longest answer here
                # fills under a third of one, so size for reading rather than for fitting
                w.text_fontsize = 9 if len(v) > 200 else 11
                w.field_flags |= 4096                      # multiline
                w.update()
                filled += 1
            elif n in radios:
                on = [s for s in (w.button_states() or {}).get("normal", []) if s != "Off"]
                if on and on[0] == radios[n]:
                    w.field_value = on[0]
                    w.update()
                    boxes += 1
            elif n in checks:
                w.field_value = bool(checks[n])
                w.update()
                boxes += 1

    doc.save(a.out)
    print(f"  {filled} text field(s), {boxes} box(es) set")
    if missing:
        print(f"  WARNING these named fields were not found in the PDF: {sorted(missing)}")
    print(f"  wrote {a.out}")
    print("  pages 11-12 (WSU IRB Determination) left blank -- that section is the IRB's")
    return 0


if __name__ == "__main__":
    sys.exit(main())
