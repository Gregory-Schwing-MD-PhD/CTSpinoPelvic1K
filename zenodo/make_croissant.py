"""Generate croissant.json for the v6 deposit, from the deposit itself.

The file carried over from the NeurIPS submission was Hugging Face's auto-generated Croissant
with hand-written RAI sections bolted on, and almost every factual field in it is now wrong:

  the creator is "anonymous_neurips" and the url points at the anonymised review repo;
  the licence says cc-by-nc-4.0, but this release is CC BY-NC-SA 4.0 -- ShareAlike is not
    optional, it is inherited from CTSpine1K;
  the description says ~650 patients from three sources, with coverage "limited to the
    lumbosacral spine and bony pelvis" and no thoracic labels -- v6 has 802 records with
    per-level ribs, femurs, FOV-limited thoracic and surgical hardware;
  it states the v20 merged L5/L6 convention, which this release specifically abandoned: L6
    is its own identifier, and that distinction is the point of the dataset;
  distribution points at a parquet conversion of an HF repo, with a GitHub issue URL where
    the sha256 belongs;
  recordSet is empty, so nothing machine-readable describes the manifest at all;
  datePublished, which the spec requires, is absent.

Rather than patch that, this builds the file from the deposit: real checksums, the manifest's
actual fields and their inferred types, the label scheme as an enumeration, and the fourteen
authors. The RAI prose is kept where it is still true, corrected where the release moved past
it, and extended where v6 added something the old file could not have known about.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter
from datetime import date
from pathlib import Path

DOI = "10.5281/zenodo.22139643"
LICENSE = "https://creativecommons.org/licenses/by-nc-sa/4.0/"
# Croissant wants MAJOR.MINOR.PATCH and the official validator warns on "v6". This is
# the semver spelling of the same release, not a different one; the deposit, the
# manuscript and the Zenodo record all designate it v6, and the description says so.
VERSION = "6.0.0"

AUTHORS = [
    ("Schwing, Gregory", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Schehr, Ashley", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Tekumulla, Annika", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Khoushi, Margret", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Christian, Ryan", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Hubers, Dane", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Mahjoub, Faris", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Saad, Hassan", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Sooch, Mia", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Siddapureddy, Sathya", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("McLellan, Michael", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Kim, Jerick", "Department of Surgery, Detroit Medical Center / Wayne State University"),
    ("Ismoilov, Miraziz", "Department of Radiology, Detroit Medical Center / Wayne State University"),
    ("Alnabahneh, Nizar", "Department of Radiology, Detroit Medical Center / Wayne State University"),
]

CONTEXT = {
    "@language": "en",
    "@vocab": "https://schema.org/",
    "citeAs": "cr:citeAs",
    "column": "cr:column",
    "conformsTo": "dct:conformsTo",
    "containedIn": "cr:containedIn",
    "cr": "http://mlcommons.org/croissant/",
    "rai": "http://mlcommons.org/croissant/RAI/",
    "data": {"@id": "cr:data", "@type": "@json"},
    "dataBiases": "cr:dataBiases",
    "dataCollection": "cr:dataCollection",
    "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
    "dct": "http://purl.org/dc/terms/",
    "extract": "cr:extract",
    "field": "cr:field",
    "fileProperty": "cr:fileProperty",
    "fileObject": "cr:fileObject",
    "fileSet": "cr:fileSet",
    "format": "cr:format",
    "includes": "cr:includes",
    "isLiveDataset": "cr:isLiveDataset",
    "jsonPath": "cr:jsonPath",
    "key": "cr:key",
    "md5": "cr:md5",
    "parentField": "cr:parentField",
    "path": "cr:path",
    "personalSensitiveInformation": "cr:personalSensitiveInformation",
    "recordSet": "cr:recordSet",
    "references": "cr:references",
    "regex": "cr:regex",
    "repeated": "cr:repeated",
    "replace": "cr:replace",
    "sc": "https://schema.org/",
    "separator": "cr:separator",
    "source": "cr:source",
    "subField": "cr:subField",
    "transform": "cr:transform",
}

DESCRIPTION = (
    "802 abdominal CT records with a unified segmentation of the spine, pelvis, per-level "
    "ribs and femurs on one coordinate frame, built so that a lumbar vertebra can be "
    "identified when the vertebral count itself is in doubt. "
    "Vertebral numbering is conventionally established by counting down from C2. No abdominal "
    "CT contains C2, so at the thoracolumbar junction a thirteenth thoracic vertebra, a rib "
    "borne by a lumbar vertebra and a stump rib produce overlapping appearances, and which "
    "label is correct depends on a count the field of view does not support. This release is "
    "annotated to make that decidable where it can be and says so plainly where it cannot: "
    "L6 carries its own identifier rather than being merged with L5, and lumbar ribs have "
    "their own classes rather than being forced to be rib 12. "
    "The deposit contains the labels and the crosswalk from each annotation to the TCIA CT "
    "series it was drawn on, which the source collections never published. The CT images "
    "themselves are not included; fetch_from_tcia.py rebuilds them. "
    "This release is designated v6; the version field carries its semantic-versioning "
    "spelling, 6.0.0."
)

# What is still true from the NeurIPS file, corrected where v6 moved past it.
RAI_LIMITATIONS = [
    "Single-modality CT only; no MRI, radiographic, or clinical correlation is included.",
    "The source cohort is the TCIA CT Colonography screening study, which skews older-adult "
    "(50 and over) and US-based; paediatric and non-US populations are absent.",
    "Thoracic ground truth is field-of-view limited and does not extend to T1; 553 of 802 "
    "records have their topmost labelled vertebra cut by the edge of the reconstruction.",
    "Rib numbering is inferred by counting down from the top of the visible field, which on "
    "an abdominal scan is not the top of the thorax. The lowest rib is reliable; numbers "
    "above it are less so.",
    "A null Castellvi field means UNGRADED, not absence of a transitional vertebra: 33 of "
    "802 records carry a radiologist grade and the other 769 were not read for it.",
    "Postural measurements are supine. Pelvic incidence is not postural and needs no such "
    "caveat, but every patient was scanned twice and prone and supine acquisitions must not "
    "be pooled.",
    "Eleven records carry surgical instrumentation. An iatrogenic fusion is indistinguishable "
    "from a congenital one to a distance measurement, so these must be excluded from any "
    "analysis of the gap between the lowest lumbar vertebra and the sacrum.",
    "Label strength varies by structure and the release does not average over it: vertebral "
    "labels derive from radiologist-supervised sources, pelvic labels on records that lacked "
    "one are pseudolabelled, and the rib layer is a pseudolabel whose human review was "
    "triaged by an automated rule rather than being exhaustive.",
    "The shipped folds are cross-validation folds covering the whole cohort. There is no "
    "held-out test set, and with 15 lumbarisation and 15 sacralisation records a per-fold "
    "metric on the rare classes is estimated from about three cases.",
]

RAI_BIASES = [
    "Indication bias: all source CT volumes were acquired for colorectal cancer screening, so "
    "the cohort skews older and is enriched for screening-eligible populations.",
    "Geographic bias: TCIA CT Colonography is predominantly US-based; non-US populations are "
    "not represented.",
    "Demographic bias inherited from the upstream screening cohort, including likely "
    "under-representation of younger adults and of some racial and ethnic groups.",
    "Class imbalance across transitional strata: 769 records are labelled normal against 17 "
    "sacralisation, 14 lumbarisation and 2 semi-sacralisation.",
    "Anatomy bias: the field of view of a colonography acquisition differs from that of "
    "trauma or oncology CT, which may limit transfer to those domains.",
    "Half the recorded ages are rounded to a multiple of five, so age-stratified analyses "
    "should not assume year-level precision.",
]

RAI_USES = [
    "Benchmarking spinopelvic CT segmentation under transitional anatomical variation.",
    "Training and evaluating methods that identify a vertebral level from its own local "
    "morphology rather than by counting from a landmark outside the field of view -- the "
    "failure mode behind wrong-level spine surgery.",
    "Transitional vertebra detection and Castellvi-style classification.",
    "Auditing existing public segmentation tools for failure modes on transitional anatomy.",
    "Re-deriving reference morphometry at a sample size cadaveric series cannot reach, with "
    "the caveat that a screening cohort aged 50 and over does not make the range "
    "representative of a younger or surgical population.",
    "Building and checking patient-specific biomechanical models, for which 351 patients "
    "carry annotation on two acquisitions in different positions.",
    "Opportunistic screening research, including trabecular attenuation and vertebral "
    "morphometry.",
    "NOT intended for clinical decision-making, diagnosis, or surgical planning without "
    "independent validation and appropriate regulatory clearance.",
]

RAI_PSI = (
    "All source data are redistributed from public, de-identified releases (TCIA CT "
    "COLONOGRAPHY, CTSpine1K, CTPelvic1K). This deposit contains derived label masks and a "
    "tabular manifest; it contains no CT images and no DICOM headers. The manifest carries "
    "only de-identified technical and demographic attributes released publicly by the source "
    "-- scanner manufacturer and model, slice thickness, reconstruction kernel, patient "
    "position, coarse age and sex -- and no direct identifiers, dates of service, or "
    "institution-level identifiers. The spinopelvic field of view excludes facial anatomy, so "
    "re-identification by facial reconstruction is not a live risk for these volumes."
)

RAI_SOCIAL = (
    "The dataset targets a known and clinically consequential failure mode: silent "
    "miscounting and mislabelling of vertebrae in patients with lumbosacral transitional "
    "anatomy, which propagates into surgical planning and research pipelines. Wrong-level "
    "spine surgery is most often attributed to exactly this variation. Patients with "
    "anatomical variants are systematically underserved by tools trained on typical spines, "
    "and improving robustness on that population is the motivation. Foreseeable risks are "
    "over-reliance on automated labels in contexts the dataset was not designed for, and "
    "amplification of the upstream screening cohort's demographic skew if models trained on "
    "it are deployed without site-specific validation."
)

JSON_TYPE = {str: "sc:Text", bool: "sc:Boolean", int: "sc:Integer", float: "sc:Float"}


def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deposit", default="data/zenodo_upload")
    ap.add_argument("--out", default="croissant.json")
    a = ap.parse_args()
    D = Path(a.deposit)

    recs = json.loads((D / "manifest.json").read_text(encoding="utf-8"))
    recs = recs if isinstance(recs, list) else recs.get("records", list(recs.values()))
    scheme = json.loads((D / "dataset_labels.json").read_text(encoding="utf-8"))
    id_to_name = scheme.get("id_to_name", scheme)

    # --- distribution: real files, real checksums ---------------------------------
    dist = []
    for name, fmt, desc in (
        ("labels.zip", "application/zip",
         "Archive of the 802 label volumes; extracts to labels/."),
        ("manifest.json", "application/json",
         "One record per case: the TCIA series crosswalk, demographics, scanner, "
         "transitional label and Castellvi grade."),
        ("splits_5fold.json", "application/json",
         "Frozen patient-grouped, transitional-stratified five-fold cross-validation splits."),
        ("dataset_labels.json", "application/json",
         "The label scheme: identifier to structure name."),
        ("SHA256SUMS.txt", "text/plain",
         "Checksums for the 802 labels by extracted path and for the loose files."),
        ("README.md", "text/markdown", "Documentation, rebuild path and usage notes."),
        ("KNOWN_ISSUES.md", "text/markdown",
         "What to filter before which analysis; read before use."),
    ):
        p = D / name
        if not p.exists():
            continue
        dist.append({
            "@type": "cr:FileObject", "@id": name, "name": name,
            "description": desc,
            "contentUrl": f"https://zenodo.org/records/22139643/files/{name}",
            "encodingFormat": fmt, "sha256": sha256(p),
        })

    dist.append({
        "@type": "cr:FileSet", "@id": "label-volumes", "name": "label-volumes",
        "description": "The 802 gzipped NIfTI label volumes inside labels.zip. Each shares "
                       "its affine exactly with the CT rebuilt by fetch_from_tcia.py, so no "
                       "resampling or reorientation is needed to overlay them.",
        "containedIn": {"@id": "labels.zip"},
        "encodingFormat": "application/gzip",
        "includes": "labels/*_label.nii.gz",
    })

    # --- recordSet: the manifest's real fields, typed from the data ---------------
    types = {}
    for r in recs:
        for k, v in r.items():
            if v is None:
                types.setdefault(k, None)
                continue
            types[k] = JSON_TYPE.get(type(v), "sc:Text")
    fields = []
    for k in sorted(types):
        fields.append({
            "@type": "cr:Field", "@id": f"records/{k}", "name": k,
            "dataType": types[k] or "sc:Text",
            "source": {"fileObject": {"@id": "manifest.json"},
                       "extract": {"jsonPath": f"$[*].{k}"}},
        })

    label_field = {
        "@type": "cr:RecordSet", "@id": "label-scheme", "name": "label-scheme",
        "description": "The segmentation label scheme: every identifier that appears in a "
                       "label volume, and the structure it denotes.",
        "key": {"@id": "label-scheme/id"},
        "field": [
            {"@type": "cr:Field", "@id": "label-scheme/id", "name": "id",
             "dataType": "sc:Integer",
             "source": {"fileObject": {"@id": "dataset_labels.json"},
                        "extract": {"jsonPath": "$.id_to_name"}}},
            {"@type": "cr:Field", "@id": "label-scheme/name", "name": "name",
             "dataType": "sc:Text",
             "source": {"fileObject": {"@id": "dataset_labels.json"},
                        "extract": {"jsonPath": "$.id_to_name"}}},
        ],
        "data": [{"label-scheme/id": int(k), "label-scheme/name": v}
                 for k, v in sorted(id_to_name.items(), key=lambda kv: int(kv[0]))],
    }

    record_sets = [
        {"@type": "cr:RecordSet", "@id": "records", "name": "records",
         "description": "One row per annotated CT record, including the SeriesInstanceUID "
                        "crosswalk that maps each annotation to the TCIA series it was drawn "
                        "on.",
         "key": {"@id": "records/volume_id"},
         "field": fields},
        label_field,
    ]

    lstv = Counter(str(r.get("lstv_label")) for r in recs)
    doc = {
        "@context": CONTEXT,
        "@type": "sc:Dataset",
        "conformsTo": "http://mlcommons.org/croissant/1.1",
        "name": "CTSpinoPelvic1K",
        "description": DESCRIPTION,
        "version": VERSION,
        "license": LICENSE,
        "url": f"https://doi.org/{DOI}",
        "identifier": f"https://doi.org/{DOI}",
        "datePublished": date.today().isoformat(),
        "citeAs": ("G. Schwing, A. Schehr, A. Tekumulla, et al., CTSpinoPelvic1K: spine, "
                   "pelvis, ribs and femurs in one coordinate frame, annotated for "
                   f"lumbosacral transitional anatomy, Zenodo (2026), "
                   f"https://doi.org/{DOI}"),
        "creator": [{"@type": "sc:Person", "name": n, "affiliation":
                     {"@type": "sc:Organization", "name": aff}} for n, aff in AUTHORS],
        "publisher": {"@type": "sc:Organization", "name": "Zenodo"},
        "keywords": ["computed tomography", "spine", "pelvis", "segmentation",
                     "lumbosacral transitional vertebra", "Castellvi classification",
                     "vertebral labelling", "spinopelvic parameters", "ribs", "femur",
                     "surgical instrumentation", "medical imaging"],
        "isLiveDataset": False,
        "rai:dataCollection": (
            "Patient-level crosswalk of public sources: TCIA CT COLONOGRAPHY provides the "
            "de-identified CT (prone and supine per patient), CTSpine1K provides "
            "VerSe-convention vertebral masks on the COLONOG subset, and CTPelvic1K provides "
            "sacrum and hip masks. For each record the annotation is placed on the TCIA "
            "series with the highest bone coverage, separately per anatomy, and the resulting "
            "series identifier is published in manifest.json -- the mapping the source "
            "collections never released. Ribs and femurs were pseudolabelled and reviewed by "
            "trained annotators through a slot-based review tool with adjudication; "
            "transitional phenotype and Castellvi grade were read by clinicians. Vertebral "
            "identity is anchored on the twelfth rib rather than assigned by counting."
        ),
        "rai:dataLimitations": RAI_LIMITATIONS,
        "rai:dataBiases": RAI_BIASES,
        "rai:dataUseCases": RAI_USES,
        "rai:personalSensitiveInformation": RAI_PSI,
        "rai:dataSocialImpact": RAI_SOCIAL,
        "rai:hasSyntheticData": False,
        "distribution": dist,
        "recordSet": record_sets,
    }

    Path(a.out).write_text(json.dumps(doc, indent=2, ensure_ascii=False) + "\n",
                           encoding="utf-8")
    print(f"  wrote {a.out}")
    print(f"    {len(dist)} distribution entr(ies), {len(fields)} manifest field(s), "
          f"{len(label_field['data'])} label id(s)")
    print(f"    creators {len(AUTHORS)}, licence {LICENSE}")
    print(f"    transitional labels in the manifest: {dict(lstv)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
