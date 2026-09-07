"""paper/cns2026/make_cns_deck.py — the CNS 2026 talk on CTSpinoPelvic1K and the one-shot detector,
built on the lab's CNS template (CNS2026_Schwing_Abstract418.pptx) so it matches the other talk.

The template's slides are reused as the design: every content slide is a copy of one of its
"Section Header" slides with the section tag, title, footer and logo kept and the body
replaced. Result slides that depend on the folds still training carry a visible
[PENDING] marker and a one-line description of the number that will go there, so the deck
can be rehearsed now and filled in the day the folds finish.

    python paper/cns2026/make_cns_deck.py --abstract "Abstract NNN" --session "..." --out CNS2026_Schwing_LSTV.pptx
"""
from __future__ import annotations

import argparse
import copy
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
TEMPLATE = ROOT / "CNS2026_Schwing_Abstract418.pptx"
FIGS = HERE / "figs"
TEAL = RGBColor(0x0E, 0x5F, 0x58)
INK = RGBColor(0x1A, 0x1C, 0x18)
PENDING = RGBColor(0xB5, 0x53, 0x3C)


# ------------------------------------------------------------------ slide plumbing
def dup_slide(prs, src):
    """Copy a slide (shapes + picture relationships) onto a new slide of the same layout."""
    dst = prs.slides.add_slide(src.slide_layout)
    for shp in list(dst.shapes):            # drop layout placeholders
        shp._element.getparent().remove(shp._element)
    rel_map = {}
    for rel in src.part.rels.values():
        if "image" in rel.reltype:
            rel_map[rel.rId] = dst.part.relate_to(rel._target, rel.reltype)
    for shp in src.shapes:
        el = copy.deepcopy(shp._element)
        for blip in el.iter("{http://schemas.openxmlformats.org/drawingml/2006/main}blip"):
            r = blip.get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed")
            if r in rel_map:
                blip.set("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed", rel_map[r])
        dst.shapes._spTree.append(el)
    return dst


def shape(slide, name):
    for s in slide.shapes:
        if s.name == name:
            return s
    return None


def remove(slide, *names):
    for n in names:
        s = shape(slide, n)
        if s is not None:
            s._element.getparent().remove(s._element)


def set_text(shp, lines, size=None, bold_first=False, color=None):
    """Replace a text frame's content; keeps the first run's formatting as the base."""
    tf = shp.text_frame
    base = tf.paragraphs[0].runs[0] if tf.paragraphs and tf.paragraphs[0].runs else None
    font_name = base.font.name if base is not None else None
    font_size = base.font.size if base is not None else None
    font_color = None
    try:
        font_color = base.font.color.rgb if base is not None and base.font.color and base.font.color.type else None
    except Exception:  # noqa: BLE001
        font_color = None
    for p in list(tf.paragraphs)[1:]:
        p._p.getparent().remove(p._p)
    p0 = tf.paragraphs[0]
    for r in list(p0.runs):
        r._r.getparent().remove(r._r)
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        p = p0 if i == 0 else tf.add_paragraph()
        run = p.add_run()
        run.text = line
        if font_name:
            run.font.name = font_name
        run.font.size = Pt(size) if size else font_size
        if font_color is not None:
            run.font.color.rgb = font_color
        if color is not None:
            run.font.color.rgb = color
        if bold_first and i == 0:
            run.font.bold = True


def textbox(slide, left, top, width, height, lines, size=16, color=INK, bold=False, align=None):
    tb = slide.shapes.add_textbox(Emu(left), Emu(top), Emu(width), Emu(height))
    tf = tb.text_frame
    tf.word_wrap = True
    if isinstance(lines, str):
        lines = [lines]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        r = p.add_run(); r.text = line
        r.font.size = Pt(size); r.font.color.rgb = color; r.font.bold = bold
        if align:
            p.alignment = align
    return tb


def picture(slide, path, left, top, width=None, height=None):
    kw = {}
    if width:
        kw["width"] = Emu(width)
    if height:
        kw["height"] = Emu(height)
    return slide.shapes.add_picture(str(path), Emu(left), Emu(top), **kw)


def content_slide(prs, proto, section, title):
    s = dup_slide(prs, proto)
    set_text(shape(s, "Text Placeholder 12"), section)
    set_text(shape(s, "Title 11"), title)
    return s


def pending(slide, left, top, width, what):
    textbox(slide, left, top, width, 500000,
            ["[PENDING] " + what], size=13, color=PENDING, bold=True)


# ---------------------------------------------------------------------- content
def build(abstract: str, session: str, out: Path):
    prs = Presentation(str(TEMPLATE))
    T = list(prs.slides)              # 17 template slides
    footer_text = f"CNS 2026  ·  Washington, DC  ·  {abstract}"

    # ---- 1 title (edit in place)
    s = T[0]
    set_text(shape(s, "TextBox 1030"), "Naming a Lumbar Level Without Counting From C2: CTSpinoPelvic1K and a One-Shot Detector for Transitional Anatomy")
    set_text(shape(s, "TextBox 1031"), "Gregory J. Schwing, MD, PhD¹   ·   Ashley Schehr²   ·   the OpenSpineConsortium annotators²   ·   Miraziz Ismoilov, MD³   ·   Nizar Alnabahneh, MD³")
    set_text(shape(s, "TextBox 1032"), ["¹ Department of Surgery, Detroit Medical Center / Wayne State University, Detroit, MI",
                                        "² Wayne State University School of Medicine   ·   ³ Department of Radiology, Detroit Medical Center / Wayne State University"])
    set_text(shape(s, "TextBox 1033"), [abstract + "   ·   " + session, "CNS 2026, Washington, DC"])
    set_text(shape(s, "TextBox 1034"), footer_text)
    set_text(shape(s, "Text Placeholder 12"), "Detroit Medical Center / Wayne State University  ·  OpenSpineConsortium")

    # ---- 2 disclosures
    s = T[1]
    set_text(shape(s, "TextBox 24"), ["The authors have no relevant financial relationships to disclose.",
                                      "All imaging is public and de-identified (TCIA CT COLONOGRAPHY); the WSU IRB determined the work is not human-participant research (2 September 2026).",
                                      "Dataset, code and weights are released under open licences; nothing presented is a medical device."])
    set_text(shape(s, "TextBox 23"), footer_text)

    # ---- 3 overview
    s = T[2]
    set_text(shape(s, "TextBox 23"), footer_text)
    items = ["The problem — a lumbar level is named by counting from C2, and the study a lumbar case is planned on has no C2",
             "The dataset — 802 CT records with spine, pelvis, ribs, femora and hardware in one frame, and a class for every anomaly",
             "The claim — identity is local: shape alone separates T12 from L1 at AUC 0.99 in the same patient",
             "The model — a one-shot network that never counts, plus a decoder that turns its probabilities into a posterior over the readings",
             "What it says about development, and about the operating room"]
    for name, txt in zip(("TextBox 25", "TextBox 27", "TextBox 29", "TextBox 31", "TextBox 33"), items):
        set_text(shape(s, name), txt)

    proto_text = T[3]        # background slide with a text column
    proto_two = T[4]         # already / not described boxes
    proto_flow = T[6]        # three boxes with arrows
    proto_fig = T[7]         # picture left, text right
    proto_fig_wide = T[9]    # wide picture, text right
    proto_concl = T[14]
    proto_limits = T[15]
    proto_thanks = T[16]

    new = []

    # ---- 4 background: the count
    s = content_slide(prs, proto_text, "Background", "The count fails exactly where the anatomy is transitional")
    remove(s, "Oval 25", "Oval 26", "Oval 27", "TextBox 28", "TextBox 29", "TextBox 30")
    set_text(shape(s, "TextBox 24"), [
        "Wrong-level spine surgery runs at about one in 3,100 spinal procedures, and the cause most often cited is transitional anatomy.¹",
        "A lumbosacral transitional vertebra is present in roughly one patient in ten; a thoracolumbar variant (a stump twelfth rib, a lumbar rib, a thirteenth thoracic vertebra) in more.",
        "The gold standard names a level by counting caudally from C2 on whole-spine imaging. A lumbar surgical case is planned on a lumbar study, T12 to S1. C2 is never in it.",
        "So the name rests on local morphology, and the question is whether morphology is enough."])
    shape(s, "TextBox 24").width = Emu(6000000)
    picture(s, FIGS / "fig_fov.png", 6500000, 1371600, width=5300000)
    set_text(shape(s, "TextBox 31"), "1. Epstein NE. Surg Neurol Int 2021;12:286.   Field of view of 802 abdominopelvic CTs: the thoracic column enters from above; C2 never.")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 5 background: the three readings
    s = content_slide(prs, proto_two, "Background", "The same count, three anatomies")
    set_text(shape(s, "Rounded Rectangle 24"), "Four rib-free bodies above the sacrum")
    set_text(shape(s, "TextBox 25"), ["An L1 that carries a lumbar rib and is counted as thoracic,",
                                      "or an L5 assimilated to the sacrum (sacralization).",
                                      "Same count. Opposite surgical anatomy."])
    set_text(shape(s, "Rounded Rectangle 26"), "Six rib-free bodies above the sacrum")
    set_text(shape(s, "TextBox 27"), ["A true sixth lumbar vertebra (A),",
                                      "a T12 whose ribs are aplastic, counted as lumbar (B),",
                                      "or a first sacral segment that separated (C, lumbarization)."])
    set_text(shape(s, "Rounded Rectangle 28"), "Can the vertebra's own shape, its ribs and its junction say which one it is, without the count?")
    set_text(shape(s, "TextBox 29"), "2. Konin GP, Walz DM. AJNR 2010;31:1778–1786.   3. Poolman TM, et al. J Anat 2023;243:311–318.")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 6 methods: dataset
    s = content_slide(prs, proto_fig, "Methods", "CTSpinoPelvic1K: two anchors on every record, and a class for every anomaly")
    remove(s, "Picture 24")
    picture(s, FIGS / "fig_anchors.png", 411480, 1371600, width=5600000)
    set_text(shape(s, "TextBox 25"), [
        "802 abdominopelvic CT records from TCIA CT COLONOGRAPHY, joining CTSpine1K's vertebrae and CTPelvic1K's pelvis on one series, which neither source had done.",
        "Added on every record: per-level ribs (thirteen per side), femora, a separate S1, and surgical hardware as its own class.",
        "Lumbar levels are anchored on the lowest rib-bearing vertebra and on S1, not on a count. L6, T13, lumbar ribs and stump ribs are recorded as themselves.",
        "33 records carry a two-reader Castellvi grade. Sixteen carry a lumbar rib; 98 a stump twelfth rib; 18 a sixth lumbar body.",
        "v10 archived on Zenodo (10.5281/zenodo.22139642); build archive 10.5281/zenodo.22647933; submitted to Medical Physics."])
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 7 methods: validation
    s = content_slide(prs, proto_fig_wide, "Methods", "Validated at three layers before any number was believed")
    remove(s, "Picture 24")
    picture(s, FIGS / "fig_validation.png", 256032, 1554480, width=7900000)
    set_text(shape(s, "TextBox 25"), [
        "Geometry first: 802 of 802 records pass the invariants (one component per bone, sided structures on their side, labels on their CT grid).",
        "Rib–vertebra incidence across 5,749 evaluable ribs: 2 residual offsets, 0.035 percent.",
        "Derived spinopelvic measures match published reference values."])
    set_text(shape(s, "TextBox 26"), "Every measurement in this talk was written to a file by a script that can be re-run on the released data.")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 8 methods: the model as a flow
    s = content_slide(prs, proto_flow, "Methods", "A network that never counts, and a decoder that reasons about the sequence")
    set_text(shape(s, "Rounded Rectangle 24"), "One-shot segmenter")
    set_text(shape(s, "TextBox 25"), ["nnU-Net ResEnc-L, 24 classes: T10–T13, L1–L6,", "sacrum, ribs 1–11, rib 12, rib 13, lumbar rib, hips, femur"])
    set_text(shape(s, "Rounded Rectangle 27"), "Instances + type probabilities")
    set_text(shape(s, "TextBox 28"), ["per vertebra: P(thoracic), P(lumbar), P(sacral)", "from the softmax; rib contact per body"])
    set_text(shape(s, "Rounded Rectangle 30"), "Monotone decode")
    set_text(shape(s, "TextBox 31"), ["thoracic above lumbar above sacrum;", "posterior over the rib-free count and over A / B / C"])
    set_text(shape(s, "TextBox 32"), [
        "Every class is a local appearance. To call a voxel L1 rather than T12 the network has to read the body's own shape and facets; to call a rib lumbar rather than twelfth it has to read the vertebra beneath it. That is the morphology classifier and the segmenter trained as one model.",
        "What a semantic network cannot do is notice that its own sequence does not fit twelve, five and a sacrum. The decoder adds only that: instances, type probabilities, and the most likely monotone sequence, with the probability of each reading reported rather than a verdict forced.",
        "Rare-class exposure is enforced: records carrying a lumbar rib, a T13 or a lumbosacral variant fill half of every training queue. Five patient-grouped, LSTV-stratified folds; 500 epochs each on H200 GPUs."], size=14)
    set_text(shape(s, "TextBox 33"), "4. Isensee F, et al. Nat Methods 2021;18:203–211.   5. Wald T, et al. TMLR 2025 (ResEnc).")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 9 results: shape separability
    s = content_slide(prs, proto_fig_wide, "Results", "Identity is local: T12 and L1 differ in shape inside the same patient")
    remove(s, "Picture 24")
    picture(s, FIGS / "fig_levelatlas.png", 256032, 1554480, width=7900000)
    set_text(shape(s, "TextBox 25"), [
        "Eleven shape features per vertebra, each divided by the patient's own median across levels so size is removed and only shape remains.",
        "Logistic regression, cross-validated by patient: thoracic against lumbar AUC 0.998; T12 against L1 AUC 0.990, accuracy 97.3 percent over 1,485 vertebrae.",
        "Transverse-process span carries most of it. Costal facets and facet orientation are not in that feature set; the network sees them."])
    set_text(shape(s, "TextBox 26"), "Morphometry by level with its spread, 802 records: the reference the model is measured against.")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 10 results: six-lumbar readings
    s = content_slide(prs, proto_fig_wide, "Results", "Eighteen six-lumbar spines, read from the labels: where is the extra body?")
    remove(s, "Picture 24")
    picture(s, FIGS / "six_lumbar_readings.png", 256032, 1554480, width=7900000)
    set_text(shape(s, "TextBox 25"), [
        "Every feature as a z-score against 643 normal five-lumbar spines.",
        "Bottom body: a transverse process 2–5 SD taller than a normal L5's, sitting 2 SD closer to the ilium, or a sacrum a segment short, reads as a lumbarized S1.",
        "Top body: the whole column scored as labelled and with every name moved up one; a labelled L1 that fits T12 better reads as an aplastic-rib T12.",
        "11 true L6  ·  4 lumbarized S1  ·  1 aplastic-rib T12  ·  2 with evidence at both ends."])
    set_text(shape(s, "TextBox 26"), "The label convention hides an aplastic twelfth rib (the lowest rib-bearing body is defined as T12), which is why the column-wide test, not the rib length, finds reading B.")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 11 results: the reading sheet
    s = content_slide(prs, proto_fig, "Results", "What the evidence looks like on the CT")
    remove(s, "Picture 24")
    picture(s, FIGS / "reading_sheet_0376.png", 411480, 1371600, height=4600000)
    set_text(shape(s, "TextBox 25"), [
        "Record 0376, Castellvi IV. Top left: the labelled L1 at pedicle level. Top right: coronal MIP through it, no rib. Bottom left: the labelled sixth body sitting on both alae. Bottom right: the disc beneath it and the sacral height.",
        "The bottom test calls it sacral-type; the column-wide test calls the top thoracic-type. A double shift is a real anatomy (presacral count 25 with a rib-less T12), and the field of view cannot exclude it.",
        "That is the point: on this study a verdict is not available to a human either. A probability is, and it is what the decoder reports."])
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 12 results: co-occurrence
    s = content_slide(prs, proto_two, "Results", "Do the two borders shift together? First look at 802 records")
    set_text(shape(s, "Rounded Rectangle 24"), "Thoracolumbar border")
    set_text(shape(s, "TextBox 25"), ["Stump twelfth rib (12th:11th length below 0.33): 98 of 788, 12.4 percent.",
                                      "Lumbar rib: 16 of 802, 2.0 percent.",
                                      "No six-lumbar record carries an aplastic twelfth rib by rib length; one does by column shape."])
    set_text(shape(s, "Rounded Rectangle 26"), "Lumbosacral border")
    set_text(shape(s, "TextBox 27"), ["Sacralization: stump ribs in 4 of 14 against 94 of 758 elsewhere, odds ratio 2.8, p = 0.09.",
                                      "Lumbarization: no lumbar rib among the 14.",
                                      "Direction agrees with Nagata 2025; power does not, at 33 graded records."])
    set_text(shape(s, "Rounded Rectangle 28"), "A single homeotic shift, cranial or caudal, predicts these pairings. Reading the whole cohort, and VerSe, is what tests it at full power.")
    set_text(shape(s, "TextBox 29"), "6. Nagata K, et al. Skeletal Radiol 2025;54:2169–2177.   7. Du Plessis AM, et al. J Anat 2018;232:850–856.")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 13 results: the network (pending)
    s = content_slide(prs, proto_fig_wide, "Results", "The one-shot network on held-out folds")
    remove(s, "Picture 24")
    pending(s, 256032, 1554480, 7900000, "figure: per-class Dice, typical versus transitional records, five folds (tools/eval_oneshot.py)")
    pending(s, 256032, 2200000, 7900000, "figure: rib-free count accuracy by transitional status; posterior calibration (tools/decode_sequence.py)")
    pending(s, 256032, 2850000, 7900000, "table: where ground-truth twelfth-rib, thirteenth-rib and lumbar-rib voxels went; lumbar-rib recall")
    set_text(shape(s, "TextBox 25"), [
        "Three numbers decide whether the claim holds, and none of them is overall Dice:",
        "1. level identity accuracy on transitional records, reported apart from typical ones;",
        "2. whether the network ever says 'lumbar rib' or 'T13' (a model that never does still scores 94 percent on short ribs);",
        "3. the posterior on the eighteen six-lumbar spines against the label-based readings."])
    set_text(shape(s, "TextBox 26"), "Folds launched 7 September 2026 on four H200 nodes; expected complete about 12 September.")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 14 results: X-ray
    s = content_slide(prs, proto_two, "Results", "The same idea on the image the surgeon looks at")
    set_text(shape(s, "Rounded Rectangle 24"), "A multi-source radiograph corpus")
    set_text(shape(s, "TextBox 25"), ["33,943 vertebra instances on 5,401 films: BUU-LSPINE 19,991; AASCE full-spine 8,069; NHANES II 3,217; Mendeley 2,666.",
                                      "Plus 4,000 rendered radiographs from CTSpinoPelvic1K whose corners are exact by construction.",
                                      "Levels recorded as annotated or inferred from position, never invented."])
    set_text(shape(s, "Rounded Rectangle 26"), "A class-agnostic corner detector")
    set_text(shape(s, "TextBox 27"), ["Baseline on 6,073 instances: AP50 0.957, corners within 10 percent of body size 77 percent.",
                                      "Retraining on the full corpus now. [PENDING] v2 numbers.",
                                      "Fails today on outside films: post-operative, instrumented, thoracic levels present."])
    set_text(shape(s, "Rounded Rectangle 28"), "Aim: the level named on the intraoperative film, from local shape, before the incision, with a probability attached.")
    set_text(shape(s, "TextBox 29"), "8. Klinwichit P, et al. BUU-LSPINE, 2023.   9. Wu H, et al. AASCE MICCAI 2019.")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 15 conclusions
    s = content_slide(prs, proto_concl, "Conclusions", "Conclusions")
    set_text(shape(s, "TextBox 24"), [
        "A lumbar level can be named without counting from C2, because a T12 and an L1 are different shapes in the same patient, and a lumbarized S1 keeps sacral features. CTSpinoPelvic1K makes that measurable on 802 records, with the anomalies recorded as themselves rather than forced into an ordinary level.",
        "Six rib-free bodies are, in this cohort, a true L6 or a caudally shifted sacral segment; the thoracolumbar border is not the usual culprit. Where the evidence sits at both ends, the honest output is a probability, and that is what the decoder gives.",
        "For the surgeon the origin of the border vertebra decides where the mobile junction is, what the pedicle will take, which root exits beneath it, and which endplate the pelvic parameters were measured on."])
    set_text(shape(s, "Rounded Rectangle 25"), "Identity by local shape, with a posterior over the readings, is the deliverable a wrong-level detector needs and a count can never give.")
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 16 limitations
    s = content_slide(prs, proto_limits, "Limitations", "Limitations and next steps")
    set_text(shape(s, "TextBox 24"), ["Limitations",
                                      "Thoracic ground truth is field-of-view limited; no scan contains C2, so the readings are morphological consistency, not developmental truth.",
                                      "33 Castellvi grades, two residents' consensus; the label-based readings rest on 18 six-lumbar records.",
                                      "Supine screening cohort over 50; ribs are triaged-review pseudolabels; no held-out test set beyond the folds."])
    set_text(shape(s, "TextBox 25"), ["Next steps",
                                      "Run the released weights over the 374 VerSe scans to add S1 and the anomaly classes, and read Castellvi on the whole cohort.",
                                      "Replace the S1 carve with one that follows anatomy.",
                                      "Train the corner detector on the enlarged radiograph corpus and on real intraoperative films with the operated level labelled.",
                                      "Students: one command onboards a new annotator (openspineconsortium.com/onboarding)."])
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- 17 thanks
    s = content_slide(prs, proto_thanks, "Thank you", "Acknowledgments and data availability")
    set_text(shape(s, "TextBox 24"), [
        "Annotators and coauthors: Ashley Schehr, Annika Tekumulla, Margret Khoushi, Ryan Christian, Dane Hubers, Faris Mahjoub, Hassan Saad, Mia Sooch, Sathyagopal Siddapureddy, Michael McLellan, Jerick Kim (WSU School of Medicine); Miraziz Ismoilov and Nizar Alnabahneh (Radiology, DMC/WSU).",
        "Data: CTSpinoPelvic1K v10, doi.org/10.5281/zenodo.22139642 (CC BY-NC-SA 4.0); build archive doi.org/10.5281/zenodo.22647933.",
        "Code and weights: github.com/OpenSpineConsortium/CTSpinoPelvic1K; huggingface.co/OpenSpineConsortium.",
        "Imaging: TCIA CT COLONOGRAPHY; annotations derive from CTSpine1K and CTPelvic1K."])
    set_text(shape(s, "Rounded Rectangle 25"), ["Questions", "gregory.schwing@med.wayne.edu", "openspineconsortium.com"])
    set_text(shape(s, "TextBox 23"), footer_text)
    new.append(s)

    # ---- drop the template's own content slides 4..17 (indices 3..16), keep 1..3
    xml_slides = prs.slides._sldIdLst
    RID = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"
    entries = list(xml_slides)                 # same order as prs.slides at this point
    for idx in range(16, 2, -1):
        r = entries[idx]
        xml_slides.remove(r)
        prs.part.drop_rel(r.get(RID))
    prs.save(str(out))
    print("wrote", out, "slides:", len(prs.slides))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--abstract", default="Abstract [number pending]")
    ap.add_argument("--session", default="[session pending]")
    ap.add_argument("--out", default=str(Path.home() / "OneDrive/Desktop/CNS2026_Schwing_LSTV.pptx"))
    a = ap.parse_args()
    build(a.abstract, a.session, Path(a.out))
