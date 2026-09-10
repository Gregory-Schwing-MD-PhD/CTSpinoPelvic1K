r"""Final pass: the prone/supine mismatch is explained three times; state it once, in the
methods, and trim the last repeated wording elsewhere.
"""
import re
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "main.tex"
s = p.read_text(encoding="utf-8")
before = len(s)


def sub_once(text, old, new):
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    hits = pat.findall(text)
    assert len(hits) == 1, f"expected one match, found {len(hits)}:\n---\n{old[:150]}\n---"
    return pat.sub(lambda _m: new, text, count=1)


edits = [
    # the mechanism and its consequence, merged into one paragraph
    ("That mapping matters because every patient in CT~COLONOGRAPHY was scanned \\emph{twice}, prone and supine, and an annotation carries a patient identifier and nothing finer. Turning a patient from supine to prone alters lumbar lordosis and the alignment of each segment against the next, so outlines drawn on one series are the wrong \\emph{shape} for the other and no rigid realignment recovers them. The crosswalk (Fig.~\\ref{fig:pipeline}) therefore resolves each annotation to a patient and then, within that patient's volumes, to the series whose bone \\emph{agrees} with it, where candidates are thresholded at bone attenuation and scored by overlap with the mask.",
     "That mapping matters because every patient in CT~COLONOGRAPHY was scanned \\emph{twice},\nprone and supine, and an annotation carries a patient identifier and nothing finer, so\nneither collection recorded which acquisition it had labeled. Turning a patient from supine\nto prone alters lumbar lordosis and the alignment of each segment against the next, so\noutlines drawn on one series are the wrong \\emph{shape} for the other and no rigid\nrealignment recovers them. The crosswalk (Fig.~\\ref{fig:pipeline}) therefore resolves each\nannotation to a patient and then, within that patient's volumes, to the series whose bone\n\\emph{agrees} with it, where candidates are thresholded at bone attenuation and scored by\noverlap with the mask. It mattered for 351 patients, nearly half the cohort, whose two\ncollections had labeled different acquisitions of the same person; those records are\nreleased but held out of any analysis that assumes a single posture, two labeled postures of\none patient being a resource rather than a defect."),

    ("CTSpine1K and CTPelvic1K each annotated a patient independently, so their labels are not guaranteed to come from the same CT series (Fig.~\\ref{fig:pipeline}). This turned out to matter for 351 patients, nearly half the cohort, because the two collections had labeled different acquisitions of the same person, one prone and one supine, with neither recording which. Because posture changes spinal alignment, a spine label from one acquisition cannot simply be paired with a pelvic label from the other. These 351 records are released but held out of any analysis that assumes a single posture; two labeled postures of one patient are a resource, not a defect.",
     "PARAGRAPH_TO_DELETE"),

    # the introduction need not rehearse the mechanism the methods set out
    ("It is not a file merge: every patient was scanned prone and supine and neither source recorded which series it drew on, so each annotation had to be traced back to its own series (Sec.~\\ref{sec:sources}). Done once, that crosswalk yields a spine-and-pelvis frame at 802 patients, and the bones neither set had, ribs, femora, S1 and hardware, could then be annotated against it rather than from scratch. Two of the most-used public bone annotation sets were halves of one dataset; this release is the whole.",
     "It is not a file merge, because each annotation had to be traced back to the series it was\ndrawn on (Sec.~\\ref{sec:sources}). Done once, that crosswalk yields a spine-and-pelvis frame\nat 802 patients, and the bones neither set had, ribs, femora, S1 and hardware, could then be\nannotated against it rather than from scratch."),

    # the consequence of the gap is already in the clause before it
    ("No public collection can therefore record a six-lumbar spine together with both of its borders, and a segmenter trained on them inherits the gap: faced with an enumeration anomaly it must shift the whole column by a level or absorb a vertebra into its neighbor.",
     "No public collection can therefore record a six-lumbar spine together with both of its\nborders, and a segmenter trained on them must, faced with an enumeration anomaly, shift the\nwhole column by a level or absorb a vertebra into its neighbor."),

    # the two classical properties can be named without the aside
    ("reproducing two properties of the classical figures, body height crossing over at T11--T12 and reversing by L4--L5 and pedicle width widening steeply into the lower lumbar spine, and adding the width of each distribution, which is the part needed to judge an individual.",
     "reproducing two properties of the classical figures, body height crossing over at T11--T12\nand reversing by L4--L5 and pedicle width widening into the lower lumbar spine, and adding\nthe width of each distribution, the part needed to judge an individual."),

    # the ground-truth statement and the positional statement each open with the same frame
    ("\\emph{What the release does in addition.} Every vertebra carries the identifier its radiologist-sourced annotation assigned, corrected where the source was wrong; those labels are the ground truth. Alongside them, every released measurement is also expressed against the two landmarks present in essentially every abdominal scan, the sacrum below and the lowest rib-bearing vertebra above,",
     "\\emph{What the release does in addition.} Every vertebra carries the identifier its\nradiologist-sourced annotation assigned, corrected where the source was wrong; those labels\nare the ground truth. Every released measurement is also expressed against the two landmarks\npresent in essentially every abdominal scan, the sacrum below and the lowest rib-bearing\nvertebra above,"),

    # the reviewers' training and the audit trail are one clause
    ("Reviewers were medical students working through \\consortium~\\cite{osc2026}, trained against a written labeling protocol, with every correction recorded against the annotator who made it.",
     "Reviewers were medical students working through \\consortium~\\cite{osc2026}, trained against\na written labeling protocol, with every correction attributed to its annotator."),
]

for old, new in edits:
    s = sub_once(s, old, new)

# remove the now-duplicated paragraph together with its blank line
s = re.sub(r"\n*PARAGRAPH_TO_DELETE\n*", "\n\n", s)

p.write_text(s, encoding="utf-8")
print(f"applied {len(edits)} edits; {before} -> {len(s)} chars ({before - len(s)} saved)")
