r"""Apply A. Schehr's second pass, from 'updated references.docx'.

Adopted here, with the reasons:

  * TABLE II WAS WRONG on two rows, and she is right about both. Vialle's abstract reads
    "60 degrees 10 degrees for maximum lumbar lordosis ... 55 degrees +/- 10.6 degrees for
    pelvic incidence". The table carried 54.7 +/- 10.6 for pelvic incidence, which is this
    dataset's own measured value copied into the reference column, and 43 +/- 11.2 for
    lumbar lordosis, which appears in Vialle nowhere. Corrected to 55 +/- 10.6 and 60 +/- 10.
    Her further point holds: with the right reference the comparison is stronger, because a
    supine lordosis of 52.7 degrees sitting below a standing 60 is the direction posture
    predicts, whereas the erroneous 43 had the cohort above the reference.

  * SIXTA 2012, the EAST practice management guideline on screening for thoracolumbar
    injuries in blunt trauma, is a peer-reviewed indexed guideline and a better citation
    than a best-practices document. Added alongside TQIP rather than replacing it, because
    TQIP is the source that states the whole-spine escalation rule in as many words
    ("screen the entire spine whenever an injury of the spine is identified") and Sixta's
    abstract does not.

  * BERRY 1987 is a primary cadaveric morphometry study, 30 skeletons, and its abstract
    states the per-level behaviour directly: "Vertebral body height increases caudally ...
    the major spinal canal diameter slightly increase[s] caudally". Added to the
    differ-by-level citation and to the reference-morphometry sentence.
"""
import re
from pathlib import Path

MPDA = Path(__file__).resolve().parent.parent


def sub_once(text, old, new):
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    hits = pat.findall(text)
    assert len(hits) == 1, f"expected one match, found {len(hits)}:\n{old[:140]}"
    return pat.sub(lambda _m: new, text, count=1)


# ---------------------------------------------------------------- bibliography
bib = (MPDA / "ctspinopelvic1k.bib").read_text(encoding="utf-8")
ADD = r"""
% Added by A. Schehr's verification pass, September 2026. Verified against PubMed
% 23114489: an EAST practice management guideline, peer reviewed and indexed, on screening
% for thoracolumbar injury in blunt trauma.
@article{sixta2012,
  author  = {Sixta, Sherry and Moore, Forrest O. and Ditillo, Michael F. and Fox, Adam D. and
             Garcia, Alejandro J. and Holena, Daniel and Joseph, Bellal and Tyrie, Leslie and
             Cotton, Bryan and others},
  title   = {Screening for thoracolumbar spinal injuries in blunt trauma: an {Eastern
             Association for the Surgery of Trauma} practice management guideline},
  journal = {J. Trauma Acute Care Surg.},
  volume  = {73},
  number  = {5},
  pages   = {S326--S332},
  year    = {2012},
  doi     = {10.1097/TA.0b013e31827559b8},
  note    = {doi:10.1097/TA.0b013e31827559b8},
}

% Added by A. Schehr's verification pass. Verified against PubMed 3616751: 27 dimensions
% from T2, T7, T12 and L1-L5 across 30 skeletons; the abstract states the caudal trend in
% body height and canal diameter that the manuscript cites it for.
@article{berry1987,
  author  = {Berry, J. L. and Moran, J. M. and Berg, W. S. and Steffee, A. D.},
  title   = {A morphometric study of human lumbar and selected thoracic vertebrae},
  journal = {Spine (Phila. Pa. 1976)},
  volume  = {12},
  number  = {4},
  pages   = {362--367},
  year    = {1987},
  doi     = {10.1097/00007632-198705000-00010},
  note    = {doi:10.1097/00007632-198705000-00010},
}
"""
if "sixta2012" not in bib:
    (MPDA / "ctspinopelvic1k.bib").write_text(bib.rstrip("\n") + "\n" + ADD, encoding="utf-8")
    print("bib: added sixta2012 and berry1987")

# ---------------------------------------------------------------- manuscript
p = MPDA / "main.tex"
s = p.read_text(encoding="utf-8")

# Table II, the two wrong reference values
s = sub_once(s, r"Pelvic incidence & 54.7\si{\degree} & 54.7 $\pm$ 10.6 \\",
             r"Pelvic incidence & 54.7\si{\degree} & 55 $\pm$ 10.6 \\")
s = sub_once(s, r"Lumbar lordosis (supine) & 52.7\si{\degree} & 43 $\pm$ 11.2 (standing) \\",
             r"Lumbar lordosis & 52.7\si{\degree} & 60 $\pm$ 10 \\")

# say what the corrected comparison shows
s = sub_once(
    s,
    "Pelvic incidence is the strongest of these checks because it is a morphological property of the pelvis rather than a posture, and so is directly comparable to a standing reference cohort.",
    "Pelvic incidence is the strongest of these checks because it is a morphological property\nof the pelvis rather than a posture, and so is directly comparable to a standing reference\ncohort; it agrees to within a degree. Lumbar lordosis sits below the standing reference, the\ndirection recumbency predicts.",
)

# the trauma claim gains the peer-reviewed guideline
s = sub_once(s, "whole-spine coverage is reserved for trauma with an identified injury~\\cite{tqip2018}",
             "whole-spine coverage is reserved for trauma with an identified\ninjury~\\cite{sixta2012,tqip2018}")

# differ-by-level gains a primary source that states the trend outright
s = sub_once(s, "endplate and canal dimensions differ by level,\\cite{panjabi1991,panjabi1992,benzel2015}",
             "endplate and canal dimensions differ by level,\\cite{panjabi1991,panjabi1992,berry1987}")

# and so does the reference-morphometry sentence
s = sub_once(
    s,
    "come from twelve cadaveric specimens, reported as means with the standard error of the mean\\cite{panjabi1991,panjabi1992} and redrawn in the textbooks as one curve per level.\\cite{benzel2015} Neither shows whether the patient in front of the reader is unusual: a standard error describes the mean, not the spread of individuals.",
    "come from cadaveric series of twelve to thirty\nspecimens\\cite{panjabi1991,panjabi1992,berry1987} and are redrawn in the textbooks as one\ncurve per level.\\cite{benzel2015} A standard error, where one is reported, describes the\nmean and not the spread of individuals, so neither tells the reader whether the patient in\nfront of them is unusual.",
)

p.write_text(s, encoding="utf-8")
print("main.tex: table II corrected, sixta2012 and berry1987 cited")
