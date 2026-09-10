r"""Second tightening pass. Matching is whitespace-insensitive, so the source's line
wrapping does not matter; each pattern must still match exactly once.
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
    # the forward pointer is redundant now that the limitations state it outright
    ("counting- and rib-based rules.\\cite{schinz2026} The lumbosacral junction is not treated this way here, for the reason in Section~\\ref{sec:limitations}.",
     "counting- and rib-based rules.\\cite{schinz2026}"),

    # the comparator breakdown repeats Table 1, and the field of view is stated twice
    ("To our knowledge it is the largest spine-and-pelvis annotated CT cohort acquired under a single protocol; the comparators pool several collections (CTSpine1K four, CTPelvic1K seven), are multi-site by design (VerSe), or are routine clinical scans under many protocols (TotalSegmentator). With the protocol held fixed, scanner effects on a model become measurable rather than confounded. The price is an abdominal field of view, in which only the lowest thoracic levels are present and C2 never is (Sec.~\\ref{sec:applications}).",
     "To our knowledge it is the largest spine-and-pelvis annotated CT cohort acquired under a\nsingle protocol, where the comparators pool collections, are multi-site by design, or are\nroutine clinical scans under many protocols (Table~\\ref{tab:priorart}). With the protocol\nheld fixed, scanner effects on a model become measurable rather than confounded. The price\nis an abdominal field of view, in which only the lowest thoracic levels are present\n(Sec.~\\ref{sec:applications})."),

    # the reclassification plan is the future-directions item, stated there
    ("No record here carries a T13; the sixteen lumbar-rib cases were read as lumbar-type bodies. Where a border vertebra could not be settled by its readers the label stands as read, and reclassifying those records with a classifier trained on local vertebral morphology (Sec.~\\ref{sec:applications}) is the intended path for a later release.",
     "No record here carries a T13; the sixteen lumbar-rib cases were read as lumbar-type\nbodies, and where a border vertebra could not be settled by its readers the label stands as\nread."),

    # the two ways are announced and then each is announced again
    ("and they disturb identification in two ways at once. They change what the border vertebra and its ribs \\emph{look like}. A thoracolumbar border vertebra may carry",
     "and they disturb identification in two ways at once. They change what the border vertebra\nand its ribs \\emph{look like}: a thoracolumbar border vertebra may carry"),

    # the closing flourish restates the paragraph's opening
    ("on the same patients, and no one joined them, although the join needs no new imaging and no new radiologist. It is not a file merge:",
     "on the same patients, and no one joined them, although the join needs no new imaging and no\nnew radiologist. It is not a file merge:"),

    ("Two of the most-used public bone annotation sets were halves of one dataset; this release is the whole.",
     "Two of the most-used public bone annotation sets were halves of one dataset; this release\nis the whole."),

    # the reason the rule is sound is self-evident from the rule
    ("since a single bone carrying two numbers is a numbering error by construction, whatever the anatomy. The rule selected 152 records;",
     "since a single bone carrying two numbers is a numbering error whatever the anatomy. The\nrule selected 152 records;"),

    # the denominator argument can be made in one clause
    ("The denominator is stated because quoting the two offsets against all 11{,}548 ribs would halve the rate by counting ribs the check never examined. Both offsets are field-of-view truncations on individually characterized cases, one at $+1$ level and one at $-1$, reported rather than suppressed.",
     "The denominator is stated because quoting the two offsets against all 11{,}548 ribs would\nhalve the rate by counting ribs the check never examined. Both offsets are field-of-view\ntruncations, one at $+1$ level and one at $-1$."),

    # the classical-figure parenthesis can carry its own weight
    ("The same measures are given here from 802 records with the distribution attached (Fig.~\\ref{fig:levelatlas}), reproducing two properties of the classical figures (body height crosses over, dorsal exceeding ventral at T11--T12 and reversing by L4--L5, and pedicle width widens steeply into the lower lumbar spine) and adding the width of each distribution, which is the part needed to judge an individual.",
     "The same measures are given here from 802 records with the distribution attached\n(Fig.~\\ref{fig:levelatlas}), reproducing two properties of the classical figures, body\nheight crossing over at T11--T12 and reversing by L4--L5 and pedicle width widening steeply\ninto the lower lumbar spine, and adding the width of each distribution, which is the part\nneeded to judge an individual."),

    # the positional argument is made twice, once in general and once by example
    ("so that the measurement is positional rather than nominal, a vertebra by how many rib-free vertebrae separate it from the sacrum, a lowest rib by its length as a fraction of the rib above it, a transverse process by its span and its distance from the ala. None of these changes with the name a reader gives the junction, so cases that disagree about the name remain comparable.",
     "so that the measurement is positional rather than nominal: a vertebra by how many rib-free\nvertebrae separate it from the sacrum, a lowest rib by its length as a fraction of the rib\nabove it, a transverse process by its span and its distance from the ala. None of these\nchanges with the name a reader gives the junction, so cases that disagree about the name\nremain comparable."),

    # the rotation figures make the point; the sentence introducing them need not
    ("so as to remove the patient's rotation within the scanner. That rotation is not negligible, with a median of 4.5\\textdegree{}, a maximum of 16.1\\textdegree{}, and 508 of 801 records above 3\\textdegree{}.",
     "so as to remove the patient's rotation within the scanner, which is not negligible: a median\nof 4.5\\textdegree{}, a maximum of 16.1\\textdegree{}, and 508 of 801 records above\n3\\textdegree{}."),

    # the safe direction and the unsafe one need not each be justified twice
    ("Pelvis onto spine-only records is the safe direction, since the sacrum and innominate bones carry no enumeration to get wrong, and quality is reported as held-out performance on the \\emph{pelvic\\_only} records, withheld from training for that purpose. The reverse direction carries the whole problem this dataset addresses, since a pseudolabeled spine must commit to a count and on a transitional vertebra commits to the commoner reading; those 20 records were corrected by hand, tractable at that scale.",
     "Pelvis onto spine-only records is the safe direction, since the sacrum and innominate bones\ncarry no enumeration to get wrong, and quality is reported as held-out performance on the\n\\emph{pelvic\\_only} records, withheld from training for that purpose. The reverse direction\ncarries the whole problem this dataset addresses, because a pseudolabeled spine must commit\nto a count, and on a transitional vertebra it commits to the commoner reading; those 20\nrecords were corrected by hand."),
]

for old, new in edits:
    s = sub_once(s, old, new)

p.write_text(s, encoding="utf-8")
print(f"applied {len(edits)} edits; {before} -> {len(s)} chars ({before - len(s)} saved)")
