r"""Final pass: collapse the six identical rows of the metadata table into one, and remove
the last of the repeated wording. Whitespace-insensitive matching; each pattern must match
exactly once.
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
    # six rows carrying the identical value collapse into one
    ("""Identifier, image path, label path, configuration, match type & 802 & 100.0\\% \\\\
Patient position, tube potential, section thickness, kernel & 802 & 100.0\\% \\\\
Transitional label and class & 802 & 100.0\\% \\\\
Spine and pelvic annotation origin & 802 & 100.0\\% \\\\
Instrumentation and partial-annotation flags & 802 & 100.0\\% \\\\
Image/label alignment check & 802 & 100.0\\% \\\\""",
     """Identifiers and paths, configuration and match type; patient position, tube potential,
section thickness and kernel; transitional label and class; spine and pelvic annotation
origin; instrumentation and partial-annotation flags; image/label alignment check
& 802 & 100.0\\% \\\\"""),

    # the sentence after the table repeats the caption
    ("""Table~\\ref{tab:completeness} lists the key data types and metadata against the fraction of
records for which each is populated.

""", ""),

    # the comparison with whole-spine CT is made twice in the same paragraph
    ("marks a stump twelfth rib in 98 of the 788 records with a measurable pair (12.4\\%), against 12.6\\% on whole-spine CT~\\cite{nagata2025}; 16 records carry a lumbar rib (2.0\\%, against 1.5\\%). The co-occurrence reported there, stump ribs with sacralization and lumbar ribs with lumbarization, is testable here only where the junction was read;",
     "marks a stump twelfth rib in 98 of the 788 records with a measurable pair (12.4\\%, against\n12.6\\% on whole-spine CT~\\cite{nagata2025}); 16 records carry a lumbar rib (2.0\\%, against\n1.5\\%). The co-occurrence reported there is testable here only where the junction was read;"),

    # the same point about size opens and closes the sentence
    ("Existing public collections cannot address this, and size is not the reason (Table~\\ref{tab:priorart}). Spine datasets omit the pelvis; pelvic datasets do not number the vertebrae; TotalSegmentator, the whole-body scheme, has no class for a sixth lumbar vertebra or for a rib on a lumbar vertebra; and VerSe, the one collection with an L6 class, stops at the sacrum and carries no ribs.",
     "Existing public collections cannot address this, and size is not the reason\n(Table~\\ref{tab:priorart}): spine datasets omit the pelvis, pelvic datasets do not number\nthe vertebrae, TotalSegmentator has no class for a sixth lumbar vertebra or for a rib on a\nlumbar vertebra, and VerSe, the one collection with an L6 class, stops at the sacrum and\ncarries no ribs."),

    # the reason for count-free measures is given in the limitations and again here
    ("Transitional labels come from two independent sources and are not uniformly adjudicated, which is why the measures reported here are count-free, and the Castellvi grades are a consensus of two radiology resident physicians.",
     "Transitional labels come from two independent sources and are not uniformly adjudicated,\nwhich is why the measures reported here are count-free; the Castellvi grades are a consensus\nof two radiology resident physicians."),
]

for old, new in edits:
    s = sub_once(s, old, new)

p.write_text(s, encoding="utf-8")
print(f"applied {len(edits)} edits; {before} -> {len(s)} chars ({before - len(s)} saved)")
