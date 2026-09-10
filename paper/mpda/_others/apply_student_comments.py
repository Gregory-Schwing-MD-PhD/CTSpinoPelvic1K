r"""Apply the student-review comments (CTSP_OTHERSTUDENTS.docx) that are corrections rather
than matters of taste or artefacts of the Word conversion.

Applied here:
  * an uncited prevalence figure gets a source (Jerick Kim, Ryan Christian)
  * TCIA is expanded at first use (Jerick Kim)
  * "the two available automatic sources" names them (Annika Tekumulla)
  * "never show what usual looks like" softened (Mia Sooch)
  * the left_hip sentence reworded (Mia Sooch)
  * a forward pointer to the figure that shows a transitional variant (Mia Sooch)
  * "single protocol" glossed in plain words (Mia Sooch)
  * the orphan sentence "Its limitations are stated..." given a subject (Maggie Khoushi)

Not applied, and why, is written up in STUDENT_REVIEW_RESPONSE.md.
"""
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "main.tex"
s = p.read_text(encoding="utf-8")

edits = [
    # Jerick Kim + Ryan Christian: "ref?" on the prevalence range
    (r"""LSTV is common,
reported between 4\% and 30\% depending on definition, and wrong-level surgery is its most
serious consequence.""",
     r"""LSTV is common, 16.3\%
in a consecutive whole-spine CT series~\cite{nagata2025} and 4--30\% across series
depending on definition,~\cite{koninwalz2010,lian2018} and wrong-level surgery is its most
serious consequence."""),

    # Mia Sooch: show the reader what a transitional variant looks like, early
    (r"""set yields the same count, and the morphology that would separate the readings is not""",
     r"""set yields the same count (Fig.~\ref{fig:anchors}), and the morphology that would
separate the readings is not"""),

    # Jerick Kim: define TCIA and the other acronyms at first use
    (r"""Both sources draw part of their imaging from the TCIA CT colonography
collection~\cite{colonog,tcia}.""",
     r"""Both sources draw part of their imaging from The Cancer Imaging Archive (TCIA) CT
colonography collection (COLONOG)~\cite{colonog,tcia}."""),

    # Mia Sooch: say what "protocol" means here
    (r"""COLONOG is the only collection common to
both and the only one acquired under a single protocol,""",
     r"""COLONOG is the only collection common to
both and the only one acquired under a single protocol, meaning one scan prescription and
patient preparation for every case,"""),

    # Annika Tekumulla: name the two automatic sources where they are first invoked
    (r"""No public rib annotation covers this cohort, and the two available automatic sources fail
in complementary ways.""",
     r"""No public rib annotation covers this cohort, and the two available automatic
sources~\cite{moller2026,totalsegmentator} fail in complementary ways."""),

    # Mia Sooch: reword the sided-structure sentence
    (r"""Testing the ribs alone passed all
802 records while four carried \texttt{left\_hip} on the patient's right and pooling all
sided structures into one test still missed one of those four, because a large
correctly-sided structure masks a smaller transposed one.""",
     r"""Testing the ribs alone passed all
802 records, four of which carried a \texttt{left\_hip} label on the patient's right side;
pooling every sided structure into one test still missed one of those four records, because
a large correctly-sided structure masks a smaller transposed one."""),

    # Mia Sooch: "never" is too strong for a point estimate
    (r"""They
cannot tell a reader whether the patient in front of them is unusual, because they never
show what usual looks like.""",
     r"""They
cannot tell a reader whether the patient in front of them is unusual, because a point
estimate does not show what usual looks like."""),

    # Maggie Khoushi: the sentence has no subject
    (r"""Its limitations are stated at the level of detail a reader would
need to catch them independently.""",
     r"""The release states its own limitations in enough detail that a reader
can find them without being told."""),
]

for old, new in edits:
    n = s.count(old)
    assert n == 1, f"expected one match, found {n}:\n{old[:110]}"
    s = s.replace(old, new)

p.write_text(s, encoding="utf-8")
print("applied", len(edits), "student-review edits")
