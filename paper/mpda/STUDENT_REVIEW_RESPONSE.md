# Response to the student review (CTSP_OTHERSTUDENTS.docx)

31 comments from five reviewers: Maggie Khoushi (13), Mia Sooch (8), Jerick Kim (5), Annika
Tekumulla (3), Ryan Christian (2). The file carries comments only, no tracked changes.
Extraction script: `_others/extract.py`; full text with anchors in `_others/comments.txt`.

Nine comments produced edits (`_others/apply_student_comments.py` and the consolidation pass). Two were already
answered by A. Schehr's revision pass. Nine describe artefacts of the PDF-to-Word
conversion rather than defects in the manuscript. Six are matters of house style or float
placement that LaTeX and the journal decide. Four are judgement calls; the two biggest, consolidating the limitations and clarifying the pseudolabelling sentence, have since been carried out.

---

## Applied

**1. The prevalence figure had no citation.** Jerick Kim wrote "ref?" on "LSTV is common,
reported between 4% and 30% depending on definition", and Ryan Christian seconded it. They
were right: the sentence carried no reference. It now reads "LSTV is common, reported in
16.3% of a whole-spine CT series [nagata2025]". The 16.3% is quoted from Nagata's own results.
The 4–30% range that was in the sentence has been dropped rather than cited: it is
conventionally attributed to Konin and Walz, whose full text is paywalled, and no source
readable from here states it. See `CITATION_CHECK.md`.

**2. TCIA was never expanded.** Jerick Kim: "define. and other acronyms used". First use now
reads "The Cancer Imaging Archive (TCIA) CT colonography collection (COLONOG)".

**3. "The two available automatic sources" did not say which.** Annika Tekumulla asked
"source?". Both are now cited at that sentence rather than only in the two that follow.

**4. "Never show what usual looks like."** Mia Sooch: "strong statement. replace 'never'".
Changed to "a point estimate does not show what usual looks like", which is the literal
point and no longer an absolute.

**5. The sided-structure sentence was hard to parse.** Mia Sooch flagged "four carried left
hip" and "those four". Reworded so the subject of each clause is explicit.

**6. No picture of a transitional variant near the description.** Mia Sooch asked for one.
Figure 2 already shows exactly this; the ambiguity sentence now points at it, so the reader
meets the picture where the problem is described rather than four pages later.

**7. "Protocol" was unexplained.** Mia Sooch: "Dumb this down". Glossed in place as "one scan
prescription and patient preparation for every case".

**8. An orphan sentence.** Maggie Khoushi: "unclear what this means / is referring to" on
"Its limitations are stated at the level of detail a reader would need to catch them
independently." The pronoun had no antecedent. It was first rewritten with a subject and then
removed altogether in the limitations consolidation, since the limitations block it gestured
at now states them outright.

---

## Already answered by Ashley Schehr's pass

**9. Annika Tekumulla, on the 0.990 AUC: "what about for LSTV?"** — the same question
A. Schehr raised. The manuscript now states in the limitations why no lumbosacral classifier
is attempted: 15 records in each rare stratum leave three per fold.

**10. Annika Tekumulla, on reference 3: "MRI? Didn't focus on this in the paper."** — also
raised by A. Schehr. The ACR CT practice parameter is now cited for the CT claim, and the
MRI parameter only for the T12-to-S1 span it states in levels.

---

## Artefacts of the Word conversion, not defects

The review copy was made by converting the typeset PDF to Word. Several comments describe
damage introduced by that conversion:

- Maggie Khoushi: "this sentence has a random subheading and drop down menu linked to it"
  and "actually there are a lot of these randomly" — Word content controls created during
  conversion. None exist in the LaTeX source.
- Maggie Khoushi: "ll,i ll — what is happening here" — Figure 1 is drawn in TikZ, and
  extracting a vector drawing to Word leaves loose letters. The figure itself is intact.
- Maggie Khoushi: "a lot of random gaps / strange formatting gaps mid-sentence" — line-break
  hyphens and inter-word spacing from two-column justification. Thank her for fixing what
  she could; nothing needs fixing in the source.
- Ryan Christian: "Where is the start of this sentence?" — a table landed mid-paragraph in
  the conversion. In the typeset article floats sit at the top of a page.
- Maggie Khoushi's related note that "the table should not split the paragraph… the constant
  cut offs mid-paragraph are really disruptive" — same cause. Worth one look at the final
  PDF to confirm no float lands badly, but the Word copy is not evidence of it.
- Mia Sooch, twice: "move the figure to after this sentence" — float placement, which the
  class controls.
- Maggie Khoushi: "some of the figure labels are above and some are below" — the journal's
  convention, which the class enforces: table captions above, figure captions below. This is
  correct as it stands.
- Maggie Khoushi: "the image for figure 6 is above when it is actually discussed" — a figure
  may be placed before its first mention when it is a full-width float; the reference is what
  ties them together.

---

## Style, for you to settle

**11. Run-in emphasis.** Maggie Khoushi twice: "I don't think phrases in the body should be
bolded or italicized… need to pick one and fix all the rest." The manuscript uses italic
run-in headings (*How S1 is cut.*, *Versions and parameters.*) which is a standard device in
this journal and is used consistently: italic for run-in headings inside a section, bold only
for the structured-abstract labels. I would keep it, but it is one command to change if you
disagree.

**12. Voice.** Maggie Khoushi and Jerick Kim both note the prose switches between active and
passive and that the introduction's flow could be better. No specific sentence was named, so
nothing was changed. If you want this addressed, the introduction is the place to spend it.

---

## Judgement calls left for you

**13. Cut the demographics?** Jerick Kim: "is this necessary? … removing info like this may
help the manuscript be more concise", seconded by Maggie Khoushi. I did not cut it. A
dataset article is required to describe the cohort, and the sex and age breakdown with its
missing-field counts is exactly the descriptive material the article type asks for. It is
also the paragraph a reviewer would ask for if it were absent.

**14. Consolidate the limitations? — DONE.** Jerick Kim: "there are limitations mentioned
multiple times throughout the manuscript… consolidate into one at the end or keep as is",
seconded by Maggie Khoushi. They were right, and this has now been carried out at the
author's direction. The limitations paragraph in the Discussion repeated the coverage and
rib-triage points already made in Section V, so Section V now carries them all under one
heading (coverage and cohort; label strength; absent structures and the absence of a
lumbosacral classifier) and the Discussion's duplicate is gone. The same pass found the
prone/supine mismatch explained three times, in the Introduction and twice in the methods,
and the metadata table carrying six rows that all read "802, 100.0%". Both were consolidated.
That work took the article from 11 published pages to 10, the limit.

**15. "Clarify this sentence, hard to follow." — DONE.** Mia Sooch, on the pseudolabelling
direction. The sentence now reads "because a pseudolabeled spine must commit to a count, and
on a transitional vertebra it commits to the commoner reading," which separates the two
clauses she was reading as one.

**16. "Change the wording."** Mia Sooch, on "an annotation carries a patient identifier and
nothing finer". No specific problem named; the phrase is accurate. Left alone.
