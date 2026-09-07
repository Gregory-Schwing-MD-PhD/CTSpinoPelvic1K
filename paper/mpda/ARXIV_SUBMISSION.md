# arXiv submission — CTSpinoPelvic1K (v10 manuscript, 7 September 2026)

Everything the arXiv form asks for, so it can be pasted rather than reconstructed.

**Upload:** `paper/mpda/CTSpinoPelvic1K_arxiv.tar.gz` (about 820 KB: `main.tex`,
`figure_captions.tex` and six figure PDFs, nothing else). `make_arxiv.sh` builds it and then
compiles it in a clean temporary tree, which is the only test that predicts what arXiv will do:
0 undefined references, no errors, 11 two-column pages. The bibliography is a `thebibliography`
environment inside `main.tex`, so there is no `.bbl` to forget. Figure 1 is drawn in TikZ inside
`main.tex`, so seven figures appear from six PDFs. The preview of exactly what arXiv will
render is `CTSpinoPelvic1K_arxiv_preview.pdf`.

**Form:** the two-column `reprint` form with authors and affiliations in the file and every
review redaction removed (`\reviewfalse`). The double-spaced, line-numbered, anonymized
`preprint` form is the journal's review copy and is not what goes to arXiv.

**How the site works now (2026):** on "Prepare Files" choose the tar.gz with the file picker
(drag-and-drop is disabled), click Upload, then Check Files. The processor auto-detects as
PDFLaTeX and the top-level file as `main.tex`; leave both. Accept, confirm any files it
suggests deleting (there should be none), preview the PDF, then Continue to the metadata page.
Submit before 14:00 Eastern on a weekday and it is announced at 20:00 the same day.

---

## Title

CTSpinoPelvic1K: spine, pelvis, ribs and femora in one coordinate frame, annotated for lumbosacral transitional anatomy

## Authors, in order

Gregory Schwing, Ashley Schehr, Annika Tekumulla, Margret Khoushi, Ryan Christian,
Dane Hubers, Faris Mahjoub, Hassan Saad, Mia Sooch, Sathyagopal Siddapureddy, Michael McLellan,
Jerick Kim, Miraziz Ismoilov, Nizar Alnabahneh

Affiliations as set in the manuscript: Schwing is Department of Surgery, Detroit Medical
Center and Wayne State University; the eleven annotators are School of Medicine, Wayne State
University; Ismoilov and Alnabahneh are Department of Radiology, Detroit Medical Center and
Wayne State University. All Detroit, Michigan, USA.

## Abstract

Paste `arxiv_abstract.txt` (the manuscript abstract, 299 words, TeX stripped). arXiv's
abstract field takes plain text; the four bold section labels are kept as words.

## Categories

- **Primary: `physics.med-ph`** (Medical Physics). It is a Medical Physics Dataset Article.
- Cross-list `eess.IV` (Image and Video Processing), where segmentation datasets are read.
- Cross-list `cs.CV`, the benchmark audience for the level-identification question.

## Comments field

    11 pages, 7 figures, 3 tables. Dataset (802 annotated CT records, CC BY-NC-SA 4.0) at
    https://doi.org/10.5281/zenodo.22139642; build archive at
    https://doi.org/10.5281/zenodo.22647933; code at
    https://github.com/OpenSpineConsortium/CTSpinoPelvic1K. Submitted to Medical Physics
    as a Medical Physics Dataset Article.

## Licence

**CC BY-NC-SA 4.0**, to match the dataset. ShareAlike is inherited from CTSpine1K, from which
the vertebral annotations derive, and is not ours to drop; keeping the preprint under the same
terms avoids a preprint more permissive than the data it describes. Medical Physics (Wiley)
permits posting the submitted manuscript on a preprint server.

## Journal reference and DOI

Leave both blank. arXiv adds a journal reference or DOI later without a new version, so this
is filled in on acceptance rather than guessed now.

---

## Before you click submit

- **Both DOIs resolve.** https://doi.org/10.5281/zenodo.22139642 (concept, resolves to v10)
  and https://doi.org/10.5281/zenodo.22642578 (v10 itself) return the published records; the
  manuscript cites them through the `\datasetdoi` and `\datasetversiondoi` macros, so the text
  cannot drift from the form.
- **Ethics.** The Wayne State IRB issued its not-human-participant-research determination on
  2 September 2026 and the manuscript states it; nothing is pending.
- **Authors.** Fourteen, matching the title page, the Zenodo creators and the sign-off sheet.
  Confirm every coauthor has returned the sign-off before posting; arXiv is public the same
  evening.
- **The preview.** Open `CTSpinoPelvic1K_arxiv_preview.pdf`: page 1 must show all fourteen
  names once each and no "[removed for double-anonymized review]" anywhere; page 11 is the
  end of the reference list.
