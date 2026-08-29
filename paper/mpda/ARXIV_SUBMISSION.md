# arXiv submission — CTSpinoPelvic1K

Everything the arXiv form asks for, so it can be pasted rather than reconstructed.

**Upload:** `paper/mpda/CTSpinoPelvic1K_arxiv.tar.gz` (464 KB — `main.tex` and five figure
PDFs, nothing else). It was compiled in a clean temporary tree with no access to this
machine, which is the only test that predicts what arXiv will do: 0 undefined references, no
errors. The bibliography is a `thebibliography` environment inside `main.tex` rather than
BibTeX, so there is no `.bbl` to forget — the commonest cause of an arXiv build failure, and
avoided here by construction.

Submit the **preprint** form (the default `documentclass` in the file). The `reprint` option
exists only to measure the published page count against the journal's ten-page limit and is
not what a preprint should look like.

---

## Title

CTSpinoPelvic1K: a fused spine and pelvis CT segmentation dataset built for lumbosacral
transitional anatomy

## Authors, in order

Gregory Schwing, Ashley Schehr, Annika Tekumulla, Margret Khoushi, Ryan Christian,
Dane Hubers, Faris Mahjoub, Hassan Saad, Mia Sooch, Sathya Siddapureddy, Michael McLellan,
Jerick Kim, Miraziz Ismoilov, Nizar Alnabahneh

Affiliations as set in the manuscript: Schwing is Department of Surgery, Detroit Medical
Center / Wayne State University; the eleven annotators are Wayne State University School of
Medicine; Ismoilov and Alnabahneh are Department of Radiology, Detroit Medical Center /
Wayne State University.

## Categories

- **Primary: `physics.med-ph`** — Medical Physics. This is a Medical Physics Dataset Article
  and that is the matching archive.
- Cross-list `eess.IV` — Image and Video Processing, where segmentation datasets are read.
- Cross-list `cs.CV` — the benchmark audience for the level-identification question.

## Comments field

    10 pages, 5 figures, 2 tables. Dataset archived at https://doi.org/10.5281/zenodo.22139643
    (802 annotated CT records, CC BY-NC-SA 4.0). Submitted to Medical Physics as a Medical
    Physics Dataset Article.

## Licence

**CC BY-NC-SA 4.0**, to match the dataset. ShareAlike is inherited from CTSpine1K, from which
the vertebral annotations derive, and is not ours to drop; keeping the preprint under the same
terms avoids a preprint more permissive than the data it describes.

## Journal reference and DOI

Leave both blank. arXiv lets you add a journal reference or DOI later **without posting a new
version** — "no new article version will be generated when journal reference, DOI or report
number information is added" — so this is filled in on acceptance rather than guessed now.

---

## Before you click submit

- **The dataset DOI resolves.** `https://doi.org/10.5281/zenodo.22139643` returns HTTP 200 and
  lands on the published record. The manuscript cites it in three places, all through the
  `\datasetdoi` macro, so they cannot drift apart.
- **The ethics statement says submitted, not obtained.** A determination has gone to the WSU
  IRB and no letter has been issued. When it arrives, change "has been submitted to" to "was
  issued by" and add the number — that is an arXiv v2, or simply a correction before the
  journal submission if the letter comes first.
- **Conflict-of-interest declarations.** The statement covers all fourteen authors on the
  corresponding author's word. Collect the rest before the journal submission; arXiv does not
  ask.

## What changed to reach ten pages

Recorded so it can be judged or reverted rather than discovered later.

| change | pages | content lost |
|---|---|---|
| `table*[b]` → `[t]` | 12 → 11 | none — a double-column float cannot be bottom-placed, so LaTeX deferred it past the bibliography onto its own page |
| `fig:hardware` full-width → single column | 13 → 12 | none — it is 794×559, and at the two-column measure it rendered about five inches tall |
| captions trimmed on three floats | — | none — each restated the body paragraph beside it |
| applications condensed to one subsection | — | argument kept, restatement dropped |
| `fig:opportunistic` removed | — | one figure; its claim remains in the text |
| `tab:castellvi` removed | 11 → 10 | the cross-tabulation; its finding is stated in the body and the full table is reproducible from the released manifest |
| four uncited references removed | — | none — they were numbered and attached to no claim |

Two citations were *restored* in the same pass: `pickhardt2013` and `ribsegv2` had lost their
citations when text around them was trimmed, while the claims they support remained.
