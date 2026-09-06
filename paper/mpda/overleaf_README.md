# CTSpinoPelvic1K — Medical Physics dataset article

Upload this whole folder to Overleaf (New Project -> Upload Project, or drag the zip in).

## First: set the main document

Overleaf does not always pick it automatically, and when it does not you get
`Emergency stop / job aborted, file error in nonstop mode` with a log that used almost no
memory — which looks alarming and means only that it tried to compile the wrong file.

**Menu -> Main document -> `main.tex`**, then recompile. Or open `main.tex` in the editor
and click *Recompile* from there.

## Compiling

- Main document: `main.tex`
- Compiler: **pdfLaTeX**
- Two passes are needed for cross-references. Overleaf does this automatically; if a
  reference shows as `??`, recompile once.
- No bibtex/biber pass. The bibliography is an inline `thebibliography` environment, so
  everything is in `main.tex`.

## The page limit is real, and it is not the number Overleaf shows you

Medical Physics allows **ten published pages**. `main.tex` is set to `preprint`, which is
single column and double spaced, so Overleaf will show roughly twice that. To check the real
count, change one word on line 16:

    \documentclass[aapm,mph,amsmath,amssymb,preprint]{revtex4-2}
    \documentclass[aapm,mph,amsmath,amssymb,reprint]{revtex4-2}

and recompile. It is currently **10 pages** in `reprint`. Please change it back to
`preprint` before committing, and if you add text, check the reprint count — the article is
at the limit, so anything added has to be paid for.

## Figures

`figures/` holds four built PDFs. They are generated from the released measurements and the
label volumes, neither of which is in this project, so edit the captions here and ask for a
regenerated figure if the plot itself needs to change.

| file | what it is |
|---|---|
| `fig_priorart.pdf` | the comparison against other public collections |
| `fig_anchors.pdf` | the two anchors and the interval between them, rendered from the labels |
| `fig_countfree.pdf` | count-free measures |
| `fig_validation.pdf` | derived measures against published reference ranges |
| `fig_opportunistic.pdf` | opportunistic screening measures |

Figure 1 is drawn in TikZ inside `main.tex` and needs no file.

## Before submission

Grep the source for `[CO-AUTHOR]` and `PENDING`. Outstanding at the time of writing:

- the Zenodo DOI is a placeholder (`\datasetdoi` on line 27)
- every author must confirm their own affiliation and supply a conflict-of-interest
  statement; the current declaration covers all authors and only the first has been asked
- the IRB determination is marked PENDING in the Ethics section and must not be submitted
  as written
- the repository URL in Data Availability
- twelve of seventeen references still have unverified volume/page fields; the five that
  were checked are named in a comment above the bibliography
