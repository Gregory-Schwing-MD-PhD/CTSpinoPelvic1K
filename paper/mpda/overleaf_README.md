# CTSpinoPelvic1K — Medical Physics Dataset and Software article

Upload this whole zip to Overleaf (New Project -> Upload Project). It is built by
`paper/mpda/make_submission.py`; edit the sources in the repository, not only here.

## First: set the main document

Overleaf does not always pick it automatically, and when it does not you get
`Emergency stop / job aborted, file error in nonstop mode` with a log that used almost no
memory — which looks alarming and means only that it tried to compile the wrong file.

**Menu -> Main document -> `main.tex`**, then recompile.

## What compiles

| file | what it is |
|---|---|
| `main.tex` | the article, as submitted: review format, anonymised, no abstract |
| `supplementary.tex` | the supporting information (inputs `supplement_body.tex`, `census_table.tex`) |
| `title_page.tex` | the separate title page: authors, affiliations, declarations |

Compiler **pdfLaTeX**, two passes (Overleaf does this; if a reference shows `??`, recompile).
No bibtex/biber: each bibliography is an inline `thebibliography`.

## Four switches at the top of `main.tex`

| switch | as shipped | meaning |
|---|---|---|
| `\reviewtrue` | on | redacts authors, the consortium name and the repository URLs |
| `\captionlisttrue` | on | lists the figure captions again after the references, as the form asks |
| `\supplementfalse` | off | appends the supplement (on only for arXiv) |
| `\abstractfalse` | off | the abstract is entered in the submission form, not the main document |

The abstract stays in `main.tex` behind `\ifabstract` because it is the source of the
form's text. Do not switch it on for the journal: the submission was returned once for
carrying one.

## The page limit is ten PUBLISHED pages, not what Overleaf shows

The shipped form is single column and double spaced, so it runs to about 25 pages. To see
the published count, change all four of these and recompile:

    preprint,linenumbers]{revtex4-2}  ->  reprint]{revtex4-2}     (line 22)
    \reviewtrue                       ->  \reviewfalse            (real authors add a page)
    \captionlisttrue                  ->  \captionlistfalse       (captions print once)
    \abstractfalse                    ->  \abstracttrue           (the published article prints it)

That is what `build.sh --reprint` does. It is **10 pages with about six body lines to
spare**; anything added has to be paid for. Change all four back before committing.

## Figures

Figure 1 is drawn in TikZ inside `main.tex`. The rest are built from the released
measurements and label volumes, which are not in this project, so edit captions here and
ask for a regenerated figure if a plot must change. Captions stay at 60 words or fewer.

| file | where |
|---|---|
| `figures/fig_anchors.pdf` | Fig. 2, four phenotypes at the lumbar borders |
| `figures/fig_levelatlas.pdf` | Fig. 3, morphometry by level |
| `figures/fig_hardware.pdf` | Fig. S1, instrumentation gallery |
| `figures/fig_validation.pdf` | Fig. S2, spinopelvic measures against standing references |
| `figures/fig_fov.pdf` | Fig. S3, records carrying each vertebral level |

## Still open

- `[CO-AUTHOR]` in `main.tex`: every author must supply their own conflict-of-interest
  statement; the current declaration covers all of them and only the first author has been
  asked.
