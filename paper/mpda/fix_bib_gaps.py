r"""fix_bib_gaps.py -- close three defects the bibtex conversion exposed.

1. \cite{nnunet} and \cite{ribseg} were never defined. They are not new breakage from the
   move to bibtex; the hand-written list never carried them either, so the manuscript has
   been silently printing "[?]" for the two tools that do the actual segmentation. Only
   running bibtex surfaced them.

     nnunet -- added, verified against Crossref (Nat Methods 2021;18(2):203-211).
     ribseg -- a stale second key for Moller's rib network, which line 473 already cites
               correctly as moller2026. Repointed rather than given a duplicate entry.

2. The Epstein DOI (10.25259/SNI_402_2021) contains underscores. bibtex passes them through
   verbatim, and an unescaped underscore is a subscript operator in text mode -- the source
   of the "Missing $ inserted" errors. Escaped in the .bib so the fix survives regeneration.
"""
from pathlib import Path

HERE = Path(__file__).parent

bib = HERE / "ctspinopelvic1k.bib"
s = bib.read_text(encoding="utf-8")

# 1. underscores in a DOI are subscripts in LaTeX text mode
assert s.count("SNI_402_2021") == 2, "expected the SNI doi in both doi and note fields"
s = s.replace("note    = {doi:10.25259/SNI_402_2021}",
              r"note    = {doi:10.25259/SNI\_402\_2021}")

# 2. nnU-Net, verified against Crossref
nnunet = r"""
% Verified against Crossref (10.1038/s41592-020-01008-z): Nat Methods 2021;18(2):203-211.
@article{nnunet,
  author  = {Isensee, Fabian and Jaeger, Paul F. and Kohl, Simon A. A. and
             Petersen, Jens and Maier-Hein, Klaus H.},
  title   = {{nnU-Net}: a self-configuring method for deep learning-based biomedical
             image segmentation},
  journal = {Nat. Methods},
  volume  = {18},
  number  = {2},
  pages   = {203--211},
  year    = {2021},
  doi     = {10.1038/s41592-020-01008-z},
  note    = {doi:10.1038/s41592-020-01008-z},
}

@article{totalsegmentator,"""
assert "@article{nnunet," not in s
s = s.replace("\n@article{totalsegmentator,", nnunet, 1)
bib.write_text(s, encoding="utf-8")

# 3. the stale duplicate key for Moller's network
main = HERE / "main.tex"
m = main.read_text(encoding="utf-8")
old = r"M\"oller's~\cite{ribseg} released weights"
assert m.count(old) == 1
m = m.replace(old, r"M\"oller's~\cite{moller2026} released weights")
main.write_text(m, encoding="utf-8")

print("bib: escaped the SNI underscores, added the verified nnunet entry")
print("main.tex: repointed the stale \cite{ribseg} at Moller's network to moller2026")
