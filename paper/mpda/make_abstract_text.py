"""The abstract as submittable plain text, from the SOURCE.

Not from the PDF: line breaking makes a real hyphen and a hyphenation hyphen look
identical, so de-hyphenating turned "LSTV-stratified" into "LSTVstratified" and
"triaged-review" into "triagedreview", and it split the DOI across a line. The source has
no line breaking to undo; only the handful of macros need resolving, and they are listed
here so a change to one is visible.
"""
import io
import os
import re
import sys

# relative to this file: the packet build runs it from inside WSL
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
TEX = os.path.join(HERE, "main.tex")
s = io.open(TEX, encoding="utf-8").read()

MACROS = {
    r"\datasetdoi": "10.5281/zenodo.22139642",
    r"\datasetversiondoi": "10.5281/zenodo.22797014",
}

a = s.index(r"\begin{abstract}") + len(r"\begin{abstract}")
b = s.index(r"\end{abstract}")
t = s[a:b]

t = re.sub(r"(?m)^\s*%.*$", "", t)
t = re.sub(r"\\doiurl\{\\datasetdoi\}", "https://doi.org/" + MACROS[r"\datasetdoi"], t)
t = re.sub(r"\\doiurl\{\\datasetversiondoi\}", "https://doi.org/" + MACROS[r"\datasetversiondoi"], t)
t = re.sub(r"\\textbf\{([^{}]*)\}", r"\1", t)
t = re.sub(r"\\emph\{([^{}]*)\}", r"\1", t)
t = re.sub(r"\\texttt\{([^{}]*)\}", r"\1", t)
t = t.replace("~", " ").replace("\\%", "%").replace("\\&", "&")
t = t.replace("---", "\u2014").replace("--", "\u2013")
t = re.sub(r"\\[a-zA-Z@]+\*?", "", t)
t = t.replace("{", "").replace("}", "")
t = re.sub(r"\s*\n\s*", " ", t)
t = re.sub(r"\s{2,}", " ", t).strip()

for h in ("Acquisition and Validation Methods:", "Data Format and Usage Notes:",
          "Potential Applications:"):
    t = t.replace(h, "\n\n" + h)
t = t.replace(" Limitations:", "\nLimitations:")

leftovers = re.findall(r"\\[a-zA-Z]+|[{}]", t)
if leftovers:
    print("!! unresolved LaTeX:", set(leftovers), file=sys.stderr)

print(t)
n = len([w for w in re.split(r"\s+", t) if re.search(r"[A-Za-z0-9]", w)])
print("\n" + "-" * 72)
print("%d words (limit 300)%s" % (n, "  -- OVER by %d" % (n - 300) if n > 300 else "  ok"))

OUTF = os.path.join(ROOT, "dist", "submission", "Abstract.txt")
os.makedirs(os.path.dirname(OUTF), exist_ok=True)
io.open(OUTF, "w", encoding="utf-8", newline="\n").write(t + "\n")
print("written to dist/submission/Abstract.txt")
