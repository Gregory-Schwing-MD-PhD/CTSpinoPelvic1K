r"""Apply what the manually retrieved PDFs settle.

  * Konin and Walz state the prevalence range verbatim: "LSTVs are common in the general
    population, with a reported prevalence of 4%-30%." It was removed for want of a
    readable source; restore it, cited to them.
  * VERIDAH's Table 1 reads "CT Cohort ... Number of Scans 1536 | Train 1171 | Test 365",
    so 1,536 is the whole in-house CT cohort, not the training split. The manuscript called
    it "1,536 training scans", which overstates the training set by 365. Correct it.
"""
import re
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "main.tex"
s = p.read_text(encoding="utf-8")


def sub_once(text, old, new):
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    hits = pat.findall(text)
    assert len(hits) == 1, f"expected one match, found {len(hits)}:\n{old[:120]}"
    return pat.sub(lambda _m: new, text, count=1)


s = sub_once(
    s,
    "LSTV is common, reported in 16.3\\% of a whole-spine CT series~\\cite{nagata2025}, and wrong-level surgery is its most serious consequence.",
    "LSTV is common, reported at 4--30\\% in the general\npopulation~\\cite{koninwalz2010} and in 16.3\\% of a consecutive whole-spine CT\nseries,~\\cite{nagata2025} and wrong-level surgery is its most serious consequence.",
)

s = sub_once(
    s,
    "Its weights label L6 and T13 on CT, but its training scans are in-house and unreleased, so the capability exists as weights, not as data.",
    "Its weights label L6 and T13 on CT, but its 1{,}536-scan CT cohort is in-house and\nunreleased, so the capability exists as weights, not as data.",
)

p.write_text(s, encoding="utf-8")
print("applied 2 corrections from the retrieved PDFs")
