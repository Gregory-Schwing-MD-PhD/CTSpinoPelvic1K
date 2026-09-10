r"""The last few lines. Same rule: nothing but repeated wording comes out."""
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
    ("The cut preserves the sub-division volume of the preceding release, changing the orientation of the S1/S2 boundary and not how much sacrum is called S1; per-record geometry ships with the archive.",
     "The cut preserves the sub-division volume of the preceding release, changing the\norientation of the S1/S2 boundary and not how much sacrum is called S1; per-record geometry\nships with the archive."),

    ("Five rib-free vertebrae above the sacrum is typical; four and six are where transitional anatomy sits, and which of the two is present does not by itself determine what to call it.",
     "Five rib-free vertebrae above the sacrum is typical; four and six are where transitional\nanatomy sits, and which is present does not by itself determine what to call it."),

    ("This scheme can record both, VerSe's T13 (28) with its rib pair (46, 59) for a thoracic-type thirteenth vertebra, and 60/61 for a lumbar-type body bearing a rudimentary rib, so either reading is recorded as itself.",
     "This scheme records both: VerSe's T13 (28) with its rib pair (46, 59) for a thoracic-type\nthirteenth vertebra, and 60/61 for a lumbar-type body bearing a rudimentary rib."),

    ("Vertebral labels derive from radiologist-supervised source annotations, corrected where those sources were wrong; pelvic labels on records that lacked one are pseudolabeled. The rib layer is a pseudolabel whose review was triaged rather than exhaustive, so its guarantee is the absence of the failure modes that rule detects.",
     "Vertebral labels derive from radiologist-supervised source annotations, corrected where\nthose sources were wrong; pelvic labels on records that lacked one are pseudolabeled. The\nrib layer is a pseudolabel whose review was triaged, so its guarantee is the absence of the\nfailure modes that rule detects."),

    ("Its weights label L6 and T13 on CT, but its 1{,}536 training scans are in-house and unreleased, so the capability exists as weights, not as data.",
     "Its weights label L6 and T13 on CT, but its training scans are in-house and unreleased, so\nthe capability exists as weights, not as data."),
]

for old, new in edits:
    s = sub_once(s, old, new)

p.write_text(s, encoding="utf-8")
print(f"applied {len(edits)} edits; {before} -> {len(s)} chars ({before - len(s)} saved)")
