r"""What the two Panjabi scans settle.

Panjabi 1991 (Thoracic Human Vertebrae, Spine 16(8):888-901), abstract:
    "based on a study of 144 vertebrae ... Means and standard errors of the means for
     linear, angular, and area dimensions of vertebral bodies, spinal canal, pedicle,
     pars articularis, spinous and transverse processes, and rib articulations are
     provided for all thoracic vertebrae."

Panjabi 1992 (Human Lumbar Vertebrae, Spine 17(3):299-306), abstract and methods:
    "based on a study of 60 vertebrae ... Means and standard errors of the means for
     linear, angular, and area dimensions of vertebral bodies, spinal canal, pedicle ...
     were obtained for all lumbar vertebrae."
    "Twelve fresh autopsy spine specimens (axis to last lumbar vertebra) were obtained"

Two consequences for the manuscript:

  * "differ by level" is directly supported: both papers report per-level dimensions of
     exactly the parts named, and each describes transitional behaviour at the ends.

  * the sentence about reference morphometry was loose in two ways. It said "a few dozen
    specimens" when the series is twelve, and it implied the primary sources carry no
    dispersion when in fact they report the standard error of the mean. What has no spread
    is the textbook redrawing, which is what a surgeon consults. Rewritten to say exactly
    that, which is a sharper argument anyway: a standard error describes the mean, not the
    variability of individuals.
"""
import re
from pathlib import Path

p = Path(__file__).resolve().parent.parent / "main.tex"
s = p.read_text(encoding="utf-8")


def sub_once(text, old, new):
    pat = re.compile(r"\s+".join(re.escape(w) for w in old.split()))
    hits = pat.findall(text)
    assert len(hits) == 1, f"expected one match, found {len(hits)}:\n{old[:130]}"
    return pat.sub(lambda _m: new, text, count=1)


s = sub_once(
    s,
    "The values a surgeon consults for vertebral body height, canal dimensions and pedicle width come from cadaveric series of a few dozen specimens, plotted as single values with no indication of spread.\\cite{benzel2015} They cannot tell a reader whether the patient in front of them is unusual, because a point estimate does not show what usual looks like.",
    "The values a surgeon consults for vertebral body height, canal dimensions and pedicle\nwidth come from twelve cadaveric specimens, reported as means with the standard error of\nthe mean\\cite{panjabi1991,panjabi1992} and redrawn in the textbooks as one curve per\nlevel.\\cite{benzel2015} Neither shows whether the patient in front of the reader is\nunusual: a standard error describes the mean, not the spread of individuals.",
)

p.write_text(s, encoding="utf-8")
print("applied the Panjabi correction")
