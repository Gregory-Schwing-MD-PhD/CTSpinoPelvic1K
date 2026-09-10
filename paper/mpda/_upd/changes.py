"""Show every insertion and deletion in the updated-references docx, with its paragraph."""
import re
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
HERE = Path(__file__).resolve().parent


def runs(el):
    out = []
    for e in el.iter():
        if e.tag == W + "t":
            out.append(e.text or "")
        elif e.tag == W + "delText":
            out.append("")
    return "".join(out)


doc = ET.parse(HERE / "word/document.xml").getroot()
lines = []
for i, p in enumerate(doc.iter(W + "p")):
    ins = ["".join(t.text or "" for t in e.iter(W + "t")) for e in p.iter(W + "ins")]
    dels = ["".join(t.text or "" for t in e.iter(W + "delText")) for e in p.iter(W + "del")]
    ins = [x for x in ins if x.strip()]
    dels = [x for x in dels if x.strip()]
    if not ins and not dels:
        continue
    ctx = re.sub(r"\s+", " ", runs(p))[:230]
    lines.append(f"\n--- para {i}: {ctx}")
    if dels:
        lines.append(f"    DELETED : {' | '.join(dels)[:400]}")
    if ins:
        lines.append(f"    INSERTED: {' | '.join(ins)[:400]}")

(HERE / "changes.txt").write_text("\n".join(lines), encoding="utf-8")
print(f"{len(lines)} lines")
