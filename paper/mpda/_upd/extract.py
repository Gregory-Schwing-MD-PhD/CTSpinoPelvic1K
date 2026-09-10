"""Pull every comment out of the student review docx, with the text it is anchored to."""
from pathlib import Path
from xml.etree import ElementTree as ET

W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
HERE = Path(__file__).resolve().parent


def text_of(el):
    out = []
    for e in el.iter():
        if e.tag == W + "t":
            out.append(e.text or "")
        elif e.tag == W + "delText":
            out.append("[-" + (e.text or "") + "-]")
        elif e.tag == W + "tab":
            out.append(" ")
    return "".join(out)


comments = {}
for c in ET.parse(HERE / "word/comments.xml").getroot().findall(W + "comment"):
    comments[c.get(W + "id")] = (
        c.get(W + "author"),
        " ".join(text_of(p) for p in c.findall(W + "p")).strip(),
    )

doc = ET.parse(HERE / "word/document.xml").getroot()
paras = list(doc.iter(W + "p"))
lines = []
for i, p in enumerate(paras):
    ids = [e.get(W + "id") for e in p.iter(W + "commentRangeStart")]
    if not ids:
        continue
    ptext = text_of(p)
    for cid in ids:
        author, body = comments.get(cid, ("?", "?"))
        anchor, on = [], False
        for e in p.iter():
            if e.tag == W + "commentRangeStart" and e.get(W + "id") == cid:
                on = True
            elif e.tag == W + "commentRangeEnd" and e.get(W + "id") == cid:
                on = False
            elif on and e.tag == W + "t":
                anchor.append(e.text or "")
        lines.append(f"\n### comment {cid} — {author}  (paragraph {i})")
        lines.append(f"ANCHOR: {''.join(anchor)[:300]}")
        lines.append(f"PARA:   {ptext[:400]}")
        lines.append(f"SAYS:   {body}")

(HERE / "comments.txt").write_text("\n".join(lines), encoding="utf-8")
(HERE / "fulltext.txt").write_text("\n".join(text_of(p) for p in paras), encoding="utf-8")
print(len(comments), "comments;", len(paras), "paragraphs")
