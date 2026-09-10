"""Pull supporting sentences from open-access full text and from the two guideline PDFs,
for the references whose PubMed record carries no abstract.

Adds a "fulltext_hits" list to each entry of sources.json: sentences from the source that
contain the terms the manuscript's claim turns on.
"""
import json
import re
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
UA = {"User-Agent": "citation-check/1.0 (mailto:gregory.schwing@med.wayne.edu)"}
S = json.loads((HERE / "sources.json").read_text(encoding="utf-8"))


def get(url, timeout=60):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read()


def sentences(text):
    text = re.sub(r"\s+", " ", text)
    return re.split(r"(?<=[.!?]) (?=[A-Z(])", text)


def pmc(pmcid):
    xml = get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pmc&id={pmcid}&retmode=xml").decode("utf-8", "ignore")
    xml = re.sub(r"<(ref-list|back|table-wrap|fig)\b.*?</\1>", " ", xml, flags=re.S)
    return re.sub(r"<[^>]+>", " ", xml)


WANT = {
    "duplessis2018": ("PMC5879990", ["thirteenth", "13th", "lumbar rib", "transitional", "classif"]),
    "poolman2023":   ("PMC10335368", ["thirteenth", "13th", "lumbar rib", "rudimentary", "numeric variation"]),
    "nagata2025":    (None, ["rib abnormalit", "hypoplastic", "sacralization", "lumbarization", "lumbar rib"]),
}

for key, (pmcid, terms) in WANT.items():
    hits = []
    try:
        if pmcid:
            txt = pmc(pmcid)
        else:
            txt = S[key]["abstract"]
        for s in sentences(txt):
            s = s.strip()
            if 60 < len(s) < 420 and any(t.lower() in s.lower() for t in terms):
                hits.append(s)
        S[key]["fulltext_hits"] = hits[:14]
        S[key]["where"] = (S[key]["where"] + f"; {pmcid} full text") if pmcid else S[key]["where"]
        print(f"{key}: {len(hits)} candidate sentences")
    except Exception as e:
        print(f"{key}: ERROR {e}")
    time.sleep(0.4)

# --- the two guideline PDFs and the trauma imaging guideline ---------------------------
PDFS = {
    "acrct2022":  ("https://www.asnr.org/wp-content/uploads/2019/06/CT-Spine-1.pdf",
                   ["lumbar spine from T12", "entire lumbar", "T12", "coverage"]),
    "acrmri2023": ("https://www.asnr.org/wp-content/uploads/2019/06/MR-Adult-Spine.pdf",
                   ["Lumbar spine:", "T12 to S1", "entire lumbar"]),
    "tqip2018":   ("https://www.facs.org/media/oxdjw5zj/imaging_guidelines.pdf",
                   ["whole spine", "entire spine", "thoracolumbar", "cervical spine CT", "screening"]),
}
try:
    import pymupdf
    for key, (url, terms) in PDFS.items():
        try:
            raw = get(url)
            f = HERE / f"{key}.pdf"
            f.write_bytes(raw)
            d = pymupdf.open(f)
            txt = " ".join(p.get_text() for p in d)
            hits = [s.strip() for s in sentences(txt)
                    if 40 < len(s) < 400 and any(t.lower() in s.lower() for t in terms)]
            S[key]["fulltext_hits"] = hits[:12]
            S[key]["where"] = f"{f.name}, {len(d)} pages"
            print(f"{key}: {len(hits)} candidate sentences from {len(d)}-page PDF")
        except Exception as e:
            print(f"{key}: ERROR {e}")
except ImportError:
    print("pymupdf missing; PDFs skipped")

(HERE / "sources.json").write_text(json.dumps(S, indent=1), encoding="utf-8")
print("updated sources.json")
