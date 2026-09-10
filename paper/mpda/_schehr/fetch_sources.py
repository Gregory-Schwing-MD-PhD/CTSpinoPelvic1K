"""Collect, for every reference in the manuscript, the sentences that cite it and the
source's own abstract, so each claim can be checked against the source's words.

Writes _schehr/sources.json:
    {key: {"doi":..., "contexts":[manuscript sentences], "abstract": "...", "where": "..."}}

Abstracts come from PubMed (searched by DOI), then arXiv, then the Crossref abstract
field. Documents with no such record (the ACR parameters, the TQIP guideline, the Benzel
textbook) are left empty and checked by hand against the PDFs.
"""
import json
import re
import time
import urllib.parse
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
MPDA = HERE.parent
UA = {"User-Agent": "citation-check/1.0 (mailto:gregory.schwing@med.wayne.edu)"}


def get(url, timeout=45):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout).read().decode("utf-8", "ignore")


# ---- 1. citation contexts from the manuscript ------------------------------------------
src = (MPDA / "main.tex").read_text(encoding="utf-8")
body = src[src.index(r"\section{Introduction}"):src.index(r"\begin{thebibliography}")]
body = re.sub(r"%.*", "", body)
flat = re.sub(r"\s+", " ", body)

contexts = {}
for m in re.finditer(r"\\cite\{([^}]*)\}", flat):
    start = flat.rfind(".", 0, max(0, m.start() - 1))
    end = flat.find(".", m.end())
    sent = flat[start + 1: end + 1].strip()
    sent = re.sub(r"\\(emph|textbf|texttt)\{([^{}]*)\}", r"\2", sent)
    for k in (k.strip() for k in m.group(1).split(",")):
        contexts.setdefault(k, [])
        if sent not in contexts[k]:
            contexts[k].append(sent)

# ---- 2. DOIs from the bib --------------------------------------------------------------
bib = (MPDA / "ctspinopelvic1k.bib").read_text(encoding="utf-8")
dois = {}
for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S):
    d = re.search(r"\bdoi\s*=\s*\{([^}]*)\}", m.group(2))
    dois[m.group(1)] = d.group(1) if d else ""

out = {}
for key, doi in dois.items():
    rec = {"doi": doi, "contexts": contexts.get(key, []), "abstract": "", "where": ""}
    try:
        if "arxiv" in doi.lower():
            aid = re.search(r"(\d{4}\.\d{4,5})", doi).group(1)
            xml = get(f"http://export.arxiv.org/api/query?id_list={aid}")
            a = re.search(r"<summary>(.*?)</summary>", xml, re.S)
            if a:
                rec["abstract"] = re.sub(r"\s+", " ", a.group(1)).strip()
                rec["where"] = f"arXiv:{aid}"
        elif doi:
            q = urllib.parse.quote(f"{doi}[doi]")
            js = json.loads(get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term={q}&retmode=json"))
            ids = js["esearchresult"]["idlist"]
            if ids:
                txt = get(f"https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id={ids[0]}&rettype=abstract&retmode=text")
                rec["abstract"] = re.sub(r"\n(?!\n)", " ", txt).strip()
                rec["where"] = f"PubMed {ids[0]}"
            else:
                js = json.loads(get(f"https://api.crossref.org/works/{urllib.parse.quote(doi)}"))
                ab = js["message"].get("abstract", "")
                if ab:
                    rec["abstract"] = re.sub(r"<[^>]+>", " ", ab)
                    rec["abstract"] = re.sub(r"\s+", " ", rec["abstract"]).strip()
                    rec["where"] = "Crossref abstract"
    except Exception as e:
        rec["where"] = f"ERROR {e}"
    out[key] = rec
    print(f"{key:18s} {len(rec['abstract']):5d} chars  {rec['where']}")
    time.sleep(0.35)

(HERE / "sources.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
print("\nwrote sources.json;", sum(1 for r in out.values() if r["abstract"]), "of", len(out), "have text")
