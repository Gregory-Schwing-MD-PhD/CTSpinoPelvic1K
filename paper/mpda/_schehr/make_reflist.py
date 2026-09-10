"""Build a download list for every reference in the manuscript: resolver link, and whether
a free full text exists (PubMed Central / arXiv), using what was already fetched into
sources.json by the citation check."""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
bib = (HERE.parent / "ctspinopelvic1k.bib").read_text(encoding="utf-8")
src = json.loads((HERE / "sources.json").read_text(encoding="utf-8")) if (HERE / "sources.json").exists() else {}

main = (HERE.parent / "main.tex").read_text(encoding="utf-8")
live = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in main.split("\n"))
order, seen = [], set()
for g in re.findall(r"\\cite\{([^}]*)\}", live):
    for k in (x.strip() for x in g.split(",")):
        if k and k not in seen:
            seen.add(k)
            order.append(k)


def field(body, name):
    m = re.search(r"\b" + name + r"\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", body, re.I)
    return re.sub(r"\s+", " ", m.group(1)).strip() if m else ""


entries = {m.group(1): m.group(2) for m in re.finditer(r"@\w+\{([^,]+),(.*?)\n\}", bib, re.S)}

# Ask NCBI which PubMed records have a free full text in PubMed Central, so the list says
# "free" only where it really is.
import urllib.request

pmc_of = {}
_pmids = {k: m.group(1) for k, v in src.items()
          if (m := re.search(r"PubMed (\d+)", v.get("where", "")))}
if _pmids:
    try:
        _u = ("https://www.ncbi.nlm.nih.gov/pmc/utils/idconv/v1.0/?format=json&ids="
              + ",".join(sorted(set(_pmids.values())))
              + "&tool=citation-check&email=gregory.schwing@med.wayne.edu")
        _rec = json.loads(urllib.request.urlopen(_u, timeout=60).read())
        _by = {r.get("pmid"): r.get("pmcid") for r in _rec.get("records", []) if r.get("pmcid")}
        pmc_of = {k: _by[p] for k, p in _pmids.items() if _by.get(p)}
        print(f"  {len(pmc_of)} of {len(set(_pmids.values()))} PubMed records have free full text")
    except Exception as exc:
        print("  PMC lookup skipped:", type(exc).__name__)

rows = []
for i, key in enumerate(order, 1):
    b = entries.get(key, "")
    title = re.sub(r"[{}]", "", field(b, "title"))
    authors = field(b, "author").split(" and ")
    first = authors[0].split(",")[0] if authors and authors[0] else ""
    first = re.sub(r"[{}]", "", first)
    year = field(b, "year")
    journal = field(b, "journal") or field(b, "howpublished") or field(b, "publisher")
    journal = journal.split(chr(92) + "url")[0]
    journal = re.sub(r"[{}]", "", journal).strip().rstrip(".").strip()
    doi = field(b, "doi")
    where = src.get(key, {}).get("where", "")
    pmcid = re.search(r"(PMC\d+)", where) or (re.match(r"(PMC\d+)", pmc_of[key]) if key in pmc_of else None)
    pmid = re.search(r"PubMed (\d+)", where)
    arx = re.search(r"arXiv:([\d.]+)", where)
    if doi and "arxiv" in doi.lower():
        link = f"https://arxiv.org/abs/{arx.group(1)}" if arx else f"https://doi.org/{doi}"
        free = "arXiv, free"
    elif pmcid:
        link = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid.group(1)}/"
        free = "free full text"
    elif doi:
        link = f"https://doi.org/{doi}"
        free = "publisher, may be paywalled"
    else:
        link = re.search(r"url\{([^}]*)\}", b)
        link = link.group(1) if link else ""
        free = "document, free"
    extra = f"  PubMed {pmid.group(1)}" if pmid and not pmcid else ""
    rows.append((i, key, first, year, title, journal, link, free, extra))

out = ["# Reference download list — CTSpinoPelvic1K dataset article",
       "",
       f"{len(rows)} references, numbered as they appear in the manuscript. Paste the links into",
       "Mendeley's web importer, or drag the PDFs in after downloading.",
       ""]
for i, key, first, year, title, journal, link, free, extra in rows:
    out.append(f"{i}. **{first} {year}** — {title}")
    out.append(f"   {journal}  ·  `{key}`  ·  {free}{extra}")
    out.append(f"   {link}")
    out.append("")

(HERE.parent / "REFERENCE_DOWNLOADS.md").write_text("\n".join(out), encoding="utf-8")
free_n = sum(1 for r in rows if "free" in r[7])
print(f"wrote REFERENCE_DOWNLOADS.md: {len(rows)} references, {free_n} with a free full text")
