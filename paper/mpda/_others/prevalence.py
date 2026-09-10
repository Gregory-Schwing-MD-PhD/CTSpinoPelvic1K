"""Find a citable, quotable statement of LSTV prevalence for the uncited 4-30% claim."""
import json
import re
import urllib.request

UA = {"User-Agent": "citation-check/1.0 (mailto:gregory.schwing@med.wayne.edu)"}


def get(u, t=60):
    return urllib.request.urlopen(urllib.request.Request(u, headers=UA), timeout=t).read().decode("utf-8", "ignore")


for name, pmid in [("lian2018", "29564611"), ("koninwalz2010", "20203111"), ("nagata2025", "40381031")]:
    try:
        j = json.loads(get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=EXT_ID:{pmid}&resultType=core&format=json"))
        r = j["resultList"]["result"][0]
        pmcid = r.get("pmcid")
        print(f"== {name}: openAccess={r.get('isOpenAccess')} pmcid={pmcid}")
        if pmcid:
            xml = get(f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML")
            txt = re.sub(r"<[^>]+>", " ", xml)
            txt = re.sub(r"\s+", " ", txt)
            for s in re.split(r"(?<=[.]) (?=[A-Z])", txt):
                s = s.strip()
                if "%" in s and re.search(r"prevalence|incidence|ranges|reported", s, re.I) and 40 < len(s) < 380:
                    print("   *", s[:340])
    except Exception as e:
        print(name, "ERR", e)
