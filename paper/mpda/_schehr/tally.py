"""Per-reference tally: for each of the 38 keys, was text from the source actually read and
matched against the sentence that cites it? Written from the record of what was retrieved,
not from the narrative, so the count in CITATION_CHECK.md can be corrected."""
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent

# how the supporting text was obtained for each key
READ = {
    "lian2018": "PubMed abstract", "acr2021": "PDF, appropriateness tables",
    "acrct2022": "PDF", "acrmri2023": "PDF", "koninwalz2010": "abstract + PDF p.1778",
    "ctspine1k": "Crossref abstract", "ctpelvic1k": "PubMed abstract",
    "spineps2025": "PubMed abstract", "veridah2026": "arXiv abstract + PDF Table 1",
    "otake2012": "PubMed abstract", "lo2015": "PubMed abstract", "desilva2016": "PubMed abstract",
    "nass2013": "PDF, recommendations", "colonog": "TCIA collection page",
    "tcia": "PMC full text", "duplessis2018": "PMC full text", "poolman2023": "PMC full text",
    "nagata2025": "PubMed abstract", "zindrick1987": "PubMed abstract",
    "hughes2006": "PubMed abstract", "versedata2021": "PMC full text", "verse2021": "PubMed abstract",
    "ribsegv2": "PubMed abstract", "totalsegmentator": "PubMed abstract",
    "moller2026": "arXiv abstract", "osc2026": "PubMed abstract", "vialle2005": "PubMed abstract",
    "castellvi1984": "PubMed abstract", "pickhardt2013": "PubMed abstract",
    "epstein2021": "PubMed abstract", "fea2026": "PMC full text", "schinz2026": "PubMed abstract",
    "mody2008": "PubMed abstract", "tqip2018": "PDF §8", "benzel2015": "PDF chapter 1",
    "panjabi1991": "PDF, scanned; abstract and methods read from the page image",
    "panjabi1992": "PDF, scanned; abstract and methods read from the page image",
    "sixta2012": "PubMed abstract (A. Schehr's find)",
    "berry1987": "PubMed abstract (A. Schehr's find)",
}
NOT_READ = {
    "nnunet": "no abstract in PubMed; cited only as the framework the released weights run "
              "under, which is what its title states.",
}

main = (HERE.parent / "main.tex").read_text(encoding="utf-8")
live = "\n".join(re.sub(r"(?<!\\)%.*$", "", ln) for ln in main.split("\n"))
keys, seen = [], set()
for g in re.findall(r"\\cite\{([^}]*)\}", live):
    for k in (x.strip() for x in g.split(",")):
        if k and k not in seen:
            seen.add(k)
            keys.append(k)

read = [k for k in keys if k in READ]
unread = [k for k in keys if k in NOT_READ]
missing = [k for k in keys if k not in READ and k not in NOT_READ]

lines = ["## Per-reference tally", "",
         f"{len(keys)} references cited. **{len(read)} verified against text retrieved from the "
         f"source itself. {len(unread)} not.**", ""]
lines.append("| # | key | how the source was read |")
lines.append("|---|---|---|")
for i, k in enumerate(keys, 1):
    if k in READ:
        lines.append(f"| {i} | `{k}` | {READ[k]} |")
lines += ["", "### Not verified against source text", ""]
for k in keys:
    if k in NOT_READ:
        lines.append(f"- **`{k}`** — {NOT_READ[k]}")
if missing:
    lines += ["", "### Unaccounted for", ""] + [f"- `{k}`" for k in missing]

(HERE / "tally.md").write_text("\n".join(lines), encoding="utf-8")
print(f"{len(keys)} cited | {len(read)} verified from source text | {len(unread)} not | unaccounted {missing}")
for k in unread:
    print("  not read:", k)
