r"""Rebuild ctspinopelvic1k.bib from Crossref records, keeping the manuscript's keys.

Authors, volume, issue, pages, year and DOI come from the Crossref BibTeX pulled for each
DOI (crossref/<key>.bib). Titles are kept from the previously verified .bib where one
exists, because Crossref drops subtitles (Panjabi 1992, Lo 2015). Journal names are
abbreviated as Medical Physics prints them. Entries without a DOI (two ACR practice
parameters, the TQIP guideline, the Benzel book) are written by hand at the end, with the
URLs Ashley Schehr located.
"""
import html
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "ctspinopelvic1k.bib"
# the pre-rebuild file, kept as the source of hand-verified titles, DOI casing and the
# article numbers Crossref omits. Reading OUT here would read this script's own output.
OLD = (HERE / "ctspinopelvic1k.bib.before").read_text(encoding="utf-8")

ABBR = {
    "European Spine Journal": "Eur. Spine J.",
    "Journal of the American College of Radiology": "J. Am. Coll. Radiol.",
    "American Journal of Neuroradiology": "AJNR Am. J. Neuroradiol.",
    "Machine Learning for Biomedical Imaging": "Mach. Learn. Biomed. Imaging",
    "International Journal of Computer Assisted Radiology and Surgery": "Int. J. Comput. Assist. Radiol. Surg.",
    "European Radiology": "Eur. Radiol.",
    "Physics in Medicine and Biology": "Phys. Med. Biol.",
    "Physics in Medicine & Biology": "Phys. Med. Biol.",
    "Spine": "Spine (Phila. Pa. 1976)",
    "SPINE": "Spine (Phila. Pa. 1976)",
    "The Spine Journal": "Spine J.",
    "Journal of Digital Imaging": "J. Digit. Imaging",
    "Journal of Anatomy": "J. Anat.",
    "Skeletal Radiology": "Skeletal Radiol.",
    "American Journal of Roentgenology": "AJR Am. J. Roentgenol.",
    "Scientific Data": "Sci. Data",
    "Medical Image Analysis": "Med. Image Anal.",
    "IEEE Transactions on Medical Imaging": "IEEE Trans. Med. Imaging",
    "Radiology: Artificial Intelligence": "Radiol. Artif. Intell.",
    "Cureus": "Cureus",
    "Nature Methods": "Nat. Methods",
    "The Journal of Bone & Joint Surgery": "J. Bone Joint Surg. Am.",
    "The Journal of Bone and Joint Surgery-American Volume": "J. Bone Joint Surg. Am.",
    "Annals of Internal Medicine": "Ann. Intern. Med.",
    "Surgical Neurology International": "Surg. Neurol. Int.",
    "Journal of Clinical Medicine": "J. Clin. Med.",
}

# year of the printed volume where Crossref gives the online-first year instead
YEAR_FIX = {"nnunet": "2021", "spineps2025": "2025", "schinz2026": "2026"}

# protect acronyms and proper nouns from the style's sentence-casing
PROTECT = ["CT", "MRI", "MR", "VerSe", "CTSpine1K", "TCIA", "nnU-Net", "RibSeg", "TotalSegmentator",
           "SPINEPS", "VERIDAH", "OpenSpineConsortium", "LevelCheck", "ACR", "T2", "3D-2D",
           "ACRIN 6664", "Cancer Imaging Archive", "COLONOGRAPHY", "Appropriateness Criteria"]

# keys in manuscript order of first citation are irrelevant here; bibtex orders by citation.
KEYS = ["lian2018", "acr2021", "koninwalz2010", "ctspine1k", "ctpelvic1k", "spineps2025",
        "veridah2026", "otake2012", "lo2015", "desilva2016", "nass2013", "colonog", "tcia",
        "duplessis2018", "poolman2023", "nagata2025", "zindrick1987", "panjabi1991",
        "panjabi1992", "hughes2006", "versedata2021", "verse2021", "ribsegv2",
        "totalsegmentator", "moller2026", "osc2026", "nnunet", "vialle2005", "castellvi1984",
        "pickhardt2013", "epstein2021", "fea2026", "schinz2026", "mody2008"]


def field(src, name):
    m = re.search(r"\b" + name + r"\s*=\s*\{((?:[^{}]|\{[^{}]*\})*)\}", src, re.I)
    return html.unescape(m.group(1).strip()) if m else ""


def old_title(key):
    m = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\}", OLD, re.S)
    return field(m.group(1), "title") if m else ""


import unicodedata

BS = chr(92)   # a literal backslash, kept out of string literals so nothing can eat it
LATEX = {"ö": "{" + BS + '"o}', "ü": "{" + BS + '"u}', "ä": "{" + BS + '"a}', "Ö": "{" + BS + '"O}',
         "é": "{" + BS + "'e}", "É": "{" + BS + "'E}", "è": "{" + BS + "`e}", "ç": "{" + BS + "c{c}}",
         "ñ": "{" + BS + "~n}", "á": "{" + BS + "'a}", "í": "{" + BS + "'i}", "ó": "{" + BS + "'o}",
         "ú": "{" + BS + "'u}", "ß": "{" + BS + "ss}", "č": "{" + BS + "v{c}}", "Š": "{" + BS + "v{S}}",
         "š": "{" + BS + "v{s}}", "ł": "{" + BS + "l}", "ø": "{" + BS + "o}", "å": "{" + BS + "aa}",
         "’": "'", "‘": "'", "–": "--", "—": "---", " ": " "}


def ascii_tex(t):
    for k, v in LATEX.items():
        t = t.replace(k, v)
    out = []
    for c in t:
        if ord(c) < 128:
            out.append(c)
        else:
            base = unicodedata.normalize("NFKD", c)
            out.append("".join(ch for ch in base if ord(ch) < 128) or "?")
    return "".join(out)


def fix_name(n):
    n = ascii_tex(n.strip())
    if n.upper() == n and any(c.isalpha() for c in n):   # PANJABI, MANOHAR M. -> Panjabi, Manohar M.
        n = " ".join(w.capitalize() if len(w) > 1 else w for w in n.split(" "))
        n = re.sub(r"\b(Mc)(\w)", lambda m: m.group(1) + m.group(2).upper(), n)
    return n


def protect(t):
    t = re.sub(r"\s+", " ", t).strip()
    t = t.replace("®", "")
    for p in sorted(PROTECT, key=len, reverse=True):
        if "{" + p + "}" in t:
            continue
        t = re.sub(r"(?<![\w{])" + re.escape(p) + r"(?![\w}])", "{" + p + "}", t)
    return t


def wrap(s, indent=13, width=88):
    words, lines, cur = s.split(" "), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width and cur:
            lines.append(cur); cur = w
        else:
            cur = (cur + " " + w).strip()
    lines.append(cur)
    return ("\n" + " " * indent).join(lines)


entries = []
for key in KEYS:
    raw = (HERE / "crossref" / f"{key}.bib").read_bytes()
    try:
        src = raw.decode("utf-8")
    except UnicodeDecodeError:
        src = raw.decode("latin-1")
    kind = "article" if src.lstrip().startswith("@article") else "misc"
    names = [fix_name(a) for a in field(src, "author").split(" and ")]
    # The journal prints "First Author et al." beyond six authors, and medphy.bst triggers
    # that on a trailing "others". The full list is kept in the file so the entry is
    # complete and checkable; only the printed form is shortened.
    if len(names) > 6:
        names.append("others")
    authors = " and ".join(names)
    title = old_title(key) or field(src, "title")
    if key == "colonog":
        title = "Data from {CT COLONOGRAPHY} ({ACRIN} 6664)"
    title = protect(re.sub(r"\s*\n\s*", " ", title))
    journal = ABBR.get(field(src, "journal"), field(src, "journal"))
    if kind == "article" and field(src, "journal") not in ABBR:
        print("  UNABBREVIATED journal for", key, ":", field(src, "journal"))
    volume, number, pages = field(src, "volume"), field(src, "number"), field(src, "pages").replace("–", "--").replace("—", "--")
    year = YEAR_FIX.get(key) or field(src, "year")
    doi = field(src, "doi") or field(src, "DOI")
    m = re.search(r"@\w+\{" + re.escape(key) + r",(.*?)\n\}", OLD, re.S)
    # keep the DOI in its registered case (Crossref lowercases some); the old file's form is the registered one
    if m and field(m.group(1), "doi"):
        doi = field(m.group(1), "doi")
    # Crossref omits article numbers used in place of a page range (Sci. Data 8:284,
    # Radiol. Artif. Intell. 5:e230024); fall back to the verified file for those.
    if m and not pages:
        pages = field(m.group(1), "pages")
    # Crossref lowercases some registrant prefixes (10.1097/brs...); the registered form is
    # uppercase and the rest of the list prints it that way, so normalise for consistency.
    doi = re.sub(r"^10\.1097/brs\.", "10.1097/BRS.", doi)
    doi_note = doi.replace("_", r"\_")
    lines = [f"@{kind}{{{key},", f"  author  = {{{wrap(authors)}}},", f"  title   = {{{wrap(title)}}},"]
    if kind == "article":
        lines.append(f"  journal = {{{journal}}},")
        if volume: lines.append(f"  volume  = {{{volume}}},")
        if number: lines.append(f"  number  = {{{number}}},")
        if pages:  lines.append(f"  pages   = {{{pages}}},")
    else:
        pub = field(src, "publisher")
        if key == "colonog":
            lines.append("  howpublished = {The Cancer Imaging Archive, version 2 [dataset]},")
        elif pub:
            lines.append(f"  howpublished = {{{pub}}},")
        arx = re.search(r"arxiv\.(\d{4}\.\d{5})", doi, re.I)
        if arx:
            lines.append(f"  note    = {{arXiv:{arx.group(1)}. doi:{doi_note}}},")
    lines.append(f"  year    = {{{year}}},")
    lines.append(f"  doi     = {{{doi}}},")
    if kind == "article" or key == "colonog":
        lines.append(f"  note    = {{doi:{doi_note}}},")
    lines.append("}")
    entries.append("\n".join(lines))

HAND = r"""
% No DOI exists for the ACR practice parameters or the ACS TQIP guideline; the URLs are the
% documents' own hosts (located by A. Schehr, September 2026). The MRI parameter is the one
% that states the lumbar coverage in levels, T12 to S1; the CT parameter is cited beside it
% because the planning study in question is CT.
@misc{acrmri2023,
  author       = {{American College of Radiology}},
  title        = {{ACR}--{ASNR}--{SABI}--{SSR} practice parameter for the performance of magnetic
                  resonance imaging ({MRI}) of the adult spine, revised 2023 ({Resolution} 6)},
  howpublished = {American College of Radiology, Reston, VA.
                  \url{https://www.asnr.org/wp-content/uploads/2019/06/MR-Adult-Spine.pdf}},
  year         = {2023},
}

@misc{acrct2022,
  author       = {{American College of Radiology}},
  title        = {{ACR}--{ASNR}--{ASSR}--{SPR} practice parameter for the performance of computed
                  tomography ({CT}) of the spine, revised 2022 ({Resolution} 23)},
  howpublished = {American College of Radiology, Reston, VA.
                  \url{https://www.asnr.org/wp-content/uploads/2019/06/CT-Spine-1.pdf}},
  year         = {2022},
}

@misc{tqip2018,
  author       = {{American College of Surgeons Committee on Trauma}},
  title        = {{ACS TQIP} best practices guidelines in imaging},
  howpublished = {American College of Surgeons, Chicago.
                  \url{https://www.facs.org/media/oxdjw5zj/imaging_guidelines.pdf}},
  year         = {2018},
}

% Print ISBN 978-1-60406-924-2; publisher page https://shop.thieme.com/Biomechanics-of-Spine-Stabilization/9781604069242
@book{benzel2015,
  author    = {Benzel, Edward C.},
  title     = {Biomechanics of Spine Stabilization},
  edition   = {3rd},
  publisher = {Thieme},
  address   = {New York},
  year      = {2015},
  note      = {ISBN 978-1-60406-924-2},
}
"""

HEADER = r"""% ctspinopelvic1k.bib -- references for the Medical Physics Dataset and Software Article.
%
% Use with the journal's medphy.bst:
%     \bibliographystyle{./medphy}
%     \bibliography{./ctspinopelvic1k}
% then run bibtex and paste the .bbl into the manuscript (inline_bbl.py), as the template
% README instructs.
%
% REBUILT 9 September 2026 (_schehr/make_bib.py) from the Crossref record of every DOI, so
% that author lists, volumes, issues, pages and DOIs are the registered ones rather than
% transcriptions. Titles are the previously verified ones where Crossref drops a subtitle.
% Journal names are abbreviated as the journal prints them. The four items without a DOI
% (two ACR practice parameters, the ACS TQIP guideline, the Benzel textbook) are at the end
% with the URL or ISBN of the document itself.
%
% Two entries were added at the co-author's request: schinz2026, the vertebral-shape
% classification of the thoracolumbar junction, and mody2008, the primary source for the
% one-in-3,110 wrong-level rate that epstein2021 quotes.
"""

OUT.write_text(HEADER + "\n" + "\n\n".join(entries) + "\n" + HAND, encoding="utf-8")
print(f"wrote {OUT.name}: {len(entries)} Crossref entries + 4 hand entries")
