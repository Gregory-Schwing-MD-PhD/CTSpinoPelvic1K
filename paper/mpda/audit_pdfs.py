"""Read the fixes back out of every rendered PDF. A build that succeeds is not evidence
the PDF says the right thing -- the submission packet once shipped a stale DOI from a
prebuilt preprint and the build reported success throughout."""
import re
from pathlib import Path

import fitz

ROOT = Path(r"C:\Users\grego\OneDrive\Desktop\CTSpinoPelvic1K-1")
PDFS = [
    ("preprint",        "paper/mpda/CTSpinoPelvic1K_dataset_article.pdf"),
    ("reprint check",   "paper/mpda/CTSpinoPelvic1K_reprint_check.pdf"),
    ("arxiv preview",   "paper/mpda/CTSpinoPelvic1K_arxiv_preview.pdf"),
    ("submission main", "dist/submission/Main_Document_CTSpinoPelvic1K.pdf"),
    ("submission title", "dist/submission/Title_Page_CTSpinoPelvic1K.pdf"),
]

print("%-17s %5s  %-9s %-7s %-7s %-8s %s" %
      ("pdf", "pages", "Aichmair", "noLurie", "v11DOI", "noOldDOI", "hasSI"))
bad = 0
for name, rel in PDFS:
    p = ROOT / rel
    if not p.exists():
        print("%-17s MISSING %s" % (name, rel))
        bad += 1
        continue
    d = fitz.open(str(p))
    t = re.sub(r"\s+", "", "".join(pg.get_text() for pg in d))
    is_title = "title" in name
    checks = {
        "Aichmair": ("Aichmair" in t) or is_title,
        "noLurie": "Lurie" not in t,
        "v11DOI": ("22797014" in t) or is_title,
        "noOldDOI": "22642578" not in t,
        # the journal wants supplements linked below the Conclusion, so the section is
        # REQUIRED in the article; what must not appear is a dangling S-reference
        "hasSI": bool(re.search("SUPPORTINGINFORMATION", t, re.I)) or is_title,
    }
    row = "  ".join("%-7s" % ("ok" if v else "FAIL") for v in checks.values())
    print("%-17s %5d  %s" % (name, d.page_count, row))
    bad += sum(1 for v in checks.values() if not v)

print("\n%s" % ("ALL PDFS CARRY THE FIXES" if bad == 0 else "%d PROBLEM(S)" % bad))
raise SystemExit(1 if bad else 0)
