from pathlib import Path
import re
p = Path("main.tex"); s = p.read_text(encoding="utf-8")
def words(t):
    t = re.sub(r"\cite\{[^}]*\}", "", t)
    t = re.sub(r"\[a-zA-Z]+\*?", "", t)
    return len(re.sub(r"[{}~]", " ", t).split())
a = s.index(r"\begin{abstract}"); b = s.index(r"\end{abstract}")
before = words(s[a:b])
# phrasing only; no claim is dropped, and margin is left because a journal's counter
# will not agree with mine to the word
T = [("It is intended for lumbosacral", "Intended for lumbosacral"),
     ("Existing collections either omit the pelvis", "Existing collections omit either the pelvis"),
     ("or omit the classes a transitional vertebra requires", "or the classes a transitional vertebra requires"),
     ("802 CT records carrying radiologist-sourced", "802 CT records with radiologist-sourced"),
     ("anatomical\nvariant studies, surgical planning and opportunistic screening research",
      "variant studies, surgical planning, and opportunistic screening"),
     ("whose review\nwas triaged rather than exhaustive", "whose review was triaged, not exhaustive"),
     ("pairs sharing an identical\naffine", "pairs sharing one affine"),
     ("LSTV-stratified five-fold\ncross-validation splits", "LSTV-stratified five-fold splits")]
for o, n in T:
    if o in s: s = s.replace(o, n, 1)
    else: print("  skipped (not found):", o[:40])
p.write_text(s, encoding="utf-8")
a = s.index(r"\begin{abstract}"); b = s.index(r"\end{abstract}")
print(f"abstract: {before} -> {words(s[a:b])} words (limit 300)")
