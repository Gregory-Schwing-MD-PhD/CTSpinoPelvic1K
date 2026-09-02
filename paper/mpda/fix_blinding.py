r"""fix_blinding.py -- close the two identity leaks the first blinding pass missed.

The first pass blinded the author list by name, from a list written by hand. It therefore
missed a SECOND \author block further down -- the two radiology readers -- which was left
uncommented along with its affiliation. That is exactly the failure mode a by-name list
invites, so this pass works structurally instead: every remaining uncommented \author and
\affiliation is commented out, whatever the name inside it.

Also blinds the Ethics section, which named the institution whose review board issued the
determination. The guidelines say no institutional affiliations anywhere in the manuscript,
and a named review board identifies the authors as surely as a named department.

The reference list keeps its self-citation: the guidelines permit citing the authors' own
publications provided the running text refers to them in the third person, which it does.
"""
from pathlib import Path

p = Path(__file__).parent / "main.tex"
lines = p.read_text(encoding="utf-8").split("\n")

PLACEHOLDER = r"\author{[Author names removed for double-anonymised review]}"
n_auth = n_aff = 0
for i, l in enumerate(lines):
    st = l.lstrip()
    if l == PLACEHOLDER or st.startswith("%"):
        continue
    if st.startswith(r"\author{") and "removed for double-anonymised" not in l:
        lines[i] = "%" + l.lstrip()
        n_auth += 1
    elif st.startswith(r"\affiliation{") and "removed for double-anonymised" not in l:
        lines[i] = "%" + l.lstrip()
        n_aff += 1

s = "\n".join(lines)

old_ethics = "determination to that effect has been submitted to the Wayne State University Institutional"
new_ethics = "determination to that effect has been submitted to the authors' institutional"
assert old_ethics in s, "ethics anchor missing"
s = s.replace(old_ethics, new_ethics)

p.write_text(s, encoding="utf-8")
print(f"commented {n_auth} stray \\author and {n_aff} stray \\affiliation lines")
print("blinded the named review board in Ethics")
