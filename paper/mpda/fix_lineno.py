r"""fix_lineno.py -- use REVTeX's own line numbering instead of the lineno package.

Continuous line numbering is a Medical Physics submission requirement. Supplying it with
the lineno package breaks against amsmath: every display equation raises "Improper
\prevdepth" and "You can't use \prevdepth in restricted horizontal mode", and the run
produced a PDF only because nonstopmode carried on past ten errors.

REVTeX 4.2 accepts `linenumbers` as a class option and has no such conflict, so the
requirement is met by the class rather than by a package fighting it.
"""
from pathlib import Path

p = Path(__file__).parent / "main.tex"
s = p.read_text(encoding="utf-8")

old_class = r"\documentclass[aapm,mph,amsmath,amssymb,preprint]{revtex4-2}"
new_class = r"\documentclass[aapm,mph,amsmath,amssymb,preprint,linenumbers]{revtex4-2}"
assert old_class in s
s = s.replace(old_class, new_class)

old_pkg = ("% Continuous line numbering is a submission requirement, not a preference.\n"
           "\\usepackage[mathlines]{lineno}\n")
new_pkg = ("% Continuous line numbering is a submission requirement. It comes from the\n"
           "% `linenumbers' REVTeX class option above, NOT from the lineno package:\n"
           "% lineno conflicts with amsmath and raises \\prevdepth errors on every\n"
           "% display equation.\n")
assert old_pkg in s
s = s.replace(old_pkg, new_pkg)

old_cmd = ("% Continuous line numbering: a submission requirement for Medical Physics.\n"
           "\\linenumbers\n")
assert old_cmd in s
s = s.replace(old_cmd, "")

p.write_text(s, encoding="utf-8")
print("switched to REVTeX native line numbering")
