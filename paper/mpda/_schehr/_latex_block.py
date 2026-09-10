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
