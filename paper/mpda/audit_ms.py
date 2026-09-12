"""paper/mpda/audit_ms.py -- check the manuscript against the data it claims to describe.

Every number here has been wrong at least once, and none of them announced it. The recurring
failure in this project is a value that is wrong but plausible, so the checks below all
compare the TEXT against something outside the text: a CSV, the label files, the build log.

    python paper/mpda/audit_ms.py
"""
from __future__ import annotations

import csv
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEX = HERE / "main.tex"
MORPH = ROOT / "morphometrics"

fails, warns, oks = [], [], []


def ok(msg):
    oks.append(msg)


def fail(msg):
    fails.append(msg)


def warn(msg):
    warns.append(msg)


s = TEX.read_text(encoding="utf-8")


def num(rows, key):
    out = []
    for r in rows:
        v = (r.get(key) or "").strip()
        try:
            out.append(float(v))
        except ValueError:
            pass
    return out


def median(v):
    v = sorted(v)
    n = len(v)
    return None if not n else (v[n // 2] if n % 2 else 0.5 * (v[n // 2 - 1] + v[n // 2]))


# ---------------------------------------------------------------- references
keys_defined = set(re.findall(r"\\bibitem\{([^}]+)\}", s))
keys_cited = set()
for m in re.finditer(r"\\cite\{([^}]+)\}", s):
    keys_cited.update(k.strip() for k in m.group(1).split(","))

dangling = sorted(keys_cited - keys_defined)
uncited = sorted(keys_defined - keys_cited)
if dangling:
    fail(f"cited but not defined: {dangling}")
else:
    ok(f"all {len(keys_cited)} cited keys are defined")
if uncited:
    fail(f"defined but never cited: {uncited}")
else:
    ok(f"all {len(keys_defined)} bibliography entries are cited")

# every entry should carry a year and a locator
for key in sorted(keys_defined):
    m = re.search(r"\\bibitem\{" + re.escape(key) + r"\}(.*?)(?=\\bibitem\{|\\end\{thebibliography\})",
                  s, re.S)
    body = m.group(1) if m else ""
    if not re.search(r"\(\d{4}\)|\b(19|20)\d{2}\b", body):
        warn(f"{key}: no year found")
    if not re.search(r"doi:|\\url\{|ISBN|arXiv", body, re.I):
        warn(f"{key}: no DOI, URL, ISBN or arXiv locator")

# ---------------------------------------------------------------- S1 withdrawal
if re.search(r"S1 carved from the sacrum \(29\)", s):
    fail("label scheme still lists S1 (29) as a released class")
else:
    ok("label scheme no longer lists a carved S1")
if "Identifier 29 is retired" in s:
    ok("id 29 is explicitly recorded as retired")
else:
    warn("no sentence stating that identifier 29 is retired")

# ---------------------------------------------------------------- spinopelvic
f = MORPH / "surgical_morphometrics.csv"
if f.exists():
    rows = list(csv.DictReader(open(f)))
    claimed = {}
    # the table reports MEAN +- SD: the reference series (Veilleux, Hasegawa) report
    # means, and comparing our median against their mean was costing 0.5-0.6 degrees to
    # the right skew in pelvic incidence and tilt. The check follows the table.
    m = re.search(r"Pelvic incidence & ([\d.]+)", s)
    if m:
        claimed["pelvic_incidence_deg"] = float(m.group(1))
    m = re.search(r"Sacral slope & ([\d.]+)", s)
    if m:
        claimed["sacral_slope_deg"] = float(m.group(1))
    m = re.search(r"Pelvic tilt & ([\d.]+)", s)
    if m:
        claimed["pelvic_tilt_deg"] = float(m.group(1))
    for k, want in claimed.items():
        vals = num(rows, k)
        got = (sum(vals) / len(vals)) if vals else None
        if got is None:
            warn(f"{k}: no values in {f.name}")
        elif abs(got - want) > 0.15:
            fail(f"Table II {k}: manuscript {want}, data mean {got:.1f}  ({f})")
        else:
            ok(f"Table II {k} = {want} matches the data mean ({got:.1f})")
    m = re.search(r"over the (\d+) of 802 records", s)
    if m:
        want_n = int(m.group(1))
        got_n = len(num(rows, "pelvic_incidence_deg"))
        if want_n != got_n:
            fail(f"caption says n={want_n}, data has {got_n} pelvic incidences")
        else:
            ok(f"caption n={want_n} matches the data")
else:
    warn(f"{f} not present; spinopelvic numbers unchecked")

# ---------------------------------------------------------------- count-free
f = MORPH / "transition_morphometrics.csv"
if f.exists():
    rows = list(csv.DictReader(open(f)))
    meas = [r for r in rows if (r.get("rib12_11_ratio_min") or "").strip()]
    stump = [r for r in meas if float(r["rib12_11_ratio_min"]) < 0.33]
    m = re.search(r"stump twelfth rib in (\d+) of the (\d+) records", s)
    if m:
        cs, cn = int(m.group(1)), int(m.group(2))
        if cn != len(meas):
            fail(f"text says {cn} measurable pairs, data has {len(meas)}")
        else:
            ok(f"measurable rib pairs {cn} matches the data")
        if cs != len(stump):
            fail(f"text says {cs} stump ribs, data has {len(stump)}")
        else:
            ok(f"stump ribs {cs} matches the data")
    lab = [(r.get("lstv_label") or "").strip().upper() for r in rows]
    m = re.search(r"sacralization in (\d+)", s)
    if m:
        want, got = int(m.group(1)), lab.count("SACRALIZATION")
        (ok if want == got else fail)(
            f"sacralizations: text {want}, data {got}")
    m = re.search(r"lumbarization in (\d+)", s)
    if m:
        want, got = int(m.group(1)), lab.count("LUMBARIZATION")
        (ok if want == got else fail)(
            f"lumbarizations: text {want}, data {got}")
else:
    warn(f"{f} not present; count-free numbers unchecked")

# ---------------------------------------------------------------- figures
figdir = HERE / "figures"
for m in re.finditer(r"includegraphics\[[^\]]*\]\{(figures/[^}]+)\}", s):
    p = HERE / m.group(1)
    if not p.exists():
        fail(f"figure missing: {m.group(1)}")
    else:
        ok(f"figure present: {p.name}")

# ---------------------------------------------------------------- build
print("building --reprint ...", flush=True)
try:
    r = subprocess.run(["wsl", "-e", "bash", "-lc",
                        "cd /mnt/c/Users/grego/OneDrive/Desktop/CTSpinoPelvic1K-1/paper/mpda "
                        "&& ./build.sh --reprint 2>&1 | tail -40"],
                       capture_output=True, text=True, timeout=900)
    out = r.stdout + r.stderr
    m = re.search(r"PAGES: (\d+)", out)
    if m:
        pages = int(m.group(1))
        (ok if pages <= 10 else fail)(f"reprint is {pages} pages (limit 10)")
    else:
        fail("build produced no page count")
    if re.search(r"^!", out, re.M):
        fail("LaTeX errors in the build output")
    else:
        ok("no LaTeX errors")
    if "undefined" in out.lower() and "There were undefined references" in out:
        fail("undefined references remain")
    else:
        ok("no undefined references")
except Exception as exc:                                            # noqa: BLE001
    warn(f"build not run: {type(exc).__name__}: {exc}")

# ---------------------------------------------------------------- report
print()
for m in oks:
    print(f"  ok    {m}")
for m in warns:
    print(f"  warn  {m}")
for m in fails:
    print(f"  FAIL  {m}")
print(f"\n{len(oks)} ok, {len(warns)} warnings, {len(fails)} failures")
sys.exit(1 if fails else 0)
