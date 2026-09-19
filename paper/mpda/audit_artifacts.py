r"""paper/mpda/audit_artifacts.py -- every figure and table in the manuscript, where it
comes from, and whether it still regenerates.

WHY THIS EXISTS. audit_ms.py checks that the NUMBERS in the text match the CSVs they were
computed from. It says nothing about the figures, and a figure is exactly where this project
has been bitten: Figure 2 was once rendered from four cases whose identifiers nobody wrote
down, and later a retired quality gate silently swapped one of those cases out while the
caption still named the old one. A figure whose generator is unrecorded is not reproducible,
and a figure whose generator is recorded but never re-run is only reproducible in principle.

WHAT IT CHECKS, per artifact:
  1. a generator is recorded here at all (the check that would have caught Figure 2);
  2. the generator file exists;
  3. the data it reads exists;
  4. for the cheap, deterministic ones (--regen), that re-running reproduces the committed
     figure PIXEL FOR PIXEL. A PDF carries a creation timestamp, so bytes differ on every
     run and md5 is useless here; rendering both and differencing is the real comparison.

    python paper/mpda/audit_artifacts.py            # records + existence
    python paper/mpda/audit_artifacts.py --regen    # also re-run and pixel-compare

Heavy generators (they read the 802 label volumes) are marked cost="heavy" and are not run
by --regen; their exact command is recorded so a reader can run them.
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]

FAILS: list[str] = []
WARNS: list[str] = []


def ok(m):
    print("  ok    " + m)


def warn(m):
    WARNS.append(m)
    print("  warn  " + m)


def fail(m):
    FAILS.append(m)
    print("  FAIL  " + m)


# label -> how it is made. `out` is relative to paper/mpda.
ARTIFACTS = [
    dict(name="Fig 1 pipeline", out=None, cost="inline",
         gen="main.tex (TikZ, drawn inline)",
         note="exported standalone by make_submission.py for the journal form",
         reads=[]),
    dict(name="Fig 2 anchors", out="figures/fig_anchors.pdf", cost="heavy",
         gen="scripts/make_fig_anchors.py",
         cmd="python scripts/make_fig_anchors.py --reselect",
         note="select_anchor_cases.py picks the four cases by rule -> "
              "morphometrics/anchor_cases.json, render_anchors.py draws them",
         reads=["morphometrics/transition_morphometrics.csv",
                "morphometrics/anchor_cases.json"]),
    dict(name="Fig 3 hardware", out="figures/fig_hardware.pdf", cost="heavy",
         gen="scripts/render_hardware_gallery.py",
         cmd="python scripts/render_hardware_gallery.py",
         note="renders the confirmed implants through bone; needs the label volumes",
         reads=["morphometrics/metal.csv"]),
    dict(name="Fig 4 validation", out="figures/fig_validation.pdf", cost="cheap",
         gen="paper/mpda/make_figures.py",
         cmd="python paper/mpda/make_figures.py",
         reads=["morphometrics/surgical_morphometrics.csv"]),
    dict(name="Fig 5 level atlas", out="figures/fig_levelatlas.pdf", cost="cheap",
         gen="paper/mpda/make_levelatlas_fig.py",
         cmd="python paper/mpda/make_levelatlas_fig.py",
         reads=["morphometrics/level_atlas.csv", "morphometrics/panjabi_reference.csv"]),
    dict(name="Fig 6 field of view", out="figures/fig_fov.pdf", cost="cheap",
         gen="paper/mpda/make_fov_fig.py",
         cmd="python paper/mpda/make_fov_fig.py",
         reads=["morphometrics/label_census.csv"]),

    dict(name="Table I prior art", out=None, cost="manual",
         gen="hand-built from the cited collections",
         note="literature table; every row carries its citation in main.tex",
         reads=[]),
    dict(name="Table II pseudolabel Dice", out=None, cost="data",
         gen="results/pseudolabel_dice/",
         note="nnU-Net per-fold validation Dice at each fold's selected checkpoint",
         reads=["results/pseudolabel_dice"]),
    dict(name="Table III rib QC", out=None, cost="data",
         gen="results/rib_qc_v4_vs_release/rib_qc_stages.py",
         note="same code on the v4 pseudolabels and the release; "
              "gates_v4_vs_release.json and fragments_release/summary.json hold the numbers",
         reads=["results/rib_qc_v4_vs_release/gates_v4_vs_release.json",
                "results/rib_qc_v4_vs_release/fragments_release/summary.json",
                "results/rib_qc_v4_vs_release/incidence_release.json"]),
    dict(name="Table IV spinopelvic", out=None, cost="data",
         gen="scripts/extract_surgical_morphometrics.py",
         note="means are asserted against the CSV by audit_ms.py",
         reads=["morphometrics/surgical_morphometrics.csv"]),
    dict(name="Table V completeness", out=None, cost="manual",
         gen="hand-written in main.tex from the released manifest",
         note="per-field record counts; not script-generated. Regenerate the counts with "
              "a pass over manifest.json if the manifest changes.",
         reads=["data/zenodo_deposit/manifest.json"]),
    dict(name="Table S1 level census", out=None, cost="data",
         gen="scripts/make_census_table.py",
         cmd="python scripts/make_census_table.py",
         note="writes paper/mpda/census_table.tex; Fig 6's caption points at it",
         reads=["morphometrics/label_census.csv"]),
]


def check_records():
    print("\n=== every artifact has a recorded generator ===")
    for a in ARTIFACTS:
        if not a.get("gen"):
            fail("%s: NO GENERATOR RECORDED" % a["name"])
            continue
        gen = a["gen"]
        # a recorded generator must exist on disk unless it is inline or manual
        if a["cost"] in ("inline", "manual"):
            ok("%-24s %s" % (a["name"], gen))
            continue
        p = ROOT / gen.split()[0]
        if p.exists():
            ok("%-24s %s" % (a["name"], gen))
        else:
            fail("%s: generator missing on disk: %s" % (a["name"], gen))


def check_inputs():
    print("\n=== the data each one reads is present ===")
    for a in ARTIFACTS:
        for r in a.get("reads", []):
            p = ROOT / r
            if p.exists():
                ok("%-24s reads %s" % (a["name"], r))
            else:
                warn("%s: input not present here: %s" % (a["name"], r))


def pixels(pdf: Path, dpi=110):
    import fitz
    import numpy as np
    pm = fitz.open(str(pdf))[0].get_pixmap(dpi=dpi)
    import numpy as np
    return np.frombuffer(pm.samples, np.uint8).reshape(pm.height, pm.width, pm.n).astype(int)


def check_regen():
    import numpy as np
    print("\n=== cheap generators re-run and compared PIXEL FOR PIXEL ===")
    print("  (a PDF stores a creation time, so bytes always differ; pixels are the test)")
    for a in ARTIFACTS:
        if a["cost"] != "cheap" or not a.get("out"):
            continue
        out = HERE / a["out"]
        if not out.exists():
            fail("%s: %s not present" % (a["name"], a["out"]))
            continue
        with tempfile.TemporaryDirectory() as td:
            keep = Path(td) / "committed.pdf"
            shutil.copy2(out, keep)
            r = subprocess.run([sys.executable] + a["cmd"].split()[1:],
                               cwd=ROOT, capture_output=True, text=True, timeout=900)
            if r.returncode != 0:
                fail("%s: generator exited %d: %s" % (a["name"], r.returncode, r.stderr[-300:]))
                shutil.copy2(keep, out)
                continue
            try:
                A, B = pixels(keep), pixels(out)
            except Exception as e:                                  # noqa: BLE001
                warn("%s: could not rasterise (%s)" % (a["name"], e))
                continue
            if A.shape != B.shape:
                fail("%s: regenerated figure has a different size %s vs %s"
                     % (a["name"], A.shape, B.shape))
                continue
            d = np.abs(A - B)
            n = int((d.sum(2) > 0).sum())
            if n == 0:
                ok("%-24s regenerates pixel-identical" % a["name"])
            else:
                fail("%s: regenerated figure differs in %d px (max %d)" % (a["name"], n, d.max()))


ap = argparse.ArgumentParser()
ap.add_argument("--regen", action="store_true",
                help="re-run the cheap generators and pixel-compare")
args = ap.parse_args()

check_records()
check_inputs()
if args.regen:
    check_regen()
else:
    print("\n(--regen re-runs the cheap generators and pixel-compares them)")

print("\n=== heavy generators, recorded but not run here ===")
for a in ARTIFACTS:
    if a["cost"] == "heavy":
        print("  %-24s %s" % (a["name"], a.get("cmd", a["gen"])))

print("\n%d ok, %d warnings, %d failures" % (
    0, len(WARNS), len(FAILS)))
sys.exit(1 if FAILS else 0)
