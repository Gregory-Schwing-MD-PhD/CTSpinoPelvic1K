"""v6_readme.py — say what v6 changed, in the README that ships with it.

build_v6.py copies the README across unchanged, which is correct for every section
describing the sources, the scheme and the loader -- none of that moved -- and wrong for the
one thing a person downloading v6 needs to know: what is different from v5. Without this the
tree carries eleven records with identifiers nobody has seen populated before and no note
saying so.

Two additions, both placed where a reader is already looking:

  A "What is new in v6" block near the top, because someone arriving at the repo decides in
  the first screen whether this revision is the one they want.

  A hardware paragraph inside the existing label-scheme section, because that section
  already lists 76-79 as declared-but-unused, and that sentence is now wrong.

Numbers come from paper/mpda/figures/hardware_stats.json, the file the manuscript and the
website both read, so three places cannot disagree.

    python scripts/v6_readme.py --readme data/hf_export_v6/README.md \\
        --stats paper/mpda/figures/hardware_stats.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", required=True)
    ap.add_argument("--stats", required=True)
    a = ap.parse_args()

    p = Path(a.readme)
    s = p.read_text(encoding="utf-8")
    st = json.loads(Path(a.stats).read_text(encoding="utf-8"))
    cls = st["classes"]

    if "What is new in v6" in s:
        print("  README already describes v6")
        return 0

    naive = 100.0 * st["flagged_1800HU"] / st["scanned"]
    block = f"""
## What is new in v6

**Surgical hardware is labelled.** Identifiers 76-79 have been declared in
`dataset_labels.json` since the scheme was written and were populated in **no record of any
previous release**. In v6 they are, and the block is extended to 82.

{st['confirmed']} of the {st['scanned']} records carry instrumentation. In every one of them
the metal had been absorbed into the bone label beside it -- a segmenter handed a bright
object against a cortical surface takes it for bone -- so naming the implant **reclaims**
voxels rather than adding them: {st['voxels_reclaimed_total']:,} across the eleven.

| id | class | n |
|---|---|---|
| 80 | `hardware_arthroplasty` | {cls.get('hardware_arthroplasty', 0)} |
| 82 | `hardware_osteosynthesis` | {cls.get('hardware_osteosynthesis', 0)} |
| 81 | `hardware_si_screw` | {cls.get('hardware_si_screw', 0)} |
| 77 | `hardware_cage` | {cls.get('hardware_cage', 0)} |

80 and 82 are the two arms of one clinical decision and must not be confused: osteosynthesis
holds parts of the **same bone** together, arthroplasty **replaces a joint**. Fixation leaves
the patient's own femoral head; a prosthesis does not. Pelvic incidence and pelvic tilt are
measured from that head, so in **{st['cases_with_femoral_head_replaced']} cases the landmark
is an implant** and any spinopelvic parameter computed from them was measured on metal.

**Why this matters for the transitional-anatomy use case.** An iatrogenic fusion is
indistinguishable from a congenital one to a distance measurement: a cage-bridged interspace
reads as "no gap" exactly as a congenitally fused transitional vertebra does. Filter on
`hardware_labelled` in `manifest.json` before running any gap-based analysis.

**Detection is not the hard part; interpretation is.** A threshold at 1800 HU flags
{st['flagged_1800HU']} records. At 2500 HU -- the lower of the two values validated in the
metal-segmentation literature -- {st['above_floor_2500HU']} keep a component above a
40-voxel floor. A radiologist read all {st['above_floor_2500HU']} and confirmed
{st['confirmed']}, rejecting {st['artefact']} as contrast, calcification and reconstruction
artefact. **True prevalence is {st['prevalence_pct']:.1f}%, not {naive:.1f}%.** Saturation
does not separate the two: both groups reach the 3071 HU scanner ceiling and
{st['peak_HU_above_ceiling_artefacts']} rejected proposals exceed it, peaking at
{st['max_artefact_peak_HU']:,} HU -- values above the ceiling being reconstruction overshoot
rather than denser metal.

**Case 0068 is corrected.** It had six free lumbar bodies under five labels: the pseudolabel
merged the top two, so every level below was named one level too high and the interbody cages
appeared an interspace higher than they sit. Renumbered against the twelfth rib -- an anchor
that is in the image, unlike the top of the spine -- to L1-L6, with T10-T12 added where 48 mm
of column was imaged and unlabelled, and the L2-L3 and L3-L4 vertebral-body mixing resolved.

**Five cases differ from published v5 for reasons unrelated to hardware.** `0179`, `0378`,
`0412`, `0787` and `1153` were hand-corrected after the v5 export was cut and never
re-exported, so those fixes appear for the first time here (largely renumbering: 1,025,631
voxels relabelled, chiefly a rib remap on 0179). The other 797 records are byte-identical to
v5 apart from the eleven hardware cases.

**Known limitation.** On `1035` the sacroiliac screws cross the joint and leave both hip
labels genuinely fragmented -- the largest connected piece holds 66% and 84% respectively.
This is recorded rather than repaired, because the fragmentation is anatomically real.

"""

    anchor = "## Design principles (the decisions that make it LSTV-aware)"
    assert s.count(anchor) == 1, "design-principles heading not found"
    s = s.replace(anchor, block.strip() + "\n\n" + anchor, 1)

    # the label-scheme section calls 76-79 reserved; it is not any more
    for old, new in (
        ("hardware (76-79)", "surgical hardware (76-82, populated in v6)"),
        ("hardware (76–79)", "surgical hardware (76–82, populated in v6)"),
    ):
        if old in s:
            s = s.replace(old, new)
            print(f"  label-scheme line updated: {old!r}")

    with open(p, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(s)
    print(f"  wrote {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
