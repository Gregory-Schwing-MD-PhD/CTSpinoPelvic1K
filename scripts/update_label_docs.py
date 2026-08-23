"""scripts/update_label_docs.py — put the new classes into the release's own label docs.

Five classes are live in label_scheme.py and absent from everything the release actually
ships: rib_left_lumbar (74), rib_right_lumbar (75), and the hardware block (76-79).
Downstream code reads dataset_labels.json and dataset_interface.py, not label_scheme.py, so
a class missing from those is a class that silently does not exist -- voxels carrying it get
dropped, or worse, treated as an unknown index.

Regenerates the id->name map from label_scheme (the single source of truth) rather than
editing entries by hand, so the release cannot drift from the code again.

    python scripts/update_label_docs.py --export-dir data/hf_export_v5
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import label_scheme as LS                                          # noqa: E402

NEW_NOTES = {
    74: ("A rib articulating with a LUMBAR vertebra. Given its own class rather than being "
         "forced to be 'rib 12': a 13th rib is a finding, and numbering it as the twelfth "
         "consumed the id the T12 rib needed."),
    75: "As 74, right side.",
    76: ("Surgical instrumentation whose subtype is not distinguished. Neither bone nor any "
         "anatomical class; labelled rather than ignored because a cage bridging a disc "
         "space makes two vertebrae look fused to any distance measurement, and that must "
         "stay separable from congenital fusion."),
    77: "Interbody cage or spacer.",
    78: "Pedicle screws and rods.",
    79: "Plates and other fixation.",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    a = ap.parse_args()
    ed = Path(a.export_dir)

    d = LS.label_dict()
    by_id = {int(v): k for k, v in d.items()}

    p = ed / "dataset_labels.json"
    old = {}
    if p.exists():
        try:
            old = json.loads(p.read_text())
        except json.JSONDecodeError:
            old = {}
    prev_ids = set()
    if isinstance(old, dict):
        for k in ("labels", "classes", "id_to_name"):
            if isinstance(old.get(k), dict):
                prev_ids = {int(x) for x in old[k]}
                break

    payload = {
        "scheme": "CTSpinoPelvic1K v5 (VerSe-native)",
        "source_of_truth": "scripts/label_scheme.py",
        "ignore_label": LS.IGNORE_LABEL,
        "id_to_name": {str(i): by_id[i] for i in sorted(by_id)},
        "name_to_id": {by_id[i]: i for i in sorted(by_id)},
        "notes": {str(k): v for k, v in NEW_NOTES.items()},
    }
    p.write_text(json.dumps(payload, indent=1))
    added = sorted(set(by_id) - prev_ids) if prev_ids else sorted(NEW_NOTES)
    print(f"  dataset_labels.json: {len(by_id)} classes"
          f"{'  (added ' + ', '.join(f'{i}={by_id[i]}' for i in added) + ')' if added else ''}")

    # README: an appended section, so hand-written prose above it is untouched
    rp = ed / "README.md"
    if rp.exists():
        txt = rp.read_text(encoding="utf-8", errors="replace")
        marker = "## Label classes added in v5"
        block = [marker, "",
                 "| id | name | note |", "|---|---|---|"]
        for i in sorted(NEW_NOTES):
            block.append(f"| {i} | `{by_id.get(i, '?')}` | {NEW_NOTES[i]} |")
        block += ["",
                  "The full id-to-name map is `dataset_labels.json`, generated from "
                  "`scripts/label_scheme.py`, which is the single source of truth.", ""]
        new = "\n".join(block)
        if marker in txt:
            head = txt.split(marker)[0]
            txt = head + new
        else:
            txt = txt.rstrip() + "\n\n" + new
        rp.write_text(txt, encoding="utf-8")
        print(f"  README.md: documented {len(NEW_NOTES)} new classes")
    else:
        print("  ! no README.md in the export dir")
    return 0


if __name__ == "__main__":
    sys.exit(main())
