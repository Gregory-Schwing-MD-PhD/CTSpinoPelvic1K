"""when_did_v5_change.py — date the edit that moved the working copy of v5 past the export.

The published tree and the working tree disagree, and the file times say which way: 1153's
published copy is stamped 2026-08-20 10:42 and the working copy 23:30 the same day, thirteen
hours later and 3,873 bytes smaller. Something rewrote the working directory after the export
was cut.

This groups every label in both trees by the hour it was last written, and lists the job logs
from the same window. A pass that rewrote part of the release shows up as a cluster of files
sharing one timestamp, and the log written beside it names what did it.

READ ONLY -- it stats files and reads log headers.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import sys
from pathlib import Path


def stamp(p: Path) -> str:
    return dt.datetime.fromtimestamp(p.stat().st_mtime).strftime("%Y-%m-%d %H:%M")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--working", default="data/v5_final")
    ap.add_argument("--published", default="data/hf_export_v5/labels")
    ap.add_argument("--logs", default="logs")
    a = ap.parse_args()

    for tag, d in (("working  (v5_final)", Path(a.working)),
                   ("published (hf_export_v5)", Path(a.published))):
        files = list(d.glob("*_label.nii.gz"))
        if not files:
            print(f"  {tag}: nothing found at {d}")
            continue
        by_hour = collections.Counter(stamp(p)[:13] for p in files)
        print(f"\n  {tag}: {len(files)} labels, written in {len(by_hour)} distinct hour(s)")
        for hour, n in sorted(by_hour.items(), key=lambda x: -x[1])[:6]:
            print(f"      {n:>4} files   {hour}:xx")

    # which jobs were running in the hours the working copy was rewritten
    wf = list(Path(a.working).glob("*_label.nii.gz"))
    hours = {stamp(p)[:13] for p in wf}
    logs = sorted(Path(a.logs).glob("*.out"), key=lambda p: p.stat().st_mtime)
    hits = [p for p in logs if stamp(p)[:13] in hours]
    print(f"\n  job logs written in the same hour(s) as the working labels: {len(hits)}")
    for p in hits[-14:]:
        head = ""
        try:
            for line in p.read_text(errors="ignore").splitlines()[:14]:
                s = line.strip()
                if s.startswith("===") or "python3" in s or "scripts/" in s:
                    head = s[:88]
                    break
        except OSError:
            pass
        print(f"      {stamp(p)}  {p.name:<34} {head}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
