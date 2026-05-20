"""
reviewtool/cli.py — annotator-facing CLI.

Commands:
  reviewtool login   --service URL --key KEY
  reviewtool next    [--workdir DIR] [--itksnap itksnap]   # claim→edit→submit
  reviewtool adjudicate [...]                               # disagreements
  reviewtool status

All HF download/upload is hidden: CT + pseudo come straight from the public
v2 repo; the corrected label is uploaded *through the Space*, so the
annotator never holds the dataset token — only their reviewer API key.

The decision logic (`build_submission`) is pure and unit-tested; everything
else is I/O glue (subprocess + HTTP).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import subprocess
from pathlib import Path
from typing import Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
from review import diff, labels_descriptor  # noqa: E402

CONFIG = Path.home() / ".reviewtool" / "config.json"


# ── pure core (tested) ───────────────────────────────────────────────────────

def build_submission(pseudo, edited, region: str,
                     source_label_sha256: str) -> Tuple[str, dict]:
    """Decide accept vs corrected from the pseudo→edited diff and build the
    review record. Pure (arrays in, dict out)."""
    d = diff.label_diff(pseudo, edited)
    decision = "accept" if d["n_voxels_changed"] == 0 else "corrected"
    record = {
        "decision": decision,
        "region_reviewed": region,
        "source_label_sha256": source_label_sha256,
        "diff": d,
    }
    return decision, record


# ── config / http ────────────────────────────────────────────────────────────

def _cfg() -> dict:
    if not CONFIG.exists():
        sys.exit("not logged in — run: reviewtool login --service URL --key KEY")
    return json.loads(CONFIG.read_text())


def _api():
    import requests          # local import so `login` works before install note
    cfg = _cfg()
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {cfg['api_key']}"
    return s, cfg["service_url"].rstrip("/")


def _fetch(job: dict, filename: str, dest: Path) -> Path:
    """Download a dataset file via huggingface_hub so it works for public,
    gated, OR private v2 repos — using the reviewer's `huggingface-cli
    login` token (or HF_TOKEN env). No dataset write access needed."""
    import shutil
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        repo_id=job["v2_repo"], repo_type="dataset",
        filename=filename, revision=job.get("source_revision"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    return dest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path):
    import numpy as np
    import nibabel as nib
    return np.asarray(nib.load(str(path)).dataobj)


def _launch_itksnap(itksnap: str, ct: Path, seg: Path, labels: Path) -> None:
    print(f"\nOpening ITK-SNAP — edit the segmentation, Save Segmentation to:\n"
          f"  {seg}\nthen quit ITK-SNAP to continue.\n")
    try:
        subprocess.run([itksnap, "-g", str(ct), "-s", str(seg),
                        "-l", str(labels)], check=False)
    except FileNotFoundError:
        sys.exit(f"'{itksnap}' not found on PATH — install ITK-SNAP or pass "
                 f"--itksnap /path/to/itksnap")


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_login(a):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    CONFIG.write_text(json.dumps(
        {"service_url": a.service, "api_key": a.key}, indent=2))
    print(f"saved {CONFIG}")


def _claim(s, base, path="/claim"):
    r = s.post(base + path, timeout=60)
    if r.status_code == 204:
        return None
    r.raise_for_status()
    return r.json()


def cmd_next(a):
    s, base = _api()
    job = _claim(s, base)
    if job is None:
        print("nothing to claim — all cases assigned/done.")
        return
    work = Path(a.workdir) / job["case_id"]
    work.mkdir(parents=True, exist_ok=True)
    ct = _fetch(job, job["ct_file"], work / "ct.nii.gz")
    pseudo = _fetch(job, job["label_file"], work / "pseudo.nii.gz")
    seg = work / "seg.nii.gz"
    seg.write_bytes(pseudo.read_bytes())                # edit a copy
    labels = work / "labels.txt"
    labels.write_text(job.get("labels_descriptor")
                      or labels_descriptor.descriptor_text())

    print(f"case {job['case_id']}  (review the {job['region_to_review']} region)")
    _launch_itksnap(a.itksnap, ct, seg, labels)

    decision, record = build_submission(
        _load(pseudo), _load(seg), job["region_to_review"], _sha256(pseudo))
    print(f"decision={decision}  voxels_changed={record['diff']['n_voxels_changed']}")

    files = {"label": ("label.nii.gz", open(seg, "rb"), "application/gzip")}
    data = {"claim_token": job["claim_token"], "record": json.dumps(record)}
    r = s.post(base + "/submit", data=data, files=files, timeout=300)
    r.raise_for_status()
    print("submitted ->", r.json())


def cmd_adjudicate(a):
    s, base = _api()
    job = _claim(s, base, "/adjudication/next")
    if job is None:
        print("nothing to adjudicate.")
        return
    work = Path(a.workdir) / (job["case_id"] + "__adj")
    work.mkdir(parents=True, exist_ok=True)
    ct = _fetch(job, job["ct_file"], work / "ct.nii.gz")
    pseudo = _fetch(job, job["label_file"], work / "pseudo.nii.gz")
    seg = work / "seg.nii.gz"
    seg.write_bytes(pseudo.read_bytes())
    labels = work / "labels.txt"
    labels.write_text(job.get("labels_descriptor")
                      or labels_descriptor.descriptor_text())
    print(f"ADJUDICATE {job['case_id']}  IRR={job.get('irr')}\n"
          f"two reviewers disagreed on the {job['region_to_review']} region; "
          f"produce the deciding label.")
    _launch_itksnap(a.itksnap, ct, seg, labels)

    files = {"label": ("label.nii.gz", open(seg, "rb"), "application/gzip")}
    data = {"claim_token": job["claim_token"], "decision": "corrected",
            "notes": a.notes}
    r = s.post(base + "/adjudicate", data=data, files=files, timeout=300)
    r.raise_for_status()
    print("adjudicated ->", r.json())


def cmd_status(a):
    s, base = _api()
    r = s.get(base + "/status", timeout=60)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="reviewtool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("login"); p.add_argument("--service", required=True)
    p.add_argument("--key", required=True); p.set_defaults(fn=cmd_login)

    for name, fn in (("next", cmd_next), ("adjudicate", cmd_adjudicate)):
        p = sub.add_parser(name)
        p.add_argument("--workdir", default=str(Path.home() / ".reviewtool" / "work"))
        p.add_argument("--itksnap", default="itksnap")
        if name == "adjudicate":
            p.add_argument("--notes", default="")
        p.set_defaults(fn=fn)

    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)

    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
