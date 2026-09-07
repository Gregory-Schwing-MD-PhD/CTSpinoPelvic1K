"""hf/push_v9.py -- push the v9 labels and metadata to a Hugging Face dataset mirror.

    python hf/push_v9.py --repo OpenSpineConsortium/CTSpinoPelvic1K --token-file ~/.hf_osc_token
    python hf/push_v9.py --repo anonymous-mlhc/CTSpinoPelvic1K --token-file ~/.hf_org_token

Uploads the 802 remapped label volumes (data/zenodo_deposit/labels -> labels/) and the
metadata files that changed with them: manifest.json, dataset_labels.json, KNOWN_ISSUES.md.
The CT volumes and the splits are unchanged and are not touched. The dataset card
(README.md on the hub) is NEVER uploaded from here: the Zenodo README is a different
document, and pushing it once overwrote both cards. Edit the card on the hub or with
--card <file> deliberately.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parents[1]
DEPOSIT = ROOT / "data" / "zenodo_deposit"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token-file", default=None)
    ap.add_argument("--labels", default=str(DEPOSIT / "labels"))
    ap.add_argument("--card", default=None, help="explicit dataset card to upload as README.md")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    tok = None
    if a.token_file:
        tok = Path(os.path.expanduser(a.token_file)).read_text().strip()
    api = HfApi(token=tok)
    files = ["manifest.json", "dataset_labels.json", "KNOWN_ISSUES.md"]
    labels = Path(a.labels)
    n = len(list(labels.glob("*_label.nii.gz")))
    print(f"{a.repo}: {n} label volumes from {labels}; files {files}")
    if a.dry_run:
        return 0
    api.upload_folder(repo_id=a.repo, repo_type="dataset", folder_path=str(labels),
                      path_in_repo="labels", allow_patterns=["*_label.nii.gz"],
                      commit_message="v10: thirteen rib identifiers per side (34-46, 47-59); lumbar ribs 60-61, hardware 62-68")
    for f in files:
        api.upload_file(repo_id=a.repo, repo_type="dataset", path_or_fileobj=str(DEPOSIT / f),
                        path_in_repo=f, commit_message=f"v10: {f}")
    if a.card:
        api.upload_file(repo_id=a.repo, repo_type="dataset", path_or_fileobj=a.card,
                        path_in_repo="README.md", commit_message="v10: dataset card")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
