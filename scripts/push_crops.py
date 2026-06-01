"""
push_crops.py — upload the review crops (+ crops_index.json) to the v2 dataset
repo under `crops/`, so the review Space triages to exactly those cases and
serves the small crops to reviewers.

  HF_TOKEN=hf_xxx python scripts/push_crops.py \
      --crops data/review_crops --repo_id ORG/CTSpinoPelvic1K --revision v2

After this, restart the review Space: it reads crops/crops_index.json, seeds
ONLY the flagged cases, and `reviewtool next` serves the crop (few MB) instead
of the full 200 MB volume.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crops", required=True, type=Path,
                    help="local crops dir (must contain crops_index.json)")
    ap.add_argument("--repo_id", required=True, help="v2 dataset repo, e.g. ORG/Name")
    ap.add_argument("--revision", default="v2", help="branch/revision (default v2)")
    ap.add_argument("--token", default=os.environ.get("HF_TOKEN"))
    args = ap.parse_args()

    if not (args.crops / "crops_index.json").exists():
        sys.exit(f"no crops_index.json in {args.crops} — run export_review_crops.py first")
    if not args.token:
        sys.exit("no HF token — set HF_TOKEN or pass --token")

    from huggingface_hub import upload_folder
    print(f"uploading {args.crops} -> {args.repo_id}@{args.revision}:crops/ ...")
    upload_folder(folder_path=str(args.crops), path_in_repo="crops",
                  repo_id=args.repo_id, repo_type="dataset", revision=args.revision,
                  token=args.token, commit_message="add review crops + crops_index")
    print("done. Restart the review Space to triage to the flagged worklist.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
