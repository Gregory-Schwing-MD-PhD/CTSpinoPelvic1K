"""
review_service/deploy_v4_spaces.py — stand up the three v4 annotation Spaces.

Creates (idempotently) one Docker Space + one private review ledger PER task
(ribs / ls_nerve / iliolumbar), all reading the public dataset at @v3, sets each
Space's env vars, stores the HF token as each Space's SECRET, and uploads the
service code. Run ONE command; the token is read from the environment and is never
written to any file in the repo (so it can't be committed / scanned / auto-revoked).

    HF_TOKEN=hf_xxx python review_service/deploy_v4_spaces.py
    # optional overrides:
    #   ORG=anonymous-mlhc  ADJUDICATORS="user1,user2"  PRIVATE_SPACE=0
    #   DATASET=anonymous-mlhc/CTSpinoPelvic1K  SOURCE_REVISION=v3

Prereq: the dataset's @v3 branch must carry the CT volumes + manifest (run the
clean ship_v3 first); each Space seeds its cases from there on boot.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# task -> repo-name slug. The TASK env value the Space uses is the dict key.
TASKS = {"ribs": "ribs", "ls_nerve": "nerve", "iliolumbar": "ili"}

SPACE_README = """---
title: CTSpinoPelvic1K review ({task})
emoji: "\U0001FA90"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

Private annotation service for the CTSpinoPelvic1K **{task}** task. See
docs/annotation/ in the dataset repo. Reviewers: `reviewtool login --service <this url>`.
"""


def main() -> int:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("ERROR: set HF_TOKEN in your environment first:\n"
              "  HF_TOKEN=hf_xxx python review_service/deploy_v4_spaces.py")
        return 1

    org = os.environ.get("ORG", "anonymous-mlhc")
    dataset = os.environ.get("DATASET", f"{org}/CTSpinoPelvic1K")
    revision = os.environ.get("SOURCE_REVISION", "v3")
    adjudicators = os.environ.get("ADJUDICATORS", "")
    private_space = os.environ.get("PRIVATE_SPACE", "0") == "1"

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    for task, slug in TASKS.items():
        space_id = f"{org}/CTSpinoPelvic1K-review-{slug}"
        review_repo = f"{org}/CTSpinoPelvic1K-reviews-{slug}"
        print(f"\n=== {task} -> Space {space_id} | ledger {review_repo} ===")

        # 1) private review ledger (auto-created on first write too; make it now)
        api.create_repo(repo_id=review_repo, repo_type="dataset",
                        private=True, exist_ok=True)

        # 2) the Space
        api.create_repo(repo_id=space_id, repo_type="space", space_sdk="docker",
                        private=private_space, exist_ok=True)

        # 3) env vars (public) + token (secret) — set BEFORE upload so the first
        #    build boots configured.
        variables = {
            "TASK": task,
            "V2_REPO": dataset,
            "SOURCE_REVISION": revision,
            "REVIEW_REPO": review_repo,
            "TAU": os.environ.get("TAU", "0.9"),
            "IRR_MODE": os.environ.get("IRR_MODE", "per_class_min"),
        }
        if adjudicators:
            variables["ADJUDICATORS"] = adjudicators
        for k, v in variables.items():
            api.add_space_variable(repo_id=space_id, key=k, value=v)
        api.add_space_secret(repo_id=space_id, key="HF_TOKEN", value=token)
        print("  vars:", ", ".join(f"{k}={v}" for k, v in variables.items()))
        print("  secret: HF_TOKEN set (value not logged)")

        # 4) upload the service code in the layout the Dockerfile expects:
        #    /Dockerfile + /review_service/* + /scripts/review/*
        api.upload_file(path_or_fileobj=str(ROOT / "review_service" / "Dockerfile"),
                        path_in_repo="Dockerfile", repo_id=space_id,
                        repo_type="space")
        api.upload_file(path_or_fileobj=SPACE_README.format(task=task).encode(),
                        path_in_repo="README.md", repo_id=space_id,
                        repo_type="space")
        api.upload_folder(folder_path=str(ROOT / "review_service"),
                          path_in_repo="review_service", repo_id=space_id,
                          repo_type="space",
                          ignore_patterns=["__pycache__/*", "*.pyc", "Dockerfile"])
        api.upload_folder(folder_path=str(ROOT / "scripts" / "review"),
                          path_in_repo="scripts/review", repo_id=space_id,
                          repo_type="space",
                          ignore_patterns=["__pycache__/*", "*.pyc"])
        print(f"  uploaded code -> https://huggingface.co/spaces/{space_id}")

    print("\nAll three Spaces created/updated. They build, then seed from "
          f"{dataset}@{revision} on boot. URLs:")
    for slug in TASKS.values():
        print(f"  https://{org.lower()}-ctspinopelvic1k-review-{slug}.hf.space")
    if not adjudicators:
        print("\nNote: ADJUDICATORS empty — set faculty HF usernames later via each "
              "Space's Variables, or re-run with ADJUDICATORS=\"user1,user2\".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
