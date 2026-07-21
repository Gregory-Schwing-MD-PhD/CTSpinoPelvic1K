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

# name -> per-Space deploy config. The rib Space is the v4 in-place CORRECTION of the
# QC-flagged duplicate ribs (TASK=rib_fix reads @v4 + seeds rib_worklist.json; the
# server-side CHECK=ribs gate rejects any submit that still has a duplicate/split rib).
# The other two are the v3 additive overlays (no server gate).
TASKS = {
    "ribs":       {"slug": "ribs",  "task": "rib_fix",    "revision": "v4", "check": "ribs"},
    "spine_extend": {"slug": "spine", "task": "spine_extend", "revision": "v4", "check": "spine_extend"},
    "class_mixing": {"slug": "classfix", "task": "class_mixing", "revision": "v4", "check": "class_mixing"},
    "ls_nerve":   {"slug": "nerve", "task": "ls_nerve",   "revision": "v3", "check": "none"},
    "iliolumbar": {"slug": "ili",   "task": "iliolumbar", "revision": "v3", "check": "none"},
}

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
    only = {t.strip() for t in os.environ.get("ONLY", "").replace(",", " ").split() if t.strip()}

    from huggingface_hub import HfApi
    api = HfApi(token=token)

    for name, cfg in TASKS.items():
        if only and name not in only:                # ONLY=ribs -> deploy just that Space
            continue
        slug, task = cfg["slug"], cfg["task"]
        space_id = f"{org}/CTSpinoPelvic1K-review-{slug}"
        review_repo = f"{org}/CTSpinoPelvic1K-reviews-{slug}"
        print(f"\n=== {name} (TASK={task}@{cfg['revision']} check={cfg['check']}) "
              f"-> Space {space_id} | ledger {review_repo} ===")

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
            "SOURCE_REVISION": cfg["revision"],
            "CHECK": cfg["check"],                  # server-side QC gate on submit
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
        #    /Dockerfile + /review_service/* + /scripts/review/* + /scripts/<qc modules>
        api.upload_file(path_or_fileobj=str(ROOT / "review_service" / "Dockerfile"),
                        path_in_repo="Dockerfile", repo_id=space_id,
                        repo_type="space")
        api.upload_file(path_or_fileobj=SPACE_README.format(task=name).encode(),
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
        # server-side QC gate deps: the CHECK=ribs gate imports these from scripts/
        for fn in ("review_anatomy_qc.py", "label_scheme.py"):
            api.upload_file(path_or_fileobj=str(ROOT / "scripts" / fn),
                            path_in_repo=f"scripts/{fn}", repo_id=space_id,
                            repo_type="space")
        print(f"  uploaded code -> https://huggingface.co/spaces/{space_id}")

    print(f"\nAll {len(TASKS)} Spaces created/updated. They build, then seed on boot "
          f"(ribs from {dataset}@v4 via rib_worklist.json; overlays from @v3). URLs:")
    for cfg in TASKS.values():
        print(f"  https://{org.lower()}-ctspinopelvic1k-review-{cfg['slug']}.hf.space")
    if not adjudicators:
        print("\nNote: ADJUDICATORS empty — set faculty HF usernames later via each "
              "Space's Variables, or re-run with ADJUDICATORS=\"user1,user2\".")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
