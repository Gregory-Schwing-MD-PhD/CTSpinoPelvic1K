"""zenodo/mirror_ct_to_hf.py — publish the release CTs to a personal HuggingFace repo.

WHY THIS IS NEEDED BEFORE THE DEPOSIT CAN LINK TO ONE. The personal repo
`gregoryschwingmdphd/CTSpinoPelvic1K` already exists and is public, but it is the older
export: of the 802 records in this release only 529 are present, 273 are absent entirely,
and the filenames follow a different convention (`0001_spine_ct.nii.gz` and
`0001_pelvic_ct.nii.gz` where the release has one `NNNN_ct.nii.gz` per record). Pointing
users at it as "the CT images" would be wrong for a third of the dataset and silently wrong
for the rest, since the same case id means a different volume there.

RUN THIS WHERE THE DATA IS, WHICH IS NOT A LAPTOP. The CTs are about 193 GB. Run it on the
grid against the export directory rather than pulling them down and pushing them back.

    export HF_TOKEN=...                       # a token that can WRITE to the target repo
    python mirror_ct_to_hf.py --ct data/hf_export_v5/ct \\
        --manifest data/hf_export_v5/manifest.json \\
        --repo gregoryschwingmdphd/CTSpinoPelvic1K --revision v5

IT UPLOADS TO A BRANCH, NOT TO main. The existing main holds the older export and other
people may already depend on it; a v5 branch adds the release without disturbing that, and
the deposit links to the branch explicitly.

It verifies before uploading that every record in the manifest has a CT and that no extra
CTs are present, because a mirror that is quietly missing 273 files is exactly the situation
this exists to prevent.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def rec_id(r):
    return Path(str(r.get("label_file", ""))).name.split("_")[0]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ct", required=True, help="directory of NNNN_ct.nii.gz")
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--repo", default="gregoryschwingmdphd/CTSpinoPelvic1K")
    ap.add_argument("--revision", default="v5")
    ap.add_argument("--check", action="store_true", help="verify only, upload nothing")
    a = ap.parse_args()

    tok = os.environ.get("HF_TOKEN")
    if not tok and not a.check:
        print("  ! HF_TOKEN is not set, and it must be able to WRITE to the target repo.")
        return 2

    man = json.loads(Path(a.manifest).read_text(encoding="utf-8"))
    recs = man if isinstance(man, list) else man.get("records", list(man.values()))
    want = {rec_id(r) for r in recs}

    ctdir = Path(a.ct)
    have = {p.name.split("_")[0] for p in ctdir.glob("*_ct.nii.gz")}
    missing, extra = sorted(want - have), sorted(have - want)
    print(f"  manifest records: {len(want)}")
    print(f"  CT files present: {len(have)}")
    if missing:
        print(f"  ! {len(missing)} record(s) have no CT: {missing[:8]}")
    if extra:
        print(f"  ! {len(extra)} CT file(s) are not in the manifest: {extra[:8]}")
    if missing:
        print("  Refusing: a mirror that is missing records is worse than no mirror,")
        print("  because the case ids still resolve and quietly return the wrong thing.")
        return 1

    total = sum(p.stat().st_size for p in ctdir.glob("*_ct.nii.gz"))
    print(f"  {len(have)} file(s), {total / 1e9:.1f} GB")
    if a.check:
        print("  --check: nothing uploaded")
        return 0

    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    api.create_repo(a.repo, repo_type="dataset", exist_ok=True)
    try:
        api.create_branch(a.repo, repo_type="dataset", branch=a.revision, exist_ok=True)
    except TypeError:                       # older hub without exist_ok
        try:
            api.create_branch(a.repo, repo_type="dataset", branch=a.revision)
        except Exception:                                             # noqa: BLE001
            pass

    print(f"  uploading to {a.repo}@{a.revision} ...", flush=True)
    api.upload_folder(
        folder_path=str(ctdir),
        path_in_repo="ct",
        repo_id=a.repo,
        repo_type="dataset",
        revision=a.revision,
        allow_patterns=["*_ct.nii.gz"],
        commit_message=(
            "the CT images for the 802-record release\n\n"
            "One volume per record, named to match the released labels, so a case id means\n"
            "the same acquisition here and in the label deposit. The previous export on main\n"
            "covers 529 of these records under a different naming convention and is left\n"
            "untouched on its own branch."
        ),
    )
    print(f"  done: https://huggingface.co/datasets/{a.repo}/tree/{a.revision}/ct")
    return 0


if __name__ == "__main__":
    sys.exit(main())
