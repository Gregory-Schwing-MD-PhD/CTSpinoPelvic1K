"""push_osc_release.py -- publish the release tree to the OpenSpineConsortium namespace.

upload_large_folder, not upload_folder: 802 CT volumes come to ~195 GB, and this needs to
survive a dropped connection without restarting the transfer. It keeps its own progress
cache under the folder, so a re-run resumes rather than re-uploads.

The token is read from a file (default ~/.hf_osc_token, mode 0600) so it never appears in a
command line, a SLURM --export, or a job log.

    python scripts/push_osc_release.py --folder data/hf_export_osc \
        --repo OpenSpineConsortium/CTSpinoPelvic1K --workers 8
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import HfApi


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--folder", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--token-file", default=os.path.expanduser("~/.hf_osc_token"))
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    token = Path(a.token_file).read_text().strip()
    if not token:
        raise SystemExit(f"empty token file: {a.token_file}")

    api = HfApi(token=token)
    who = api.whoami()["name"]
    folder = Path(a.folder)
    n_ct = len(list((folder / "ct").glob("*.nii.gz")))
    n_lab = len(list((folder / "labels").glob("*.nii.gz")))
    print(f"  as       : {who}")
    print(f"  folder   : {folder}  ({n_ct} ct, {n_lab} labels)", flush=True)
    print(f"  repo     : {a.repo}", flush=True)
    if n_ct != 802 or n_lab != 802:
        raise SystemExit(f"expected 802/802, found {n_ct}/{n_lab}")

    api.upload_large_folder(
        folder_path=str(folder),
        repo_id=a.repo,
        repo_type="dataset",
        num_workers=a.workers,
        print_report=True,
    )
    print("  upload call returned", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
