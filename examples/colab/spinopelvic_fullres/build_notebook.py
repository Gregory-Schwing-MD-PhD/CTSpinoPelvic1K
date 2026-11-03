#!/usr/bin/env python3
"""Build the Colab notebook wrapper for the full-resolution fresh NRRD run."""

from __future__ import annotations

import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
RUNNER = (HERE / "spinopelvic_fullres_nrrd_colab.py").read_text()


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.strip("\n").splitlines()],
    }


cells = [
    code(
        r'''
# @title 1. GPU and workspace check
from pathlib import Path
import os, platform, shutil, subprocess, sys

print("Python:", platform.python_version())
try:
    print(subprocess.check_output(["nvidia-smi"], text=True))
except Exception as e:
    raise RuntimeError("Select Runtime > Change runtime type > GPU, then rerun.") from e

ROOT = Path("/content/spinopelvic_fullres_fresh")
UPLOAD = ROOT / "00_uploaded_inputs"
UPLOAD.mkdir(parents=True, exist_ok=True)
print("Upload folder:", UPLOAD)
print("Free /content GiB:", round(shutil.disk_usage("/content").free / 2**30, 2))
'''
    ),
    code(
        r'''
# @title 2. Upload fresh NRRD/NIfTI files
from google.colab import files
from pathlib import Path
import shutil

UPLOAD = Path("/content/spinopelvic_fullres_fresh/00_uploaded_inputs")
UPLOAD.mkdir(parents=True, exist_ok=True)
uploaded = files.upload()
for name in uploaded:
    src = Path("/content") / name
    dst = UPLOAD / name
    if src.exists() and src != dst:
        shutil.move(str(src), str(dst))
print("Files ready:")
for p in sorted(UPLOAD.glob("*")):
    print(p, p.stat().st_size)
'''
    ),
    code(
        '''
# @title 3. Save the full runner script
from pathlib import Path

RUNNER = __RUNNER_LITERAL__
path = Path("/content/spinopelvic_fullres_fresh/spinopelvic_fullres_nrrd_colab.py")
path.write_text(RUNNER)
print("Wrote:", path)
'''.replace("__RUNNER_LITERAL__", json.dumps(RUNNER))
    ).copy(),
    code(
        r'''
# @title 4. Run full-resolution spinopelvic segmentation
import getpass, os, runpy, sys

token = getpass.getpass("Hugging Face token, or Enter if public access works: ").strip()
if token:
    os.environ["HF_TOKEN"] = token

sys.argv = ["/content/spinopelvic_fullres_fresh/spinopelvic_fullres_nrrd_colab.py"]
runpy.run_path(sys.argv[0], run_name="__main__")
'''
    ),
    code(
        r'''
# @title 5. Download final ZIP again if needed
from google.colab import files
from pathlib import Path

zip_path = Path("/content/spinopelvic_fullres_fresh/09_export/Spinopelvic_Fullres_Fresh_RESULTS.zip")
print(zip_path, zip_path.exists(), zip_path.stat().st_size if zip_path.exists() else None)
files.download(str(zip_path))
'''
    ),
]

nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
        "language_info": {"name": "python"},
        "accelerator": "GPU",
    },
    "cells": cells,
}

(HERE / "Spinopelvic_Fullres_Fresh_GPU.ipynb").write_text(json.dumps(nb, indent=1))
print(HERE / "Spinopelvic_Fullres_Fresh_GPU.ipynb")
