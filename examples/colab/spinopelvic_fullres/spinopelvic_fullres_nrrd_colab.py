#!/usr/bin/env python3
"""Full-resolution spinopelvic-seg Colab runner for uploaded NRRD/NIfTI CT files.

Research-only workflow. Outputs are segmentation candidates for human review, not
clinical diagnoses.
"""

from __future__ import annotations

import argparse
import csv
import getpass
import hashlib
import json
import os
import shutil
import subprocess
import sys
import textwrap
import time
import zipfile
from pathlib import Path


RUN_ROOT = Path("/content/spinopelvic_fullres_fresh")
INPUT_ROOT = RUN_ROOT / "00_uploaded_inputs"
NIFTI_ROOT = RUN_ROOT / "01_converted_nifti"
NNINPUT_ROOT = RUN_ROOT / "02_nninput"
MODEL_ROOT = RUN_ROOT / "03_model"
NNUNET_RAW = RUN_ROOT / "04_nnunet_raw"
NNUNET_PREPROCESSED = RUN_ROOT / "05_nnunet_preprocessed"
NNUNET_RESULTS = RUN_ROOT / "06_nnunet_results"
PRED_ROOT = RUN_ROOT / "07_predictions"
REPORT_ROOT = RUN_ROOT / "08_reports"
EXPORT_ROOT = RUN_ROOT / "09_export"
LOG_ROOT = RUN_ROOT / "logs"
SOURCE_ROOT = RUN_ROOT / "source" / "spinopelvic-seg"

MODEL_REPO = "anonymous-neurips-ED/spinopelvic-seg-checkpoints"
CODE_REPO = "https://github.com/anonymous-mlhc/spinopelvic-seg.git"
DATASET_NAME = "Dataset803_SpineSurgCTFullMerged"

LABELS = {
    0: "background",
    1: "L1",
    2: "L2",
    3: "L3",
    4: "L4",
    5: "last_lumbar",
    6: "sacrum",
    7: "left_hip",
    8: "right_hip",
    9: "ignore_or_auxiliary",
}

COLORS = {
    1: "#e41a1c",
    2: "#377eb8",
    3: "#4daf4a",
    4: "#984ea3",
    5: "#ff7f00",
    6: "#ffff33",
    7: "#a65628",
    8: "#f781bf",
    9: "#999999",
}

LIMITS = (
    "Research-only spinopelvic segmentation candidate. This does not diagnose "
    "LSTV/Castellvi class or replace radiology review. Verify numbering and "
    "left/right laterality anatomically, especially for cropped or nonstandard CT."
)


def run(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> None:
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env, cwd=str(cwd) if cwd else None)


def capture(cmd: list[str], *, env: dict[str, str] | None = None, cwd: Path | None = None) -> str:
    print("+", " ".join(cmd), flush=True)
    return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT, env=env, cwd=str(cwd) if cwd else None)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def case_id(index: int) -> str:
    return f"CASE_{index:03d}"


def safe_name(path: Path) -> str:
    clean = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in path.stem)
    return clean[:80] or "ct"


def prepare_dirs() -> None:
    for path in (
        RUN_ROOT,
        INPUT_ROOT,
        NIFTI_ROOT,
        NNINPUT_ROOT,
        MODEL_ROOT,
        NNUNET_RAW,
        NNUNET_PREPROCESSED,
        NNUNET_RESULTS,
        PRED_ROOT,
        REPORT_ROOT,
        EXPORT_ROOT,
        LOG_ROOT,
        SOURCE_ROOT.parent,
    ):
        path.mkdir(parents=True, exist_ok=True)


def install_environment(skip_install: bool) -> None:
    if skip_install:
        return
    run([sys.executable, "-m", "pip", "install", "-q", "--upgrade", "pip"])
    run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "nnunetv2",
            "huggingface_hub",
            "SimpleITK",
            "nibabel",
            "pydicom",
            "matplotlib",
            "numpy",
            "scipy",
            "tqdm",
            "gdown",
        ]
    )


def clone_source() -> None:
    if SOURCE_ROOT.exists() and any(SOURCE_ROOT.iterdir()):
        print(f"Using existing source checkout: {SOURCE_ROOT}", flush=True)
    else:
        run(["git", "clone", CODE_REPO, str(SOURCE_ROOT)])
    req = SOURCE_ROOT / "requirements.txt"
    if req.exists():
        run([sys.executable, "-m", "pip", "install", "-q", "-r", str(req)])


def login_hf(token: str | None) -> None:
    if not token:
        token = os.environ.get("HF_TOKEN")
    if token is None and sys.stdin.isatty():
        token = getpass.getpass("Hugging Face token, or Enter if public access works: ").strip()
    if token:
        from huggingface_hub import login

        login(token=token, add_to_git_credential=False)


def download_model() -> None:
    from huggingface_hub import snapshot_download

    print(f"Downloading/checking model: {MODEL_REPO}", flush=True)
    snapshot_download(repo_id=MODEL_REPO, repo_type="model", local_dir=str(MODEL_ROOT))
    target = NNUNET_RESULTS / DATASET_NAME
    source = MODEL_ROOT / DATASET_NAME
    if target.exists():
        shutil.rmtree(target)
    if source.exists():
        shutil.copytree(source, target)
    else:
        shutil.copytree(MODEL_ROOT, target)
    (RUN_ROOT / "model_manifest.json").write_text(
        json.dumps(
            {
                "repo": MODEL_REPO,
                "dataset_target": str(target),
                "files": {str(p.relative_to(MODEL_ROOT)): sha256(p) for p in MODEL_ROOT.rglob("*") if p.is_file()},
            },
            indent=2,
        )
    )


def download_drive_folder(url: str | None, folder_id: str | None) -> None:
    if not url and not folder_id:
        return
    import gdown

    target = INPUT_ROOT / "drive_folder"
    target.mkdir(parents=True, exist_ok=True)
    if list(target.glob("*.nrrd")) or list(target.glob("*.nii")) or list(target.glob("*.nii.gz")):
        print(f"Reusing existing CT files in {target}", flush=True)
        return
    if url:
        print(f"Downloading CT folder from Google Drive into {target}", flush=True)
        gdown.download_folder(url=url, output=str(target), quiet=False, use_cookies=False)
    else:
        folder_url = f"https://drive.google.com/drive/folders/{folder_id}"
        print(f"Downloading CT folder from Google Drive into {target}", flush=True)
        gdown.download_folder(url=folder_url, output=str(target), quiet=False, use_cookies=False)


def find_inputs(explicit: list[str]) -> list[Path]:
    paths = [Path(x) for x in explicit]
    if not paths:
        drive_root = INPUT_ROOT / "drive_folder"
        roots = [drive_root] if drive_root.exists() and any(drive_root.rglob("*")) else [INPUT_ROOT, Path("/content"), Path("/content/drive/MyDrive")]
        for root in roots:
            if root.exists():
                for pattern in ("*.nrrd", "*.nii", "*.nii.gz"):
                    paths.extend(root.rglob(pattern))
    seen: set[Path] = set()
    out: list[Path] = []
    for path in paths:
        path = path.expanduser()
        if path.exists() and path.is_file() and path.resolve() not in seen:
            seen.add(path.resolve())
            out.append(path)
    if not out:
        raise FileNotFoundError("No NRRD/NIfTI files found. Upload files into /content/spinopelvic_fullres_fresh/00_uploaded_inputs.")
    return sorted(out, key=lambda p: p.name)


def convert_inputs(inputs: list[Path]) -> list[dict[str, object]]:
    import numpy as np
    import SimpleITK as sitk

    shutil.rmtree(NNINPUT_ROOT, ignore_errors=True)
    NNINPUT_ROOT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for index, source in enumerate(inputs):
        cid = case_id(index)
        out = NIFTI_ROOT / f"{cid}_{safe_name(source)}.nii.gz"
        nn_case = NNINPUT_ROOT / f"{cid}_0000.nii.gz"
        img = sitk.ReadImage(str(source))
        arr = sitk.GetArrayViewFromImage(img)
        if img.GetDimension() != 3 or img.GetNumberOfComponentsPerPixel() != 1:
            raise ValueError(f"{source} is not a scalar 3D CT")
        if not np.isfinite(arr).all() or np.ptp(arr) == 0:
            raise ValueError(f"{source} is empty/constant/non-finite")
        for key in img.GetMetaDataKeys():
            img.EraseMetaData(key)
        sitk.WriteImage(img, str(out), True)
        shutil.copy2(out, nn_case)
        rows.append(
            {
                "case_id": cid,
                "source": str(source),
                "converted": str(out),
                "nnunet_input": str(nn_case),
                "bytes": source.stat().st_size,
                "sha256": sha256(source),
                "size_xyz": list(img.GetSize()),
                "spacing_xyz_mm": list(img.GetSpacing()),
                "pixel_type": img.GetPixelIDTypeAsString(),
                "min": float(arr.min()),
                "max": float(arr.max()),
                "nonzero_voxels": int(np.count_nonzero(arr)),
            }
        )
    (RUN_ROOT / "input_manifest.json").write_text(json.dumps(rows, indent=2))
    with (RUN_ROOT / "input_manifest.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def nnunet_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(
        {
            "nnUNet_raw": str(NNUNET_RAW),
            "nnUNet_preprocessed": str(NNUNET_PREPROCESSED),
            "nnUNet_results": str(NNUNET_RESULTS),
            "MPLCONFIGDIR": str(RUN_ROOT / "matplotlib"),
            "nnUNet_extTrainer": str(SOURCE_ROOT),
        }
    )
    Path(env["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
    return env


def run_prediction(force: bool, device: str) -> None:
    base_cmd = [
        "nnUNetv2_predict", "-d", "803", "-c", "3d_fullres",
        "-p", "nnUNetResEncUNetPlans_100G",
        "-tr", "nnUNetTrainerWandB_500ep_LSTVOversample",
        "-f", "0", "1", "2", "3", "4", "-chk", "checkpoint_best.pth",
        "-npp", "1", "-nps", "1", "-device", device,
        "--disable_tta", "--continue_prediction",
    ]
    log_path = LOG_ROOT / "nnunet_predict.log"
    env = nnunet_env()
    failures = []
    with log_path.open("a") as log:
        for case_file in sorted(NNINPUT_ROOT.glob("CASE_*_0000.nii.gz")):
            cid = case_file.name.split("_0000")[0]
            out_file = PRED_ROOT / f"{cid}.nii.gz"
            if out_file.exists() and not force:
                print(f"Reusing prediction: {out_file}", flush=True)
                continue
            temp_in = RUN_ROOT / "single_case_in" / cid
            shutil.rmtree(temp_in, ignore_errors=True)
            temp_in.mkdir(parents=True, exist_ok=True)
            shutil.copy2(case_file, temp_in / case_file.name)
            cmd = base_cmd + ["-i", str(temp_in), "-o", str(PRED_ROOT)]
            log.write("\n\n### " + cid + "\n" + " ".join(cmd) + "\n")
            log.flush()
            print(f"Predicting {cid}", flush=True)
            proc = subprocess.run(cmd, env=env, text=True, stdout=log, stderr=subprocess.STDOUT)
            if proc.returncode != 0:
                failures.append({"case_id": cid, "returncode": proc.returncode})
                print(f"FAILED {cid}; continuing", flush=True)
            else:
                print(f"Done {cid}", flush=True)
    if failures:
        (LOG_ROOT / "prediction_failures.json").write_text(json.dumps(failures, indent=2))
        print(f"{len(failures)} nnUNet case(s) failed; rendering successful cases anyway", flush=True)


def render_reports(manifest: list[dict[str, object]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import nibabel as nib
    import numpy as np
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from scipy import ndimage

    colors = ["#000000"] + [COLORS.get(i, "#ffffff") for i in range(1, max(LABELS) + 1)]
    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors))
    all_rows: list[dict[str, object]] = []

    for row in manifest:
        cid = str(row["case_id"])
        ct_path = Path(str(row["converted"]))
        pred_path = PRED_ROOT / f"{cid}.nii.gz"
        if not pred_path.exists():
            pred_path = PRED_ROOT / f"{cid}_0000.nii.gz"
        if not pred_path.exists():
            print(f"Skipping {cid}: missing prediction", flush=True)
            continue
        case_dir = REPORT_ROOT / cid
        img_dir = case_dir / "overlays"
        mask_dir = case_dir / "masks"
        img_dir.mkdir(parents=True, exist_ok=True)
        mask_dir.mkdir(parents=True, exist_ok=True)

        ct_img = nib.load(str(ct_path))
        pred_img = nib.load(str(pred_path))
        if ct_img.shape != pred_img.shape:
            raise ValueError(f"{cid} shape mismatch: CT {ct_img.shape} vs prediction {pred_img.shape}")
        ct = np.asanyarray(nib.as_closest_canonical(ct_img).dataobj)
        pred_c = nib.as_closest_canonical(pred_img)
        pred = np.asanyarray(pred_c.dataobj).astype(np.int16)
        present = [int(x) for x in np.unique(pred) if int(x) > 0]
        voxel_ml = abs(np.linalg.det(pred_img.affine[:3, :3])) / 1000
        stats_rows = []
        for lab in present:
            mask = pred == lab
            count = int(mask.sum())
            if not count:
                continue
            nib.save(
                nib.Nifti1Image(mask.astype("uint8"), pred_c.affine, pred_c.header),
                mask_dir / f"{lab:02d}_{LABELS.get(lab, 'label')}.nii.gz",
            )
            center = ndimage.center_of_mass(mask)
            stats = {
                "case_id": cid,
                "label_id": lab,
                "label": LABELS.get(lab, "unknown"),
                "voxels": count,
                "volume_ml": round(count * voxel_ml, 3),
                "tags": "prediction; requires_human_review",
            }
            stats_rows.append(stats)
            all_rows.append(stats)

        planes = [("sagittal", 0), ("coronal", 1), ("axial", 2)]
        for plane, axis in planes:
            occupancy = np.count_nonzero(pred, axis=tuple(a for a in range(3) if a != axis))
            index = int(occupancy.argmax()) if occupancy.max() else pred.shape[axis] // 2
            image = np.take(ct, index, axis=axis).T
            mask = np.take(pred, index, axis=axis).T
            fig, ax = plt.subplots(figsize=(9, 9), layout="constrained")
            ax.imshow(image, cmap="gray", vmin=-500, vmax=1300, origin="lower")
            ax.imshow(np.ma.masked_equal(mask, 0), cmap=cmap, norm=norm, alpha=0.48, origin="lower", interpolation="nearest")
            for lab in sorted(np.unique(mask)):
                if lab <= 0:
                    continue
                yy, xx = ndimage.center_of_mass(mask == lab)
                ax.text(
                    xx,
                    yy,
                    LABELS.get(int(lab), str(int(lab))),
                    color="white",
                    fontsize=9,
                    ha="center",
                    bbox={"facecolor": COLORS.get(int(lab), "#333333"), "alpha": 0.85, "edgecolor": "white", "pad": 2},
                )
            ax.set_title(f"{cid} {plane} overlay slice {index}")
            ax.axis("off")
            fig.savefig(img_dir / f"{cid}_{plane}_ct_overlay.png", dpi=180)
            plt.close(fig)

        (case_dir / "case_report.json").write_text(
            json.dumps(
                {
                    "case_id": cid,
                    "source": row["source"],
                    "ct": str(ct_path),
                    "prediction": str(pred_path),
                    "labels": stats_rows,
                    "limitations": LIMITS,
                },
                indent=2,
            )
        )
        with (case_dir / "label_stats.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "label_id", "label", "voxels", "volume_ml", "tags"])
            writer.writeheader()
            writer.writerows(stats_rows)

    if all_rows:
        with (REPORT_ROOT / "all_case_label_stats.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["case_id", "label_id", "label", "voxels", "volume_ml", "tags"])
            writer.writeheader()
            writer.writerows(all_rows)
    (REPORT_ROOT / "label_schema.json").write_text(json.dumps({str(k): v for k, v in LABELS.items()}, indent=2))
    (REPORT_ROOT / "LIMITATIONS.txt").write_text(textwrap.fill(LIMITS, 88) + "\n")


def zip_results() -> Path:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    archive = EXPORT_ROOT / "Spinopelvic_Fullres_Fresh_RESULTS.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root in (PRED_ROOT, REPORT_ROOT, LOG_ROOT):
            for path in sorted(root.rglob("*")):
                if path.is_file():
                    z.write(path, path.relative_to(RUN_ROOT))
        for path in (RUN_ROOT / "input_manifest.json", RUN_ROOT / "input_manifest.csv", RUN_ROOT / "model_manifest.json"):
            if path.exists():
                z.write(path, path.relative_to(RUN_ROOT))
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failed at {bad}")
    (EXPORT_ROOT / "ZIP_SHA256.txt").write_text(f"{sha256(archive)}  {archive.name}\n")
    return archive


def zip_single_case(case_id: str) -> Path:
    EXPORT_ROOT.mkdir(parents=True, exist_ok=True)
    archive = EXPORT_ROOT / f"{case_id}_Spinopelvic_RESULT.zip"
    if archive.exists():
        archive.unlink()
    with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for root in (PRED_ROOT, REPORT_ROOT / case_id, LOG_ROOT):
            if root.exists():
                for path in sorted(root.rglob("*")):
                    if path.is_file():
                        z.write(path, path.relative_to(RUN_ROOT))
        for path in (RUN_ROOT / "input_manifest.json", RUN_ROOT / "input_manifest.csv", RUN_ROOT / "model_manifest.json"):
            if path.exists():
                z.write(path, path.relative_to(RUN_ROOT))
    with zipfile.ZipFile(archive) as z:
        bad = z.testzip()
        if bad:
            raise RuntimeError(f"ZIP integrity failed at {bad}")
    return archive


def main() -> None:
    # Colab's launcher can leak its own flags into sys.argv when this script is
    # executed through `colab exec -f`; keep only arguments meant for this file.
    if Path(sys.argv[0]).name.startswith("colab_kernel_launcher"):
        sys.argv = [__file__]
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="*", help="NRRD/NIfTI CT paths. If omitted, auto-searches upload folders.")
    parser.add_argument("--hf-token", default=None, help="Hugging Face token. Prefer HF_TOKEN env var or getpass in notebook.")
    parser.add_argument("--drive-folder-url", default=None, help="Public/shared Google Drive folder URL containing CT NRRD/NIfTI files.")
    parser.add_argument("--drive-folder-id", default=None, help="Google Drive folder ID containing CT NRRD/NIfTI files.")
    parser.add_argument("--only-case", default=None, help="Run/report one converted case id such as CASE_006.")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="nnUNet inference device.")
    parser.add_argument("--skip-install", action="store_true")
    parser.add_argument("--force", action="store_true", help="Rerun prediction even if predictions exist.")
    args = parser.parse_args()

    started = time.time()
    prepare_dirs()
    install_environment(args.skip_install)
    clone_source()
    login_hf(args.hf_token)
    download_model()
    download_drive_folder(args.drive_folder_url, args.drive_folder_id)
    inputs = find_inputs(args.inputs)
    manifest = convert_inputs(inputs)
    if args.only_case:
        keep = [r for r in manifest if r["case_id"] == args.only_case]
        if not keep:
            raise ValueError(f"Requested {args.only_case}, but manifest has {[r['case_id'] for r in manifest]}")
        for p in NNINPUT_ROOT.glob("CASE_*_0000.nii.gz"):
            if not p.name.startswith(args.only_case + "_"):
                p.unlink()
        manifest = keep
    run_prediction(force=args.force, device=args.device)
    render_reports(manifest)
    if args.only_case:
        archive = zip_single_case(args.only_case)
        status = {
            "status": "single_case_complete_or_partial",
            "archive": str(archive),
            "archive_sha256": sha256(archive),
            "case_count": 1,
            "limitations": LIMITS,
        }
        (RUN_ROOT / f"{args.only_case}_run_status.json").write_text(json.dumps(status, indent=2))
        print(json.dumps(status, indent=2), flush=True)
        return
    archive = zip_results()
    status = {
        "status": "complete_with_possible_case_failures" if (LOG_ROOT / "prediction_failures.json").exists() else "complete",
        "archive": str(archive),
        "archive_sha256": sha256(archive),
        "case_count": len(manifest),
        "elapsed_seconds": round(time.time() - started, 2),
        "limitations": LIMITS,
    }
    (RUN_ROOT / "run_status.json").write_text(json.dumps(status, indent=2))
    print(json.dumps(status, indent=2), flush=True)
    try:
        from google.colab import files

        files.download(str(archive))
    except Exception:
        print(f"Download ZIP manually from: {archive}", flush=True)


if __name__ == "__main__":
    main()
