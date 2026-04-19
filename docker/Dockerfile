# CTSpinoPelvic1K — lean dataset construction + visualization image.
# Bare necessities for: TCIA download, dcm2niix conversion, fused mask
# placement, QC rendering, HF export, figure generation.
#
# NO torch, NO MONAI, NO wandb — those belong in the training/inference image.
#
# Build:   docker build -f docker/Dockerfile -t ctspinopelvic1k:latest .
# HPC:     singularity pull ctspinopelvic1k.sif docker://<user>/ctspinopelvic1k:latest

FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# System deps: dcm2niix for DICOM→NIfTI, build-essential for scipy/numpy wheels,
# git+ca-certificates for HF hub, libopenjp2 for JPEG2000 DICOM pixel decoding.
RUN apt-get update && apt-get install -y --no-install-recommends \
        dcm2niix \
        build-essential \
        git \
        curl \
        ca-certificates \
        libopenjp2-7 \
        libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/ctspinopelvic1k

# Install Python deps up front for better layer caching.
# pydicom is left unpinned so the resolver can pick a version compatible with
# whatever tcia-utils wants (tcia-utils 3.x supports pydicom >=2.3).
RUN pip install --upgrade pip && pip install \
        numpy==1.26.4 \
        scipy==1.13.1 \
        nibabel==5.2.1 \
        "pydicom>=2.3,<3" \
        SimpleITK==2.3.1 \
        scikit-image==0.22.0 \
        matplotlib==3.8.4 \
        tqdm==4.66.4 \
        pandas==2.2.2 \
        scikit-learn==1.4.2 \
        huggingface_hub==0.24.5 \
        "tcia-utils>=3.0,<4" \
        requests==2.32.3

# Copy pipeline source (scripts + SLURM launchers).
COPY scripts/ /opt/ctspinopelvic1k/scripts/
COPY slurm/   /opt/ctspinopelvic1k/slurm/
COPY Makefile pyproject.toml README.md /opt/ctspinopelvic1k/

# PYTHONPATH is empty at image build time, so don't try to append to it here.
# Callers can extend it at runtime via `docker run -e PYTHONPATH=...`.
ENV PYTHONPATH=/opt/ctspinopelvic1k/scripts

# Default command prints help.
CMD ["python", "-c", "print('CTSpinoPelvic1K dataset construction container. See scripts/ and slurm/.')"]
