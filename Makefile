# =============================================================================
# CTSpinoPelvic1K  Makefile
#
# User-facing entry points for the three pipeline stages.  Every stage is a
# single `sbatch` under the hood; overrides are passed via environment vars.
#
# Typical workflow:
#     make build-container         # once
#     make download-raw            # Stage 1
#     make create-dataset          # Stage 2  (QC off by default)
#     HF_TOKEN=hf_xxx make export-dataset PUSH=1     # Stage 3  (QC on by default)
#
# See `make help` for the full list of targets.
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# ── Paths ────────────────────────────────────────────────────────────────────
ROOT_DIR      := $(CURDIR)
DATA_DIR      ?= $(ROOT_DIR)/data
LOGS_DIR      ?= $(ROOT_DIR)/logs
CONFIGS_DIR   ?= $(ROOT_DIR)/configs

# ── Containers (two .sif files, pulled from Docker Hub) ─────────────────────
# All non-TS pipeline stages use the lean image.  Only Stage 4 (TotalSegmentator
# benchmark) uses the CUDA image.  See scripts/hpc_pull.sh for pull details.
CONTAINER     ?= $(ROOT_DIR)/containers/ctspinopelvic1k.sif
TS_CONTAINER  ?= $(ROOT_DIR)/containers/ctspinopelvic1k-ts.sif

# ── HuggingFace ──────────────────────────────────────────────────────────────
HF_REPO_ID ?= anonymous-mlhc/CTSpinoPelvic1K
HF_TOKEN   ?=
PUSH       ?= 0

# ── Download scope ───────────────────────────────────────────────────────────
# TCIA_SCOPE: "all" (~3451 series) or "filtered" (~1194, CTPelvic1K patients only)
TCIA_SCOPE ?= all

# ── Parallelism ──────────────────────────────────────────────────────────────
WORKERS         ?= 32
DCM2NIIX_WORKERS ?= 16

# ── Stage 2 control ──────────────────────────────────────────────────────────
# DEBUG_N: limit to first N patients (0 = all)
# DEBUG_TOKENS: comma-separated list of patient tokens; overrides DEBUG_N
# CREATE_SKIP_QC / CREATE_SKIP_QC_ORIFIX: QC OFF by default for create-dataset.
#   QC figures are mostly useful for the export stage's curated set, not for
#   the bulk Stage 2 run. Set to 0 to re-enable.
DEBUG_N             ?= 0
DEBUG_TOKENS        ?=
CREATE_SKIP_QC        ?= 1
CREATE_SKIP_QC_ORIFIX ?= 1

# ── Stage 3 control ──────────────────────────────────────────────────────────
# MANIFEST_FILE: which Stage 2 manifest Stage 3 consumes.  Default is the
# orientation-fixed manifest so AP-inverted cases (e.g., token 480) are
# exported with flipped CT + masks.  Override with MANIFEST_FILE=placed_manifest.json
# to export from the un-fixed manifest.
# SKIP_QC: 0 by default for export-dataset (QC images ARE generated here, since
#   they accompany the published HF dataset).
MANIFEST_FILE ?= placed_manifest_orientation_fixed.json
SKIP_QC       ?= 0
NO_PIR        ?= 0
SKIP_EXPORT   ?= 0
HF_PRIVATE    ?= 0
HF_WORKERS    ?= 8

# ── Stage 4 control (TotalSegmentator benchmark) ─────────────────────────────
TS_WINDOW_MM    ?= 40.0
DOCKERHUB_USER  ?= gregoryschwingmdphd


# =============================================================================
# help
# =============================================================================
.PHONY: help
help:  ## Show this help
	@echo ""
	@echo "CTSpinoPelvic1K  —  dataset construction pipeline"
	@echo ""
	@echo "Setup:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  grep -E '^(build-container|hpc-pull|hpc-pull-now|docker-push|install-dev|test|lint|check-syntax):' | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m  %s\n", $$1, $$2}'
	@echo ""
	@echo "Pipeline (in order):"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  grep -E '^(download-raw|create-dataset|export-dataset|benchmark-totalseg):' | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m  %s\n", $$1, $$2}'
	@echo ""
	@echo "Inspection / utilities:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  grep -vE '^(build-container|hpc-pull|hpc-pull-now|docker-push|install-dev|test|lint|check-syntax|download-raw|create-dataset|export-dataset|benchmark-totalseg|help):' | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m  %s\n", $$1, $$2}'
	@echo ""
	@echo "Common env overrides (set via VAR=value before the target):"
	@echo "  HF_TOKEN        HuggingFace access token (required for push)"
	@echo "  PUSH=1          Push to HF during export-dataset"
	@echo "  TCIA_SCOPE      all | filtered  (default: all)"
	@echo "  DEBUG_N=5       Limit create-dataset to first 5 patients"
	@echo "  DEBUG_TOKENS=\"145,184,205\"   Specific tokens only (comma- or space-separated)"
	@echo "  WORKERS=16      Override CPU count"
	@echo "  MANIFEST_FILE   Stage 2 manifest for Stage 3"
	@echo "                  default: placed_manifest_orientation_fixed.json"
	@echo "                  override: MANIFEST_FILE=placed_manifest.json"
	@echo ""
	@echo "QC image generation (defaults shown):"
	@echo "  CREATE_SKIP_QC=1         create-dataset Step D off (original-manifest QC)"
	@echo "  CREATE_SKIP_QC_ORIFIX=1  create-dataset Step E off (post-flip QC)"
	@echo "    -> set either to 0 to re-enable QC during create-dataset"
	@echo "  SKIP_QC=0                export-dataset QC ON (default; QC ships with HF dataset)"
	@echo ""


# =============================================================================
# Setup — pull containers from Docker Hub
# =============================================================================
.PHONY: build-container
build-container: hpc-pull  ## Alias for hpc-pull — submits slurm/hpc_pull.sh

.PHONY: check-container
check-container:
	@test -f $(CONTAINER) || { \
	  echo "ERROR: container not found at $(CONTAINER)"; \
	  echo "       Run:  sbatch slurm/hpc_pull.sh    (or: make hpc-pull)"; \
	  exit 1; \
	}

.PHONY: check-ts-container
check-ts-container:
	@test -f $(TS_CONTAINER) || { \
	  echo "ERROR: TS container not found at $(TS_CONTAINER)"; \
	  echo "       Run:  sbatch slurm/hpc_pull.sh    (or: make hpc-pull)"; \
	  exit 1; \
	}


# =============================================================================
# Stage 1 — download raw data
# =============================================================================
.PHONY: download-raw
download-raw: check-container  ## Stage 1 — download TCIA + CTSpine1K + CTPelvic1K
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting Stage 1: download-raw ..."
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),TCIA_SCOPE=$(TCIA_SCOPE),HF_TOKEN=$(HF_TOKEN) \
	       slurm/download_raw.sh


# =============================================================================
# Stage 2 — create dataset
# =============================================================================
# QC images are OFF by default here. They're only useful for spotting
# AP-inverted cases for the manual flip list, which is a one-time review.
# The export stage generates a curated QC set that ships with the dataset.
# Re-enable with CREATE_SKIP_QC=0 (and/or CREATE_SKIP_QC_ORIFIX=0).
.PHONY: create-dataset
create-dataset: check-container  ## Stage 2 — build PatientDB + place masks (QC off by default)
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting Stage 2: create-dataset ..."
	@echo "  CREATE_SKIP_QC        = $(CREATE_SKIP_QC)   (Step D, original-manifest QC)"
	@echo "  CREATE_SKIP_QC_ORIFIX = $(CREATE_SKIP_QC_ORIFIX)   (Step E, post-flip QC)"
	DEBUG_TOKENS='$(DEBUG_TOKENS)' \
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),DEBUG_N=$(DEBUG_N),WORKERS=$(WORKERS),DCM2NIIX_WORKERS=$(DCM2NIIX_WORKERS),SKIP_QC=$(CREATE_SKIP_QC),SKIP_QC_ORIFIX=$(CREATE_SKIP_QC_ORIFIX) \
	       slurm/create_dataset.sh


# =============================================================================
# Stage 3 — export + push
# =============================================================================
.PHONY: export-dataset
export-dataset: check-container  ## Stage 3 — split, export, optionally push to HF
	@mkdir -p $(LOGS_DIR)
	@if [ "$(PUSH)" = "1" ] && [ -z "$(HF_TOKEN)" ]; then \
	  echo "ERROR: PUSH=1 requires HF_TOKEN.  Prepend HF_TOKEN=hf_xxx to the command."; \
	  exit 1; \
	fi
	@if [ ! -f $(DATA_DIR)/placed/$(MANIFEST_FILE) ]; then \
	  echo "ERROR: manifest not found at $(DATA_DIR)/placed/$(MANIFEST_FILE)"; \
	  echo "       Either run 'make create-dataset' first, or override with"; \
	  echo "       MANIFEST_FILE=placed_manifest.json make export-dataset ..."; \
	  exit 1; \
	fi
	@echo "Submitting Stage 3: export-dataset"
	@echo "  MANIFEST_FILE = $(MANIFEST_FILE)"
	@echo "  PUSH          = $(PUSH)"
	@echo "  SKIP_QC       = $(SKIP_QC)   (0 = QC images generated for HF dataset)"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),HF_TOKEN=$(HF_TOKEN),PUSH=$(PUSH),HF_REPO_ID=$(HF_REPO_ID),HF_WORKERS=$(HF_WORKERS),HF_PRIVATE=$(HF_PRIVATE),SKIP_QC=$(SKIP_QC),NO_PIR=$(NO_PIR),SKIP_EXPORT=$(SKIP_EXPORT),MANIFEST_FILE=$(MANIFEST_FILE) \
	       slurm/export_dataset.sh


# =============================================================================
# Inspection / utilities
# =============================================================================
.PHONY: status
status:  ## Show disk usage and stage completion status
	@echo ""
	@echo "Pipeline status  ($(ROOT_DIR))"
	@echo "===================================================================="
	@printf "  %-40s %s\n" "TCIA raw ($(DATA_DIR)/tcia):" \
	  "$$(du -sh $(DATA_DIR)/tcia 2>/dev/null | cut -f1 || echo 'NOT DOWNLOADED')"
	@printf "  %-40s %s\n" "CTSpine1K ($(DATA_DIR)/ctspine1k):" \
	  "$$(du -sh $(DATA_DIR)/ctspine1k 2>/dev/null | cut -f1 || echo 'NOT DOWNLOADED')"
	@printf "  %-40s %s\n" "CTPelvic1K ($(DATA_DIR)/ctpelvic1k):" \
	  "$$(du -sh $(DATA_DIR)/ctpelvic1k 2>/dev/null | cut -f1 || echo 'NOT DOWNLOADED')"
	@echo ""
	@printf "  %-40s %s\n" "patient_db.json:" \
	  "$$(test -f $(DATA_DIR)/patient_db.json && echo BUILT || echo MISSING)"
	@printf "  %-40s %s\n" "placed_manifest.json:" \
	  "$$(test -f $(DATA_DIR)/placed/placed_manifest.json && echo BUILT || echo MISSING)"
	@printf "  %-40s %s\n" "placed_manifest_orientation_fixed.json:" \
	  "$$(test -f $(DATA_DIR)/placed/placed_manifest_orientation_fixed.json && echo BUILT || echo MISSING)"
	@printf "  %-40s %s\n" "Active manifest for export (MANIFEST_FILE):" \
	  "$(MANIFEST_FILE)"
	@printf "  %-40s %s\n" "hf_export/:" \
	  "$$(du -sh $(DATA_DIR)/hf_export 2>/dev/null | cut -f1 || echo 'NOT EXPORTED')"
	@echo ""
	@echo "Active SLURM jobs:"
	@squeue -u $$USER -o "  %.10i %.15j %.8T %.10M %R" 2>/dev/null | tail -n +2 || echo "  (no jobs)"
	@echo ""

.PHONY: logs
logs:  ## Tail the most recent SLURM log
	@latest=$$(ls -t $(LOGS_DIR)/*.out 2>/dev/null | head -1); \
	if [ -z "$$latest" ]; then \
	  echo "No logs in $(LOGS_DIR)/"; \
	else \
	  echo "Tailing: $$latest"; \
	  tail -f $$latest; \
	fi

.PHONY: clean-logs
clean-logs:  ## Remove old SLURM log files
	@find $(LOGS_DIR) -name "*.out" -o -name "*.err" | xargs rm -f 2>/dev/null || true
	@echo "Cleaned $(LOGS_DIR)/"

.PHONY: clean-data
clean-data:  ## DANGER — remove all staged data (asks for confirmation)
	@echo "This will permanently delete $(DATA_DIR)"
	@read -p "Type 'yes' to continue: " ans; [ "$$ans" = "yes" ] || exit 1
	rm -rf $(DATA_DIR)/*
	@echo "Removed all data."


# =============================================================================
# Stage 4 — TotalSegmentator benchmark (uses ctspinopelvic1k-ts.sif)
# =============================================================================
.PHONY: benchmark-totalseg
benchmark-totalseg: check-ts-container  ## Stage 4 — zero-shot TotalSegmentator benchmark on whole dataset
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting Stage 4: benchmark-totalseg ..."
	sbatch --export=ALL,SIF_PATH=$(TS_CONTAINER),DATASET_DIR=$(DATA_DIR)/hf_export \
	       slurm/benchmark_totalseg.sh

.PHONY: build-manifest
build-manifest:  ## Build external training manifest (NIfTI paths) from placed_manifest
	python scripts/build_manifest.py \
	    --placed_manifest $(DATA_DIR)/placed/$(MANIFEST_FILE) \
	    --patient_db      $(DATA_DIR)/patient_db.json \
	    --out             $(DATA_DIR)/matched/colonog_training_manifest.json \
	    --nifti_dir       $(DATA_DIR)/tcia_nifti \
	    --placed_spine_dir         $(DATA_DIR)/placed/spine \
	    --placed_mask_dir          $(DATA_DIR)/placed/fused \
	    --placed_pelvic_native_dir $(DATA_DIR)/placed/pelvic_native

.PHONY: render-lstv-gt
render-lstv-gt:  ## Render publication LSTV panel (ground-truth labels)
	python scripts/render_lstv_examples.py \
	    --source gt \
	    --manifest  $(DATA_DIR)/placed/$(MANIFEST_FILE) \
	    --spine_dir $(DATA_DIR)/placed/spine \
	    --fused_dir $(DATA_DIR)/placed/fused \
	    --pelv_dir  $(DATA_DIR)/placed/pelvic_native \
	    --out_dir   $(DATA_DIR)/figures/lstv_gt

.PHONY: render-lstv-ts
render-lstv-ts:  ## Render LSTV panel using TS predictions (needs --ts_pred_dir)
	@if [ -z "$(TS_PRED_DIR)" ]; then \
	  echo "ERROR: set TS_PRED_DIR=path/to/ts_preds"; exit 1; \
	fi
	python scripts/render_lstv_examples.py \
	    --source ts \
	    --manifest      $(DATA_DIR)/placed/$(MANIFEST_FILE) \
	    --ts_pred_dir   $(TS_PRED_DIR) \
	    --hf_export_dir $(DATA_DIR)/hf_export \
	    --spine_dir     $(DATA_DIR)/placed/spine \
	    --fused_dir     $(DATA_DIR)/placed/fused \
	    --pelv_dir      $(DATA_DIR)/placed/pelvic_native \
	    --out_dir       $(DATA_DIR)/figures/lstv_ts


# =============================================================================
# Docker / HPC container plumbing
# =============================================================================
.PHONY: docker-push
docker-push:  ## Build + push both Docker images (run on workstation, not HPC)
	@DOCKERHUB_USER=$(DOCKERHUB_USER) bash scripts/docker_push.sh

.PHONY: hpc-pull
hpc-pull:  ## Submit slurm job that pulls both .sif images on HPC
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting slurm/hpc_pull.sh  (DOCKERHUB_USER=$(DOCKERHUB_USER)) ..."
	sbatch --export=ALL,DOCKERHUB_USER=$(DOCKERHUB_USER) slurm/hpc_pull.sh

.PHONY: hpc-pull-now
hpc-pull-now:  ## Pull .sif images immediately on the current node (not via slurm)
	@DOCKERHUB_USER=$(DOCKERHUB_USER) bash scripts/hpc_pull.sh


# =============================================================================
# Development
# =============================================================================
.PHONY: check-syntax
check-syntax:  ## Syntax-check all Python and Bash scripts
	@echo "Checking Python syntax..."
	@for f in scripts/*.py tools/*.py; do \
	  if [ -f "$$f" ]; then python3 -m py_compile "$$f" && echo "  OK  $$f" || exit 1; fi; \
	done
	@echo "Checking Bash syntax..."
	@for f in slurm/*.sh scripts/*.sh; do \
	  if [ -f "$$f" ]; then bash -n "$$f" && echo "  OK  $$f" || exit 1; fi; \
	done
	@echo "All scripts pass syntax check."

.PHONY: lint
lint: check-syntax  ## Alias for check-syntax

.PHONY: install-dev
install-dev:  ## Editable install incl. dev/test tooling (pytest, ruff)
	python3 -m pip install -e ".[dev]"

.PHONY: test
test:  ## Run the pytest suite (auto-installs dev extras if pytest missing)
	@python3 -c "import pytest" 2>/dev/null || $(MAKE) install-dev
	python3 -m pytest

.PHONY: clean
clean: clean-logs  ## Remove logs and __pycache__
	@find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true
	@echo "Cleaned __pycache__ and logs."
