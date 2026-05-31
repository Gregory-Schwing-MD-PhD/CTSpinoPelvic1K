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
# No default repo — supply per invocation: make export-dataset HF_REPO_ID=org/Name
HF_REPO_ID ?=
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
# WIPE_REMOTE is a flag on `make hf-push` ONLY (clears remote files before
# the push; repo/URL/history preserved). Never honored by `make hf-stage`.
WIPE_REMOTE   ?= 0
# HF_REVISION: push to a branch instead of main (e.g. v2 for the pseudo-
# labelled full release). Empty = main. hf-push only.
HF_REVISION   ?=

# ── Pseudo-label (Stage 3.5 — build the full v2 tree) ────────────────────────
# Empty values fall through to slurm/pseudolabel.sh's own defaults.
NNUNET_SIF     ?=        # nnU-Net+CUDA container (REQUIRED unless DRY_RUN=1)
NNUNET_RESULTS ?=        # checkpoint download dir (default: nnunet/results)
HF_EXPORT_DIR  ?=        # v1 source tree   (default: data/hf_export)
PSEUDO_OUT_DIR ?=        # v2 output tree   (default: data/hf_export_v2)
MODELS_CONFIG  ?=        # default: configs/pseudolabel_models.json
DRY_RUN        ?= 0
SKIP_DOWNLOAD  ?= 0
PSEUDO_LIMIT   ?= 0
# Effective tree hf-push validates AND uploads: HF_EXPORT_DIR if set
# (e.g. the v2 tree), else the default v1 staged tree. Keeps the
# preflight guard honest for the v2 push instead of always checking v1.
HF_PUSH_DIR    := $(if $(strip $(HF_EXPORT_DIR)),$(HF_EXPORT_DIR),$(DATA_DIR)/hf_export)

# ── seg-compare (CPU; quantify model-vs-intensity disagreement) ─────────────
COMPARE_CSV     ?=
COMPARE_WORKERS ?=
COMPARE_NO_ASSD ?= 0
COMPARE_CSV     := $(strip $(COMPARE_CSV))
COMPARE_WORKERS := $(strip $(COMPARE_WORKERS))
COMPARE_NO_ASSD := $(strip $(COMPARE_NO_ASSD))

# ── Intensity refinement (Stage 3.6 — CT-intensity bone seg on the v2 tree) ──
# CPU-only post-step (lean container). Empty values fall through to the
# slurm script's own defaults.
REFINE_OUT_DIR ?=
REFINE_MODE    ?= clip
REFINE_GROW    ?= 3
REFINE_PCTL    ?= 10
REFINE_ERODE   ?= 1
REFINE_FILL    ?= 1
REFINE_WORKERS ?=
REFINE_LIMIT   ?= 0
REFINE_DRY_RUN ?= 0
REFINE_OVERWRITE ?= 0
# Strip any trailing whitespace from values defined with ?= so the comma-
# separated sbatch --export list doesn't split mid-arg.
REFINE_OUT_DIR := $(strip $(REFINE_OUT_DIR))
REFINE_MODE    := $(strip $(REFINE_MODE))
REFINE_GROW    := $(strip $(REFINE_GROW))
REFINE_PCTL    := $(strip $(REFINE_PCTL))
REFINE_ERODE   := $(strip $(REFINE_ERODE))
REFINE_FILL    := $(strip $(REFINE_FILL))
REFINE_WORKERS := $(strip $(REFINE_WORKERS))
REFINE_LIMIT   := $(strip $(REFINE_LIMIT))
REFINE_DRY_RUN := $(strip $(REFINE_DRY_RUN))
REFINE_OVERWRITE := $(strip $(REFINE_OVERWRITE))

# ── compete-refine + change-review control ───────────────────────────────────
REVIEW_OUT_DIR    ?=
PURITY_TOL        ?= 0.15
MIN_BLEED_VOX     ?= 50
BONE_FLOOR        ?= 150
RUN_REFINE        ?= 1
REVIEW_AXIS       ?= 2
REVIEW_MAX_SLICES ?= 12
REVIEW_NO_PNGS    ?= 0
REVIEW_OUT_DIR    := $(strip $(REVIEW_OUT_DIR))
PURITY_TOL        := $(strip $(PURITY_TOL))
MIN_BLEED_VOX     := $(strip $(MIN_BLEED_VOX))
BONE_FLOOR        := $(strip $(BONE_FLOOR))
RUN_REFINE        := $(strip $(RUN_REFINE))
REVIEW_AXIS       := $(strip $(REVIEW_AXIS))
REVIEW_MAX_SLICES := $(strip $(REVIEW_MAX_SLICES))
REVIEW_NO_PNGS    := $(strip $(REVIEW_NO_PNGS))

# ── vertebra QC (neighbour-mixing metrics) ───────────────────────────────────
QC_MANUAL_CSV     ?=
QC_PSEUDO_CSV     ?=
QC_LIMIT          ?= 0
QC_MANUAL_CSV     := $(strip $(QC_MANUAL_CSV))
QC_PSEUDO_CSV     := $(strip $(QC_PSEUDO_CSV))
QC_LIMIT          := $(strip $(QC_LIMIT))

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
	  grep -E '^(download-raw|create-dataset|hf-stage|hf-push|pseudolabel|intensity-refine|export-dataset|benchmark-totalseg):' | \
	  awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-22s\033[0m  %s\n", $$1, $$2}'
	@echo ""
	@echo "Inspection / utilities:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  grep -vE '^(build-container|hpc-pull|hpc-pull-now|docker-push|install-dev|test|lint|check-syntax|download-raw|create-dataset|hf-stage|hf-push|pseudolabel|intensity-refine|export-dataset|benchmark-totalseg|help):' | \
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
.PHONY: hf-stage
hf-stage: check-container  ## Stage 3a — build data/hf_export/ ONLY (no network, no push)
	@mkdir -p $(LOGS_DIR)
	@if [ ! -f $(DATA_DIR)/placed/$(MANIFEST_FILE) ]; then \
	  echo "ERROR: manifest not found at $(DATA_DIR)/placed/$(MANIFEST_FILE)"; \
	  echo "       Either run 'make create-dataset' first, or override with"; \
	  echo "       MANIFEST_FILE=placed_manifest.json make hf-stage ..."; \
	  exit 1; \
	fi
	@echo "Submitting Stage 3a: hf-stage  (export only — NO push)"
	@echo "  MANIFEST_FILE = $(MANIFEST_FILE)"
	@echo "  SKIP_QC       = $(SKIP_QC)   (0 = QC images generated for HF dataset)"
	@echo "  -> when staged, push separately with:  make hf-push HF_REPO_ID=org/Name"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),PUSH=0,SKIP_EXPORT=0,HF_REPO_ID=,HF_WORKERS=$(HF_WORKERS),HF_PRIVATE=$(HF_PRIVATE),SKIP_QC=$(SKIP_QC),NO_PIR=$(NO_PIR),MANIFEST_FILE=$(MANIFEST_FILE) \
	       slurm/export_dataset.sh

.PHONY: hf-push
hf-push: check-container  ## Stage 3b — push an already-staged data/hf_export/ to HF (the ONLY target that touches the remote)
	@mkdir -p $(LOGS_DIR)
	@if [ -z "$(HF_REPO_ID)" ]; then \
	  echo "ERROR: hf-push requires an explicit HF_REPO_ID — there is no default repo."; \
	  echo "       The same export is pushed to multiple venue repos, so the"; \
	  echo "       target must be explicit:"; \
	  echo "         HF_TOKEN=hf_xxx HF_REPO_ID=org/Name make hf-push"; \
	  exit 1; \
	fi
	@if [ -z "$(HF_TOKEN)" ]; then \
	  echo "ERROR: hf-push requires HF_TOKEN.  Prepend HF_TOKEN=hf_xxx to the command."; \
	  exit 1; \
	fi
	@if [ ! -f "$(HF_PUSH_DIR)/manifest.json" ]; then \
	  echo "ERROR: nothing staged at $(HF_PUSH_DIR)/ (no manifest.json)."; \
	  echo "       For the v1 partial: run 'make hf-stage' first."; \
	  echo "       For v2: run 'make pseudolabel' first, then re-run with"; \
	  echo "       HF_EXPORT_DIR=\$$(pwd)/data/hf_export_v2 HF_REVISION=v2 ..."; \
	  echo "       hf-push never re-runs the export."; \
	  exit 1; \
	fi
	@echo "Submitting Stage 3b: hf-push  (push only — export is NOT re-run)"
	@echo "  HF_EXPORT_DIR = $(HF_PUSH_DIR)"
	@echo "  HF_REPO_ID    = $(HF_REPO_ID)"
	@echo "  HF_REVISION   = $(HF_REVISION)  (empty = main branch)"
	@echo "  WIPE_REMOTE   = $(WIPE_REMOTE)  (1 = clear all files first; repo/URL/history kept)"
	@echo "  Resources     : 2 cpu / 16G (push is a network upload; overrides the 24c/128G export header so it schedules fast)"
	sbatch --cpus-per-task=2 --mem=16G \
	       --export=ALL,SIF_PATH=$(CONTAINER),HF_TOKEN=$(HF_TOKEN),PUSH=1,SKIP_EXPORT=1,HF_REPO_ID=$(HF_REPO_ID),HF_REVISION=$(HF_REVISION),HF_EXPORT_DIR=$(HF_PUSH_DIR),HF_WORKERS=$(HF_WORKERS),HF_PRIVATE=$(HF_PRIVATE),WIPE_REMOTE=$(WIPE_REMOTE),MANIFEST_FILE=$(MANIFEST_FILE) \
	       slurm/export_dataset.sh

.PHONY: export-dataset
export-dataset: check-container  ## DEPRECATED — split into 'make hf-stage' + 'make hf-push'
	@if [ "$(PUSH)" = "1" ]; then \
	  echo "ERROR: 'make export-dataset PUSH=1' is removed. Push is now a"; \
	  echo "       separate, explicit step. Stage and push are decoupled:"; \
	  echo "         make hf-stage"; \
	  echo "         HF_TOKEN=hf_xxx HF_REPO_ID=org/Name make hf-push"; \
	  exit 1; \
	fi
	@echo "NOTE: 'make export-dataset' is deprecated and now stages ONLY."
	@echo "      It will NOT push. Run 'make hf-push' separately to push."
	@$(MAKE) hf-stage


# =============================================================================
# Stage 3.5 — pseudo-label completion (build the full v2 tree)
# =============================================================================
.PHONY: pseudolabel
pseudolabel:  ## Stage 3.5 — complete partial cases via out-of-fold nnU-Net (DRY_RUN=1 to plan)
	@mkdir -p $(LOGS_DIR)
	@if [ "$(DRY_RUN)" != "1" ] && [ -z "$(NNUNET_SIF)" ]; then \
	  echo "ERROR: pseudolabel needs NNUNET_SIF=<nnU-Net+CUDA container>"; \
	  echo "       (ctspinopelvic1k-ts.sif ships nnunetv2), or DRY_RUN=1 to plan."; \
	  echo "  DRY_RUN=1 make pseudolabel"; \
	  echo "  NNUNET_SIF=$(TS_CONTAINER) make pseudolabel"; \
	  exit 1; \
	fi
	@echo "Submitting Stage 3.5: pseudolabel  (DRY_RUN=$(DRY_RUN))"
	@echo "  NNUNET_SIF     = $(NNUNET_SIF)"
	@echo "  out-of-fold from staged splits_5fold.json (no train->label leak)"
	@echo "  -> then publish v2 to a branch:"
	@echo "     HF_TOKEN=hf_xxx HF_REPO_ID=org/Name HF_REVISION=v2 \\"
	@echo "       HF_EXPORT_DIR=\$$(pwd)/data/hf_export_v2 make hf-push"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),NNUNET_SIF=$(NNUNET_SIF),NNUNET_RESULTS=$(NNUNET_RESULTS),HF_EXPORT_DIR=$(HF_EXPORT_DIR),PSEUDO_OUT_DIR=$(PSEUDO_OUT_DIR),MODELS_CONFIG=$(MODELS_CONFIG),DRY_RUN=$(DRY_RUN),SKIP_DOWNLOAD=$(SKIP_DOWNLOAD),PSEUDO_LIMIT=$(PSEUDO_LIMIT),HF_TOKEN=$(HF_TOKEN) \
	       slurm/pseudolabel.sh


# =============================================================================
# Stage 3.6 — intensity refinement (CT-intensity bone seg of the pseudo region)
# =============================================================================
.PHONY: intensity-refine
intensity-refine: check-container  ## Stage 3.6 — CT-intensity bone refine of the pseudo region (CPU; REFINE_DRY_RUN=1 to plan)
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting Stage 3.6: intensity-refine  (CPU, REFINE_DRY_RUN=$(REFINE_DRY_RUN))"
	@echo "  v1 manual = $(if $(strip $(HF_EXPORT_DIR)),$(HF_EXPORT_DIR),$(DATA_DIR)/hf_export)"
	@echo "  v2 pseudo = $(if $(strip $(PSEUDO_OUT_DIR)),$(PSEUDO_OUT_DIR),$(DATA_DIR)/hf_export_v2)"
	@echo "  refined   = $(if $(strip $(REFINE_OUT_DIR)),$(REFINE_OUT_DIR),$(DATA_DIR)/hf_export_v2_refined)"
	@echo "  mode=$(REFINE_MODE)  grow_iters=$(REFINE_GROW)  percentile=$(REFINE_PCTL)  erode=$(REFINE_ERODE)  fill=$(REFINE_FILL)"
	@echo "  -> then publish the refined tree to the v2 branch:"
	@echo "     HF_TOKEN=hf_xxx HF_REPO_ID=org/Name HF_REVISION=v2 WIPE_REMOTE=1 \\"
	@echo "       HF_EXPORT_DIR=\$$(pwd)/data/hf_export_v2_refined make hf-push"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),HF_EXPORT_DIR=$(HF_EXPORT_DIR),PSEUDO_OUT_DIR=$(PSEUDO_OUT_DIR),REFINE_OUT_DIR=$(REFINE_OUT_DIR),REFINE_MODE=$(REFINE_MODE),REFINE_GROW=$(REFINE_GROW),REFINE_PCTL=$(REFINE_PCTL),REFINE_ERODE=$(REFINE_ERODE),REFINE_FILL=$(REFINE_FILL),REFINE_WORKERS=$(REFINE_WORKERS),REFINE_LIMIT=$(REFINE_LIMIT),REFINE_DRY_RUN=$(REFINE_DRY_RUN),REFINE_OVERWRITE=$(REFINE_OVERWRITE) \
	       slurm/intensity_refine.sh


# NOTE: use := (not ?=) so a stray exported PCTL_SWEEP/GROW_SWEEP in the shell
# environment can't silently shrink the sweep. Command-line overrides
# (make sweep-refine PCTL_SWEEP=5) still win, since CLI beats makefile assignment.
PCTL_SWEEP := 5,10,15,20,30
GROW_SWEEP := 0,1,2
SWEEP_CSV  ?=
BEST_JSON  ?=
PCTL_SWEEP := $(strip $(PCTL_SWEEP))
GROW_SWEEP := $(strip $(GROW_SWEEP))
SWEEP_CSV  := $(strip $(SWEEP_CSV))
BEST_JSON  := $(strip $(BEST_JSON))


.PHONY: sweep-refine
sweep-refine: check-container  ## Sweep (pctl, grow), pick best, build refined tree, run compare + eval — all in ONE job
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting sweep-refine (CPU, all stages, one job)"
	@echo "  pctl_sweep = $(PCTL_SWEEP)"
	@echo "  grow_sweep = $(GROW_SWEEP)"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),HF_EXPORT_DIR=$(HF_EXPORT_DIR),PSEUDO_OUT_DIR=$(PSEUDO_OUT_DIR),REFINE_OUT_DIR=$(REFINE_OUT_DIR),PRED_DIR=$(PRED_DIR),MODELS_CONFIG=$(MODELS_CONFIG),SWEEP_CSV=$(SWEEP_CSV),BEST_JSON=$(BEST_JSON),COMPARE_CSV=$(COMPARE_CSV),EVAL_CSV=$(EVAL_CSV),PCTL_SWEEP=$(PCTL_SWEEP),GROW_SWEEP=$(GROW_SWEEP),REFINE_WORKERS=$(REFINE_WORKERS) \
	       slurm/sweep_refine.sh


.PHONY: vertebra-qc
vertebra-qc: check-container  ## GT-free neighbour-mixing QC on manual vs pseudo trees (CPU)
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting vertebra-qc (manual vs pseudo neighbour-mixing metrics)"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),HF_EXPORT_DIR=$(HF_EXPORT_DIR),PSEUDO_OUT_DIR=$(PSEUDO_OUT_DIR),QC_MANUAL_CSV=$(QC_MANUAL_CSV),QC_PSEUDO_CSV=$(QC_PSEUDO_CSV),QC_LIMIT=$(QC_LIMIT) \
	       slurm/vertebra_qc.sh


.PHONY: refine-eval
refine-eval: check-container  ## Stage 3.6 + compare + eval-vs-manual, all in ONE SLURM job (CPU)
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting refine+eval (three stages, one job)"
	@echo "  mode=$(REFINE_MODE)  grow=$(REFINE_GROW)  percentile=$(REFINE_PCTL)  erode=$(REFINE_ERODE)  fill=$(REFINE_FILL)"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),HF_EXPORT_DIR=$(HF_EXPORT_DIR),PSEUDO_OUT_DIR=$(PSEUDO_OUT_DIR),REFINE_OUT_DIR=$(REFINE_OUT_DIR),PRED_DIR=$(PRED_DIR),MODELS_CONFIG=$(MODELS_CONFIG),COMPARE_CSV=$(COMPARE_CSV),EVAL_CSV=$(EVAL_CSV),REFINE_MODE=$(REFINE_MODE),REFINE_GROW=$(REFINE_GROW),REFINE_PCTL=$(REFINE_PCTL),REFINE_ERODE=$(REFINE_ERODE),REFINE_FILL=$(REFINE_FILL),REFINE_LIMIT=$(REFINE_LIMIT),REFINE_WORKERS=$(REFINE_WORKERS),COMPARE_NO_ASSD=$(COMPARE_NO_ASSD),EVAL_NO_ASSD=$(EVAL_NO_ASSD) \
	       slurm/refine_eval.sh


.PHONY: refine-review
# This job IS the compete-refine + review; default to compete (not the global
# clip) and to bounded-grow 0 (so a hip seed can't grow into an unpredicted
# femur). A command-line REFINE_MODE=/REFINE_GROW= still overrides these.
refine-review: REFINE_MODE := compete
refine-review: REFINE_GROW := 0
refine-review: check-container  ## compete-refine + 2D change-overlay review, ONE SLURM job (CPU). RUN_REFINE=0 to review an existing tree only.
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting compete-refine + change-review (one job)"
	@echo "  mode=$(REFINE_MODE)  purity_tol=$(PURITY_TOL)  min_bleed_vox=$(MIN_BLEED_VOX)  bone_floor=$(BONE_FLOOR)  RUN_REFINE=$(RUN_REFINE)"
	@echo "  refined   = $(if $(strip $(REFINE_OUT_DIR)),$(REFINE_OUT_DIR),$(DATA_DIR)/hf_export_v2_compete)"
	@echo "  review    = $(if $(strip $(REVIEW_OUT_DIR)),$(REVIEW_OUT_DIR),$(DATA_DIR)/refine_review)"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),HF_EXPORT_DIR=$(HF_EXPORT_DIR),PSEUDO_OUT_DIR=$(PSEUDO_OUT_DIR),REFINE_OUT_DIR=$(REFINE_OUT_DIR),REVIEW_OUT_DIR=$(REVIEW_OUT_DIR),REFINE_MODE=$(REFINE_MODE),REFINE_PCTL=$(REFINE_PCTL),REFINE_ERODE=$(REFINE_ERODE),REFINE_GROW=$(REFINE_GROW),REFINE_FILL=$(REFINE_FILL),PURITY_TOL=$(PURITY_TOL),MIN_BLEED_VOX=$(MIN_BLEED_VOX),BONE_FLOOR=$(BONE_FLOOR),REFINE_WORKERS=$(REFINE_WORKERS),REFINE_LIMIT=$(REFINE_LIMIT),REFINE_OVERWRITE=$(REFINE_OVERWRITE),RUN_REFINE=$(RUN_REFINE),REVIEW_AXIS=$(REVIEW_AXIS),REVIEW_MAX_SLICES=$(REVIEW_MAX_SLICES),REVIEW_NO_PNGS=$(REVIEW_NO_PNGS) \
	       slurm/refine_review.sh


.PHONY: eval-vs-manual
eval-vs-manual: check-container  ## Quantify model accuracy vs MANUAL ground truth on the scoped manual side (CPU)
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting eval-vs-manual (CPU)"
	@echo "  v1 manual = $(if $(strip $(HF_EXPORT_DIR)),$(HF_EXPORT_DIR),$(DATA_DIR)/hf_export)"
	@echo "  preds     = $(if $(strip $(PRED_DIR)),$(PRED_DIR),$(DATA_DIR)/hf_export_v2_work/preds)"
	@echo "  csv       = $(if $(strip $(EVAL_CSV)),$(EVAL_CSV),$(DATA_DIR)/eval_vs_manual.csv)"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),HF_EXPORT_DIR=$(HF_EXPORT_DIR),PRED_DIR=$(PRED_DIR),MODELS_CONFIG=$(MODELS_CONFIG),EVAL_CSV=$(EVAL_CSV),EVAL_WORKERS=$(EVAL_WORKERS),EVAL_NO_ASSD=$(EVAL_NO_ASSD) \
	       slurm/eval_vs_manual.sh


EVAL_CSV     ?=
EVAL_WORKERS ?=
EVAL_NO_ASSD ?= 0
PRED_DIR     ?=
EVAL_CSV     := $(strip $(EVAL_CSV))
EVAL_WORKERS := $(strip $(EVAL_WORKERS))
EVAL_NO_ASSD := $(strip $(EVAL_NO_ASSD))
PRED_DIR     := $(strip $(PRED_DIR))


.PHONY: seg-compare
seg-compare: check-container  ## Quantify model-vs-intensity segmentation disagreement (CPU)
	@mkdir -p $(LOGS_DIR)
	@echo "Submitting seg-compare (CPU)"
	@echo "  v1 manual = $(if $(strip $(HF_EXPORT_DIR)),$(HF_EXPORT_DIR),$(DATA_DIR)/hf_export)"
	@echo "  model v2  = $(if $(strip $(PSEUDO_OUT_DIR)),$(PSEUDO_OUT_DIR),$(DATA_DIR)/hf_export_v2)"
	@echo "  refined   = $(if $(strip $(REFINE_OUT_DIR)),$(REFINE_OUT_DIR),$(DATA_DIR)/hf_export_v2_refined)"
	@echo "  csv       = $(if $(strip $(COMPARE_CSV)),$(COMPARE_CSV),$(DATA_DIR)/seg_compare.csv)"
	sbatch --export=ALL,SIF_PATH=$(CONTAINER),HF_EXPORT_DIR=$(HF_EXPORT_DIR),PSEUDO_OUT_DIR=$(PSEUDO_OUT_DIR),REFINE_OUT_DIR=$(REFINE_OUT_DIR),COMPARE_CSV=$(COMPARE_CSV),COMPARE_WORKERS=$(COMPARE_WORKERS),COMPARE_NO_ASSD=$(COMPARE_NO_ASSD) \
	       slurm/seg_compare.sh


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
