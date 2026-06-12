---
title: CTSpinoPelvic1K Review
emoji: 🦴
colorFrom: blue
colorTo: red
sdk: docker
app_port: 7860
pinned: false
---

# CTSpinoPelvic1K review service (Phase 2)

FastAPI coordination service for **distributed double-review + adjudication**
of the v2 pseudo labels. Reviewers authenticate with their **own HuggingFace
login** (`hf auth login`); the Space verifies the token via `whoami` and uses
the username as their identity (open mode — any HF user may review; only
`ADJUDICATORS` may adjudicate). The Space holds the single dataset **write**
token, which reviewers never see. CT + pseudo labels are fetched by the client
straight from the **public v2** repo; only corrected labels are uploaded
through this service into a **private review repo**, the source of truth for
claims / status / reviews / finalized labels.

## Deploy as a HuggingFace Space (Docker SDK)

Push this repo (or a subtree containing `review_service/` + `scripts/review/`)
to a Space whose SDK is **docker**. Set these **Space secrets / variables**:

| name | example | notes |
|---|---|---|
| `HF_TOKEN` | `hf_...` (write) | secret; write access to `REVIEW_REPO` |
| `REVIEW_REPO` | `anonymous-neurips-ED/CTSpinoPelvic1K-reviews` | private dataset (auto-created) |
| `V2_REPO` | `anonymous-neurips-ED/CTSpinoPelvic1K` | public source of CT + pseudo |
| `SOURCE_REVISION` | `v2` | branch holding the pseudo labels |
| `ADJUDICATORS` | `drsmith,drokafor` | HF usernames allowed to adjudicate; any other HF-authenticated user is a primary reviewer |
| `REVIEWER_KEYS` | `{"k_snr":{"id":"snr","role":"adjudicator"}}` | optional/legacy: bearer values matching a key here bypass HF identity |
| `TASK` | `lstv` | `lstv` (default) = correct v2 pseudo-labels; `rib_anchor` = the v4 add-the-anchor pass over **v3** (see below) |
| `TAU` | `0.9` | IRR agreement threshold (per-class min-Dice) |
| `IRR_MODE` | `per_class_min` | or `overall` |

On first boot it seeds review cases from `manifest.json` (idempotent). The
`TASK` env selects what gets seeded:
- `lstv` (default): one case per `spine_only` / `pelvic_native` record to
  correct the v2 pseudo-labels (or, if `crops/crops_index.json` is present, only
  the QC-flagged worklist).
- `rib_anchor`: one case per dense-labelled **v3** record (all configs), serving
  the existing v3 label as the editable base — see the next section.

For **local dev** (no HF), set `LOCAL_STORE_DIR=/tmp/reviewstore` and run:
```bash
pip install -r review_service/requirements.txt
LOCAL_STORE_DIR=/tmp/rs V2_REPO=org/X REVIEWER_KEYS='{"k":{"id":"a","role":"adjudicator"}}' \
  uvicorn review_service.app:app --port 7860
```

## Rib-anchor (v4) task — serving v3 to students

The v4 pass asks annotators to **add the counting anchor** (`last_rib_vertebra`
= 11, `rib` = 12) onto the existing v3 dense label, and to tidy any class-mixing
/ partly-coloured vertebrae while they are in the case. Rationale +
how-to-annotate: `docs/RIB_ANCHOR_RATIONALE.md`,
`docs/STUDENT_ANNOTATION_PROTOCOL.md`.

Because the claim serves CT **and** label from a single `repo@revision`, the v3
revision must carry the **CT volumes**, not just labels. One-time publish:

```bash
# 1. Build the FULL v3 tree WITH CTs (corrected labels swapped in).
#    V3_LABELS_ONLY=0 is the only change from the QC build.
V3_LABELS_ONLY=0 sbatch slurm/reduce_v3.sh
#    (if has_l6/splits metadata needs the spine-authoritative refresh, run
#     WRITE=1 RESPLIT=1 sbatch slurm/refresh_lstv_v3.sh against the same tree.)

# 2. Push it to the dataset's v3 revision (branch).
HF_TOKEN=hf_xxx HF_REPO_ID=<org>/CTSpinoPelvic1K HF_REVISION=v3 \
    HF_EXPORT_DIR=$(pwd)/data/hf_export_v3 make hf-push
```

Then deploy a **second Space** (or re-point this one) with:

| name | value |
|---|---|
| `TASK` | `rib_anchor` |
| `V2_REPO` | `<org>/CTSpinoPelvic1K` (the CT+label source — now carries v3) |
| `SOURCE_REVISION` | `v3` |
| `REVIEW_REPO` | `<org>/CTSpinoPelvic1K-reviews-rib` (a fresh private repo) |
| `ADJUDICATORS` | the resident/radiologist usernames |

Keep it on a **separate `REVIEW_REPO`** so the rib-anchor claims/finals don't
mix with the LSTV review. Students then run the same client:
`reviewtool login --service <rib-space-url>` → `reviewtool next` (it prints the
find-the-anchor instructions and opens ITK-SNAP with the 12-class palette).
Closing the loop folds the finalized anchor labels into **v4** via the same
`reduce_to_v3` machinery (point `--out data/hf_export_v4`, push `HF_REVISION=v4`).

## API

| method | path | auth | body |
|---|---|---|---|
| POST | `/claim` | reviewer | — → assigns a primary slot (A≠B), returns CT/pseudo URLs + claim token + label descriptor |
| POST | `/submit` | reviewer | form: `claim_token`, `record` (JSON), `label` (file). On 2nd primary, computes IRR → auto-finalize (≥τ) or `needs_adjudication` |
| GET | `/adjudication/next` | adjudicator | → a disagreeing case + both reviews + IRR |
| POST | `/adjudicate` | adjudicator | form: `claim_token`, `decision`, `notes`, `label` (file) |
| POST | `/admin/build_finals` | adjudicator | writes `finals.json` for `reduce_to_v3` |
| GET | `/status` | reviewer | JSON summary |
| GET | `/` | — | HTML dashboard |

Auth: `Authorization: Bearer <hf_token>` (the reviewer's HuggingFace token; the
Space resolves it to a username via `whoami`). A legacy minted `<api_key>` is
also accepted if present in `REVIEWER_KEYS`.

## Closing the loop → v3
When review is complete, pull the private repo's `finals.json` + corrected
labels and run:
```bash
python scripts/review/reduce_to_v3.py --v2 data/hf_export_v2 \
    --finals finals.json --labels_root <pulled review repo> --out data/hf_export_v3
# then: HF_TOKEN=... HF_REPO_ID=org/Name HF_REVISION=v3 \
#         HF_EXPORT_DIR=$(pwd)/data/hf_export_v3 make hf-push
```

The state machine + IRR + provenance live in `scripts/review/` (Phase 1,
unit-tested); this service is a thin, write-serialized layer over it.
