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
of the v2 pseudo labels. Holds the single HF write token; reviewers
authenticate with per-reviewer API keys and never see the token. CT +
pseudo labels are fetched by the client straight from the **public v2**
repo; only corrected labels are uploaded through this service into a
**private review repo**, which is the source of truth for claims / status /
reviews / finalized labels.

## Deploy as a HuggingFace Space (Docker SDK)

Push this repo (or a subtree containing `review_service/` + `scripts/review/`)
to a Space whose SDK is **docker**. Set these **Space secrets / variables**:

| name | example | notes |
|---|---|---|
| `HF_TOKEN` | `hf_...` (write) | secret; write access to `REVIEW_REPO` |
| `REVIEW_REPO` | `anonymous-neurips-ED/CTSpinoPelvic1K-reviews` | private dataset (auto-created) |
| `V2_REPO` | `anonymous-neurips-ED/CTSpinoPelvic1K` | public source of CT + pseudo |
| `SOURCE_REVISION` | `v2` | branch holding the pseudo labels |
| `REVIEWER_KEYS` | `{"k_alice":{"id":"alice","role":"primary"},"k_snr":{"id":"snr","role":"adjudicator"}}` | API-key → reviewer map |
| `TAU` | `0.9` | IRR agreement threshold (per-class min-Dice) |
| `IRR_MODE` | `per_class_min` | or `overall` |

On first boot it seeds one review case per `spine_only` / `pelvic_native`
record in the v2 `manifest.json` (idempotent).

For **local dev** (no HF), set `LOCAL_STORE_DIR=/tmp/reviewstore` and run:
```bash
pip install -r review_service/requirements.txt
LOCAL_STORE_DIR=/tmp/rs V2_REPO=org/X REVIEWER_KEYS='{"k":{"id":"a","role":"adjudicator"}}' \
  uvicorn review_service.app:app --port 7860
```

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

Auth: `Authorization: Bearer <api_key>`.

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
