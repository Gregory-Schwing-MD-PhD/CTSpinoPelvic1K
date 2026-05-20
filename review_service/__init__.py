"""CTSpinoPelvic1K review service (Phase 2).

A small FastAPI app (deployable as a HuggingFace Space) that coordinates
distributed double-review + adjudication of the v2 pseudo labels. It holds
the single HF write token; reviewers authenticate with per-reviewer API
keys and never touch the token. Heavy NIfTIs are fetched by clients
directly from the public v2 repo; only corrected labels are uploaded
through the service into a private review repo, which is the source of
truth for claims/status/reviews.

Layers:
  store.py    pluggable persistence (LocalBackend for dev/tests; HFBackend
              = the private review repo for production) + domain ops.
  service.py  backend-agnostic claim/submit/IRR/adjudicate/status core,
              built on scripts/review (schema, diff). Fully unit-tested.
  app.py      thin FastAPI wrapper: auth, file upload, HTML dashboard.
"""
