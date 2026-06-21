"""
review_service/app.py — FastAPI front-end for the review service.

Thin layer over service.ReviewService: per-reviewer API-key auth, multipart
label uploads, an in-process write lock (single-worker Space => atomic
claim/submit), and a tiny HTML dashboard. The HF write token lives only
here (server-side); reviewers send their API key, never the token.

Deploy as a HuggingFace Space (see Dockerfile + README). Configure via env:
  HF_TOKEN            write token for the private review repo (Space secret)
  REVIEW_REPO         org/CTSpinoPelvic1K-reviews  (private dataset)
  V2_REPO             org/CTSpinoPelvic1K          (public; source of CT+pseudo)
  SOURCE_REVISION     v2                            (branch holding pseudo)
  ADJUDICATORS        comma/space-separated HF usernames allowed to adjudicate;
                      any other HF-authenticated user is a primary reviewer
  REVIEWER_KEYS       (optional/legacy) JSON {"<api_key>":{"id":..,"role":..}};
                      bearer values matching a key here bypass HF identity
  TAU, IRR_MODE       agreement threshold + mode (default 0.9 / per_class_min)
  LOCAL_STORE_DIR     optional: use a local dir instead of HFBackend (dev)

Auth: reviewers send their own HuggingFace token (from `hf auth login`) as the
bearer; the Space verifies it with `whoami` and uses the returned username as
their identity (open mode — any HF user may review). The Space holds the only
dataset WRITE token; reviewers never see it.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import threading
import time
from pathlib import Path
from typing import Optional

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from fastapi import (Depends, FastAPI, Form, Header, HTTPException,  # noqa: E402
                     UploadFile, File)
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

import store as store_mod      # noqa: E402
import service as svc          # noqa: E402
from review import schema      # noqa: E402   (overlay-task registry)

# ── config + wiring ──────────────────────────────────────────────────────────

def _load_keys() -> dict:
    raw = os.environ.get("REVIEWER_KEYS")
    if not raw and os.environ.get("REVIEWER_KEYS_FILE"):
        raw = Path(os.environ["REVIEWER_KEYS_FILE"]).read_text()
    return json.loads(raw) if raw else {}


def _build_service() -> svc.ReviewService:
    local = os.environ.get("LOCAL_STORE_DIR")
    if local:
        backend = store_mod.LocalBackend(local)
    else:
        backend = store_mod.HFBackend(
            repo_id=os.environ["REVIEW_REPO"],
            token=os.environ["HF_TOKEN"])
    store = store_mod.ReviewStore(backend)
    return svc.ReviewService(
        store,
        v2_repo=os.environ.get("V2_REPO", "org/CTSpinoPelvic1K"),
        source_revision=os.environ.get("SOURCE_REVISION", "v2"),
        tau=float(os.environ.get("TAU", svc.diff.DEFAULT_TAU)),
        irr_mode=os.environ.get("IRR_MODE", svc.diff.DEFAULT_MODE),
        claim_ttl_seconds=int(os.environ.get("CLAIM_TTL_SECONDS", "7200")),
    )


app = FastAPI(title="CTSpinoPelvic1K review service")
KEYS = _load_keys()                       # legacy minted keys (optional)
ADJUDICATORS = {                          # HF usernames allowed to adjudicate
    u.strip().lower()
    for u in os.environ.get("ADJUDICATORS", "").replace(",", " ").split()
    if u.strip()
}
_WHOAMI_CACHE: dict = {}                   # sha256(token) -> (username, ts)
_WHOAMI_TTL = 600                          # re-verify a token every 10 min
SERVICE: Optional[svc.ReviewService] = None
_LOCK = threading.Lock()       # serialize writes (single-worker Space)


# TASK selects what this Space serves (run ONE Space per task — see
# docs/annotation/ and docs/REVIEW.md for the per-task Space + ledger map):
#   "lstv"       (default) — correct v2 pseudo-labels (init_cases_from_manifest,
#                 optional crops/crops_index triage). SOURCE_REVISION defaults v2.
#   v4 overlays (schema.OVERLAY_TASKS) — ADD structures onto the v3 label
#   (init_overlay_cases); point SOURCE_REVISION at v3:
#     "rib_anchor" minimal LSTV rostral anchor (11/12)
#     "ribs"       per-rib segmentation, reusing reserved ids 26–49
#     "ls_nerve"   L4/L5/S1 nerve roots (53–58)
#     "iliolumbar" iliolumbar ligament (51/52)
TASK = os.environ.get("TASK", "lstv").strip().lower()


@app.on_event("startup")
def _startup():
    global SERVICE
    SERVICE = _build_service()
    # Seed / gap-fill from the manifest on EVERY boot. Both seeders are
    # idempotent and cheap (one repo LIST + a single batched commit of ONLY the
    # missing cases, or a no-op if none). Running it unconditionally self-heals
    # a partial seed — e.g. a first seed truncated by HF's commit-rate limit —
    # which a "seed only if the store is empty" guard would leave incomplete.
    try:
        from huggingface_hub import hf_hub_download
        mp = hf_hub_download(
            repo_id=SERVICE.v2_repo, repo_type="dataset",
            filename="manifest.json", revision=SERVICE.source_revision)
        data = json.loads(Path(mp).read_text())
        recs = data if isinstance(data, list) else data.get("records", [])

        if TASK in schema.OVERLAY_TASKS:
            # v4 overlay pass (rib_anchor | ribs | ls_nerve | iliolumbar): serve
            # the v3 label as the editable base; the student ADDS this task's
            # overlay onto it. Point SOURCE_REVISION at v3 (one Space per task).
            n = store_mod.init_overlay_cases(
                SERVICE.store, recs, task=TASK,
                source_revision=SERVICE.source_revision)
            tag = f"{TASK} (v4 overlay) case(s) from {SERVICE.v2_repo}@{SERVICE.source_revision}"
        else:
            # TRIAGE: if the repo carries crops/crops_index.json (the QC-flagged
            # worklist), seed ONLY those cases and attach their review-crop info.
            crops_index = None
            try:
                cp = hf_hub_download(repo_id=SERVICE.v2_repo, repo_type="dataset",
                                     filename="crops/crops_index.json",
                                     revision=SERVICE.source_revision)
                crops_index = {e["label_file"]: e
                               for e in json.loads(Path(cp).read_text())}
                print(f"[startup] crops_index: triaging to {len(crops_index)} flagged case(s)")
            except Exception:
                print("[startup] no crops/crops_index.json — seeding the full manifest")
            n = store_mod.init_cases_from_manifest(
                SERVICE.store, recs, source_revision=SERVICE.source_revision,
                crops_index=crops_index)
            tag = f"review case(s) from {SERVICE.v2_repo}"

        if n:
            print(f"[startup] task={TASK}: seeded {n} new {tag}")
        else:
            print(f"[startup] task={TASK}: all cases already present; nothing to seed")
    except Exception as e:                           # noqa: BLE001
        print(f"[startup] could not seed/gap-fill cases: {e}")


def _hf_username(token: str) -> Optional[str]:
    """Verified HuggingFace username for a token (cached), or None if invalid.

    The token is the reviewer's own (read-scoped) HF token; we call whoami to
    confirm identity and only cache sha256(token) -> username, never the token."""
    h = hashlib.sha256(token.encode()).hexdigest()
    now = time.time()
    hit = _WHOAMI_CACHE.get(h)
    if hit and now - hit[1] < _WHOAMI_TTL:
        return hit[0]
    try:
        from huggingface_hub import whoami as hf_whoami
        name = hf_whoami(token=token).get("name")
    except Exception:                       # invalid token / network
        return None
    if name:
        _WHOAMI_CACHE[h] = (name, now)
    return name


def auth(authorization: str = Header(default="")) -> dict:
    cred = authorization[7:] if authorization.lower().startswith("bearer ") \
        else authorization
    cred = cred.strip()
    if not cred:
        raise HTTPException(401, "missing credentials")
    who = KEYS.get(cred)                    # 1) legacy minted reviewer key
    if who:
        return who                          # {"id":..., "role":...}
    username = _hf_username(cred)           # 2) HuggingFace identity (open mode)
    if not username:
        raise HTTPException(
            401, "invalid credentials — run `hf auth login`, then "
                 "`reviewtool login --service <url>`")
    role = "adjudicator" if username.lower() in ADJUDICATORS else "primary"
    return {"id": username, "role": role}


def _require(role: str, who: dict):
    if role == "adjudicator" and who.get("role") != "adjudicator":
        raise HTTPException(403, "adjudicator role required")


# ── endpoints ────────────────────────────────────────────────────────────────

@app.post("/claim")
def claim(who: dict = Depends(auth)):
    with _LOCK:
        out = SERVICE.claim(who["id"])
    if out is None:
        return JSONResponse({"detail": "nothing to claim"}, status_code=204)
    return out


@app.post("/submit")
async def submit(claim_token: str = Form(...), record: str = Form(...),
                 label: Optional[UploadFile] = File(default=None),
                 who: dict = Depends(auth)):
    data = await label.read() if label is not None else None
    name = label.filename if label is not None else "label.nii.gz"
    try:
        with _LOCK:
            return SERVICE.submit(claim_token, json.loads(record), data, name)
    except svc.ReviewError as e:
        raise HTTPException(400, str(e))


@app.get("/adjudication/next")
def adjudication_next(who: dict = Depends(auth)):
    _require("adjudicator", who)
    with _LOCK:
        out = SERVICE.adjudication_next(who["id"])
    if out is None:
        return JSONResponse({"detail": "nothing to adjudicate"}, status_code=204)
    return out


@app.post("/adjudicate")
async def adjudicate(claim_token: str = Form(...), decision: str = Form(...),
                     notes: str = Form(default=""),
                     label: Optional[UploadFile] = File(default=None),
                     who: dict = Depends(auth)):
    _require("adjudicator", who)
    data = await label.read() if label is not None else None
    name = label.filename if label is not None else "label.nii.gz"
    try:
        with _LOCK:
            return SERVICE.adjudicate(claim_token, decision, data, name, notes)
    except svc.ReviewError as e:
        raise HTTPException(400, str(e))


@app.post("/admin/build_finals")
def build_finals(who: dict = Depends(auth)):
    _require("adjudicator", who)
    with _LOCK:
        return SERVICE.build_finals()


@app.get("/status")
def status(who: dict = Depends(auth)):
    return SERVICE.status_summary()


@app.get("/", response_class=HTMLResponse)
def dashboard():
    s = SERVICE.status_summary()
    rows = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                   for k, v in s["by_status"].items())
    revs = "".join(f"<tr><td>{k}</td><td>{v}</td></tr>"
                   for k, v in s["reviews_by_reviewer"].items())
    irr = "—" if s["irr_mean"] is None else f"{s['irr_mean']:.3f}"
    return f"""<!doctype html><meta charset=utf-8>
<title>CTSpinoPelvic1K review</title>
<style>body{{font:14px system-ui;margin:2rem}}table{{border-collapse:collapse}}
td,th{{border:1px solid #ccc;padding:.3rem .6rem}}</style>
<h1>CTSpinoPelvic1K review</h1>
<p>{s['n_cases']} cases · mean IRR {irr} ({s['n_irr_evaluated']} evaluated)</p>
<h3>By status</h3><table><tr><th>status</th><th>n</th></tr>{rows}</table>
<h3>Reviews by reviewer</h3><table><tr><th>reviewer</th><th>n</th></tr>{revs}</table>
"""
