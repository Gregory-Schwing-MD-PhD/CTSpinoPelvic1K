"""
review_service/store.py — persistence for the review service.

A small backend abstraction (text/bytes/list/exists) so the domain layer
(ReviewStore) is identical whether state lives on a local filesystem
(LocalBackend — dev + tests) or in a private HuggingFace dataset repo
(HFBackend — production, the source of truth). The repo layout:

    cases/<case_id>.json                 claim/slot/status + case metadata
    reviews/<case_id>/<review_id>.json   one immutable review record
    reviews/<case_id>/<slot>_label.nii.gz  the reviewer's corrected label
    reviews/<case_id>/final_label.nii.gz   the finalized label (v3 source)
    finals.json                          aggregated index for reduce_to_v3
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import List, Optional

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from review import schema  # noqa: E402


# ── backends ─────────────────────────────────────────────────────────────────

class LocalBackend:
    """Filesystem-backed store (dev + tests)."""

    def __init__(self, root):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _p(self, path: str) -> Path:
        return self.root / path

    def exists(self, path: str) -> bool:
        return self._p(path).exists()

    def read_text(self, path: str) -> Optional[str]:
        p = self._p(path)
        return p.read_text() if p.exists() else None

    def write_text(self, path: str, text: str) -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def read_bytes(self, path: str) -> Optional[bytes]:
        p = self._p(path)
        return p.read_bytes() if p.exists() else None

    def write_bytes(self, path: str, data: bytes) -> None:
        p = self._p(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)

    def write_many(self, files: dict, commit_message: str = "") -> None:
        for path, data in files.items():
            if isinstance(data, str):
                self.write_text(path, data)
            else:
                self.write_bytes(path, data)

    def list(self, prefix: str) -> List[str]:
        base = self._p(prefix)
        if not base.exists():
            return []
        return sorted(
            str(p.relative_to(self.root)).replace("\\", "/")
            for p in base.rglob("*") if p.is_file()
        )


class HFBackend:
    """Private HF dataset repo as the store (production).

    Each write is a commit; reads go through the HF cache. Fine for a
    small reviewer team. NOTE: not exercised by local tests (needs network
    + a token); the domain logic is validated via LocalBackend.
    """

    def __init__(self, repo_id: str, token: str, repo_type: str = "dataset"):
        from huggingface_hub import HfApi, create_repo
        self.api = HfApi(token=token)
        self.repo_id = repo_id
        self.repo_type = repo_type
        self.token = token
        create_repo(repo_id=repo_id, repo_type=repo_type, private=True,
                    exist_ok=True, token=token)

    def exists(self, path: str) -> bool:
        try:
            return path in self.api.list_repo_files(
                repo_id=self.repo_id, repo_type=self.repo_type,
                token=self.token)
        except Exception:
            return False

    def read_bytes(self, path: str) -> Optional[bytes]:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.utils import EntryNotFoundError
        try:
            local = hf_hub_download(
                repo_id=self.repo_id, repo_type=self.repo_type,
                filename=path, token=self.token,
                force_download=True)        # always fetch latest
            return Path(local).read_bytes()
        except (EntryNotFoundError, Exception):
            return None

    def read_text(self, path: str) -> Optional[str]:
        b = self.read_bytes(path)
        return b.decode("utf-8") if b is not None else None

    def write_bytes(self, path: str, data: bytes) -> None:
        self.api.upload_file(
            path_or_fileobj=data, path_in_repo=path,
            repo_id=self.repo_id, repo_type=self.repo_type, token=self.token,
            commit_message=f"review: write {path}")

    def write_text(self, path: str, text: str) -> None:
        self.write_bytes(path, text.encode("utf-8"))

    def write_many(self, files: dict, commit_message: str = "review: batch write") -> None:
        """Write many files in ONE commit.

        Every other write here is its own commit (upload_file). HF caps
        repo commits at 128/hour, so seeding 128 cases as 128 separate
        commits trips a 429 mid-seed. This collapses a batch into a single
        create_commit. `files` maps path_in_repo -> str | bytes.
        """
        from huggingface_hub import CommitOperationAdd
        ops = []
        for path, data in files.items():
            if isinstance(data, str):
                data = data.encode("utf-8")
            ops.append(CommitOperationAdd(path_in_repo=path, path_or_fileobj=data))
        if not ops:
            return
        self.api.create_commit(
            repo_id=self.repo_id, repo_type=self.repo_type, token=self.token,
            operations=ops, commit_message=commit_message)

    def list(self, prefix: str) -> List[str]:
        try:
            return sorted(
                f for f in self.api.list_repo_files(
                    repo_id=self.repo_id, repo_type=self.repo_type,
                    token=self.token)
                if f.startswith(prefix))
        except Exception:
            return []


# ── domain store ─────────────────────────────────────────────────────────────

class ReviewStore:
    """Domain operations over a backend. JSON in/out, blob put/get."""

    def __init__(self, backend):
        self.b = backend

    # cases ------------------------------------------------------------------
    def case_path(self, case_id: str) -> str:
        return f"cases/{case_id}.json"

    def get_case(self, case_id: str) -> Optional[dict]:
        t = self.b.read_text(self.case_path(case_id))
        return json.loads(t) if t else None

    def put_case(self, case: dict) -> None:
        self.b.write_text(self.case_path(case["case_id"]),
                          json.dumps(case, indent=2))

    def put_cases(self, cases: List[dict]) -> None:
        """Persist many cases in a SINGLE backend commit (see write_many)."""
        files = {self.case_path(c["case_id"]): json.dumps(c, indent=2)
                 for c in cases}
        if files:
            self.b.write_many(files,
                              commit_message=f"review: seed {len(files)} cases")

    def list_cases(self) -> List[dict]:
        out = []
        for p in self.b.list("cases/"):
            if p.endswith(".json"):
                t = self.b.read_text(p)
                if t:
                    out.append(json.loads(t))
        return out

    # reviews + labels -------------------------------------------------------
    def put_review(self, record: dict) -> str:
        cid = schema.case_id(record["token"], record["config"])
        path = f"reviews/{cid}/{record['review_id']}.json"
        self.b.write_text(path, json.dumps(record, indent=2))
        return path

    def put_label(self, case_id: str, name: str, data: bytes) -> str:
        path = f"reviews/{case_id}/{name}"
        self.b.write_bytes(path, data)
        return path

    def get_label_bytes(self, path: str) -> Optional[bytes]:
        return self.b.read_bytes(path)

    # finalized index --------------------------------------------------------
    def put_finals(self, finals: dict) -> None:
        self.b.write_text("finals.json", json.dumps(finals, indent=2))

    def get_finals(self) -> dict:
        t = self.b.read_text("finals.json")
        return json.loads(t) if t else {}


def init_cases_from_manifest(store: ReviewStore, records: List[dict],
                             source_revision: str = "v2",
                             crops_index: Optional[dict] = None) -> int:
    """Create one review case per scoped (spine_only/pelvic_native) record.

    Idempotent: never clobbers a case that already has claims/reviews. The
    `region_to_review` is the pseudo-filled side (the only thing reviewers
    touch); priority defaults to 0 (raise it later for low-confidence).

    TRIAGE + CROPS: if `crops_index` (keyed by pseudo label_file -> crop entry)
    is given, ONLY the cases in it are seeded — i.e. the QC-flagged worklist —
    and each case carries a `crop` block (small ct/seg crop paths + voxel
    origin) so the client can review a few-MB crop and paste the edit back to
    full-res. Without it, the full manifest is seeded as before.

    All new cases are written in a SINGLE commit (store.put_cases). Writing
    one commit per case trips HF's 128-commits/hour limit on first boot;
    existence is checked from a single file LIST (not 128 downloads) so a
    re-boot is cheap and never re-commits cases that already exist.
    """
    existing = {p[len("cases/"):-len(".json")]
                for p in store.b.list("cases/")
                if p.startswith("cases/") and p.endswith(".json")}
    new_cases = []
    for rec in records:
        cfg = rec.get("config")
        # fused = radiologist gold on BOTH regions; "both" means re-check the whole
        # label. We only enqueue fused cases that the QC FLAGGED (i.e. in a
        # crops_index) — never the full gold set.
        region = {"spine_only": "pelvis", "pelvic_native": "spine",
                  "fused": "both"}.get(cfg)
        if region is None:                       # out of scope
            continue
        if cfg == "fused" and crops_index is None:
            continue                             # don't review gold unless triaged
        if crops_index is not None and rec.get("label_file") not in crops_index:
            continue                             # triage: only the flagged worklist
        cid = schema.case_id(rec.get("token"), cfg)
        if cid in existing:                      # don't overwrite live state
            continue
        case = {
            "case_id": cid,
            "token": str(rec.get("token")),
            "config": cfg,
            "stratum": rec.get("lstv_label") or "normal",
            "priority": 0,
            "source_revision": source_revision,
            "ct_file": rec.get("ct_file"),
            "pseudo_label_file": rec.get("label_file"),
            "region_to_review": region,
            "prov_before": {"spine": rec.get("prov_spine"),
                            "pelvis": rec.get("prov_pelvis")},
            "slots": {},
            "final": None,
        }
        if crops_index is not None:
            e = crops_index[rec["label_file"]]
            case["crop"] = {"ct_crop": e["ct_crop"], "seg_crop": e["seg_crop"],
                            "origin": e["origin"],
                            # LSTV phenotype -> reviewtool warns to count levels
                            # (transitional vertebra is where the draft duplicates).
                            "lstv_class": int(rec.get("lstv_class") or 0),
                            "lstv_label": rec.get("lstv_label") or "normal"}
        new_cases.append(case)
    store.put_cases(new_cases)                   # single commit (no-op if empty)
    return len(new_cases)


# rib-anchor (v4) seeding ----------------------------------------------------
# Only configs with REAL radiologist spine GT are eligible. The anchor is
# defined relationally as "the vertebra directly above ground-truth L1" — so it
# is only trustworthy where L1 itself is radiologist truth. `fused` and
# `spine_only` carry the CTSpine1K spine GT; `pelvic_native` has a PSEUDOLABELLED
# spine (its L1 is a model guess), so its anchor would be untrustworthy — those
# cases are dropped from the numbering task entirely.
RIB_ANCHOR_CONFIGS = frozenset({"fused", "spine_only"})


def init_rib_anchor_cases(store: ReviewStore, records: List[dict],
                          source_revision: str = "v3",
                          include_configs: frozenset = RIB_ANCHOR_CONFIGS) -> int:
    """Seed the v4 rib-anchor task: one case per spine-GT v3 record.

    Unlike the pseudo-label review (init_cases_from_manifest), this serves the
    EXISTING v3 label as the editable base — the anchor (`last_rib_vertebra` 11
    + `rib` 12) is defined as "the vertebra above ground-truth L1"; reviewers
    confirm/segment it and may tidy class-mixing / partly coloured vertebrae
    (docs/RIB_ANCHOR_RATIONALE.md). region_to_review is "rib_anchor" so
    IRR/provenance treat it as the add-the-anchor pass.

    Only spine-GT configs are enqueued (fused / spine_only). `pelvic_native`
    cases have a pseudolabelled spine — an untrusted L1 — so their anchor cannot
    be trusted and they are excluded. Idempotent: never clobbers a case that
    already has claims/reviews. All new cases land in a SINGLE commit.
    """
    existing = {p[len("cases/"):-len(".json")]
                for p in store.b.list("cases/")
                if p.startswith("cases/") and p.endswith(".json")}
    new_cases = []
    for rec in records:
        cfg = rec.get("config")
        if cfg not in include_configs:
            continue
        if not rec.get("ct_file") or not rec.get("label_file"):
            continue                             # need both to serve + edit
        cid = schema.case_id(rec.get("token"), cfg)
        if cid in existing:                      # don't overwrite live state
            continue
        new_cases.append({
            "case_id": cid,
            "token": str(rec.get("token")),
            "config": cfg,
            "task": "rib_anchor",
            "stratum": rec.get("lstv_label") or "normal",
            "priority": 0,
            "source_revision": source_revision,
            "ct_file": rec.get("ct_file"),
            # the v3 label IS the base the student edits (adds 11/12 onto)
            "pseudo_label_file": rec.get("label_file"),
            "region_to_review": "rib_anchor",
            "prov_before": {"spine": rec.get("prov_spine"),
                            "pelvis": rec.get("prov_pelvis")},
            "slots": {},
            "final": None,
        })
    store.put_cases(new_cases)                   # single commit (no-op if empty)
    return len(new_cases)
