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
                             source_revision: str = "v2") -> int:
    """Create one review case per scoped (spine_only/pelvic_native) record.

    Idempotent: never clobbers a case that already has claims/reviews. The
    `region_to_review` is the pseudo-filled side (the only thing reviewers
    touch); priority defaults to 0 (raise it later for low-confidence).
    """
    n = 0
    for rec in records:
        cfg = rec.get("config")
        region = {"spine_only": "pelvis", "pelvic_native": "spine"}.get(cfg)
        if region is None:                       # fused / out of scope
            continue
        cid = schema.case_id(rec.get("token"), cfg)
        if store.get_case(cid):                  # don't overwrite live state
            continue
        store.put_case({
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
        })
        n += 1
    return n
