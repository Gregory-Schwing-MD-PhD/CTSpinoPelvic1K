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

    def delete_many(self, paths, commit_message: str = "") -> None:
        for path in paths:
            try:
                self._p(path).unlink()
            except OSError:
                pass

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

    def delete_many(self, paths, commit_message: str = "review: batch delete") -> None:
        """Delete many files in ONE commit (paths must exist; callers pass existing case files)."""
        from huggingface_hub import CommitOperationDelete
        ops = [CommitOperationDelete(path_in_repo=p) for p in paths]
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

    def prune_unassigned_not_in(self, keep_tokens, keep_region=None) -> int:
        """Self-heal: delete UNASSIGNED cases that are not current — token not in `keep_tokens`,
        or (if `keep_region` given) a different region_to_review. NEVER touches a case with a
        claim/review (status != 'unassigned'), so in-progress / finished work is preserved.
        Lets a stale ledger fix itself on boot instead of needing a manual wipe."""
        keep = {str(t) for t in (keep_tokens or set())}
        stale = []
        for c in self.list_cases():
            current = str(c.get("token")) in keep
            if keep_region is not None:
                current = current and c.get("region_to_review") == keep_region
            if current:
                continue
            if schema.derive_status(c) != "unassigned":
                continue                       # in-progress / done -> keep, never destroy work
            stale.append(self.case_path(c["case_id"]))
        if stale:
            self.b.delete_many(stale, commit_message=f"self-heal: prune {len(stale)} stale case(s)")
        return len(stale)

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


# v4 overlay-task seeding -----------------------------------------------------
# Only configs with REAL radiologist spine GT are eligible for ALL overlay tasks.
# Each overlay is anchored to ground-truth lumbar/sacral anatomy: the rib anchor
# is "the vertebra above GT L1"; rib NUMBERING is derived from GT thoracic
# costovertebral adjacency; the iliolumbar ligament arises from the GT L5
# transverse process; LS-nerve roots are read against GT L4/L5/S1. `fused` and
# `spine_only` carry the CTSpine1K spine GT; `pelvic_native` has a PSEUDOLABELLED
# spine (an untrusted L1), so its overlays can't be trusted — those cases are
# dropped from every overlay task.
SPINE_GT_CONFIGS = frozenset({"fused", "spine_only"})
RIB_ANCHOR_CONFIGS = SPINE_GT_CONFIGS            # back-compat alias


def init_overlay_cases(store: ReviewStore, records: List[dict], task: str,
                       source_revision: str = "v3",
                       include_configs: frozenset = SPINE_GT_CONFIGS) -> int:
    """Seed a v4 overlay task (one Space + ledger per task; see docs/annotation/).

    `task` is one of schema.OVERLAY_TASKS (rib_anchor | ribs | ls_nerve |
    iliolumbar). Unlike the pseudo-label review (init_cases_from_manifest), this
    serves the EXISTING v3 label as the editable base — the student ADDS the
    task's overlay structures onto it (and may tidy class-mixing / partial
    vertebrae). `region_to_review` is the task name, so IRR runs over the overlay
    classes and provenance treats it as an additive pass (the spine/pelvis source
    axes are unchanged — see schema.provenance_after).

    Only spine-GT configs are enqueued. Idempotent: never clobbers a case that
    already has claims/reviews. All new cases land in a SINGLE commit.
    """
    if task not in schema.OVERLAY_TASKS:
        raise ValueError(f"unknown overlay task {task!r}; expected one of "
                         f"{schema.OVERLAY_TASKS}")
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
            "task": task,
            "stratum": rec.get("lstv_label") or "normal",
            "priority": 0,
            "source_revision": source_revision,
            "ct_file": rec.get("ct_file"),
            # the v3 label IS the base the student edits (adds the overlay onto)
            "pseudo_label_file": rec.get("label_file"),
            "region_to_review": task,
            "prov_before": {"spine": rec.get("prov_spine"),
                            "pelvis": rec.get("prov_pelvis")},
            "slots": {},
            "final": None,
        })
    store.put_cases(new_cases)                   # single commit (no-op if empty)
    return len(new_cases)


def init_rib_anchor_cases(store: ReviewStore, records: List[dict],
                          source_revision: str = "v3",
                          include_configs: frozenset = SPINE_GT_CONFIGS) -> int:
    """Back-compat wrapper: seed the rib-anchor overlay (see init_overlay_cases)."""
    return init_overlay_cases(store, records, task="rib_anchor",
                              source_revision=source_revision,
                              include_configs=include_configs)


def init_rib_fix_cases(store: ReviewStore, records: List[dict],
                       worklist_tokens, source_revision: str = "v4") -> int:
    """Seed the v4 RIB-CORRECTION task: ONLY the QC-flagged duplicate cases (those whose
    manifest token is in `worklist_tokens`), serving the EXISTING v4 label as the editable
    base — the student corrects the rib in place (weld / relabel / delete). region_to_review
    is 'ribs' so IRR + the server-side QC gate run over the rib numbering. Unlike the overlay
    seeder this is NOT limited to spine-GT configs (ribs exist on every config). Idempotent;
    all new cases land in a SINGLE commit. Seeds nothing if `worklist_tokens` is empty (so a
    missing worklist can never accidentally enqueue all 802)."""
    want = {str(t) for t in (worklist_tokens or set())}
    if not want:
        return 0
    existing = {p[len("cases/"):-len(".json")]
                for p in store.b.list("cases/")
                if p.startswith("cases/") and p.endswith(".json")}
    new_cases = []
    for rec in records:
        if str(rec.get("token")) not in want:
            continue
        if not rec.get("ct_file") or not rec.get("label_file"):
            continue                             # need both to serve + edit
        cfg = rec.get("config")
        cid = schema.case_id(rec.get("token"), cfg)
        if cid in existing:                      # don't overwrite live state
            continue
        new_cases.append({
            "case_id": cid,
            "token": str(rec.get("token")),
            "config": cfg,
            "task": "ribs",
            "stratum": rec.get("lstv_label") or "normal",
            "priority": 0,
            "source_revision": source_revision,
            "ct_file": rec.get("ct_file"),
            # the v4 label IS the base the student corrects in place
            "pseudo_label_file": rec.get("label_file"),
            "region_to_review": "ribs",
            "prov_before": {"spine": rec.get("prov_spine"),
                            "pelvis": rec.get("prov_pelvis")},
            "slots": {},
            "final": None,
        })
    store.put_cases(new_cases)                   # single commit (no-op if empty)
    return len(new_cases)


def init_spine_extend_cases(store: ReviewStore, records: List[dict],
                            worklist_tokens, source_revision: str = "v4") -> int:
    """Seed the SPINE-EXTENSION task: ONLY the flagged cases (token in `worklist_tokens`), serving the
    v4 label as the editable base. The student ADDS thoracic vertebrae that are in the FOV but not yet
    labelled, numbering upward. region_to_review='spine' so IRR runs over the vertebrae; the server
    normalizer keeps ONLY their additions (existing GT is force-restored). Idempotent; single commit;
    seeds nothing if the worklist is empty (a missing worklist can never enqueue all 802)."""
    want = {str(t) for t in (worklist_tokens or set())}
    if not want:
        return 0
    existing = {p[len("cases/"):-len(".json")]
                for p in store.b.list("cases/")
                if p.startswith("cases/") and p.endswith(".json")}
    new_cases = []
    for rec in records:
        if str(rec.get("token")) not in want:
            continue
        if not rec.get("ct_file") or not rec.get("label_file"):
            continue
        cid = schema.case_id(rec.get("token"), rec.get("config"))
        if cid in existing:
            continue
        new_cases.append({
            "case_id": cid,
            "token": str(rec.get("token")),
            "config": rec.get("config"),
            "task": "spine",
            "stratum": rec.get("lstv_label") or "normal",
            "priority": 0,
            "source_revision": source_revision,
            "ct_file": rec.get("ct_file"),
            "pseudo_label_file": rec.get("label_file"),   # v4 label is the editable base
            "region_to_review": "spine",
            "prov_before": {"spine": rec.get("prov_spine"),
                            "pelvis": rec.get("prov_pelvis")},
            "slots": {},
            "final": None,
        })
    store.put_cases(new_cases)
    return len(new_cases)
