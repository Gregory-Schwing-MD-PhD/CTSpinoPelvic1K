"""
review_service/service.py — backend-agnostic review orchestration.

The heart of the service: claim → submit → (auto-finalize | adjudicate),
built entirely on the Phase-1 contract (scripts/review: schema, diff) plus
a ReviewStore. No FastAPI, no network here, so it is fully unit-testable
with store.LocalBackend (see tests/test_review_service.py).

Double-review + adjudication:
  * each case has 2 primary slots; a reviewer can hold at most one (A≠B).
  * on the 2nd primary submit, IRR = per-class min-Dice between the two
    reviewers' resulting labels. ≥ τ → auto-finalize (keep the more
    conservative label); < τ → needs_adjudication.
  * an adjudicator submits the deciding label → finalize.
A reviewer uploads a resulting label for BOTH accept and corrected (on
accept it is the unchanged pseudo) so IRR is always computable from the
store; only `reject` carries no label.
"""
from __future__ import annotations

import io
import json
import sys
import tempfile
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from review import diff, schema           # noqa: E402
from review import labels_descriptor      # noqa: E402

from store import ReviewStore             # noqa: E402  (sibling)


class ReviewError(Exception):
    pass


def _descriptor_for_case(case) -> str:
    """ITK-SNAP palette matching the served labels: VerSe-native for v3/v4 (spine 1-28, ribs
    34-57) vs the v2 LSTV scheme (1-9). Keeps label names correct in the editor."""
    rev = str((case or {}).get("source_revision", "v2")).lstrip("vV")
    if (rev.isdigit() and int(rev) >= 3) or (case or {}).get("region_to_review") == "ribs":
        return labels_descriptor.verse_native_descriptor_text()
    return labels_descriptor.descriptor_text()


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _sha256(b: bytes) -> str:
    import hashlib
    return hashlib.sha256(b).hexdigest()


def _load_label_array(data: bytes, name: str):
    """Load a label volume from bytes. .npy (tests) or .nii.gz (production)."""
    import numpy as np
    if name.endswith(".npy"):
        return np.load(io.BytesIO(data))
    suffix = ".nii.gz" if name.endswith(".nii.gz") else Path(name).suffix
    import nibabel as nib
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        f.write(data)
        tmp = f.name
    try:
        return np.asarray(nib.load(tmp).dataobj)
    finally:
        Path(tmp).unlink(missing_ok=True)


class ReviewService:
    def __init__(self, store: ReviewStore, *, v2_repo: str,
                 source_revision: str = "v2", tau: float = diff.DEFAULT_TAU,
                 irr_mode: str = diff.DEFAULT_MODE, claim_ttl_seconds: int = 3 * 24 * 3600,
                 check: str = "none"):
        self.store = store
        self.v2_repo = v2_repo
        self.source_revision = source_revision
        self.tau = tau
        self.irr_mode = irr_mode
        self.ttl = claim_ttl_seconds
        # server-side anatomy QC gate run on every submitted label (none|spine|ribs|both).
        # "ribs" rejects any submission whose ribs still have a duplicate/split number.
        self.check = (check or "none").strip().lower()

    # ── helpers ─────────────────────────────────────────────────────────────
    def _blob_url(self, rel: str) -> str:
        return (f"https://huggingface.co/datasets/{self.v2_repo}/resolve/"
                f"{self.source_revision}/{rel}")

    @staticmethod
    def _mint_token(case_id: str, slot: str) -> str:
        return f"{case_id}::{slot}::{uuid.uuid4().hex}"

    @staticmethod
    def _parse_token(token: str) -> Tuple[str, str]:
        parts = token.split("::")
        if len(parts) != 3:
            raise ReviewError("malformed claim token")
        return parts[0], parts[1]      # case_id, slot

    def _read_label(self, path: Optional[str],
                    files: Optional[dict] = None) -> Optional[bytes]:
        """Label bytes, preferring this op's not-yet-committed `files` batch
        over the committed store — so IRR/finalize can read a label written
        earlier in the same submit, before its single commit lands."""
        if not path:
            return None
        if files and path in files:
            d = files[path]
            return bytes(d) if isinstance(d, (bytes, bytearray)) else None
        return self.store.get_label_bytes(path)

    def _agree(self, case: dict, files: Optional[dict] = None) -> Optional[bool]:
        """IRR between the two primary slots' labels (None if <2 labels)."""
        labels = []
        for s in ("1", "2"):
            slot = case["slots"].get(s, {})
            lp = slot.get("label_path")
            if not slot.get("done") or not lp:
                return None
            data = self._read_label(lp, files)
            if data is None:
                return None
            labels.append(_load_label_array(data, lp))
        r = diff.irr(labels[0], labels[1], tau=self.tau, mode=self.irr_mode)
        case["irr"] = {k: r[k] for k in ("metric", "min_class_dice",
                                         "mode", "tau", "agree")}
        return bool(r["agree"])

    def _qc_gate(self, label_bytes: Optional[bytes], label_name: str) -> None:
        """Server-side anatomy QC: REJECT a submission whose label fails self.check
        (e.g. 'ribs' -> no duplicate/split rib number). Fails CLOSED — if the QC
        cannot run, the submission is rejected, so nothing un-verified is committed."""
        if self.check in (None, "", "none") or label_bytes is None:
            return
        import os as _os
        import sys as _sys
        import tempfile
        for _p in ("scripts",                                  # Space layout: /scripts/*
                   _os.path.join(_os.path.dirname(__file__), "..", "scripts")):
            if _os.path.isdir(_p) and _p not in _sys.path:
                _sys.path.insert(0, _p)
        try:
            import numpy as np
            import nibabel as nib
            import review_anatomy_qc as RA
        except Exception as exc:                               # cannot verify -> fail closed
            raise ReviewError(f"server QC unavailable ({exc}); submission rejected")
        suffix = ".nii.gz" if label_name.endswith(".gz") else ".nii"
        tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
        try:
            tf.write(label_bytes); tf.close()
            img = nib.load(tf.name)
            ok, msgs = RA.check_label(self.check, np.asanyarray(img.dataobj), img.affine)
        except Exception as exc:
            raise ReviewError(f"server QC failed to run ({exc}); submission rejected")
        finally:
            try:
                _os.unlink(tf.name)
            except OSError:
                pass
        if not ok:
            bad = "; ".join(m for m in msgs if m.startswith("X")) or "; ".join(msgs)
            raise ReviewError(f"label fails {self.check} QC: {bad}")

    # ── claim ───────────────────────────────────────────────────────────────
    def claim(self, reviewer_id: str) -> Optional[dict]:
        """Assign this reviewer an open primary slot (priority-first), or
        None if nothing is claimable by them right now."""
        now_s = schema.utcnow()
        candidates = []
        for case in self.store.list_cases():
            if case.get("final"):
                continue
            slot = schema.claimable_primary_slot(case, reviewer_id, now=now_s)
            if slot is None:
                continue
            status = schema.derive_status(case)
            # prefer untouched cases, then by priority, then case_id
            order = (0 if status == "unassigned" else 1,
                     -int(case.get("priority", 0)), case["case_id"])
            candidates.append((order, case, slot))
        if not candidates:
            return None
        candidates.sort(key=lambda x: x[0])
        _, case, slot = candidates[0]

        token = self._mint_token(case["case_id"], slot)
        expires = (_now() + timedelta(seconds=self.ttl)).isoformat()
        case.setdefault("slots", {})[slot] = {
            "reviewer": reviewer_id, "claimed_at": now_s,
            "expires_at": expires, "claim_token": token,
            "done": False, "review_id": None, "decision": None,
            "label_path": None,
        }
        self.store.put_case(case)
        return {
            "case_id": case["case_id"], "token": case["token"],
            "config": case["config"], "slot": slot,
            "region_to_review": case["region_to_review"],
            "claim_token": token, "expires_at": expires,
            "v2_repo": self.v2_repo, "source_revision": self.source_revision,
            "ct_file": case["ct_file"], "label_file": case["pseudo_label_file"],
            "ct_url": self._blob_url(case["ct_file"]),
            "pseudo_label_url": self._blob_url(case["pseudo_label_file"]),
            "source_label_sha256": case.get("source_label_sha256", ""),
            "prov_before": case["prov_before"],
            "labels_descriptor": _descriptor_for_case(case),
            # Triage-crop review: small ct/seg crop + voxel origin so the client
            # downloads a few MB and pastes its edit back into the full label.
            "crop": case.get("crop"),
        }

    def amend_next(self, reviewer_id: str) -> Optional[dict]:
        """Serve this reviewer their next slot that was RE-OPENED for amendment (its earlier
        submission failed the strengthened QC). Bypasses the double-review distinctness guard
        (they are fixing THEIR OWN slot) and hands back their own previous label as the editable
        base, not the pseudo. Resubmit goes through the normal /submit QC gate (fails closed)."""
        for case in self.store.list_cases():
            # We are only running the RIB correction task now. Never re-serve a leftover
            # spine/pelvis pseudolabel amend (its base label lives elsewhere and 404s the client).
            if case.get("region_to_review") != "ribs":
                continue
            for slot in schema.PRIMARY_SLOTS:
                sl = case.get("slots", {}).get(slot)
                if (not sl or sl.get("done") or not sl.get("amend")
                        or sl.get("reviewer") != reviewer_id):
                    continue
                token = self._mint_token(case["case_id"], slot)
                sl["claim_token"] = token
                sl["claimed_at"] = schema.utcnow()
                sl["expires_at"] = (_now() + timedelta(seconds=self.ttl)).isoformat()
                self.store.put_case(case)
                base = sl.get("amend_base") or case["pseudo_label_file"]
                return {
                    "case_id": case["case_id"], "token": case["token"],
                    "config": case["config"], "slot": slot,
                    "region_to_review": case["region_to_review"],
                    "claim_token": token, "expires_at": sl["expires_at"],
                    "v2_repo": self.v2_repo, "source_revision": self.source_revision,
                    "ct_file": case["ct_file"], "label_file": base,
                    "ct_url": self._blob_url(case["ct_file"]),
                    "pseudo_label_url": self._blob_url(base),
                    "source_label_sha256": case.get("source_label_sha256", ""),
                    "prov_before": case["prov_before"],
                    "labels_descriptor": _descriptor_for_case(case),
                    "crop": case.get("crop"),
                    "amend": True, "amend_reason": sl.get("amend_reason", ""),
                    # public-repo fallback if the private amend base can't be streamed
                    "orig_pseudo_file": case["pseudo_label_file"],
                }
        return None

    def amend_base_bytes(self, reviewer_id: str, case_id: str, slot: str) -> bytes:
        """Stream a reviewer their OWN earlier submission (the amend base) from the PRIVATE review
        repo. Students sign in with their own HF login and CANNOT read that repo directly, so the
        Space (which holds the write token) reads the label and hands it back. Scoped to the caller's
        own slot only."""
        case = self.store.get_case(case_id)
        if case is None:
            raise ReviewError(f"unknown case {case_id}")
        sl = case.get("slots", {}).get(slot)
        if not sl or sl.get("reviewer") != reviewer_id:
            raise ReviewError("that slot is not yours to amend")
        rel = sl.get("amend_base") or f"reviews/{case_id}/{slot}_label.nii.gz"
        data = self.store.b.read_bytes(rel)
        if data is None:
            raise ReviewError(f"amend base not found in review repo: {rel}")
        return data

    def defer(self, claim_token: str) -> dict:
        """A reviewer declines a scan: release their claim so the case returns to the queue for
        SOMEONE ELSE. The reviewer is recorded in `deferred_by` so they are never re-served it."""
        case_id, slot = self._parse_token(claim_token)
        case = self.store.get_case(case_id)
        if case is None:
            raise ReviewError(f"unknown case {case_id}")
        sl = case.get("slots", {}).get(slot)
        if not sl or sl.get("claim_token") != claim_token:
            raise ReviewError("claim token does not match an open claim")
        if sl.get("done"):
            raise ReviewError("slot already submitted; cannot defer")
        reviewer = sl.get("reviewer")
        dby = case.setdefault("deferred_by", [])
        if reviewer and reviewer not in dby:
            dby.append(reviewer)
        case.get("slots", {}).pop(slot, None)          # release -> case re-enters the queue
        self.store.put_case(case)
        return {"case_id": case_id, "deferred": True, "status": schema.derive_status(case)}

    def me_stats(self, reviewer_id: str) -> dict:
        """A reviewer's OWN progress (private, self-service): how many of their submissions passed
        the strengthened QC, the pass %, and how many are re-opened for them to amend. Reads the
        per-slot QC verdict stamped at seed/submit time -- no heavy recompute."""
        import collections as _c
        total = passed = amend_pending = 0
        fails = _c.Counter()
        for case in self.store.list_cases():
            for slot in schema.PRIMARY_SLOTS:
                sl = case.get("slots", {}).get(slot)
                if not sl or sl.get("reviewer") != reviewer_id or "qc_pass" not in sl:
                    continue
                total += 1
                passed += bool(sl.get("qc_pass"))
                if sl.get("amend") and not sl.get("done"):
                    amend_pending += 1
                for c in (sl.get("qc_fail_checks") or []):
                    fails[c] += 1
        return {"reviewer": reviewer_id, "submissions": total, "passed": passed,
                "pass_pct": (round(100 * passed / total) if total else None),
                "amend_pending": amend_pending, "fail_by_check": dict(fails)}

    # ── submit ──────────────────────────────────────────────────────────────
    def submit(self, claim_token: str, record: dict,
               label_bytes: Optional[bytes] = None,
               label_name: str = "label.nii.gz") -> dict:
        case_id, slot = self._parse_token(claim_token)
        case = self.store.get_case(case_id)
        if case is None:
            raise ReviewError(f"unknown case {case_id}")
        sl = case.get("slots", {}).get(slot)
        if not sl or sl.get("claim_token") != claim_token:
            raise ReviewError("claim token does not match an open claim")

        new_sha = _sha256(label_bytes) if label_bytes is not None else None
        if sl.get("done"):
            # Idempotent retry: a client re-sending the SAME work (e.g. after a
            # lost response / network blip) must succeed, not error — that is
            # what lets `reviewtool resume` recover without losing a finished
            # edit. A resubmit with DIFFERENT content under a done slot is a
            # real conflict and still raises.
            if sl.get("label_sha256") == new_sha:
                return {"case_id": case_id, "duplicate": True,
                        "status": schema.derive_status(case, case.get("agree")),
                        "irr": case.get("irr")}
            raise ReviewError("slot already submitted with different content")

        region = case["region_to_review"]
        decision = record.get("decision", "accept")
        record = dict(record)
        record.setdefault("review_id",
                           schema.review_id(case["token"], case["config"],
                                            sl["reviewer"]))
        record.update(token=case["token"], config=case["config"],
                       source_revision=self.source_revision,
                       reviewer_id=sl["reviewer"], role="primary",
                       region_reviewed=region,
                       prov_before=case["prov_before"])
        record["prov_after"] = schema.provenance_after(
            case["prov_before"], region, decision)

        # Accumulate every file this submit writes; commit them in ONE atomic
        # commit (was 2-4 separate commits -> 2-4x the HF commit-rate cost, and
        # a partial-write window between them).
        files: Dict[str, object] = {}

        label_path = None
        if decision in ("accept", "corrected"):
            if label_bytes is None:
                raise ReviewError(f"{decision} requires a label upload")
            self._qc_gate(label_bytes, label_name)       # reject if it fails QC (fails closed)
            label_path = f"reviews/{case_id}/{slot}_label{_ext(label_name)}"
            files[label_path] = label_bytes
            record["artifact"] = label_path
            record["corrected_label_sha256"] = new_sha
            if not record.get("diff"):
                record["diff"] = {"n_voxels_changed": 0}

        errs = schema.validate_review_record(record)
        if errs:
            raise ReviewError("invalid review record: " + "; ".join(errs))

        files[f"reviews/{case_id}/{record['review_id']}.json"] = \
            json.dumps(record, indent=2)
        sl.update(done=True, review_id=record["review_id"],
                  decision=decision, label_path=label_path,
                  label_sha256=new_sha, submitted_at=schema.utcnow())
        # it passed the QC gate to get here -> mark clean and clear any amend re-open
        sl["qc_pass"] = True
        sl["qc_fail_checks"] = []
        sl.pop("amend", None); sl.pop("amend_base", None); sl.pop("amend_reason", None)

        # evaluate once both primaries are in
        agree = None
        if len(schema.primary_done(case)) >= schema.N_PRIMARY:
            agree = self._agree(case, files)
            case["agree"] = agree          # persist so derive_status is stateless
            if agree is True:
                self._finalize_from_primaries(case, files)
        files[self.store.case_path(case_id)] = json.dumps(case, indent=2)
        self.store.b.write_many(
            files, commit_message=f"review: submit {case_id}/{slot}")
        return {"case_id": case_id, "status": schema.derive_status(case, agree),
                "irr": case.get("irr")}

    def _finalize_from_primaries(self, case: dict, files: dict) -> None:
        """Auto-finalize an agreed case: keep the more-conservative label
        (fewest voxels changed; tie -> slot 1). The final label is added to
        `files` so it rides the submit's single commit (reads prefer the
        pending batch, since the chosen label may be this submit's own)."""
        best_slot, best_changed = None, None
        for s in ("1", "2"):
            sl = case["slots"][s]
            rev = self._get_review(case, sl["review_id"], files)
            changed = (rev.get("diff", {}) or {}).get("n_voxels_changed", 0)
            if best_changed is None or changed < best_changed:
                best_slot, best_changed = s, changed
        sl = case["slots"][best_slot]
        rev = self._get_review(case, sl["review_id"], files)
        final_label = f"reviews/{case['case_id']}/final_label" \
                      f"{_ext(sl['label_path'] or '.nii.gz')}"
        data = self._read_label(sl["label_path"], files)
        if data is not None:
            files[final_label] = data
        decision = "corrected" if best_changed and best_changed > 0 else "accept"
        case["final"] = {
            "decision": decision,
            "prov_after": rev["prov_after"],
            "label_rel": final_label,
            "final_review_id": rev["review_id"],
            "by": "agreement", "at": schema.utcnow(),
            "irr": case.get("irr"),
        }

    def _get_review(self, case: dict, review_id: str,
                    files: Optional[dict] = None) -> dict:
        path = f"reviews/{case['case_id']}/{review_id}.json"
        if files and path in files:
            return json.loads(files[path])
        t = self.store.b.read_text(path)
        return json.loads(t) if t else {}

    # ── adjudication ─────────────────────────────────────────────────────────
    def adjudication_next(self, adjudicator_id: str) -> Optional[dict]:
        for case in self.store.list_cases():
            if schema.derive_status(case) != "needs_adjudication":
                continue
            adj = case["slots"].get(schema.ADJ_SLOT)
            if adj and adj.get("reviewer") != adjudicator_id \
                    and not _expired(adj):
                continue                       # someone else adjudicating
            token = self._mint_token(case["case_id"], schema.ADJ_SLOT)
            case["slots"][schema.ADJ_SLOT] = {
                "reviewer": adjudicator_id, "claimed_at": schema.utcnow(),
                "expires_at": (_now() + timedelta(seconds=self.ttl)).isoformat(),
                "claim_token": token, "done": False,
            }
            self.store.put_case(case)
            reviews = [self._get_review(case, case["slots"][s]["review_id"])
                       for s in ("1", "2")]
            return {
                "case_id": case["case_id"], "token": case["token"],
                "config": case["config"], "claim_token": token,
                "region_to_review": case["region_to_review"],
                "v2_repo": self.v2_repo, "source_revision": self.source_revision,
                "ct_file": case["ct_file"], "label_file": case["pseudo_label_file"],
                "ct_url": self._blob_url(case["ct_file"]),
                "pseudo_label_url": self._blob_url(case["pseudo_label_file"]),
                "irr": case.get("irr"),
                "reviews": reviews,
                "labels_descriptor": _descriptor_for_case(case),
            }
        return None

    def adjudicate(self, claim_token: str, decision: str,
                   label_bytes: Optional[bytes] = None,
                   label_name: str = "label.nii.gz",
                   notes: str = "") -> dict:
        case_id, slot = self._parse_token(claim_token)
        if slot != schema.ADJ_SLOT:
            raise ReviewError("not an adjudication token")
        case = self.store.get_case(case_id)
        if case is None:
            raise ReviewError(f"unknown case {case_id}")
        adj = case["slots"].get(schema.ADJ_SLOT)
        if not adj or adj.get("claim_token") != claim_token:
            raise ReviewError("claim token does not match the adjudication claim")
        if adj.get("done"):
            # Idempotent retry (see submit): a re-sent adjudication is a no-op.
            return {"case_id": case_id, "duplicate": True,
                    "status": schema.derive_status(case)}

        region = case["region_to_review"]
        files: Dict[str, object] = {}
        if decision == "reject":
            case["final"] = {"decision": "reject",
                             "prov_after": case["prov_before"],
                             "by": adj["reviewer"], "at": schema.utcnow(),
                             "notes": notes}
        else:
            if label_bytes is None:
                raise ReviewError("adjudication requires a final label")
            label_path = f"reviews/{case_id}/final_label{_ext(label_name)}"
            files[label_path] = label_bytes
            rec = schema.ReviewRecord(
                review_id=schema.review_id(case["token"], case["config"],
                                           adj["reviewer"], round=2),
                token=case["token"], config=case["config"],
                source_revision=self.source_revision,
                source_label_sha256=case.get("source_label_sha256", ""),
                reviewer_id=adj["reviewer"], role="adjudicator",
                decision="corrected", region_reviewed=region,
                diff={"n_voxels_changed": -1}, corrected_label_sha256=_sha256(label_bytes),
                artifact=label_path, prov_before=case["prov_before"],
                prov_after=schema.provenance_after(case["prov_before"], region,
                                                   "corrected"), notes=notes,
            ).to_dict()
            files[f"reviews/{case_id}/{rec['review_id']}.json"] = \
                json.dumps(rec, indent=2)
            case["final"] = {"decision": "corrected",
                             "prov_after": rec["prov_after"],
                             "label_rel": label_path,
                             "final_review_id": rec["review_id"],
                             "by": adj["reviewer"], "at": schema.utcnow(),
                             "irr": case.get("irr")}
        adj.update(done=True, submitted_at=schema.utcnow())
        files[self.store.case_path(case_id)] = json.dumps(case, indent=2)
        self.store.b.write_many(
            files, commit_message=f"review: adjudicate {case_id}")
        return {"case_id": case_id, "status": schema.derive_status(case)}

    # ── status + finals ───────────────────────────────────────────────────
    def status_summary(self) -> dict:
        from collections import Counter
        cases = self.store.list_cases()
        by_status = Counter(schema.derive_status(c) for c in cases)
        by_stratum = Counter(c.get("stratum", "?") for c in cases)
        by_reviewer: Counter = Counter()
        irr_metrics: List[float] = []
        for c in cases:
            for s in ("1", "2", schema.ADJ_SLOT):
                sl = c.get("slots", {}).get(s)
                if not sl or not sl.get("done"):
                    continue
                # public board shows only PASSED student submissions (+ adjudications);
                # a re-opened / QC-failing slot is not counted until the student fixes it.
                if s != schema.ADJ_SLOT and not sl.get("qc_pass"):
                    continue
                by_reviewer[sl.get("reviewer")] += 1
            if c.get("irr"):
                irr_metrics.append(c["irr"].get("metric"))
        return {
            "n_cases": len(cases),
            "by_status": dict(by_status),
            "by_stratum": dict(by_stratum),
            "reviews_by_reviewer": dict(by_reviewer),
            "n_irr_evaluated": len(irr_metrics),
            "irr_mean": (sum(irr_metrics) / len(irr_metrics)) if irr_metrics else None,
        }

    def build_finals(self) -> dict:
        finals = {}
        for c in self.store.list_cases():
            if c.get("final"):
                finals[c["case_id"]] = {
                    k: c["final"].get(k)
                    for k in ("decision", "prov_after", "label_rel",
                              "final_review_id")
                }
        self.store.put_finals(finals)
        return finals


def _ext(name: str) -> str:
    return ".nii.gz" if name.endswith(".nii.gz") else (Path(name).suffix or ".nii.gz")


def _expired(slot: dict) -> bool:
    exp = slot.get("expires_at")
    return bool(exp and exp < schema.utcnow() and not slot.get("done"))
