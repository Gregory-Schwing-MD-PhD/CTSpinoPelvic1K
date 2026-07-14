"""
reviewtool/cli.py — annotator-facing CLI.

Commands:
  reviewtool login   --service URL          (uses your `hf auth login` identity)
  reviewtool next    [--workdir DIR] [--itksnap itksnap]   # claim→edit→submit
  reviewtool adjudicate [...]                               # disagreements
  reviewtool status
  reviewtool resume                                         # re-upload saved edits

All HF download/upload is hidden: CT + pseudo come straight from the public
v2 repo; the corrected label is uploaded *through the Space*. The reviewer
authenticates with their own HuggingFace login (`hf auth login`) — the Space
verifies the username and holds the only dataset *write* token, which the
reviewer never sees. (A legacy `--key` is still accepted if configured.)

The decision logic (`build_submission`) is pure and unit-tested; everything
else is I/O glue (subprocess + HTTP).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
from review import diff, labels_descriptor, schema  # noqa: E402
from export_review_crops import crop_dirname  # noqa: E402

CONFIG = Path.home() / ".reviewtool" / "config.json"

# An "active claim" is persisted the moment a case is claimed (before any
# edit), so a finished segmentation is NEVER lost to a flaky upload, a crash,
# or a HuggingFace commit-rate 429: the edited seg.nii.gz stays in its workdir
# and `reviewtool resume` re-uploads it. Submit is idempotent server-side, so
# replaying work the server already recorded is a safe no-op.
ACTIVE_DIR = Path.home() / ".reviewtool" / "active"


def _save_active(job: dict, work: Path, kind: str = "review", **extra) -> None:
    ACTIVE_DIR.mkdir(parents=True, exist_ok=True)
    payload = {"kind": kind, "job": job, "workdir": str(work), **extra}
    (ACTIVE_DIR / f"{Path(work).name}.json").write_text(
        json.dumps(payload, indent=2))


def _clear_active(work) -> None:
    p = ACTIVE_DIR / f"{Path(work).name}.json"
    if p.exists():
        p.unlink()


# ── pure core (tested) ───────────────────────────────────────────────────────

def build_submission(pseudo, edited, region: str,
                     source_label_sha256: str) -> Tuple[str, dict]:
    """Decide accept vs corrected from the pseudo→edited diff and build the
    review record. Pure (arrays in, dict out)."""
    d = diff.label_diff(pseudo, edited)
    decision = "accept" if d["n_voxels_changed"] == 0 else "corrected"
    record = {
        "decision": decision,
        "region_reviewed": region,
        "source_label_sha256": source_label_sha256,
        "diff": d,
    }
    return decision, record


def _resume_action(record: dict, accepted: bool) -> str:
    """Pure: decide what `resume` should do with a saved REVIEW claim.

    A claim is persisted at claim time — BEFORE any edit — purely for
    durability, so the fact that a saved claim exists does NOT mean work was
    done. Only replay it if the seg genuinely differs from the pseudo (a real
    correction), or it was explicitly recorded as a deliberate accept after a
    clean editor session. Otherwise it's a never-opened / never-saved case and
    must NOT be auto-submitted as an 'accept'."""
    if record["diff"]["n_voxels_changed"] > 0:
        return "submit"
    return "submit" if accepted else "skip"


# ── config / http ────────────────────────────────────────────────────────────

def _cfg() -> dict:
    if not CONFIG.exists():
        sys.exit("not logged in — run: reviewtool login --service URL --key KEY")
    return json.loads(CONFIG.read_text())


def _api():
    import requests          # local import so `login` works before install note
    cfg = _cfg()
    token = cfg.get("api_key")              # legacy minted key, if any
    if not token:
        from huggingface_hub import get_token
        token = get_token()                 # from `hf auth login`
        if not token:
            sys.exit("not authenticated — run `hf auth login` first, then "
                     "`reviewtool login --service <url>`.")
    s = requests.Session()
    s.headers["Authorization"] = f"Bearer {token}"
    return s, cfg["service_url"].rstrip("/")


def _fetch(job: dict, filename: str, dest: Path) -> Path:
    """Download a dataset file via huggingface_hub so it works for public,
    gated, OR private v2 repos — using the reviewer's `huggingface-cli
    login` token (or HF_TOKEN env). No dataset write access needed."""
    import shutil
    from huggingface_hub import hf_hub_download
    cached = hf_hub_download(
        repo_id=job["v2_repo"], repo_type="dataset",
        filename=filename, revision=job.get("source_revision"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, dest)
    return dest


def _fetch_adj_label(s, base, case_id: str, slot: str, dest: Path) -> Path:
    """Download a reviewer's SUBMITTED label (slot '1'/'2') THROUGH the Space (adjudicator only).
    The reviewer labels live in the PRIVATE review repo, which a personal HF login can't read
    directly — the Space holds the token and streams them."""
    r = s.get(base + "/adjudication/base",
              params={"case": case_id, "slot": slot}, timeout=120)
    r.raise_for_status()
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(r.content)
    return dest


def _auto_merge_case(s, base, job, work, ct_path, seg_path):
    """3-way auto-merge of the two reviewers' labels against the shared pseudo (see
    scripts/auto_adjudicate.py). Writes the merged label into `seg_path` and, if any voxels are
    irreconcilable, the conflict mask into work/conflict.nii.gz. Returns a summary dict, or None if
    it couldn't run (caller then adjudicates manually from reviewer 1's label)."""
    try:
        import numpy as np
        import nibabel as nib
        import auto_adjudicate as AA
        r1 = _fetch_adj_label(s, base, job["case_id"], "1", work / "r1_label.nii.gz")
        r2 = _fetch_adj_label(s, base, job["case_id"], "2", work / "r2_label.nii.gz")
        pseudo = _fetch(job, job["label_file"], work / "pseudo.nii.gz")
        pimg = nib.load(str(pseudo)); P = np.asanyarray(pimg.dataobj); aff = pimg.affine
        A = np.asanyarray(nib.load(str(r1)).dataobj)
        B = np.asanyarray(nib.load(str(r2)).dataobj)
        ct = np.asanyarray(nib.load(str(ct_path)).dataobj)
        region = job.get("region_to_review") or "ribs"
        chk = region if region in ("ribs", "spine", "both") else "ribs"
        res = AA.auto_adjudicate(P, A, B, ct, aff, check=chk)
        nib.save(nib.Nifti1Image(res["final"].astype(P.dtype), aff, pimg.header), str(seg_path))
        conflict_path = None
        if res["conflict_mask"].any():
            conflict_path = work / "conflict.nii.gz"
            nib.save(nib.Nifti1Image(res["conflict_mask"].astype(np.uint8), aff, pimg.header),
                     str(conflict_path))
        st = res["stats"]; raw = max(1, st["conflict_l0"])
        return {"decision": res["decision"], "residual": st["residual_conflict"],
                "resolved_pct": 100.0 * (1 - st["residual_conflict"] / raw),
                "qc_ok": res["qc_ok"], "conflict_path": conflict_path,
                "summary": (f"agree+one-sided auto; HU-resolved "
                            f"{st['hu_kept_label'] + st['hu_to_background']} voxels; "
                            f"{st['class_conflict']} class-conflicts; "
                            f"{st['residual_conflict']} left for you")}
    except Exception as exc:                             # noqa: BLE001
        print(f"  (auto-merge unavailable [{str(exc)[:90]}] — adjudicating from reviewer 1's label)")
        return None


def _open_reviewer_window(s, base, job, ct, work, a, slot):
    """Open ONE reviewer's FULL submitted label read-only beside the editor (v4 palette), so the
    adjudicator can compare it against the base they're editing and pick the better one. Returns
    [Popen] or []."""
    try:
        lp = _fetch_adj_label(s, base, job["case_id"], slot, Path(work) / f"r{slot}_full.nii.gz")
        desc = Path(work) / "verse_labels.txt"
        if not desc.exists():
            desc.write_text(labels_descriptor.verse_native_descriptor_text())
        print(f"  READ-ONLY reviewer {slot} window opened (their full label) to compare.")
        return [_launch_itksnap_bg(a.itksnap or _default_itksnap(), Path(ct), lp, desc)]
    except Exception as exc:                             # noqa: BLE001
        print(f"  (reviewer {slot} compare window unavailable: {str(exc)[:70]})")
        return []


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path):
    import numpy as np
    import nibabel as nib
    return np.asarray(nib.load(str(path)).dataobj)


def _default_itksnap() -> str:
    """Best-effort ITK-SNAP executable so reviewers rarely need --itksnap.

    Order: $REVIEWTOOL_ITKSNAP → anything named itksnap on PATH → the standard
    per-OS install location. EVERY candidate (the env override included) is
    verified to resolve to a runnable executable; a stale/broken one is skipped
    so detection falls through to the next option instead of handing subprocess
    a path that doesn't exist. Falls back to the bare name 'itksnap' (whose
    later FileNotFoundError prints the --itksnap hint)."""
    import os
    import shutil

    def _runnable(cand) -> Optional[str]:
        """Resolve cand (a bare name or an explicit path) to a runnable
        executable, or None. shutil.which handles PATH lookup AND verifies an
        explicit path is an executable file; the is_file/X_OK backstop covers
        an extensionless explicit path on Windows that which would miss."""
        if not cand:
            return None
        hit = shutil.which(str(cand))
        if hit:
            return hit
        p = Path(cand)
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
        return None

    env = os.environ.get("REVIEWTOOL_ITKSNAP")
    hit = _runnable(env)
    if hit:
        return hit
    if env:
        print(f"warning: REVIEWTOOL_ITKSNAP={env!r} is not a runnable "
              "executable; ignoring it and auto-detecting ITK-SNAP.",
              file=sys.stderr)

    if sys.platform == "darwin":
        # The .app GUI binary BLOCKS until its window closes; the bin/ wrapper
        # (and `open -a`) DETACH and return immediately, which makes the watch
        # loop think the reviewer quit the instant ITK-SNAP launches -> it
        # auto-submits with no edit. Always prefer the blocking binary on Mac.
        hit = _runnable("/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP")
        if hit:
            return hit
    for name in ("itksnap", "ITK-SNAP"):
        hit = _runnable(name)
        if hit:
            return hit
    candidates: list = []
    if sys.platform == "darwin":
        candidates.append("/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP")
        candidates.append("/Applications/ITK-SNAP.app/Contents/bin/itksnap")
    elif sys.platform.startswith("win"):
        for base in (r"C:\Program Files", r"C:\Program Files (x86)"):
            candidates += sorted(Path(base).glob("ITK-SNAP*/bin/ITK-SNAP.exe"),
                                 reverse=True)               # newest version first
    else:  # linux
        candidates += ["/usr/bin/itksnap", "/snap/bin/itksnap",
                       "/usr/local/bin/itksnap"]
    for c in candidates:
        hit = _runnable(c)
        if hit:
            return hit
    return "itksnap"


def _qc_hold_alert(case: str, seg, check: str = "ribs") -> None:
    """LOUD alert when ITK-SNAP closed cleanly but the label still FAILS the quick QC, so we
    refuse to upload it. Says what to do: reopen & finish, re-run the QC, or --force override."""
    bar = "!" * 74
    seg = str(seg)
    print(f"\n{bar}")
    print(f"  NOT UPLOADED - this label still FAILS the quick QC (the X items above).")
    print(f"  Your claim is KEPT; your saved edit is at:\n    {seg}")
    print(f"{bar}")
    print(f"  Before it can be submitted:")
    print(f"    - REOPEN and finish (a clean File > Quit re-checks and submits ONLY if it PASSES):")
    print(f"        python -m reviewtool edit {case}")
    print(f"    - RE-RUN the quick QC on the saved edit (no ITK-SNAP):")
    print(f'        python scripts/review_anatomy_qc.py "{seg}" --check {check}')
    print(f"    - OVERRIDE and submit anyway (adjudicator only - e.g. an FOV-truncated rib the")
    print(f"      QC can't know about): add  --force  to the command.")


def _reopen_held_if_any(a, kinds=None) -> bool:
    """A case of the SAME kind that's already CLAIMED but NOT yet submitted blocks handing out
    a new one: reopen it instead (with instructions). Returns True if it took over (caller
    should stop). `kinds` filters by claim kind ({'adjudicate'} for adjudicate, {'review'} for
    next) so a stale review claim never blocks adjudication and vice-versa."""
    held = []
    for f in sorted(ACTIVE_DIR.glob("*.json")) if ACTIVE_DIR.exists() else []:
        try:
            d = json.loads(f.read_text())
        except Exception:                                # noqa: BLE001 — skip a corrupt marker
            continue
        if kinds and d.get("kind") not in kinds:
            continue
        held.append(Path(d["workdir"]).name)
    if not held:
        return False
    bar = "=" * 74
    print(f"\n{bar}")
    print(f"  You have {len(held)} claimed case(s) NOT yet submitted — REOPENING instead of")
    print(f"  claiming a new one (finish/submit the current one first):")
    for n in held:
        print(f"    - {n}")
    print(f"  (submit it, or drop it, before a new case is handed out.)")
    print(bar)
    a.case = held[0] if len(held) == 1 else None         # cmd_edit reopens it (lists if >1)
    cmd_edit(a)
    return True


def _abnormal_close_alert(rc: int, case: str, seg, check: str = "ribs") -> None:
    """LOUD, actionable alert when ITK-SNAP closed abnormally (non-zero exit) so the case
    was NOT submitted. Makes clear the last edit didn't land and how to recover: reopen,
    re-submit (a clean quit submits the saved edit), or just re-run the cheap QC."""
    bar = "!" * 74
    seg = str(seg)
    print(f"\n{bar}")
    print(f"  ITK-SNAP CLOSED ABNORMALLY (exit {rc}) - the last case did NOT go through.")
    print(f"  Nothing was submitted; your claim is KEPT. Your last SAVED edit is safe at:")
    print(f"    {seg}")
    print(f"{bar}")
    print(f"  Do ONE of these:")
    print(f"    - REOPEN to keep editing (a CLEAN File > Quit then submits it):")
    print(f"        python -m reviewtool edit {case}")
    print(f"    - RE-SUBMIT as-is: run the reopen above and immediately File > Quit.")
    print(f"    - RE-RUN the cheap QC on the saved edit WITHOUT opening ITK-SNAP:")
    print(f'        python scripts/review_anatomy_qc.py "{seg}" --check {check}')
    print(f"  TIP: quit ITK-SNAP via File > Quit, not the window X / task-kill - a forced")
    print(f"       close returns a crash code and blocks the submit.")
    _itksnap_failure_hint(rc)


def _itksnap_failure_hint(rc: int) -> None:
    """Platform-specific troubleshooting when ITK-SNAP won't start."""
    if sys.platform.startswith("linux"):
        print(f"  ITK-SNAP exited {rc}. On Ubuntu this is almost always the Qt "
              "xcb library. Fix ONE of:\n"
              "    sudo apt-get install -y libxcb-cursor0                 # with sudo\n"
              "    conda install -y -c conda-forge xcb-util-cursor && \\\n"
              "      cp $CONDA_PREFIX/lib/libxcb-cursor.so.0* ~/itksnap/lib/   # no sudo\n"
              "  Also: `unset REVIEWTOOL_ITKSNAP` if it points at a stale path.")
    elif sys.platform == "darwin":
        print(f"  ITK-SNAP exited {rc}. On macOS, if it's Gatekeeper-quarantined:\n"
              "    xattr -dr com.apple.quarantine /Applications/ITK-SNAP.app\n"
              "  or open ITK-SNAP once from Finder to approve it.")
    elif sys.platform.startswith("win"):
        print(f"  ITK-SNAP exited {rc}. Ensure it's installed, then pass e.g.\n"
              '    --itksnap "C:\\Program Files\\ITK-SNAP 4.2\\bin\\ITK-SNAP.exe"\n'
              "  or set the REVIEWTOOL_ITKSNAP env var to the .exe path.")


def _itksnap_env():
    """ITK-SNAP launch env that silences the harmless macOS Qt trackpad/gesture
    spam (qt.pointer.dispatch / QGestureManager 'no target window'). Cosmetic —
    ITK-SNAP works regardless; this just keeps the terminal readable."""
    import os
    e = dict(os.environ)
    rules = e.get("QT_LOGGING_RULES", "")
    e["QT_LOGGING_RULES"] = (rules + ";qt.pointer.*=false;qt.gui.*=false").lstrip(";")
    return e


def _launch_itksnap_bg(itksnap: str, ct: Path, seg: Path, labels: Path):
    """Open ITK-SNAP NON-blocking (for a persistent reference window). Returns a
    Popen handle (or None if it couldn't launch)."""
    try:
        return subprocess.Popen([itksnap, "-g", str(ct), "-s", str(seg),
                                 "-l", str(labels)], env=_itksnap_env())
    except (FileNotFoundError, OSError):
        return None


def _launch_itksnap(itksnap: str, ct: Path, seg: Path, labels: Path) -> int:
    """Open ITK-SNAP on the case and return its exit code. A non-zero code
    means it failed to start or crashed (e.g. missing Qt platform lib) — the
    caller must treat that as 'no edit captured' and NOT submit."""
    print(f"\nOpening ITK-SNAP ({itksnap}) — edit the segmentation, Save "
          f"Segmentation to:\n  {seg}\nthen quit ITK-SNAP to continue.\n")
    try:
        proc = subprocess.run([itksnap, "-g", str(ct), "-s", str(seg),
                               "-l", str(labels)], check=False, env=_itksnap_env())
    except FileNotFoundError:
        sys.exit(f"'{itksnap}' not found — install ITK-SNAP, add it to PATH, "
                 f"set REVIEWTOOL_ITKSNAP, or pass --itksnap /path/to/itksnap")
    return proc.returncode


# ── commands ─────────────────────────────────────────────────────────────────

def cmd_login(a):
    CONFIG.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"service_url": a.service}
    if a.key:                               # optional legacy minted key
        cfg["api_key"] = a.key
    CONFIG.write_text(json.dumps(cfg, indent=2))
    print(f"saved {CONFIG}")
    from huggingface_hub import get_token
    if not a.key and not get_token():
        print("note: you're not logged in to HuggingFace yet — "
              "run `hf auth login` before `reviewtool next`.")


def _claim(s, base, path="/claim", method="post", params=None):
    # /claim is POST; /adjudication/next is GET — use the matching verb.
    # `params` lets the adjudicator pick a SPECIFIC case (work a ranked list, not ledger order).
    r = ((s.get(base + path, params=params, timeout=60)) if method == "get"
         else s.post(base + path, timeout=60))
    if r.status_code == 204:
        return None
    r.raise_for_status()
    return r.json()


def _post_resilient(s, url, *, data, label_path, timeout=300,
                    max_tries=4, backoff=3.0):
    """POST a submit/adjudicate with the edited label, retrying transient
    failures (socket errors, 5xx, a brief 429). The label is read into memory
    so each attempt can rebuild a fresh multipart body. Returns the final
    requests.Response; the caller inspects the status."""
    import requests
    label_bytes = Path(label_path).read_bytes()
    r = None
    for attempt in range(1, max_tries + 1):
        files = {"label": ("label.nii.gz", label_bytes, "application/gzip")}
        try:
            r = s.post(url, data=data, files=files, timeout=timeout)
        except requests.RequestException:
            if attempt == max_tries:
                raise
            time.sleep(backoff * attempt)
            continue
        if (r.status_code == 429 or 500 <= r.status_code < 600) \
                and attempt < max_tries:
            time.sleep(backoff * attempt)
            continue
        return r
    return r


def _finish_submit(r, work) -> bool:
    """Interpret a submit/adjudicate response. The active claim is cleared
    ONLY on success — on any failure it is kept so `resume` can retry without
    the reviewer redoing the segmentation."""
    if r.status_code == 429:
        print("\nRATE-LIMITED by HuggingFace (commit cap reached this hour).\n"
              "Your edit is SAVED locally — nothing is lost. Re-run later:\n"
              "  python -m reviewtool resume\n")
        return False
    if not r.ok:
        print(f"submit failed [{r.status_code}]: {r.text[:300]}\n"
              "Your edit is saved locally; `reviewtool resume` will retry.")
        return False
    out = r.json()
    _clear_active(work)
    print("submitted ->", out,
          "(already recorded)" if out.get("duplicate") else "")
    return True


def _submit_review(s, base, job, work, seg, record) -> bool:
    data = {"claim_token": job["claim_token"], "record": json.dumps(record)}
    return _finish_submit(
        _post_resilient(s, base + "/submit", data=data, label_path=seg), work)


def _submit_adjudication(s, base, job, work, seg, notes) -> bool:
    data = {"claim_token": job["claim_token"], "decision": "corrected",
            "notes": notes}
    return _finish_submit(
        _post_resilient(s, base + "/adjudicate", data=data, label_path=seg),
        work)


def _qc_feedback(ct_path, before_path, after_path) -> None:
    """Print a draft->edit QC comparison so the reviewer sees whether their fix
    cleared the flag. Computes the same metrics the triage used. Degrades
    SILENTLY if scipy / the QC modules aren't installed (it's just feedback)."""
    try:
        import numpy as np
        import nibabel as nib
        from vertebra_topology_qc import vertebra_topology_metrics
        from structure_qc import structure_metrics
    except Exception:
        return
    try:
        bimg = nib.load(str(before_path))
        aff = bimg.affine
        before = np.asarray(bimg.dataobj)
        after = np.asarray(nib.load(str(after_path)).dataobj)
    except Exception:
        return

    def _metrics(lab):
        out = {}
        # off-bone leak dropped from triage -> not computed (also the slow part)
        for fn in (lambda l: vertebra_topology_metrics(l, aff),
                   lambda l: structure_metrics(l, aff)):
            try:
                out.update(fn(lab))
            except Exception:
                pass
        return out

    mb, ma = _metrics(before), _metrics(after)
    if not ma:
        return
    print("  QC check (draft -> your edit):")
    elevated = False
    for key, name, thr in (("off_main_frac", "vertebra mixing", 0.005),):
        if key not in ma:
            continue
        vb, va = mb.get(key), ma.get(key)
        ok = va is not None and va <= thr
        elevated = elevated or not ok
        print(f"    {name:16s} {('%.4f' % vb) if vb is not None else '-':>8} -> "
              f"{('%.4f' % va) if va is not None else '-':>8}   "
              f"{'OK' if ok else 'STILL HIGH'} (target <= {thr})")
    for key, name in (("n_order_inversions", "level order"),
                      ("duplication_flag", "duplicated piece"),
                      ("lr_swap", "L/R hip swap"),
                      ("vertebra_gap", "missing level")):
        vb, va = mb.get(key, 0), ma.get(key, 0)
        if not vb and not va:
            continue
        ok = not va
        elevated = elevated or not ok
        print(f"    {name:16s} {vb} -> {va}   {'OK' if ok else 'STILL FLAGGED'}")
    print("  ! some checks still elevated - recheck the flagged region "
          "(or it may be a true case the metric can't fully clear)."
          if elevated else "  OK - all automated checks pass on your edit.")


def _watch_itksnap(itksnap, ct, seg, labels, *, qc_ct, before, live_qc=False) -> int:
    """Open ITK-SNAP NON-blocking and wait for the reviewer to quit. With
    live_qc=True, re-runs the QC on every Save so they see progress without
    closing; by default it just notes the save (no recompute — faster, which
    matters with AI-assisted fixes). Returns ITK-SNAP's exit code."""
    extra = ("  Edit, then **Save Segmentation** (Ctrl-S) any time to see a live\n"
             "  QC update here. " if live_qc else "  Edit, **Save Segmentation** "
             "(Ctrl-S), then ")
    print(f"\nOpening ITK-SNAP ({itksnap}) — WATCH mode.\n"
          f"{extra}Quit ITK-SNAP when you're done — it submits then.\n")
    proc = _launch_itksnap_bg(itksnap, ct, seg, labels)
    if proc is None:
        print(f"'{itksnap}' not found — install ITK-SNAP, add it to PATH, set "
              f"REVIEWTOOL_ITKSNAP, or pass --itksnap /path/to/itksnap")
        return -1
    try:
        last = Path(seg).stat().st_mtime if Path(seg).exists() else 0.0
    except OSError:
        last = 0.0
    while proc.poll() is None:
        time.sleep(1.5)
        try:
            m = Path(seg).stat().st_mtime if Path(seg).exists() else last
        except OSError:
            continue
        if m > last + 1e-6:                     # a Save happened
            last = m
            if live_qc:
                print("  [saved] recomputing QC ...")
                _qc_feedback(qc_ct, before, seg)
            else:
                print("  [saved]")
    return proc.returncode if proc.returncode is not None else 0


def _watch_anatomy(itksnap, ct, seg, labels, *, check="spine"):
    """Open ITK-SNAP non-blocking; run the anatomy QC in a BACKGROUND thread on each Save
    (the editor never waits on it) and print PASS / exactly what to fix when it finishes.
    Used by `review-cases` and `next` so students get a guiding, non-blocking gate.

    Returns ``(itksnap_rc, passed)`` where ``passed`` is True / False / None
    (None = the QC could not run, e.g. scipy/nibabel missing). The caller GATES
    a case as 'resolved' (saved + uploaded) on ``passed is True`` — a still-failing
    or un-checkable edit is never accepted."""
    import os as _os
    import sys as _sys
    import threading
    for _p in (_os.path.join(_os.path.dirname(__file__), "..", "scripts"), "scripts"):
        if _os.path.isdir(_p) and _p not in _sys.path:
            _sys.path.insert(0, _p)
    state = {"passed": None, "running": False}          # verdict + a single-flight guard
    lock = threading.Lock()

    def _qc(header=None):
        try:
            import numpy as np
            import nibabel as nib
            import review_anatomy_qc as RA
            img = nib.load(str(seg))
            if header:
                print(header, flush=True)
            state["passed"] = bool(RA.report(check, np.asanyarray(img.dataobj), img.affine))
        except Exception as exc:                        # noqa: BLE001
            print(f"  (anatomy QC could not run: {str(exc)[:120]})")
            state["passed"] = None                      # cannot verify -> caller fails closed
        finally:
            with lock:
                state["running"] = False

    def _qc_bg(header=None):
        with lock:                                      # one QC at a time; the poll re-triggers
            if state["running"]:
                return False
            state["running"] = True
        threading.Thread(target=_qc, args=(header,), daemon=True).start()
        return True

    if check in ("ribs", "both"):
        print("  How to fix a flagged rib (a number in two pieces): go to that rib number,\n"
              "    * ONE broken rib      -> WELD: paint across the gap with that rib's label\n"
              "    * TWO different ribs  -> RELABEL the wrong piece to its correct number\n"
              "    * NOT a rib (TP / bowel) -> DELETE it (set to 0)\n"
              "  Use nnInteractive, don't hand-trace. Full guide: docs/RIB_REVIEW_GUIDE.md")
    print(f"Opening ITK-SNAP ({itksnap}) — edit, **Save (Ctrl-S)** to re-check (runs in the\n"
          f"  background — keep working), quit once it prints OK.\n")
    proc = _launch_itksnap_bg(itksnap, ct, seg, labels)
    if proc is None:
        print(f"'{itksnap}' not found — install ITK-SNAP, add it to PATH, set "
              f"REVIEWTOOL_ITKSNAP, or pass --itksnap /path/to/itksnap")
        return -1, None
    _qc_bg(header="\n--- current QC for this case (running in the background) ---")  # don't delay launch
    try:
        last = Path(seg).stat().st_mtime if Path(seg).exists() else 0.0
    except OSError:
        last = 0.0

    while proc.poll() is None:
        time.sleep(1.5)
        try:
            m = Path(seg).stat().st_mtime if Path(seg).exists() else last
        except OSError:
            continue
        if m > last + 1e-6:                             # a Save happened
            # kick the QC in the background; if one is still running, leave `last` so the next
            # poll re-checks this Save once it frees up (no missed feedback, no blocking).
            if _qc_bg(header="  [saved] re-running QC in the background ..."):
                last = m
    # ITK-SNAP quit: let any in-flight QC finish (~10s cap), then a final authoritative check.
    for _ in range(40):
        with lock:
            if not state["running"]:
                break
        time.sleep(0.25)
    _qc()                                               # authoritative final verdict on the saved file
    rc = proc.returncode if proc.returncode is not None else 0
    return rc, state["passed"]


def _qc_startup(ct_path, draft_path) -> None:
    """Print the DRAFT's current QC so the reviewer knows what to focus on — this
    is the 'WHY FLAGGED' for the live flow (the Space doesn't carry the reason).
    Degrades silently without scipy / the QC modules."""
    try:
        import numpy as np
        import nibabel as nib
        from vertebra_topology_qc import vertebra_topology_metrics
        from structure_qc import structure_metrics
    except Exception:
        return
    try:
        dimg = nib.load(str(draft_path))
        aff = dimg.affine
        draft = np.asarray(dimg.dataobj)
    except Exception:
        return
    m = {}
    # off-bone leak is intentionally NOT computed: leak was dropped from triage
    # (too hard to fix by hand) and the fill-holes over a whole-scan CT is what
    # made fused cases take minutes. Topology + structure only (fast).
    for fn in (lambda l: vertebra_topology_metrics(l, aff),
               lambda l: structure_metrics(l, aff)):
        try:
            m.update(fn(draft))
        except Exception:
            pass
    if not m:
        return
    foci = []
    if m.get("off_main_frac", 0) > 0.005:
        sc = (m.get("split_classes") or "")
        foci.append(f"vertebra MIXING  (off_main={m['off_main_frac']:.3f}; target <= 0.005)"
                    + (f"  [split: {sc.replace(';', ', ')}]" if sc else ""))
    if m.get("n_order_inversions", 0):
        foci.append("level ORDER wrong (a vertebra is out of sequence)")
    if m.get("duplication_flag", 0):
        dc = (m.get("dup_classes") or "")
        foci.append("DUPLICATED structure" + (f"  [{dc}]" if dc else
                                              " (a stray disconnected piece)"))
    if m.get("lr_swap", 0):
        foci.append("L/R HIP SWAP")
    if m.get("vertebra_gap", 0):
        foci.append("MISSING level (a gap in the sequence)")
    print("  WHY FLAGGED - focus your edit here:")
    if foci:
        for f in foci:
            print(f"    * {f}")
    else:
        print("    * nothing obvious in the automated metrics - may be a "
              "borderline flag; give the structure a quick look.")
    print("  (edit, Save (Ctrl-S), and watch these clear to OK above.)")


def _open_space_reference(job, snap, work):
    """Download the gold reference example (crops/reference/ in the v2 repo, if
    present) and open it in a second ITK-SNAP window for comparison. None if no
    reference is published."""
    try:
        rct = _fetch(job, "crops/reference/ct.nii.gz", work / "ref_ct.nii.gz")
        rseg = _fetch(job, "crops/reference/seg.nii.gz", work / "ref_seg.nii.gz")
    except Exception:
        return None
    rlbl = work / "ref_labels.txt"
    rlbl.write_text(labels_descriptor.descriptor_text())
    print("  opened a GOLD reference example in a second window "
          "(tile the two windows to compare).")
    return _launch_itksnap_bg(snap, rct, rseg, rlbl)




_LSTV_NAMES = {0: "Normal (L1-L5 + sacrum)", 1: "Lumbarization (L6 present)",
               2: "Semi-sacralization (borderline)", 3: "Sacralization (L5 fused to sacrum)"}
_LSTV_HINT = {
    1: "an L6 EXISTS. Count up from the sacrum and make sure L1-L6 are all "
       "distinct — the draft often DUPLICATES or shifts the extra level.",
    2: "the L5/S1 boundary is genuinely ambiguous. Label what the bone shows; "
       "don't force a clean split. Unsure -> smallest safe edit, tell the lead.",
    3: "L5 is fused to the sacrum. Check the L5<->sacrum boundary — the draft "
       "tends to bleed between them.",
}


_LSTV_MANIFEST_CACHE: dict = {}


def _lstv_lookup(job: dict, work):
    """LSTV (class, label) for this case from the v2 manifest — so it shows even
    when the seeded crop predates the lstv field. Cached per v2_repo per process
    (manifest.json is ~1 MB and HF-cached, so this is cheap)."""
    repo = job.get("v2_repo")
    if repo not in _LSTV_MANIFEST_CACHE:
        table: dict = {}
        try:
            p = _fetch(job, "manifest.json", Path(work) / "v2_manifest.json")
            recs = json.loads(Path(p).read_text())
            if isinstance(recs, dict):
                recs = recs.get("records") or recs.get("cases") or []
            for r in recs:
                t = str(r.get("token"))
                if t and t not in table:
                    table[t] = (int(r.get("lstv_class") or 0),
                                r.get("lstv_label") or "")
        except Exception:
            pass
        _LSTV_MANIFEST_CACHE[repo] = table
    tok = str(job.get("token") or str(job.get("case_id", "")).split("__")[0])
    return _LSTV_MANIFEST_CACHE[repo].get(tok)


def _print_lstv(job: dict, crop: dict, work) -> None:
    """Print the case's LSTV phenotype so the reviewer counts levels carefully.
    Prefers the seeded crop value; falls back to the v2 manifest so it ALWAYS
    prints (including 'Normal'). Silent only if the manifest can't be read."""
    c = label = None
    if crop and "lstv_class" in crop:
        c, label = int(crop.get("lstv_class") or 0), crop.get("lstv_label")
    else:
        hit = _lstv_lookup(job, work)
        if hit:
            c, label = hit
    if c is None:
        return
    print(f"  LSTV STATUS: {_LSTV_NAMES.get(c) or label or '?'}")
    if c in _LSTV_HINT:
        print(f"    -> {_LSTV_HINT[c]}")


_QC_CSV_CACHE: dict = {}


def _qc_table(job: dict, work):
    """(token,config) -> precomputed qc_master row, from crops/qc_master.csv on
    the v2 repo (cached per repo). Returns None if it isn't published — the
    signal to fall back to a local QC scan."""
    repo = job.get("v2_repo")
    if repo not in _QC_CSV_CACHE:
        table = None
        try:
            import csv as _csv
            p = _fetch(job, "crops/qc_master.csv", Path(work) / "qc_master.csv")
            table = {(str(r.get("token")), str(r.get("config"))): r
                     for r in _csv.DictReader(open(p))}
        except Exception:
            table = None
        _QC_CSV_CACHE[repo] = table
    return _QC_CSV_CACHE[repo]


def _precomputed_focus(job: dict, work):
    """WHY-FLAGGED lines straight from the published qc_master.csv (no local
    compute). None => not published (fall back to a local scan). Leak is omitted
    (dropped from triage)."""
    table = _qc_table(job, work)
    if table is None:
        return None
    cid = str(job.get("case_id", ""))
    tok = str(job.get("token") or (cid.split("__", 1)[0] if "__" in cid else cid))
    cfg = str(job.get("config") or (cid.split("__", 1)[1] if "__" in cid else ""))
    row = table.get((tok, cfg)) or next(
        (v for (t, _c), v in table.items() if t == tok), None)
    if row is None:
        return []
    foci = []
    if _is_flag(row.get("mixing_flag")):
        sc = (row.get("split_classes") or "").strip()
        detail = f"  [split: {sc.replace(';', ', ')}]" if sc else ""
        foci.append(f"vertebra MIXING  (off_main={row.get('off_main_frac', '?')}; "
                    f"target <= 0.005){detail}")
    if str(row.get("n_order_inversions", "0")).strip() not in ("", "0", "0.0"):
        foci.append("level ORDER wrong (a vertebra is out of sequence)")
    if _is_flag(row.get("struct_flag")):
        if _is_flag(row.get("duplication_flag")):
            dc = (row.get("dup_classes") or "").strip()
            foci.append("DUPLICATED structure"
                        + (f"  [{dc}]" if dc else " (a stray disconnected piece)"))
        if _is_flag(row.get("lr_swap")):
            foci.append("L/R HIP SWAP")
        if str(row.get("vertebra_gap", "0")).strip() not in ("", "0", "0.0"):
            foci.append("MISSING level (a gap in the sequence)")
        if _is_flag(row.get("pelvis_incomplete")):
            foci.append("pelvis INCOMPLETE")
    return foci


def _class_split_lines(label_path):
    """Cheap LOCAL per-class split/duplication detail from one label map, e.g.
    ['L3 (63/37)', 'L4 (70/31)'] — the two biggest pieces of any class that's
    split >=10%. Used to show which levels to fix when the precomputed CSV
    predates the split_classes column. None on any error."""
    try:
        import numpy as np
        import nibabel as nib
        from scipy.ndimage import label as _cc, generate_binary_structure
        lab = np.asarray(nib.load(str(label_path)).dataobj)
    except Exception:
        return None
    names = {1: "L1", 2: "L2", 3: "L3", 4: "L4", 5: "L5", 6: "L6",
             7: "sacrum", 8: "left hip", 9: "right hip"}
    st = generate_binary_structure(lab.ndim, lab.ndim)
    out = []
    for c in range(1, 10):
        m = lab == c
        n = int(m.sum())
        if n == 0:
            continue
        arr, k = _cc(m, structure=st)
        if k < 2:
            continue
        sizes = np.sort(np.bincount(arr.ravel())[1:])[::-1]
        if sizes[1] >= 0.10 * n:                       # real split, not a speck
            out.append("%s (%d/%d)" % (names[c], round(100 * sizes[0] / n),
                                       round(100 * sizes[1] / n)))
    return out or None


def _show_startup(job: dict, crop: dict, work, qc_ct, before, run_qc: bool):
    """Print LSTV + WHY FLAGGED at case open. Uses the precomputed qc_master.csv
    if it's published on v2 (instant, no compute); else a local QC scan."""
    _print_lstv(job, crop, work)
    if not run_qc:
        return
    foci = _precomputed_focus(job, work)
    if foci is None:                                  # not published -> local scan
        _qc_startup(qc_ct, before)
        return
    print("  WHY FLAGGED - focus your edit here:")
    for f in (foci or ["nothing obvious in the metrics — give the structure a "
                       "quick look."]):
        print(f"    * {f}")
    # If the published CSV predates the split/dup detail, fill it in locally (cheap).
    if foci and not any("[" in f for f in foci):
        detail = _class_split_lines(before)
        if detail:
            print("    split/duplicated levels: " + ", ".join(detail))
    print("  (edit, Save (Ctrl-S), and watch these clear to OK above.)")


def _descriptor_for_job(job) -> str:
    """ITK-SNAP palette that MATCHES the served labels so names show correctly: VerSe-native
    for v3/v4 (spine 1-28, S1 29, hips/femurs 30-33, ribs 34-57) vs the v2 LSTV scheme (1-9).
    Without this a v4 label shows e.g. value 22 as 'Label 22' with no name."""
    j = job or {}
    rev = str(j.get("source_revision", "v2")).lstrip("vV")
    verse = (rev.isdigit() and int(rev) >= 3) or j.get("region_to_review") == "ribs"
    return (labels_descriptor.verse_native_descriptor_text() if verse
            else labels_descriptor.descriptor_text())


def cmd_next(a):
    s, base = _api()
    # NOTE: no reopen-held gate here -- a stale local claim (e.g. an old spine-pseudolabel task in
    # the same Space) must NEVER block a student from claiming a new rib case. Use `resume`/`edit`
    # to finish an unsubmitted case; it does not stop `next`.
    if getattr(a, "amend", False):                       # fix YOUR re-opened (QC-failed) cases
        job = _claim(s, base, "/amend/next", method="get")
        if job is None:
            print("nothing to amend — no cases re-opened for you (or you've fixed them all).")
            return
        print("AMEND: this is YOUR earlier submission, re-opened because it failed the "
              "strengthened QC. Your own label is loaded — fix the flagged items and re-save.")
    else:
        job = _claim(s, base)
        if job is None:
            print("nothing to claim — all cases assigned/done.")
            return
    work = Path(a.workdir) / job["case_id"]
    work.mkdir(parents=True, exist_ok=True)
    _save_active(job, work, kind="review")              # durable before any edit
    crop = job.get("crop")
    # The full pseudo LABEL is small (~MB) and is what we submit; with a crop we
    # avoid the ~200 MB CT and only pull the few-MB crop CT + crop mask.
    if getattr(a, "amend", False):
        # The amend base is YOUR earlier submission — it lives in the PRIVATE review repo, which
        # you can't read directly, so the Space streams it to you. Fall back to the public auto-
        # label (re-do the fix from scratch) if your old submission can't be served.
        pseudo = work / "pseudo.nii.gz"
        try:
            r = s.get(base + "/amend/base",
                      params={"case": job["case_id"], "slot": job["slot"]}, timeout=120)
            r.raise_for_status()
            pseudo.write_bytes(r.content)
        except Exception as exc:                          # noqa: BLE001
            print(f"  (could not load your earlier label [{str(exc)[:80]}] — "
                  "starting from the auto-label instead)")
            pseudo = _fetch(job, job.get("orig_pseudo_file") or job["label_file"],
                            work / "pseudo.nii.gz")
    else:
        pseudo = _fetch(job, job["label_file"], work / "pseudo.nii.gz")
    seg = work / "seg.nii.gz"
    labels = work / "labels.txt"
    # palette must MATCH the served labels (VerSe-native for v3/v4, v2 LSTV otherwise),
    # else ITK-SNAP shows e.g. value 22 as "Label 22" with no name.
    labels.write_text(_descriptor_for_job(job))
    snap = a.itksnap or _default_itksnap()
    ctx_proc = None
    if crop:
        snap_ct = _fetch(job, crop["ct_crop"], work / "ct.nii.gz")    # crop CT (few MB)
        snap_seg = work / "crop_edit.nii.gz"
        snap_seg.write_bytes(_fetch(job, crop["seg_crop"],
                                    work / "crop_seg.nii.gz").read_bytes())
        note = "crop review"
        if getattr(a, "full", False):
            # Pull the FULL scan from the original repo and open it read-only
            # beside the crop, to verify the rest of the volume looks fine.
            full_ct = _fetch(job, job["ct_file"], work / "full_ct.nii.gz")
            print("  also opening the FULL scan (read-only) for context ...")
            ctx_proc = _launch_itksnap_bg(snap, full_ct, pseudo, labels)
    else:
        snap_ct = _fetch(job, job["ct_file"], work / "ct.nii.gz")     # full CT
        snap_seg = seg
        seg.write_bytes(pseudo.read_bytes())                          # edit a copy
        note = "review"

    region = job.get("region_to_review")
    if region == "both":
        region_note = "the WHOLE scan (radiologist gold being re-checked)"
    elif region == "rib_anchor":
        region_note = "ADD the counting anchor"
    else:
        region_note = f"the {region} region"
    print(f"case {job['case_id']}  ({note} — {region_note})")
    if region == "rib_anchor":
        print("  TASK: scroll up, find the lowest vertebra with a TRUE "
              "articulating rib.\n"
              "  Paint it as 11=last_rib_vertebra and its proximal rib as "
              "12=rib (AI-assisted).\n"
              "  The vertebra just below it (no rib) is L1; count down to the "
              "sacrum.\n"
              "  Also fix any class-mixing / partly-coloured vertebra you see. "
              "Save when done.\n"
              "  Full guide: docs/STUDENT_ANNOTATION_PROTOCOL.md")
    before_qc = (work / "crop_seg.nii.gz") if crop else pseudo
    # Print LSTV + WHY FLAGGED in the background so ITK-SNAP opens immediately —
    # the manifest lookup and QC scan must not delay the window.
    import threading

    def _startup_info():
        _show_startup(job, crop, work, snap_ct, before_qc,
                      not getattr(a, "no_qc", False))
    threading.Thread(target=_startup_info, daemon=True).start()

    ref_proc = None
    if getattr(a, "reference", False):                  # opt-in: gold example window
        ref_proc = _open_space_reference(job, snap, work)

    # mtime of the seg the editor opens, BEFORE the session. A real save (an
    # edit OR a deliberate accept) advances it; if it never advances, ITK-SNAP
    # detached/closed before the reviewer saved (the macOS bin/ wrapper bug) and
    # we must NOT submit a phantom voxels_changed=0 'accept'.
    pre_mtime = snap_seg.stat().st_mtime if snap_seg.exists() else 0.0
    rib_passed = None
    if region == "ribs" and not getattr(a, "no_watch", False):
        # Live rib QC on every Save (same engine as review-cases): show exactly what's still
        # wrong + how to fix it, and gate the submit client-side so students don't quit into a
        # server rejection. The server CHECK=ribs gate is still the hard backstop on /submit.
        rc, rib_passed = _watch_anatomy(snap, snap_ct, snap_seg, labels, check="ribs")
    elif getattr(a, "no_watch", False):                 # watch is the default
        rc = _launch_itksnap(snap, snap_ct, snap_seg, labels)
    else:
        rc = _watch_itksnap(snap, snap_ct, snap_seg, labels,
                            qc_ct=snap_ct, before=before_qc,
                            live_qc=getattr(a, "live_qc", False))
    for p in (ctx_proc, ref_proc):
        if p is not None:
            p.terminate()
    if rc != 0:
        _abnormal_close_alert(rc, job["case_id"], snap_seg,
                              check="ribs" if region == "ribs" else "both")
        return
    saved = snap_seg.exists() and snap_seg.stat().st_mtime > pre_mtime + 1e-6
    if not saved:
        print("\nITK-SNAP closed but you never SAVED, so no edit was captured — "
              "nothing was submitted. Your claim is kept.\n"
              "  • Edit, then Save Segmentation (Ctrl-S / Cmd-S) at least once "
              "before quitting (one save is also how you 'accept' a correct draft).\n"
              "  • If ITK-SNAP opened then instantly closed/submitted, it "
              "DETACHED instead of staying open — relaunch it directly:\n"
              f"      python -m reviewtool edit {job['case_id']}"
              + (" --itksnap /Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP"
                 if sys.platform == "darwin" else ""))
        return
    if crop:                                            # fold the crop edit into full-res
        _paste_edit_to_full(snap_seg, crop["origin"], pseudo, seg)

    # only recompute QC at the end if the reviewer explicitly asked for live QC
    # (default is no recompute — fix, save, quit, submit; fast for AI-assisted edits)
    if getattr(a, "live_qc", False) and getattr(a, "no_watch", False):
        _qc_feedback(snap_ct, before_qc, snap_seg)

    if region == "ribs" and rib_passed is not True:
        print("\n[HELD] the ribs still fail QC (a rib number is in two pieces) — NOT "
              "submitting (the server would reject it too).\n"
              "  Re-open and finish the fix, then quit once it PASSES:\n"
              f"      python -m reviewtool edit {job['case_id']}")
        return
    decision, record = build_submission(
        _load(pseudo), _load(seg), job["region_to_review"], _sha256(pseudo))
    print(f"decision={decision}  voxels_changed={record['diff']['n_voxels_changed']}")
    if decision == "accept":
        # 0 voxels changed after a clean editor session = a DELIBERATE accept.
        # Tag the saved claim so a failed upload can be replayed by `resume`
        # without being mistaken for a never-opened case.
        _save_active(job, work, kind="review", accepted=True)
    _submit_review(s, base, job, work, seg, record)


def cmd_adjudicate(a):
    s, base = _api()
    # An explicit --case means "serve me THIS one" — a stale held claim must not hijack it.
    if not getattr(a, "case", None) and _reopen_held_if_any(a, kinds={"adjudicate"}):
        return
    _adj_case = getattr(a, "case", None)
    job = _claim(s, base, "/adjudication/next", method="get",
                 params=({"case": _adj_case} if _adj_case else None))
    if job is None:
        print(f"nothing to adjudicate{f' for {_adj_case}' if _adj_case else ''}.")
        return
    work = Path(a.workdir) / (job["case_id"] + "__adj")
    work.mkdir(parents=True, exist_ok=True)
    _save_active(job, work, kind="adjudicate", notes=a.notes)
    ct = _fetch(job, job["ct_file"], work / "ct.nii.gz")
    seg = work / "seg.nii.gz"
    labels = work / "labels.txt"
    # region-correct palette (v4 VerSe-native for ribs, v2 jet otherwise) — same idx↔structure
    # as the server, so this just guarantees the current colours without a Space redeploy.
    labels.write_text(_descriptor_for_job(job))
    region = job.get("region_to_review")
    base_slot = str(getattr(a, "base", None) or "1")
    other_slot = "2" if base_slot == "1" else "1"
    auto = None
    if getattr(a, "auto", False):
        # OPT-IN: 3-way merge the two reviewers vs the shared pseudo. If nothing irreconcilable
        # remains (and QC is clean) the case AUTO-FINALIZES with no editing; otherwise the conflicts
        # are highlighted. Off by default — a merge can erode low-HU rib heads, so picking a human
        # label is the trusted path.
        auto = _auto_merge_case(s, base, job, work, ct, seg)
        if auto is not None and auto["decision"] == "auto_finalize" \
                and not getattr(a, "review_auto", False):
            print(f"ADJUDICATE {job['case_id']}: AUTO-MERGED — no irreconcilable conflict, QC clean.\n"
                  f"  {auto['summary']}. Auto-finalizing (pass --review_auto to eyeball it first).")
            _submit_adjudication(s, base, job, work, seg, (a.notes or "").strip() + " [auto-merged]")
            return
        if auto is not None:
            print(f"  auto-merge resolved {auto['resolved_pct']:.0f}% of the disagreement; "
                  f"{auto['residual']} voxels remain as conflicts — highlighted for you to paint.")
    if auto is None:
        # DEFAULT: adjudicate by SELECTION — load the chosen reviewer's label as the editable base;
        # the OTHER reviewer opens read-only beside it, so you pick the better one and edit it in.
        try:
            _fetch_adj_label(s, base, job["case_id"], base_slot, seg)
            print(f"  base = reviewer {base_slot}'s label; reviewer {other_slot} opens read-only to "
                  f"compare. If {other_slot} is the better base, re-run with  --base {other_slot}.")
        except Exception as exc:                          # noqa: BLE001
            print(f"  (couldn't load reviewer {base_slot} [{str(exc)[:70]}] — from the auto-label)")
            pseudo = _fetch(job, job["label_file"], work / "pseudo.nii.gz")
            seg.write_bytes(pseudo.read_bytes())
    print(f"ADJUDICATE {job['case_id']}  IRR={job.get('irr')}\n"
          f"two reviewers disagreed on the {region} region; produce the deciding label.")
    if getattr(a, "accept", False):
        # The chosen annotator's label is right AS-IS -> finalize it WITHOUT opening ITK-SNAP.
        # Still QC-gated, so an accept can never wave a bad label through.
        import numpy as _np
        import nibabel as _nib
        import review_anatomy_qc as _RA
        _img = _nib.load(str(seg))
        _chk = region if region in ("ribs", "spine", "both") else "ribs"
        _ok, _msgs = _RA.check_label(_chk, _np.asanyarray(_img.dataobj), _img.affine)
        if not _ok and not getattr(a, "force", False):
            print(f"  ACCEPT REFUSED — reviewer {base_slot}'s label FAILS QC:")
            for _m in _msgs:
                if _m.startswith("X"):
                    print(f"     {_m}")
            print("  edit it instead (drop --accept), or --force to override.")
            return
        print(f"  ACCEPT: reviewer {base_slot}'s label is correct as-is (QC OK) → finalizing.")
        _submit_adjudication(s, base, job, work, seg,
                             (a.notes or "").strip() + f" [accepted reviewer {base_slot} as-is]")
        return
    snap = a.itksnap or _default_itksnap()
    pre_mtime = seg.stat().st_mtime if seg.exists() else 0.0
    # read-only reference windows beside the editor (best-effort)
    dis_procs = ([] if getattr(a, "no_disagreement", False)
                 else _open_disagreement_ref(s, base, job, ct, work, a))
    if auto is None:                                     # pick-better: show the OTHER reviewer's FULL label
        dis_procs += _open_reviewer_window(s, base, job, ct, work, a, other_slot)
    if auto and auto.get("conflict_path"):               # --auto: show the conflict blobs to paint
        dis_procs.append(_launch_itksnap_bg(snap, ct, auto["conflict_path"], labels))
    if region in ("ribs", "spine", "both"):   # live QC as you decide; gate the submit on it
        rc, passed = _watch_anatomy(snap, ct, seg, labels, check=region)
    else:
        rc, passed = _launch_itksnap(snap, ct, seg, labels), True   # no QC for this region
    for _p in dis_procs:
        if _p is not None:
            _p.terminate()
    chk = region if region in ("ribs", "spine", "both") else "ribs"
    saved = seg.exists() and seg.stat().st_mtime > pre_mtime + 1e-6
    if not saved:                            # only NOW does the exit code matter (nothing saved)
        if rc != 0:
            _abnormal_close_alert(rc, f"{job['case_id']}__adj", seg, check=chk)
        else:
            print(f"\nITK-SNAP closed but you never SAVED - nothing submitted; your claim is "
                  f"kept.\n  Re-open: python -m reviewtool edit {job['case_id']}__adj")
        return
    # You SAVED -> the submit is decided by QC, not by ITK-SNAP's (Windows-flaky) exit code.
    if passed is not True and not getattr(a, "force", False):       # QC GATE - don't upload junk
        _qc_hold_alert(f"{job['case_id']}__adj", seg, check=chk)
        return
    _submit_adjudication(s, base, job, work, seg, a.notes)


def cmd_skip(a):
    """Defer the scan(s) you have claimed — release back to the queue for someone else, and clear
    the local claim. Use it for a scan you don't want (bad scan, unsure) or to clear STALE local
    claims that error with "claim token does not match an open claim" (the server already moved on).
    `--all` clears every held claim at once. Your submitted/passed work is never touched."""
    s, base = _api()
    claims = []
    for f in (sorted(ACTIVE_DIR.glob("*.json")) if ACTIVE_DIR.exists() else []):
        try:
            d = json.loads(f.read_text())
        except Exception:                                # noqa: BLE001
            continue
        if d.get("kind") == "review":
            claims.append(d)
    if a.case:
        claims = [d for d in claims if Path(d["workdir"]).name == a.case]
    if not claims:
        print("no claimed scan to defer — you're not holding one. `reviewtool next` to claim.")
        return
    if len(claims) > 1 and not a.case and not getattr(a, "all", False):
        print("you're holding more than one — pass which to skip (`reviewtool skip <case>`), "
              "or clear them all with `reviewtool skip --all`:")
        for d in claims:
            print("   ", Path(d["workdir"]).name)
        return
    for d in claims:
        job = d["job"]; cid = job.get("case_id", Path(d["workdir"]).name)
        try:
            r = s.post(base + "/defer", data={"claim_token": job.get("claim_token")}, timeout=60)
            r.raise_for_status()
            print(f"deferred {cid} — released back to the queue for another reviewer.")
        except Exception as exc:                         # noqa: BLE001
            # stale/expired/already-reassigned: nothing to release server-side; just drop it locally
            print(f"cleared stale local claim {cid} (server had already moved on).")
        _clear_active(Path(d["workdir"]))
    print("Done. Get your next case with:  python -m reviewtool next --amend   (or  next)")


def cmd_mystats(a):
    """Private self-service scorecard: YOUR own submissions, pass-rate, and amend queue."""
    s, base = _api()
    r = s.get(base + "/me/stats", timeout=60)
    r.raise_for_status()
    d = r.json()
    n, p, pct = d.get("submissions", 0), d.get("passed", 0), d.get("pass_pct")
    print(f"\nYour review scorecard ({d.get('reviewer', 'you')}):")
    print(f"  submissions checked : {n}")
    print(f"  passing the QC      : {p}" + (f"   ({pct}%)" if pct is not None else ""))
    print(f"  still to fix (amend): {d.get('amend_pending', 0)}")
    fb = d.get("fail_by_check") or {}
    if fb:
        print("  fails by check      : " + ", ".join(f"{k}={v}" for k, v in fb.items()))
    if d.get("amend_pending"):
        print("\n  Fix your flagged cases:  python -m reviewtool next --amend")


def cmd_status(a):
    s, base = _api()
    r = s.get(base + "/status", timeout=60)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))


def cmd_reference(a):
    """Open the published gold reference example in ITK-SNAP (standalone — not
    tied to a claimed case). Downloads crops/reference/ from the v2 dataset."""
    from huggingface_hub import hf_hub_download
    work = Path(a.workdir) / "_reference"
    work.mkdir(parents=True, exist_ok=True)
    try:
        ct = hf_hub_download(repo_id=a.repo, repo_type="dataset", revision=a.revision,
                             filename="crops/reference/ct.nii.gz", local_dir=str(work))
        seg = hf_hub_download(repo_id=a.repo, repo_type="dataset", revision=a.revision,
                              filename="crops/reference/seg.nii.gz", local_dir=str(work))
    except Exception:
        print(f"no gold reference is published (crops/reference/ not found on "
              f"{a.repo}@{a.revision}).")
        return
    lbl = work / "reference_labels.txt"
    lbl.write_text(labels_descriptor.descriptor_text())
    print("opening the gold reference example — close it when done.")
    _launch_itksnap(a.itksnap or _default_itksnap(), Path(ct), Path(seg), lbl)
    return 0


def cmd_final_pass(a):
    """Local final QC pass over the flagged (finalized) cases.

    Reads straight from HuggingFace — the review ledger (REVIEW_REPO) for the
    finalized worklist + each case's corrected (final) label, and the v2 dataset
    for the matching crop CT — so no review Space is involved. Opens each crop
    CT with its corrected label in ITK-SNAP, one at a time, and records an
    approve/flag verdict. Resumable: re-running skips cases already logged
    (pass --redo to re-review them). The flag list is your kick-back worklist.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    work = Path(a.workdir).expanduser()
    work.mkdir(parents=True, exist_ok=True)
    log_path = Path(a.log).expanduser() if a.log else work / "verdicts.json"
    verdicts = json.loads(log_path.read_text()) if log_path.exists() else {}

    # 1) pull just the ledger's tiny case files and derive the finalized worklist
    print(f"loading the review ledger from {a.review_repo} ...")
    ledger = snapshot_download(repo_id=a.review_repo, repo_type="dataset",
                               revision=a.review_revision,
                               allow_patterns=["cases/*"])
    cases = []
    for cf in sorted((Path(ledger) / "cases").glob("*.json")):
        try:
            cases.append(json.loads(cf.read_text()))
        except Exception as exc:                       # noqa: BLE001
            print(f"  skip {cf.name}: {exc}")
    flagged = [c for c in cases
               if schema.derive_status(c) == "finalized" and c.get("final")]
    if a.only == "corrected":
        flagged = [c for c in flagged
                   if c["final"].get("decision") == "corrected"]
    flagged.sort(key=lambda c: c["case_id"])
    todo = [c for c in flagged if a.redo or c["case_id"] not in verdicts]
    print(f"{len(flagged)} flagged case(s); {len(todo)} to review "
          f"({len(flagged) - len(todo)} already logged in {log_path.name}).")
    if not todo:
        print(f"nothing left — every flagged case has a verdict. log: {log_path}")
        return _final_pass_summary(verdicts, log_path)

    itksnap = a.itksnap or _default_itksnap()
    labels = work / "labels.txt"
    labels.write_text(labels_descriptor.descriptor_text())
    print("per case, after ITK-SNAP closes: [Enter]=approve  f=flag  s=skip  q=quit\n")

    for i, c in enumerate(todo, 1):
        cid, fin = c["case_id"], c["final"]
        ct_rel = c.get("ct_file")
        lbl_rel = fin.get("label_rel")
        print(f"[{i}/{len(todo)}] {cid}   decision={fin.get('decision')}")
        if not ct_rel or not lbl_rel:
            print("  missing ct_file or final label_rel — skipping.")
            continue
        cdir = work / "cases" / cid
        try:
            ct = hf_hub_download(repo_id=a.v2_repo, repo_type="dataset",
                                 revision=a.revision, filename=ct_rel,
                                 local_dir=str(cdir))
            seg = hf_hub_download(repo_id=a.review_repo, repo_type="dataset",
                                  revision=a.review_revision, filename=lbl_rel,
                                  local_dir=str(cdir))
        except Exception as exc:                       # noqa: BLE001
            print(f"  download failed ({exc}); skipping.")
            continue
        rc = _launch_itksnap(itksnap, Path(ct), Path(seg), labels)
        if rc != 0:
            print(f"  ITK-SNAP exited {rc} — no verdict recorded.")
            _itksnap_failure_hint(rc)
            if input("  [q]uit / anything else = continue: ").strip().lower() \
                    in ("q", "quit"):
                break
            continue
        ans = input("  verdict [Enter=approve / f=flag / s=skip / q=quit]: ") \
            .strip().lower()
        if ans in ("q", "quit"):
            print("  stopping; progress saved.")
            break
        if ans in ("s", "skip"):
            continue
        note = ""
        if ans in ("f", "flag"):
            verdict, note = "flag", input("  what's wrong (short note): ").strip()
        else:
            verdict = "approve"
        verdicts[cid] = {"verdict": verdict, "note": note,
                         "decision": fin.get("decision"),
                         "label_rel": lbl_rel}
        log_path.write_text(json.dumps(verdicts, indent=2))   # save after each

    return _final_pass_summary(verdicts, log_path)


def _final_pass_summary(verdicts: dict, log_path: Path) -> int:
    appr = [k for k, v in verdicts.items() if v["verdict"] == "approve"]
    flg = [k for k, v in verdicts.items() if v["verdict"] == "flag"]
    print("\n=== final pass ===")
    print(f"  approved : {len(appr)}")
    print(f"  flagged  : {len(flg)}")
    for k in flg:
        print(f"    - {k}: {verdicts[k].get('note', '')}")
    print(f"  log: {log_path}")
    return 0


def cmd_resume(a):
    """Re-upload any edits whose submit didn't confirm (crash / 429 / network).
    The edited seg.nii.gz + downloaded pseudo are still in each workdir, so the
    review record is recomputed and re-sent; the server is idempotent."""
    s, base = _api()
    pend = sorted(ACTIVE_DIR.glob("*.json")) if ACTIVE_DIR.exists() else []
    if not pend:
        print("nothing to resume — no saved claims.")
        return
    print(f"{len(pend)} saved claim(s) to resume.")
    for f in pend:
        st = json.loads(f.read_text())
        job, work = st["job"], Path(st["workdir"])
        kind = st.get("kind", "review")
        seg = work / "seg.nii.gz"
        if not seg.exists():
            print(f"skip {work.name}: edited label missing ({seg})")
            continue
        pseudo = work / "pseudo.nii.gz"
        if kind == "adjudicate":
            # An adjudication is meaningless without an actual deciding edit.
            if pseudo.exists() and seg.read_bytes() == pseudo.read_bytes():
                print(f"skip {work.name}: no edits detected — never reviewed or "
                      f"not saved. Review it with `reviewtool edit {work.name}`.")
                continue
            print(f"resuming {work.name} ({kind}) ...")
            _submit_adjudication(s, base, job, work, seg, st.get("notes", ""))
        else:
            if not pseudo.exists():
                print(f"skip {work.name}: pseudo label missing; cannot recompute diff")
                continue
            _, record = build_submission(
                _load(pseudo), _load(seg), job["region_to_review"],
                _sha256(pseudo))
            if _resume_action(record, st.get("accepted", False)) == "skip":
                print(f"skip {work.name}: no edits detected — never reviewed or "
                      f"not saved (would submit the raw pseudo-label as an "
                      f"accept). Review it with `reviewtool edit {work.name}`.")
                continue
            print(f"resuming {work.name} ({kind}) ...")
            _submit_review(s, base, job, work, seg, record)


def cmd_edit(a):
    """Re-open an already-claimed case in ITK-SNAP and submit it — `next`
    without the claim+download. Use after an ITK-SNAP crash or a fresh install
    to finish cases sitting in ~/.reviewtool/active/ from a downloaded workdir."""
    s, base = _api()
    pend = sorted(ACTIVE_DIR.glob("*.json")) if ACTIVE_DIR.exists() else []
    names = [Path(json.loads(f.read_text())["workdir"]).name for f in pend]
    if not pend:
        print("no saved claims to edit — run `reviewtool next` to claim one.")
        return
    target = a.case
    if target is None:
        if len(pend) == 1:
            target = names[0]
        else:
            print("multiple saved claims — pass one as `reviewtool edit <case>`:")
            for n in names:
                print("  ", n)
            return
    match = [f for f, n in zip(pend, names) if n == target]
    if not match:
        print(f"no saved claim named {target!r}. Saved claims:")
        for n in names:
            print("  ", n)
        return

    st = json.loads(match[0].read_text())
    job, work = st["job"], Path(st["workdir"])
    kind = st.get("kind", "review")
    labels = work / "labels.txt"
    # rewrite every launch so the palette matches the served labels (VerSe-native for v3/v4)
    labels.write_text(_descriptor_for_job(job))
    pseudo = work / "pseudo.nii.gz"
    seg = work / "seg.nii.gz"                     # full-res output (built on save)
    crop = job.get("crop")

    if crop:                                      # crop review: open the CROP ct+mask
        snap_ct = work / "ct.nii.gz"              # crop CT (matches crop mask dims)
        snap_seg = work / "crop_edit.nii.gz"      # the crop mask the reviewer edits
        need = [snap_ct, snap_seg, pseudo]
    else:                                         # full-scan review
        snap_ct, snap_seg, need = work / "ct.nii.gz", seg, [work / "ct.nii.gz", seg]
    missing = [p.name for p in need if not p.exists()]
    if missing:
        print(f"workdir incomplete ({work}): missing {', '.join(missing)} — "
              "re-claim with `reviewtool next`.")
        return

    print(f"editing {work.name} ({kind}) — review the "
          f"{job['region_to_review']} region")
    before_qc = (work / "crop_seg.nii.gz") if crop else pseudo
    # LSTV + WHY FLAGGED in the background so ITK-SNAP opens immediately.
    import threading

    def _startup_info():
        _show_startup(job, crop, work, snap_ct, before_qc, kind != "adjudicate")
    threading.Thread(target=_startup_info, daemon=True).start()
    snap = a.itksnap or _default_itksnap()
    ref_proc = (_open_space_reference(job, snap, work)
                if getattr(a, "reference", False) else None)
    pre_mtime = snap_seg.stat().st_mtime if snap_seg.exists() else 0.0
    region = job.get("region_to_review")
    rib_passed = None
    if region == "ribs":                          # live rib QC + client gate (same as `next`)
        rc, rib_passed = _watch_anatomy(snap, snap_ct, snap_seg, labels, check="ribs")
    else:
        rc = _watch_itksnap(snap, snap_ct, snap_seg, labels,
                            qc_ct=snap_ct, before=before_qc,
                            live_qc=getattr(a, "live_qc", False))
    if ref_proc is not None:
        ref_proc.terminate()
    saved = snap_seg.exists() and snap_seg.stat().st_mtime > pre_mtime + 1e-6
    if not saved:                            # only NOW does the exit code matter (nothing saved)
        if rc != 0:
            _abnormal_close_alert(rc, work.name, snap_seg,
                                  check="ribs" if region == "ribs" else "both")
        else:
            print("ITK-SNAP closed but you never SAVED - no edit captured, nothing "
                  "submitted; your claim is kept.\n"
                  "  Edit, then Save Segmentation (Ctrl-S / Cmd-S) before quitting "
                  "(one save also 'accepts' a correct draft)."
                  + (" If it auto-closed, relaunch with --itksnap "
                     "/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP"
                     if sys.platform == "darwin" else ""))
        return
    # You SAVED -> proceed on QC, not on ITK-SNAP's (Windows-flaky) exit code.

    if crop:                                      # fold the crop edit into full-res
        _paste_edit_to_full(snap_seg, crop["origin"], pseudo, seg)

    if kind == "adjudicate":
        if region == "ribs" and rib_passed is not True and not getattr(a, "force", False):
            _qc_hold_alert(work.name, snap_seg, check="ribs")   # gate the adjudicator too
            return
        _submit_adjudication(s, base, job, work, seg, st.get("notes", ""))
    else:
        if region == "ribs" and rib_passed is not True:
            print("\n[HELD] the ribs still fail QC (a rib number is in two pieces) — NOT "
                  "submitting (the server would reject it). Re-open and finish:\n"
                  f"      python -m reviewtool edit {work.name}")
            return
        _, record = build_submission(
            _load(pseudo), _load(seg), job["region_to_review"], _sha256(pseudo))
        print(f"decision={record['decision']}  "
              f"voxels_changed={record['diff']['n_voxels_changed']}")
        if record["decision"] == "accept":
            _save_active(job, work, kind="review", accepted=True)
        _submit_review(s, base, job, work, seg, record)


def _load_manifest_local(path: Path) -> list:
    payload = json.loads(Path(path).read_text())
    if isinstance(payload, dict):
        payload = payload.get("records", payload.get("cases", []))
    return [r for r in payload if isinstance(r, dict)]


def _is_flag(v) -> bool:
    return str(v).strip() in ("1", "1.0", "True", "true")


def _flag_hint(row: dict) -> str:
    """Human 'what to look for' from a merged-QC row, so the reviewer opens each
    case knowing the suspected defect."""
    h = []
    if _is_flag(row.get("mixing_flag")):
        h.append(f"vertebra MIXING (off_main={row.get('off_main_frac','?')}, "
                 f"order-inv={row.get('n_order_inversions','?')})")
    if _is_flag(row.get("leak_flag")):
        h.append(f"OFF-BONE leak (off_bone={row.get('off_bone_frac','?')})")
    if _is_flag(row.get("struct_flag")):
        s = []
        if _is_flag(row.get("lr_swap")):
            s.append("L/R HIP SWAP")
        if str(row.get("vertebra_gap", "0")).strip() not in ("", "0", "0.0"):
            s.append("missing level")
        if _is_flag(row.get("pelvis_incomplete")):
            s.append("pelvis incomplete")
        if _is_flag(row.get("duplication_flag")):
            s.append("duplicated structure")
        h.append("structure: " + (", ".join(s) if s else "issue"))
    return "  |  ".join(h) or "(flagged)"


def _fixlist_rows(rows: list, only_flagged: bool = True) -> list:
    """Rows to review, preserving the CSV's worst-first order."""
    if not only_flagged:
        return list(rows)
    return [r for r in rows if _is_flag(r.get("needs_review"))
            or _is_flag(r.get("mixing_flag")) or _is_flag(r.get("leak_flag"))
            or _is_flag(r.get("struct_flag"))]


def _paste_crop_to_full(crop_seg: Path, crop_json: Path, full_src: Path, dst: Path):
    """Fold an edited ROI crop back into the full-res mask: start from the
    original full label, overwrite the crop's voxel box with the edit, save."""
    import json as _json
    import numpy as np
    import nibabel as nib
    from export_review_crops import paste_back
    meta = _json.loads(Path(crop_json).read_text())
    full_img = nib.load(str(full_src))
    full = np.asarray(full_img.dataobj)
    crop = np.asarray(nib.load(str(crop_seg)).dataobj)
    merged = paste_back(full, crop, meta["origin"])
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(merged.astype(full.dtype), full_img.affine,
                             full_img.header), str(dst))


def _paste_edit_to_full(edit_seg: Path, origin, full_label: Path, dst: Path):
    """Paste an edited crop mask into the full-res label at voxel `origin`, save
    to dst. Lets a crop-reviewed case submit a full label (unchanged server)."""
    import numpy as np
    import nibabel as nib
    from export_review_crops import paste_back
    fi = nib.load(str(full_label))
    full = np.asarray(fi.dataobj)
    crop = np.asarray(nib.load(str(edit_seg)).dataobj)
    merged = paste_back(full, crop, origin)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(merged.astype(full.dtype), fi.affine, fi.header), str(dst))


def cmd_fix_list(a):
    """Walk a merged-QC CSV worst-first, open each flagged case in ITK-SNAP for
    a student to fix, and save the corrected label into a reviewed tree. LOCAL
    (no Space): reads the pseudo tree on disk; resumable (skips cases already in
    the out tree)."""
    import csv as _csv
    import shutil as _shutil
    qc = Path(a.qc_csv)
    if not qc.exists():
        sys.exit(f"QC CSV not found: {qc}")
    rows = _fixlist_rows(list(_csv.DictReader(open(qc))), only_flagged=not a.all)
    if a.limit:
        rows = rows[:a.limit]
    if not rows:
        print(f"nothing to fix — no flagged cases in {qc}")
        return

    tree = Path(a.tree)
    index = {(str(r.get("token")), str(r.get("config"))): r
             for r in _load_manifest_local(tree / "manifest.json")}
    out = Path(a.out) if a.out else tree.parent / (tree.name + "_reviewed")
    (out / "labels").mkdir(parents=True, exist_ok=True)
    labels_txt = out / "labels.txt"
    labels_txt.write_text(labels_descriptor.descriptor_text())
    itksnap = a.itksnap or _default_itksnap()

    # Optional persistent "good example" window beside the review window.
    ref_proc = None
    if a.reference:
        rdir = Path(a.reference)
        rct, rseg, rlbl = rdir / "ct.nii.gz", rdir / "seg.nii.gz", rdir / "labels.txt"
        if not rlbl.exists():
            rlbl.write_text(labels_descriptor.descriptor_text())
        if rct.exists() and rseg.exists():
            print(f"opening reference example beside review: {rdir.name} "
                  "(leave it open to compare against)\n")
            ref_proc = _launch_itksnap_bg(itksnap, rct, rseg, rlbl)
        else:
            print(f"reference {rdir} missing ct/seg — continuing without the example panel")

    print(f"{len(rows)} flagged case(s) to review — corrected labels go to {out}\n"
          f"(edit in ITK-SNAP, Save Segmentation, quit to advance; Ctrl-C to stop)\n")
    n_done = n_skip = 0
    for i, row in enumerate(rows, 1):
        key = (str(row.get("token")), str(row.get("config")))
        rec = index.get(key)
        if not rec or not rec.get("label_file") or not rec.get("ct_file"):
            print(f"[{i}/{len(rows)}] {key}: not in manifest — skip"); n_skip += 1
            continue
        src_ct, src_pseudo = tree / rec["ct_file"], tree / rec["label_file"]
        dst = out / rec["label_file"]
        if dst.exists() and not a.redo:
            n_skip += 1
            continue                                     # already reviewed
        if not src_ct.exists() or not src_pseudo.exists():
            print(f"[{i}/{len(rows)}] {key}: missing CT/label in tree — skip")
            n_skip += 1
            continue
        print(f"[{i}/{len(rows)}] token={key[0]}  {key[1]}")
        print(f"    WHY FLAGGED: {_flag_hint(row)}")

        if a.crops:
            # Open the small ROI crop; paste the edit back into the full-res mask.
            cdir = Path(a.crops) / crop_dirname(key[0], key[1])
            cj = cdir / "crop.json"
            if not (cdir / "ct.nii.gz").exists() or not cj.exists():
                print(f"    no crop in {cdir} — run export_review_crops.py; skip")
                n_skip += 1
                continue
            crop_seg = cdir / "seg.nii.gz"
            ctx = None
            if a.full and src_ct.exists():
                print("    opening FULL scan (read-only) for context ...")
                ctx = _launch_itksnap_bg(itksnap, src_ct, src_pseudo, labels_txt)
            rc = _launch_itksnap(itksnap, cdir / "ct.nii.gz", crop_seg, cdir / "labels.txt")
            if ctx is not None:
                ctx.terminate()
            if rc != 0:
                print(f"    not saving — rerun to redo.")
                _itksnap_failure_hint(rc)
                n_skip += 1
                continue
            _paste_crop_to_full(crop_seg, cj, src_pseudo, dst)
            print(f"    saved full-res -> {dst}")
            n_done += 1
            continue

        dst.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(str(src_pseudo), str(dst))         # editable copy; SAVE here
        rc = _launch_itksnap(itksnap, src_ct, dst, labels_txt)
        if rc != 0:
            print(f"    left the unedited copy — rerun to redo.")
            _itksnap_failure_hint(rc)
        n_done += 1
    if ref_proc is not None:
        ref_proc.terminate()
    print(f"\ndone: {n_done} opened, {n_skip} skipped. Reviewed tree: {out}")


# Approx sizes of the v2 dataset tree (measured June 2026) so `download` can
# warn before a huge pull. ct/ dominates; labels + manifest are tiny.
_DL = {
    "full":   (253.0, None,                                 "v2",   "everything in v2 (ct + labels + crops + manifest)"),
    "ct":     (241.0, ["ct/*", "manifest.*", "splits_*.json"], "v2", "full-res CT volumes only (+ manifest/splits)"),
    "labels": (1.9,   ["labels/*", "manifest.*", "splits_*.json"], "v2", "label maps + manifest/splits (tiny; pair with CTs you already have)"),
    "crops":  (10.0,  ["crops/*"],                          "v2",   "the review ROI crops only (the flagged worklist)"),
    "sample": (10.7,  None,                                 "main", "the anonymized review sample (main branch)"),
}


def cmd_download(a):
    """Download the dataset from HuggingFace. `--what` picks how much."""
    from huggingface_hub import snapshot_download
    approx, patterns, default_rev, desc = _DL[a.what]
    rev = a.revision or default_rev
    out = Path(a.out).expanduser()
    print(f"download '{a.what}': {desc}")
    print(f"  repo={a.repo}@{rev}  ->  {out}")
    print(f"  approx size: ~{approx:.0f} GB")
    if approx >= 50 and not a.yes:
        print(f"\n  This is LARGE (~{approx:.0f} GB) and may take a long time / lots "
              f"of disk.\n  If you really want it, re-run with --yes. (Most users "
              f"want --what labels or --what crops.)")
        return 1
    out.mkdir(parents=True, exist_ok=True)
    snapshot_download(repo_id=a.repo, repo_type="dataset", revision=rev,
                      local_dir=str(out), allow_patterns=patterns,
                      max_workers=a.workers)
    print(f"\ndone -> {out}")
    return 0


def cmd_review_cases(a):
    """Review SPECIFIC cases (by --tokens and/or --config) pulled straight from a
    HuggingFace dataset revision in ITK-SNAP, saving the corrected labels locally
    and (optionally) pushing them back. LOCAL editing, NO review Space — the way to
    owner-review a small, named cohort (e.g. the pelvic_native spines). Resumable:
    a case already in --out is skipped (use --redo to re-open).

      python -m reviewtool review-cases --repo anonymous-mlhc/CTSpinoPelvic1K \
          --revision v4 --config pelvic_native --out ./pelvic_review [--push]
      python -m reviewtool review-cases --repo ... --revision v4 --tokens 22,46,61 --out ./r
    """
    from huggingface_hub import hf_hub_download
    import shutil as _shutil
    work = Path(a.out).expanduser()
    (work / "labels").mkdir(parents=True, exist_ok=True)
    man = json.loads(Path(hf_hub_download(a.repo, "manifest.json", repo_type="dataset",
                                          revision=a.revision)).read_text())
    if isinstance(man, dict):
        man = man.get("records") or man.get("cases") or []
    toks = set(t.strip() for t in a.tokens.split(",")) if a.tokens else None
    sel = [r for r in man
           if (toks is None or str(r.get("token")) in toks)
           and (a.config is None or r.get("config") == a.config)]
    if a.limit:
        sel = sel[:a.limit]
    if not sel:
        print("no matching cases (check --tokens / --config against the manifest).")
        return
    labels_txt = work / "labels.txt"
    labels_txt.write_text(labels_descriptor.verse_native_descriptor_text())  # real v3/v4 ids
    itksnap = a.itksnap or _default_itksnap()
    print(f"{len(sel)} case(s) to review from {a.repo}@{a.revision} — corrected labels "
          f"-> {work}/labels/\n(edit in ITK-SNAP, Save Segmentation, quit to advance; "
          f"Ctrl-C to stop)\n")
    n = 0
    for i, rec in enumerate(sel, 1):
        cf, lf = rec.get("ct_file"), rec.get("label_file")
        if not cf or not lf:
            print(f"[{i}] token {rec.get('token')}: no ct/label in manifest — skip"); continue
        dst = work / lf
        if dst.exists() and not a.redo:
            print(f"[{i}] token {rec.get('token')}: already reviewed — skip"); continue
        cdir = work / "_dl" / str(rec.get("token"))
        cdir.mkdir(parents=True, exist_ok=True)
        try:
            ct = hf_hub_download(a.repo, cf, repo_type="dataset", revision=a.revision,
                                 local_dir=str(cdir), token=a.token or os.environ.get("HF_TOKEN"))
            lbl = hf_hub_download(a.repo, lf, repo_type="dataset", revision=a.revision,
                                  local_dir=str(cdir), token=a.token or os.environ.get("HF_TOKEN"))
        except Exception as exc:                       # noqa: BLE001
            print(f"[{i}] token {rec.get('token')}: download failed ({str(exc)[:80]}) — skip"); continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        _shutil.copy2(lbl, str(dst))                   # editable copy — SAVE here in ITK-SNAP
        print(f"[{i}/{len(sel)}] token={rec.get('token')}  {rec.get('config')}  "
              f"(LSTV={rec.get('lstv_label', '?')})")
        if a.check == "none":                              # no QC gate requested
            rc = _launch_itksnap(itksnap, Path(ct), dst, labels_txt)
            if rc != 0:
                print("    ITK-SNAP failed — rerun to redo this case."); _itksnap_failure_hint(rc); continue
            n += 1
            continue
        # QC-GATED: a case is kept as resolved (and later uploaded) ONLY if it PASSES.
        passed = None
        while True:
            rc, passed = _watch_anatomy(itksnap, Path(ct), dst, labels_txt, check=a.check)
            if rc != 0:
                print("    ITK-SNAP failed — rerun to redo this case."); _itksnap_failure_hint(rc)
                passed = False
                break
            if passed:
                break
            why = ("still FAILS the rib QC" if passed is False
                   else "could not be QC-checked (install scipy + nibabel)")
            print(f"    [LOCKED] token {rec.get('token')} {why} -> NOT saved as resolved, "
                  f"NOT uploaded.")
            try:
                ans = input("    reopen in ITK-SNAP to keep fixing? [Y/n] ").strip().lower()
            except EOFError:
                ans = "n"
            if ans == "n":
                break
        if not passed:                                     # unresolved: drop the local copy so it
            try:                                           # is neither pushed nor counted, and
                dst.unlink()                               # re-downloads/re-opens on the next run
            except OSError:
                pass
            print(f"    token {rec.get('token')}: left UNRESOLVED (re-run to finish).")
            continue
        n += 1
    print(f"\ndone: {n} reviewed. Corrected labels in {work}/labels/ (same paths as the dataset).")
    if a.push:                                          # push whatever is in --out (this run or prior)
        from huggingface_hub import HfApi, CommitOperationAdd
        tok = a.token or os.environ.get("HF_TOKEN")
        if not tok:
            print("--push needs a write token (--token or HF_TOKEN)."); return
        ops = [CommitOperationAdd(path_in_repo=r["label_file"], path_or_fileobj=str(work / r["label_file"]))
               for r in sel if (work / r["label_file"]).exists()]
        if not ops:
            print("nothing to push — no reviewed labels in --out yet."); return
        HfApi(token=tok).create_commit(
            repo_id=a.repo, repo_type="dataset", revision=a.revision, operations=ops,
            commit_message=f"review-cases: corrected {len(ops)} label(s)")
        print(f"pushed {len(ops)} corrected label(s) to {a.repo}@{a.revision}. "
              f"(promote main separately if needed.)")
    elif not a.push:
        print("Not pushed. When satisfied, re-run with --push (or fold into your tree + re-export).")


def _disagreement_descriptor() -> str:
    """ITK-SNAP palette for the 4-class reviewer-disagreement map."""
    return (
        "################################################\n"
        "# ITK-SNAP disagreement map (READ-ONLY view)\n"
        "# 1 agree  2 reviewer-1 only  3 reviewer-2 only  4 conflict\n"
        "################################################\n"
        '    0     0    0    0        0  0  0    "Clear Label"\n'
        '    1   130  130  130        1  1  1    "1 agree (both)"\n'
        '    2   240   40   40        1  1  1    "2 reviewer-1 only"\n'
        '    3    40  110  240        1  1  1    "3 reviewer-2 only"\n'
        '    4   255  215    0        1  1  1    "4 conflict (differ)"\n'
    )


def _build_disagreement_map(a_arr, b_arr, region: str):
    """4-class map over the region's id range: 1 agree, 2 reviewer-1 only, 3 reviewer-2 only,
    4 both-labelled-but-different. Returns (uint8 map, lo, hi)."""
    import numpy as np
    lo, hi = {"ribs": (34, 57), "spine": (1, 29)}.get(region, (1, 57))
    ina = (a_arr >= lo) & (a_arr <= hi)
    inb = (b_arr >= lo) & (b_arr <= hi)
    out = np.zeros(a_arr.shape, dtype=np.uint8)
    out[ina & inb & (a_arr == b_arr)] = 1
    out[ina & ~inb] = 2
    out[inb & ~ina] = 3
    out[ina & inb & (a_arr != b_arr)] = 4
    return out, lo, hi


def _print_disagreement_summary(a_arr, b_arr, lo: int, hi: int) -> None:
    """Per-structure contested list (worst overlap-Dice first) so you know WHICH ribs differ."""
    import os as _os
    import sys as _sys
    for _p in (_os.path.join(_os.path.dirname(__file__), "..", "scripts"), "scripts"):
        if _os.path.isdir(_p) and _p not in _sys.path:
            _sys.path.insert(0, _p)
    try:
        import label_scheme as LS
        names = {v: k for k, v in LS.label_dict().items()}
    except Exception:                                    # noqa: BLE001
        names = {}
    import numpy as np
    # counts per id in 3 whole-volume passes (bincount), not 24 masked .sum() passes
    ina = (a_arr >= lo) & (a_arr <= hi)
    inb = (b_arr >= lo) & (b_arr <= hi)
    ca = np.bincount(a_arr[ina].astype(np.int64).ravel(), minlength=hi + 1)
    cb = np.bincount(b_arr[inb].astype(np.int64).ravel(), minlength=hi + 1)
    same = ina & (a_arr == b_arr)
    ci = np.bincount(a_arr[same].astype(np.int64).ravel(), minlength=hi + 1)
    rows = []
    for i in range(lo, hi + 1):
        na, nb, inter = int(ca[i]), int(cb[i]), int(ci[i])
        if na == 0 and nb == 0:
            continue
        dice = (2 * inter / (na + nb)) if (na + nb) else 1.0
        if dice >= 0.999:
            continue
        rows.append((dice, names.get(i, f"id {i}"), na, nb))
    if not rows:
        print("  reviewers AGREE on every structure in range — nothing contested.")
        return
    rows.sort()
    print("  contested structures (reviewer-1 vs reviewer-2), worst overlap first:")
    for dice, nm, na, nb in rows:
        tag = "MISSING in one" if (na == 0 or nb == 0) else "differ"
        print(f"    {nm:<16} R1={na:>7}  R2={nb:>7}  overlap-Dice={dice:.2f}  ({tag})")


def cmd_disagreement(a):
    """READ-ONLY: show WHERE the two reviewers disagree for an adjudication case in ITK-SNAP.

    Downloads only — the two reviewers' stored labels + the case CT are FETCHED, never
    modified, and NOTHING is written to the review repo. Student segmentations are untouched.
    Toggle the label visibility in ITK-SNAP to isolate conflicts (hide '1 agree')."""
    from huggingface_hub import hf_hub_download
    import numpy as np
    import nibabel as nib
    work = Path(a.workdir).expanduser() / f"{a.case}__disagree"
    work.mkdir(parents=True, exist_ok=True)
    rr, rrev = a.review_repo, a.review_revision

    def _dl(repo, rev, fn):
        return hf_hub_download(repo_id=repo, repo_type="dataset", revision=rev,
                               filename=fn, local_dir=str(work))

    try:
        casef = _dl(rr, rrev, f"cases/{a.case}.json")
    except Exception as exc:                             # noqa: BLE001
        print(f"could not read case {a.case} from {rr}@{rrev}: {str(exc)[:120]}\n"
              f"  check the case id, and that `hf auth login` has READ access to the private "
              f"review repo.")
        return
    case = json.loads(Path(casef).read_text())
    region = case.get("region_to_review", "ribs")
    ctrev = a.ct_revision or case.get("source_revision", "v4")

    labs = {}
    for slot in ("1", "2"):
        sl = (case.get("slots", {}) or {}).get(slot) or {}
        lp = sl.get("label_path") or f"reviews/{a.case}/{slot}_label.nii.gz"
        try:
            labs[slot] = _dl(rr, rrev, lp)
        except Exception as exc:                         # noqa: BLE001
            print(f"  reviewer {slot}: label not found ({lp}): {str(exc)[:80]}")
    if len(labs) < 2:
        print(f"need BOTH reviewers' labels to show a disagreement — only {len(labs)} present "
              f"(has the 2nd review been submitted for {a.case}?).")
        return

    ct = _dl(a.ct_repo, ctrev, case["ct_file"])
    A, B = nib.load(labs["1"]), nib.load(labs["2"])
    a_arr, b_arr = np.asanyarray(A.dataobj), np.asanyarray(B.dataobj)
    if a_arr.shape != b_arr.shape:
        print(f"reviewer label shapes differ {a_arr.shape} vs {b_arr.shape} — cannot diff.")
        return
    dmap, lo, hi = _build_disagreement_map(a_arr, b_arr, region)
    out = work / "disagreement.nii.gz"
    nib.save(nib.Nifti1Image(dmap, A.affine, A.header), str(out))
    desc = work / "disagreement_labels.txt"
    desc.write_text(_disagreement_descriptor())

    print(f"\nDISAGREEMENT for {a.case}  (region: {region}, reviewer-1 vs reviewer-2)")
    _print_disagreement_summary(a_arr, b_arr, lo, hi)
    print("\n  Opening ITK-SNAP (READ-ONLY view — nothing is written back). Colours:")
    print("    1 agree (grey)   2 reviewer-1 only (red)   3 reviewer-2 only (blue)   "
          "4 conflict (yellow)")
    print("  Toggle a label's eye in the Segmentation Labels panel to isolate it — hide")
    print("  '1 agree' to see ONLY where they differ. This is just a viewer; make the actual")
    print("  decision in your adjudicate/edit window.")
    _launch_itksnap(a.itksnap or _default_itksnap(), Path(ct), out, desc)


def _open_disagreement_ref(s, base, job, ct, work, a):
    """Best-effort: open ONE read-only DISAGREE window (v4 palette, colored ribs) beside the
    editor — just the places the two reviewers differ, so you know where to focus. Streams the two
    reviewer labels THROUGH the Space (adjudicator only — a personal login can't read the private
    review repo directly). Never writes to the ledger. Returns a list with the one Popen handle (or [])."""
    import numpy as np
    import nibabel as nib
    cid = job["case_id"]
    try:
        l1 = _fetch_adj_label(s, base, cid, "1", Path(work) / "r1_label.nii.gz")
        l2 = _fetch_adj_label(s, base, cid, "2", Path(work) / "r2_label.nii.gz")
        i1 = nib.load(str(l1))
        A, B = np.asanyarray(i1.dataobj), np.asanyarray(nib.load(l2).dataobj)
        if A.shape != B.shape:
            print("  (disagree: reviewer label shapes differ — skipping the reference window)")
            return []
        dis = A.copy(); dis[A == B] = 0                          # reviewer-1's label where they differ
        m = (A != B) & (A == 0); dis[m] = B[m]                   # ...fill reviewer-2 where r1 is bg
        ds = Path(work) / "disagree.nii.gz"
        nib.save(nib.Nifti1Image(dis.astype(A.dtype), i1.affine, i1.header), str(ds))
        desc = Path(work) / "verse_labels.txt"
        desc.write_text(labels_descriptor.verse_native_descriptor_text())
        print(f"\n  READ-ONLY DISAGREE window opened ({int((A != B).sum())} voxels differ). "
              f"Click 'Update' in its 3D pane to see the spots to focus on in 3D.")
        return [_launch_itksnap_bg(a.itksnap or _default_itksnap(), Path(ct), ds, desc)]
    except Exception as exc:                             # noqa: BLE001
        print(f"  (disagree window unavailable: {str(exc)[:90]} — the editor still opens; "
              f"pass --no_disagreement to skip it.)")
        return []


def main(argv=None) -> int:
    for _s in (sys.stdout, sys.stderr):                  # never crash on a non-cp1252 char
        try:
            _s.reconfigure(errors="replace")
        except Exception:                                # noqa: BLE001 — older streams
            pass
    ap = argparse.ArgumentParser(prog="reviewtool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("login"); p.add_argument("--service", required=True)
    p.add_argument("--key", default=None,
                   help="(optional) legacy reviewer key; omit to use your "
                        "HuggingFace login")
    p.set_defaults(fn=cmd_login)

    p = sub.add_parser("mystats",
                       help="your OWN private scorecard: submissions, QC pass-rate, amend queue")
    p.set_defaults(fn=cmd_mystats)

    p = sub.add_parser("skip",
                       help="defer a scan you claimed / clear stale local claims (put back in queue)")
    p.add_argument("case", nargs="?", default=None, help="case id (omit if you hold only one)")
    p.add_argument("--all", action="store_true", help="clear ALL your held/stale local claims")
    p.set_defaults(fn=cmd_skip)

    for name, fn in (("next", cmd_next), ("adjudicate", cmd_adjudicate)):
        p = sub.add_parser(name)
        p.add_argument("--workdir", default=str(Path.home() / ".reviewtool" / "work"))
        p.add_argument("--itksnap", default=None,
                       help="ITK-SNAP executable (auto-detected if omitted)")
        if name == "next":
            p.add_argument("--full", action="store_true",
                           help="for a crop case, ALSO pull the full scan from the "
                                "original repo and open it read-only beside the crop "
                                "(to verify the rest of the volume looks fine)")
            p.add_argument("--no_qc", action="store_true",
                           help="skip the QC focus/progress prints (needs scipy; "
                                "QC is on by default)")
            p.add_argument("--no_watch", action="store_true",
                           help="don't keep ITK-SNAP open for live QC; use the "
                                "old quit-to-submit behaviour")
            p.add_argument("--reference", action="store_true",
                           help="also open the gold reference example in a 2nd "
                                "window (off by default)")
            p.add_argument("--live_qc", action="store_true",
                           help="re-run the QC on every Save (live progress to OK). "
                                "Off by default — faster, since AI-assisted fixes "
                                "rarely need re-verification.")
            p.add_argument("--amend", action="store_true",
                           help="fix YOUR OWN cases re-opened because their earlier submission "
                                "failed QC (loads your previous label to edit, not the pseudo)")
        if name == "adjudicate":
            p.add_argument("--notes", default="")
            p.add_argument("--force", action="store_true",
                           help="submit even if the quick QC fails (adjudicator override, "
                                "e.g. an FOV-truncated rib the QC can't know about)")
            p.add_argument("--review_repo",
                           default="anonymous-mlhc/CTSpinoPelvic1K-reviews-ribs",
                           help="private ledger for the disagreement 2nd window")
            p.add_argument("--no_disagreement", action="store_true",
                           help="don't auto-open the reviewer-disagreement reference window")
            p.add_argument("--accept", action="store_true",
                           help="the --base annotator's label is right AS-IS: finalize it without "
                                "opening ITK-SNAP (still QC-gated)")
            p.add_argument("--case", default=None,
                           help="adjudicate a SPECIFIC case (work a ranked list, e.g. worst-first)")
            p.add_argument("--base", choices=["1", "2"], default="1",
                           help="which reviewer's label to load as the editable base (default 1); "
                                "the other opens read-only to compare")
            p.add_argument("--auto", action="store_true",
                           help="use the 3-way auto-merge instead of picking a reviewer (opt-in; a "
                                "merge can erode low-HU rib heads, so it's off by default)")
            p.add_argument("--review_auto", action="store_true",
                           help="with --auto: open even auto-finalizable merges to eyeball before submit")
        p.set_defaults(fn=fn)

    p = sub.add_parser("edit",
                       help="re-open an already-claimed case in ITK-SNAP and "
                            "submit it (no re-download)")
    p.add_argument("case", nargs="?", default=None,
                   help="case id of a saved claim (omit if only one is saved)")
    p.add_argument("--itksnap", default=None,
                   help="ITK-SNAP executable (auto-detected if omitted)")
    p.add_argument("--reference", action="store_true",
                   help="also open the gold reference example in a 2nd window")
    p.add_argument("--live_qc", action="store_true",
                   help="re-run QC on every Save (default off — faster)")
    p.add_argument("--force", action="store_true",
                   help="submit even if the quick QC fails (adjudicator override)")
    p.set_defaults(fn=cmd_edit)

    p = sub.add_parser("disagreement",
                       help="READ-ONLY: view where the two reviewers disagree for an "
                            "adjudication case in ITK-SNAP (nothing is written back)")
    p.add_argument("case", help="case id, e.g. 104__spine_only")
    p.add_argument("--review_repo", default="anonymous-mlhc/CTSpinoPelvic1K-reviews-ribs",
                   help="private review ledger repo (default: the rib review repo)")
    p.add_argument("--review_revision", default="main")
    p.add_argument("--ct_repo", default="anonymous-mlhc/CTSpinoPelvic1K",
                   help="dataset repo holding the CT")
    p.add_argument("--ct_revision", default=None,
                   help="default: the case's own source_revision (v4)")
    p.add_argument("--workdir", default=str(Path.home() / ".reviewtool" / "work"))
    p.add_argument("--itksnap", default=None,
                   help="ITK-SNAP executable (auto-detected if omitted)")
    p.set_defaults(fn=cmd_disagreement)

    p = sub.add_parser("fix-list",
                       help="review QC-flagged cases from a merged-QC CSV in "
                            "ITK-SNAP (local; writes a corrected tree)")
    p.add_argument("qc_csv", help="merged QC CSV (scripts/merge_qc.py output)")
    p.add_argument("--tree", required=True,
                   help="pseudo tree on disk (manifest.json + ct/ + labels/)")
    p.add_argument("--out", default=None,
                   help="reviewed-tree output dir (default: <tree>_reviewed)")
    p.add_argument("--crops", default=None,
                   help="review the small ROI crops in this dir "
                        "(from export_review_crops.py); edits paste back to full-res")
    p.add_argument("--reference", default=None,
                   help="a crop dir (ct/seg/labels) of a GOOD example to open in a "
                        "second ITK-SNAP window beside each case for comparison")
    p.add_argument("--full", action="store_true",
                   help="with --crops, ALSO open the full scan (read-only) beside "
                        "the crop to verify the rest of the volume looks fine")
    p.add_argument("--itksnap", default=None)
    p.add_argument("--all", action="store_true",
                   help="open every row, not just flagged ones")
    p.add_argument("--redo", action="store_true",
                   help="re-open cases already present in the out tree")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_fix_list)

    p = sub.add_parser("review-cases",
                       help="owner-review SPECIFIC cases (by --tokens and/or --config) "
                            "pulled from a HF revision in ITK-SNAP; save corrected "
                            "labels locally (+ optional --push). No Space.")
    p.add_argument("--repo", default="anonymous-mlhc/CTSpinoPelvic1K",
                   help="HuggingFace dataset repo id")
    p.add_argument("--revision", default="v4", help="branch/tag to review (default v4)")
    p.add_argument("--tokens", default=None,
                   help="comma-separated manifest tokens to review (e.g. 22,46,61)")
    p.add_argument("--config", default=None,
                   help="only this config (e.g. pelvic_native)")
    p.add_argument("--out", default=str(Path.home() / ".reviewtool" / "review_cases"),
                   help="output dir; corrected labels land in <out>/labels/")
    p.add_argument("--push", action="store_true",
                   help="commit the corrected labels back to --repo@--revision "
                        "(needs --token/HF_TOKEN write access)")
    p.add_argument("--token", default=None, help="HF token (or HF_TOKEN env)")
    p.add_argument("--itksnap", default=None,
                   help="ITK-SNAP executable (auto-detected if omitted)")
    p.add_argument("--check", choices=("spine", "ribs", "both", "none"), default="spine",
                   help="anatomy QC to run on each Save: 'spine' (pelvic-native pseudo-spine: "
                        "no class mixing, ascending+contiguous vertebrae), 'ribs' (rib numbering: "
                        "consecutive numbers per side and each number a single connected piece -- "
                        "flags any duplicate/split rib), 'both', or 'none'. Default spine; use "
                        "'ribs' for the v4 rib-correction task.")
    p.add_argument("--redo", action="store_true",
                   help="re-open cases already saved in --out")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_review_cases)

    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("resume"); p.set_defaults(fn=cmd_resume)

    p = sub.add_parser("reference",
                       help="open the published GOLD reference example in ITK-SNAP "
                            "(separate window, on demand)")
    p.add_argument("--workdir", default=str(Path.home() / ".reviewtool" / "work"))
    p.add_argument("--itksnap", default=None,
                   help="ITK-SNAP executable (auto-detected if omitted)")
    p.add_argument("--repo", default="gregoryschwingmdphd/CTSpinoPelvic1K",
                   help="HuggingFace dataset repo id")
    p.add_argument("--revision", default="v2", help="branch/revision (default v2)")
    p.set_defaults(fn=cmd_reference)

    p = sub.add_parser("final-pass",
                       help="local final QC pass over the flagged (finalized) "
                            "cases: open each crop CT + its corrected label in "
                            "ITK-SNAP, record approve/flag (resumable)")
    p.add_argument("--workdir",
                   default=str(Path.home() / ".reviewtool" / "final_pass"),
                   help="where crop CT + label downloads and the verdict log live")
    p.add_argument("--log", default=None,
                   help="verdict log path (default: <workdir>/verdicts.json)")
    p.add_argument("--only", choices=("all", "corrected"), default="all",
                   help="'all' = every finalized case (default); 'corrected' = "
                        "only those whose decision actually changed a label")
    p.add_argument("--redo", action="store_true",
                   help="re-review cases already in the log (default: skip them)")
    p.add_argument("--review-repo",
                   default="gregoryschwingmdphd/CTSpinoPelvic1K-reviews-triaged",
                   help="the review ledger dataset (cases/ + reviews/)")
    p.add_argument("--review-revision", default="main",
                   help="branch/tag of the review ledger (default main)")
    p.add_argument("--v2-repo", default="gregoryschwingmdphd/CTSpinoPelvic1K",
                   help="dataset repo holding the crop CTs")
    p.add_argument("--revision", default="v2",
                   help="branch/tag for the crop CTs (default v2)")
    p.add_argument("--itksnap", default=None,
                   help="ITK-SNAP executable (auto-detected if omitted)")
    p.set_defaults(fn=cmd_final_pass)

    p = sub.add_parser("download",
                       help="download the dataset from HuggingFace "
                            "(full ~253GB / ct ~241GB / labels ~2GB / crops ~10GB / sample ~11GB)")
    p.add_argument("--what", choices=list(_DL), default="crops",
                   help="how much to pull (default: crops — the review worklist, ~10GB)")
    p.add_argument("--out", default=str(Path.home() / "CTSpinoPelvic1K_data"),
                   help="destination dir")
    p.add_argument("--repo", default="gregoryschwingmdphd/CTSpinoPelvic1K",
                   help="HuggingFace dataset repo id")
    p.add_argument("--revision", default=None,
                   help="branch/tag (default: v2, or main for --what sample)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--yes", action="store_true",
                   help="confirm a large (>=50 GB) download")
    p.set_defaults(fn=cmd_download)

    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
