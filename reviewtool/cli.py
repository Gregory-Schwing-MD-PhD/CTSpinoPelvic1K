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
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, Tuple

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))
from review import diff, labels_descriptor  # noqa: E402
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


def _launch_itksnap_bg(itksnap: str, ct: Path, seg: Path, labels: Path):
    """Open ITK-SNAP NON-blocking (for a persistent reference window). Returns a
    Popen handle (or None if it couldn't launch)."""
    try:
        return subprocess.Popen([itksnap, "-g", str(ct), "-s", str(seg),
                                 "-l", str(labels)])
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
                               "-l", str(labels)], check=False)
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


def _claim(s, base, path="/claim"):
    r = s.post(base + path, timeout=60)
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


def _watch_itksnap(itksnap, ct, seg, labels, *, qc_ct, before) -> int:
    """Open ITK-SNAP NON-blocking and recompute the QC every time `seg` is saved,
    so the reviewer sees progress live without closing. Returns ITK-SNAP's exit
    code when they finally quit."""
    print(f"\nOpening ITK-SNAP ({itksnap}) — WATCH mode.\n"
          f"  Edit, then **Save Segmentation** (Ctrl-S) any time to see a live QC\n"
          f"  update here. Quit ITK-SNAP when you're done — it submits then.\n")
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
            print("  [saved] recomputing QC ...")
            _qc_feedback(qc_ct, before, seg)
    return proc.returncode if proc.returncode is not None else 0


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
        foci.append(f"vertebra MIXING  (off_main={m['off_main_frac']:.3f}; target <= 0.005)")
    if m.get("n_order_inversions", 0):
        foci.append("level ORDER wrong (a vertebra is out of sequence)")
    if m.get("duplication_flag", 0):
        foci.append("DUPLICATED structure (a stray disconnected piece)")
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
    rlbl.write_text(job.get("labels_descriptor")
                    or labels_descriptor.descriptor_text())
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


def _print_lstv(crop: dict) -> None:
    """Print the case's LSTV phenotype so the reviewer counts levels carefully.
    Only when the seed carried it (post-reseed); silent on older seeds."""
    if not crop or "lstv_class" not in crop:
        return
    c = int(crop.get("lstv_class") or 0)
    name = _LSTV_NAMES.get(c) or (crop.get("lstv_label") or "?")
    print(f"  LSTV STATUS: {name}")
    if c in _LSTV_HINT:
        print(f"    -> {_LSTV_HINT[c]}")


def cmd_next(a):
    s, base = _api()
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
    pseudo = _fetch(job, job["label_file"], work / "pseudo.nii.gz")
    seg = work / "seg.nii.gz"
    labels = work / "labels.txt"
    labels.write_text(job.get("labels_descriptor")
                      or labels_descriptor.descriptor_text())
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

    region_note = ("the WHOLE scan (radiologist gold being re-checked)"
                   if job.get("region_to_review") == "both"
                   else f"the {job['region_to_review']} region")
    print(f"case {job['case_id']}  ({note} — {region_note})")
    _print_lstv(crop)
    before_qc = (work / "crop_seg.nii.gz") if crop else pseudo
    if not getattr(a, "no_qc", False):                  # WHY FLAGGED / what to fix
        # Run async so ITK-SNAP opens immediately — on a fused WHOLE-scan crop the
        # QC still takes a bit, and we don't want a blank terminal while it runs.
        import threading
        threading.Thread(target=_qc_startup, args=(snap_ct, before_qc),
                         daemon=True).start()

    ref_proc = None
    if not getattr(a, "no_reference", False):           # gold example beside the case
        ref_proc = _open_space_reference(job, snap, work)

    # mtime of the seg the editor opens, BEFORE the session. A real save (an
    # edit OR a deliberate accept) advances it; if it never advances, ITK-SNAP
    # detached/closed before the reviewer saved (the macOS bin/ wrapper bug) and
    # we must NOT submit a phantom voxels_changed=0 'accept'.
    pre_mtime = snap_seg.stat().st_mtime if snap_seg.exists() else 0.0
    if getattr(a, "no_watch", False):                   # watch is the default
        rc = _launch_itksnap(snap, snap_ct, snap_seg, labels)
    else:
        rc = _watch_itksnap(snap, snap_ct, snap_seg, labels,
                            qc_ct=snap_ct, before=before_qc)
    for p in (ctx_proc, ref_proc):
        if p is not None:
            p.terminate()
    if rc != 0:
        print(f"\nITK-SNAP exited with code {rc} — it failed to start or "
              f"crashed, so NO edit was captured. Not submitting.\n"
              f"Your claim is saved (nothing sent to the server). Fix ITK-SNAP, "
              f"then re-open this case with:\n"
              f"  python -m reviewtool edit {job['case_id']}")
        _itksnap_failure_hint(rc)
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

    # watch (default) already printed QC live on each save; in --no_watch show once.
    if not getattr(a, "no_qc", False) and getattr(a, "no_watch", False):
        _qc_feedback(snap_ct, before_qc, snap_seg)

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
    job = _claim(s, base, "/adjudication/next")
    if job is None:
        print("nothing to adjudicate.")
        return
    work = Path(a.workdir) / (job["case_id"] + "__adj")
    work.mkdir(parents=True, exist_ok=True)
    _save_active(job, work, kind="adjudicate", notes=a.notes)
    ct = _fetch(job, job["ct_file"], work / "ct.nii.gz")
    pseudo = _fetch(job, job["label_file"], work / "pseudo.nii.gz")
    seg = work / "seg.nii.gz"
    seg.write_bytes(pseudo.read_bytes())
    labels = work / "labels.txt"
    labels.write_text(job.get("labels_descriptor")
                      or labels_descriptor.descriptor_text())
    print(f"ADJUDICATE {job['case_id']}  IRR={job.get('irr')}\n"
          f"two reviewers disagreed on the {job['region_to_review']} region; "
          f"produce the deciding label.")
    rc = _launch_itksnap(a.itksnap or _default_itksnap(), ct, seg, labels)
    if rc != 0:
        print(f"\nITK-SNAP exited with code {rc} — no deciding label captured. "
              f"Not submitting. Your claim is saved; fix ITK-SNAP and re-open "
              f"with:\n  python -m reviewtool edit {job['case_id']}__adj")
        _itksnap_failure_hint(rc)
        return
    _submit_adjudication(s, base, job, work, seg, a.notes)


def cmd_status(a):
    s, base = _api()
    r = s.get(base + "/status", timeout=60)
    r.raise_for_status()
    print(json.dumps(r.json(), indent=2))


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
    if not labels.exists():
        labels.write_text(job.get("labels_descriptor")
                          or labels_descriptor.descriptor_text())
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
    pre_mtime = snap_seg.stat().st_mtime if snap_seg.exists() else 0.0
    rc = _launch_itksnap(a.itksnap or _default_itksnap(), snap_ct, snap_seg, labels)
    if rc != 0:
        print(f"ITK-SNAP exited with code {rc} — no edit captured. "
              "Not submitting; your claim is kept.")
        _itksnap_failure_hint(rc)
        return
    if not (snap_seg.exists() and snap_seg.stat().st_mtime > pre_mtime + 1e-6):
        print("ITK-SNAP closed but you never SAVED — no edit captured, nothing "
              "submitted; your claim is kept.\n"
              "  Edit, then Save Segmentation (Ctrl-S / Cmd-S) before quitting "
              "(one save also 'accepts' a correct draft)."
              + (" If it auto-closed, relaunch with --itksnap "
                 "/Applications/ITK-SNAP.app/Contents/MacOS/ITK-SNAP"
                 if sys.platform == "darwin" else ""))
        return

    if crop:                                      # fold the crop edit into full-res
        _paste_edit_to_full(snap_seg, crop["origin"], pseudo, seg)

    if kind == "adjudicate":
        _submit_adjudication(s, base, job, work, seg, st.get("notes", ""))
    else:
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="reviewtool", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("login"); p.add_argument("--service", required=True)
    p.add_argument("--key", default=None,
                   help="(optional) legacy reviewer key; omit to use your "
                        "HuggingFace login")
    p.set_defaults(fn=cmd_login)

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
            p.add_argument("--no_reference", action="store_true",
                           help="don't open the gold reference example window")
        if name == "adjudicate":
            p.add_argument("--notes", default="")
        p.set_defaults(fn=fn)

    p = sub.add_parser("edit",
                       help="re-open an already-claimed case in ITK-SNAP and "
                            "submit it (no re-download)")
    p.add_argument("case", nargs="?", default=None,
                   help="case id of a saved claim (omit if only one is saved)")
    p.add_argument("--itksnap", default=None,
                   help="ITK-SNAP executable (auto-detected if omitted)")
    p.set_defaults(fn=cmd_edit)

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

    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("resume"); p.set_defaults(fn=cmd_resume)

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
