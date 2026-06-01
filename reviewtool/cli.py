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

    for name in ("itksnap", "ITK-SNAP"):
        hit = _runnable(name)
        if hit:
            return hit
    candidates: list = []
    if sys.platform == "darwin":
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
    if crop:
        snap_ct = _fetch(job, crop["ct_crop"], work / "ct.nii.gz")    # crop CT (few MB)
        snap_seg = work / "crop_edit.nii.gz"
        snap_seg.write_bytes(_fetch(job, crop["seg_crop"],
                                    work / "crop_seg.nii.gz").read_bytes())
        note = "crop review"
    else:
        snap_ct = _fetch(job, job["ct_file"], work / "ct.nii.gz")     # full CT
        snap_seg = seg
        seg.write_bytes(pseudo.read_bytes())                          # edit a copy
        note = "review"

    print(f"case {job['case_id']}  ({note} — {job['region_to_review']} region)")
    rc = _launch_itksnap(a.itksnap or _default_itksnap(), snap_ct, snap_seg, labels)
    if rc != 0:
        print(f"\nITK-SNAP exited with code {rc} — it failed to start or "
              f"crashed, so NO edit was captured. Not submitting.\n"
              f"Your claim is saved (nothing sent to the server). Fix ITK-SNAP, "
              f"then re-open this case with:\n"
              f"  python -m reviewtool edit {job['case_id']}")
        _itksnap_failure_hint(rc)
        return
    if crop:                                            # fold the crop edit into full-res
        _paste_edit_to_full(snap_seg, crop["origin"], pseudo, seg)

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
    ct, seg, labels = work / "ct.nii.gz", work / "seg.nii.gz", work / "labels.txt"
    if not ct.exists() or not seg.exists():
        print(f"workdir incomplete ({work}): missing ct/seg — re-claim with "
              "`reviewtool next`.")
        return
    if not labels.exists():
        labels.write_text(job.get("labels_descriptor")
                          or labels_descriptor.descriptor_text())

    print(f"editing {work.name} ({kind}) — review the "
          f"{job['region_to_review']} region")
    rc = _launch_itksnap(a.itksnap or _default_itksnap(), ct, seg, labels)
    if rc != 0:
        print(f"ITK-SNAP exited with code {rc} — no edit captured. "
              "Not submitting; your claim is kept.")
        _itksnap_failure_hint(rc)
        return

    if kind == "adjudicate":
        _submit_adjudication(s, base, job, work, seg, st.get("notes", ""))
    else:
        pseudo = work / "pseudo.nii.gz"
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
            rc = _launch_itksnap(itksnap, cdir / "ct.nii.gz", crop_seg, cdir / "labels.txt")
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
    p.add_argument("--itksnap", default=None)
    p.add_argument("--all", action="store_true",
                   help="open every row, not just flagged ones")
    p.add_argument("--redo", action="store_true",
                   help="re-open cases already present in the out tree")
    p.add_argument("--limit", type=int, default=0)
    p.set_defaults(fn=cmd_fix_list)

    p = sub.add_parser("status"); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("resume"); p.set_defaults(fn=cmd_resume)

    args = ap.parse_args(argv)
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
