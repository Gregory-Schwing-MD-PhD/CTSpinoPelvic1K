"""Unit test for reviewtool's pure decision logic (build_submission).
The HTTP/ITK-SNAP glue isn't unit-tested (subprocess + network); the
accept-vs-corrected decision + record assembly is."""
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
for p in (_ROOT / "scripts", _ROOT / "reviewtool"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import cli  # noqa: E402  (reviewtool/cli.py)


def _vol():
    a = np.zeros((6, 6, 6), dtype=np.int16)
    a[0:4, 0, 0] = 5            # manual spine
    a[0:3, 1, 0] = 7            # pseudo-filled sacrum
    return a


def test_no_edit_is_accept():
    pseudo = _vol()
    decision, rec = cli.build_submission(pseudo, pseudo.copy(), "pelvis", "sha")
    assert decision == "accept"
    assert rec["diff"]["n_voxels_changed"] == 0
    assert rec["region_reviewed"] == "pelvis"
    assert rec["source_label_sha256"] == "sha"


def test_edit_is_corrected_with_diff():
    pseudo = _vol()
    edited = pseudo.copy()
    edited[0:5, 1, 0] = 7        # grew sacrum by 2 voxels
    decision, rec = cli.build_submission(pseudo, edited, "pelvis", "sha")
    assert decision == "corrected"
    assert rec["diff"]["n_voxels_changed"] == 2
    assert rec["diff"]["regions_touched"] == ["pelvis"]


def test_default_itksnap_honors_valid_env(tmp_path, monkeypatch):
    fake = tmp_path / "itksnap"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)
    monkeypatch.setenv("REVIEWTOOL_ITKSNAP", str(fake))
    assert cli._default_itksnap() == str(fake)


def test_default_itksnap_ignores_broken_env(monkeypatch):
    # A stale env override pointing at a non-existent file must NOT be returned
    # (that's what was handing subprocess a bad path); detection falls through.
    broken = "/usr/bin/itksnap-stale-0.0.0/bin/itksnap"
    monkeypatch.setenv("REVIEWTOOL_ITKSNAP", broken)
    assert cli._default_itksnap() != broken
