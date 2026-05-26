# reviewtool (Phase 3)

> **Reviewers:** the full step-by-step onboarding (HF account + token,
> installing ITK-SNAP, connecting, doing reviews) is in
> **[docs/REVIEW.md](../docs/REVIEW.md)**. This file is the terse reference.

Local CLI an annotator runs to review v2 pseudo labels in ITK-SNAP. All HF
download/upload is hidden — CT + pseudo come from the v2 repo (via
`huggingface_hub`), and the corrected label is uploaded **through the review
Space** (which holds the dataset write token). The annotator authenticates with
their **own HuggingFace login** (`hf auth login`); the Space verifies the
username via `whoami`. (A legacy `--key` is still accepted if configured.)

## Install
```bash
pip install requests huggingface_hub numpy nibabel    # + ITK-SNAP on PATH
```

## Use
```bash
hf auth login              # sign in with your own HF Read token
reviewtool login --service https://<your-space>.hf.space

reviewtool next            # claim a case → opens ITK-SNAP → on quit, diffs + submits
#   In ITK-SNAP: correct ONLY the indicated region, "Save Segmentation"
#   over the seg.nii.gz it opened, then quit.

reviewtool status          # progress dashboard (JSON)
reviewtool resume          # re-send any edit whose upload was interrupted

# adjudicators only:
reviewtool adjudicate --notes "took reviewer A's sacrum boundary"
```

`reviewtool` auto-detects ITK-SNAP (PATH / standard install dir / `$REVIEWTOOL_ITKSNAP`).
Flags: `--workdir DIR` (scratch, default `~/.reviewtool/work`),
`--itksnap /path/to/itksnap` for a nonstandard install.

Notes
- The palette is locked (`labels.txt` written from the service's descriptor)
  so label values stay consistent across reviewers — don't renumber labels.
- `next` decides **accept** vs **corrected** automatically from whether you
  changed any voxels; **reject** (unusable scan) is an adjudicator action.
- Edit only the region the prompt names (the pseudo-filled side); the manual
  region is gold and must not be touched.
