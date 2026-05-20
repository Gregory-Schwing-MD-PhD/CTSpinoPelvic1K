"""reviewtool — local CLI for CTSpinoPelvic1K pseudo-label review (Phase 3).

Wraps ITK-SNAP and the review Space so an annotator just runs
`reviewtool next`: it claims a case, fetches the CT + pseudo label from the
public v2 repo, opens ITK-SNAP with the locked 10-class palette, then on
exit diffs the edit, builds the review record, and uploads through the
Space (which holds the HF token — the annotator never does).
"""
