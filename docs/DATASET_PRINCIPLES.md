# CTSpinoPelvic1K — dataset principles (the governing law)

The dataset is an **append-only, lossless** artifact. It only ever **grows**:
each version adds structures (pelves → counting anchor → ribs → thoracic levels
→ …). We never strip a mask out, and above all we never destroy **radiologist
ground truth** — the one component that cannot be regenerated. Pseudolabels can
be recomputed, splits redone, schemes re-derived; a radiologist's read cannot.
Treat it as **write-once**.

## The invariants

1. **Radiologist ground truth is canonical and immutable.** It is never
   overwritten, never deleted, never replaced by a model. A correction to GT is
   only valid with radiologist sign-off, and it is recorded as a new read, not a
   silent edit.

2. **Additive, never destructive.** A new version is a **superset** of the
   labels in the previous one. If a structure was labelled once, it stays
   labelled. The roadmap is purely additive: pelvis (done) → last rib-bearing
   vertebra / counting anchor (done) → rib (reserved, class 12) → all available
   thoracic levels → …

3. **Class IDs are reserved forward, never renumbered.** Once an ID means a
   thing it means that thing forever (e.g. `11 = last_rib_vertebra`,
   `12 = rib`). New structures take new IDs. This is what makes every version a
   drop-in superset.

4. **The master is lossless; releases are filtered *views*.** The master
   retains every mask and the native source labelling. A named release (v1, v2,
   …) may *filter* the master for a purpose — e.g. v2 excludes `pelvic_native`
   from the training view because shipping it would require a pseudolabelled
   spine — but filtering a release **does not delete the underlying GT**. The
   `pelvic_native` pelvis read still exists in the master and is reused (here,
   as the pelvis-pseudolabel validation set). "We don't ship it in v2" is never
   "we threw it away."

5. **Pseudolabels never touch a GT region.** Model completion fills only the
   region a human did not annotate (and only background/ignore voxels); a manual
   voxel is never altered. The fidelity-critical structures stay human.

6. **The training scheme is a *derived view*, not the artifact.** The dataset
   keeps the full, native label richness (e.g. all available thoracic levels in
   their VerSe numbering); the merged/contiguous scheme a particular model
   trains on is produced by a documented conversion script. Reducing for
   training must never reach back and reduce the artifact. (Keeping only T12 in
   an early export was this rule being violated — the reduction leaked into the
   artifact; retaining all thoracic levels restores it.)

## Why this matters

A dataset that only grows is one you can build on forever without fear: every
downstream model, split, and paper that used version *n* remains valid against
version *n+1*, because *n+1* added to *n* rather than changing it. The moment you
allow destructive edits, every prior result is silently at risk. Append-only is
what lets a benchmark accumulate value instead of churning it.

See also [LABELING_GUIDE.md](LABELING_GUIDE.md) and
[RIB_ANCHOR_RATIONALE.md](RIB_ANCHOR_RATIONALE.md).
