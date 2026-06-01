# How to fix a segmentation in ITK-SNAP

This is the "what do I actually *do* in ITK-SNAP" guide. Read it once; keep it
open while you review your first few cases.

Every case you get has already been **flagged by automated quality checks** —
you're not hunting blind. When the tool opens a case, the terminal prints a
**`WHY FLAGGED:`** line telling you the *suspected* problem (a leak, a mixed-up
vertebra, a stray piece, a left/right swap). Your job is to look at that, decide
if it's real, and fix it. **Most fixes are small** (erasing a few stray voxels);
a minority need real re-drawing.

---

## 0. The two things to remember

1. **Only edit the region named in the terminal** — `spine` (L1–L6) **or**
   `pelvis` (sacrum + both hips). The other region is expert manual annotation —
   don't touch it.
2. **Never renumber or recolor labels.** The palette is locked:
   `L1=1, L2=2, L3=3, L4=4, L5=5, L6=6, sacrum=7, left hip=8, right hip=9`.
   Always paint with the *correct existing label* from the list on the left.

---

## 1. Find your way around ITK-SNAP (60-second tour)

- **Four panels**: three CT slice views (axial = top-down, sagittal = side,
  coronal = front) and a **3D** view (bottom-right). Click anywhere to move the
  crosshair; all views jump to that point.
- **Scroll through slices**: mouse-wheel over a panel (or arrow keys). This is
  how you move *through* the body.
- **Zoom / pan**: right-drag to zoom, middle-drag (or scroll-click) to pan.
- **The label list** (left side): the colors are L1–L6, sacrum, hips. The
  highlighted one is the **active label** — what your brush will paint.
- **3D view**: click **Update** (or the mesh icon) to render the whole
  structure in 3D — the fastest way to spot a stray piece or a missing chunk.

### The three tools you'll use
- **Paintbrush** (brush icon): paints the active label. Adjust size in the tool
  panel; round brush is fine.
- **Eraser**: set the active label to **"Clear Label"** (value 0) at the top of
  the label list, then paint — that *removes* label (this is how you fix leaks).
- **Polygon** (lasso icon): click around an outline, then **Accept** to fill it
  with the active label — good for re-drawing a bigger area in one slice.

### Reading the CT (this is how you know what's bone)
Bone segmentation is about following the CT brightness:
- **Bright white** = dense **cortical bone** (the outer shell). The label edge
  should sit right at this bright rim.
- **Speckled gray inside the white shell** = **marrow** — this *is* part of the
  bone, keep it labeled.
- **Mid-gray** outside the bone = **muscle / soft tissue** — should **not** be
  labeled.
- **Black** = **air / lung / outside the body** — should **never** be labeled.

> Rule of thumb: if a colored voxel sits on gray or black (not on/inside the
> white bone), it's wrong — erase it.

---

## 2. Identify the vertebra level (the genuinely hard part)

When a flag says a vertebra is the *wrong level* (e.g. an L3 that should be L4),
you have to know which level it is. Radiologists **count from a fixed landmark**:

1. Go to the **sagittal** view (side-on) — you'll see the spine as a stack.
2. Find the **sacrum** (big triangular bone at the bottom, label 7). The
   vertebra sitting directly on it is **L5** (or **L6** if this patient has six
   lumbar vertebrae).
3. Count **upward**: L5 → L4 → L3 → L2 → L1.
4. If a reference example is open beside your case, compare the pattern.

**Honest caveat:** some people have transitional anatomy (an extra/partial
bottom vertebra — "LSTV": a 6th lumbar, a sacralized L5, a lumbarized S1).
Counting is then genuinely ambiguous *even for experts*. **If you're not sure of
the level, don't guess** — make no level change, note it, and let the second
reviewer / adjudicator settle it. Two-reviewer agreement is there exactly for
these.

---

## 3. Fix-by-flag recipes

Match the terminal's `WHY FLAGGED:` to the recipe.

### "OFF-BONE leak" — the most common, the easiest
The label spills past the cortex into soft tissue or air.
1. Set the active label to **Clear Label** (eraser).
2. Scroll through the flagged structure slice by slice.
3. Wherever the color extends **off the white bone** onto gray/black, **erase
   it** back to the bright cortical edge.
4. Don't over-erase — keep the marrow (gray *inside* the white shell).

### "vertebra MIXING" — voxels of one level inside another
A vertebra has stray pieces labeled as it, sitting in/around a *neighbor*.
1. Use the **3D view** + scroll to find the out-of-place voxels (e.g. L4-colored
   voxels down on the L5 body).
2. Decide the **correct** label for those voxels (Section 2 — count levels).
3. Set the active label to that correct level and **paint over** the wrong-color
   voxels (painting replaces the old label). Erase any that are off-bone.

### "structure: duplicated" — a stray disconnected blob
A second, separate chunk carries the same label (often a speck far from the main
structure).
1. **3D view** makes the stray piece obvious (a floating island).
2. Click onto it, switch to the slice views, set active label to **Clear
   Label**, and **erase the whole stray piece**.
3. If the "stray" piece is actually real bone that got the *wrong* label, relabel
   it instead (paint with the correct one).

### "structure: L/R HIP SWAP" — left/right reversed
`left hip` (8) and `right hip` (9) are on the wrong sides.
- This is a relabel of two large structures and is easy to get wrong. **Flag it
  for the project lead / adjudicator** rather than swapping by hand, unless
  you're confident — note which case it was.

### "missing level" / "pelvis incomplete" — something absent
A vertebra or a hip/sacrum is missing.
1. Confirm in the CT that the bone **is actually there** (sometimes it's at the
   edge of the scan and genuinely cut off).
2. If present but unlabeled: paint it in with the correct label (polygon tool is
   fastest, slice by slice).
3. If genuinely **not in the scan** (cut off): leave it and tell the lead — don't
   invent anatomy.

---

## 4. The whole-case workflow

1. Run `reviewtool next` — it opens the case and prints `WHY FLAGGED:`.
2. **(Optional) open the whole scan for context:** `reviewtool next --full`
   opens the full CT read-only in a second window, so you can confirm the rest of
   the scan looks fine beyond the cropped region.
3. Scroll to the flagged area; apply the recipe above.
4. **Check your work in 3D** (Update the mesh) — the structure should look like
   one clean bone with smooth borders.
5. **Segmentation → Save Segmentation Image**, saving **over the file it opened**
   (`seg.nii.gz` / `crop_edit.nii.gz`), then **quit ITK-SNAP**. The tool uploads
   automatically.
6. **Nothing actually wrong?** The flag can be a false alarm. If the draft is
   already correct, just save and quit — that records a valid "accept".

---

## 5. When you're unsure — the honest rule

You will hit cases you can't confidently fix (ambiguous level, weird/abnormal
anatomy, a swap you're not sure about). **Do not guess aggressively.** Make the
smallest correct edit you're confident in, or none, and note the case for the
lead. The two-reviewer + adjudication design exists to catch and resolve exactly
these — a careful "I left this because X" is far more useful than a confident
wrong relabel.

> Optional speed-up for heavy re-drawing: ITK-SNAP's **nnInteractive** AI tool
> (free, runs on Google Colab) can rough-in a structure you then clean up. See
> [REVIEWERS_AI_EDITING.md](REVIEWERS_AI_EDITING.md).
