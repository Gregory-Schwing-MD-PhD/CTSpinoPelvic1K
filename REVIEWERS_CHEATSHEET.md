# CTSpinoPelvic1K — Reviewer Cheat Sheet
*Print this and keep it beside ITK-SNAP.*

## The loop (one case, ~20–30 min)
1. **`python -m reviewtool next`**  *(Windows: `py -m reviewtool next`)*
2. Read the **`WHY FLAGGED`** list — that's exactly what to fix.
3. Edit in ITK-SNAP → **Save (Ctrl-S / ⌘-S)** → the terminal re-runs the QC.
4. Repeat *edit → save* until **every check reads `OK`**.
5. **Quit ITK-SNAP** → it submits automatically. Run `next` for the next case.

## Reading the terminal
```
WHY FLAGGED - focus your edit here:
  * OFF-BONE LEAK  (off_bone=0.072; target <= 0.058)   <- fix this
```
After each **Save**:
```
  off-bone leak   0.072 -> 0.008   OK        (draft -> your edit)
OK - all automated checks pass on your edit.  <- DONE: quit ITK-SNAP
```
`STILL HIGH` / `STILL FLAGGED` = keep going. (Occasionally a true case can't fully
clear — if you're sure it's right, quit anyway; the second reviewer/adjudicator
settles it.)

## Flag → fix (in ITK-SNAP)
| Terminal says | What to do |
|---|---|
| **OFF-BONE LEAK** | Erase label that's off the bone (set label = **Clear Label**, paint). |
| **vertebra MIXING** | Relabel stray voxels to the correct level (count up from the sacrum). |
| **DUPLICATED structure** | Find the stray piece (3D view) and erase it. |
| **L/R HIP SWAP** | Don't guess — flag it for the project lead. |
| **MISSING level** | Paint it in — or flag it if it's cut off the scan. |
| **"the WHOLE scan"** | Fused gold case: edit **both** regions (no one-region limit). |

## Rules
- **Edit only the region named** (spine *or* pelvis) — unless it says "the WHOLE scan".
- **Don't renumber/recolor:** `L1–L6 = 1–6, sacrum = 7, left hip = 8, right hip = 9`.
- **Reading the CT:** bright white = bone (label edge sits here) · gray *inside*
  the white shell = marrow (keep it) · gray outside = soft tissue (don't label) ·
  black = air (never label).
- **Unsure?** (weird/abnormal anatomy, ambiguous level) → smallest safe edit, or
  none, and tell the lead. A careful "I left this because X" beats a wrong fix.

## Helpers
- A **gold reference example** opens in a second window — tile the two windows to compare.
- Full guide + 3 short video tutorials: **[REVIEWERS_FIXING.md](REVIEWERS_FIXING.md)**.
- Saved but upload failed (crash / rate-limit)? `reviewtool resume`. Progress: `reviewtool status`.

## Credit
- **~210 cases total. A case is *done* when the QC reads `OK`.**
- **Authorship:** **Gregory Schwing — first author**; **Loren Schwiebert — senior
  (last) author**. **Complete at least 10 QC-passing cases to qualify for
  co-authorship**, then you earn 2nd, 3rd, 4th … position by volume — the more
  cases you complete, the higher you rank. Your HuggingFace username is recorded
  with every case, so it's tracked and fair.
