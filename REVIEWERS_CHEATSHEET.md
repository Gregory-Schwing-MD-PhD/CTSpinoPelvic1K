# CTSpinoPelvic1K — Reviewer Cheat Sheet
*Print this and keep it beside ITK-SNAP.*

## The loop (one case, ~20–30 min)
1. **`python -m reviewtool next`**  *(Windows: `py -m reviewtool next`)*
2. Read the **`WHY FLAGGED`** list — that's exactly what to fix (it names the
   split levels).
3. Edit in ITK-SNAP → **Save (Ctrl-S / ⌘-S)** → **Quit** → it submits. Run
   `next` for the next case.

**Options** (add to `next`): `--reference` = open a gold example beside it ·
`--live_qc` = re-run the checks on every Save and watch them clear to **OK**.
Standalone gold example: `reviewtool reference`.

## Reading the terminal
```
LSTV STATUS: Normal (L1-L5 + sacrum)
WHY FLAGGED - focus your edit here:
  * vertebra MIXING  (off_main=0.088; target <= 0.005)  [split: L3(63/37), L4(70/31)]
  * DUPLICATED structure  [L3,L4]
```
The `[split: …]` / `[L3,L4]` tags tell you **which levels** to fix. With
`--live_qc`, each Save reprints them clearing toward `OK`.

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
- Want a **gold reference example** beside your case? Add `--reference` to `next`
  (or run `reviewtool reference`), then tile the two windows to compare.
- Full guide + 3 short video tutorials: **[REVIEWERS_FIXING.md](REVIEWERS_FIXING.md)**.
- Saved but upload failed (crash / rate-limit)? `reviewtool resume`. Progress: `reviewtool status`.

## Credit
- **~210 cases total. A case is *done* when the QC reads `OK`.**
- **Authorship:** **Gregory Schwing — first author**; **Loren Schwiebert — senior
  (last) author**. **Complete at least 10 QC-passing cases to qualify for
  co-authorship**, then you earn 2nd, 3rd, 4th … position by volume — the more
  cases you complete, the higher you rank. Your HuggingFace username is recorded
  with every case, so it's tracked and fair.
