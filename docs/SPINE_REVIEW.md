# CTSpinoPelvic1K — Spine Review

One Space for all spine corrections. For each case, follow this order:

## The procedure (every case)
1. **Check for class-mixing / a duplicate.** Is a vertebra split into two disconnected pieces, or does one mask cover **two vertebral bodies** (a duplicated level)?
2. **Find the LAST FULL RIB.** The vertebra it attaches to is your **T12 anchor** (a *full* rib — long, curving — not a little stump). Everything is numbered from here: below T12 → L1, L2, L3…
3. **If a lumbar vertebra is duplicated:** the count is short one — **add an L6 caudally** and renumber so the lumbar run is L1→L6, anchored to T12.
4. **If not:** just **fix the class-mixing** — merge/relabel the stray piece so each bone is one clean mask. Don't renumber.
5. **Always:** **segment every thoracic vertebra in the field of view**, numbering **upward** from T12.

## Rules
- **Stay in the spine.** Only correct vertebrae/sacrum/pelvis here. If you spot a **rib** problem, don't fix it here — `python -m reviewtool flag "reason"`.
- **Ribs are protected** automatically; you can only edit ids for the spine/pelvis.
- **When a transitional level is unclear, `flag` it** for the radiologist instead of guessing.
- The QC on Save checks: every bone one piece, ascending/contiguous, **last full rib on T12**, and **no vertebra covering two bodies** — it won't let a bad count through.

## Commands
```bash
hf auth login
python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-spine.hf.space
python -m reviewtool next          # edit in ITK-SNAP, Save, quit
python -m reviewtool resume        # submit
python -m reviewtool flag "possible L6"   # send a transitional case to the radiologist
```
