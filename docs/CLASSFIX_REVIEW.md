# CTSpinoPelvic1K — Fix Split Bones (class-mixing)

A small set of cases have a **bone split into two disconnected pieces** — usually a vertebra (or a hip) whose main body is fine but has a **stray blob** of the same label floating off to the side. Your job: make each bone **one clean piece**.

Same tools as the other tasks, pointed at a new Space.

---

## What you're doing

For each flagged case, ITK-SNAP will show a bone that the QC says is "split into pieces." Find the **stray piece** and fix it — one of:
- **Delete it** if it's just noise / a floating speck.
- **Relabel it** to the correct neighboring bone if it actually belongs to the vertebra next to it.
- **Connect it** to the main body if it's genuinely part of the same bone.

Result: every bone is a single connected label.

---

## The rules (strict — the QC enforces them)

1. **Only touch the STRAY piece.** Do **not** renumber or delete a bone's **main body** — the QC will reject that ("you renumbered/removed a real bone").
2. **Don't touch the ribs** — they're protected automatically.
3. **You do NOT decide anything about L6 or lumbar ribs.** If a case looks like a transitional/6th-lumbar question, that's handled separately by the radiologist — just fix the split you're shown, or **skip** the case if you can't cleanly fix it.
4. The QC checks, live on every Save: **each bone is one piece**, the **column stays in order and contiguous**, and **no main body was renumbered/deleted**.

---

## Setup (once)
```bash
hf auth login
python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-classfix.hf.space
```

## Fix a case
```bash
python -m reviewtool next        # claims a case, opens CT + label in ITK-SNAP
# -> merge/relabel/delete the STRAY piece, SAVE, watch the QC, then QUIT
python -m reviewtool resume      # submits; then `next` for the following case
```
- `python -m reviewtool skip` — if a case is unclear or looks transitional.
- Switch back to ribs/spine anytime: `reviewtool login --service <that Space's url>`.

**Thanks — this is the last cleanup pass. Almost done!**
