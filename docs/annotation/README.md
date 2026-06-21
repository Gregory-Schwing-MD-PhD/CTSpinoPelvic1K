# CTSpinoPelvic1K v4 — Annotation tasks (med-student guides)

Three v4 overlay tasks add new structures onto the **good v3 labels**. Each runs as
its **own HuggingFace Space + private ledger**, all reading
`anonymous-mlhc/CTSpinoPelvic1K@v3`. You never edit the existing v3 structures —
you **add** your task's overlay onto the label that's served to you.

| Task | Guide | `TASK` | Space | Private ledger (`REVIEW_REPO`) | Paints ids |
|---|---|---|---|---|---|
| **Ribs** | [ribs.md](ribs.md) | `ribs` | `anonymous-mlhc/CTSpinoPelvic1K-review-ribs` | `anonymous-mlhc/CTSpinoPelvic1K-reviews-ribs` | **26–49** (reuses v3 reserved) |
| **LS nerves** | [ls_nerves.md](ls_nerves.md) | `ls_nerve` | `…-review-nerve` | `…-reviews-nerve` | **53–58** |
| **Iliolumbar** | [iliolumbar.md](iliolumbar.md) | `iliolumbar` | `…-review-ili` | `…-reviews-ili` | **51/52** |

> The earlier `lstv` (v2 pseudo-correction) and `rib_anchor` Spaces are separate and
> unchanged. "What is what": the **public dataset** `…/CTSpinoPelvic1K` holds the CTs
> + labels (branches v1/v2/v3); each **private `-reviews-*` ledger** holds that one
> task's claims/records/labels/`finals.json`; each **`-review-*` Space** is the
> service that hands cases out and collects them. One task = one Space = one ledger.

---

## Label space (read this once)

You paint into the **dataset id scheme** so your work drops straight in:

- Existing v3 structures (do **not** touch): 1–6 L1–L6, 7 S1, 8 sacrum, 9/10 hips,
  11/12 femurs, 13–25 T1–T13, 50 ignore. These appear as **grey context** in your
  ITK-SNAP palette.
- Your task's overlay ids are the only ones you paint (see each guide).
- The palette is locked per task — launch ITK-SNAP with the task's label file:
  ```bash
  python scripts/review/labels_descriptor.py --task <ribs|ls_nerve|iliolumbar> --out labels.txt
  ```
  **Never renumber or recolour labels** — if your "rib_left_11" isn't everyone's id
  48, the IRR/merge is silently wrong.

---

## One-time setup (any task)

```bash
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K
pip install requests huggingface_hub numpy nibabel
hf auth login          # paste your HuggingFace token (free account)
```
ITK-SNAP must be installed (`reviewtool` auto-detects it).

## Reviewing (same shape for every task)

Each task guide has the exact copy-paste block with **its** Space URL. The shape:

```bash
python -m reviewtool login --service <YOUR TASK'S SPACE URL>
python -m reviewtool next      # claims a case + opens ITK-SNAP; annotate, save & close to submit
python -m reviewtool next      # ...repeat for each case
python -m reviewtool status    # your progress
```
`next` auto-assigns the next case (double-review enforces A≠B; no cherry-picking).
If a session drops mid-edit, `python -m reviewtool resume` re-sends it. Use
**AI-assist** (nnInteractive scribbles) to start each structure, then correct by hand.

**Double review + IRR:** every case is independently annotated by **two** students
(slots 1 and 2; you can't hold both). Agreement = per-class Dice; if it clears the
threshold τ the case auto-finalizes, otherwise it goes to an **adjudicator**
(faculty) who decides the final label. This is your inter-rater reliability data.

## Reference standard & quality

- An **expert adjudicator** (spine/neuro faculty) resolves disagreements and spot-QCs.
- Where a case has paired **MRI** (esp. for nerves), validate against it.
- Flag anything ambiguous in `reviewtool` notes rather than guessing — a flagged
  case is more useful than a confidently-wrong label.

## Deploying the three Spaces (maintainer)

Each task is one Docker Space (same `review_service/` code, different env). Create
three, set env per task, keep **1 replica** (the write-lock assumes a single worker):

```bash
for t in ribs nerve ili; do
  hf repos create anonymous-mlhc/CTSpinoPelvic1K-review-$t --type space --space-sdk docker
done
# push Dockerfile + review_service/*.py + scripts/review/*.py to each Space
```

Per-Space **Variables & secrets** (all read the public dataset at v3):

| var | ribs | nerve | iliolumbar |
|---|---|---|---|
| `TASK` | `ribs` | `ls_nerve` | `iliolumbar` |
| `V2_REPO` | `anonymous-mlhc/CTSpinoPelvic1K` | (same) | (same) |
| `SOURCE_REVISION` | `v3` | `v3` | `v3` |
| `REVIEW_REPO` | `…/CTSpinoPelvic1K-reviews-ribs` | `…-reviews-nerve` | `…-reviews-ili` |
| `HF_TOKEN` *(secret)* | write token | write token | write token |
| `ADJUDICATORS` | faculty HF usernames | … | … |
| `TAU` / `IRR_MODE` | `0.9` / `per_class_min` | (same) | (same) |

On boot each Space pulls `manifest.json` from `@v3` and seeds spine-GT cases via
`init_overlay_cases(task=…)` (idempotent). The private `-reviews-*` ledgers are
auto-created on first write.

## Credit & code

Per the OpenSpineToolbox policy, any code you write (AI-assist scripts, QC, analysis)
must be PR'd into <https://github.com/Gregory-Schwing-MD-PhD/OpenSpineToolbox> to
count toward your contribution.
