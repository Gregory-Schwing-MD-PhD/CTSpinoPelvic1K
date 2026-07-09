# CTSpinoPelvic1K — v4 Rib Numbering Correction Guide

A short, AI-assisted review task: fix the handful of ribs in **v4** where the
automatic pipeline left a rib **number split across two pieces**. The rib
*levels* are already correct (the dataset has **no missing rib numbers**) — you
are only resolving **duplicates**: one rib number that shows up as two separate
blobs. Most cases take under a minute with nnInteractive.

> **The QC gate:** your edit is accepted only if its ribs pass QC. When you submit, the
> service checks the ribs **server-side** and **rejects** any case that still has a rib
> number in two pieces — so nothing un-QC'd can land in the dataset. See §5.

---

## Why this matters — your fix becomes the X-ray dataset

These CT rib masks are not the end product. We project each segmented CT into a
**synthetic radiograph (a DRR)** and carry the 3-D masks down with it to build
**XRSpinoPelvic1K** — the first **densely-segmented spinal-radiograph dataset**,
in which *every* structure (vertebrae, sacrum, pelvis, femurs, **and ribs**) is
labelled on a plain-film-style image.

For those radiographs to be **fully and correctly masked, every rib in the CT must
be complete and correctly numbered.** A rib left in two pieces, or two ribs sharing
one number, projects into a **wrong or broken rib outline** on the synthetic X-ray
— a defect in the very labels the dataset exists to provide. **Your correction in
the CT is what makes the corresponding radiograph mask correct.**

Why a *radiograph* dataset at all: plain films are the modality used **in the
operating room**. A densely-segmented radiograph set lets us train **automatic
vertebral level-counting on X-ray, even in a limited field of view** — aimed
squarely at **preventing wrong-site surgery**. The ribs are part of the anatomy
that makes that counting reliable, so clean rib labels are essential, not cosmetic.

![rib cage](rib_review_example_0231_cage.png)

*A rib cage in the dataset, each rib coloured by its number (the flagged rib is
ringed). Every one of these projects to a labelled rib on the synthetic radiograph
— which is why getting the count and the pieces right matters.*

---

## 1. What you are fixing (30 seconds)

The build trusts TotalSegmentator's rib numbering (reliable level-to-level) and only repairs
it. The one residual error is a **DUPLICATE**: a single rib number (e.g. "right rib 9") appears
as **two (or more) disconnected pieces**.

**Almost always, both pieces ARE that rib** — the algorithm just split one rib into chunks. A
rib is a long curved arc, so the two pieces can even be **far apart** (e.g. a back chunk and a
front chunk with lung in between) and still be the same single rib. **Your default fix is to
CONNECT the pieces into one rib.**

Only two exceptions:
- a piece is genuinely a **different rib** that got the wrong number → **relabel** that piece;
- a piece **isn't a rib at all** (a transverse process, bowel, vascular calcification) →
  **delete** it.

When in doubt, the pieces are usually **one rib → connect**. There are **no missing numbers** to
add — only these duplicates. The terminal names the side, the rib number, and how far apart the
pieces are (distance is just a hint — far-apart pieces are still often one rib, as in 0007 below).

---

## 2. Examples

**CONNECT — the common case (0007, "right rib 9", pieces ~9 cm apart).**
Both pieces are **rib 9**: the algorithm split one long rib, and the straight-line gap happens
to cross the lung — but it is still a **single rib**. → **Connect the two pieces** into one rib
(do NOT relabel). Far apart ≠ a different rib.
![connect](rib_review_example_0007_mislabel.png)

**RELABEL — the exception (0231, "left rib 8").**
Here the two pieces are actually **two different rib arcs** that both got numbered 8. → **Relabel**
the wrong piece to its correct number (count from its neighbours). If you can't tell whether two
pieces are one rib or two, treat them as **one → connect**.
![relabel](rib_review_example_0231_split.png)

---

## 3. Setup (once — then `git pull` each session)

Get the code, install the client, sign in with your own free HuggingFace account, and
point the tool at the rib review service:

```bash
# first time only:
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K
git pull                                       # EVERY session — never run stale code
pip install requests huggingface_hub numpy nibabel scipy   # scipy: the live rib QC
hf auth login                                  # your own free HuggingFace login
python -m reviewtool login \
  --service https://anonymous-mlhc-ctspinopelvic1k-review-ribs.hf.space
# ITK-SNAP installed, with the AI-assisted (nnInteractive / DLS) backend configured
```

Run everything as `python -m reviewtool …` from inside the `CTSpinoPelvic1K` folder.
You sign in with your **own** HuggingFace login — there is no separate reviewer key.

**Set up AI-assisted segmentation — you'll use it to make every fix (don't hand-trace).** It's
ITK-SNAP's nnInteractive / Deep-Learning-Service backend; one-time setup (free Colab-GPU option)
is in **[REVIEWERS_AI_EDITING.md](../REVIEWERS_AI_EDITING.md)**. The GPU server runs from this
Colab notebook (open it, follow REVIEWERS_AI_EDITING.md):
**<https://colab.research.google.com/drive/14IHpQJtIxjnK2qdUcyxPmZjUXdH9F55n?usp=sharing>**

---

## 3b. Check your work & fix your flagged cases (AMEND)

We've strengthened the automatic QC. **`git pull` first** (every session), then check how your
submissions did — this is **private, only you see your own numbers**:

```bash
python -m reviewtool mystats
```

It prints your **submissions**, your **QC pass %**, how many are **still to fix**, and the fails by
check. If you have cases to fix, **amend your own work** (you do NOT start over — it re-opens your
own previous label):

```bash
python -m reviewtool next --amend
```

This hands you one of *your* flagged cases with **your own segmentation loaded**. Fix the flagged
items (the terminal names them on each Save), **Save** over `seg.nii.gz`, and quit — it re-submits
only once it passes QC. Repeat `python -m reviewtool next --amend` until `mystats` shows **0 to fix**.

**What BLOCKS your submission (you must fix these):**
- **one rib bone = one number** — a single connected rib bone must carry exactly ONE rib label
  (don't split one rib across two numbers, and don't give two ribs the same number).
- **each spine/pelvis bone is one clean class** — e.g. don't leave half a hip labelled a different
  class (the bone must be one dominant connected piece).

These are the only two hard blocks, and they're always fixable.

**Advisory — the tool shows these but they do NOT block your upload:**
- a rib in two pieces, a missing rib number, a gap between a rib and its vertebra, or
  **"rib N not incident on T-N."** These are very often just the rib **exiting the field of view**
  (partially scanned), which is **not fixable** — so they never block you, and you will **not** be
  handed the same case again for them. **But** if the rib is fully in view and genuinely broken,
  please still connect it; if it's cut off by the edge of the scan, **leave it and close — it will submit.**

### Don't chase the "halo"
When you re-segment a rib with nnInteractive, you'll often see tiny speckles of the **old** label
still clinging to the rib's outer surface. **Do not spend time cleaning those up.** Post-processing
automatically absorbs stray wrong-class specks along a corrected bone's border. Fix the actual
structure (connect / relabel / delete the real piece) and move on — the surface flecks are handled
for you.

### Skip a scan you don't want
Don't want a particular case (bad scan, unsure, or a rib simply cut off by the scan edge)? Release
it back to the queue for someone else:

```bash
python -m reviewtool skip            # the scan you're holding
python -m reviewtool skip --all      # clear any stuck/stale local claims at once
```

Your submitted work is never touched by `skip`.

### Your claimed cases are protected
A case you claim is held for **3 days**, and a case you're **amending can never be taken from you**
— so careful, multi-session work won't be reassigned mid-flight. Just try to submit or amend within
a few days of claiming; your progress and passed cases are safe.

### See problems in 3D
The fastest way to spot these is ITK-SNAP's **3D view**. Watch this short tutorial on building a 3D
render of your segmentation to catch splits, gaps, and mislabels at a glance:
**[3D render in ITK-SNAP — tutorial](https://drive.google.com/file/d/1R_6FtSncJcwyPm4SoYnRynVQtHBmX7Yr/view?usp=sharing)**
> Tip: after any edit, click **Update** in the 3D pane to rebuild the mesh. If it stays blank, fully
> quit and relaunch ITK-SNAP once — the 3D view needs a working OpenGL/GPU context.

---

## 4. Do a case, step by step

1. **Claim the next case** — no batches, no tokens; the server hands you one:
   ```bash
   python -m reviewtool next
   ```

2. It downloads the case and **opens ITK-SNAP** with the CT + the v4 label (locked
   palette). The terminal names the case and that you're reviewing the **ribs**.

3. **Find the rib that's in two pieces** (the terminal names it) and look at both on the CT.
   Ask: *are both pieces the same rib?* — **usually yes** (one rib the algorithm split, even if
   the pieces are far apart) → connect. Relabel only if a piece is clearly a **different** rib;
   delete only if a piece **isn't a rib** (transverse process, bowel, calcification).

4. **Fix it with AI-assisted segmentation (nnInteractive) — don't hand-trace** (setup:
   [REVIEWERS_AI_EDITING.md](../REVIEWERS_AI_EDITING.md)):
   - **CONNECT (the usual fix):** with that rib's label active, scribble across the gap so the
     two pieces become **one** connected rib.
   - **RELABEL (a different rib):** paint the wrong piece with its correct rib number.
   - **DELETE (not a rib):** set the stray piece to background (0).

5. **Save** (Ctrl-S, over the `seg.nii.gz` it opened). The rib QC re-runs and prints
   **exactly what's still wrong** (or `OK ribs`). Keep fixing and saving until it's clean.

6. **Quit** ITK-SNAP. If the ribs **PASS**, your edit is submitted. If a rib still
   fails, the tool **holds** the case (does not submit) and tells you to re-open and
   finish: `python -m reviewtool edit <case_id>`. (The server re-checks on submit as a
   backstop — see §5.)

Repeat `python -m reviewtool next` for as many as you like. Each case is shown to
**two** reviewers independently; disagreements go to a senior adjudicator. Check your
progress any time with `python -m reviewtool status`.

---

## 5. The QC gate — a bad rib can't be committed

Checked in **two places**, so nothing un-QC'd is ever committed:

- **In the tool (live):** every Save re-runs the rib QC and prints what's still wrong; if you
  quit while a rib still has a duplicate, the tool **holds** the case and does **not** submit it.
- **On the server (backstop):** the service re-runs the rib QC on submit and **rejects** any
  case that still has a duplicate/split rib. It **fails closed** — if the QC can't run, the
  submit is rejected.

So a bad rib can't be submitted from the tool, and even if it were, the server would refuse it.
Just make each rib **one clean piece per number** and quit when it prints `OK ribs`.

---

## 6. One-paragraph summary

`python -m reviewtool login --service https://anonymous-mlhc-ctspinopelvic1k-review-ribs.hf.space`,
then `python -m reviewtool next`. Each case has one rib number in two pieces — **usually one rib
the algorithm split, so CONNECT the pieces** with AI-assisted segmentation (nnInteractive,
[REVIEWERS_AI_EDITING.md](../REVIEWERS_AI_EDITING.md)); only relabel a piece that's a different
rib, or delete a piece that isn't a rib. Save, quit, and your edit is submitted; the server accepts it only once
the ribs are clean (a remaining duplicate is rejected and the case returns). Two reviewers
see each case; disagreements are adjudicated.
