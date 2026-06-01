# CTSpinoPelvic1K — Reviewer Guide (Mac)

Thanks for helping review the spine + pelvis segmentations!

**What you'll do:** Each CT scan already has an AI-generated *draft* segmentation.
You'll open one case at a time in **ITK-SNAP**, fix mistakes in **one region**,
save, and quit. The tool downloads the scan, measures your edits, and uploads the
result. Two reviewers see each case independently; disagreements go to a senior
adjudicator. No prior command-line experience needed.

## Before you start — a free HuggingFace account

You sign in with a free HuggingFace account (there's **no separate reviewer
key**):

1. Make an account at <https://huggingface.co/join>.
2. Create a **Read** token: Settings → Access Tokens → **New token** → type
   **Read** → copy it.

You'll paste that token during `hf auth login` in step 2.

## 1. Install ITK-SNAP

Download from <http://www.itksnap.org> (Downloads), open the `.dmg`, and drag
**ITK-SNAP** into your Applications folder. The review tool finds it
automatically.

## 2. Get the tool and sign in

Open **Terminal** (Applications → Utilities → Terminal) and run these one at a
time:

```bash
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K
python3 -m pip install requests huggingface_hub numpy nibabel scipy
hf auth login
python3 -m reviewtool login --service https://gregoryschwingmdphd-ctspinopelvic1k-review-triaged.hf.space
```

> - `hf auth login` asks for the **Read** token you copied — paste it (it stays
>   hidden as you type) and press Enter. Answer "n" if it asks about git
>   credentials.
> - If `git` prompts to install "command line developer tools," click
>   **Install**, then re-run the `git clone` line.
> - If you get `command not found: python3`, install Python from
>   <https://www.python.org/downloads/> and reopen Terminal.

Steps 1–2 are done **once**.

## 3. Review a case

```bash
python3 -m reviewtool next
```

This claims a **flagged** case and prints a **`WHY FLAGGED - focus your edit
here`** list (your target). It downloads a small **crop** (fast), opens it in
ITK-SNAP, and opens a **gold reference example** in a second window to compare
against. Then:

1. **Read the `WHY FLAGGED` focus list. New to editing? Open
   [REVIEWERS_FIXING.md](REVIEWERS_FIXING.md)** — ITK-SNAP tools, video
   tutorials, and a fix recipe per flag.
2. **Only edit the region named** — `spine` (L1–L6) **or** `pelvis` (sacrum +
   hips); don't touch the other. *(If it says **"the WHOLE scan"** — a gold case
   being re-checked — edit the whole label.)* **Don't renumber/recolor labels** —
   `L1–L6 = 1–6, sacrum = 7, left hip = 8, right hip = 9`.
3. Fix it, then **Save Segmentation** (**⌘-S**). The terminal **re-runs the QC
   and shows progress** (`off-bone leak 0.072 -> 0.008 OK`). Keep editing and
   saving until the checks read **OK** — that's your goal. (Tile the two windows
   to compare against the gold example.)
4. **Quit ITK-SNAP** when OK — it uploads (`submitted -> ...`). Nothing wrong?
   Save once and quit (a valid "accept").

Repeat `python3 -m reviewtool next` for the next case. Check progress with:

```bash
python3 -m reviewtool status
```

> **Optional — faster editing with AI:** if a draft needs heavy re-drawing, you
> can use ITK-SNAP's nnInteractive AI tool (runs on free Google Colab — no GPU
> needed, which matters on a Mac). Entirely optional; see
> [REVIEWERS_AI_EDITING.md](REVIEWERS_AI_EDITING.md).

## If an upload gets interrupted

Internet drop, crash, or a "rate-limited" message **after you saved**? Your edit
is **not lost** — it's on your computer. Recover it with:

```bash
python3 -m reviewtool resume
```

Safe to run anytime, even if everything already went through.

## Rules of thumb

- **One region only** — the one named in the terminal. Manual labels are gold.
- **Don't renumber or recolor labels.**
- Draft already correct → just save and quit.
- Scan unusable (wrong anatomy / corrupt)? Don't fix it — tell the project lead.

## Troubleshooting

- **`command not found: python3`** → install Python from
  <https://www.python.org/downloads/>, then reopen Terminal.
- **`invalid credentials` / `401`** → you're not signed in to HuggingFace (or the
  token expired). Run `hf auth login` again with a valid **Read** token.
- **`itksnap not found`** → ITK-SNAP isn't in Applications; reinstall it there, or
  add `--itksnap /Applications/ITK-SNAP.app/Contents/bin/itksnap` to the `next`
  command.
- **macOS won't open ITK-SNAP ("unidentified developer")** → right-click the app
  in Applications → **Open** → **Open** once to approve it.
- **`No module named reviewtool`** → make sure you're in the folder:
  `cd CTSpinoPelvic1K`.
- **"nothing to claim"** → all cases assigned/done for now; check back later.

Stuck on anything else? Send the project lead a copy of what the terminal
printed.
