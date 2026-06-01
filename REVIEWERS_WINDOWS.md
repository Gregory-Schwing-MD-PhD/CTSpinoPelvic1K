# CTSpinoPelvic1K — Reviewer Guide (Windows)

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

You'll paste that token during `hf auth login` in step 3.

## 1. Install Python

Download from <https://www.python.org/downloads/> and run the installer. **On the
first screen, check "Add python.exe to PATH"**, then click *Install Now*.

## 2. Install ITK-SNAP

Download from <http://www.itksnap.org> (Downloads) and run the installer,
accepting the defaults. The review tool finds it automatically.

## 3. Get the tool and sign in

Open **PowerShell** (Start menu → type "PowerShell") and run these one at a time:

```powershell
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K
py -m pip install requests huggingface_hub numpy nibabel scipy
hf auth login
py -m reviewtool login --service https://gregoryschwingmdphd-ctspinopelvic1k-review-triaged.hf.space
```

> - `hf auth login` asks for the **Read** token you copied — paste it (it stays
>   hidden as you type) and press Enter. Answer "n" if it asks about git
>   credentials.
> - No `git`? Install it from <https://git-scm.com/download/win> (defaults are
>   fine), reopen PowerShell, and re-run the `git clone` line.

Steps 1–3 are done **once**.

## 4. Review a case

```powershell
py -m reviewtool next
```

This claims a **flagged** case and prints a **`WHY FLAGGED - focus your edit
here`** list (what the automated checks think is wrong — that's your target). It
downloads a small **crop** (a few MB, fast), opens it in ITK-SNAP, and opens a
**gold reference example** in a second window to compare against. Then:

1. **Read the `WHY FLAGGED` focus list** — a leak, a mixed-up vertebra, a stray
   piece, etc. **New to editing? Keep [REVIEWERS_FIXING.md](REVIEWERS_FIXING.md)
   open** — it has a fix recipe for each kind of flag, and links to short video
   tutorials.
2. **Only edit the region named** — `spine` (L1–L6) **or** `pelvis` (sacrum +
   both hips). Don't touch the other region. *(If it says **"the WHOLE scan"** —
   a radiologist gold case being re-checked — edit the whole label; both regions
   are fair game.)* **Don't renumber or recolor labels** — locked:
   `L1–L6 = 1–6`, `sacrum = 7`, `left hip = 8`, `right hip = 9`.
3. Fix it, then **Save Segmentation** (**Ctrl-S**). The terminal **re-runs the QC
   and shows your progress** — e.g. `off-bone leak 0.072 -> 0.008 OK`. Keep
   editing and saving until the checks read **OK**: that's your measurable goal.
4. **Quit ITK-SNAP** when the checks are OK — it uploads your result
   (`submitted -> ...`).
   - **Tile the two windows** (your case + the gold example) to compare.
   - Nothing actually wrong (the flag can be a false alarm)? Just save once and
     quit — that's a valid "accept".

Repeat `py -m reviewtool next` for the next case. Check progress with:

```powershell
py -m reviewtool status
```

> **Optional — faster editing with AI:** if a draft needs heavy re-drawing, you
> can use ITK-SNAP's nnInteractive AI tool (runs on free Google Colab — no GPU
> needed). Entirely optional; see
> [REVIEWERS_AI_EDITING.md](REVIEWERS_AI_EDITING.md).

## If an upload gets interrupted

Internet drop, crash, or a "rate-limited" message **after you saved**? Your edit
is **not lost** — it's on your computer. Recover it with:

```powershell
py -m reviewtool resume
```

Safe to run anytime, even if everything already went through.

## Rules of thumb

- **One region only** — the one named in the terminal. Manual labels are gold.
- **Don't renumber or recolor labels.**
- Draft already correct → just save and quit.
- Scan unusable (wrong anatomy / corrupt)? Don't fix it — tell the project lead.

## Troubleshooting

- **`py` not recognized** → reinstall Python and check "Add python.exe to PATH"
  (step 1), then reopen PowerShell.
- **`invalid credentials` / `401`** → you're not signed in to HuggingFace (or the
  token expired). Run `hf auth login` again with a valid **Read** token.
- **`itksnap not found`** → it installed somewhere nonstandard; add
  `--itksnap "C:\Program Files\ITK-SNAP 4.0\bin\ITK-SNAP.exe"` (adjust the
  version) to the `next` command.
- **`No module named reviewtool`** → make sure you're in the folder:
  `cd CTSpinoPelvic1K`.
- **"nothing to claim"** → all cases assigned/done for now; check back later.

Stuck on anything else? Send the project lead a copy of what the terminal
printed.
