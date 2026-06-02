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

> **Already cloned this before?** Update first (recent reviewer-tool fixes):
> `cd CTSpinoPelvic1K; git pull`

## 4. Review a case

```powershell
py -m reviewtool next
```

This claims a **flagged** case and prints a **`WHY FLAGGED - focus your edit
here`** list (what the automated checks think is wrong — that's your target; it
even names the split levels, e.g. `L3(63/37)`). It downloads a small **crop** (a
few MB, fast) and opens it in ITK-SNAP — **one window**. Then:

1. **Read the `WHY FLAGGED` focus list** — a mixed-up vertebra, a stray piece,
   etc. **New to editing? Keep [REVIEWERS_FIXING.md](REVIEWERS_FIXING.md) open** —
   a fix recipe for each kind of flag, plus short video tutorials.
2. **Only edit the region named** — `spine` (L1–L6) **or** `pelvis` (sacrum +
   both hips). Don't touch the other region. *(If it says **"the WHOLE scan"** —
   a radiologist gold case being re-checked — edit the whole label; both regions
   are fair game.)* **Don't renumber or recolor labels** — locked:
   `L1–L6 = 1–6`, `sacrum = 7`, `left hip = 8`, `right hip = 9`.
3. Fix it, then **Save Segmentation** (**Ctrl-S**) and **quit ITK-SNAP** — it
   uploads your result (`submitted -> ...`). Nothing actually wrong (the flag can
   be a false alarm)? Just save once and quit — that's a valid "accept".

> **Optional helpers** (add to the `next` command):
> - **`--reference`** — also open a **gold example** in a second window to compare to.
> - **`--live_qc`** — re-run the checks on every Save and watch them clear to **OK**
>   (off by default; with AI-assisted fixes you usually don't need it).
>
> e.g. `py -m reviewtool next --reference --live_qc`. You can also open the gold
> example on its own anytime: `py -m reviewtool reference`.

Repeat `py -m reviewtool next` for the next case. Check progress with:

```powershell
py -m reviewtool status
```

## Optional: download all cases at once (browse / offline)

To pull every flagged case in one go instead of one-by-one, download just the
crops (~10 GB — only the review set):

```powershell
py -m reviewtool download --what crops
```

They land in `$HOME\CTSpinoPelvic1K_data\crops\`, one folder per case
(`<token>__<config>\`) with `ct.nii.gz`, `seg.nii.gz`, and `labels.txt`. To look
at one, open ITK-SNAP → **File ▸ Open Main Image** → that case's `ct.nii.gz`,
then **Segmentation ▸ Open Segmentation** → its `seg.nii.gz`. Or from PowerShell
(adjust the ITK-SNAP version in the path):

```powershell
& "C:\Program Files\ITK-SNAP 4.2\bin\ITK-SNAP.exe" `
  -g $HOME\CTSpinoPelvic1K_data\crops\<case>\ct.nii.gz `
  -s $HOME\CTSpinoPelvic1K_data\crops\<case>\seg.nii.gz `
  -l $HOME\CTSpinoPelvic1K_data\crops\<case>\labels.txt
```

Bulk download is for browsing/offline. To submit corrections that count toward
the dataset, still use `py -m reviewtool next` (it claims + uploads).

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
