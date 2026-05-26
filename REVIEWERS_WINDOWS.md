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
py -m pip install requests huggingface_hub numpy nibabel
hf auth login
py -m reviewtool login --service https://gregoryschwingmdphd-ctspinopelvic1k-review.hf.space
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

This claims a case, downloads its CT + draft, and opens ITK-SNAP. Then:

1. The terminal says **which region to review** — `spine` (L1–L6) **or** `pelvis`
   (sacrum + both hips). **Only edit that region**; the other is an expert manual
   annotation — don't touch it.
2. Fix the draft with the brush / polygon tools. **Don't renumber or recolor
   labels** — the palette is locked: `L1–L6 = 1–6`, `sacrum = 7`,
   `left hip = 8`, `right hip = 9`.
3. **Segmentation → Save Segmentation Image**, saving **over the file it opened**
   (`seg.nii.gz`). Then **quit ITK-SNAP**.
4. On quit the tool uploads your result — you'll see `submitted -> ...`. Nothing
   to fix? Just save and quit; that's a valid "accept".

Repeat `py -m reviewtool next` for the next case. Check progress with:

```powershell
py -m reviewtool status
```

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
