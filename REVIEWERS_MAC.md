# CTSpinoPelvic1K — Reviewer Guide (Mac)

Thanks for helping review the spine + pelvis segmentations!

**What you'll do:** Each CT scan already has an AI-generated *draft* segmentation.
You'll open one case at a time in **ITK-SNAP**, fix mistakes in **one region**,
save, and quit. The tool downloads the scan, measures your edits, and uploads the
result. Two reviewers see each case independently; disagreements go to a senior
adjudicator. No prior command-line experience needed.

## Before you start — your reviewer key

Your project lead will send you **your personal reviewer key** — a code that
looks like `k_3f9a8c1d…`. You'll paste it in once, in step 2 below.

- It is created and sent **to you** by the project lead — you do **not** sign up
  for it anywhere.
- It is **not** a HuggingFace token; you don't need a HuggingFace account at all.
- No key yet? Ask the project lead before continuing.

## 1. Install ITK-SNAP

Download from <http://www.itksnap.org> (Downloads), open the `.dmg`, and drag
**ITK-SNAP** into your Applications folder. The review tool finds it
automatically.

## 2. Get the review tool and connect

Open **Terminal** (Applications → Utilities → Terminal) and run these one at a
time. Replace `<YOUR_REVIEWER_KEY>` with the key your project lead sent you:

```bash
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K
python3 -m pip install requests huggingface_hub numpy nibabel
python3 -m reviewtool login --service https://gregoryschwingmdphd-ctspinopelvic1k-review.hf.space --key <YOUR_REVIEWER_KEY>
```

> - If `git` prompts to install "command line developer tools," click
>   **Install**, then re-run the `git clone` line.
> - If you get `command not found: python3`, install Python from
>   <https://www.python.org/downloads/> and reopen Terminal.

Step 1–2 are done **once**.

## 3. Review a case

```bash
python3 -m reviewtool next
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

Repeat `python3 -m reviewtool next` for the next case. Check progress with:

```bash
python3 -m reviewtool status
```

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
- **`itksnap not found`** → ITK-SNAP isn't in Applications; reinstall it there, or
  add `--itksnap /Applications/ITK-SNAP.app/Contents/bin/itksnap` to the `next`
  command.
- **macOS won't open ITK-SNAP ("unidentified developer")** → right-click the app
  in Applications → **Open** → **Open** once to approve it.
- **`No module named reviewtool`** → make sure you're in the folder:
  `cd CTSpinoPelvic1K`.
- **"nothing to claim"** → all cases assigned/done for now; check back later.
- **permission / `401` downloading a scan** → the dataset is access-controlled;
  tell the project lead (you'd then make a free HuggingFace Read token and run
  `hf auth login`).

Stuck on anything else? Send the project lead a copy of what the terminal
printed.
