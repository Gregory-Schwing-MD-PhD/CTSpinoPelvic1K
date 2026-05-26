# CTSpinoPelvic1K — Reviewer Guide (macOS / Windows)

> **On Linux or Windows WSL?** ITK-SNAP installs differently there — use
> **[REVIEWERS_LINUX.md](REVIEWERS_LINUX.md)** instead of this guide.

Thanks for helping review the spine + pelvis segmentations!

**What you'll do:** Each CT scan already has an AI-generated *draft* segmentation.
Your job is to open one case at a time in **ITK-SNAP**, fix mistakes in **one
region**, save, and quit. The tool does everything else — downloading the scan,
measuring your edits, and uploading the result. Two reviewers see each case
independently; disagreements go to a senior adjudicator. No prior command-line
or machine-learning experience needed.

**You need two things from the project lead** (they'll send these):
- your **reviewer key** (looks like `k_...`)
- the **service URL** (the one in the commands below)

---

## 1. One-time setup (~15 min)

### macOS

1. **Install ITK-SNAP** — download from <http://www.itksnap.org>, open the
   `.dmg`, and drag **ITK-SNAP** into your Applications folder.
2. Open **Terminal** (Applications → Utilities → Terminal) and run:

```bash
# Download the review tool
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K

# Install its dependencies
python3 -m pip install requests huggingface_hub numpy nibabel

# Connect to the review service (paste the key the project lead sent you)
python3 -m reviewtool login \
  --service https://gregoryschwingmdphd-ctspinopelvic1k-review.hf.space \
  --key <YOUR_REVIEWER_KEY>
```

> If `git` prompts to install "command line developer tools," click **Install**
> and re-run the `git clone` line afterward.

### Windows

1. **Install ITK-SNAP** — download from <http://www.itksnap.org> and run the
   installer (accept the defaults).
2. **Install Python** from <https://www.python.org/downloads/> — on the first
   screen, **check "Add Python to PATH"**.
3. Open **PowerShell** (Start menu → type "PowerShell") and run the same
   commands, but use `py` instead of `python3` and put each command on one line:

```powershell
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K
py -m pip install requests huggingface_hub numpy nibabel
py -m reviewtool login --service https://gregoryschwingmdphd-ctspinopelvic1k-review.hf.space --key <YOUR_REVIEWER_KEY>
```

You only do this section **once**. (The reviewtool finds ITK-SNAP automatically
in the standard install location — you don't need to tell it where it is.)

---

## 2. Review a case

Run this each time you want to do a case (use `py` instead of `python3` on
Windows):

```bash
python3 -m reviewtool next
```

This will:

1. **Claim** a case and **download** its CT + draft segmentation.
2. **Open ITK-SNAP** with the scan and the draft.
3. Wait while you work. **In ITK-SNAP:**
   - The terminal tells you **which region to review** — `spine` (labels L1–L6)
     **or** `pelvis` (sacrum + both hips). **Only edit that region.** The other
     region is an expert manual annotation — **do not touch it.**
   - Use the brush / polygon tools to fix the draft. **Do not renumber or
     recolor labels** — the palette is locked:
     `L1–L6 = 1–6`, `sacrum = 7`, `left hip = 8`, `right hip = 9`.
   - **Segmentation → Save Segmentation Image**, saving **over the file it
     opened** (`seg.nii.gz`). Then **quit ITK-SNAP.**
4. On quit, the tool measures your edits and **uploads** the result. If you
   changed nothing, it records **"accept"** (the draft was already correct);
   otherwise **"corrected."** You'll see `submitted -> ...`.

Repeat `python3 -m reviewtool next` for the next case.

### Check progress anytime

```bash
python3 -m reviewtool status
```

---

## 3. If an upload gets interrupted

If your internet drops, the tool crashes, or you see a "rate-limited" message
**after you saved** — your edit is **not lost**. It's saved on your computer.
Just run:

```bash
python3 -m reviewtool resume
```

to re-send any pending case(s). It's always safe to run, even if everything
already went through (re-sending work the server already has does nothing).

---

## Rules of thumb

- **One region only** — the one named in the terminal. The manual labels in the
  other region are gold; leave them exactly as they are.
- **Don't renumber or recolor labels.** Keep the palette as loaded.
- A draft that's already correct → just **save and quit** (that's a valid
  "accept").
- If a scan is **unusable** (wrong anatomy, corrupt, the structure isn't even in
  the image), don't try to fix it — **tell the project lead**; they or an
  adjudicator will handle it.

---

## Troubleshooting

- **`itksnap not found`** → ITK-SNAP isn't installed in the standard place. Pass
  its full path, e.g. on macOS:
  `python3 -m reviewtool next --itksnap /Applications/ITK-SNAP.app/Contents/bin/itksnap`
  (on Windows: `--itksnap "C:\Program Files\ITK-SNAP 4.0\bin\ITK-SNAP.exe"`).
- **`command not found: python3`** (macOS) → install Python from
  <https://www.python.org/downloads/>, then reopen Terminal.
- **`reviewtool` "No module named reviewtool"** → make sure you're inside the
  `CTSpinoPelvic1K` folder (`cd CTSpinoPelvic1K`).
- **"nothing to claim"** → all cases are assigned or done for now; check back
  later or ping the project lead.
- **A permission / `401` error while downloading a scan** → the dataset is
  access-controlled. Create a free account at <https://huggingface.co/join>,
  make a **Read** token (Settings → Access Tokens → New token), run
  `hf auth login` and paste it, then click **"Agree / Access repository"** on
  the dataset page the project lead links you. Retry `reviewtool next`.

Stuck on anything else? Send the project lead a copy of what the terminal
printed.
