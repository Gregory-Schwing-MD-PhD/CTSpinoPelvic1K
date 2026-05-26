# CTSpinoPelvic1K — Reviewer Guide (Linux / WSL)

Thanks for helping review the spine + pelvis segmentations!

> **On macOS or native Windows?** Use **[REVIEWERS.md](REVIEWERS.md)** instead —
> ITK-SNAP installs much more simply there. This guide is for **Linux** and
> **Windows WSL (Ubuntu)**, where ITK-SNAP isn't in the package manager and you
> install the official Linux build by hand.

**What you'll do:** Each CT scan already has an AI-generated *draft*
segmentation. Your job is to open one case at a time in **ITK-SNAP**, fix
mistakes in **one region**, save, and quit. The tool does everything else —
downloading the scan, measuring your edits, and uploading the result. Two
reviewers see each case independently; disagreements go to a senior adjudicator.

**You need two things from the project lead** (they'll send these):
- your **reviewer key** (looks like `k_...`)
- the **service URL** (already in the commands below)

---

## 1. One-time setup

### a) Get the review tool

```bash
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K
python3 -m pip install requests huggingface_hub numpy nibabel
```

### b) Install ITK-SNAP (Linux build)

`apt install itksnap` does **not** work on recent Ubuntu — install the official
Linux build instead:

1. Download the **Linux (gcc64) `.tar.gz`** from <http://www.itksnap.org> →
   Downloads.
   - **WSL:** download it in your Windows browser; WSL sees your Downloads at
     `/mnt/c/Users/<YourWindowsName>/Downloads/`.
   - **Native Linux:** save it to `~/Downloads/` (or `wget` the link).
2. Extract it and tell `reviewtool` where it is:

```bash
cd ~
# adjust the path/glob to wherever you saved it:
tar xzf /mnt/c/Users/<YourWindowsName>/Downloads/itksnap-*-Linux-*.tar.gz
ls ~/itksnap-*/bin/itksnap        # confirm the binary exists

# remember it permanently (reviewtool reads this first):
echo "export REVIEWTOOL_ITKSNAP=$(ls ~/itksnap-*/bin/itksnap | head -1)" >> ~/.bashrc
source ~/.bashrc
```

> **WSL only:** the ITK-SNAP window displays through WSLg (Windows 11). Check
> that `echo $DISPLAY` prints something like `:0`. If it's blank, update WSL
> (`wsl --update` in Windows PowerShell) and reopen your terminal.

### c) Connect to the review service

```bash
python3 -m reviewtool login \
  --service https://gregoryschwingmdphd-ctspinopelvic1k-review.hf.space \
  --key <YOUR_REVIEWER_KEY>
```

You only do this whole section **once**.

---

## 2. Review a case

Run this each time you want to do a case:

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
already went through.

---

## Rules of thumb

- **One region only** — the one named in the terminal. The manual labels in the
  other region are gold; leave them exactly as they are.
- **Don't renumber or recolor labels.** Keep the palette as loaded.
- A draft that's already correct → just **save and quit** (a valid "accept").
- If a scan is **unusable** (wrong anatomy, corrupt), don't try to fix it —
  **tell the project lead**; they or an adjudicator will handle it.

---

## Troubleshooting

- **`itksnap not found` / nothing opens** → re-check step 1b: confirm
  `ls ~/itksnap-*/bin/itksnap` shows the file, and that
  `echo $REVIEWTOOL_ITKSNAP` prints its full path. You can also pass it
  explicitly: `python3 -m reviewtool next --itksnap ~/itksnap-*/bin/itksnap`.
- **ITK-SNAP errors about a missing library** (e.g. `libGL.so.1`,
  `libxcb-*`) → install them and retry:
  `sudo apt-get install -y libopengl0 libgl1 libxcb-xinerama0 libxcb-cursor0`.
- **`$DISPLAY` is blank (WSL)** → WSLg isn't active; run `wsl --update` in
  Windows PowerShell, then reopen the terminal.
- **`No module named reviewtool`** → make sure you're inside the
  `CTSpinoPelvic1K` folder (`cd CTSpinoPelvic1K`).
- **"nothing to claim"** → all cases are assigned or done for now; check back
  later or ping the project lead.
- **A permission / `401` error while downloading a scan** → the dataset is
  access-controlled. Make a free **Read** token at
  <https://huggingface.co/settings/tokens>, run `hf auth login`, paste it, then
  click **"Agree / Access repository"** on the dataset page the project lead
  links you, and retry.

Stuck on anything else? Send the project lead a copy of what the terminal
printed.
