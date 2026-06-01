# CTSpinoPelvic1K — Reviewer Guide (Linux / WSL)

Thanks for helping review the spine + pelvis segmentations!

**What you'll do:** Each CT scan already has an AI-generated *draft* segmentation.
You'll open one case at a time in **ITK-SNAP**, fix mistakes in **one region**,
save, and quit. The tool downloads the scan, measures your edits, and uploads the
result. Two reviewers see each case independently; disagreements go to a senior
adjudicator.

> This guide is for **Linux** and **Windows WSL (Ubuntu)**, where ITK-SNAP isn't
> in the package manager and you install the official Linux build by hand. On
> plain Windows or Mac, use [REVIEWERS_WINDOWS.md](REVIEWERS_WINDOWS.md) or
> [REVIEWERS_MAC.md](REVIEWERS_MAC.md) instead.

## Before you start — a free HuggingFace account

You sign in with a free HuggingFace account (there's **no separate reviewer
key**):

1. Make an account at <https://huggingface.co/join>.
2. Create a **Read** token: Settings → Access Tokens → **New token** → type
   **Read** → copy it.

You'll paste that token during `hf auth login` in step 1c.

## 1. One-time setup

Do these in order, top to bottom. Watch which folder you're in — the prompt
shows it (`~$` is your home folder; `~/CTSpinoPelvic1K$` is the repo).

> **Important:** every `reviewtool` command (Step 3 below and Section 2) must be
> run from **inside the `CTSpinoPelvic1K` folder**. If you open a new terminal
> later, run `cd ~/CTSpinoPelvic1K` first — otherwise you'll get
> `No module named reviewtool`.

### Step 1 — Install ITK-SNAP

`apt install itksnap` does **not** work on recent Ubuntu, so install the
official Linux build: it's a `.tar.gz` you unpack into a **home-directory**
folder (`~/itksnap`) and add to your `PATH`. (Not `/usr/bin` or `/usr/local` —
those belong to the package manager.)

1. **Download** the **Linux (gcc64) `.tar.gz`** from <http://www.itksnap.org> →
   Downloads.
   - **WSL:** download it in your Windows browser. WSL sees your Windows
     Downloads at `/mnt/c/Users/<YourWindowsName>/Downloads/`.
   - **Native Linux:** save it to `~/Downloads/`.

2. From your **home folder**, locate the file you downloaded, unpack it into
   `~/itksnap`, and add it to your PATH (replace the `tar` filename with your
   real one — browsers sometimes rename the download):

```bash
cd ~
ls -t ~/Downloads/*.tar.gz /mnt/c/Users/*/Downloads/*.tar.gz 2>/dev/null   # find your download

mkdir -p ~/itksnap
tar -C ~/itksnap --strip-components=1 -xzf ~/Downloads/itksnap-4.4.0-Linux-x86_64.tar.gz   # <- your file

echo 'export PATH="$HOME/itksnap/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

which itksnap        # should print  /home/<you>/itksnap/bin/itksnap
```

`--strip-components=1` strips the version-stamped top folder, so the program
always lands at exactly `~/itksnap/bin/itksnap` and `reviewtool` finds it on
your `PATH` automatically.

> **WSL only:** the ITK-SNAP window displays through WSLg (Windows 11). Check
> that `echo $DISPLAY` prints something like `:0`. If it's blank, run
> `wsl --update` in Windows PowerShell and reopen your terminal.

### Step 2 — Get the review tool

```bash
cd ~
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K          # stay in this folder for Step 3 and for reviewing
python3 -m pip install requests huggingface_hub numpy nibabel
```

> **Already cloned it before?** Don't re-clone — update it instead:
> `cd ~/CTSpinoPelvic1K && git pull`.

### Step 3 — Sign in (from inside `~/CTSpinoPelvic1K`)

```bash
hf auth login        # paste your Read token; answer "n" to git creds
python3 -m reviewtool login --service https://gregoryschwingmdphd-ctspinopelvic1k-review-triaged.hf.space
```

Section 1 is done **once**.

## 2. Review a case

From inside `~/CTSpinoPelvic1K` (run `cd ~/CTSpinoPelvic1K` first if you're not
there):

```bash
python3 -m reviewtool next
```

This claims a **flagged** case, downloads a small **crop** (fast), and opens
ITK-SNAP. The terminal prints a **`WHY FLAGGED:`** line with the suspected
problem. **New to editing? Open [REVIEWERS_FIXING.md](REVIEWERS_FIXING.md)** —
ITK-SNAP tools + a fix recipe per flag. (Add `--full` to also open the whole
scan read-only for context.) Then:

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

> **Optional — faster editing with AI:** if a draft needs heavy re-drawing, you
> can use ITK-SNAP's nnInteractive AI tool (free Google Colab, a local NVIDIA
> GPU, or a GPU server). Entirely optional; see
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

- **`invalid credentials` / `401`** → you're not signed in to HuggingFace (or the
  token expired). Run `hf auth login` again with a valid **Read** token.
- **`itksnap not found` / nothing opens** → run `which itksnap`; it should print
  a path ending in `/bin/itksnap`. If it prints nothing, redo the PATH step in
  1b and **open a new terminal** (or `source ~/.bashrc`). You can also point at
  it directly: `python3 -m reviewtool next --itksnap ~/itksnap/bin/itksnap`.
- **ITK-SNAP errors about a missing library** (`libGL.so.1`, `libxcb-*`) →
  `sudo apt-get install -y libopengl0 libgl1 libxcb-xinerama0 libxcb-cursor0`.
- **`$DISPLAY` is blank (WSL)** → WSLg isn't active; run `wsl --update` in
  Windows PowerShell, then reopen the terminal.
- **`No module named reviewtool`** → make sure you're in the folder:
  `cd CTSpinoPelvic1K`.
- **"nothing to claim"** → all cases assigned/done for now; check back later.

Stuck on anything else? Send the project lead a copy of what the terminal
printed.
