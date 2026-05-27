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

### a) Get the review tool

```bash
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git
cd CTSpinoPelvic1K
python3 -m pip install requests huggingface_hub numpy nibabel
```

### b) Install ITK-SNAP (Linux build)

`apt install itksnap` does **not** work on recent Ubuntu — install the official
Linux build instead. It's just a `.tar.gz`: you unpack it into a folder and add
that folder's `bin/` to your `PATH`.

**Step 1 — download** the **Linux (gcc64) `.tar.gz`** from
<http://www.itksnap.org> → Downloads.
- **WSL:** download it in your Windows browser; WSL sees it at
  `/mnt/c/Users/<YourWindowsName>/Downloads/`.
- **Native Linux:** save it to `~/Downloads/` (or `wget` the link there).

**Step 2 — unpack into `~/itksnap` and put it on your PATH.** Replace the path
in the first line with your actual download (e.g. the WSL Downloads path):

```bash
mkdir -p ~/itksnap
tar -C ~/itksnap --strip-components=1 -xzf ~/Downloads/itksnap-*-Linux-*.tar.gz

echo 'export PATH="$HOME/itksnap/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc

which itksnap        # should print  /home/<you>/itksnap/bin/itksnap
```

`--strip-components=1` drops the version-stamped top folder, so the program
always lands at exactly `~/itksnap/bin/itksnap`. `reviewtool` finds anything
named `itksnap` on your `PATH` automatically — no extra config.

> **Already unpacked it somewhere else?** You don't need to move it — just add
> *that* folder's `bin` to your PATH and reload, e.g.:
> ```bash
> echo 'export PATH="/full/path/to/itksnap-<version>-Linux-x86_64/bin:$PATH"' >> ~/.bashrc
> source ~/.bashrc
> which itksnap
> ```

> **WSL only:** the ITK-SNAP window displays through WSLg (Windows 11). Check
> that `echo $DISPLAY` prints something like `:0`. If it's blank, run
> `wsl --update` in Windows PowerShell and reopen your terminal.

### c) Sign in

```bash
hf auth login        # paste the Read token you copied; answer "n" to git creds
python3 -m reviewtool login --service https://gregoryschwingmdphd-ctspinopelvic1k-review.hf.space
```

Section 1 is done **once**.

## 2. Review a case

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
