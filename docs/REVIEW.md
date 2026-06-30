# CTSpinoPelvic1K — distributed segmentation review

This page has two parts:

- **[Part A — Reviewers](#part-a--reviewers-start-here)** — if you were asked to
  correct CT segmentations, start here. No prior command-line or ML experience
  needed.
- **[Part B — Maintainer setup](#part-b--maintainer-setup)** — only the person
  running the project needs this (stand up the backend, build the final
  dataset). Reviewers can ignore it.

---

## Part A — Reviewers (start here)

### What you'll be doing
Each CT scan already has an **AI-generated draft segmentation** of the spine
and pelvis. Your job is to open it in **ITK-SNAP**, **fix mistakes in one
region**, save, and quit. The tool handles everything else (downloading the
scan, measuring your changes, uploading the result). You'll do this one case
at a time; two reviewers see each case independently, and disagreements go to
a senior adjudicator.

You will need three things, set up once: **(1)** a free HuggingFace account +
Read token (this is how you sign in — there's no separate reviewer key),
**(2)** ITK-SNAP + the `reviewtool` program, **(3)** the **service URL** the
maintainer sent you.

### 1. Make a HuggingFace account + token (5 min)
1. Go to **https://huggingface.co/join** and create a free account.
2. (If the maintainer says the dataset is *gated*) open the dataset page they
   linked and click **“Agree / Access repository”**.
3. Get a token: **profile → Settings → Access Tokens → + Create new token →
   type “Read”** → name it `ctspine-review` → **Create** → **copy** the
   `hf_...` string (you won't see it again).

This token only lets you *download* scans. You never get the dataset's write
token — uploads of your corrections go through the project's server.

### 2. Install the tools (15 min, once)
- **Python 3.10+** — https://www.python.org/downloads/ (on macOS/Linux it's
  usually already there; check with `python3 --version`).
- **ITK-SNAP** — http://www.itksnap.org/pmwiki/pmwiki.php?n=Downloads.SNAP4
  (install the desktop app for your OS).
- **reviewtool** — open a terminal and run:
  ```bash
  git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git   # first time
  cd CTSpinoPelvic1K
  git pull                          # if you already cloned it: UPDATE to the latest code first
  pip install requests huggingface_hub numpy nibabel scipy
  ```
  You'll run the tool as `python -m reviewtool …` from inside that
  `CTSpinoPelvic1K` folder. **Run `git pull` at the start of every session** —
  an out-of-date copy can have the wrong QC and your edits may not gate correctly.

> **ITK-SNAP path:** `reviewtool` auto-detects ITK-SNAP in the standard
> location — `/Applications/ITK-SNAP.app/Contents/bin/itksnap` on macOS,
> `C:\Program Files\ITK-SNAP *\bin\ITK-SNAP.exe` on Windows, or anything named
> `itksnap` on your `PATH`. You only need to point it out if you installed it
> somewhere nonstandard: pass `--itksnap "/full/path/to/itksnap"` to the
> commands below, or set `REVIEWTOOL_ITKSNAP=/full/path` once.

### 3. Connect (once)
```bash
hf auth login                  # paste your Read token when prompted
python -m reviewtool login --service <SERVICE_URL>
```
You sign in with your own HuggingFace account — there's no separate reviewer
key. `<SERVICE_URL>` comes from the maintainer (it looks like
`https://<org>-ctspinopelvic1k-review-triaged.hf.space`).

### 4. Review a case
```bash
python -m reviewtool next
```
This will:
1. **Claim** a case and **download** its CT + draft segmentation.
2. **Open ITK-SNAP** with the scan and the draft, using a fixed colour
   palette.
3. Wait while you work. **In ITK-SNAP:**
   - The terminal tells you **which region to review** — `spine` (labels
     L1–L6) **or** `pelvis` (sacrum + both hips). **Only edit that region.**
     The other region is an expert manual annotation — **do not touch it.**
   - Use the brush/polygon tools to fix the draft. **Do not renumber or
     recolour labels** (the palette is locked: L1–L6 = 1–6, sacrum = 7,
     left hip = 8, right hip = 9).
   - **Segmentation → Save Segmentation Image**, saving over the file it
     opened (`seg.nii.gz`), then **quit ITK-SNAP**.
4. On quit, the tool measures your edits and **uploads** the result. If you
   changed nothing, it records **“accept”** (the draft was already correct);
   otherwise **“corrected.”**

Repeat `python -m reviewtool next` for the next case. Check overall progress
anytime with:
```bash
python -m reviewtool status
```

**If an upload is interrupted** (network drop, the tool crashes, or you see a
"rate-limited" message), your edit is **not lost** — it's saved on disk. Just
run:
```bash
python -m reviewtool resume
```
to re-send any pending case(s). It's safe to run anytime, even if everything
already went through (re-sending work the server already has is a no-op).

### Rules of thumb
- **One region only** — the one named in the terminal. Manual labels are gold.
- **Don't renumber labels.** Keep the palette as loaded.
- A draft that's already correct → just save & quit (that's a valid “accept”).
- If a scan is **unusable** (wrong anatomy, corrupt), don't try to fix it —
  tell the maintainer; they (or an adjudicator) will exclude it.

### Troubleshooting
- **`itksnap not found`** → pass `--itksnap /full/path/...` (see box above).
- **`401`/permission error on download** → run `huggingface-cli login` again,
  and make sure you clicked “Agree” on the dataset page (gated datasets).
- **“nothing to claim”** → all cases are assigned/done for now; check back
  later or ping the maintainer.
- **upload failed / “rate-limited” / tool crashed after you saved** → your
  edit is safe on disk; run `python -m reviewtool resume` to send it.
- **Adjudicators only:** `python -m reviewtool adjudicate --notes "…"` shows a
  case where two reviewers disagreed (with both their results); you produce
  the deciding label the same way (edit → save → quit).

---

## Part B — Maintainer setup

You stand up one backend; reviewers just point `reviewtool` at it. The
heavy NIfTIs live on HuggingFace; a small **HF Space** (FastAPI) coordinates
claims/locking/adjudication and holds the single write token.

### Repos
| repo | visibility | role |
|---|---|---|
| `<org>/CTSpinoPelvic1K` | **public** (or gated) | dataset — `main`=v1 (manual), branch `v2`=pseudo, branch `v3`=corrected |
| `<org>/CTSpinoPelvic1K-reviews` | **private** | review coordination (claims/status/reviews/corrected labels) |
| `<org>/CTSpinoPelvic1K-review` | Space (Docker) | the review service |

### 1. Publish v2 (the thing being reviewed)
After `make hf-stage` (v1) and `make pseudolabel` (v2), push v2 to a branch:
```bash
HF_TOKEN=<write> HF_REPO_ID=<org>/CTSpinoPelvic1K \
  HF_REVISION=v2 HF_EXPORT_DIR=$(pwd)/data/hf_export_v2 make hf-push
```

### 2. Deploy the review Space (Docker SDK — HF builds it, no Docker Hub)
```bash
hf repos create <org>/CTSpinoPelvic1K-review --type space --space-sdk docker
git clone https://huggingface.co/spaces/<org>/CTSpinoPelvic1K-review
cd CTSpinoPelvic1K-review
SRC=/path/to/CTSpinoPelvic1K
cp "$SRC/review_service/Dockerfile" Dockerfile
cp "$SRC/review_service/README.md"  README.md          # has the sdk:docker front-matter
mkdir -p review_service scripts/review
cp "$SRC"/review_service/*.py "$SRC"/review_service/requirements.txt review_service/
cp "$SRC"/scripts/review/*.py scripts/review/
git add -A && git commit -m "review service" && git push
```
Then in the Space's **Settings → Variables and secrets**:

| name | value |
|---|---|
| `HF_TOKEN` *(secret)* | write token for `…-reviews` |
| `REVIEW_REPO` | `<org>/CTSpinoPelvic1K-reviews` |
| `V2_REPO` | `<org>/CTSpinoPelvic1K` |
| `SOURCE_REVISION` | `v2` |
| `ADJUDICATORS` | `drsmith,drokafor` (HF usernames; everyone else who signs in is a primary reviewer) |
| `TAU` | `0.9` |

Reviewers authenticate with their own HuggingFace login (open mode) — there are
no per-reviewer keys to mint or distribute. `REVIEWER_KEYS` is still honored if
set (legacy), but isn't needed.

On boot it seeds one review case per `spine_only`/`pelvic_native` record from
the v2 manifest. The Space URL (`https://<org>-ctspinopelvic1k-review-triaged.hf.space`)
is the `--service` value you give reviewers; the dashboard is at `/`.

### 3. Onboard reviewers
Send each reviewer the per-OS setup guide — **[REVIEWERS.md](../REVIEWERS.md)**
(it links the Windows / Mac / Linux versions). The service URL is already filled
into those guides. Reviewers sign in with their own free HuggingFace account —
**no keys to mint or distribute.** For senior **adjudicators**, add their HF
username to the `ADJUDICATORS` Space variable (everyone else is a primary
reviewer automatically). Keep the Space at **1 replica** (the atomic-claim lock
assumes a single worker).

> **Smoke-test before onboarding:** sign in with your own HF account
> (`hf auth login`), then `reviewtool login --service <url>` and run
> `python -m reviewtool next` on one case end-to-end (claim → tiny edit →
> save → quit); confirm a record + label appear in the private `…-reviews`
> repo. The service core is unit-tested, but verify the live HTTP + ITK-SNAP
> path once.

### 4. Build v3 when review is done
```bash
# adjudicator (or you) writes the finalized index:
curl -X POST -H "Authorization: Bearer k_snr" <SERVICE_URL>/admin/build_finals
# pull reviews + corrected labels, fold into v3, publish:
hf download <org>/CTSpinoPelvic1K-reviews --repo-type dataset --local-dir data/reviews_pull
python scripts/review/reduce_to_v3.py --v2 data/hf_export_v2 \
    --finals data/reviews_pull/finals.json --labels_root data/reviews_pull \
    --out data/hf_export_v3
HF_TOKEN=<write> HF_REPO_ID=<org>/CTSpinoPelvic1K \
  HF_REVISION=v3 HF_EXPORT_DIR=$(pwd)/data/hf_export_v3 make hf-push
```

Design details (state machine, IRR, label-source tracking) live in
[`scripts/review/`](../scripts/review); the service in
[`review_service/`](../review_service); the client in
[`reviewtool/`](../reviewtool).
