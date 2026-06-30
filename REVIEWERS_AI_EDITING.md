# Optional: AI-assisted editing with nnInteractive

**This is optional.** You can review every case with ITK-SNAP's normal brush and
polygon tools — nothing here is required. But if a draft needs a lot of
re-drawing, ITK-SNAP's **Deep Learning Service (DLS)** lets you re-segment a
structure with a few clicks or scribbles using the **nnInteractive** model,
which is often much faster than brushing voxel by voxel.

nnInteractive needs an **NVIDIA GPU**, so you run a small "server" somewhere that
has one and point ITK-SNAP at it. Three ways, easiest first.

## Option A — Google Colab (recommended; no GPU of your own needed)

1. Open our notebook (package versions are already pinned, so it just works —
   don't `pip install` anything yourself):
   <https://colab.research.google.com/drive/14IHpQJtIxjnK2qdUcyxPmZjUXdH9F55n?usp=sharing>
2. **Runtime → Change runtime type → GPU** (a free T4 is fine).
3. **Generate your OWN ngrok token and paste it in.** Make a free account at
   <https://dashboard.ngrok.com/signup>, open **Your Authtoken**, copy it, and paste it into the
   cell that says `NGROK_AUTHTOKEN="..."`. Everyone needs their **own** token — a shared/blank one
   fails.
4. **Paste your Hugging Face read token** into the `HF_TOKEN="..."` cell (use the same `hf_…` read
   token you made for reviewing). **This is required** — without it the model download hits Hugging
   Face's anonymous rate limit on Colab's shared IPs and the session dies with *"Error creating
   session on DLS server: Internal Server Error."*
5. **Install — the restart dance, do it exactly:**
   1. **Runtime → Run all.**
   2. While the install cells run, Colab pops up **"Restart session"** — click **Cancel** each time
      and let it keep going until **every** install cell has finished.
   3. Now do **Runtime → Restart session and run all.**
   4. When the **"Restart session"** popup appears again, click **Cancel** this time — the packages
      are already installed; just let the rest (especially the **server** cell) run to the end.
   (The red pip "dependency conflict" lines are **expected** — the notebook pins older versions on
   purpose. Ignore them.)
6. When the server cell is ready it prints a banner like:
   `Server: xxxx.ngrok-free.dev   Port: 443` — copy **both** the address and the port.
7. Connect ITK-SNAP with that address + port **443** (see **Connect ITK-SNAP** below).

> Colab disconnects when idle or after a few hours. If the AI stops responding, re-run (Run all →
> cancel the restarts → Restart-and-run-all → cancel) and reconnect — your review work is safe
> (saved locally, submitted as usual).

## Option B — A local NVIDIA GPU (Windows/Linux only)

Macs have no NVIDIA GPU, so skip this on a Mac. On a Windows/Linux machine with
an NVIDIA card and current CUDA drivers + Python 3.10+: in ITK-SNAP go
**Preferences → AI Extensions → New… → Local computer**, pick a Python
executable and a package directory, then press **Setup Python Packages** (this
installs `itksnap-dls` and downloads the models). Status turns **Connected**.

## Option C — A GPU server you can reach (e.g. the WSU grid)

If you have access to a Linux GPU server, install and run the service there:

```bash
conda create -n itksnap-dls python=3.12
conda activate itksnap-dls
pip install itksnap-dls
python -m itksnap_dls          # serves on port 8911 by default
```

Then connect from ITK-SNAP with **Use SSH tunnel** checked (see below).

## Connect ITK-SNAP to the server

The first time, click the **AI** button under the paintbrush tool → **Yes, configure** (or open
**Preferences → AI Extensions → New…**). Choose **Network connection to GPU server**, then:
- **Server address:** paste the banner's address **exactly** — just the host (e.g.
  `xxxx.ngrok-free.dev`), with **no** `https://`, **no** port, and **no** trailing slash.
- **Port:** **443** for Colab (a self-run server uses **8911**).

The status should turn **Connected**. (Firewalled server / Option C: tick **Use SSH tunnel** and
enter your SSH username; ITK-SNAP prompts for the password.)

Once connected, the **AI** tool gives you point / scribble / lasso prompts that
nnInteractive turns into a 3D segmentation.

## Using it within the review — important

- It only changes *how you draw*. **Everything else is the same:** review only
  the region the terminal named, **Save Segmentation Image** over `seg.nii.gz`,
  then quit. `reviewtool` measures and submits exactly as before.
- **Keep the locked palette.** nnInteractive fills the **active label**, so set the active label
  to the right structure *before* prompting. The label names show in the Active-label list and
  under the cursor (e.g. `rib_left_8`, `L3`). Never renumber labels.
- **You still decide the anatomy.** nnInteractive segments *a* structure; it
  doesn't know vertebral levels — you assign the correct level, and the LSTV
  judgment stays yours.
- Touch **only the assigned region**; the other region is manual gold.

## Troubleshooting

- **"Error creating session on DLS server: Internal Server Error"** — and the Colab log shows a
  `json.decoder.JSONDecodeError: Expecting value: line 1 column 1 (char 0)` coming from
  `itksnap_dls/segment.py … hf.snapshot_download → repo_info → model_info → r.json()`:
  the model download was **unauthenticated**, so Hugging Face returned an empty body (anonymous
  rate-limit on Colab's shared IP). **Fix:** set your HF read token before the server cell —
  `import os; os.environ["HF_TOKEN"]=os.environ["HUGGING_FACE_HUB_TOKEN"]="hf_…"` (Option A step 4) —
  then re-run the server cell. If it persists, the nnInteractive model repo is **gated**: open its
  HF page with that account and click **Agree**. This is **not** a wifi/ngrok problem — connecting
  fine but failing on *use* is exactly this.
- **You don't actually need the AI to connect two rib pieces:** set the rib's label active, pick
  the brush, and paint one stroke across the gap to join them. That's the fix — start now even if
  the AI server is being sorted out.
- **Can't connect at all:** the address must be the bare host (no `https://`/port/slash), the port
  **443**, and the Colab **server** cell must still be running (banner visible).

Full, version-current details are in the official
**[ITK-SNAP DLS Quick Start](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html)**.
