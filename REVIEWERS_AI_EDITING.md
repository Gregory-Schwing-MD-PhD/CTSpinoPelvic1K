# Optional: AI-assisted editing with nnInteractive

**This is optional.** You can review every case with ITK-SNAP's normal brush and
polygon tools — nothing here is required. But if a draft needs a lot of
re-drawing, ITK-SNAP's **Deep Learning Service (DLS)** lets you re-segment a
structure with a few clicks or scribbles using the **nnInteractive** model,
which is often much faster than brushing voxel by voxel.

nnInteractive needs an **NVIDIA GPU**, so you run a small "server" somewhere that
has one and point ITK-SNAP at it. Three ways, easiest first.

## Option A — Google Colab (recommended; no GPU of your own needed)

1. Open the official notebook:
   <https://colab.research.google.com/drive/1AtB2yZgB6KUHb6e0LHVVE9pjsaLANVr5?usp=sharing>
2. **Runtime → Change runtime type → GPU** (a free T4 is fine).
3. The notebook uses a secure **ngrok** tunnel. Make a free account at
   <https://dashboard.ngrok.com/signup>, copy your **authtoken**, and paste it
   where the notebook asks.
4. Run the cells top to bottom. When it's ready it prints a banner like
   `Server: xxxx.ngrok-free.app   Port: 443` — keep that.
5. Connect ITK-SNAP using that address + port (see **Connect ITK-SNAP** below).

> Colab disconnects when idle or after a few hours. If the AI stops responding,
> re-run the notebook and reconnect — your review work is unaffected (it's saved
> locally and submitted as usual).

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

The first time, click the **AI** button under the paintbrush tool → **Yes,
configure** (or open **Preferences → AI Extensions → New…**). Choose **Network
connection to GPU server**, then enter the **Server address** and **Port** from
your server (Colab uses port **443**; a self-run server defaults to **8911**).
For a firewalled server (Option C), tick **Use SSH tunnel** and enter your SSH
username; ITK-SNAP will prompt for the password.

Once connected, the **AI** tool gives you point / scribble / lasso prompts that
nnInteractive turns into a 3D segmentation.

## Using it within the review — important

- It only changes *how you draw*. **Everything else is the same:** review only
  the region the terminal named, **Save Segmentation Image** over `seg.nii.gz`,
  then quit. `reviewtool` measures and submits exactly as before.
- **Keep the locked palette.** nnInteractive fills the **active label**, so set
  the active label to the right structure *before* prompting — `L1–L6 = 1–6`,
  `sacrum = 7`, `left hip = 8`, `right hip = 9`. Never renumber labels.
- **You still decide the anatomy.** nnInteractive segments *a* structure; it
  doesn't know vertebral levels — you assign the correct level, and the LSTV
  judgment stays yours.
- Touch **only the assigned region**; the other region is manual gold.

Full, version-current details are in the official
**[ITK-SNAP DLS Quick Start](https://itksnap-dls.readthedocs.io/en/latest/quick_start.html)**.
