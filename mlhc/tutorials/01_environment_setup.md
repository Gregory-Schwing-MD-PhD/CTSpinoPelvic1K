# Tutorial 1 — Setting Up Your Environment

*AI Imaging Workshop · WSUSOM · the 9:30 AM "Python / SLURM / Good Programming
Practices" block.*

By the end of this tutorial you will have, on the WSU grid (and/or your own
laptop):

1. **Miniconda** — Python plus a package manager that keeps each project's
   software isolated and reproducible.
2. An **imaging environment** with the libraries we use to read and manipulate
   CT scans and segmentation labels (`nibabel`, `SimpleITK`, `pydicom`, …).
3. **Nextflow** — the tool we use to run reproducible, parallel pipelines on
   the SLURM cluster.

You only do this once. Everything is copy-paste; read the one-line explanation
above each block so you know *what* you are running, not just *that* it runs.

---

## 0. Open a terminal

Everything below is typed into a **terminal** (a text window where you run
commands). The compute for this workshop lives on the **WSU grid**, so that is
where we will install things.

**Log in to the grid** (replace `youraccessid` with your WSU AccessID):

```bash
ssh youraccessid@grid.wayne.edu
```

> **Laptop instead of the grid?** The same commands work on **macOS** (open the
> *Terminal* app) and on **Linux**. On **Windows**, first install **WSL**
> (Windows Subsystem for Linux) by running `wsl --install` in PowerShell, then
> open the "Ubuntu" app — that gives you a Linux terminal. The only thing that
> changes per-platform is the Miniconda *installer file* in Step 1; we note the
> Mac variants there.

You know you are in the right place when each line starts with a prompt like
`[youraccessid@warrior ~]$`.

---

## 1. Install Miniconda

**What it is:** Miniconda bundles Python and `conda`, a program that downloads
software libraries and keeps them in self-contained "environments" so two
projects can never break each other's dependencies.

**Download the installer** (Linux x86-64 — this is the grid and most laptops):

```bash
cd ~
curl -L -o miniconda.sh \
  https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
```

> **On a Mac**, swap the URL for your chip:
> - Apple Silicon (M1/M2/M3): `Miniconda3-latest-MacOSX-arm64.sh`
> - Intel Mac: `Miniconda3-latest-MacOSX-x86_64.sh`

**Run the installer** (`-b` = accept defaults, `-p` = where to put it):

```bash
bash miniconda.sh -b -p ~/miniconda3
rm miniconda.sh                     # tidy up the installer
```

**Turn conda on** so your shell knows where to find it, now and in every future
session:

```bash
~/miniconda3/bin/conda init bash
source ~/.bashrc                    # reload the shell so 'conda' works now
```

You should now see `(base)` at the start of your prompt — that is conda's
default environment, telling you it is active.

**Use the open, community package channel** (`conda-forge`) by default — it has
the most up-to-date scientific and medical-imaging packages:

```bash
conda config --add channels conda-forge
conda config --set channel_priority strict
```

---

## 2. Create the imaging environment

**Why a separate environment:** we keep the workshop's libraries in their own
named space (`imaging`) so that installing or upgrading something later never
disturbs anything else. Activating it is how you "enter" that space.

**Create it with Python 3.11:**

```bash
conda create -n imaging python=3.11 -y
conda activate imaging
```

Your prompt now shows `(imaging)` instead of `(base)` — you are inside the new
environment. (Run `conda activate imaging` at the start of any future session.)

**Install the imaging libraries** — these are the tools for reading and editing
CT scans and label maps:

```bash
conda install -y \
  numpy scipy pandas matplotlib \
  nibabel pydicom simpleitk scikit-image
```

What each one is for:

| Package        | What you use it for                                              |
|----------------|-----------------------------------------------------------------|
| `numpy`        | the array — every CT volume and label map is a 3-D NumPy array  |
| `scipy`        | image operations: connected components, morphology, resampling  |
| `nibabel`      | read/write **NIfTI** (`.nii.gz`) — the format our labels ship in |
| `pydicom`      | read **DICOM** — the raw format scanners produce                |
| `SimpleITK`    | resampling, registration, format conversion (DICOM ↔ NIfTI)     |
| `scikit-image` | extra image analysis (region labelling, measurements)           |
| `pandas`       | tables — reading the dataset manifest / writing results to CSV  |
| `matplotlib`   | plotting slices and figures                                     |

---

## 3. Install Nextflow (for running pipelines on SLURM)

**What it is:** Nextflow runs multi-step analysis pipelines and submits the work
to the SLURM scheduler for you — in parallel, and reproducibly (the same command
gives the same result, on your laptop or the cluster). Nextflow needs **Java**,
which conda installs alongside it:

```bash
conda install -y -c bioconda nextflow
```

That single command pulls in both Nextflow and a compatible Java runtime into
your `imaging` environment.

---

## 4. Check that everything works

**Python libraries** — this prints the version of each and a final "OK":

```bash
python - <<'PY'
import numpy, scipy, pandas, matplotlib, nibabel, pydicom, SimpleITK, skimage
print("numpy     ", numpy.__version__)
print("scipy     ", scipy.__version__)
print("nibabel   ", nibabel.__version__)
print("pydicom   ", pydicom.__version__)
print("SimpleITK ", SimpleITK.__version__)
print("skimage   ", skimage.__version__)
print("ALL IMPORTS OK")
PY
```

**Nextflow and Java:**

```bash
nextflow -version
java -version
```

If you see version numbers and `ALL IMPORTS OK`, your environment is ready.

**A 30-second real test** — make a tiny 3-D volume, save it as NIfTI, read it
back. This is exactly the read/write loop you'll use on real scans:

```bash
python - <<'PY'
import numpy as np, nibabel as nib
vol = np.random.randint(0, 100, size=(8, 8, 8)).astype(np.int16)   # fake volume
nib.save(nib.Nifti1Image(vol, affine=np.eye(4)), "test.nii.gz")    # write
back = np.asarray(nib.load("test.nii.gz").dataobj)                 # read
print("round-trip identical:", np.array_equal(vol, back))
PY
rm -f test.nii.gz
```

`round-trip identical: True` means you can load, edit, and save medical images.

---

## 5. Everyday use (after today)

Each time you log in and want to work, just activate the environment:

```bash
conda activate imaging
```

To leave it:

```bash
conda deactivate
```

To see your environments, or remove one and start over:

```bash
conda env list                      # list all environments
conda env remove -n imaging         # delete 'imaging' if you want a clean redo
```

---

## Troubleshooting

- **`conda: command not found`** after Step 1 → you missed `source ~/.bashrc`,
  or open a fresh terminal. As a fallback: `source ~/miniconda3/bin/activate`.
- **`CondaToSNonInteractiveError` / channel/terms-of-service errors** → you are
  hitting Anaconda's default channels. The `conda config` commands in Step 1
  (switch to `conda-forge`) fix this; re-run them, then retry.
- **A package install hangs on "Solving environment"** → it's just slow. You can
  install the much faster drop-in solver `mamba` once
  (`conda install -y -n base -c conda-forge mamba`) and then use `mamba install`
  in place of `conda install` everywhere.
- **`nextflow: command not found`** → make sure `imaging` is active
  (`conda activate imaging`); Nextflow was installed *into* that environment.
- **Out of disk space in your home directory on the grid** → conda environments
  are large. Ask whether to place `~/miniconda3` on a project/scratch filesystem
  instead of home.

---

**Next tutorial:** *DICOM vs. NIfTI — what's actually inside a medical image
file*, where we open a real CT scan and its segmentation with the tools you just
installed.
