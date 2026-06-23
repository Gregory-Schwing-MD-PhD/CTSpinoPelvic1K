# Email draft — student onboarding

**Subject: Spine annotation miniproject — getting started**

Hi everyone,

You'll be helping build the next version of our CT spine/pelvis dataset
(**CTSpinoPelvic1K**). The current version (**v3**) already has the **bones**
segmented — vertebrae, sacrum, S1, hips, and femurs — **except ribs**, which is one
of the things you'll be adding.

**Pick a task** (just go to that task's link):
- **Ribs** *(easiest)* → https://anonymous-mlhc-ctspinopelvic1k-review-ribs.hf.space
- **Lumbosacral nerves** *(hardest)* → https://anonymous-mlhc-ctspinopelvic1k-review-nerve.hf.space
- **Iliolumbar ligament** → https://anonymous-mlhc-ctspinopelvic1k-review-ili.hf.space

**How it works** (one-time setup, then repeat per case):
```bash
git clone https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K.git && cd CTSpinoPelvic1K
pip install requests huggingface_hub numpy nibabel
hf auth login                                    # free HuggingFace account
python -m reviewtool login --service <your task's URL above>
python -m reviewtool next                         # opens ITK-SNAP with AI-assist; annotate, save & close to submit
```
Then just repeat `python -m reviewtool next` for each case (`python -m reviewtool status` shows your progress).

Every case is done independently by **two** students so we can measure agreement —
so just claim whatever the tool gives you. Step-by-step protocols (with reference
images) for each task are here: **`docs/annotation/`** in the dataset repo.

Cases go live within ~24 hours (once v3 finishes uploading) — I'll confirm when
they're ready to claim.

Any code you write goes into **OpenSpineToolkit** via a pull request (its README
walks you through it).

Thanks!
Greg
