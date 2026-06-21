# Email draft — team / collaborator + student update

**Subject: CTSpinoPelvic1K — vision, v4 annotation tasks, roadmap & how to contribute**

Hi all,

A consolidated update on where CTSpinoPelvic1K is headed, the annotation tasks
starting now, the publication timeline, and how to contribute code.

---

## The vision

I've come across a paper that segmented **lumbosacral nerves on non-contrast CT**,
so I'd like to add these nerves — as they branch off the spinal cord — to
CTSpinoPelvic1K. For those unaware, the latest version of the dataset has the
**femurs** segmented, which can be used to isolate the **hip joint** by finding
where the femur mask interfaces with the hip mask.

While the fidelity of the LS nerves won't be as high as MRI, it will permit mapping
of the **foramina** and **Kambin's triangle** by finding where the LS nerve
interfaces with the vertebra. This simplifies locating the foramina on the vertebra,
permits analysis of **stenosis**, and in the future could help **plan surgeries**.

The **LSTV project** is also moving forward. By adding the **ribs**, we get a
**rostral anchor** for the lumbar vertebrae, which — together with the **S1
endplate** as the **caudal anchor** — brackets the start and stop of the lumbar
classes. (The **iliolumbar ligament**, which arises from the L5 transverse process,
gives a third, independent level anchor.) I don't expect a neural net to segment
anomalies automatically from a dataset this small — but by also bringing the
**VerSeFusion** dataset up to high-quality annotations, we may get the samples
needed to train a network to segment spinal anomalies as accurately as normal spines.

---

## Dataset status (v3 — bone)

v3 = v2 + femurs + carved S1 + GT thoracic. The build pipeline had a scratch-fill
bug that was silently shipping bone-less labels; that's fixed (per-case temp
cleanup, fail-fast so a broken run can't push, resume that only skips
genuinely-completed cases). v3 is currently re-segmenting on the grid and has not
yet been pushed to HF — once it finishes cleanly it auto-pushes to `@v3` and
promotes `@main` to match.

---

## Annotation tasks (students)

Each task is AI-assisted, with every case independently annotated by **two** students
for inter-rater reliability (per-class Dice; faculty adjudication on disagreement).
Per-structure protocols (with reference images):

**On CTSpinoPelvic1K v3 (n = 801):**
- **Ribs** *(easier)* — [guide](https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K/blob/master/docs/annotation/ribs.md)
- **LS nerves** *(hard)* — [guide](https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K/blob/master/docs/annotation/ls_nerves.md)
- **Iliolumbar ligament** — [guide](https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K/blob/master/docs/annotation/iliolumbar.md)

**On VerSeFusion (m = 200):** vertebra + ribs + LS nerves, re-annotated to
high quality (the anomaly-training set; tooling to follow).

Each CTSpinoPelvic1K task runs as its own HuggingFace Space + private review ledger
(all reading `@v3`). The three Spaces are **deployed and configured but idle until
v3 lands** — once it's pushed we factory-reboot them and cases populate. Workflow +
the Space/ledger map are here:
[docs/annotation/](https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K/blob/master/docs/annotation/README.md).
I'll confirm when cases are ready to claim (within ~24 h of the v3 push).

---

## Publication roadmap — one project = one paper, each with aims + deadline

Full matrix: [docs/ROADMAP.md](https://github.com/Gregory-Schwing-MD-PhD/CTSpinoPelvic1K/blob/master/docs/ROADMAP.md).

**Paper 1 — CTSpinoPelvic1K v3 (bone) dataset** · *MLHC 2026 — in review (conf Aug 12–14, 2026); backups: BIBM, ML4H Findings, Scientific Data*
- Aim 1: Build v3 (femurs + carved S1 + GT thoracic on v2).
- Aim 2: Baseline nnU-Net (pseudolabels L1–L5 + sacrum) + the reviewer-requested ablation.
- Aim 3: Publish v3 to HF (`@v3`, promote `@main`).

**Paper 2 — Spinopelvic parameters from CT masks (OpenSpineToolbox)** · *BIBM 2026 — Jul 5 (full paper), or SPIE Medical Imaging 2027 — abstract Aug 5, 2026 (mss Jan 27, 2027)*
- Aim 1: Develop PI extraction code (femoral-head sphere-fit + S1 endplate + sagittal projection).
- Aim 2: Manually verify PI (MAE / ICC / Bland–Altman vs manual).
- Aim 3: Extend to LL + PI–LL mismatch (develop + verify).

**Paper 3 — Lumbosacral nerve segmentation & Kambin's-triangle mapping** · *MICCAI 2027 (~Feb) / MIDL 2027 (~Dec 2026); clinical: AJNR / World Neurosurgery*
- Aim 1: Annotate L4/L5/S1 nerve roots on v3 (student task + IRR).
- Aim 2: Train + validate a nerve segmentation model (vs MRI subset).
- Aim 3: Derive Kambin's-triangle / foraminal geometry + LSTV neural enumeration.

**Paper 4 — v4 dataset: ribs, iliolumbar ligament & LSTV/TLTV detection** · *BIBM 2027 (~Jul) / Scientific Data (rolling) / MICCAI 2027*
- Aim 1: Segment ribs on v3 (student task; numbering read from GT thoracic).
- Aim 2: Segment the iliolumbar ligament (student task).
- Aim 3: LSTV/TLTV detection via three anchors (rib / S1-bony / neural + iliolumbar) + VerSeFusion augmentation; publish v4.

---

## Contributing code (required)

Anyone who develops code for their miniproject must upload it to GitHub, or the
contribution to the toolbox will be removed.

**Repository:** <https://github.com/Gregory-Schwing-MD-PhD/OpenSpineToolbox>

**What to do:** upload your miniproject code via a pull request. The
[README](https://github.com/Gregory-Schwing-MD-PhD/OpenSpineToolbox/blob/main/README.md)
walks you through it step by step — create a GitHub account → fork the repo → clone
your fork → make a branch → drop your code into your project's folder → push → open a
pull request. There's a pre-created folder for every miniproject under `projects/`,
each with a short README — find yours and copy your code in. Start from the shared
contract in [`SPEC.md`](https://github.com/Gregory-Schwing-MD-PhD/OpenSpineToolbox/blob/main/SPEC.md)
(how to read the masks; build PI first). *(First-time Git note: GitHub no longer
takes your password on the command line — use `gh auth login` or a Personal Access
Token; there's a troubleshooting section in the README.)*

Happy to walk anyone through their piece.

Best,
Gregory Schwing, MD, PhD
PGY-1, Surgery-Preliminary · NIH F30 Fellow · Research Director, OpenSpineConsortium
