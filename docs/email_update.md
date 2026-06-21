# Email draft — team / collaborator update

**Subject: CTSpinoPelvic1K — v3 status, v4 annotation launch, and publication timeline**

Hi all,

A consolidated update on where the dataset, the toolbox, and the annotation effort
stand, plus the deadlines we're aiming at.

**Dataset (v3 — bone)**
v3 = v2 + femurs + carved S1 + GT thoracic. The build pipeline had a scratch-fill
bug that was silently shipping bone-less labels; that's fixed (per-case temp
cleanup, fail-fast so a broken run can't push, resume that only skips
genuinely-completed cases). v3 is currently re-segmenting on the grid and has not
yet been pushed to HF — once it finishes cleanly it auto-pushes to `@v3` and
promotes `@main` to match.

**v4 annotation (now standing up)**
Three med-student tasks, each AI-assisted with double-annotation + inter-rater
reliability (per-class Dice, faculty adjudication on disagreement):
- **Ribs** (easier) — paint into the reserved ids 26–49; rib number read off the adjacent GT thoracic vertebra.
- **LS nerves** (hard) — L4/L5/S1 roots; serves Kambin's-triangle mapping *and* LSTV neural enumeration.
- **Iliolumbar ligament** — LSTV level-anchor cross-check.

Each runs as its own HuggingFace Space + private review ledger (all reading `@v3`).
The three Spaces are deployed and configured but idle until v3 lands — once it's
pushed we factory-reboot them and cases populate. Annotation protocols are in the
repo under `docs/annotation/`.

**Toolbox (OpenSpineToolbox)**
Spinopelvic-parameter extraction from the masks, built in order of clinical utility
starting with **Pelvic Incidence** (the one sagittal parameter valid on supine CT).
Shared mask-I/O contract + the PI spec are in `SPEC.md`; SVA/TPA are out of scope
for this FOV.

**Publication plan — one project = one paper, each with aims + deadline**

**Paper 1 — CTSpinoPelvic1K v3 (bone) dataset** · *MLHC 2026 — in review (conf Aug 12–14, 2026); backups: BIBM, ML4H Findings, Scientific Data*
- Aim 1: Build v3 — add femurs + carve S1 + GT thoracic onto v2.
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

Full matrix is in `docs/ROADMAP.md`.

**Students:** any code you write goes to OpenSpineToolbox via pull request — see its README.

Happy to walk anyone through their piece.

Best,
Gregory Schwing, MD, PhD
PGY-1, Surgery-Preliminary · NIH F30 Fellow · Research Director, OpenSpineConsortium
