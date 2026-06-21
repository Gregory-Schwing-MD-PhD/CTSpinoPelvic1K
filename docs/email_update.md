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

**Publication plan & deadlines**
- **P1 — v3 dataset paper:** at **MLHC 2026** (in review). Backup: BIBM if needed; ML4H Findings for visibility; Scientific Data as the durable citation.
- **P2 — toolbox (PI/LL):** **BIBM 2026 — Jul 5** (full paper) or **SPIE Medical Imaging 2027 — abstract Aug 5** (2–4 pp; manuscript Jan 27, 2027).
- **P3 — LS-nerve segmentation:** MICCAI/MIDL 2027; AJNR/World Neurosurgery.
- **P4 — ribs + iliolumbar + LSTV/TLTV (+VerSeFusion):** the v4 dataset paper — Scientific Data / MICCAI 2027.

Full project + deadline matrix is in `docs/ROADMAP.md`.

**Students:** any code you write goes to OpenSpineToolbox via pull request — see its README.

Happy to walk anyone through their piece.

Best,
Gregory Schwing, MD, PhD
PGY-1, Surgery-Preliminary · NIH F30 Fellow · Research Director, OpenSpineConsortium
