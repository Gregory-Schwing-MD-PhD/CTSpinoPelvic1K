# Surgical hardware in CTSpinoPelvic1K

The label scheme has carried a hardware block since case 0068 was deferred, and until v6 it
was populated in **no record at all**. This is what the block now means, what is actually in
the cohort, and what a reader confirmed.

## Why it matters, in one paragraph

**An iatrogenic fusion is indistinguishable from a congenital one to a distance
measurement.** The transitional-anatomy result measures the gap between the lowest lumbar
vertebra and the sacrum, and a cage-bridged interspace reads as "no gap" exactly like a
congenitally fused transitional vertebra. Separately, **pelvic incidence and pelvic tilt are
measured from the femoral head centre** — and in nine of these cases the femoral head is
metal. Any spinopelvic parameter computed from them was measured on an implant.

## The classes

| id | name | what it is |
|---|---|---|
| 76 | `hardware` | metal that no rule could name; a placeholder, not an answer |
| 77 | `hardware_cage` | interbody cage or spacer, in the disc space |
| 78 | `hardware_screw_rod` | pedicle screws and rods — spinal posterior instrumentation |
| 79 | `hardware_plate` | plates and other flat fixation |
| 80 | `hardware_arthroplasty` | joint **replacement**: femoral stem, head, acetabular cup |
| 81 | `hardware_si_screw` | iliosacral fixation crossing the sacroiliac joint |
| 82 | `hardware_osteosynthesis` | metal holding parts of the **same bone** together |

### 80 against 82 — the distinction that matters most

These are the two arms of the same clinical decision, and the femoral-neck literature is
written as exactly that dichotomy — meta-analyses are titled
["arthroplasty vs. osteosynthesis"](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5602948/).

- **Osteosynthesis** holds a fractured bone together: plates, screws, intramedullary nails.
- **Arthroplasty** replaces all or part of a joint with a prosthesis.

Getting it wrong inverts a measurement rather than merely mislabelling a voxel: **fixation
leaves the patient's own femoral head**, a prosthesis does not, and the femoral head centre
is the landmark PI and PT are taken from.

82 was briefly named `hardware_fracture_fix`, which was a description rather than a term.

## What is in the cohort

Eleven records, from 52 candidate proposals. Every one read by a radiologist.

| case | class | notes |
|---|---|---|
| 0974 | arthroplasty | bilateral **total** hip, cups both sides |
| 0515 | arthroplasty | bilateral **total** hip, cups both sides |
| 1003 | arthroplasty | right; **cup status unread** — the only possible hemiarthroplasty |
| 0443 | arthroplasty | left total, cup present |
| 0671 | arthroplasty | left total, cup present |
| 0188 | arthroplasty | right total, cup present |
| 0485 | arthroplasty | left total, cup present |
| 1128 | arthroplasty | left total, cup present **with supplementary screws into the ilium** |
| 0247 | osteosynthesis | three parallel cannulated screws, 7.2–7.7 mm × 77–93 mm, inverted-triangle configuration, left femoral neck |
| 1035 | si_screw | sacroiliac screws crossing the joint |
| 0068 | cage | paired threaded cylindrical interbody cages, 27 × 14 × 14 mm, L5–L6 |

### 1128 — supplementary transacetabular screw fixation

The acetabular component is anchored by screws passing through the cup into the ilium. This
is standard, optional, and worth recording because the metal leaves the joint and enters the
innominate bone.

Most cementless acetabular cups are **multihole**: implanted press-fit, with holes for
optional screws. Whether the screws are needed is genuinely contested — a
[systematic review and meta-analysis](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8831679/)
concluded press-fit alone achieves sufficient stability and additional screws are not
required, while
[in-vitro work](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4363363/) found that where the
press-fit is poor, cup stability depends on the screws and more screws means more stability.
So their presence says something about the operation: a surgeon reaching for them usually
had a reason — deficient bone stock, dysplasia, revision, or a press-fit that did not feel
solid.

Where they may go is not free. The
[Wasielewski acetabular quadrant system](https://pubmed.ncbi.nlm.nih.gov/2324135/) divides
the acetabulum by a line from the anterior superior iliac spine through its centre to the
posterior fovea, and a perpendicular at the midpoint. **The posterosuperior and
posteroinferior quadrants are safe; the anterior quadrants are not** — the external iliac
vessels and the obturator neurovascular bundle lie behind them. The **posterosuperior**
quadrant carries the best bone stock and the widest margin of safety, which is where
adjunctive screws normally go.

**For this dataset the consequence is concrete:** transacetabular screws put metal inside
the ilium, so on 1128 the hardware displaces innominate bone and not only femur. Any pelvic
morphometry on that case is measuring around an implant.

## What could not be determined automatically

**Whether a cup is present.** Two methods, both wrong on a case with a known answer:

1. *Counting which label the metal displaced* — a stem takes femur voxels, a cup takes hip
   voxels. This reads the **segmenter's** opinion, not the anatomy: a bone segmenter handed
   one contiguous cup-head-stem blob assigns it wherever it touches most. It called 0974's
   right side a hemiarthroplasty on 117 hip voxels; a reader saw a cup.
2. *Fill fraction in cross-section* — a cup is a shell, a bare head is solid. But **the
   metal ball sits inside the cup**, so a total hip is solid in section too. It called both
   of 0974's sides solid.

Cup status is therefore a **reader field**. Nine of the eleven carry a reader's answer; 1003
does not.

**How many screws anchor 1128's cup.** A morphological opening was meant to strip the thin
screws from the bulky cup and stem. It peeled a shell off the cup instead and returned rim
fragments, none elongated. The screws are recorded from the reader's observation, uncounted.

## Detection, and why the counts moved

- `scan_hardware.py` flagged **84** cases at **1800 HU**. That is below anything published;
  the metal-segmentation literature validates **2500** (DSC 82.9%) and **3000** (DSC 84.2%).
- At 2500 HU, only **52** have a component clearing the 40-voxel floor. Not 3000: these
  series clip at **3071**, so a 3000 threshold measures the saturation plateau.
- Of those 52, a reader called **41 artefact**. The real count is **11**.

**Saturation does not separate implant from artefact**, which was the obvious test and
fails. Nearly every rejected proposal reaches the ceiling too, and several exceed it —
**11,798 HU on 0878, 9,534 on 0763, 7,438 on 0027**. Values above the ceiling are
reconstruction overshoot around something dense, not denser metal. What works is a real
surgical site **and** volume: every confirmed implant is ≥2,586 mm³, every rejection
≤1,768 mm³.

The search region must be the **whole labelled skeleton**, not the spine. A spine-only shell
missed every arthroplasty — on 0188 that is 174,302 saturated voxels reported as "no metal".
