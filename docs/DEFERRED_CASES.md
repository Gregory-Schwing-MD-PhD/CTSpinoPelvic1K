# Deferred cases — hand annotation required, do not batch

Cases pulled out of the normal review flow because an automated or semi-automated pass
would make them worse, not better. Each entry says what is wrong, why the usual tools fail
on it, and what would actually be needed. **Nothing here should be assigned as routine
work**, and none of these should be silently included in derived measurements until
resolved.

| case | token | deferred | blocker | in morphometrics? |
|---|---|---|---|---|
| 0068 | 46 | 2026-08-19 | interbody cage fuses L5–L6; boundary is implicit | **resolved 2026-08-26** — see below |

---

## 0068 — instrumented L5–L6, pseudolabelled spine, six lumbar bodies

**Status: DEFERRED. Requires hand annotation by someone who can read the construct.**

### What is wrong

Three problems stacked on one case, and each one makes the others harder.

**An interbody cage bridges the L5–L6 interspace.** Dense metal fills the disc space, so
there is no image gradient where the boundary between the two vertebrae should be. The
pseudolabel merged them into a single connected object. No segmenter — AI-assisted or
otherwise — can find that boundary, because it is not in the image. It has to be inferred
from anatomy and drawn by hand.

**The numbering is off by one below the merge.** The case has six free lumbar bodies. The
pseudolabel produced five labels, so what it calls L4–L5 is really L5–L6, and the merged
object holds the bottom two. Splitting it is not enough; everything above has to shift.

**The spine labels were never reviewed.** `prov_spine: pseudo`, `config: pelvic_native`,
`match_type: pelvic_only`. The pelvis was annotated by hand from CTPelvic1K; the vertebrae
are model output that no human has checked. `lstv_vertebral` is `None` — no spine-side LSTV
read exists for this case at all.

### Why the record disagrees with the anatomy

`lstv_label: NORMAL`, inherited from `lstv_pelvic: NORMAL` — a judgement made from pelvic
morphology, without counting lumbar vertebrae. The case has six free lumbar bodies and an
instrumented lumbosacral junction. The label is not wrong so much as answering a different
question, which is the same pattern seen on 0008 and 0033.

### Why it must be excluded from the morphometrics until fixed

**An iatrogenic fusion is indistinguishable from a congenital one to a distance
measurement.** The transitional-anatomy analysis measures gaps between the lowest lumbar
vertebra and the sacrum, and a cage-bridged interspace reads as "no gap" exactly like a
congenitally fused transitional vertebra. Left in, this case can contribute a false
positive to the co-occurrence result.

The concern generalises: any instrumented case still unlabelled in the release carries the
same risk, and there is currently no field in the manifest recording surgical hardware. A
metal threshold scan over the 802 would give the count and the case list; until that is
run, the instrumented population is unknown.

### What was added because of this case

The label scheme gained a hardware block, since instrumentation is neither bone nor any
existing class, and `ignore` would have erased the information rather than recording it:

    76  hardware             instrumentation, subtype not distinguished
    77  hardware_cage        interbody cage / spacer
    78  hardware_screw_rod   pedicle screws and rods
    79  hardware_plate       plates and other fixation

Subtypes are named where identifiable: collapsing `cage` into generic `hardware` later is a
one-line merge, whereas splitting a generic label back into subtypes means revisiting every
case that used it.

### What would be needed to resolve it

1. Split the merged L5/L6 at the anatomic disc position — hand-drawn, with the ITK-SNAP
   paintbrush constrained to *paint over* the merged label so nothing else can be damaged.
2. Renumber the lumbar run to L1–L6, anchored at the sacrum and counted up.
3. Label the cage as 77, and any screws or rods as 78. Metal thresholds cleanly at 2000+ HU
   and should not be painted by hand.
4. Record whether the L5–L6 fusion is surgical only, or surgical *on top of* a pre-existing
   transitional vertebra — those are different findings and only a reader can separate them.

### State of the file

The label on disk is untouched — nothing was saved during the session that deferred it.

---

## 0068 — resolved 2026-08-26

Everything the deferred entry said about the anatomy held up. Six free lumbar bodies, five
labels, the top two merged under one label, and the cage at L5–L6 — all confirmed by
measurement. The entry above is left as written; this records what was done.

### The count, settled by two independent anchors

The deferred entry proposed counting **up from the sacrum**. The renumbering used the
**twelfth rib** instead: `rib_left_12` and `rib_right_12` both articulate with one vertebra
at about 5 mm, and the vertebra bearing the twelfth rib is T12 by definition. Both anchors
give **six free lumbar bodies**, which is worth more than either alone — this is a case
where counting from the top is exactly what cannot be done, because the scan starts
mid-thoracic.

`scripts/split_and_renumber_0068.py` finds how many bodies each label covers by restricting
to the anterior column, where the disc is a real gap and the posterior elements do not run
continuously past it. It reported five labels over six bodies with the merge at L1, split it,
and shifted everything below:

    L1 -> L1 + L2 ;  L2 -> L3 ;  L3 -> L4 ;  L4 -> L5 ;  L5 -> L6

The merged label is split by **nearest body**, not by a flat plane at the disc: a plane cuts
the laminae at an arbitrary height and leaves half of one on the wrong vertebra. The two
bodies become seeds and every voxel of the merged label joins the nearer one, so the
posterior elements follow their own body.

### The hardware, read rather than assumed

Two **threaded cylindrical interbody cages**, hollow, 27 × 14 × 14 mm each, screwed into the
L5–L6 disc space side by side and symmetric about the midline — BAK/Ray type. Labelled `77`.

**There is no posterior instrumentation anywhere in the volume.** A whole-volume census
found only two dense objects touching the skeleton; every other dense component is 24–75 mm
away from any labelled bone, which is tagged stool from the colonography prep saturating the
scanner exactly as titanium does. No screws, no rods, so nothing is labelled `78`. That is
not hardware missing from the scan — standalone threaded cages without posterior fixation is
the technique this implant was used with.

The cage voxels were **taken back from the vertebrae that held them**: the bone segmenter had
absorbed the metal into L5 and L6, so 100% of the cage sat inside an existing label and
naming it meant a subtraction, not an addition.

Threshold: **2500 HU**, the lower of the two values validated in the metal-segmentation
literature. Not 3000, which scores marginally better in that literature but on this series
would measure the saturation plateau — these images clip at 3071 HU, so a 3000 threshold
keeps only what is within 71 HU of the ceiling.

### The thoracic levels that were never labelled

The case carried no thoracic vertebra at all while imaging 48 mm of column above the
lumbar spine. T12, T11 and a truncated T10 were added. Keeping the truncated ones is the
release convention, not a judgement: 553 of 802 records (69%) have their top vertebra cut by
the field of view, with a median labelled height of 13.6 mm and a range down to 0.8 mm.

### Still open — needs a reader, not a script

Point 4 of the original list stands: **whether the L5–L6 fusion is surgical only, or surgical
on top of a pre-existing transitional vertebra.** Those are different findings and no
measurement separates them. `lstv_vertebral` remains unread for this case.

### Manifest

`hardware`, `hardware_components`, `hardware_mm3` and `hardware_bridges_interspace` now exist
for all 802 records, populated from the metal scan — the instrumented population is **84
cases**, which the deferred entry asked for and could not have. The read-level fields
(`hardware_type`, `hardware_level`, `fusion`) are populated for 0068 only; a null there means
**not yet read**, never "no hardware". `has_l6` and `n_lumbar_labels` are corrected to
`true` and `6`.

