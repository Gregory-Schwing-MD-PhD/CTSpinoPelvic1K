# Deferred cases — hand annotation required, do not batch

Cases pulled out of the normal review flow because an automated or semi-automated pass
would make them worse, not better. Each entry says what is wrong, why the usual tools fail
on it, and what would actually be needed. **Nothing here should be assigned as routine
work**, and none of these should be silently included in derived measurements until
resolved.

| case | token | deferred | blocker | in morphometrics? |
|---|---|---|---|---|
| 0068 | 46 | 2026-08-19 | interbody cage fuses L5–L6; boundary is implicit | **exclude** |

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
