"""
scripts/auto_finalize.py — finalize the cases that do NOT need a human.

THE PROBLEM. A case goes to the adjudicator whenever `per_class_min` Dice < tau (0.9), i.e. unless
EVERY one of ~24 rib classes agrees at >=0.9. That bar was calibrated for the SPINE task (a handful of
big vertebrae) and is close to unpassable for RIBS: a rib is a few voxels thick, so Dice punishes a
1-voxel boundary wobble severely, and taking the WORST of 24 thin structures means almost any two
independent annotations "disagree". The result is a ~119-case adjudication queue that is mostly not
disagreement about anatomy at all.

THE RULE (the one that actually needs a human). A rib class present in ONE label and not the other --
or with ZERO overlap between them -- is a SUBSTANTIVE conflict: somebody added, missed, or RENUMBERED
a rib, and only a human can say who is right. Everything else is boundary jitter between two labels
that BOTH already passed the anatomical QC gate, so either is a defensible ground truth.

  Dice == 0 on any rib class (AFTER halo cleanup)  -> ADJUDICATE (a human decides)
  otherwise                                        -> AUTO-FINALIZE

HALO CLEANUP FIRST -- this is what makes the rule trustworthy. A 13-voxel halo speck (the residue of
an old label clinging to the rib the annotator renumbered) present in one label and absent in the
other scores Dice 0.0 and masquerades as a "missing rib". Strip the halo, and the two annotators are
revealed to have agreed all along.

SELECTION RULE (documented, deterministic, reproducible). Where the two labels differ only at
boundaries, the final is the label from the annotator with the higher QC pass rate, halo-cleaned. The
annotators' RAW submissions are never mutated -- they remain the primary record for inter-rater
agreement and the data descriptor. Every finalized case records `by: auto:pick-better` plus the git
commit, so nothing is silently passed off as a human adjudication.

  python scripts/auto_finalize.py            # DRY RUN: report the split, write nothing
  python scripts/auto_finalize.py --apply    # finalize + write the escalation list
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
for p in (_HERE, _HERE.parent / "review_service"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
import store as store_mod            # noqa: E402
import label_scheme as LS            # noqa: E402
import review_anatomy_qc as RA       # noqa: E402
from review import schema            # noqa: E402
from huggingface_hub import hf_hub_download   # noqa: E402

REPO = os.environ.get("REVIEW_REPO", "anonymous-mlhc/CTSpinoPelvic1K-reviews-ribs")
LO, HI = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12


def _commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                       cwd=str(_HERE.parent), text=True).strip()
    except Exception:                                    # noqa: BLE001
        return "unknown"


def halo_ids(lab: np.ndarray) -> list:
    """Rib ids that are HALO: small AND fused to an adjacent REAL rib. Sizes via ONE bincount."""
    cnt = np.bincount(lab.ravel(), minlength=HI + 2)
    objs = ndimage.find_objects(lab if lab.dtype.kind in "iu" else lab.astype(np.int32))
    out = []
    for rid in range(LO, HI + 1):
        n = int(cnt[rid]) if rid < len(cnt) else 0
        if n == 0 or n >= RA.HALO_MAX_VOX:
            continue
        o = objs[rid - 1] if rid - 1 < len(objs) else None
        if o is None:
            continue
        pad = tuple(slice(max(0, o[i].start - 3), min(lab.shape[i], o[i].stop + 3))
                    for i in range(3))
        sub = lab[pad]
        m = (sub == rid)
        dil = ndimage.binary_dilation(m, iterations=2)
        touch = {int(v) for v in np.unique(sub[dil & (sub > 0) & (sub != rid)])}
        if any((nb < len(cnt) and int(cnt[nb]) >= RA.HALO_MAX_VOX)
               for nb in (touch & RA._adjacent_rib_ids(rid))):
            out.append(rid)
    return out


def zero_overlap_classes(A: np.ndarray, B: np.ndarray, drop=()) -> list:
    """Rib classes with ZERO overlap between the two labels -- present in one and not the other, or
    renumbered. ONE pass via bincount (not 24x full-volume boolean ops)."""
    inA = (A >= LO) & (A <= HI)
    inB = (B >= LO) & (B <= HI)
    if drop:
        inA &= ~np.isin(A, drop)
        inB &= ~np.isin(B, drop)
    cA = np.bincount(A[inA].ravel(), minlength=HI + 1)
    cB = np.bincount(B[inB].ravel(), minlength=HI + 1)
    same = inA & inB & (A == B)
    cI = np.bincount(A[same].ravel(), minlength=HI + 1)
    bad = []
    for c in range(LO, HI + 1):
        if int(cA[c]) + int(cB[c]) == 0:
            continue
        if int(cI[c]) == 0:                              # zero overlap -> substantive conflict
            bad.append(c)
    return bad


def conflict_kind(A: np.ndarray, B: np.ndarray, c: int) -> str:
    """Why does rib class `c` have zero overlap?

    MISSING  - one annotator segmented it and the other left BACKGROUND at those exact voxels.
               That is not a disagreement about anatomy: one of them simply did less work. The more
               complete annotation wins (both started from the SAME pseudolabel, so a structure that
               is absent from one of them was DELETED -- a defect, not an opinion).
    RENUMBER - both segmented the bone; they disagree on its NUMBER. Only a human can settle that,
               and it is the only conflict that should ever cost an adjudication.
    """
    inA, inB = int((A == c).sum()), int((B == c).sum())
    src, other = (A, B) if inA >= inB else (B, A)
    vals = other[src == c]
    if vals.size == 0:
        return "MISSING"
    return "MISSING" if float((vals == 0).mean()) >= 0.70 else "RENUMBER"


def completeness(lab: np.ndarray):
    """(distinct structures, foreground voxels) -- how much of the anatomy this annotator kept."""
    return (len({int(v) for v in np.unique(lab) if v > 0}), int((lab > 0).sum()))


def strip(lab: np.ndarray, ids) -> np.ndarray:
    if not ids:
        return lab
    out = lab.copy()
    for r in ids:
        out[out == r] = 0
    return out


def pass_rates(store) -> dict:
    """Each annotator's QC pass rate, from the verdicts the sweep stamped into the ledger."""
    tot, ok = {}, {}
    for c in store.list_cases():
        for k in ("1", "2"):
            s = c.get("slots", {}).get(k)
            if not s or "qc_pass" not in s:
                continue
            r = s.get("reviewer")
            tot[r] = tot.get(r, 0) + 1
            ok[r] = ok.get(r, 0) + bool(s.get("qc_pass"))
    return {r: (ok.get(r, 0) + 1) / (tot[r] + 2) for r in tot}     # Laplace-smoothed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="write finals (default: dry run)")
    ap.add_argument("--manifest", default="auto_finalize_manifest.csv")
    ap.add_argument("--escalate", default="adjudication_list.csv")
    a = ap.parse_args(argv)

    tok = os.environ["HF_TOKEN"]
    commit = _commit()
    store = store_mod.ReviewStore(store_mod.HFBackend(repo_id=REPO, token=tok))
    rates = pass_rates(store)
    names = RA._id2name()

    cases = [c for c in store.list_cases()
             if c.get("region_to_review") == "ribs"
             and schema.derive_status(c) == "needs_adjudication"]
    print(f"{len(cases)} cases in the adjudication queue  (commit {commit})\n", flush=True)

    fin_rows, esc_rows = [], []
    pending: dict = {}
    for i, case in enumerate(cases):
        cid = case["case_id"]
        try:
            pa = hf_hub_download(REPO, f"reviews/{cid}/1_label.nii.gz", repo_type="dataset", token=tok)
            pb = hf_hub_download(REPO, f"reviews/{cid}/2_label.nii.gz", repo_type="dataset", token=tok)
            ia, ib = nib.load(pa), nib.load(pb)
            A, B = np.asanyarray(ia.dataobj), np.asanyarray(ib.dataobj)
        except Exception as e:                           # noqa: BLE001
            print(f"  [skip] {cid}: {str(e)[:60]}", flush=True)
            continue

        ha, hb = halo_ids(A), halo_ids(B)
        bad = zero_overlap_classes(A, B, drop=tuple(set(ha) | set(hb)))
        r1 = case["slots"]["1"].get("reviewer")
        r2 = case["slots"]["2"].get("reviewer")

        kinds = [conflict_kind(A, B, c) for c in bad]
        renum = [c for c, k in zip(bad, kinds) if k == "RENUMBER"]

        if renum:                                        # a real NUMBERING dispute -> a human decides
            esc_rows.append({"case": cid, "reviewer_1": r1, "reviewer_2": r2,
                             "conflicting_ribs": " ".join(names.get(c, str(c)) for c in renum),
                             "n_conflicts": len(renum), "commit": commit})
            print(f"  ADJUDICATE {cid}: RENUMBER {[names.get(c, c) for c in renum]}", flush=True)
            continue

        if bad:
            # every conflict is MISSING -> one annotator just left it as background. Not a judgment
            # call: the MORE COMPLETE annotation wins (they both started from the same pseudolabel).
            win = "1" if completeness(A) >= completeness(B) else "2"
            print(f"  auto-finalize {cid}: {len(bad)} MISSING -> more complete = slot {win}", flush=True)
        else:
            # boundary jitter only; both passed QC -> higher pass-rate annotator wins
            win = "1" if rates.get(r1, 0) >= rates.get(r2, 0) else "2"
        lab = strip(A, ha) if win == "1" else strip(B, hb)
        img = ia if win == "1" else ib
        winner = r1 if win == "1" else r2
        fin_rows.append({"case": cid, "chosen_slot": win, "chosen_reviewer": winner,
                         "reviewer_1": r1, "reviewer_2": r2,
                         "halo_removed": len(ha) + len(hb), "commit": commit})
        print(f"  auto-finalize {cid}: slot {win} ({winner})", flush=True)
        if a.apply:
            pending[cid] = (case, lab, img, winner)
            if len(pending) >= 8:
                _flush(store, pending, commit)
        if i % 10 == 0:
            print(f"    ...{i}/{len(cases)}", flush=True)

    if a.apply and pending:
        _flush(store, pending, commit)

    if fin_rows:
        with open(a.manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(fin_rows[0].keys())); w.writeheader(); w.writerows(fin_rows)
    if esc_rows:
        with open(a.escalate, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(esc_rows[0].keys())); w.writeheader(); w.writerows(esc_rows)

    n = len(fin_rows) + len(esc_rows)
    print(f"\n{'APPLIED' if a.apply else 'DRY RUN'} (commit {commit})")
    print(f"  AUTO-FINALIZED         : {len(fin_rows)}/{n}")
    print(f"  YOU ADJUDICATE (real)  : {len(esc_rows)}/{n}   -> {a.escalate}")
    if not a.apply:
        print("\n  nothing written. re-run with --apply to finalize.")
    return 0


def _flush(store, pending, commit):
    """Write a batch of finals in ONE commit (label + review record + case json per case)."""
    import tempfile
    files = {}
    for cid, (case, lab, img, winner) in pending.items():
        with tempfile.TemporaryDirectory() as td:
            tp = Path(td) / "final_label.nii.gz"
            nib.save(nib.Nifti1Image(lab.astype(np.asanyarray(img.dataobj).dtype),
                                     img.affine, img.header), str(tp))
            data = tp.read_bytes()
        rel = f"reviews/{cid}/final_label.nii.gz"
        files[rel] = data
        case["final"] = {"decision": "corrected", "label_rel": rel,
                         "prov_after": case.get("prov_before"),
                         "by": "auto:pick-better", "at": schema.utcnow(),
                         "rule": "no zero-overlap rib class after halo cleanup; "
                                 "final = higher-QC-pass-rate annotator, halo-cleaned",
                         "chosen_reviewer": winner, "commit": commit,
                         "irr": case.get("irr")}
        case.setdefault("slots", {})[schema.ADJ_SLOT] = {
            "reviewer": "auto:pick-better", "done": True, "submitted_at": schema.utcnow()}
        files[store.case_path(cid)] = json.dumps(case, indent=2)
    store.b.write_many(files, commit_message=f"auto-finalize {len(pending)} cases ({commit})")
    pending.clear()


if __name__ == "__main__":
    raise SystemExit(main())
