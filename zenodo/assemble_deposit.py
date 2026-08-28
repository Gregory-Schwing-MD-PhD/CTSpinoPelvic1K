"""zenodo/assemble_deposit.py — build the deposit directory and check it before upload.

WHAT GOES UP IS THE LABELS AND THE CROSSWALK, NOT THE IMAGES. 193 GB of CT against 1.8 GB of
labels, and the CT is already in TCIA. What the source collections never published is the
mapping from each annotation to the series it was drawn on, and manifest.json carries it as
SeriesInstanceUIDs. A labels-plus-manifest deposit is not a reduced version of the dataset;
it is the part that did not exist before.

Everything here is a check that would otherwise be discovered by a user after download:

  * every label file named in the manifest exists, and nothing extra is present
  * every record carries at least one TCIA series identifier, or the crosswalk is not
    actually published for it
  * the Castellvi grades are populated, since the schema declared them null for a whole
    release cycle
  * no sided pair is transposed anywhere -- three records shipped with left_hip and
    right_hip swapped and the release QC could not see it, because it tested ribs only
  * the manifest declares no field it leaves empty in every record, which is the failure
    that produced the null castellvi_type in the first place

    python zenodo/assemble_deposit.py --check          # verify, build nothing
    python zenodo/assemble_deposit.py --build DIR      # verify, then assemble
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import multiprocessing
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import nibabel as nib

HIP_L, HIP_R, FEM_L, FEM_R = 30, 31, 32, 33
RIB_L = set(range(34, 46)) | {74}
RIB_R = set(range(46, 58)) | {75}


def rec_id(r):
    return Path(str(r.get("label_file", ""))).name.split("_")[0]


# How much of a sided label may lie on the far side of the midplane between the two before
# it is a laterality error rather than anatomy. The pubic symphysis and midline rib tips put
# a little across; half a bone wearing the wrong label puts about half across.
CROSSOVER = 0.25


def check_sidedness(path):
    """-> list of (pair name, what is wrong) for one volume.

    Two different faults, and the second one shipped once already:

      TRANSPOSED -- the two labels are swapped outright, so their centroids are in the wrong
      order along the left-right axis.

      CROSSOVER -- the labels are in the right order but a large piece of one carries the
      other's identifier. 1035 shipped this way: a 476,756-voxel piece of the right hip was
      labelled left, which pulled the left centroid toward the midline without pushing it
      past the right one. A centroid-order test cannot see that, so measure how much of each
      label sits beyond the plane midway between the two centroids.
    """
    img = nib.load(str(path))
    lab = np.asanyarray(img.dataobj)
    codes = nib.aff2axcodes(img.affine)
    lr = [i for i, k in enumerate(codes) if k in ("L", "R")]
    if not lr:
        return []
    ax = lr[0]
    out = []
    for name, li, ri in (("ribs", RIB_L, RIB_R), ("hips", {HIP_L}, {HIP_R}),
                         ("femurs", {FEM_L}, {FEM_R})):
        lm = np.isin(lab, list(li))
        rm = np.isin(lab, list(ri))
        if not (lm.any() and rm.any()):
            continue
        lpos = np.argwhere(lm)[:, ax]
        rpos = np.argwhere(rm)[:, ax]
        lc, rc = float(lpos.mean()), float(rpos.mean())
        if (lc >= rc) if codes[ax] == "R" else (lc <= rc):
            out.append((name, "transposed"))
            continue
        mid = (lc + rc) / 2.0
        # whichever side each label is meant to be on, count what is past the midplane
        l_over = float((lpos > mid).mean() if lc < rc else (lpos < mid).mean())
        r_over = float((rpos < mid).mean() if lc < rc else (rpos > mid).mean())
        worst = max(l_over, r_over)
        if worst >= CROSSOVER:
            side = "left" if l_over >= r_over else "right"
            out.append((name, f"{worst:.0%} of the {side} label is on the other side"))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="data/hf_export_v5")
    ap.add_argument("--extras", default="zenodo")
    ap.add_argument("--build", default=None, help="assemble into this directory")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sidedness", type=int, default=0,
                    help="verify sidedness on N records (0 = skip, -1 = all). Slow.")
    a = ap.parse_args()

    src = Path(a.src)
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    recs = man if isinstance(man, list) else man.get("records", list(man.values()))
    print(f"  manifest: {len(recs)} record(s)")

    fail = []

    # --- files line up with the manifest, in both directions ---------------------------
    want = {rec_id(r) for r in recs}
    have = {p.name.split("_")[0] for p in (src / "labels").glob("*_label.nii.gz")}
    missing, extra = sorted(want - have), sorted(have - want)
    print(f"  labels on disk: {len(have)}")
    if missing:
        fail.append(f"{len(missing)} manifest record(s) have no label file: {missing[:5]}")
    if extra:
        fail.append(f"{len(extra)} label file(s) are not in the manifest: {extra[:5]}")

    # --- the crosswalk is the point of the deposit; it must be complete ----------------
    no_uid = [rec_id(r) for r in recs
              if not (str(r.get("spine_series_uid") or "").strip()
                      or str(r.get("pelvic_series_uid") or "").strip())]
    print(f"  records with a TCIA series identifier: {len(recs) - len(no_uid)}/{len(recs)}")
    if no_uid:
        fail.append(f"{len(no_uid)} record(s) carry no series identifier, so the crosswalk "
                    f"is not published for them: {no_uid[:5]}")

    # --- declared-but-always-empty fields ----------------------------------------------
    keys = sorted({k for r in recs for k in r})
    empty = [k for k in keys
             if all(r.get(k) in (None, "", [], {}) for r in recs)]
    if empty:
        fail.append(f"field(s) declared in every record and populated in none: {empty}. "
                    f"A schema that promises a field it never fills reads as an annotation "
                    f"layer to anyone who lists the columns.")

    # --- the transitional layer ---------------------------------------------------------
    grades = Counter(str(r.get("castellvi_type")) for r in recs
                     if str(r.get("castellvi_type") or "").strip())
    print(f"  Castellvi grades populated: {sum(grades.values())} record(s) {dict(grades)}")
    if not grades:
        fail.append("no Castellvi grades are populated; the README describes a layer that "
                    "the manifest does not contain")

    lstv = Counter(str(r.get("lstv_label")) for r in recs)
    print(f"  LSTV labels: {dict(lstv)}")

    # --- sidedness, optional because it reads every volume -------------------------------
    if a.sidedness:
        files = sorted((src / "labels").glob("*_label.nii.gz"))
        if a.sidedness > 0:
            files = files[: a.sidedness]
        workers = min(len(files), max(1, (os.cpu_count() or 2) - 1))
        print(f"  checking sidedness on {len(files)} volume(s) "
              f"across {workers} process(es)...")
        bad = []
        if workers == 1:
            results = ((f, check_sidedness(f)) for f in files)
        else:
            pool = multiprocessing.Pool(workers)
            results = zip(files, pool.imap(check_sidedness, files, chunksize=4))
        for i, (f, t) in enumerate(results, 1):
            if t:
                bad.append((f.name.split("_")[0], t))
            if i % 100 == 0:
                print(f"    {i}/{len(files)}", flush=True)
        if workers > 1:
            pool.close()
            pool.join()
        swapped = [(c, f) for c, f in bad if any(w == "transposed" for _, w in f)]
        crossed = [(c, f) for c, f in bad if all(w != "transposed" for _, w in f)]
        for cid, faults in bad:
            for pair, why in faults:
                print(f"    {cid}  {pair}: {why}")
        if swapped:
            # a centroid ordering cannot come out wrong on real anatomy
            fail.append(f"{len(swapped)} record(s) have a transposed sided pair: "
                        f"{[c for c, _ in swapped][:5]}")
        if crossed:
            # a proportion past a threshold can; print it and let a person decide
            print(f"  sidedness: {len(crossed)} record(s) exceed the {CROSSOVER:.0%} "
                  f"crossover threshold -- review, not a blocker")
        if not bad:
            print("  sidedness: every checked pair is correctly sided, "
                  "in order and without crossover")

    # --- verdict -------------------------------------------------------------------------
    print()
    if fail:
        print("  NOT READY:")
        for f in fail:
            print(f"    - {f}")
        return 1
    print("  every check passed")

    if not a.build:
        return 0

    out = Path(a.build)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    for p in sorted((src / "labels").glob("*_label.nii.gz")):
        shutil.copy(p, out / "labels" / p.name)
    for name in ("manifest.json", "splits_5fold.json", "dataset_labels.json"):
        if (src / name).exists():
            shutil.copy(src / name, out / name)
    # KNOWN_ISSUES.md ships WITH the data, not beside it in a repository. Every entry
    # in it is a filter somebody has to apply before a particular analysis -- ungraded
    # is not negative, prone and supine must not be pooled, instrumented cases must
    # leave any gap measurement -- and a caveat that arrives separately from the files
    # is a caveat that arrives too late.
    # fetch_from_tcia.py SHIPS TOO. The README lists it in the table of files in this
    # deposit and tells the reader to run it; leaving it out means the deposit
    # documents a tool it does not contain, and the rebuild path -- the entire reason
    # a labels-only deposit is usable -- has no starting point.
    for name in ("README.md", "KNOWN_ISSUES.md", "fetch_from_tcia.py",
                 "reconstruct_ct.py", "LICENSE"):
        if (Path(a.extras) / name).exists():
            shutil.copy(Path(a.extras) / name, out / name)

    # A CHECKSUM MANIFEST, because a 1.8 GB download that silently truncates looks exactly
    # like one that completed. Zenodo checksums the archive it stores; this lets someone
    # verify the files they actually ended up with, offline, against a list they can cite.
    sums = out / "SHA256SUMS.txt"
    files = sorted((q for q in out.rglob("*") if q.is_file() and q != sums),
                   key=lambda q: str(q.relative_to(out)).replace("\\", "/"))
    with open(sums, "w", encoding="utf-8", newline="\n") as fh:
        for q in files:
            h = hashlib.sha256()
            with open(q, "rb") as src_fh:
                for chunk in iter(lambda: src_fh.read(1 << 20), b""):
                    h.update(chunk)
            rel = str(q.relative_to(out)).replace("\\", "/")
            fh.write(f"{h.hexdigest()}  {rel}\n")
    print(f"  wrote {sums.name} covering {len(files)} file(s)")
    print("  verify with:  sha256sum -c SHA256SUMS.txt")

    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    n = len(list((out / "labels").glob("*.nii.gz")))
    print(f"\n  assembled {out}: {n} labels, {total / 1e9:.2f} GB total")
    print("  Zenodo's default per-record limit is 50 GB; this fits without a quota request.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
