"""scripts/six_lumbar_readings.py — the three readings of a six-lumbar spine, scored from the labels.

Six rib-free bodies above the sacrum can be (A) a true sixth lumbar vertebra, (B) a T12 whose
ribs are aplastic so a thoracic body is counted as lumbar, or (C) a lumbarized S1, a sacral
segment that separated from the sacrum. The count is the same in all three. The morphology is
not, and the released labels carry the morphology:

  reading B leaves its evidence at the TOP of the rib-free run: a missing or aplastic twelfth
            rib on a body that is otherwise thoracic;
  reading C leaves its evidence at the BOTTOM: a transverse process that is broad and tall and
            sits on the ilium like an ala, a thin disc beneath it, and a sacrum that is one
            segment short;
  reading A leaves no evidence at either end: the junction looks like a normal L5 junction and
            the sacrum is full height.

Every feature is expressed as a z-score against the 644 released records with five rib-free
bodies, no transitional flag and no hardware, so "normal" is the corpus's own distribution
rather than a textbook value. The scores are reported per record, with the call the rules
make and the Castellvi grade the readers gave, so a reader can disagree with a number.

    python scripts/six_lumbar_readings.py --out results/six_lumbar_readings

Inputs: morphometrics/transition_morphometrics.csv (per-record junction and rib geometry,
extracted from the labels) and the release manifest. No volume is read.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics as st
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
FEATURES = {
    # feature: (sign toward the SACRAL-type reading, description)
    "tp_height_max_mm":     (+1, "transverse-process height, larger toward an ala"),
    "tp_gap_ilium_min_mm":  (-1, "process-to-ilium gap, smaller toward contact"),
    "disc_low_mm":          (-1, "disc beneath the lowest body, thinner toward fusion"),
    "sacrum_height_mm":     (-1, "sacral height, shorter when a segment has left it"),
}
RIB_FEATURES = ["rib12_11_ratio_min", "n_ribs_left", "n_ribs_right"]
STUMP_RATIO = 0.33          # the paper's stump-rib threshold on the 12th:11th length ratio


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--morph", default=str(HERE / "morphometrics/transition_morphometrics.csv"))
    ap.add_argument("--manifest", default=str(HERE / "data/zenodo_deposit/manifest.json"))
    ap.add_argument("--out", default=str(HERE / "results/six_lumbar_readings"))
    ap.add_argument("--sacral-threshold", type=float, default=1.5,
                    help="mean sacral-type z above which the bottom body reads as a lumbarized S1")
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = {r["case"]: r for r in csv.DictReader(open(a.morph, encoding="utf-8"))}
    for r in rows.values():
        g = [fnum(r.get("tp_gap_ilium_left_mm")), fnum(r.get("tp_gap_ilium_right_mm"))]
        g = [x for x in g if x is not None]
        r["tp_gap_ilium_min_mm"] = min(g) if g else ""
    man = json.load(open(a.manifest, encoding="utf-8"))
    case_of = lambda rec: rec["label_file"].split("/")[-1].split("_")[0]
    hw = {case_of(x) for x in man if x.get("hardware_labelled")}

    normal = [r for r in rows.values()
              if r.get("n_non_rib_bearing") == "5.0" and r.get("lstv_vertebral") == "NORMAL"
              and r.get("lstv_pelvic") == "NORMAL" and r["case"].zfill(4) not in hw]
    ref = {}
    for c in list(FEATURES) + RIB_FEATURES:
        v = sorted(x for x in (fnum(r.get(c)) for r in normal) if x is not None)
        q1, q3 = v[int(0.25 * len(v))], v[int(0.75 * len(v))]
        ref[c] = (st.median(v), max((q3 - q1) / 1.349, 1e-6))

    six = [x for x in man if x.get("has_l6")]
    table = []
    for rec in sorted(six, key=case_of):
        c = case_of(rec)
        r = rows.get(str(int(c)), {})
        z = {}
        for feat in list(FEATURES) + RIB_FEATURES:
            v = fnum(r.get(feat))
            z[feat] = None if v is None else (v - ref[feat][0]) / ref[feat][1]
        sacral = [FEATURES[f][0] * z[f] for f in FEATURES if z[f] is not None]
        sacral_score = st.mean(sacral) if sacral else None
        ratio = fnum(r.get("rib12_11_ratio_min"))
        rib_pairs = min(fnum(r.get("n_ribs_left")) or 0, fnum(r.get("n_ribs_right")) or 0)
        top_thoracic = (ratio is not None and ratio < STUMP_RATIO)   # a stump or aplastic 12th rib
        # A bilaterally fused process (Castellvi III/IV) is carried inside the sacrum label, so
        # the process features read as normal exactly where fusion is complete
        # (docs/CASTELLVI_SCREEN_BLIND_SPOT.md). There the sacrum's own height is the evidence:
        # a sacrum that has lost its first segment is short.
        short_sacrum = z["sacrum_height_mm"] is not None and z["sacrum_height_mm"] <= -2.0
        if top_thoracic:
            reading = "B: T12 with an aplastic or stump twelfth rib"
        elif (sacral_score is not None and sacral_score >= a.sacral_threshold) or short_sacrum:
            reading = "C: lumbarized S1"
        elif sacral_score is not None:
            reading = "A: true L6"
        else:
            reading = "not measurable"
        table.append({"case": c, "castellvi": rec.get("castellvi_type") or "",
                      "lstv_vertebral": r.get("lstv_vertebral", ""), "lstv_pelvic": r.get("lstv_pelvic", ""),
                      "rib_pairs_in_fov": int(rib_pairs), "rib12_11_ratio_min": ratio,
                      **{f"z_{f}": (None if z[f] is None else round(z[f], 2)) for f in FEATURES},
                      "sacral_type_score": None if sacral_score is None else round(sacral_score, 2),
                      "reading": reading})

    with open(out / "six_lumbar_readings.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0]))
        w.writeheader(); w.writerows(table)

    counts = {}
    for t in table:
        counts[t["reading"]] = counts.get(t["reading"], 0) + 1
    lines = [f"normal reference: {len(normal)} records (five rib-free bodies, no transitional flag, no hardware)",
             f"six-lumbar records: {len(table)}",
             "reference medians: " + ", ".join(f"{k} {ref[k][0]:.1f}" for k in FEATURES),
             "", "readings:"] + [f"  {k}: {v}" for k, v in sorted(counts.items())] + [
             "", f"{'case':5s} {'Castellvi':9s} {'ribs':4s} {'r12/11':6s} {'TPh':6s} {'gap':6s} {'disc':6s} {'sacH':6s} {'score':6s}  reading"]
    for t in table:
        zz = [t[f"z_{f}"] for f in FEATURES]
        lines.append(f"{t['case']:5s} {t['castellvi']:9s} {t['rib_pairs_in_fov']:<4d} "
                     f"{'' if t['rib12_11_ratio_min'] is None else f'{t['rib12_11_ratio_min']:.2f}':6s} "
                     + " ".join(f"{'' if v is None else f'{v:+.1f}':6s}" for v in zz)
                     + f" {'' if t['sacral_type_score'] is None else f'{t['sacral_type_score']:+.2f}':6s}  {t['reading']}")
    (out / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    # ---- figure: the junction evidence, one point per six-lumbar record --------------------
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(8.4, 3.4))
    xs = [t["z_tp_gap_ilium_min_mm"] for t in table]
    ys = [t["z_tp_height_max_mm"] for t in table]
    ss = [t["z_sacrum_height_mm"] for t in table]
    col = ["#b5533c" if t["reading"].startswith("C") else "#2f6f8f" for t in table]
    ax[0].axhline(0, color="0.8", lw=0.8); ax[0].axvline(0, color="0.8", lw=0.8)
    ax[0].scatter(xs, ys, c=col, s=34, edgecolor="k", linewidth=0.4)
    for t, x, y in zip(table, xs, ys):
        if x is not None and y is not None:
            ax[0].annotate(t["case"], (x, y), fontsize=6.5, xytext=(3, 2), textcoords="offset points")
    ax[0].set_xlabel("process-to-ilium gap, z against normal L5")
    ax[0].set_ylabel("transverse-process height, z")
    ax[0].set_title("(a) The bottom body's junction")
    ax[0].grid(True, alpha=0.3)
    sc = [t["sacral_type_score"] for t in table]
    order = sorted(range(len(table)), key=lambda i: (sc[i] if sc[i] is not None else -9))
    ax[1].barh([table[i]["case"] for i in order], [sc[i] or 0 for i in order],
               color=[col[i] for i in order], edgecolor="k", linewidth=0.4)
    ax[1].axvline(a.sacral_threshold, color="k", ls="--", lw=0.8)
    ax[1].set_xlabel("sacral-type score (mean signed z of four junction features)")
    ax[1].set_title("(b) Reading C above the line, reading A below")
    ax[1].grid(True, axis="x", alpha=0.3)
    ax[1].tick_params(axis="y", labelsize=7)
    fig.tight_layout()
    fig.savefig(out / "six_lumbar_readings.png", dpi=200)
    fig.savefig(out / "six_lumbar_readings.pdf")
    print("wrote", out / "six_lumbar_readings.csv", out / "six_lumbar_readings.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
