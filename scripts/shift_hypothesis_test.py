"""scripts/shift_hypothesis_test.py — is a six-lumbar spine better explained with every name shifted up one?

THE HOLE THIS CLOSES. The rib check in six_lumbar_readings.py cannot see an aplastic
twelfth rib, because the labels define the lowest rib-bearing vertebra as T12: a spine whose
true T12 has no ribs gets its T11 ribs called "rib 12", its T12 called "L1", and a sixth
"lumbar" body at the bottom, and every rib ratio looks normal. So the question has to be put
to the bodies themselves, level by level, as Greg asked: does the column fit its labels as
given (hypothesis A, the sixth body is a true L6), or does it fit better with every name
moved up one (hypothesis B, the labelled L1 is really a rib-less T12 and the labelled L6 is
really L5)?

Each vertebra is described by the same size-free shape features the separability test
uses (each body measure over the patient's own median across levels, plus six ratios), and
each named level T11..L5 gets a diagonal Gaussian fitted on the normal five-lumbar records.
A record's log-likelihood is summed over its labelled T12..L6 under both naming maps; the
labelled L6 is scored as a lowest lumbar body (L5 model) under A and as L5 under B, so the
decisive terms are the labelled T12 (T12 model against T11) and the labelled L1 (L1 model
against T12), which is where a rib-less T12 would betray itself. The same score on normal
records calibrates what "fits as labelled" looks like.

    python scripts/shift_hypothesis_test.py --out results/six_lumbar_readings
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
import extract_level_gradients as ELG  # noqa: E402

BASE = ["endplate_width_{}_mm", "body_height_{}_mm", "body_height_post_{}_mm",
        "canal_width_{}_mm", "tp_span_{}_mm"]
NAMED = ["T11", "T12", "L1", "L2", "L3", "L4", "L5"]           # levels with a reference model
MAP_A = {"T12": "T12", "L1": "L1", "L2": "L2", "L3": "L3", "L4": "L4", "L5": "L5", "L6": "L5"}
MAP_B = {"T12": "T11", "L1": "T12", "L2": "L1", "L3": "L2", "L4": "L3", "L5": "L4", "L6": "L5"}


def num(r, k):
    try:
        v = float(r.get(k, ""))
        return v if v > 0 else None
    except (TypeError, ValueError):
        return None


def feats(r, level, med):
    vals = {b: num(r, b.format(level)) for b in BASE}
    if any(v is None for v in vals.values()):
        return None
    ew, bh, bhp = vals["endplate_width_{}_mm"], vals["body_height_{}_mm"], vals["body_height_post_{}_mm"]
    cw, tp = vals["canal_width_{}_mm"], vals["tp_span_{}_mm"]
    return np.array([vals[b] / med[b] for b in BASE] +
                    [tp / ew, cw / ew, bh / ew, bh / bhp, tp / bh, cw / bh], float)


def case_median(r, levels):
    med = {}
    for b in BASE:
        v = [num(r, b.format(l)) for l in levels]
        v = [x for x in v if x is not None]
        if len(v) < 3:
            return None
        med[b] = float(np.median(v))
    return med


def loglik(x, model):
    mu, sd = model
    return float(-0.5 * np.sum(((x - mu) / sd) ** 2) - np.sum(np.log(sd)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gradients", default=str(ROOT / "morphometrics/level_gradients.csv"))
    ap.add_argument("--transition", default=str(ROOT / "morphometrics/transition_morphometrics.csv"))
    ap.add_argument("--manifest", default=str(ROOT / "data/zenodo_deposit/manifest.json"))
    ap.add_argument("--labels", default=str(ROOT / "data/zenodo_deposit/labels"))
    ap.add_argument("--out", default=str(ROOT / "results/six_lumbar_readings"))
    ap.add_argument("--controls", type=int, default=40, help="normal records scored the same way, for scale")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    man = json.load(open(a.manifest, encoding="utf-8"))
    case_of = lambda rec: rec["label_file"].split("/")[-1].split("_")[0]
    hw = {case_of(x) for x in man if x.get("hardware_labelled")}
    six = sorted(case_of(x) for x in man if x.get("has_l6"))

    trans = {r["case"].zfill(4): r for r in csv.DictReader(open(a.transition, encoding="utf-8"))}
    grad = {r["case"].zfill(4): r for r in csv.DictReader(open(a.gradients, encoding="utf-8"))}
    normal = [c for c, r in trans.items()
              if r.get("n_non_rib_bearing") == "5.0" and r.get("lstv_vertebral") == "NORMAL"
              and r.get("lstv_pelvic") == "NORMAL" and c not in hw and c in grad]

    # ---- reference: one diagonal Gaussian per named level, on normal records ---------------
    per_level = {l: [] for l in NAMED}
    for c in normal:
        r = grad[c]
        med = case_median(r, NAMED)
        if med is None:
            continue
        for l in NAMED:
            f = feats(r, l, med)
            if f is not None:
                per_level[l].append(f)
    models = {}
    for l, rows in per_level.items():
        X = np.asarray(rows)
        models[l] = (X.mean(0), np.maximum(X.std(0, ddof=1), 0.02))
    print("reference:", {l: len(v) for l, v in per_level.items()})

    # ---- measure the six-lumbar records afresh, with T10, L6 and T13 included ---------------
    ELG.LUMBAR = ELG.LEVELS = {17: "T10", 18: "T11", 19: "T12", 20: "L1", 21: "L2", 22: "L3",
                               23: "L4", 24: "L5", 25: "L6", 28: "T13"}
    measured = {}
    for c in six:
        p = Path(a.labels) / f"{c}_label.nii.gz"
        if p.exists():
            measured[c] = ELG.one(str(p))
    print(f"measured {len(measured)} six-lumbar records")

    def score(r, levels_present):
        med = case_median(r, [l for l in levels_present if l != "T10" and l != "T13"])
        if med is None:
            return None
        terms = {}
        for lab in levels_present:
            if lab not in MAP_A or lab not in MAP_B:
                continue
            f = feats(r, lab, med)
            if f is None:
                continue
            terms[lab] = (loglik(f, models[MAP_A[lab]]), loglik(f, models[MAP_B[lab]]))
        return terms

    rows = []
    for c in six:
        r = measured.get(c)
        if r is None or "error" in r:
            continue
        present = [l for l in ["T10", "T11", "T12", "L1", "L2", "L3", "L4", "L5", "L6", "T13"]
                   if num(r, f"body_height_{l}_mm") is not None]
        terms = score(r, present)
        if not terms:
            continue
        dA = sum(v[0] for v in terms.values()); dB = sum(v[1] for v in terms.values())
        rows.append({"case": c, "group": "six-lumbar", "levels": " ".join(present),
                     "logLik_A_as_labelled": round(dA, 1), "logLik_B_shift_up": round(dB, 1),
                     "delta_A_minus_B": round(dA - dB, 1),
                     **{f"d_{lab}": round(v[0] - v[1], 1) for lab, v in terms.items()},
                     "verdict": "A (as labelled)" if dA - dB > 0 else "B (shift up: labelled L1 is a rib-less T12)"})

    # ---- controls: normal records scored the same way from the existing table --------------
    rng = np.random.default_rng(0)
    ctrl = list(rng.choice(normal, size=min(a.controls, len(normal)), replace=False))
    for c in ctrl:
        r = grad[c]
        present = [l for l in NAMED if num(r, f"body_height_{l}_mm") is not None]
        terms = score(r, present)
        if not terms:
            continue
        dA = sum(v[0] for v in terms.values()); dB = sum(v[1] for v in terms.values())
        rows.append({"case": c, "group": "normal control", "levels": " ".join(present),
                     "logLik_A_as_labelled": round(dA, 1), "logLik_B_shift_up": round(dB, 1),
                     "delta_A_minus_B": round(dA - dB, 1),
                     **{f"d_{lab}": round(v[0] - v[1], 1) for lab, v in terms.items()},
                     "verdict": "A" if dA - dB > 0 else "B"})

    keys = ["case", "group", "levels", "logLik_A_as_labelled", "logLik_B_shift_up", "delta_A_minus_B",
            "d_T12", "d_L1", "d_L2", "d_L3", "d_L4", "d_L5", "d_L6", "verdict"]
    with open(out / "shift_hypothesis.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore"); w.writeheader(); w.writerows(rows)

    six_rows = [x for x in rows if x["group"] == "six-lumbar"]
    ctl_rows = [x for x in rows if x["group"] == "normal control"]
    cd = np.array([x["delta_A_minus_B"] for x in ctl_rows])
    lines = [f"per-level reference models from {len(normal)} normal records",
             f"normal controls (n={len(cd)}): delta(A-B) median {np.median(cd):+.1f}, 5th pct {np.percentile(cd, 5):+.1f}, "
             f"min {cd.min():+.1f}; {int((cd > 0).sum())}/{len(cd)} favour their own labels",
             "", "six-lumbar records: positive delta = fits as labelled (true L6); negative = fits shifted up (rib-less T12)",
             f"{'case':5s} {'delta':7s} {'T12':6s} {'L1':6s} {'L2':6s} {'L3':6s} {'L4':6s} {'L5':6s} {'L6':6s}  verdict"]
    for x in six_rows:
        lines.append(f"{x['case']:5s} {x['delta_A_minus_B']:+7.1f} " +
                     " ".join(f"{x.get(f'd_{l}', ''):>6}" for l in ["T12", "L1", "L2", "L3", "L4", "L5", "L6"]) +
                     f"  {x['verdict']}")
    nB = sum(1 for x in six_rows if x["verdict"].startswith("B"))
    lines += ["", f"verdicts: {len(six_rows) - nB} as labelled, {nB} shift up"]
    text = "\n".join(lines)
    (out / "shift_hypothesis_report.txt").write_text(text + "\n", encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
