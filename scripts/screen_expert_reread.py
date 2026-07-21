"""
scripts/screen_expert_reread.py — flag annotations that need a RADIOLOGIST's eyes, so students never
have to make a transitional-level call. Over all of v4 (EDT-free, bbox-based):

  * L6 present (id 25)                 -> lumbarization / 6th lumbar, OR a mis-count -> EXPERT decides
  * a rib whose head is on L1          -> FULL rib (> STUMP_MM: L1 should NEVER carry a full rib, so this
                                          is almost always a numbering error) or STUMP rib (<= STUMP_MM:
                                          a lumbar rib / transitional variant). Either way -> EXPERT.

Students fix the mechanical class-mixing splits; anything this screen flags is routed to you instead.

Output: docs/lstv_reread_queue.csv  (token, config, signal, detail)

    HF_TOKEN=... HF_HUB_OFFLINE=1 V4_REV=<cached commit> python scripts/screen_expert_reread.py [--limit N]
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np, nibabel as nib
from scipy import ndimage

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
import label_scheme as LS            # noqa: E402
from huggingface_hub import hf_hub_download   # noqa: E402

DS = os.environ.get("V2_REPO", "anonymous-mlhc/CTSpinoPelvic1K")
REV = os.environ.get("V4_REV", "v4")
OFFLINE = os.environ.get("HF_HUB_OFFLINE") == "1"
L1_ID, L6_ID = 20, 25
LO, HI = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12
STUMP_MM = 38.0                                       # Moller: a stump rib is <= 38 mm
MIN_VOX = 800                                         # ignore specks


def screen(lab: np.ndarray, affine) -> list:
    """Return list of (signal, detail) for anything needing expert review."""
    out = []
    spacing = np.sqrt((affine[:3, :3] ** 2).sum(axis=0))
    names = {v: k for k, v in LS.label_dict().items()}

    # L6 present?
    if int((lab == L6_ID).sum()) >= MIN_VOX:
        out.append(("L6_present", "id 25 labelled — confirm lumbarization vs mis-count"))

    # a rib on L1? map each rib to a vertebra by SI level (bbox), EDT-free
    ids = set(int(v) for v in np.unique(lab))
    rib_ids = [v for v in range(LO, HI + 1) if v in ids]
    vpres = [v for v in range(8, 26) if v in ids]
    if L1_ID in ids and rib_ids:
        objs = ndimage.find_objects(lab if lab.dtype.kind in "iu" else lab.astype(np.int32))
        R = np.asarray(affine)[:3, :3]; si = int(np.argmax(np.abs(R[2, :]))); sup = R[2, si] >= 0
        vranges = [(v, objs[v - 1][si].start, objs[v - 1][si].stop) for v in vpres if objs[v - 1]]
        for rid in rib_ids:
            sl = objs[rid - 1]
            if sl is None:
                continue
            head_si = sl[si].stop - 1 if sup else sl[si].start
            inside = [v for v, lo, hi in vranges if lo <= head_si < hi]
            v = inside[0] if inside else (min(vranges, key=lambda x: abs((x[1] + x[2]) // 2 - head_si))[0]
                                          if vranges else None)
            if v != L1_ID:
                continue
            # rib length (max extent, mm) -> full vs stump
            r = np.argwhere(lab[sl] == rid)
            if len(r) < 2:
                continue
            if len(r) > 300:
                r = r[:: len(r) // 300]
            pts = r * spacing
            length = float(np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1)).max())
            off = LS.RIB_LEFT_OFFSET if rid <= LS.RIB_LEFT_OFFSET + 12 else LS.RIB_RIGHT_OFFSET
            side = "left" if off == LS.RIB_LEFT_OFFSET else "right"
            if length > STUMP_MM:
                out.append(("full_rib_on_L1",
                            f"{side} rib on L1 is {length:.0f} mm (FULL) — L1 should not carry a full "
                            f"rib; likely a numbering error"))
            else:
                out.append(("stump_rib_on_L1",
                            f"{side} rib on L1 is {length:.0f} mm (stump) — lumbar-rib / transitional"))
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args(argv); tok = os.environ["HF_TOKEN"]
    recs = json.load(open(hf_hub_download(DS, "manifest.json", repo_type="dataset", token=tok,
                                          revision=REV, local_files_only=OFFLINE)))
    recs = recs if isinstance(recs, list) else recs.get("records", [])
    items = [(str(r.get("token")), r.get("config"), r.get("pseudo_label_file") or r.get("label_file"))
             for r in recs if (r.get("pseudo_label_file") or r.get("label_file"))]
    if a.limit:
        items = items[:: max(1, len(items) // a.limit)][:a.limit]
    print(f"screening {len(items)} v4 cases for expert-reread signals\n", flush=True)

    from concurrent.futures import ThreadPoolExecutor
    def work(it):
        t, cfg, p = it
        try:
            img = nib.load(hf_hub_download(DS, p, repo_type="dataset", token=tok, revision=REV,
                                           local_files_only=OFFLINE))
            return (t, cfg, screen(np.asanyarray(img.dataobj), img.affine))
        except Exception:
            return None

    rows = []; import collections; sig = collections.Counter(); done = 0
    with ThreadPoolExecutor(max_workers=a.workers) as ex:
        for r in ex.map(work, items):
            done += 1
            if done % 100 == 0: print(f"  ...{done}/{len(items)}", flush=True)
            if not r or not r[2]:
                continue
            t, cfg, flags = r
            for s, d in flags:
                sig[s] += 1
                rows.append({"token": t, "config": cfg, "signal": s, "detail": d})
    # write to a SEPARATE candidates file -- docs/lstv_reread_queue.csv is Greg's hand-curated queue
    # (rich schema, radiologist notes); this only proposes candidates to promote into it.
    Path("docs").mkdir(exist_ok=True)
    with open("docs/expert_reread_candidates.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["token", "config", "signal", "detail"])
        w.writeheader(); w.writerows(rows)
    print(f"\n===== EXPERT RE-READ CANDIDATES -> docs/expert_reread_candidates.csv =====")
    print(f"   flagged annotations: {len(rows)}  across {len({r['token'] for r in rows})} cases")
    for s, n in sig.most_common():
        print(f"     {s}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
