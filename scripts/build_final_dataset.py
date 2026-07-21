"""
scripts/build_final_dataset.py — assemble the FINAL unified labels from BOTH review Spaces + the v4
base, one label per case:

    base            = the v4 label
    + ribs (34-57)  from the reviews-RIBS ledger's finalized label   (student-corrected ribs)
    + thoracic adds from the reviews-SPINE ledger's finalized label   (vertebrae 8-19 added on v4 bg)

A case finalized in both gets both; a case in neither is the v4 label as-is; a case finalized as
REJECT is excluded. Output feeds straight into apply_lumbar_rib_class.py.

    HF_TOKEN=... python scripts/build_final_dataset.py --out FINAL_DIR [--v4-rev <commit>] [--limit N]

Use --v4-rev with the cached commit + HF_HUB_OFFLINE=1 to avoid re-downloading the 802 base labels.
"""
from __future__ import annotations
import argparse, csv, json, os, sys
from pathlib import Path
import numpy as np, nibabel as nib

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE)); sys.path.insert(0, str(_HERE / ".." / "review_service"))
sys.path.insert(0, str(_HERE / "review"))
import label_scheme as LS            # noqa: E402
from huggingface_hub import hf_hub_download   # noqa: E402

DS = os.environ.get("V2_REPO", "anonymous-mlhc/CTSpinoPelvic1K")
RIB_REPO = os.environ.get("RIB_REPO", "anonymous-mlhc/CTSpinoPelvic1K-reviews-ribs")
SPINE_REPO = os.environ.get("SPINE_REPO", "anonymous-mlhc/CTSpinoPelvic1K-reviews-spine")
LOR, HIR = LS.RIB_LEFT_OFFSET + 1, LS.RIB_RIGHT_OFFSET + 12     # ribs 34..57


def _final_map(repo, tok):
    """{case_id: final_label_rel} for cases finalized (not rejected) in a ledger repo."""
    from huggingface_hub import HfApi
    api = HfApi(token=tok)
    out = {}
    try:
        files = [f for f in api.list_repo_files(repo, repo_type="dataset")
                 if f.startswith("cases/") and f.endswith(".json")]
    except Exception as e:
        print(f"  ! cannot list {repo}: {e}"); return out
    for f in files:
        try:
            c = json.loads(Path(hf_hub_download(repo, f, repo_type="dataset", token=tok)).read_text())
        except Exception:
            continue
        fin = c.get("final")
        if fin and fin.get("decision") != "reject" and fin.get("label_rel"):
            out[c["case_id"]] = fin["label_rel"]
    return out


def _load(repo, rel, tok, rev=None):
    kw = {"repo_type": "dataset", "token": tok}
    if rev:
        kw["revision"] = rev
    img = nib.load(hf_hub_download(repo, rel, **kw))
    return np.asanyarray(img.dataobj), img


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--v4-rev", default="v4", help="v4 revision or a cached commit hash (with HF_HUB_OFFLINE=1)")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args(argv); tok = os.environ["HF_TOKEN"]
    out_dir = Path(a.out); out_dir.mkdir(parents=True, exist_ok=True)

    recs = json.load(open(hf_hub_download(DS, "manifest.json", repo_type="dataset", token=tok, revision=a.v4_rev)))
    recs = recs if isinstance(recs, list) else recs.get("records", [])
    rib_fin = _final_map(RIB_REPO, tok); spine_fin = _final_map(SPINE_REPO, tok)
    print(f"finalized: ribs={len(rib_fin)}  spine={len(spine_fin)}  | base cases={len(recs)}\n", flush=True)

    import schema as _sc              # scripts/review/schema.py: case_id(token, config)
    rows = []; nrib = nspine = nboth = nplain = 0; done = 0
    items = recs[:a.limit] if a.limit else recs
    for r in items:
        lf = r.get("label_file") or r.get("pseudo_label_file")
        if not lf:
            continue
        cid = _sc.case_id(r.get("token"), r.get("config"))
        done += 1
        if done % 50 == 0:
            print(f"  ...{done}/{len(items)}", flush=True)
        base, bimg = _load(DS, lf, tok, rev=a.v4_rev)
        out = base.copy(); used = ["v4"]
        if cid in rib_fin:
            rib, _ = _load(RIB_REPO, rib_fin[cid], tok)
            if rib.shape == out.shape:
                out[(out >= LOR) & (out <= HIR)] = 0            # drop base ribs
                m = (rib >= LOR) & (rib <= HIR); out[m] = rib[m]  # student-corrected ribs
                used.append("ribs"); nrib += 1
        if cid in spine_fin:
            sp, _ = _load(SPINE_REPO, spine_fin[cid], tok)
            if sp.shape == out.shape:
                add = (sp >= 8) & (sp <= 19) & (base == 0)       # thoracic vertebrae added on v4 bg
                out[add] = sp[add]; used.append("spine"); nspine += 1
        if "ribs" in used and "spine" in used:
            nboth += 1
        if used == ["v4"]:
            nplain += 1
        nib.save(nib.Nifti1Image(out.astype(base.dtype), bimg.affine, bimg.header),
                 str(out_dir / Path(lf).name))
        rows.append({"case": cid, "label": Path(lf).name, "merged": "+".join(used)})
    with open("final_build_report.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["case", "label", "merged"]); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {len(rows)} final labels -> {out_dir}")
    print(f"   with rib corrections: {nrib}   with spine additions: {nspine}   both: {nboth}   v4-only: {nplain}")
    print(f"   report -> final_build_report.csv   |   next: apply_lumbar_rib_class.py --labels {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
