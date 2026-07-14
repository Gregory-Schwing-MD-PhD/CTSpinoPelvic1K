import os, sys, json, collections, traceback
sys.path.insert(0,"review_service"); sys.path.insert(0,"scripts")
import numpy as np, nibabel as nib
import store as store_mod
from review import schema
import review_anatomy_qc as RA
from huggingface_hub import hf_hub_download
tok=os.environ["HF_TOKEN"]; REPO="anonymous-mlhc/CTSpinoPelvic1K-reviews-ribs"
st = store_mod.ReviewStore(store_mod.HFBackend(repo_id=REPO, token=tok))

def gate_fails(lab, aff, given=None):
    fails=[]
    if given is not None and not RA.spine_untouched(lab, given)[0]:
        fails.append("spine_altered")
    if not RA.rib_label_mixing(lab,aff)[0]: fails.append("rib_label_mixing")
    if not RA.structure_integrity(lab,aff)[0]: fails.append("structure_integrity")
    if not RA.rib_spine_gap(lab,aff)[0]: fails.append("rib_spine_gap")
    return fails

stats=collections.defaultdict(lambda: {"total":0,"passed":0,"fails":collections.Counter(),"reopened":0})
changed=[]; n_eval=0; n_reopened=0; n_pass=0
cases=[c for c in st.list_cases() if c.get("region_to_review")=="ribs"]
print(f"sweeping {len(cases)} rib cases...", flush=True)
for ci,case in enumerate(cases):
    cid=case["case_id"]; status=schema.derive_status(case); dirty=False
    for slot in ("1","2"):
        sl=case.get("slots",{}).get(slot)
        if not sl: continue
        rel = sl.get("label_path") or f"reviews/{cid}/{slot}_label.nii.gz"
        try:
            p=hf_hub_download(REPO, rel, repo_type="dataset", token=tok, force_download=True)
        except Exception:
            continue                         # never submitted -> skip
        try:
            img=nib.load(p); lab=np.asanyarray(img.dataobj); aff=img.affine
            given=None
            try:
                from huggingface_hub import hf_hub_download as _dl
                given=np.asanyarray(nib.load(_dl("anonymous-mlhc/CTSpinoPelvic1K",
                        case["pseudo_label_file"], repo_type="dataset", token=tok,
                        revision="v4")).dataobj)
            except Exception: pass
            fails=gate_fails(lab,aff,given)
        except Exception as e:
            print(f"  [qc err] {cid}/{slot}: {str(e)[:60]}", flush=True); continue
        passed = not fails
        rev=sl.get("reviewer","?")
        stats[rev]["total"]+=1; n_eval+=1
        sl["qc_pass"]=passed; sl["qc_fail_checks"]=fails; dirty=True
        if passed:
            stats[rev]["passed"]+=1; n_pass+=1
            sl["done"]=True
            for k in ("amend","amend_base","amend_reason"): sl.pop(k,None)
        else:
            for f in fails: stats[rev]["fails"][f]+=1
            if status!="finalized":          # kick back to the student for amend
                sl["done"]=False; sl["amend"]=True
                sl["amend_base"]=rel; sl["amend_reason"]="; ".join(fails)
                stats[rev]["reopened"]+=1; n_reopened+=1
    if dirty: changed.append(case)
    if ci%20==0: print(f"  ...{ci}/{len(cases)}  evaluated={n_eval}", flush=True)

print(f"\napplying {len(changed)} updated cases (one batch commit)...", flush=True)
st.put_cases(changed)
print(f"DONE. evaluated={n_eval}  passed={n_pass}  reopened(kicked back)={n_reopened}\n", flush=True)

# per-student scorecard
rows=sorted(stats.items(), key=lambda kv:-kv[1]["passed"])
print(f"{'student':20s} {'subs':>5s} {'pass':>5s} {'pass%':>6s}  {'reopened':>8s}  fail reasons")
out={}
for rev,d in rows:
    pct = round(100*d["passed"]/d["total"]) if d["total"] else 0
    fr = ", ".join(f"{k}:{v}" for k,v in d["fails"].most_common())
    print(f"{rev:20s} {d['total']:5d} {d['passed']:5d} {pct:5d}%  {d['reopened']:8d}  {fr}")
    out[rev]={"subs":d["total"],"passed":d["passed"],"pass_pct":pct,"reopened":d["reopened"],"fails":dict(d["fails"])}
json.dump(out, open("scratchpad_resweep_stats.json","w"), indent=2)
tot=sum(d["total"] for _,d in rows); pas=sum(d["passed"] for _,d in rows)
print(f"\nOVERALL: {pas}/{tot} submissions pass ({round(100*pas/tot) if tot else 0}%); {n_reopened} kicked back for amend.")
