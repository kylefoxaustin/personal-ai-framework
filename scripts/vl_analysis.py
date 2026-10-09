import json, glob, csv, statistics as st
SV="/home/kyle/Documents/GitHub/splat-vla"
def load(p):
    out=[]
    for ln in open(p):
        ln=ln.strip()
        if ln:
            try: out.append(json.loads(ln))
            except: pass
    return out

print("=== RESOURCE by resolution (prefill/decode, image-token-driven prefill share) ===")
print(f"{'rung':8s} {'n':>4s} {'img+prompt tok':>14s} {'prefill t/s':>11s} {'decode t/s':>10s} {'prefill_ms':>10s} {'decode_ms':>9s} {'prefill_share':>13s}")
for rung in ["0.25MP","0.50MP","1.00MP","native"]:
    fs=glob.glob(f"{SV}/out/vlbench/resource_{rung}.jsonl")
    if not fs: continue
    d=load(fs[0])
    if not d: continue
    tok=st.mean(x["prompt_tokens"] for x in d)
    pf=st.mean(x["prompt_per_second"] for x in d)
    dc=st.mean(x["predicted_per_second"] for x in d)
    pms=st.mean(x["prompt_ms"] for x in d)
    dms=st.mean(x["predicted_ms"] for x in d)
    share=100*pms/(pms+dms)
    print(f"{rung:8s} {len(d):>4d} {tok:>14.0f} {pf:>11.1f} {dc:>10.1f} {pms:>10.1f} {dms:>9.1f} {share:>12.1f}%")

print("\n=== SPECIES — intra-rater ceiling + model accuracy (human_confirmed.csv, verification-with-prior) ===")
rows=list(csv.DictReader(open(f"{SV}/bench/vlbench/labels/human_confirmed.csv")))
n=len(rows)
changed=sum(1 for r in rows if r.get("changed")=="1")
# intra-rater agreement (pass1 vs pass2)
agree=sum(1 for r in rows if r["verdict_pass1"]==r["verdict_pass2"])
print(f"  n={n}  intra-rater self-agreement={100*agree/n:.1f}% (changed {changed}/{n}={100*changed/n:.1f}%) <- the CEILING")
# model species vs human verdict (use pass2 as the settled label)
def norm(s): return (s or "").strip().lower()
match=sum(1 for r in rows if norm(r["model_claim"])==norm(r["verdict_pass2"]))
print(f"  model species == human(pass2): {match}/{n} = {100*match/n:.1f}%  (ceiling-relative: {100*match/n/ (agree/n):.1f}% of the human ceiling)")
# detection-ish: how often human said SOME animal (not nothing/human) vs model claimed present

print("\n=== SPECIES — reconcile denominators (species-only vs all-verdict) ===")
NONAN={"nothing","none","human",""}
def norm(s): return (s or "").strip().lower()
# species-only = rows where human(pass2) names an actual animal
sp=[r for r in rows if norm(r["verdict_pass2"]) not in NONAN]
sp_agree=sum(1 for r in sp if r["verdict_pass1"]==r["verdict_pass2"])
sp_match=sum(1 for r in sp if norm(r["model_claim"])==norm(r["verdict_pass2"]))
print(f"  species-only n={len(sp)}: self-agree {100*sp_agree/len(sp):.1f}% · model-match {100*sp_match/len(sp):.1f}%")
# animal-presence (detection-ish): human says animal vs not
hum_animal=sum(1 for r in rows if norm(r["verdict_pass2"]) not in NONAN)
print(f"  human verdict breakdown: animal={hum_animal}/{len(rows)}  non-animal(nothing/human)={len(rows)-hum_animal}")
from collections import Counter
print("  human pass2 classes:", dict(Counter(norm(r['verdict_pass2']) for r in rows)))
