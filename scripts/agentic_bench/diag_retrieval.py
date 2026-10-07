#!/usr/bin/env python3
"""Diagnose the i.MX 95 NPU retrieval miss: does phrasing (naming 'Neutron') matter?"""
import json, urllib.request
BASE="http://localhost:8080"
tok=json.loads(urllib.request.urlopen(urllib.request.Request(f"{BASE}/auth/login",
    data=json.dumps({"username":"kyle","password":"123456"}).encode(),
    headers={"Content-Type":"application/json"}),timeout=15).read())["token"]
H={"Content-Type":"application/json","Authorization":f"Bearer {tok}"}

def gen(prompt):
    body={"prompt":prompt,"use_rag":True,"rag_k":5,"max_tokens":150,"skip_agent_loop":True}
    d=json.loads(urllib.request.urlopen(urllib.request.Request(f"{BASE}/generate",
        data=json.dumps(body).encode(),headers=H),timeout=120).read())
    cites=d.get("context_used") or []
    srcs=[]
    for c in cites:
        c=str(c)
        for key in ("IMX95RM","IMX95IEC","IMX93","IMXRT","MCXN","Neutron"):
            if key in c: srcs.append(key)
    return d.get("text","")[:220], d.get("citations"), srcs[:8]

for q in [
    "what NPU does the i.MX 95 have and how many TOPS?",
    "What does the i.MX 95 Neutron NPU do? How many TOPS?",
    "i.MX 95 eIQ Neutron NPU TOPS performance",
]:
    txt,cites,srcs=gen(q)
    print("Q:",q)
    print("  ANSWER:",repr(txt))
    print("  source-keys hit in context:",srcs)
    print()
