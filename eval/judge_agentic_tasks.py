#!/usr/bin/env python3
"""Two-judge (cross-family) grading of the agentic-edge task-success column.

Grades the 10 agentic-bench task outputs PASS/PARTIAL/FAIL with TWO independent
judges from different model families — Claude Sonnet (Anthropic) + GPT-4o (OpenAI) —
per the repo's "two judges by default" rule (a result leaving the repo can't ride on
one model's blind spots). Reads the latest bench_<task>_*.json, grades each output
against the task intent + datasheet ground truth, writes judge verdicts + agreement.

Keys come from the ENVIRONMENT (ANTHROPIC_API_KEY, OPENAI_API_KEY). Run from an
interactive shell so ~/.bashrc exports them:  bash -ic 'python3 eval/judge_agentic_tasks.py'
--dry-run prints the prompts without calling any API (no keys needed).
"""
import argparse, glob, json, os, re, sys

OUT = "eval/results/ladder/agentic"
GROUND_TRUTH = (
    "NXP i.MX 95 datasheet facts: 6x Arm Cortex-A55 + Cortex-M33 + Cortex-M7; eIQ Neutron NPU "
    "(N3-1024S) rated 2.0 TOPS = 1024 MACs x 1.0 GHz x 2 OPS/MAC; Mali GPU; ISP; LPDDR5X memory; "
    "PCIe Gen3; Gigabit-Ethernet TSN; Energy Flex architecture. Targets automotive/industrial/"
    "consumer edge AI. NOT true: 'Neutrino' NPU, dual-A55-only, Cortex-M7-pair-only, LPDDR4X, 16GB."
)
# per-task intent + what PASS requires; the audio clip is a weather forecast ("thunderstorms,
# large hail, isolated tornadoes, heavy rain"), NOT a meeting with named attendees.
RUBRIC = {
 "email": "Draft a coherent, friendly email introducing the i.MX 95 for an edge-AI product, covering what it is, the Neutron NPU, and why it suits edge. PASS = coherent email, specifics grounded in the facts, no fabricated specs.",
 "spec_rag": "Answer what the i.MX 95 Neutron NPU does and its key features, from the datasheets. PASS = accurate (2.0 TOPS / 1024 MAC@1GHz) and grounded, no fabrication.",
 "file_ops": "Read notes and summarize them in 3 bullets. PASS = accurate bullet summary of the i.MX95 specs, no fabrication.",
 "transcribe": "Transcribe the audio clip. PASS = transcript matches the weather-forecast content (thunderstorms/large hail/isolated tornadoes/heavy rain).",
 "ocr": "Extract text from a datasheet page image. PASS = coherent, accurate extraction of the page text (AN14721 TRDC application note).",
 "run_script": "Run a sandboxed script computing the NPU TOPS. PASS = correct output (2.0 TOPS; 6xA55; LPDDR5X; Gen3), exit 0.",
 "doc_brief": "Generate a one-page i.MX 95 product brief from the datasheets. PASS = coherent, grounded brief, no fabricated specs.",
 "web_summarize": "Summarize web results about the i.MX 95. PASS = coherent AND contains actual i.MX95-specific content; PARTIAL = coherent but generic/content-free.",
 "meeting_summarize": "Summarize the recording as a meeting. The clip is a weather forecast with NO named attendees/decisions. PASS = correct summary; PARTIAL = correct gist but FABRICATED action-items/attendee names; FAIL = wrong.",
 "multi_tool_chain": "Steps: note the specs, WRITE them to a file, draft an email. PASS = actually performs the tool steps (file write) with accurate specs; FAIL = only narrates the steps and/or fabricates specs.",
}
JUDGE_INSTRUCTIONS = (
 "You are grading whether a local AI assistant completed an agentic task. Grade ONLY task success "
 "and factual correctness (not style). Return STRICT JSON only: "
 '{\"verdict\": \"PASS\"|\"PARTIAL\"|\"FAIL\", \"reason\": \"<one sentence>\"}. '
 "PARTIAL = did the task but with a material flaw (generic/content-free, or fabricated a sub-part). "
 "FAIL = did not accomplish the task or fabricated the core answer."
)

def latest(task):
    fs = sorted(glob.glob(f"{OUT}/bench_{task}_*.json"))
    return json.load(open(fs[-1])) if fs else None

def build_user_prompt(task, output):
    return (f"GROUND TRUTH: {GROUND_TRUTH}\n\nTASK: {task}\nWHAT PASS REQUIRES: {RUBRIC[task]}\n\n"
            f"ASSISTANT OUTPUT:\n\"\"\"\n{output[:2500]}\n\"\"\"\n\nGrade it. STRICT JSON only.")

def parse_verdict(text):
    m = re.search(r'\{.*\}', text, re.S)
    if not m: return {"verdict": "?", "reason": f"unparseable: {text[:120]}"}
    try:
        d = json.loads(m.group(0)); d["verdict"] = str(d.get("verdict","?")).upper(); return d
    except Exception as e:
        return {"verdict": "?", "reason": f"parse error: {e}"}

def judge_anthropic(task, output, model):
    from anthropic import Anthropic
    c = Anthropic()
    r = c.messages.create(model=model, max_tokens=300, system=JUDGE_INSTRUCTIONS,
                          messages=[{"role":"user","content":build_user_prompt(task,output)}])
    return parse_verdict(r.content[0].text)

def judge_openai(task, output, model):
    from openai import OpenAI
    c = OpenAI()
    r = c.chat.completions.create(model=model, max_tokens=300,
        messages=[{"role":"system","content":JUDGE_INSTRUCTIONS},
                  {"role":"user","content":build_user_prompt(task,output)}])
    return parse_verdict(r.choices[0].message.content)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--anthropic-model", default="claude-sonnet-4-6")
    ap.add_argument("--openai-model", default="gpt-4o")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--out", default=f"{OUT}/task_success_twojudge.json")
    a = ap.parse_args()
    tasks = list(RUBRIC)
    if a.dry_run:
        for t in tasks:
            r = latest(t); print(f"\n===== {t} =====\n{build_user_prompt(t, (r or {}).get('output',''))[:600]}")
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from _keys import ensure_keys
    missing = ensure_keys(("ANTHROPIC_API_KEY", "OPENAI_API_KEY"))
    if missing:
        sys.exit(f"ERROR: {', '.join(missing)} not set. Add them to ~/.personal-ai/keys.env "
                 f"(copy ~/.personal-ai/keys.env.example and fill in values; see docs/api-keys-setup.md).")
    results, tally = {}, {"agree":0,"split":0}
    for t in tasks:
        r = latest(t)
        if not r: print(f"  {t}: no bench json, skip"); continue
        out = r.get("output","")
        try: sv = judge_anthropic(t, out, a.anthropic_model)
        except Exception as e: sv = {"verdict":"ERR","reason":str(e)[:120]}
        try: gv = judge_openai(t, out, a.openai_model)
        except Exception as e: gv = {"verdict":"ERR","reason":str(e)[:120]}
        agree = sv.get("verdict")==gv.get("verdict")
        tally["agree" if agree else "split"] += 1
        results[t] = {"sonnet":sv, "gpt4o":gv, "agree":agree}
        print(f"  {t:18s} sonnet={sv.get('verdict'):8s} gpt4o={gv.get('verdict'):8s} {'AGREE' if agree else '<<SPLIT>>'}")
    summary = {"ts":"2026-10-07","judges":{"anthropic":a.anthropic_model,"openai":a.openai_model},
               "rule":"two judges, different families; both must be read; splits are flagged not averaged",
               "agreement":tally, "results":results}
    json.dump(summary, open(a.out,"w"), indent=2)
    print(f"\n  wrote {a.out}  ·  agree={tally['agree']} split={tally['split']}")

if __name__ == "__main__":
    main()
