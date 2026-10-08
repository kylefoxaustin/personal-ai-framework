#!/usr/bin/env python3
"""Live LLM decode bench on a Kinara Ara-2 (NXP i.MX95 ARA240), via optimum-ara.

Loads the resident Qwen .dvm, warms up, times one generation, prints ARA_RESULT JSON.
Run ON the board from /usr/share/rt-sdk-ara240/optimum-ara (needs the rt-sdk-ara2 proxy up).
NOTE: the deployed .dvm is compiled for a FIXED output length (~20 tok on the staged qwen3b),
so `tok_s` is an END-TO-END rate (prefill + fixed decode), a lower bound on steady-state decode;
a longer-output number needs a Kinara recompile. bench_board.py backs out prefill for a decode estimate.
"""
import sys, time, json
from transformers import AutoTokenizer, AutoModelForCausalLM
import optimum.ara

M = sys.argv[1] if len(sys.argv) > 1 else "/run/media/root-mmcblk0p2/qwen3b-ara"
try:
    model = AutoModelForCausalLM.from_pretrained(M)
    tok = AutoTokenizer.from_pretrained(M + "/tokenizer")
    msgs = [{"role": "user", "content": "Write a detailed paragraph about edge AI for industrial devices."}]
    s = tok.apply_chat_template(conversation=msgs, tokenize=False, add_generation_prompt=True)
    inp = tok(s); plen = len(inp["input_ids"])
    _ = model.generate(**inp)                       # warmup (first call loads onto the Ara)
    t0 = time.time(); out = model.generate(**inp); dt = time.time() - t0
    seq = out[0] if hasattr(out, "__getitem__") else out
    n_new = len(seq) - plen
    txt = tok.decode(seq)[-200:].replace("\n", " ")
    print("ARA_RESULT " + json.dumps({"decode_toks": n_new, "wall_s": round(dt, 3),
          "tok_s_e2e": round(n_new / dt, 2), "prompt_toks": plen,
          "model": "qwen2.5-3b", "device": "ARA2", "coherent_tail": txt}))
except Exception as e:
    print("ARA_ERROR " + repr(e)[:300])
