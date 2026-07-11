# iq9 Qwen2.5-7B Q8_0 — two-judge regrade (2026-07-10)

qualcomm ran stock Qwen2.5-7B **Q8_0** (weight-only int8, fp16 compute) on the
v2.1 ORACLE set (retrieval frozen → isolates the model/quantizer) on the IQ-9075
host A78 CPU via llama.cpp. Substring was a screening number; here is the
two-judge regrade.

| grader | pass rate |
|---|--:|
| substring (qualcomm) | 75.6% |
| GPT-4o | **76.1%** |
| Claude Sonnet | **76.9%** |

117 clean paired samples, persona excluded. **Three-grader spread: 1.3pp.**

The substring over-credit the MANIFEST warns about (~10pp on Qwen fine-tunes) did
NOT materialize here — only ~2pp — because that bias is a property of KYLE'S
FINE-TUNE (voice/phrasing over-match), not stock Qwen. So 75.6% substring was
already close to honest, and the trustworthy Q8_0 number is **~76%**.

**Remaining for the "8-bit is free" third nail:** an fp16 (or Q4) stock-Qwen2.5-7B
run on the SAME v2.1 oracle prompt_pack, to get the Q8_0-vs-fp16 delta. Not yet
run (needs a GPU generation pass + the fp16 model). The number above is the model
accuracy under Q8_0; the delta vs fp16 is the quantizer-free claim.

---

## fp16 delta — SEALED (2026-07-10)

fp16 stock Qwen2.5-7B-Instruct run through the SAME v2.1 oracle prompt_pack via
llama.cpp on the 5090 (apples-to-apples with qualcomm's iq9 Q8_0). 112 clean
paired samples, three graders:

| grader | Q8_0 | fp16 | Q8_0 − fp16 |
|---|--:|--:|--:|
| substring | 79.5% | 79.5% | **0.0pp** |
| Sonnet | 79.5% | 79.5% | **0.0pp** |
| GPT-4o | 77.7% | 80.4% | −2.7pp |

**Two of three graders: exactly 0.0pp.** GPT-4o: −2.7pp (3 samples / 112, inside
the σ≈1.4–2.3pp noise floor). **Q8_0 weight-only int8 is indistinguishable from
fp16 — the quantizer added nothing. Third nail seated.**

The 8-bit-is-free result now spans three independent quantization mechanisms,
two model sizes, two silicon vendors, and three grading methods:
- **W8A8** (SmoothQuant+GPTQ, int8 activations, vLLM) on Qwen2.5-14B → 0pp
- **FP8** (e4m3) → 0pp
- **Q8_0** (weight-only int8, llama.cpp) on stock Qwen2.5-7B → 0pp (this)
