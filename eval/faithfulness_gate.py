#!/usr/bin/env python3
"""Model-agnostic faithfulness gate (HHEM-2.1-Open NLI).

The production 7B fine-tune ignores grounding prompts and rubber-stamps its own output, so
faithfulness has to be enforced OUTSIDE the generator. This gate scores each claim in an answer
against the retrieved context with a 110M entailment model (vectara/hallucination_evaluation_model,
CPU, edge-feasible) and turns a confidently-fabricated answer into a SAFE abstention that keeps
only the grounded sentences.

Validated separation on i.MX95 claims: supported 0.92–0.98, fabricated 0.005–0.045 → threshold 0.5.

CPU-ONLY BY DESIGN: HHEM's deps pull in TensorFlow, which grabs the whole GPU on import and will
OOM the llm-server sharing that card. We hide the GPU from this process entirely — a 110M model
runs fine on CPU, and CPU is the edge-deployment story anyway. Keep these env vars FIRST, before
any torch/transformers/tensorflow import.
"""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""   # force CPU — never contend with the llm-server GPU
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # quiet TensorFlow's CUDA-registration spam
import re

_MODEL = None

def _model():
    global _MODEL
    if _MODEL is None:
        from transformers import AutoModelForSequenceClassification
        _MODEL = AutoModelForSequenceClassification.from_pretrained(
            "vectara/hallucination_evaluation_model", trust_remote_code=True)
    return _MODEL

def split_claims(text, min_words=4):
    """Split an answer into checkable sentences; drop trivial fragments and greetings."""
    sents = re.split(r"(?<=[.!?])\s+|\n+", (text or "").strip())
    out = []
    for s in sents:
        s = s.strip(" -*•\t")
        if len(s.split()) >= min_words and not s.lower().startswith(("hi ", "hello", "dear ", "best,", "regards")):
            out.append(s)
    return out

# HHEM's predict() batches every pair with NO truncation, and T5 attention is O(batch x L^2):
# a full rag_k=8 context or meeting transcript x ~25 claims in one call hit ~90 GB RSS and got
# OOM-killed (2026-10-11). So: score one claim at a time against bounded context windows, and
# take the best window (a claim is supported if ANY window entails it).
WINDOW_TOKENS, STRIDE_TOKENS = 1024, 768

def _windows(context):
    tok = _model().tokenzier  # (sic) HHEM's attribute name
    ids = tok(context or "", add_special_tokens=False)["input_ids"]
    if len(ids) <= WINDOW_TOKENS:
        return [context or ""]
    return [tok.decode(ids[i:i + WINDOW_TOKENS]) for i in range(0, len(ids) - STRIDE_TOKENS, STRIDE_TOKENS)]

def score(context, claims):
    if not claims:
        return []
    wins = _windows(context)
    return [max(float(_model().predict([(w, c)])[0]) for w in wins) for c in claims]

def _score_answer(answer, context):
    claims = split_claims(answer)
    return claims, score(context, claims)

def gate(answer, context, threshold=0.5, _scored=None):
    """Faithfulness verdict for an answer against its context."""
    claims, scores = _scored or _score_answer(answer, context)
    unsupported = [(c, s) for c, s in zip(claims, scores) if s < threshold]
    n = len(claims) or 1
    return {
        "n_claims": len(claims),
        "n_unsupported": len(unsupported),
        "min_score": round(min(scores), 3) if scores else None,
        "mean_score": round(sum(scores) / n, 3) if scores else None,
        "unsupported": [{"claim": c[:140], "score": round(s, 3)} for c, s in unsupported],
        "faithful": len(unsupported) == 0,
    }

ABSTAIN = ("I can't give a fully confident answer — the retrieved sources don't support some of the "
           "specifics this question needs, and I won't guess. What the sources do support: ")

def gated_answer(answer, context, threshold=0.5):
    """Return (output, verdict). If unfaithful, drop the unsupported sentences and return an
    abstention that keeps only the grounded ones — a safe partial instead of a confident fabrication."""
    claims, scores = _score_answer(answer, context)
    g = gate(answer, context, threshold, _scored=(claims, scores))
    if g["faithful"]:
        return answer, g
    kept = [c for c, s in zip(claims, scores) if s >= threshold]
    msg = ABSTAIN + (" ".join(kept) if kept else "(the sources don't cover the specifics asked).")
    return msg, g

if __name__ == "__main__":
    ctx = ("NXP i.MX 95: 6x Cortex-A55; eIQ Neutron NPU 2.0 TOPS (1024 MACs x 1.0 GHz x 2); "
           "Mali GPU; LPDDR5/LPDDR4X; PCIe Gen3; GbE-TSN.")
    ans = "The i.MX 95 has 6 Cortex-A55 cores and a 2.0 TOPS Neutron NPU. It uses LPDDR5X and 16GB of on-package HBM."
    out, g = gated_answer(ans, ctx)
    print("verdict:", g)
    print("gated output:", out)
