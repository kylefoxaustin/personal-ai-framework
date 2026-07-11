#!/usr/bin/env python3
"""Generate completions over a pre-rendered oracle prompt_pack via llama-cpp-python.

Runs INSIDE the llm-server container (has llama-cpp-python + CUDA). Raw generation:
apply the GGUF's own chat template to [system, user] and generate at temp 0. NO
retrieval — the oracle context is already baked into each prompt's `user` field,
exactly as qualcomm's iq9 Q8_0 run did. This makes the fp16 run apples-to-apples
with his Q8_0 for the quantizer delta.

    python3 gen_oracle.py <model.gguf> <prompt_pack.jsonl> <completions_out.jsonl>
"""
import json
import sys

from llama_cpp import Llama

model_path, pack_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]

llm = Llama(model_path=model_path, n_gpu_layers=-1, n_ctx=16384, verbose=False)

pack = [json.loads(l) for l in open(pack_path) if l.strip()]
print(f"loaded {len(pack)} prompts from {pack_path}", file=sys.stderr)

with open(out_path, "w") as f:
    for i, e in enumerate(pack, 1):
        r = llm.create_chat_completion(
            messages=[{"role": "system", "content": e["system"]},
                      {"role": "user", "content": e["user"]}],
            temperature=0.0,
            max_tokens=e.get("gen", {}).get("max_tokens", 512),
        )
        text = r["choices"][0]["message"]["content"]
        f.write(json.dumps({
            "uid": e["uid"], "prompt_id": e["prompt_id"],
            "sample_index": e["sample_index"], "text": text,
        }) + "\n")
        if i % 20 == 0 or i == len(pack):
            print(f"  {i}/{len(pack)}", file=sys.stderr)

print(f"wrote {out_path}", file=sys.stderr)
