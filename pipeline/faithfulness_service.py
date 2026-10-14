"""HTTP wrapper for the HHEM faithfulness gate (eval/faithfulness_gate.py), run as a CPU-only sidecar.

llm_server.py calls POST /gate when the faithfulness_gate setting (or a request override) is on.
The gate logic itself stays in eval/faithfulness_gate.py so the A/B experiment and production
use the same code.
"""
import sys, threading
from typing import List, Optional, Union
from fastapi import FastAPI
from pydantic import BaseModel

sys.path.insert(0, "/app/eval")
import faithfulness_gate as fg  # sets CUDA_VISIBLE_DEVICES="" before torch loads

app = FastAPI(title="Skippy faithfulness gate")
_lock = threading.Lock()  # one scoring pass at a time: bounds peak RAM (see the 2026-10-11 OOM)

class GateRequest(BaseModel):
    answer: str
    context: Union[str, List[str]]
    threshold: Optional[float] = 0.5

@app.on_event("startup")
def _load():
    fg._model()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/gate")
def gate(req: GateRequest):
    ctx = "\n\n".join(req.context) if isinstance(req.context, list) else req.context
    with _lock:
        output, verdict = fg.gated_answer(req.answer, ctx, threshold=req.threshold)
    return {"output": output, "verdict": verdict}
