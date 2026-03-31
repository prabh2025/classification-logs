from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import time
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))
from classify import classify_log

app = FastAPI(
    title="Log Classification API",
    description="Classify logs using Regex + BERT + LLM",
    version="2.0.0"
)

class LogRequest(BaseModel):
    source: str
    log_message: str

class LogResponse(BaseModel):
    source: str
    log_message: str
    predicted_label: str
    latency_ms: float

class BatchRequest(BaseModel):
    logs: List[LogRequest]

@app.get("/health")
def health():
    return {"status": "healthy", "version": "2.0.0"}

@app.post("/classify", response_model=LogResponse)
def classify_single(request: LogRequest):
    try:
        start = time.time()
        label = classify_log(request.source, request.log_message)
        latency = (time.time() - start) * 1000
        return LogResponse(
            source=request.source,
            log_message=request.log_message,
            predicted_label=label or "Unclassified",
            latency_ms=round(latency, 2)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/classify/batch")
def classify_batch(request: BatchRequest):
    try:
        results = []
        for log in request.logs:
            start = time.time()
            label = classify_log(log.source, log.log_message)
            latency = (time.time() - start) * 1000
            results.append({
                "source": log.source,
                "log_message": log.log_message,
                "predicted_label": label or "Unclassified",
                "latency_ms": round(latency, 2)
            })
        return {"results": results, "total": len(results)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))