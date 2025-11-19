# server/main.py
import uuid
import time
from typing import Dict, Any
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Use your existing helper
from utils import extract_frames  # adjust import path if needed

app = FastAPI(title="Deepfake Analysis API (single-endpoint)")

# Allow requests from Vite dev server (adjust origin as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173", 
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ResultPayload(BaseModel):
    verdict: str
    confidence: float

def run_analysis_inline(file_bytes: bytes, filename: str) -> Dict[str, Any]:
    """
    Run analysis synchronously and return a report dictionary.
    The returned dict contains status, percent, stageName and result (verdict+confidence).
    Replace the simulated parts with your real model inference.
    """
    report: Dict[str, Any] = {
        "status": "processing",
        "percent": 0,
        "stageName": "Queued",
        "result": None,
        "error": None,
    }

    try:
        # Stage 1: extracting frames
        report["stageName"] = "Extracting frames"
        report["percent"] = 5
        frames = extract_frames(file_bytes, max_frames=8, size=224, filename=filename)

        # Stage 2: model run
        report["stageName"] = "Running model"
        report["percent"] = 35

        # TODO: Replace the following simulated inference with your actual model inference.
        time.sleep(1.0)  # simulate compute
        import random
        is_deepfake = random.random() > 0.5
        confidence = 70.0 + random.random() * 30.0

        # Stage 3: finalizing
        report["stageName"] = "Final Classification"
        report["percent"] = 100
        report["status"] = "complete"
        report["result"] = {
            "verdict": "Deepfake" if is_deepfake else "Authentic",
            "confidence": float(confidence)
        }

        # ------------------------
        # ⭐ Console Output (Added)
        # ------------------------
        print("\n===== Deepfake Analysis Result =====")       # <-- added
        print(f"File Name: {filename}")                     # <-- added
        print(f"Verdict: {report['result']['verdict']}")    # <-- added
        print(f"Confidence: {report['result']['confidence']}\n")  # <-- added

    except HTTPException as he:
        report["status"] = "error"
        report["error"] = str(he.detail)
        print(f"[ERROR] {he.detail}")                       # <-- added
    except Exception as e:
        report["status"] = "error"
        report["error"] = str(e)
        print(f"[ERROR] {str(e)}")                          # <-- added

    return report

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    """
    Synchronous analysis endpoint.
    Upload a video file and wait for the analysis to finish in this request.
    Returns a JSON object with status, percent, stageName and result.
    """
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file uploaded")

    try:
        filename = file.filename or "uploaded_file"
        report = run_analysis_inline(contents, filename)

        if report.get("status") == "error":
            raise HTTPException(status_code=500, detail=report.get("error") or "analysis error")

        return report

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
