import os
import time
from pathlib import Path

import torch
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
import uvicorn

from openkeyscan_analyzer_server import (
    preprocess_audio,
    camelot_output,
    camelot_to_openkey,
    load_model,
    get_resource_path
)

# -----------------------------------------------------------------------------
# App
# -----------------------------------------------------------------------------
app = FastAPI(title="OpenKeyScan Analyzer API", version="1.0")

# -----------------------------------------------------------------------------
# Load model ONCE at startup (important for Railway performance)
# -----------------------------------------------------------------------------
MODEL_PATH = get_resource_path("checkpoints/openkeyscan3.pt")

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("cpu")
)

model = load_model(MODEL_PATH, device)
model.eval()

print(f"[API] Model loaded on {device}")


# -----------------------------------------------------------------------------
# Health check
# -----------------------------------------------------------------------------
@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


# -----------------------------------------------------------------------------
# Core endpoint
# -----------------------------------------------------------------------------
@app.post("/analyze/single")
async def analyze_single(file: UploadFile = File(...)):
    start = time.time()

    # Validate file
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")

    suffix = Path(file.filename).suffix.lower()

    # Save temp file (Railway-safe approach)
    temp_path = Path(f"/tmp/{file.filename}")

    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    try:
        # Preprocess (your existing logic)
        spec_tensor = preprocess_audio(temp_path)

        spec_tensor = spec_tensor.to(device)
        spec_tensor = spec_tensor.unsqueeze(0)

        # Inference
        with torch.no_grad():
            outputs = model(spec_tensor)
            pred = int(torch.argmax(outputs, dim=1).cpu().numpy()[0])

        # Format output
        camelot_str, key_text = camelot_output(pred)
        openkey_str = camelot_to_openkey(camelot_str)

        return {
            "status": "success",
            "key": key_text,
            "camelot": camelot_str,
            "openkey": openkey_str,
            "class_id": pred,
            "processing_time": round(time.time() - start, 3)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    finally:
        # Cleanup
        try:
            temp_path.unlink(missing_ok=True)
        except:
            pass


# -----------------------------------------------------------------------------
# Railway entry point
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))

    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=port,
        log_level="info"
    )
