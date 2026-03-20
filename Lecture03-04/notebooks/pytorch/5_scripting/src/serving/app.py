"""FastAPI inference server.

Start with:
    python scripts/serve.py --checkpoint runs/best.pt
"""
from __future__ import annotations

import io
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from PIL import Image

from src.serving.predictor import Predictor

import traceback

app = FastAPI(title="CIFAR-10 Classifier", version="1.0")

# Populated at startup via lifespan or by serve.py before uvicorn.run()
_predictor: Predictor | None = None


def init_predictor(checkpoint_path: str | Path, top_k: int = 5) -> None:
    global _predictor
    _predictor = Predictor.from_checkpoint(checkpoint_path, top_k=top_k)
    print(f"Predictor loaded from {checkpoint_path}")


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_loaded": _predictor is not None}


@app.post("/predict")
async def predict(file: UploadFile = File(...)) -> JSONResponse:
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        predictions = _predictor.predict(image)
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))