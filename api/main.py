"""
SketchToScene FastAPI REST API
"""

from __future__ import annotations
import io
import os
import base64
from pathlib import Path
from contextlib import asynccontextmanager

import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image

from .schemas import GenerateRequest, GenerateResponse, HealthResponse
from .utils import image_to_base64, base64_to_tensor


# ── Model loading ─────────────────────────────────────────────────────────────

MODEL = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global MODEL
    model_path = os.getenv("MODEL_PATH", "checkpoints/best")
    device = os.getenv("DEVICE", "cpu")

    if Path(model_path).exists():
        from src.model.sketch2scene import SketchToScene
        from src.model.transformer import TransformerConfig
        MODEL = SketchToScene.load(model_path, TransformerConfig(), device=device)
        print(f"✓ Model loaded from {model_path} on {device}")
    else:
        print(f"⚠ No model found at {model_path} — running without model")
    yield
    MODEL = None


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="SketchToScene API",
    description=(
        "Multimodal decoder-only transformer that converts hand-drawn sketches "
        "into realistic scene descriptions and completes image patches."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return {
        "status": "ok",
        "model_loaded": MODEL is not None,
        "device": os.getenv("DEVICE", "cpu"),
    }


@app.post("/generate", response_model=GenerateResponse)
async def generate(
    file: UploadFile = File(...),
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    complete_image: bool = True,
):
    """Upload a sketch image, get scene description + completed image."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded. Check MODEL_PATH env var.")

    # Read image
    contents = await file.read()
    try:
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as e:
        raise HTTPException(400, f"Invalid image: {e}")

    device = next(MODEL.parameters()).device
    sketch_tensor = base64_to_tensor(pil_img, device=str(device))

    description = MODEL.generate_description(
        sketch_tensor, temperature=temperature, top_k=top_k, top_p=top_p
    )

    completed_b64 = None
    if complete_image:
        completed_tensor = MODEL.complete_image(sketch_tensor, description)
        completed_b64 = image_to_base64(completed_tensor)

    sketch_b64 = image_to_base64(sketch_tensor)

    return {
        "description": description,
        "sketch_b64": sketch_b64,
        "completed_b64": completed_b64,
    }


@app.post("/describe", response_model=dict)
async def describe_only(
    file: UploadFile = File(...),
    temperature: float = 0.7,
    top_k: int = 50,
):
    """Faster endpoint: only generate text description."""
    if MODEL is None:
        raise HTTPException(503, "Model not loaded.")

    contents = await file.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    device = str(next(MODEL.parameters()).device)
    sketch_tensor = base64_to_tensor(pil_img, device=device)
    description = MODEL.generate_description(sketch_tensor, temperature=temperature, top_k=top_k)
    return {"description": description}
