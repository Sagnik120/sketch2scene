"""Pydantic schemas for the API."""
from pydantic import BaseModel
from typing import Optional


class GenerateRequest(BaseModel):
    temperature: float = 0.7
    top_k: int = 50
    top_p: float = 0.9
    complete_image: bool = True


class GenerateResponse(BaseModel):
    description: str
    sketch_b64: str
    completed_b64: Optional[str] = None


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    device: str
