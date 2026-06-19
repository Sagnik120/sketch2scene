"""
Export trained SketchToScene model to HuggingFace Hub.

Usage:
    python scripts/export_hf.py \
        --checkpoint checkpoints/best \
        --repo Sagnik120/sketch2scene \
        --token YOUR_HF_TOKEN
"""

import argparse
import shutil
from pathlib import Path

from huggingface_hub import HfApi, create_repo


MODEL_CARD = """\
---
language: en
license: mit
tags:
- multimodal
- sketch-to-image
- decoder-only-transformer
- scene-generation
- image-completion
- pytorch
pipeline_tag: image-to-text
---

# 🖼️ SketchToScene

A **decoder-only multimodal transformer** (~125M params) that converts hand-drawn sketches into:
1. **Natural language scene descriptions** (autoregressive text generation)
2. **Completed realistic image patches** (autoregressive image token generation)

## Architecture

- **VQ-VAE** with 8192-code codebook for image tokenization
- **12-layer decoder-only transformer** with RoPE positional encoding
- **Unified vocabulary**: 8192 image codes + 50257 GPT-2 BPE tokens + 6 special tokens
- All in a single forward pass — no cross-attention, no separate encoder

## Usage

```python
from huggingface_hub import snapshot_download
from src.model.sketch2scene import SketchToScene
from src.model.transformer import TransformerConfig

# Load
local = snapshot_download("Sagnik120/sketch2scene")
model = SketchToScene.load(local, TransformerConfig(), device="cpu")

# Inference
from src.inference.visualize import load_sketch
sketch = load_sketch("my_sketch.png")
description = model.generate_description(sketch)
print(description)
```

## Training Data

Trained on SketchyCOCO dataset — paired sketches, scene images, and COCO captions.

## Citation

```bibtex
@misc{sketch2scene2024,
  author = {Sagnik Chandra},
  title  = {SketchToScene: Multimodal Decoder-Only Transformer},
  year   = {2024},
  url    = {https://github.com/Sagnik120/sketch2scene}
}
```
"""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/best")
    p.add_argument("--repo",       default="Sagnik120/sketch2scene")
    p.add_argument("--token",      default=None, help="HF token (or set HF_TOKEN env var)")
    args = p.parse_args()

    import os
    token = args.token or os.getenv("HF_TOKEN")
    if not token:
        raise ValueError("Set HF_TOKEN env var or pass --token")

    api = HfApi(token=token)
    create_repo(args.repo, repo_type="model", exist_ok=True, token=token)

    ckpt = Path(args.checkpoint)
    tmp = Path("tmp_hf_upload")
    tmp.mkdir(exist_ok=True)

    # Copy model files
    for f in ckpt.glob("*.pt"):
        shutil.copy(f, tmp / f.name)

    # Write model card
    (tmp / "README.md").write_text(MODEL_CARD)

    # Upload
    api.upload_folder(
        folder_path=str(tmp),
        repo_id=args.repo,
        repo_type="model",
        token=token,
    )

    shutil.rmtree(tmp, ignore_errors=True)
    print(f"✓ Model uploaded to https://huggingface.co/{args.repo}")


if __name__ == "__main__":
    main()
