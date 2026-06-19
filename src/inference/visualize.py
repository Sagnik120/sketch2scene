"""
Inference utilities for SketchToScene.
"""

from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt


def load_sketch(path: str | Path, img_size: int = 256) -> torch.Tensor:
    """Load a sketch image and return a (1, 3, H, W) tensor."""
    img = Image.open(path).convert("RGB")
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    return tf(img).unsqueeze(0)


def tensor_to_pil(t: torch.Tensor) -> Image.Image:
    """Convert normalized (1, 3, H, W) or (3, H, W) tensor to PIL."""
    if t.dim() == 4:
        t = t.squeeze(0)
    t = (t * 0.5 + 0.5).clamp(0, 1)
    return TF.to_pil_image(t.cpu())


def visualize_results(
    sketch: torch.Tensor,
    description: str,
    completed: Optional[torch.Tensor] = None,
    save_path: Optional[str] = None,
) -> Image.Image:
    """
    Create a side-by-side visualization:
      [Sketch] | [Generated Description] | [Completed Image]
    """
    fig, axes = plt.subplots(1, 3 if completed is not None else 2, figsize=(14, 5))
    fig.patch.set_facecolor("#1a1a2e")

    def show(ax, img_tensor, title):
        pil = tensor_to_pil(img_tensor)
        ax.imshow(np.array(pil))
        ax.set_title(title, color="white", fontsize=12, fontweight="bold")
        ax.axis("off")

    show(axes[0], sketch, "Input Sketch")

    # Description panel
    axes[1].set_facecolor("#16213e")
    wrapped = _wrap_text(description, 40)
    axes[1].text(
        0.5, 0.5, wrapped,
        transform=axes[1].transAxes,
        ha="center", va="center",
        fontsize=10, color="#e2e8f0",
        wrap=True,
    )
    axes[1].set_title("Generated Description", color="white", fontsize=12, fontweight="bold")
    axes[1].axis("off")

    if completed is not None:
        show(axes[2], completed, "Completed Scene")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=150, facecolor=fig.get_facecolor())
        print(f"Saved to {save_path}")

    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    img_array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8).reshape(h, w, 3)
    plt.close(fig)
    return Image.fromarray(img_array)


def _wrap_text(text: str, width: int) -> str:
    words = text.split()
    lines, line = [], []
    for w in words:
        line.append(w)
        if len(" ".join(line)) > width:
            lines.append(" ".join(line[:-1]))
            line = [w]
    if line:
        lines.append(" ".join(line))
    return "\n".join(lines)


@torch.no_grad()
def run_inference(
    model,
    sketch_path: str,
    device: str = "cpu",
    img_size: int = 256,
    temperature: float = 0.7,
    top_k: int = 50,
    top_p: float = 0.9,
    complete_image: bool = True,
) -> dict:
    """Full inference pipeline: sketch → description + completed image."""
    sketch = load_sketch(sketch_path, img_size).to(device)

    print("Generating scene description...")
    description = model.generate_description(
        sketch,
        max_text_tokens=150,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
    )
    print(f"Description: {description}")

    completed = None
    if complete_image:
        print("Completing image...")
        completed = model.complete_image(sketch, description)

    return {
        "sketch": sketch,
        "description": description,
        "completed": completed,
    }
