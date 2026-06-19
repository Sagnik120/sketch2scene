"""API utility functions."""
from __future__ import annotations
import io
import base64

import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image


def image_to_base64(tensor: torch.Tensor) -> str:
    if tensor.dim() == 4:
        tensor = tensor.squeeze(0)
    tensor = (tensor * 0.5 + 0.5).clamp(0, 1)
    pil = TF.to_pil_image(tensor.cpu())
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def base64_to_tensor(pil_img: Image.Image, img_size: int = 256, device: str = "cpu") -> torch.Tensor:
    tf = T.Compose([
        T.Resize((img_size, img_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.5]*3, [0.5]*3),
    ])
    return tf(pil_img).unsqueeze(0).to(device)
