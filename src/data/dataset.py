"""
Dataset loader for SketchToScene.
Primary dataset: SketchyCOCO (sketch + paired COCO image + caption)
Fallback: Sketchy Database + COCO captions
QuickDraw: sketch-only mode (no scenes/)
"""

from __future__ import annotations
import json
import random
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T


def get_sketch_transform(img_size: int = 256) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.Grayscale(num_output_channels=3),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_image_transform(img_size: int = 256) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


def get_augment_transform(img_size: int = 256) -> T.Compose:
    return T.Compose([
        T.Resize((img_size + 32, img_size + 32)),
        T.RandomCrop(img_size),
        T.RandomHorizontalFlip(p=0.3),
        T.Grayscale(num_output_channels=3),
        T.ColorJitter(brightness=0.15, contrast=0.15),
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])


class SketchSceneDataset(Dataset):
    def __init__(
        self,
        sketch_dir: str | Path,
        scene_dir: str | Path | None,
        captions_path: str | Path,
        img_size: int = 256,
        split: str = "train",
        val_frac: float = 0.05,
        augment: bool = True,
        max_samples: Optional[int] = None,
    ):
        self.sketch_dir  = Path(sketch_dir)
        self.scene_dir   = Path(scene_dir) if scene_dir else None
        self.img_size    = img_size
        self.split       = split

        with open(captions_path) as f:
            self.captions: dict[str, str] = json.load(f)

        all_ids = sorted(self.captions.keys())
        random.seed(42)
        random.shuffle(all_ids)
        n_val = max(1, int(len(all_ids) * val_frac))

        if split == "train":
            self.ids = all_ids[n_val:]
        elif split in ("val", "test"):
            self.ids = all_ids[:n_val]
        else:
            self.ids = all_ids

        if max_samples:
            self.ids = self.ids[:max_samples]

        self.sketch_tf = get_augment_transform(img_size) if (augment and split == "train") else get_sketch_transform(img_size)
        self.scene_tf  = get_image_transform(img_size)

        mode = "sketch+scene" if self.scene_dir else "sketch-only"
        print(f"[{split}] {len(self.ids)} samples | mode={mode}")

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx: int) -> dict:
        sid = self.ids[idx]
        caption = self.captions[sid]

        sketch_path = self._find_image(self.sketch_dir, sid)
        sketch = Image.open(sketch_path).convert("RGB")
        sketch = self.sketch_tf(sketch)

        if self.scene_dir is not None:
            scene_path = self._find_image(self.scene_dir, sid)
            scene = Image.open(scene_path).convert("RGB")
            scene = self.scene_tf(scene)
        else:
            scene = torch.zeros_like(sketch)

        return {
            "sketch": sketch,
            "scene": scene,
            "caption": caption,
            "id": sid,
        }

    def _find_image(self, directory: Path, stem: str) -> Path:
        for ext in (".png", ".jpg", ".jpeg", ".PNG", ".JPG"):
            p = directory / f"{stem}{ext}"
            if p.exists():
                return p
        raise FileNotFoundError(f"No image found for id={stem} in {directory}")


def collate_fn(batch: list[dict]) -> dict:
    sketches  = torch.stack([b["sketch"] for b in batch])
    scenes    = torch.stack([b["scene"]  for b in batch])
    captions  = [b["caption"] for b in batch]
    ids       = [b["id"] for b in batch]
    return {"sketch": sketches, "scene": scenes, "caption": captions, "id": ids}


def build_dataloaders(
    data_root: str | Path,
    img_size: int = 256,
    batch_size: int = 8,
    num_workers: int = 4,
    max_samples: Optional[int] = None,
):
    from torch.utils.data import DataLoader

    root = Path(data_root)

    scene_dir = root / "scenes"
    scene_dir = scene_dir if scene_dir.exists() else None
    if scene_dir is None:
        print("[dataset] No scenes/ folder — sketch-only mode (QuickDraw).")

    use_pin = torch.cuda.is_available()

    train_ds = SketchSceneDataset(
        sketch_dir=root / "sketches",
        scene_dir=scene_dir,
        captions_path=root / "captions.json",
        img_size=img_size,
        split="train",
        augment=True,
        max_samples=max_samples,
    )
    val_ds = SketchSceneDataset(
        sketch_dir=root / "sketches",
        scene_dir=scene_dir,
        captions_path=root / "captions.json",
        img_size=img_size,
        split="val",
        augment=False,
        max_samples=max_samples,
    )

    train_dl = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=use_pin, drop_last=True,
    )
    val_dl = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=use_pin,
    )
    return train_dl, val_dl
