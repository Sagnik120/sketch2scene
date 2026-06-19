"""
Download and prepare datasets for SketchToScene.

Datasets:
  - sketchy_coco: SketchyCOCO (sketch + COCO image + caption)
  - quickdraw:    Google QuickDraw (synthetic sketches, good for pretraining)

Usage:
    python scripts/download_data.py --dataset sketchy_coco --output data/raw/
"""

import argparse
import os
import json
import urllib.request
from pathlib import Path


DATASETS = {
    "sketchy_coco": {
        "description": "SketchyCOCO: paired sketches, scene images, COCO captions",
        "instructions": """
SketchyCOCO requires manual download:
1. Visit: https://github.com/sysu-imsl/SketchyCOCO
2. Download 'sketches.zip' and 'scenes.zip'
3. Extract to data/raw/sketchy_coco/sketches/ and data/raw/sketchy_coco/scenes/
4. Run: python scripts/download_data.py --prepare --dataset sketchy_coco
""",
    },
    "quickdraw": {
        "description": "Google QuickDraw — 50M vector sketches, great for pretraining",
        "urls": {
            "cat":      "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/cat.npy",
            "dog":      "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/dog.npy",
            "house":    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/house.npy",
            "mountain": "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/mountain.npy",
            "tree":     "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/tree.npy",
            "car":      "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/car.npy",
            "bird":     "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/bird.npy",
        },
    },
}


def download_quickdraw(output: Path, max_per_class: int = 5000):
    """Download QuickDraw classes and convert to PNG + captions.json."""
    import numpy as np
    from PIL import Image

    sketch_dir = output / "quickdraw" / "sketches"
    sketch_dir.mkdir(parents=True, exist_ok=True)
    captions: dict[str, str] = {}

    for cls_name, url in DATASETS["quickdraw"]["urls"].items():
        npy_path = output / f"{cls_name}.npy"
        if not npy_path.exists():
            print(f"Downloading {cls_name}...")
            urllib.request.urlretrieve(url, npy_path)

        data = np.load(npy_path)[:max_per_class]  # (N, 784)
        print(f"  Processing {len(data)} {cls_name} sketches...")

        for i, vec in enumerate(data):
            sid = f"{cls_name}_{i:05d}"
            img = Image.fromarray(vec.reshape(28, 28).astype(np.uint8))
            img = img.resize((256, 256), Image.NEAREST)
            img.save(sketch_dir / f"{sid}.png")
            captions[sid] = f"A hand-drawn sketch of a {cls_name}."

    cap_path = output / "quickdraw" / "captions.json"
    with open(cap_path, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"\n✓ QuickDraw prepared: {len(captions)} samples → {output}/quickdraw/")
    print(f"  Use --data_root data/raw/quickdraw in training (no scene images, text-only mode)")


def prepare_sketchy_coco(data_root: Path):
    """Validate and create captions.json for SketchyCOCO."""
    sketch_dir = data_root / "sketchy_coco" / "sketches"
    if not sketch_dir.exists():
        print("⚠ sketches/ folder not found. Follow instructions above.")
        return

    sketches = list(sketch_dir.glob("*.png")) + list(sketch_dir.glob("*.jpg"))
    print(f"Found {len(sketches)} sketches.")

    # Generate dummy captions (replace with real COCO captions)
    captions = {p.stem: f"A scene containing {p.stem.replace('_', ' ')}." for p in sketches}
    out = data_root / "sketchy_coco" / "captions.json"
    with open(out, "w") as f:
        json.dump(captions, f, indent=2)
    print(f"✓ captions.json written: {out}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=["sketchy_coco", "quickdraw", "all"], default="quickdraw")
    p.add_argument("--output",  default="data/raw")
    p.add_argument("--prepare", action="store_true")
    p.add_argument("--max_per_class", type=int, default=5000)
    args = p.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    if args.dataset in ("quickdraw", "all"):
        download_quickdraw(out, args.max_per_class)

    if args.dataset in ("sketchy_coco", "all"):
        if args.prepare:
            prepare_sketchy_coco(out)
        else:
            print(DATASETS["sketchy_coco"]["instructions"])


if __name__ == "__main__":
    main()
