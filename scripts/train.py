"""
Training entry point for SketchToScene.

Usage:
    # Dev run on Mac M5
    python scripts/train.py --config configs/train_config.yaml --device mps --batch_size 4

    # Full run on Kaggle (CUDA)
    python scripts/train.py --config configs/train_config.yaml --device cuda --batch_size 16
"""

import argparse
import yaml
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.model.vqvae import VQVAE
from src.model.transformer import SketchToSceneTransformer, TransformerConfig
from src.model.tokenizer import UnifiedTokenizer
from src.model.sketch2scene import SketchToScene
from src.data.dataset import build_dataloaders
from src.training.trainer import Trainer, get_device


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config",      default="configs/train_config.yaml")
    p.add_argument("--device",      default=None)
    p.add_argument("--batch_size",  type=int, default=None)
    p.add_argument("--epochs",      type=int, default=None)
    p.add_argument("--data_root",   default=None)
    p.add_argument("--save_dir",    default="checkpoints")
    p.add_argument("--wandb",       action="store_true")
    p.add_argument("--debug",       action="store_true", help="Small run for debugging")
    return p.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override with CLI args
    if args.batch_size:  cfg["training"]["batch_size"] = args.batch_size
    if args.epochs:      cfg["training"]["max_epochs"] = args.epochs
    if args.data_root:   cfg["data"]["root"] = args.data_root

    device = torch.device(args.device) if args.device else get_device()

    # Tokenizer
    tokenizer = UnifiedTokenizer()
    print(tokenizer)

    # Model config
    model_cfg = TransformerConfig(
        vocab_size=tokenizer.vocab_size,
        **cfg["model"],
    )

    # Build model
    vqvae = VQVAE(
        in_channels=cfg["vqvae"]["in_channels"],
        base_channels=cfg["vqvae"]["base_channels"],
        latent_dim=cfg["vqvae"]["latent_dim"],
        n_codes=cfg["vqvae"]["n_codes"],
    )
    transformer = SketchToSceneTransformer(model_cfg)
    model = SketchToScene(vqvae, transformer, tokenizer)

    # Data
    max_samples = 100 if args.debug else None
    train_dl, val_dl = build_dataloaders(
        data_root=cfg["data"]["root"],
        img_size=cfg["data"]["img_size"],
        batch_size=cfg["training"]["batch_size"],
        num_workers=cfg["data"]["num_workers"],
        max_samples=max_samples,
    )

    # Train
    trainer = Trainer(
        model=model,
        train_loader=train_dl,
        val_loader=val_dl,
        tokenizer=tokenizer,
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
        grad_accum=cfg["training"]["grad_accum"],
        max_epochs=cfg["training"]["max_epochs"],
        warmup_steps=cfg["training"]["warmup_steps"],
        save_dir=args.save_dir,
        use_wandb=args.wandb,
        device=device,
    )
    trainer.train()


if __name__ == "__main__":
    main()
