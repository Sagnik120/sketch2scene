"""
Trainer for SketchToScene.
Supports: Apple MPS (M5 Mac), CUDA (Kaggle T4), CPU fallback.
Features: gradient accumulation, mixed precision, checkpoint saving, W&B logging (optional).
"""

from __future__ import annotations
import os
import time
import math
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False


def get_device() -> torch.device:
    if torch.cuda.is_available():
        print("Using CUDA")
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        print("Using Apple MPS")
        return torch.device("mps")
    print("Using CPU")
    return torch.device("cpu")


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        tokenizer,
        lr: float = 3e-4,
        weight_decay: float = 0.1,
        grad_clip: float = 1.0,
        grad_accum: int = 4,
        max_epochs: int = 20,
        warmup_steps: int = 500,
        save_dir: str = "checkpoints",
        log_every: int = 50,
        val_every: int = 500,
        use_wandb: bool = False,
        project_name: str = "sketch2scene",
        device: Optional[torch.device] = None,
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.tokenizer = tokenizer
        self.grad_clip = grad_clip
        self.grad_accum = grad_accum
        self.max_epochs = max_epochs
        self.warmup_steps = warmup_steps
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.log_every = log_every
        self.val_every = val_every

        self.device = device or get_device()
        self.model = self.model.to(self.device)

        # Optimizer (AdamW with decoupled decay)
        decay_params = [p for n, p in model.named_parameters() if "ln" not in n and p.ndim >= 2]
        no_decay     = [p for n, p in model.named_parameters() if "ln" in n or p.ndim < 2]
        self.optimizer = torch.optim.AdamW(
            [{"params": decay_params, "weight_decay": weight_decay},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=lr, betas=(0.9, 0.95),
        )

        # Mixed precision scaler (CUDA only; MPS doesn't support amp yet)
        self.use_amp = self.device.type == "cuda"
        self.scaler  = GradScaler() if self.use_amp else None

        # W&B
        self.use_wandb = use_wandb and HAS_WANDB
        if self.use_wandb:
            wandb.init(project=project_name, name="sketch2scene-run")

        self.step = 0
        self.best_val_loss = float("inf")

    # ── LR Schedule: linear warmup + cosine decay ────────────────────────────

    def _lr(self, step: int, total_steps: int) -> float:
        if step < self.warmup_steps:
            return step / max(1, self.warmup_steps)
        progress = (step - self.warmup_steps) / max(1, total_steps - self.warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * progress)))

    def _update_lr(self, total_steps: int):
        factor = self._lr(self.step, total_steps)
        for pg in self.optimizer.param_groups:
            pg["lr"] = pg.get("initial_lr", pg["lr"]) * factor

    # ── Validation ────────────────────────────────────────────────────────────

    @torch.no_grad()
    def _validate(self) -> float:
        self.model.eval()
        total_loss, n = 0.0, 0
        for batch in self.val_loader:
            sketch  = batch["sketch"].to(self.device)
            captions = batch["caption"]
            out = self.model(sketch, captions[0])  # use first caption for simplicity
            if out["loss"] is not None:
                total_loss += out["loss"].item()
                n += 1
        self.model.train()
        return total_loss / max(n, 1)

    # ── Training loop ─────────────────────────────────────────────────────────

    def train(self):
        total_steps = len(self.train_loader) * self.max_epochs
        # Save initial LRs
        for pg in self.optimizer.param_groups:
            pg["initial_lr"] = pg["lr"]

        print(f"\n{'='*60}")
        print(f"  SketchToScene Training")
        print(f"  Device: {self.device}")
        print(f"  Total steps: {total_steps}")
        print(f"  Grad accum: {self.grad_accum}")
        print(f"{'='*60}\n")

        self.model.train()
        self.optimizer.zero_grad()

        for epoch in range(1, self.max_epochs + 1):
            epoch_loss = 0.0
            t0 = time.time()

            for i, batch in enumerate(self.train_loader):
                sketch   = batch["sketch"].to(self.device)
                captions = batch["caption"]
                caption  = captions[0]  # single caption per batch for simplicity

                # Forward
                if self.use_amp:
                    with autocast():
                        out = self.model(sketch, caption)
                        loss = out["loss"] / self.grad_accum
                    self.scaler.scale(loss).backward()
                else:
                    out = self.model(sketch, caption)
                    loss = out["loss"] / self.grad_accum
                    loss.backward()

                epoch_loss += loss.item() * self.grad_accum

                # Gradient step
                if (i + 1) % self.grad_accum == 0:
                    if self.use_amp:
                        self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    if self.use_amp:
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()
                    self.optimizer.zero_grad()
                    self._update_lr(total_steps)
                    self.step += 1

                    # Logging
                    if self.step % self.log_every == 0:
                        lr_now = self.optimizer.param_groups[0]["lr"]
                        print(f"  [E{epoch} S{self.step}] loss={epoch_loss/(i+1):.4f} lr={lr_now:.2e}")
                        if self.use_wandb:
                            wandb.log({"train/loss": epoch_loss/(i+1), "lr": lr_now}, step=self.step)

                    # Validation
                    if self.step % self.val_every == 0:
                        val_loss = self._validate()
                        print(f"\n  ── Val loss: {val_loss:.4f} ──\n")
                        if self.use_wandb:
                            wandb.log({"val/loss": val_loss}, step=self.step)
                        if val_loss < self.best_val_loss:
                            self.best_val_loss = val_loss
                            self.model.save(self.save_dir / "best")
                            print(f"  ✓ New best model saved (val_loss={val_loss:.4f})")

            elapsed = time.time() - t0
            avg_loss = epoch_loss / len(self.train_loader)
            print(f"\nEpoch {epoch}/{self.max_epochs} | avg_loss={avg_loss:.4f} | {elapsed:.0f}s\n")
            self.model.save(self.save_dir / f"epoch_{epoch:02d}")

        print("Training complete.")
        if self.use_wandb:
            wandb.finish()
