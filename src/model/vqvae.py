"""
VQ-VAE: Vector Quantized Variational Autoencoder
Converts images into discrete codebook tokens for the transformer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.GroupNorm(32, channels),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1),
        )

    def forward(self, x):
        return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_channels: int = 3, base_channels: int = 128, latent_dim: int = 256, n_res: int = 2):
        super().__init__()
        ch = base_channels
        self.input_proj = nn.Conv2d(in_channels, ch, 3, padding=1)
        self.down = nn.Sequential(
            *[ResBlock(ch) for _ in range(n_res)],
            nn.Conv2d(ch, ch * 2, 4, stride=2, padding=1),  # /2
            *[ResBlock(ch * 2) for _ in range(n_res)],
            nn.Conv2d(ch * 2, ch * 4, 4, stride=2, padding=1),  # /4
            *[ResBlock(ch * 4) for _ in range(n_res)],
            nn.Conv2d(ch * 4, ch * 4, 4, stride=2, padding=1),  # /8
            *[ResBlock(ch * 4) for _ in range(n_res)],
        )
        self.out_proj = nn.Conv2d(ch * 4, latent_dim, 1)

    def forward(self, x):
        x = self.input_proj(x)
        x = self.down(x)
        return self.out_proj(x)


class Decoder(nn.Module):
    def __init__(self, out_channels: int = 3, base_channels: int = 128, latent_dim: int = 256, n_res: int = 2):
        super().__init__()
        ch = base_channels
        self.in_proj = nn.Conv2d(latent_dim, ch * 4, 1)
        self.up = nn.Sequential(
            *[ResBlock(ch * 4) for _ in range(n_res)],
            nn.ConvTranspose2d(ch * 4, ch * 4, 4, stride=2, padding=1),
            *[ResBlock(ch * 4) for _ in range(n_res)],
            nn.ConvTranspose2d(ch * 4, ch * 2, 4, stride=2, padding=1),
            *[ResBlock(ch * 2) for _ in range(n_res)],
            nn.ConvTranspose2d(ch * 2, ch, 4, stride=2, padding=1),
            *[ResBlock(ch) for _ in range(n_res)],
        )
        self.out_proj = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, out_channels, 3, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        x = self.in_proj(x)
        x = self.up(x)
        return self.out_proj(x)


class VectorQuantizer(nn.Module):
    """
    EMA-updated codebook (more stable than gradient-based).
    """
    def __init__(self, n_codes: int = 8192, code_dim: int = 256, beta: float = 0.25, ema_decay: float = 0.99):
        super().__init__()
        self.n_codes = n_codes
        self.code_dim = code_dim
        self.beta = beta
        self.ema_decay = ema_decay

        self.codebook = nn.Embedding(n_codes, code_dim)
        nn.init.uniform_(self.codebook.weight, -1 / n_codes, 1 / n_codes)

        # EMA buffers
        self.register_buffer("ema_cluster_size", torch.zeros(n_codes))
        self.register_buffer("ema_embed_sum", self.codebook.weight.data.clone())

    def forward(self, z: torch.Tensor):
        # z: (B, D, H, W) → flatten spatial
        B, D, H, W = z.shape
        z_flat = rearrange(z, "b d h w -> (b h w) d")  # (N, D)

        # L2 distances to codebook
        dist = (
            z_flat.pow(2).sum(1, keepdim=True)
            - 2 * z_flat @ self.codebook.weight.T
            + self.codebook.weight.pow(2).sum(1)
        )  # (N, K)
        indices = dist.argmin(dim=1)  # (N,)
        z_q = self.codebook(indices)  # (N, D)

        # EMA update during training
        if self.training:
            one_hot = F.one_hot(indices, self.n_codes).float()
            self.ema_cluster_size.mul_(self.ema_decay).add_(one_hot.sum(0) * (1 - self.ema_decay))
            embed_sum = one_hot.T @ z_flat.detach()
            self.ema_embed_sum.mul_(self.ema_decay).add_(embed_sum * (1 - self.ema_decay))
            n = self.ema_cluster_size.sum()
            smoothed = (self.ema_cluster_size + 1e-5) / (n + self.n_codes * 1e-5) * n
            self.codebook.weight.data.copy_(self.ema_embed_sum / smoothed.unsqueeze(1))

        # Straight-through estimator
        z_q_st = z + (z_q - z_flat).detach().view(B, H, W, D).permute(0, 3, 1, 2)
        z_q_spatial = z_q.view(B, H, W, D).permute(0, 3, 1, 2)

        commitment_loss = self.beta * F.mse_loss(z_flat.detach(), z_q)
        indices_spatial = indices.view(B, H, W)

        return z_q_st, z_q_spatial, commitment_loss, indices_spatial


class VQVAE(nn.Module):
    """
    Full VQ-VAE: Encoder → VQ → Decoder
    Image (B,3,H,W) → discrete tokens (B, H/8, W/8) + reconstructed image
    """
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 128,
        latent_dim: int = 256,
        n_codes: int = 8192,
        n_res: int = 2,
        beta: float = 0.25,
    ):
        super().__init__()
        self.encoder = Encoder(in_channels, base_channels, latent_dim, n_res)
        self.vq = VectorQuantizer(n_codes, latent_dim, beta)
        self.decoder = Decoder(in_channels, base_channels, latent_dim, n_res)

    def forward(self, x: torch.Tensor):
        z = self.encoder(x)
        z_q_st, z_q, commit_loss, indices = self.vq(z)
        x_recon = self.decoder(z_q_st)
        recon_loss = F.mse_loss(x_recon, x)
        total_loss = recon_loss + commit_loss
        return {
            "loss": total_loss,
            "recon_loss": recon_loss,
            "commit_loss": commit_loss,
            "x_recon": x_recon,
            "indices": indices,  # image tokens
        }

    @torch.no_grad()
    def encode_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """Encode image to discrete token indices."""
        z = self.encoder(x)
        _, _, _, indices = self.vq(z)
        return indices  # (B, H/8, W/8)

    @torch.no_grad()
    def decode_from_tokens(self, indices: torch.Tensor) -> torch.Tensor:
        """Decode discrete token indices to image."""
        z_q = self.vq.codebook(indices)  # (B, H, W, D)
        z_q = z_q.permute(0, 3, 1, 2)   # (B, D, H, W)
        return self.decoder(z_q)
