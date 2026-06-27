import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(32, ch), nn.SiLU(),
            nn.Conv2d(ch, ch, 1),
        )
    def forward(self, x): return x + self.net(x)


class Encoder(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, latent_dim=256, n_res=2):
        super().__init__()
        ch = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, ch, 3, padding=1),
            *[ResBlock(ch) for _ in range(n_res)],
            nn.Conv2d(ch, ch*2, 4, stride=2, padding=1),
            *[ResBlock(ch*2) for _ in range(n_res)],
            nn.Conv2d(ch*2, ch*4, 4, stride=2, padding=1),
            *[ResBlock(ch*4) for _ in range(n_res)],
            nn.Conv2d(ch*4, ch*4, 4, stride=2, padding=1),
            *[ResBlock(ch*4) for _ in range(n_res)],
            nn.Conv2d(ch*4, latent_dim, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)


class Decoder(nn.Module):
    def __init__(self, out_ch=3, base_ch=64, latent_dim=256, n_res=2):
        super().__init__()
        ch = base_ch
        self.net = nn.Sequential(
            nn.Conv2d(latent_dim, ch*4, 1),
            *[ResBlock(ch*4) for _ in range(n_res)],
            nn.ConvTranspose2d(ch*4, ch*4, 4, stride=2, padding=1),
            *[ResBlock(ch*4) for _ in range(n_res)],
            nn.ConvTranspose2d(ch*4, ch*2, 4, stride=2, padding=1),
            *[ResBlock(ch*2) for _ in range(n_res)],
            nn.ConvTranspose2d(ch*2, ch, 4, stride=2, padding=1),
            *[ResBlock(ch) for _ in range(n_res)],
            nn.GroupNorm(32, ch), nn.SiLU(),
            nn.Conv2d(ch, out_ch, 3, padding=1),
            nn.Tanh(),
        )
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None: nn.init.zeros_(m.bias)
    def forward(self, x): return self.net(x)


class VQ(nn.Module):
    """Normalized gradient-based VQ — loss bounded [0,4], no EMA."""
    def __init__(self, n_codes=1024, dim=256, beta=0.25):
        super().__init__()
        self.n, self.beta = n_codes, beta
        self.cb = nn.Embedding(n_codes, dim)
        nn.init.normal_(self.cb.weight, 0, 1)

    def forward(self, z):
        B, D, H, W = z.shape
        zf = rearrange(z, "b d h w -> (b h w) d")
        zn = F.normalize(zf, dim=-1)
        cn = F.normalize(self.cb.weight, dim=-1)
        d = zn.pow(2).sum(1, keepdim=True) - 2*zn@cn.T + cn.pow(2).sum(1)
        idx = d.argmin(1)
        zq_norm = cn[idx]
        codebook_loss   = F.mse_loss(zq_norm, zn.detach())
        commitment_loss = self.beta * F.mse_loss(zn, zq_norm.detach())
        zq_st = z + (zq_norm - zn).detach().view(B, H, W, D).permute(0, 3, 1, 2)
        return zq_st, codebook_loss + commitment_loss, idx.view(B, H, W)


class VQVAE(nn.Module):
    def __init__(self, in_ch=3, base_ch=64, latent_dim=256, n_codes=1024, n_res=2, beta=0.25):
        super().__init__()
        self.enc = Encoder(in_ch, base_ch, latent_dim, n_res)
        self.vq  = VQ(n_codes, latent_dim, beta)
        self.dec = Decoder(in_ch, base_ch, latent_dim, n_res)

    def forward(self, x):
        z = self.enc(x)
        zq, vq_loss, idx = self.vq(z)
        xr = self.dec(zq)
        rl = F.mse_loss(xr, x)
        return {"loss": rl+vq_loss, "recon_loss": rl, "vq_loss": vq_loss, "x_recon": xr, "indices": idx}

    @torch.no_grad()
    def encode_to_tokens(self, x):
        z = self.enc(x)
        _, _, idx = self.vq(z)
        return idx

    @torch.no_grad()
    def decode_from_tokens(self, idx):
        zq = F.normalize(self.vq.cb(idx), dim=-1).permute(0, 3, 1, 2)
        return self.dec(zq)