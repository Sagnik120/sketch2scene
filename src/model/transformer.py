"""
SketchToScene Decoder-Only Transformer (~125M parameters).

Architecture:
- Token embedding over unified vocab (image codes + text BPE)
- Learned position embedding
- 12 causal self-attention layers with RoPE
- Final LM head projecting to full unified vocab

The model autoregressively generates:
  sketch image tokens → scene description text → completed image patch tokens
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class TransformerConfig:
    vocab_size: int = 66_949        # 8192 img + 50257 text + 6 special (set from tokenizer)
    n_layer: int = 12
    n_head: int = 8
    n_embd: int = 768
    max_seq_len: int = 512
    dropout: float = 0.1
    bias: bool = True


# ── Rotary Positional Embedding ─────────────────────────────────────────────

def precompute_freqs_cis(dim: int, max_seq: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
    t = torch.arange(max_seq).float()
    freqs = torch.outer(t, freqs)  # (T, dim/2)
    return torch.polar(torch.ones_like(freqs), freqs)  # complex


def apply_rotary_emb(q: torch.Tensor, k: torch.Tensor, freqs_cis: torch.Tensor):
    def to_complex(x):
        x = x.float().reshape(*x.shape[:-1], -1, 2)
        return torch.view_as_complex(x)

    def from_complex(x):
        return torch.view_as_real(x).flatten(-2).to(q.dtype)

    T = q.shape[2]
    freqs = freqs_cis[:T]
    # (B, H, T, D/2) complex
    q_r = from_complex(to_complex(q) * freqs)
    k_r = from_complex(to_complex(k) * freqs)
    return q_r, k_r


# ── Attention ────────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.head_dim = cfg.n_embd // cfg.n_head
        self.scale = math.sqrt(self.head_dim)

        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.out = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

        # Causal mask
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(cfg.max_seq_len, cfg.max_seq_len)).view(
                1, 1, cfg.max_seq_len, cfg.max_seq_len
            ),
        )

    def forward(self, x: torch.Tensor, freqs_cis: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        H, D = self.n_head, self.head_dim

        q, k, v = self.qkv(x).split(C, dim=-1)
        q = q.view(B, T, H, D).transpose(1, 2)   # (B, H, T, D)
        k = k.view(B, T, H, D).transpose(1, 2)
        v = v.view(B, T, H, D).transpose(1, 2)

        q, k = apply_rotary_emb(q, k, freqs_cis)

        attn = (q @ k.transpose(-2, -1)) / self.scale
        attn = attn.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        attn = F.softmax(attn, dim=-1)
        attn = self.drop(attn)

        out = (attn @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(out)


# ── Feed-Forward ─────────────────────────────────────────────────────────────

class MLP(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(cfg.n_embd, 4 * cfg.n_embd, bias=cfg.bias)
        self.fc2 = nn.Linear(4 * cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.drop = nn.Dropout(cfg.dropout)

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


# ── Transformer Block ─────────────────────────────────────────────────────────

class Block(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = MLP(cfg)

    def forward(self, x, freqs_cis):
        x = x + self.attn(self.ln1(x), freqs_cis)
        x = x + self.mlp(self.ln2(x))
        return x


# ── Full Model ────────────────────────────────────────────────────────────────

class SketchToSceneTransformer(nn.Module):
    """
    Decoder-only transformer for multimodal sketch-to-scene generation.
    Handles a unified vocabulary of image patch codes + text BPE tokens.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)

        # Weight tying
        self.tok_emb.weight = self.lm_head.weight

        # RoPE
        head_dim = cfg.n_embd // cfg.n_head
        self.register_buffer(
            "freqs_cis",
            precompute_freqs_cis(head_dim, cfg.max_seq_len),
        )

        self._init_weights()
        print(f"SketchToSceneTransformer: {self.num_params()/1e6:.1f}M parameters")

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def forward(
        self,
        input_ids: torch.Tensor,        # (B, T)
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> dict:
        B, T = input_ids.shape
        assert T <= self.cfg.max_seq_len, f"Sequence too long: {T} > {self.cfg.max_seq_len}"

        x = self.tok_emb(input_ids)    # (B, T, C)
        x = self.drop(x)

        for block in self.blocks:
            x = block(x, self.freqs_cis)

        x = self.ln_f(x)
        logits = self.lm_head(x)       # (B, T, V)

        loss = None
        if labels is not None:
            # Shift: predict next token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            # Mask padding (label == -100)
            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100,
            )

        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def generate(
        self,
        prefix: torch.Tensor,           # (1, T) seed tokens
        max_new_tokens: int = 200,
        temperature: float = 0.8,
        top_k: int = 64,
        top_p: float = 0.9,
        stop_token: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with top-k + top-p sampling."""
        self.eval()
        ids = prefix.clone()
        for _ in range(max_new_tokens):
            ids_cond = ids[:, -self.cfg.max_seq_len:]
            out = self(ids_cond)
            next_logits = out["logits"][:, -1, :] / temperature

            # Top-k
            if top_k > 0:
                v, _ = torch.topk(next_logits, min(top_k, next_logits.size(-1)))
                next_logits[next_logits < v[:, -1:]] = float("-inf")

            # Top-p (nucleus)
            if top_p < 1.0:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True)
                cum_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_remove = cum_probs - F.softmax(sorted_logits, dim=-1) > top_p
                sorted_logits[sorted_remove] = float("-inf")
                next_logits.scatter_(1, sorted_idx, sorted_logits)

            probs = F.softmax(next_logits, dim=-1)
            next_id = torch.multinomial(probs, 1)
            ids = torch.cat([ids, next_id], dim=1)

            if stop_token is not None and next_id.item() == stop_token:
                break

        return ids
