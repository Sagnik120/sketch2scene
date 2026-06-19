"""
Unit tests for SketchToScene model components.
Run: pytest tests/ -v
"""

import pytest
import torch
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── VQ-VAE ───────────────────────────────────────────────────────────────────

class TestVQVAE:
    def setup_method(self):
        from src.model.vqvae import VQVAE
        self.model = VQVAE(base_channels=32, latent_dim=64, n_codes=512, n_res=1)
        self.model.eval()

    def test_forward_shape(self):
        x = torch.randn(2, 3, 64, 64)
        out = self.model(x)
        assert "loss" in out
        assert "x_recon" in out
        assert out["x_recon"].shape == x.shape

    def test_encode_decode_roundtrip(self):
        x = torch.randn(1, 3, 64, 64)
        tokens = self.model.encode_to_tokens(x)  # (1, 8, 8) for 64/8
        assert tokens.shape == (1, 8, 8)
        recon = self.model.decode_from_tokens(tokens)
        assert recon.shape == (1, 3, 64, 64)

    def test_token_range(self):
        x = torch.randn(2, 3, 64, 64)
        tokens = self.model.encode_to_tokens(x)
        assert tokens.min() >= 0
        assert tokens.max() < 512


# ── Tokenizer ────────────────────────────────────────────────────────────────

class TestTokenizer:
    def setup_method(self):
        from src.model.tokenizer import UnifiedTokenizer
        self.tok = UnifiedTokenizer()

    def test_vocab_size(self):
        assert self.tok.vocab_size > 8192 + 50000

    def test_text_roundtrip(self):
        text = "A dog runs in a park"
        ids = self.tok.encode_text(text)
        decoded = self.tok.decode_text(ids)
        assert "dog" in decoded.lower()

    def test_build_sequence(self):
        img_tokens = torch.randint(0, 8192, (8 * 8,))
        seq = self.tok.build_sequence(img_tokens, "A park scene.", max_len=256)
        assert seq["input_ids"].shape == (256,)
        assert seq["attention_mask"].shape == (256,)
        # First token should be IMG_START
        assert seq["input_ids"][0].item() == self.tok.IMG_START

    def test_is_image_token(self):
        assert self.tok.is_image_token(0)
        assert self.tok.is_image_token(8191)
        assert not self.tok.is_image_token(8192)

    def test_is_text_token(self):
        assert self.tok.is_text_token(8192)
        assert not self.tok.is_text_token(0)


# ── Transformer ───────────────────────────────────────────────────────────────

class TestTransformer:
    def setup_method(self):
        from src.model.transformer import SketchToSceneTransformer, TransformerConfig
        self.cfg = TransformerConfig(
            vocab_size=1000, n_layer=2, n_head=4, n_embd=128, max_seq_len=64
        )
        self.model = SketchToSceneTransformer(self.cfg)
        self.model.eval()

    def test_forward(self):
        ids = torch.randint(0, 1000, (2, 32))
        out = self.model(ids)
        assert out["logits"].shape == (2, 32, 1000)
        assert out["loss"] is None

    def test_loss(self):
        ids = torch.randint(0, 1000, (2, 32))
        labels = ids.clone()
        labels[:, :5] = -100  # mask prefix
        out = self.model(ids, labels=labels)
        assert out["loss"] is not None
        assert out["loss"].item() > 0

    def test_generate(self):
        prefix = torch.randint(0, 1000, (1, 10))
        out = self.model.generate(prefix, max_new_tokens=20, temperature=1.0, top_k=10)
        assert out.shape[1] <= 30  # prefix + generated

    def test_param_count(self):
        # Tiny model, should be small
        n = self.model.num_params()
        assert n > 0
        print(f"\nTiny model params: {n/1e3:.1f}K")
