"""
Unified Tokenizer for SketchToScene.
Merges image VQ codes and GPT-2 BPE text tokens into one vocabulary.

Vocabulary layout:
  [0 .. 8191]        → image patch codes (VQ-VAE codebook)
  [8192 .. 8192+N_TEXT-1] → text BPE tokens (GPT-2 vocab)
  Special tokens:
    <IMG_START>  = 8192 + N_TEXT
    <IMG_END>    = 8192 + N_TEXT + 1
    <TXT_START>  = 8192 + N_TEXT + 2
    <TXT_END>    = 8192 + N_TEXT + 3
    <PAD>        = 8192 + N_TEXT + 4
    <MASK>       = 8192 + N_TEXT + 5
"""

from __future__ import annotations
import torch
from transformers import GPT2Tokenizer


N_IMAGE_CODES = 8192
_SPECIAL = ["<IMG_START>", "<IMG_END>", "<TXT_START>", "<TXT_END>", "<PAD>", "<MASK>"]


class UnifiedTokenizer:
    def __init__(self, gpt2_name: str = "gpt2"):
        self.text_tok = GPT2Tokenizer.from_pretrained(gpt2_name)
        self.text_tok.add_special_tokens({"pad_token": "<|endoftext|>"})

        self.n_image = N_IMAGE_CODES
        self.n_text = len(self.text_tok)
        self.offset = self.n_image  # text token i → vocab id (offset + i)

        # Special token IDs
        special_start = self.n_image + self.n_text
        self.IMG_START = special_start
        self.IMG_END   = special_start + 1
        self.TXT_START = special_start + 2
        self.TXT_END   = special_start + 3
        self.PAD       = special_start + 4
        self.MASK      = special_start + 5

        self.vocab_size = self.n_image + self.n_text + len(_SPECIAL)

    # ── image ──────────────────────────────────────────────────────────────
    def image_tokens_to_ids(self, indices: torch.Tensor) -> torch.Tensor:
        """VQ indices (B, H, W) → flat token ids in unified vocab."""
        return indices.long()  # already in [0, n_image)

    def ids_to_image_tokens(self, ids: torch.Tensor) -> torch.Tensor:
        return ids.long()

    # ── text ───────────────────────────────────────────────────────────────
    def encode_text(self, text: str) -> list[int]:
        raw = self.text_tok.encode(text, add_special_tokens=False)
        return [r + self.offset for r in raw]

    def decode_text(self, ids: list[int]) -> str:
        raw = [i - self.offset for i in ids if 0 <= i - self.offset < self.n_text]
        return self.text_tok.decode(raw, skip_special_tokens=True)

    # ── sequence building ──────────────────────────────────────────────────
    def build_sequence(
        self,
        image_token_ids: torch.Tensor,  # (H*W,) flat
        scene_text: str | None = None,
        max_len: int = 512,
    ) -> dict[str, torch.Tensor]:
        """
        Build full token sequence:
          <IMG_START> img_tok... <IMG_END> <TXT_START> txt_tok... <TXT_END>

        Returns input_ids and attention_mask.
        """
        img_flat = image_token_ids.flatten().tolist()
        seq = [self.IMG_START] + img_flat + [self.IMG_END]

        if scene_text is not None:
            txt_ids = self.encode_text(scene_text)
            seq += [self.TXT_START] + txt_ids + [self.TXT_END]

        # Truncate
        seq = seq[:max_len]
        # Pad
        pad_len = max_len - len(seq)
        attn = [1] * len(seq) + [0] * pad_len
        seq  = seq + [self.PAD] * pad_len

        return {
            "input_ids": torch.tensor(seq, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
        }

    def build_generation_prefix(self, image_token_ids: torch.Tensor) -> torch.Tensor:
        """
        Prefix for autoregressive generation:
          <IMG_START> img_tok... <IMG_END> <TXT_START>
        The model then generates text tokens followed by <TXT_END>.
        """
        img_flat = image_token_ids.flatten().tolist()
        seq = [self.IMG_START] + img_flat + [self.IMG_END, self.TXT_START]
        return torch.tensor(seq, dtype=torch.long)

    # ── helpers ────────────────────────────────────────────────────────────
    def is_image_token(self, id_: int) -> bool:
        return 0 <= id_ < self.n_image

    def is_text_token(self, id_: int) -> bool:
        return self.offset <= id_ < self.offset + self.n_text

    def __repr__(self):
        return (
            f"UnifiedTokenizer("
            f"vocab_size={self.vocab_size}, "
            f"n_image={self.n_image}, "
            f"n_text={self.n_text})"
        )
