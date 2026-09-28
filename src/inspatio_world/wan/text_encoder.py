"""umT5-XXL text encoder and tokenizer: a prompt -> the 512-token context the DiT reads.

The DiT cross-attends to a fixed-length context: the prompt is cleaned (HTML entities decoded,
whitespace collapsed), tokenized with the umT5 SentencePiece vocabulary (plus `</s>`), padded or
truncated to 512 tokens, and run through the 24-layer encoder. Positions past the prompt are then
zeroed, as upstream does; the DiT still attends to all 512 positions.

umT5 differs from T5 in giving every layer its own relative-position bias table (32 buckets,
bidirectional, max distance 128). Attention is unscaled, the feed-forward is gated GELU (tanh).

Numerics follow the upstream inference path: weights and activations in bf16, RMS statistics in
float32. Only upstream's `ftfy` mojibake repair is dropped from the cleaning step.

Ref: umT5 arXiv:2304.09151
# Adapted from https://github.com/Wan-Video/Wan2.1/blob/main/wan/modules/t5.py
"""

from __future__ import annotations

import html
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, override

import torch
import torch.nn.functional as F
from einops import rearrange
from tokenizers import Tokenizer
from torch import nn

from ..types import (
    AttentionMaskBias,
    RelativePositionBuckets,
    TextContext,
    TokenIds,
    TokenMask,
    Tokens,
)
from ..utils.hub import HubModule

PAD_ID = 0


@dataclass(frozen=True, slots=True)
class UMT5Config:
    """Architecture quantities of the umT5-XXL encoder."""

    vocab_size: int = 256_384
    dim: int = 4096
    ffn_dim: int = 10_240
    num_heads: int = 64
    num_layers: int = 24
    num_buckets: int = 32
    max_distance: int = 128
    text_length: int = 512
    eps: float = 1e-6


class PromptTokenizer:
    """Clean, tokenize and pad prompts to the fixed context length."""

    def __init__(self, tokenizer_file: Path, text_length: int) -> None:
        self.tokenizer = Tokenizer.from_file(str(tokenizer_file))
        self.tokenizer.enable_truncation(text_length)
        self.tokenizer.enable_padding(length=text_length, pad_id=PAD_ID, pad_token="<pad>")

    def __call__(self, prompts: list[str]) -> tuple[TokenIds, TokenMask]:
        encodings = self.tokenizer.encode_batch([_clean(prompt) for prompt in prompts])
        ids = torch.tensor([encoding.ids for encoding in encodings], dtype=torch.long)
        mask = torch.tensor([encoding.attention_mask for encoding in encodings], dtype=torch.bool)
        return ids, mask


def _clean(text: str) -> str:
    text = html.unescape(html.unescape(text)).strip()
    return re.sub(r"\s+", " ", text).strip()


class T5LayerNorm(nn.Module):
    """T5's RMS norm without mean subtraction; statistics in float32, output in weight dtype."""

    def __init__(self, dim: int, eps: float) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    @override
    def forward(self, x: Tokens) -> Tokens:
        normed = x * torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return self.weight * normed.type_as(self.weight)


class RelativePositionBias(nn.Module):
    """Bidirectional bucketed relative-position bias: one learned scalar per (bucket, head)."""

    def __init__(self, config: UMT5Config) -> None:
        super().__init__()
        self.num_buckets = config.num_buckets
        self.max_distance = config.max_distance
        self.embedding = nn.Embedding(config.num_buckets, config.num_heads)

    @override
    def forward(self, length: int) -> AttentionMaskBias:
        positions = torch.arange(length, device=self.embedding.weight.device)
        buckets = self._bucket(positions[None, :] - positions[:, None])
        return rearrange(self.embedding(buckets), "q k h -> 1 h q k")

    def _bucket(self, relative: RelativePositionBuckets) -> RelativePositionBuckets:
        # Half the buckets per direction; exact below max_exact, log-spaced up to max_distance
        half = self.num_buckets // 2
        buckets = (relative > 0).long() * half
        distance = relative.abs()
        max_exact = half // 2
        log_ratio = torch.log(distance.float() / max_exact) / math.log(
            self.max_distance / max_exact
        )
        large = (max_exact + log_ratio * (half - max_exact)).long().clamp(max=half - 1)
        return buckets + torch.where(distance < max_exact, distance, large)


class T5Block(nn.Module):
    """Pre-norm self-attention (with its own position bias) and gated-GELU feed-forward."""

    def __init__(self, config: UMT5Config) -> None:
        super().__init__()
        dim = config.dim
        self.num_heads = config.num_heads
        self.attention_norm = T5LayerNorm(dim, config.eps)
        self.query = nn.Linear(dim, dim, bias=False)
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        self.position_bias = RelativePositionBias(config)
        self.ffn_norm = T5LayerNorm(dim, config.eps)
        self.gate = nn.Linear(dim, config.ffn_dim, bias=False)
        self.fc1 = nn.Linear(dim, config.ffn_dim, bias=False)
        self.fc2 = nn.Linear(config.ffn_dim, dim, bias=False)

    @override
    def forward(self, x: Tokens, padding_bias: AttentionMaskBias) -> Tokens:
        normed = self.attention_norm(x)
        query, key, value = (
            rearrange(layer(normed), "b l (h d) -> b h l d", h=self.num_heads)
            for layer in (self.query, self.key, self.value)
        )
        bias = self.position_bias(x.shape[1]).to(x.dtype) + padding_bias
        # T5 attention is unscaled
        attended = F.scaled_dot_product_attention(query, key, value, attn_mask=bias, scale=1.0)
        x = x + self.output(rearrange(attended, "b h l d -> b l (h d)"))

        normed = self.ffn_norm(x)
        hidden = self.fc1(normed) * F.gelu(self.gate(normed), approximate="tanh")
        return x + self.fc2(hidden)


class UMT5Encoder(nn.Module, HubModule):
    """umT5-XXL encoder stack: token ids + padding mask -> contextual embeddings."""

    config_class: ClassVar[type] = UMT5Config

    def __init__(self, config: UMT5Config) -> None:
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.blocks = nn.ModuleList([T5Block(config) for _ in range(config.num_layers)])
        self.norm = T5LayerNorm(config.dim, config.eps)

    @override
    def forward(self, ids: TokenIds, mask: TokenMask) -> TextContext:
        x = self.token_embedding(ids)
        padding_bias = torch.zeros(mask.shape, dtype=x.dtype, device=x.device)
        padding_bias = padding_bias.masked_fill(~mask, torch.finfo(x.dtype).min)
        padding_bias = rearrange(padding_bias, "b l -> b 1 1 l")
        for block in self.blocks:
            x = block(x, padding_bias)
        return self.norm(x)


class TextEncoder:
    """Prompt strings -> zero-padded umT5 context `(B, 512, 4096)`, as the DiT expects."""

    def __init__(self, encoder: UMT5Encoder, tokenizer: PromptTokenizer) -> None:
        self.encoder = encoder
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def __call__(self, prompts: list[str]) -> TextContext:
        device = self.encoder.token_embedding.weight.device
        ids, mask = self.tokenizer(prompts)
        ids, mask = ids.to(device), mask.to(device)
        context = self.encoder(ids, mask)
        return context * mask[..., None].to(context.dtype)
