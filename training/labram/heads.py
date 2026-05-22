"""Output heads on top of LaBraM's pooled CLS embedding (or patch tokens).

All heads expose:
  - `forward(features)` where `features` is `(B, D)` (CLS) — except the token
    decoder which also accepts `(B, S, D)` patch tokens for cross-attention.
  - `loss(logits, target, mask)` where `mask` is a `(B,)` boolean tensor of
    trials that carry a label for this head.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import nn

from training.labram.config import HeadConfig


def _masked_mean(loss_per_item: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean over masked-in entries; zero (with grad) if mask is empty."""
    m = mask.to(loss_per_item.dtype)
    denom = m.sum().clamp_min(1.0)
    return (loss_per_item * m).sum() / denom


class ClassifyHead(nn.Module):
    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def loss(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per = F.cross_entropy(logits, target, reduction="none")
        return _masked_mean(per, mask)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        return logits.argmax(-1)


class RegressHead(nn.Module):
    def __init__(self, embed_dim: int, dim: int, loss: str = "mse"):
        super().__init__()
        hidden = max(embed_dim, 64)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
        )
        self.dim = dim
        self.loss_kind = loss

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.mlp(features)

    def loss(self, pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.loss_kind == "huber":
            per = F.smooth_l1_loss(pred, target, reduction="none").mean(-1)
        else:
            per = F.mse_loss(pred, target, reduction="none").mean(-1)
        return _masked_mean(per, mask)

    def predict(self, pred: torch.Tensor) -> torch.Tensor:
        return pred


class GridHead(nn.Module):
    def __init__(self, embed_dim: int, rows: int, cols: int):
        super().__init__()
        self.rows = rows
        self.cols = cols
        self.fc = nn.Linear(embed_dim, rows * cols)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.fc(features)

    def loss(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        per = F.cross_entropy(logits, target, reduction="none")
        return _masked_mean(per, mask)

    def predict(self, logits: torch.Tensor) -> torch.Tensor:
        idx = logits.argmax(-1)
        return torch.stack([idx // self.cols, idx % self.cols], dim=-1)


class TokenDecoderHead(nn.Module):
    """Small autoregressive decoder cross-attending over LaBraM patch tokens.

    Vocab indices 0..V-1 are word ids; index V is `<eos>`. Padding uses -1
    so callers can keep the loss agnostic to seq length via `ignore_index=-1`.
    """

    def __init__(
        self,
        embed_dim: int,
        vocab_size: int,
        max_len: int,
        d_model: int = 128,
        n_layers: int = 2,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.max_len = max_len
        self.vocab_size = vocab_size  # excludes <eos>
        self.bos_id = vocab_size + 1
        self.eos_id = vocab_size
        self.tok_emb = nn.Embedding(vocab_size + 2, d_model)  # words + <eos> + <bos>
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        self.memory_proj = nn.Linear(embed_dim, d_model)
        layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True, activation="gelu",
        )
        self.decoder = nn.TransformerDecoder(layer, num_layers=n_layers)
        self.out_proj = nn.Linear(d_model, vocab_size + 1)  # words + <eos>
        nn.init.trunc_normal_(self.pos_emb, std=0.02)

    def _causal_mask(self, n: int, device: torch.device) -> torch.Tensor:
        return torch.triu(torch.ones(n, n, dtype=torch.bool, device=device), diagonal=1)

    def forward(self, patch_tokens: torch.Tensor, target: Optional[torch.Tensor] = None):
        """patch_tokens: (B, S, embed_dim). target: (B, L) with -1 padding (training)."""
        memory = self.memory_proj(patch_tokens)
        if target is None:
            return memory
        B, L = target.shape
        # Teacher forcing: prepend <bos>, predict shifted target.
        bos = torch.full((B, 1), self.bos_id, dtype=torch.long, device=target.device)
        tgt_in = torch.cat([bos, target[:, :-1].clamp_min(0)], dim=1)
        tgt_emb = self.tok_emb(tgt_in) + self.pos_emb[:, :L]
        out = self.decoder(
            tgt_emb, memory,
            tgt_mask=self._causal_mask(L, target.device),
        )
        return self.out_proj(out)  # (B, L, vocab+1)

    def loss(self, logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # logits: (B, L, V+1) ; target: (B, L) with -1 padding
        B, L, V = logits.shape
        per_token = F.cross_entropy(
            logits.reshape(B * L, V),
            target.reshape(B * L),
            ignore_index=-1,
            reduction="none",
        ).reshape(B, L)
        # Per-trial loss: mean over non-padded tokens.
        valid = (target != -1).to(per_token.dtype)
        denom = valid.sum(-1).clamp_min(1.0)
        per_trial = (per_token * valid).sum(-1) / denom
        return _masked_mean(per_trial, mask)

    @torch.no_grad()
    def generate(self, patch_tokens: torch.Tensor) -> torch.Tensor:
        memory = self.memory_proj(patch_tokens)
        B = memory.size(0)
        device = memory.device
        out_ids = torch.full((B, 1), self.bos_id, dtype=torch.long, device=device)
        for _ in range(self.max_len):
            L = out_ids.size(1)
            emb = self.tok_emb(out_ids) + self.pos_emb[:, :L]
            dec = self.decoder(emb, memory, tgt_mask=self._causal_mask(L, device))
            next_logits = self.out_proj(dec[:, -1])
            next_id = next_logits.argmax(-1, keepdim=True)
            out_ids = torch.cat([out_ids, next_id], dim=1)
            if (next_id == self.eos_id).all():
                break
        return out_ids[:, 1:]


def build_head(spec: HeadConfig, embed_dim: int) -> nn.Module:
    if spec.type == "classify":
        return ClassifyHead(embed_dim, spec.num_classes)
    if spec.type == "regress":
        return RegressHead(embed_dim, spec.dim, loss=spec.loss or "mse")
    if spec.type == "grid":
        return GridHead(embed_dim, spec.rows, spec.cols)
    if spec.type == "token":
        return TokenDecoderHead(embed_dim, len(spec.vocab), spec.max_len)
    raise ValueError(f"Unknown head type {spec.type!r}")


def head_needs_patch_tokens(spec: HeadConfig) -> bool:
    return spec.type == "token"
