"""Custom Cross-Attention modules that expose attention alignment weights for guided training."""

from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class GuidedMultiheadCrossAttention(nn.Module):
    """Multi-Head Cross-Attention returning both output projections and attention weights."""

    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1) -> None:
        super().__init__()
        assert d_model % nhead == 0, f"d_model ({d_model}) must be divisible by nhead ({nhead})"

        self.d_model = d_model
        self.nhead = nhead
        self.d_k = d_model // nhead

        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.d_k)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
        monotonic_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute cross-attention between stroke queries and character keys.

        Parameters
        ----------
        query: `(B, T, d_model)` stroke sequence representations
        key: `(B, L, d_model)` character memory representations
        value: `(B, L, d_model)` character memory representations
        key_padding_mask: `(B, L)` bool mask where True = valid, False = pad
        monotonic_mask: `(B, 1, T, L)` or `(B, nhead, T, L)` optional forward monotonic mask

        Returns
        -------
        output: `(B, T, d_model)`
        attn_weights: `(B, nhead, T, L)` attention alignment probabilities
        """
        b_sz, t_len, _ = query.shape
        _, l_len, _ = key.shape

        # Linear projections & reshape to (B, nhead, Seq_len, d_k)
        q = self.q_proj(query).view(b_sz, t_len, self.nhead, self.d_k).transpose(1, 2)
        k = self.k_proj(key).view(b_sz, l_len, self.nhead, self.d_k).transpose(1, 2)
        v = self.v_proj(value).view(b_sz, l_len, self.nhead, self.d_k).transpose(1, 2)

        # Scaled dot-product scores: (B, nhead, T, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if key_padding_mask is not None:
            # key_padding_mask: True = valid, False = pad
            mask = key_padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, L)
            scores = scores.masked_fill(~mask, -1e4)

        if monotonic_mask is not None:
            # Monotonic mask prevents looking backwards at already completed characters
            scores = scores.masked_fill(~monotonic_mask, -1e4)

        attn_weights = F.softmax(scores, dim=-1)
        attn_dropped = self.dropout(attn_weights)

        # Context output: (B, nhead, T, d_k) -> (B, T, d_model)
        context = torch.matmul(attn_dropped, v)
        context = context.transpose(1, 2).contiguous().view(b_sz, t_len, self.d_model)

        output = self.out_proj(context)
        return output, attn_weights


class GuidedTransformerDecoderLayer(nn.Module):
    """Transformer Decoder layer that returns cross-attention weights."""

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        # 1. Causal self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)

        # 2. Guided cross-attention
        self.cross_attn = GuidedMultiheadCrossAttention(d_model, nhead, dropout=dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout2 = nn.Dropout(dropout)

        # 3. Feedforward network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
        )
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout3 = nn.Dropout(dropout)

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor | None = None,
        memory_key_padding_mask: torch.Tensor | None = None,
        monotonic_mask: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass with pre-layer normalization.

        Returns
        -------
        tgt: `(B, T, d_model)`
        cross_attn_weights: `(B, nhead, T, L)`
        """
        # Self-attention with causal mask
        norm_tgt = self.norm1(tgt)
        sa_out, _ = self.self_attn(
            norm_tgt, norm_tgt, norm_tgt, attn_mask=tgt_mask, need_weights=False
        )
        tgt = tgt + self.dropout1(sa_out)

        # Cross-attention to text memory with monotonic mask support
        norm_tgt = self.norm2(tgt)
        ca_out, attn_weights = self.cross_attn(
            norm_tgt,
            memory,
            memory,
            key_padding_mask=memory_key_padding_mask,
            monotonic_mask=monotonic_mask,
        )
        tgt = tgt + self.dropout2(ca_out)

        # Feedforward
        norm_tgt = self.norm3(tgt)
        ffn_out = self.ffn(norm_tgt)
        tgt = tgt + self.dropout3(ffn_out)

        return tgt, attn_weights
