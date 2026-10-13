"""Neural sequence model for stroke-level handwriting synthesis.

This module implements the core neural architecture:
- TextEncoder: Contextual character embeddings via bidirectional Transformer.
- StrokeEmbedding: Discrete multi-channel token embedding fusion + writer style.
- HandwritingSynthesisModel: Autoregressive causal decoder with Cross-Attention
  and multi-head token classification.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hws.alphabet import PAD_INDEX, Alphabet
from hws.tokenizer import StrokeTokenizer


@dataclass
class ModelConfig:
    """Hyperparameters and vocabulary configurations for the synthesis model."""

    char_vocab_size: int = 78
    n_writers: int = 197
    vocab_dx: int = 512
    vocab_dy: int = 512
    vocab_dt: int = 266
    vocab_pen: int = 2
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 1200

    @classmethod
    def from_config_json(cls, path: str | Path, **kwargs: Any) -> ModelConfig:
        """Instantiate ModelConfig from processed dataset config.json."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        vocab_sizes = data.get("vocab_sizes", {})
        return cls(
            char_vocab_size=int(data.get("char_vocab_size", 78)),
            n_writers=int(data.get("n_writers", 197)),
            vocab_dx=int(vocab_sizes.get("dx", 512)),
            vocab_dy=int(vocab_sizes.get("dy", 512)),
            vocab_dt=int(vocab_sizes.get("dt", 266)),
            vocab_pen=int(vocab_sizes.get("pen", 2)),
            **kwargs,
        )


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input tensor `(B, T, d_model)`."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class TextEncoder(nn.Module):
    """Bidirectional Transformer encoder for text character sequences."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.d_model = config.d_model
        self.char_emb = nn.Embedding(
            config.char_vocab_size, config.d_model, padding_idx=PAD_INDEX
        )
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, max_len=config.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self, chars: torch.Tensor, char_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode characters into contextual memory representations.

        Parameters
        ----------
        chars:
            `(B, L)` LongTensor of character token IDs.
        char_mask:
            `(B, L)` bool mask where True indicates valid characters, False is pad.

        Returns
        -------
        `(B, L, d_model)` contextual memory representation.
        """
        x = self.char_emb(chars) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        # PyTorch Transformer uses True for elements that should be masked out (ignored)
        key_padding_mask = ~char_mask if char_mask is not None else None
        memory = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(memory)


class StrokeEmbedding(nn.Module):
    """Fused token embedding for discrete stroke channels + writer style."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.emb_dx = nn.Embedding(config.vocab_dx, config.d_model)
        self.emb_dy = nn.Embedding(config.vocab_dy, config.d_model)
        self.emb_dt = nn.Embedding(config.vocab_dt, config.d_model)
        self.emb_pen = nn.Embedding(config.vocab_pen, config.d_model)
        self.emb_writer = nn.Embedding(config.n_writers, config.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, max_len=config.max_seq_len)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        in_dx: torch.Tensor,
        in_dy: torch.Tensor,
        in_dt: torch.Tensor,
        in_pen: torch.Tensor,
        writer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse stroke token channels and optional writer style into input vectors."""
        x = (
            self.emb_dx(in_dx)
            + self.emb_dy(in_dy)
            + self.emb_dt(in_dt)
            + self.emb_pen(in_pen)
        )

        if writer is not None:
            # writer: (B,) -> (B, 1, d_model)
            w_emb = self.emb_writer(writer).unsqueeze(1)
            x = x + w_emb

        x = self.pos_enc(x)
        x = self.norm(x)
        return self.dropout(x)


class HandwritingSynthesisModel(nn.Module):
    """Autoregressive Transformer with Cross-Attention for stroke synthesis."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config

        self.text_encoder = TextEncoder(config)
        self.stroke_embedding = StrokeEmbedding(config)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer, num_layers=config.num_decoder_layers
        )
        self.norm = nn.LayerNorm(config.d_model)

        # Multi-head output classifiers
        self.head_dx = nn.Linear(config.d_model, config.vocab_dx)
        self.head_dy = nn.Linear(config.d_model, config.vocab_dy)
        self.head_dt = nn.Linear(config.d_model, config.vocab_dt)
        self.head_pen = nn.Linear(config.d_model, config.vocab_pen)

    def forward(
        self,
        chars: torch.Tensor,
        in_dx: torch.Tensor,
        in_dy: torch.Tensor,
        in_dt: torch.Tensor,
        in_pen: torch.Tensor,
        char_mask: torch.Tensor | None = None,
        in_mask: torch.Tensor | None = None,
        writer: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass predicting next-token logits across all 4 channels.

        Parameters
        ----------
        chars: `(B, L)` character token IDs
        in_dx, in_dy, in_dt, in_pen: `(B, T)` stroke input tokens (steps 0..T-1)
        char_mask: `(B, L)` bool mask (True = valid, False = pad)
        in_mask: `(B, T)` bool mask (True = valid, False = pad)
        writer: `(B,)` writer ID indices

        Returns
        -------
        dict with logits:
            `dx`: `(B, T, vocab_dx)`
            `dy`: `(B, T, vocab_dy)`
            `dt`: `(B, T, vocab_dt)`
            `pen`: `(B, T, 2)`
        """
        # Encode text context
        memory = self.text_encoder(chars, char_mask=char_mask)

        # Embed stroke inputs
        tgt = self.stroke_embedding(in_dx, in_dy, in_dt, in_pen, writer=writer)

        t_len = tgt.size(1)
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            t_len, device=tgt.device, dtype=torch.bool
        )

        tgt_key_padding = ~in_mask if in_mask is not None else None
        memory_key_padding = ~char_mask if char_mask is not None else None

        hidden = self.decoder(
            tgt=tgt,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding,
            memory_key_padding_mask=memory_key_padding,
        )
        hidden = self.norm(hidden)

        return {
            "dx": self.head_dx(hidden),
            "dy": self.head_dy(hidden),
            "dt": self.head_dt(hidden),
            "pen": self.head_pen(hidden),
        }

    def compute_loss(
        self,
        logits: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Compute multi-task Cross-Entropy loss ignoring padded -100 tokens.

        Parameters
        ----------
        logits: dict from `forward()` with keys `dx`, `dy`, `dt`, `pen`
        targets: dict with keys `target_dx`, `target_dy`, `target_dt`, `target_pen`

        Returns
        -------
        dict with `loss`, `loss_dx`, `loss_dy`, `loss_dt`, `loss_pen`
        """
        loss_dx = F.cross_entropy(
            logits["dx"].reshape(-1, self.config.vocab_dx),
            targets["target_dx"].reshape(-1),
            ignore_index=-100,
        )
        loss_dy = F.cross_entropy(
            logits["dy"].reshape(-1, self.config.vocab_dy),
            targets["target_dy"].reshape(-1),
            ignore_index=-100,
        )
        loss_dt = F.cross_entropy(
            logits["dt"].reshape(-1, self.config.vocab_dt),
            targets["target_dt"].reshape(-1),
            ignore_index=-100,
        )
        loss_pen = F.cross_entropy(
            logits["pen"].reshape(-1, self.config.vocab_pen),
            targets["target_pen"].reshape(-1),
            ignore_index=-100,
        )

        total_loss = loss_dx + loss_dy + loss_dt + loss_pen

        return {
            "loss": total_loss,
            "loss_dx": loss_dx,
            "loss_dy": loss_dy,
            "loss_dt": loss_dt,
            "loss_pen": loss_pen,
        }

    @torch.no_grad()
    def generate(
        self,
        text: str,
        alphabet: Alphabet,
        tokenizer: StrokeTokenizer,
        writer_id: int = 0,
        max_steps: int = 600,
        temperature: float = 0.8,
        top_k: int = 50,
        top_p: float = 0.9,
        device: torch.device | str = "cpu",
    ) -> np.ndarray:
        """Autoregressively synthesize handwriting strokes for an input text line.

        Parameters
        ----------
        text: English text prompt (characters must be in `alphabet`).
        alphabet: Pre-fitted `Alphabet` mapping text characters to token IDs.
        tokenizer: Pre-fitted `StrokeTokenizer` mapping tokens to displacements.
        writer_id: Integer writer style ID (0 = general/unknown).
        max_steps: Maximum stroke trajectory points to generate.
        temperature: Sampling temperature (0.1 = rigid/clean, 1.5 = wild).
        top_k: Top-k filtering threshold.
        top_p: Nucleus filtering threshold.
        device: Torch execution device.

        Returns
        -------
        `(N, 4)` float32 array of `[x, y, pen_up, time]` with coordinates integrated.
        """
        self.eval()
        char_ids = alphabet.encode(text)
        if not char_ids:
            return np.zeros((0, 4), dtype=np.float32)

        chars = torch.tensor([char_ids], dtype=torch.long, device=device)
        char_mask = torch.ones((1, len(char_ids)), dtype=torch.bool, device=device)
        writer = torch.tensor([writer_id], dtype=torch.long, device=device)

        # 1. Encode text prompt
        memory = self.text_encoder(chars, char_mask=char_mask)

        # 2. Get initial starting token (corresponds to zero displacement)
        start_offsets = np.zeros((1, 4), dtype=np.float32)
        start_token_dict = tokenizer.encode(start_offsets)

        cur_dx = [int(start_token_dict["dx"][0])]
        cur_dy = [int(start_token_dict["dy"][0])]
        cur_dt = [int(start_token_dict["dt"][0])]
        cur_pen = [int(start_token_dict["pen"][0])]

        consecutive_pen_ups = 0

        for _ in range(max_steps - 1):
            t_in_dx = torch.tensor([cur_dx], dtype=torch.long, device=device)
            t_in_dy = torch.tensor([cur_dy], dtype=torch.long, device=device)
            t_in_dt = torch.tensor([cur_dt], dtype=torch.long, device=device)
            t_in_pen = torch.tensor([cur_pen], dtype=torch.long, device=device)

            t_len = t_in_dx.size(1)
            tgt = self.stroke_embedding(
                t_in_dx, t_in_dy, t_in_dt, t_in_pen, writer=writer
            )
            causal_mask = nn.Transformer.generate_square_subsequent_mask(
                t_len, device=device, dtype=torch.bool
            )

            hidden = self.decoder(
                tgt=tgt,
                memory=memory,
                tgt_mask=causal_mask,
            )
            hidden = self.norm(hidden[:, -1:, :])  # Only need the last step

            next_dx = _sample_token(
                self.head_dx(hidden).squeeze(1), temperature, top_k, top_p
            )
            next_dy = _sample_token(
                self.head_dy(hidden).squeeze(1), temperature, top_k, top_p
            )
            next_dt = _sample_token(
                self.head_dt(hidden).squeeze(1), temperature, top_k, top_p
            )
            next_pen = _sample_token(
                self.head_pen(hidden).squeeze(1), temperature, top_k, top_p
            )

            cur_dx.append(next_dx)
            cur_dy.append(next_dy)
            cur_dt.append(next_dt)
            cur_pen.append(next_pen)

            if next_pen == 1:
                consecutive_pen_ups += 1
            else:
                consecutive_pen_ups = 0

            # Stopping condition: writing finished and pen rested up
            if consecutive_pen_ups >= 6:
                break

        # 3. Decode token channels back into continuous offsets
        generated_tokens = {
            "dx": np.asarray(cur_dx, dtype=np.int64),
            "dy": np.asarray(cur_dy, dtype=np.int64),
            "dt": np.asarray(cur_dt, dtype=np.int64),
            "pen": np.asarray(cur_pen, dtype=np.int64),
        }
        decoded_offsets = tokenizer.decode(generated_tokens)

        # 4. Integrate into absolute coordinates
        xy = np.cumsum(decoded_offsets[:, :2], axis=0)
        time = np.cumsum(decoded_offsets[:, 3:4], axis=0)
        pen_up = decoded_offsets[:, 2:3]

        return np.concatenate([xy, pen_up, time], axis=1).astype(np.float32)


def _sample_token(
    logits: torch.Tensor, temperature: float, top_k: int, top_p: float
) -> int:
    """Sample a discrete token index with temperature and nucleus (top-p) filtering."""
    if temperature <= 0.01:
        return int(torch.argmax(logits, dim=-1).item())

    scaled = logits / temperature

    # Top-k filtering
    if top_k > 0:
        val, _ = torch.topk(scaled, min(top_k, scaled.size(-1)))
        min_val = val[:, -1].unsqueeze(-1)
        scaled = torch.where(scaled < min_val, torch.full_like(scaled, -float("Inf")), scaled)

    # Top-p (nucleus) filtering
    if 0.0 < top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(scaled, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

        # Remove tokens with cumulative probability above top_p
        sorted_indices_to_remove = cumulative_probs > top_p
        # Shift mask right to keep at least first token above threshold
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = 0

        indices_to_remove = sorted_indices_to_remove.scatter(
            dim=-1, index=sorted_indices, src=sorted_indices_to_remove
        )
        scaled = scaled.masked_fill(indices_to_remove, -float("Inf"))

    probs = F.softmax(scaled, dim=-1)
    sampled = torch.multinomial(probs, num_samples=1)
    return int(sampled.item())
