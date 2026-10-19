"""Continuous Guided-Attention Handwriting Synthesis Transformer Model.

Combines causal self-attention over continuous stroke coordinates with guided
cross-attention over character text encodings, and a Bivariate Gaussian Mixture
Density Network (MDN) output head for infinite-resolution analog handwriting.
"""

from __future__ import annotations

import math
from typing import Any, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GuidedTransformerDecoderLayer
from .config import NewModelConfig


class SinusoidalPositionalEncoding(nn.Module):
    """Standard sinusoidal positional encoding for sequence coordinates."""

    def __init__(self, d_model: int, max_len: int = 2000) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encodings to input embeddings: x shape (B, T, d_model)."""
        return x + self.pe[:, : x.size(1)]


class TextEncoder(nn.Module):
    """Contextual character encoder with bidirectional self-attention and positional encoding."""

    def __init__(self, config: NewModelConfig) -> None:
        super().__init__()
        self.char_emb = nn.Embedding(config.char_vocab_size, config.d_model, padding_idx=0)
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, max_len=256)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=config.num_encoder_layers
        )
        self.norm = nn.LayerNorm(config.d_model)

    def forward(
        self, chars: torch.Tensor, char_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Encode character sequence into context vectors."""
        x = self.char_emb(chars)
        x = self.pos_enc(x)
        src_key_padding_mask = ~char_mask if char_mask is not None else None
        memory = self.encoder(x, src_key_padding_mask=src_key_padding_mask)
        return self.norm(memory)


class ContinuousStrokeEmbedding(nn.Module):
    """Continuous coordinate projection with writer style conditioning."""

    def __init__(self, config: NewModelConfig) -> None:
        super().__init__()
        # Continuous projection: [dx, dy, pen_up] -> d_model
        self.proj = nn.Linear(config.input_dim, config.d_model)
        self.emb_writer = nn.Embedding(config.n_writers, config.d_model)
        self.pos_enc = SinusoidalPositionalEncoding(config.d_model, max_len=config.max_seq_len)
        self.norm = nn.LayerNorm(config.d_model)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
        self,
        strokes: torch.Tensor,
        writer: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Embed continuous [dx, dy, pen_up] inputs and add positional encodings."""
        x = self.proj(strokes)
        if writer is not None:
            # writer: (B,) -> (B, 1, d_model)
            x = x + self.emb_writer(writer).unsqueeze(1)
        x = self.pos_enc(x)
        x = self.norm(x)
        return self.dropout(x)


class BivariateMDNHead(nn.Module):
    """Bivariate Gaussian Mixture Density Network output head."""

    def __init__(self, d_model: int, num_gaussians: int = 20) -> None:
        super().__init__()
        self.num_gaussians = num_gaussians
        # 1 (pen_up Bernoulli) + 6 * M (pi, mu_x, mu_y, sigma_x, sigma_y, rho)
        self.proj = nn.Linear(d_model, 1 + 6 * num_gaussians)

    def forward(self, x: torch.Tensor, bias: float = 0.0) -> dict[str, torch.Tensor]:
        """Project decoder hidden states to bivariate Gaussian parameters."""
        out = self.proj(x)
        m = self.num_gaussians

        # 1. Bernoulli pen-lift probability e
        e = torch.sigmoid(out[:, :, 0])

        # 2. Mixture parameters
        pi_logits = out[:, :, 1 : 1 + m]
        mu_x = out[:, :, 1 + m : 1 + 2 * m]
        mu_y = out[:, :, 1 + 2 * m : 1 + 3 * m]
        sd_x_logits = out[:, :, 1 + 3 * m : 1 + 4 * m]
        sd_y_logits = out[:, :, 1 + 4 * m : 1 + 5 * m]
        ro_logits = out[:, :, 1 + 5 * m : 1 + 6 * m]

        if bias != 0.0:
            pi = F.softmax(pi_logits * (1.0 + bias), dim=-1)
            sigma_x = torch.exp(sd_x_logits - bias)
            sigma_y = torch.exp(sd_y_logits - bias)
        else:
            pi = F.softmax(pi_logits, dim=-1)
            sigma_x = torch.exp(sd_x_logits)
            sigma_y = torch.exp(sd_y_logits)

        # Clamping for strict numerical stability
        sigma_x = torch.clamp(sigma_x, min=1e-4, max=100.0)
        sigma_y = torch.clamp(sigma_y, min=1e-4, max=100.0)
        rho = torch.tanh(ro_logits)
        rho = torch.clamp(rho, min=-0.999, max=0.999)

        return {
            "pi": pi,
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho,
            "e": e,
        }


class ContinuousGuidedTransformer(nn.Module):
    """Continuous Guided-Attention Transformer with Bivariate Gaussian MDN output."""

    def __init__(self, config: NewModelConfig) -> None:
        super().__init__()
        self.config = config

        self.text_encoder = TextEncoder(config)
        self.stroke_embedding = ContinuousStrokeEmbedding(config)

        # Stack of guided decoder layers
        self.layers = nn.ModuleList([
            GuidedTransformerDecoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout,
            )
            for _ in range(config.num_decoder_layers)
        ])
        self.final_norm = nn.LayerNorm(config.d_model)

        # Continuous Bivariate Gaussian MDN Head
        self.mdn_head = BivariateMDNHead(config.d_model, num_gaussians=config.num_gaussians)

    def _generate_causal_mask(self, sz: int, device: torch.device) -> torch.Tensor:
        """Generate upper-triangular causal attention mask where True = masked out."""
        return torch.triu(torch.ones(sz, sz, device=device), diagonal=1).bool()

    def forward(
        self,
        chars: torch.Tensor,
        strokes: torch.Tensor,
        char_mask: torch.Tensor | None = None,
        stroke_mask: torch.Tensor | None = None,
        writer: torch.Tensor | None = None,
        monotonic_mask: torch.Tensor | None = None,
        bias: float = 0.0,
    ) -> dict[str, torch.Tensor]:
        """Forward pass predicting continuous bivariate Gaussian distributions.

        Parameters
        ----------
        chars: `(B, U)` character tokens
        strokes: `(B, T, 3)` continuous input strokes `[dx, dy, pen_up]`
        char_mask: `(B, U)` True for valid characters, False for padding
        stroke_mask: `(B, T)` True for valid stroke steps, False for padding
        writer: `(B,)` writer style IDs
        monotonic_mask: optional attention mask
        bias: neatness bias

        Returns
        -------
        dict with 'pi', 'mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'rho', 'e', and 'attn_weights'
        """
        device = strokes.device
        t_len = strokes.size(1)

        # 1. Encode text context: (B, U, d_model)
        memory = self.text_encoder(chars, char_mask=char_mask)

        # 2. Embed continuous stroke inputs: (B, T, d_model)
        x = self.stroke_embedding(strokes, writer=writer)

        # 3. Causal decoder mask: (T, T)
        causal_mask = self._generate_causal_mask(t_len, device)

        # 4. Decoder layers with guided cross-attention
        attn_weights = None
        for layer in self.layers:
            x, attn_weights = layer(
                tgt=x,
                memory=memory,
                tgt_mask=causal_mask,
                memory_key_padding_mask=char_mask,
                monotonic_mask=monotonic_mask,
            )

        x = self.final_norm(x)

        # 5. Project to continuous MDN parameters
        mdn_params = self.mdn_head(x, bias=bias)
        mdn_params["attn_weights"] = attn_weights
        return mdn_params

    @torch.no_grad()
    def generate(
        self,
        chars: torch.Tensor,
        writer: torch.Tensor | None = None,
        max_steps: int = 800,
        bias: float = 1.0,
        max_dwell_steps: int = 35,
        stop_on_completion: bool = True,
        baseline_spring: float = 0.08,
    ) -> dict[str, Any]:
        """Autoregressively sample smooth continuous stroke coordinates.

        Eliminates all discrete tokenization artifacts:
        1. Outputs continuous floating-point trajectories directly via bivariate Gaussian sampling.
        2. Breaks repeated character attractor loops ('llll...') via unique text positional encodings
           and attention dwell limiting.
        3. Cleanly stops when the prompt is completed and the pen lifts.

        Parameters
        ----------
        chars: `(1, U)` character token IDs
        writer: `(1,)` optional writer style ID
        max_steps: maximum stroke sequence length
        bias: sampling neatness bias (>= 0.0; higher = neater, 0 = raw variance)
        max_dwell_steps: maximum steps permitted on a single character
        stop_on_completion: halt cleanly once prompt is consumed
        baseline_spring: restoring force coefficient to anchor handwriting to baseline

        Returns
        -------
        dict with:
            'coords': `(N, 4)` continuous `[x, y, pen_up, time]`
            'offsets': `(N, 4)` continuous `[dx, dy, pen_up, dt]`
            'attn_history': attention alignment progression
        """
        self.eval()
        device = chars.device
        u_len = chars.size(1)
        char_mask = torch.ones_like(chars, dtype=torch.bool, device=device)

        # Encode text context once
        memory = self.text_encoder(chars, char_mask=char_mask)

        # Initial start point: zero displacement with pen up [0.0, 0.0, 1.0]
        curr_strokes = torch.tensor([[[0.0, 0.0, 1.0]]], dtype=torch.float32, device=device)

        min_char_idx = 0
        curr_char_focus = 0
        dwell_counter = 0
        consecutive_pen_ups = 0
        curr_y = 0.0
        history_attn = []
        sampled_deltas = []

        char_indices = torch.arange(u_len, device=device).view(1, 1, 1, u_len)

        for step in range(max_steps):
            t_len = curr_strokes.size(1)
            causal_mask = self._generate_causal_mask(t_len, device)

            # Monotonic mask prevents attention from looking back at completed characters
            mono_mask = char_indices >= min_char_idx

            x = self.stroke_embedding(curr_strokes, writer=writer)

            attn_weights = None
            for layer in self.layers:
                x, attn_weights = layer(
                    tgt=x,
                    memory=memory,
                    tgt_mask=causal_mask,
                    memory_key_padding_mask=char_mask,
                    monotonic_mask=mono_mask,
                )

            # Compute attention center at current step
            step_attn = attn_weights[0, :, -1, :].mean(dim=0)  # (U,)
            u_bar = (torch.arange(u_len, device=device, dtype=torch.float32) * step_attn).sum().item()
            history_attn.append(step_attn.cpu().numpy())

            # Dwell Tracking & Monotonic Advance (breaks attractor loops)
            char_focus = int(round(u_bar))
            char_focus = max(min_char_idx, min(char_focus, u_len - 1))

            if char_focus == curr_char_focus:
                dwell_counter += 1
            else:
                curr_char_focus = char_focus
                dwell_counter = 1

            if dwell_counter > max_dwell_steps:
                min_char_idx = min(min_char_idx + 1, u_len - 1)
                curr_char_focus = min_char_idx
                dwell_counter = 0
            elif char_focus > min_char_idx:
                min_char_idx = max(min_char_idx, char_focus - 1)

            x = self.final_norm(x)
            last_hidden = x[:, -1:, :]

            # Project to continuous MDN parameters
            mdn = self.mdn_head(last_hidden, bias=bias)

            # Sample continuous (dx, dy) from bivariate Gaussian mixture
            pi = mdn["pi"][0, 0]          # (M,)
            mu_x = mdn["mu_x"][0, 0]      # (M,)
            mu_y = mdn["mu_y"][0, 0]      # (M,)
            sd_x = mdn["sigma_x"][0, 0]  # (M,)
            sd_y = mdn["sigma_y"][0, 0]  # (M,)
            ro = mdn["rho"][0, 0]        # (M,)
            e = mdn["e"][0, 0].item()

            comp = torch.multinomial(pi, 1).item()
            mx = mu_x[comp].item()
            my = mu_y[comp].item()
            sx = sd_x[comp].item()
            sy = sd_y[comp].item()
            r = ro[comp].item()

            cov = r * sx * sy
            sigma = torch.tensor([[sx**2, cov], [cov, sy**2]], dtype=torch.float32)
            loc = torch.tensor([mx, my], dtype=torch.float32)

            try:
                gmm = torch.distributions.MultivariateNormal(loc, sigma)
                v = gmm.sample()
                step_dx, step_dy = float(v[0].item()), float(v[1].item())
            except Exception:
                step_dx, step_dy = mx, my

            # Harmonic baseline restoring potential to prevent vertical drift
            if baseline_spring > 0.0:
                step_dy += baseline_spring * (0.0 - curr_y)
            curr_y += step_dy

            step_pen = 1.0 if (torch.rand(1).item() < e or e > 0.5) else 0.0

            if step_pen > 0.5:
                consecutive_pen_ups += 1
            else:
                consecutive_pen_ups = 0

            sampled_deltas.append([step_dx, step_dy, step_pen, 0.02])

            # Clean Early Stopping:
            if stop_on_completion:
                if u_bar >= (u_len - 1.2) and step >= 20:
                    if step_pen > 0.5 or dwell_counter >= 15:
                        break
                if consecutive_pen_ups >= 6:
                    break

            # Append to autoregressive stroke input
            next_pt = torch.tensor([[[step_dx, step_dy, step_pen]]], dtype=torch.float32, device=device)
            curr_strokes = torch.cat([curr_strokes, next_pt], dim=1)

        raw_offsets = np.asarray(sampled_deltas, dtype=np.float32)
        if len(raw_offsets) == 0:
            return {
                "coords": np.zeros((0, 4), dtype=np.float32),
                "offsets": np.zeros((0, 4), dtype=np.float32),
                "attn_history": None,
            }

        # Integrate continuous coordinates
        coords = np.zeros_like(raw_offsets, dtype=np.float32)
        coords[:, [0, 1]] = np.cumsum(raw_offsets[:, [0, 1]], axis=0)
        coords[:, 2] = raw_offsets[:, 2]
        coords[:, 3] = np.cumsum(raw_offsets[:, 3], axis=0)

        return {
            "coords": coords,
            "offsets": raw_offsets,
            "attn_history": np.array(history_attn) if history_attn else None,
        }
