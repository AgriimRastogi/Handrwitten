"""Alex Graves (2013) Handwriting Synthesis Model in PyTorch.

Reference:
    Alex Graves, "Generating Sequences With Recurrent Neural Networks",
    arXiv:1308.0850 (Section 5: Handwriting Synthesis).

Architecture:
1. Text conditioning via monotonic Gaussian soft-window attention:
       phi_t(u) = sum_{k=1}^K alpha_t^k * exp(-beta_t^k * (kappa_t^k - u)^2)
       w_t = sum_{u=1}^U phi_t(u) * c_u
   where kappa_t = kappa_{t-1} + exp(hat_kappa_t), strictly monotonic forward.
2. 3-layer stacked LSTM with residual/skip connections.
3. Bivariate Gaussian Mixture Density Network (MDN) output head predicting
   joint (dx, dy) distributions and Bernoulli pen-lift probability e.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from hws.alphabet import Alphabet, PAD_INDEX


@dataclass
class GravesConfig:
    """Hyperparameters for Graves 2013 RNN Handwriting Model."""

    char_vocab_size: int = 78
    hidden_size: int = 400
    num_layers: int = 3
    num_gaussians: int = 20           # M bivariate mixture components
    num_window_components: int = 10   # K Gaussian attention kernels
    input_dim: int = 3                # [dx, dy, pen_up]


class GravesHandwritingModel(nn.Module):
    """3-layer LSTM with Monotonic Gaussian Soft-Window Attention and Bivariate MDN."""

    def __init__(self, config: GravesConfig | None = None) -> None:
        super().__init__()
        self.config = config or GravesConfig()

        c_dim = self.config.char_vocab_size
        h_dim = self.config.hidden_size
        k_dim = self.config.num_window_components
        m_dim = self.config.num_gaussians

        # 1. Recurrent LSTM Cells (Graves Equations 46-52)
        # Layer 1 input: previous stroke (3) + previous window vector (c_dim)
        self.lstm_1 = nn.LSTMCell(3 + c_dim, h_dim)

        # Window parameter projection from h_1: [alpha, beta, kappa] * K
        self.window_proj = nn.Linear(h_dim, 3 * k_dim)

        # Layer 2 input: previous stroke (3) + current window vector (c_dim) + h_1 (h_dim)
        self.lstm_2 = nn.LSTMCell(3 + c_dim + h_dim, h_dim)

        # Layer 3 input: previous stroke (3) + current window vector (c_dim) + h_2 (h_dim)
        self.lstm_3 = nn.LSTMCell(3 + c_dim + h_dim, h_dim)

        # 2. Output MDN Projection from [h_1, h_2, h_3]
        # Size: 1 (pen) + M (pi) + M (mu_x) + M (mu_y) + M (sigma_x) + M (sigma_y) + M (rho) = 1 + 6M
        self.out_proj = nn.Linear(3 * h_dim, 1 + 6 * m_dim)

        # Initialise window projection biases to ensure gentle starting focus
        with torch.no_grad():
            self.window_proj.bias.data.zero_()
            self.window_proj.bias.data[k_dim: 2 * k_dim].fill_(0.5)     # beta
            self.window_proj.bias.data[2 * k_dim: 3 * k_dim].fill_(-1.0) # delta_kappa

    def forward(
        self,
        strokes: torch.Tensor,
        chars: torch.Tensor,
        char_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Forward pass across full sequence length with teacher-forcing.

        Parameters
        ----------
        strokes: `(B, T, 3)` continuous inputs `[dx, dy, pen_up]`
        chars: `(B, U)` character token IDs
        char_mask: `(B, U)` bool mask (True = valid char, False = pad)

        Returns
        -------
        Dictionary containing MDN parameters and attention weights:
            `pi`: `(B, T, M)` mixture weights
            `mu_x`, `mu_y`: `(B, T, M)` means
            `sigma_x`, `sigma_y`: `(B, T, M)` positive standard deviations
            `rho`: `(B, T, M)` correlations in (-1, 1)
            `e`: `(B, T)` Bernoulli pen lift probabilities
            `phi`: `(B, T, U)` attention alignment weights
        """
        device = strokes.device
        batch_size, t_steps, _ = strokes.size()
        u_len = chars.size(1)
        k_dim = self.config.num_window_components
        m_dim = self.config.num_gaussians
        h_dim = self.config.hidden_size
        c_dim = self.config.char_vocab_size

        # One-hot representation of prompt characters: (B, U, C)
        safe_chars = torch.clamp(chars, min=0, max=c_dim - 1)
        char_one_hot = F.one_hot(safe_chars, num_classes=c_dim).float()
        # Zero out padding characters
        char_one_hot = char_one_hot * char_mask.unsqueeze(-1).float()

        # Character position indices u = 0 .. U-1: (1, 1, U)
        u_positions = torch.arange(u_len, device=device, dtype=torch.float32).view(1, 1, u_len)

        # Initialise recurrent states
        h1 = torch.zeros(batch_size, h_dim, device=device)
        c1 = torch.zeros(batch_size, h_dim, device=device)
        h2 = torch.zeros(batch_size, h_dim, device=device)
        c2 = torch.zeros(batch_size, h_dim, device=device)
        h3 = torch.zeros(batch_size, h_dim, device=device)
        c3 = torch.zeros(batch_size, h_dim, device=device)

        # Initialise attention window
        w_prev = torch.zeros(batch_size, c_dim, device=device)
        kappa_prev = torch.zeros(batch_size, k_dim, device=device)

        out_list: list[torch.Tensor] = []
        phi_list: list[torch.Tensor] = []

        for t in range(t_steps):
            x_t = strokes[:, t, :]  # (B, 3)

            # Layer 1
            in_1 = torch.cat([x_t, w_prev], dim=-1)
            h1, c1 = self.lstm_1(in_1, (h1, c1))

            # Window Attention parameters from h1
            p = self.window_proj(h1)  # (B, 3K)
            hat_alpha = p[:, :k_dim]
            hat_beta = p[:, k_dim: 2 * k_dim]
            hat_kappa = p[:, 2 * k_dim:]

            alpha = torch.exp(hat_alpha).unsqueeze(-1)                   # (B, K, 1)
            beta = torch.exp(hat_beta).unsqueeze(-1)                     # (B, K, 1)
            kappa = (kappa_prev + torch.exp(hat_kappa)).unsqueeze(-1)    # (B, K, 1)
            kappa_prev = kappa.squeeze(-1)

            # Gaussian attention mixture: phi_t(u) = sum_k alpha * exp(-beta * (kappa - u)^2)
            # diff: (B, K, U)
            diff = kappa - u_positions
            phi = torch.sum(alpha * torch.exp(-beta * (diff ** 2)), dim=1)  # (B, U)
            # Mask out padding positions
            phi = phi * char_mask.float()
            phi_list.append(phi)

            # Compute soft window character vector: w_t = sum_u phi_t(u) * c_u
            # (B, 1, U) x (B, U, C) -> (B, C)
            w_curr = torch.bmm(phi.unsqueeze(1), char_one_hot).squeeze(1)
            w_prev = w_curr

            # Layer 2
            in_2 = torch.cat([x_t, w_curr, h1], dim=-1)
            h2, c2 = self.lstm_2(in_2, (h2, c2))

            # Layer 3
            in_3 = torch.cat([x_t, w_curr, h2], dim=-1)
            h3, c3 = self.lstm_3(in_3, (h3, c3))

            # Output projection from stacked hidden states
            y_t = self.out_proj(torch.cat([h1, h2, h3], dim=-1))  # (B, 1 + 6M)
            out_list.append(y_t)

        outputs = torch.stack(out_list, dim=1)  # (B, T, 1 + 6M)
        phis = torch.stack(phi_list, dim=1)     # (B, T, U)

        # Parse MDN components
        e = torch.sigmoid(outputs[:, :, 0])
        pi = F.softmax(outputs[:, :, 1: 1 + m_dim], dim=-1)
        mu_x = outputs[:, :, 1 + m_dim: 1 + 2 * m_dim]
        mu_y = outputs[:, :, 1 + 2 * m_dim: 1 + 3 * m_dim]
        sigma_x = torch.exp(torch.clamp(outputs[:, :, 1 + 3 * m_dim: 1 + 4 * m_dim], -7.0, 7.0))
        sigma_y = torch.exp(torch.clamp(outputs[:, :, 1 + 4 * m_dim: 1 + 5 * m_dim], -7.0, 7.0))
        rho = torch.clamp(torch.tanh(outputs[:, :, 1 + 5 * m_dim: 1 + 6 * m_dim]), -0.995, 0.995)

        return {
            "pi": pi,
            "mu_x": mu_x,
            "mu_y": mu_y,
            "sigma_x": sigma_x,
            "sigma_y": sigma_y,
            "rho": rho,
            "e": e,
            "phi": phis,
        }

    def compute_loss(
        self,
        predictions: dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Compute exact negative log-likelihood of bivariate Gaussian MDN + Bernoulli pen-lift.

        Parameters
        ----------
        predictions: Output dictionary from `forward()`
        targets: `(B, T, 3)` ground truth `[dx, dy, pen_up]`
        mask: `(B, T)` boolean mask (True = valid stroke step, False = pad)

        Returns
        -------
        Dictionary containing `loss`, `loss_xy`, and `loss_pen`.
        """
        dx = targets[:, :, 0:1]     # (B, T, 1)
        dy = targets[:, :, 1:2]     # (B, T, 1)
        pen = targets[:, :, 2]      # (B, T)

        pi = predictions["pi"]            # (B, T, M)
        mu_x = predictions["mu_x"]        # (B, T, M)
        mu_y = predictions["mu_y"]        # (B, T, M)
        sigma_x = predictions["sigma_x"]  # (B, T, M)
        sigma_y = predictions["sigma_y"]  # (B, T, M)
        rho = predictions["rho"]          # (B, T, M)
        e = predictions["e"]              # (B, T)

        # Standardise coordinates
        norm_x = (dx - mu_x) / sigma_x
        norm_y = (dy - mu_y) / sigma_y

        one_minus_rho2 = torch.clamp(1.0 - rho ** 2, min=1e-5)

        z = (norm_x ** 2) + (norm_y ** 2) - (2.0 * rho * norm_x * norm_y)
        exp_term = -z / (2.0 * one_minus_rho2)

        # Log probability density of each component j
        log_norm_const = (
            math.log(2.0 * math.pi)
            + torch.log(sigma_x)
            + torch.log(sigma_y)
            + 0.5 * torch.log(one_minus_rho2)
        )
        log_gaussian = exp_term - log_norm_const  # (B, T, M)

        # Log sum exp across mixture components: log P(x, y) = logsumexp(log pi + log N)
        log_prob_xy = torch.logsumexp(torch.log(torch.clamp(pi, min=1e-12)) + log_gaussian, dim=-1)  # (B, T)
        loss_xy = -log_prob_xy

        # Bernoulli pen-lift loss (Binary cross-entropy)
        e_safe = torch.clamp(e, min=1e-7, max=1.0 - 1e-7)
        loss_pen = -(pen * torch.log(e_safe) + (1.0 - pen) * torch.log(1.0 - e_safe))  # (B, T)

        # Apply sequence length mask
        valid_mask = mask.float()
        n_valid = torch.clamp(valid_mask.sum(), min=1.0)

        total_loss_xy = (loss_xy * valid_mask).sum() / n_valid
        total_loss_pen = (loss_pen * valid_mask).sum() / n_valid
        total_loss = total_loss_xy + total_loss_pen

        return {
            "loss": total_loss,
            "loss_xy": total_loss_xy,
            "loss_pen": total_loss_pen,
        }

    @torch.no_grad()
    def generate(
        self,
        text: str,
        alphabet: Alphabet,
        max_steps: int = 800,
        bias: float = 1.0,
        device: torch.device | str = "cpu",
        char_step_scale: float = 0.10,
    ) -> np.ndarray:
        """Synthesize handwriting strokes for input text using monotonic Gaussian soft-window.

        Parameters
        ----------
        text: Text string to synthesize
        alphabet: Pre-fitted `Alphabet`
        max_steps: Safety maximum trajectory step limit
        bias: Sampling neatness bias (b >= 0. Higher = neater, 0 = raw variance)
        device: Torch execution device
        char_step_scale: Attention progression scaling factor (calibrated for stroke-to-character ratio)

        Returns
        -------
        `(N, 4)` float32 array of `[x, y, pen_up, time]` with integrated coordinates.
        """
        self.eval()
        char_ids = alphabet.encode(text)
        if not char_ids:
            return np.zeros((0, 4), dtype=np.float32)

        u_len = len(char_ids)
        c_dim = self.config.char_vocab_size
        k_dim = self.config.num_window_components
        m_dim = self.config.num_gaussians
        h_dim = self.config.hidden_size

        chars = torch.tensor([char_ids], dtype=torch.long, device=device)
        safe_chars = torch.clamp(chars, min=0, max=c_dim - 1)
        char_one_hot = F.one_hot(safe_chars, num_classes=c_dim).float()  # (1, U, C)
        u_positions = torch.arange(u_len, device=device, dtype=torch.float32).view(1, 1, u_len)

        # Initial recurrent states
        h1 = torch.zeros(1, h_dim, device=device)
        c1 = torch.zeros(1, h_dim, device=device)
        h2 = torch.zeros(1, h_dim, device=device)
        c2 = torch.zeros(1, h_dim, device=device)
        h3 = torch.zeros(1, h_dim, device=device)
        c3 = torch.zeros(1, h_dim, device=device)

        w_prev = torch.zeros(1, c_dim, device=device)
        kappa_prev = torch.zeros(1, k_dim, device=device)

        # Initial point is origin with pen on paper: [dx=0, dy=0, pen_up=0]
        x_t = torch.zeros(1, 3, device=device)

        sampled_offsets: list[tuple[float, float, float]] = []

        for step in range(max_steps):
            # Layer 1
            in_1 = torch.cat([x_t, w_prev], dim=-1)
            h1, c1 = self.lstm_1(in_1, (h1, c1))

            # Window Attention
            p = self.window_proj(h1)
            hat_alpha = p[:, :k_dim]
            hat_beta = p[:, k_dim: 2 * k_dim]
            hat_kappa = p[:, 2 * k_dim:]

            alpha = torch.exp(hat_alpha).unsqueeze(-1)
            beta = torch.exp(hat_beta).unsqueeze(-1)
            kappa = (kappa_prev + torch.exp(hat_kappa) * char_step_scale).unsqueeze(-1)
            kappa_prev = kappa.squeeze(-1)

            diff = kappa - u_positions
            phi = torch.sum(alpha * torch.exp(-beta * (diff ** 2)), dim=1)  # (1, U)

            w_curr = torch.bmm(phi.unsqueeze(1), char_one_hot).squeeze(1)   # (1, C)
            w_prev = w_curr

            # Layer 2 & 3
            in_2 = torch.cat([x_t, w_curr, h1], dim=-1)
            h2, c2 = self.lstm_2(in_2, (h2, c2))

            in_3 = torch.cat([x_t, w_curr, h2], dim=-1)
            h3, c3 = self.lstm_3(in_3, (h3, c3))

            # MDN Parameters
            y_t = self.out_proj(torch.cat([h1, h2, h3], dim=-1))

            # ── BIASED SAMPLING (Graves Section 5.3) ──────────────────────────
            e = torch.sigmoid(y_t[:, 0]).item()
            pen_up = 1.0 if np.random.rand() < e else 0.0

            pi_logits = y_t[:, 1: 1 + m_dim] * (1.0 + bias)
            pi = F.softmax(pi_logits, dim=-1).squeeze(0).cpu().numpy()

            mu_x = y_t[:, 1 + m_dim: 1 + 2 * m_dim].squeeze(0).cpu().numpy()
            mu_y = y_t[:, 1 + 2 * m_dim: 1 + 3 * m_dim].squeeze(0).cpu().numpy()

            # Variance reduced by bias for neater lines
            sigma_x = (
                torch.exp(torch.clamp(y_t[:, 1 + 3 * m_dim: 1 + 4 * m_dim], -7.0, 7.0))
                * math.exp(-bias)
            ).squeeze(0).cpu().numpy()

            sigma_y = (
                torch.exp(torch.clamp(y_t[:, 1 + 4 * m_dim: 1 + 5 * m_dim], -7.0, 7.0))
                * math.exp(-bias)
            ).squeeze(0).cpu().numpy()

            rho = torch.clamp(
                torch.tanh(y_t[:, 1 + 5 * m_dim: 1 + 6 * m_dim]), -0.995, 0.995
            ).squeeze(0).cpu().numpy()

            # Sample component j from categorical distribution pi
            j = int(np.random.choice(m_dim, p=pi))

            # Fast exact closed-form bivariate normal sampling
            z1 = float(np.random.randn())
            z2 = float(np.random.randn())
            sample_dx = float(mu_x[j] + sigma_x[j] * z1)
            sample_dy = float(mu_y[j] + sigma_y[j] * (rho[j] * z1 + math.sqrt(max(1e-5, 1.0 - rho[j] ** 2)) * z2))
            sampled_offsets.append((sample_dx, sample_dy, float(pen_up)))

            # Feed sampled point back to next step
            x_t = torch.tensor([[sample_dx, sample_dy, pen_up]], dtype=torch.float32, device=device)

            # Organic Graves stopping condition: window reached end of text
            avg_kappa = float(kappa_prev.mean().item())
            if avg_kappa > u_len + 0.5:
                # Give a few buffer strokes for the last letter
                break

        if not sampled_offsets:
            return np.zeros((0, 4), dtype=np.float32)

        arr = np.array(sampled_offsets, dtype=np.float32)  # (N, 3): [dx, dy, pen_up]
        xy = np.cumsum(arr[:, :2], axis=0)
        pen = arr[:, 2:3]
        time = np.arange(len(arr), dtype=np.float32).reshape(-1, 1) * 0.02  # Synthetic timing

        return np.concatenate([xy, pen, time], axis=1).astype(np.float32)
