"""Loss functions for Continuous Guided-Attention Handwriting Synthesis.

Combines Bivariate Gaussian Mixture Density Network (MDN) negative log-likelihood
with diagonal guided attention loss for continuous trajectory generation.
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_guided_attention_loss(
    attn_weights: torch.Tensor,
    stroke_lens: torch.Tensor,
    char_lens: torch.Tensor,
    sigma: float = 0.2,
) -> torch.Tensor:
    """Compute diagonal guided attention loss enforcing monotonic left-to-right alignment.

    Parameters
    ----------
    attn_weights: `(B, nhead, T, L)` attention alignment weights
    stroke_lens: `(B,)` valid stroke lengths per sample
    char_lens: `(B,)` valid character lengths per sample
    sigma: width corridor of the allowed diagonal band

    Returns
    -------
    Scalar loss penalty for off-diagonal attention drift
    """
    # Average across attention heads: (B, T, L)
    a_mean = attn_weights.mean(dim=1)
    batch_size, max_t, max_l = a_mean.shape
    device = a_mean.device

    losses = []
    two_sigma_sq = 2.0 * (sigma**2)

    for b in range(batch_size):
        t_b = int(stroke_lens[b].item())
        l_b = int(char_lens[b].item())
        if t_b <= 1 or l_b <= 1:
            continue

        # Normalized coordinates [0, 1]
        t_grid = torch.linspace(0, 1, t_b, device=device).unsqueeze(1)  # (T_b, 1)
        l_grid = torch.linspace(0, 1, l_b, device=device).unsqueeze(0)  # (1, L_b)

        # Distance penalty matrix: 0 on diagonal, ~1 far off-diagonal
        w = 1.0 - torch.exp(-((t_grid - l_grid) ** 2) / two_sigma_sq)  # (T_b, L_b)

        # Slice actual valid region
        a_valid = a_mean[b, :t_b, :l_b]

        # Normalized by query time steps t_b (sum across keys l_b is already ~1.0)
        sample_loss = (a_valid * w).sum() / t_b
        losses.append(sample_loss)

    if not losses:
        return torch.tensor(0.0, device=device)

    return torch.stack(losses).mean()


class ContinuousGuidedSynthesisLoss(nn.Module):
    """Bivariate Gaussian MDN Loss combined with Guided Monotonic Attention."""

    def __init__(self, guided_weight: float = 1.0, sigma: float = 0.2) -> None:
        super().__init__()
        self.guided_weight = guided_weight
        self.sigma = sigma

    def forward(
        self,
        predictions: dict[str, torch.Tensor],
        targets: torch.Tensor,
        mask: torch.Tensor,
        attn_weights: torch.Tensor | None = None,
        stroke_lens: torch.Tensor | None = None,
        char_lens: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute exact negative log-likelihood of bivariate Gaussian MDN + pen loss + alignment.

        Parameters
        ----------
        predictions: dict containing 'pi', 'mu_x', 'mu_y', 'sigma_x', 'sigma_y', 'rho', 'e'
        targets: `(B, T, 3)` ground truth `[dx, dy, pen_up]`
        mask: `(B, T)` boolean mask (True = valid stroke step, False = pad)
        attn_weights: `(B, nhead, T, L)` cross-attention weights
        stroke_lens: `(B,)` valid stroke lengths
        char_lens: `(B,)` valid character lengths

        Returns
        -------
        Dictionary containing `total_loss`, `loss_xy`, `loss_pen`, and `loss_guided`.
        """
        dx = targets[:, :, 0:1]     # (B, T, 1)
        dy = targets[:, :, 1:2]     # (B, T, 1)
        pen = targets[:, :, 2]      # (B, T)

        pi = predictions["pi"]            # (B, T, M)
        mu_x = predictions["mu_x"]        # (B, T, M)
        mu_y = predictions["mu_y"]        # (B, T, M)
        sigma_x = torch.clamp(predictions["sigma_x"], min=1e-4)  # (B, T, M)
        sigma_y = torch.clamp(predictions["sigma_y"], min=1e-4)  # (B, T, M)
        rho = torch.clamp(predictions["rho"], min=-0.999, max=0.999)  # (B, T, M)
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
        loss_mdn = total_loss_xy + total_loss_pen

        # Guided alignment loss
        if attn_weights is not None and stroke_lens is not None and char_lens is not None:
            l_guided = compute_guided_attention_loss(
                attn_weights, stroke_lens, char_lens, sigma=self.sigma
            )
        else:
            l_guided = torch.tensor(0.0, device=dx.device)

        total = loss_mdn + (self.guided_weight * l_guided)

        return {
            "total_loss": total,
            "loss_mdn": loss_mdn,
            "loss_xy": total_loss_xy,
            "loss_pen": total_loss_pen,
            "loss_guided": l_guided,
        }
