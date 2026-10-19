"""Unit tests for the continuous new_model package."""

import numpy as np
import pytest
import torch

import sys
from pathlib import Path

handwritten_dir = Path(__file__).resolve().parent.parent
if str(handwritten_dir) not in sys.path:
    sys.path.insert(0, str(handwritten_dir))

from new_model.config import NewModelConfig
from new_model.loss import ContinuousGuidedSynthesisLoss, compute_guided_attention_loss
from new_model.model import ContinuousGuidedTransformer


@pytest.fixture
def tiny_config():
    return NewModelConfig(
        char_vocab_size=20,
        n_writers=5,
        input_dim=3,
        num_gaussians=10,
        d_model=32,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=64,
        dropout=0.0,
        max_seq_len=50,
    )


def test_continuous_model_forward_shape(tiny_config):
    model = ContinuousGuidedTransformer(tiny_config)
    model.eval()

    b, l, t = 2, 8, 12
    chars = torch.randint(1, 19, (b, l))
    char_mask = torch.ones((b, l), dtype=torch.bool)

    # Continuous strokes: [dx, dy, pen_up]
    strokes = torch.randn((b, t, 3), dtype=torch.float32)
    stroke_mask = torch.ones((b, t), dtype=torch.bool)
    writer = torch.tensor([0, 1], dtype=torch.long)

    out = model(
        chars=chars,
        strokes=strokes,
        char_mask=char_mask,
        stroke_mask=stroke_mask,
        writer=writer,
    )

    m = tiny_config.num_gaussians
    assert out["pi"].shape == (b, t, m)
    assert out["mu_x"].shape == (b, t, m)
    assert out["mu_y"].shape == (b, t, m)
    assert out["sigma_x"].shape == (b, t, m)
    assert out["sigma_y"].shape == (b, t, m)
    assert out["rho"].shape == (b, t, m)
    assert out["e"].shape == (b, t)
    assert out["attn_weights"].shape == (b, tiny_config.nhead, t, l)

    # Check softmax sum to 1
    assert torch.allclose(out["pi"].sum(dim=-1), torch.ones((b, t)), atol=1e-5)


def test_continuous_guided_loss_computation(tiny_config):
    b, l, t = 2, 6, 10
    nhead = tiny_config.nhead

    # Synthetic diagonal attention
    attn_weights = torch.zeros((b, nhead, t, l))
    for step in range(t):
        char_idx = int(step * (l / t))
        attn_weights[:, :, step, char_idx] = 1.0

    stroke_lens = torch.tensor([t, t], dtype=torch.long)
    char_lens = torch.tensor([l, l], dtype=torch.long)

    loss_val = compute_guided_attention_loss(attn_weights, stroke_lens, char_lens, sigma=0.2)
    assert loss_val.item() >= 0.0
    assert torch.isfinite(loss_val)

    # Off-diagonal attention should yield higher penalty than diagonal
    off_diag_attn = torch.zeros((b, nhead, t, l))
    for step in range(t):
        char_idx = l - 1 - int(step * (l / t))
        off_diag_attn[:, :, step, char_idx] = 1.0

    off_diag_loss = compute_guided_attention_loss(off_diag_attn, stroke_lens, char_lens, sigma=0.2)
    assert off_diag_loss.item() > loss_val.item()

    # Full MDN loss test
    criterion = ContinuousGuidedSynthesisLoss(guided_weight=0.1)
    m = tiny_config.num_gaussians
    predictions = {
        "pi": torch.full((b, t, m), 1.0 / m),
        "mu_x": torch.zeros((b, t, m)),
        "mu_y": torch.zeros((b, t, m)),
        "sigma_x": torch.ones((b, t, m)),
        "sigma_y": torch.ones((b, t, m)),
        "rho": torch.zeros((b, t, m)),
        "e": torch.full((b, t), 0.1),
    }
    targets = torch.zeros((b, t, 3), dtype=torch.float32)
    mask = torch.ones((b, t), dtype=torch.bool)

    loss_dict = criterion(
        predictions=predictions,
        targets=targets,
        mask=mask,
        attn_weights=attn_weights,
        stroke_lens=stroke_lens,
        char_lens=char_lens,
    )

    assert "total_loss" in loss_dict
    assert torch.isfinite(loss_dict["total_loss"])
    assert loss_dict["loss_xy"].item() > 0.0


def test_continuous_model_generation(tiny_config):
    model = ContinuousGuidedTransformer(tiny_config)
    model.eval()

    chars = torch.randint(1, 19, (1, 5))
    writer = torch.tensor([0], dtype=torch.long)

    gen = model.generate(chars, writer=writer, max_steps=15, bias=0.5, stop_on_completion=False)
    assert "coords" in gen
    assert "offsets" in gen
    assert gen["coords"].shape[0] == 15
    assert gen["coords"].shape[1] == 4
    assert gen["offsets"].shape[0] == 15
    assert gen["offsets"].shape[1] == 4


def test_repeated_characters_advances_without_barcode_loop(tiny_config):
    """Verifies that repetitive character sequences advance monotonically

    without getting trapped in an infinite barcode attractor loop.
    """
    model = ContinuousGuidedTransformer(tiny_config)
    model.eval()

    l_len = 8
    chars = torch.full((1, l_len), 5, dtype=torch.long)
    writer = torch.tensor([0], dtype=torch.long)

    max_dwell = 4
    gen = model.generate(
        chars,
        writer=writer,
        max_steps=40,
        bias=0.5,
        max_dwell_steps=max_dwell,
        stop_on_completion=False,
    )

    attn_history = gen["attn_history"]
    assert attn_history is not None
    assert len(attn_history) == 40

    centers = [(np.arange(l_len) * step_attn).sum() for step_attn in attn_history]
    first_quarter_mean = np.mean(centers[:10])
    last_quarter_mean = np.mean(centers[-10:])

    assert last_quarter_mean > first_quarter_mean
    assert last_quarter_mean > 3.0, f"Expected attention to advance, but got {last_quarter_mean}"


def test_monotonic_mask_forward(tiny_config):
    """Verifies that forward pass correctly processes monotonic masks."""
    model = ContinuousGuidedTransformer(tiny_config)
    model.eval()

    b, l, t = 1, 6, 8
    chars = torch.randint(1, 19, (b, l))
    char_mask = torch.ones((b, l), dtype=torch.bool)
    strokes = torch.randn((b, t, 3), dtype=torch.float32)
    stroke_mask = torch.ones((b, t), dtype=torch.bool)

    mono_mask = torch.ones((b, 1, t, l), dtype=torch.bool)
    mono_mask[:, :, 4:, 0] = False

    out = model(
        chars=chars,
        strokes=strokes,
        char_mask=char_mask,
        stroke_mask=stroke_mask,
        monotonic_mask=mono_mask,
    )
    assert out["attn_weights"].shape == (b, tiny_config.nhead, t, l)
    assert torch.all(out["attn_weights"][:, :, 4:, 0] < 1e-4)
