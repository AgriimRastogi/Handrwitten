"""Unit tests for the neural sequence model (hws.model)."""

from pathlib import Path

import numpy as np
import pytest
import torch

from hws.alphabet import Alphabet
from hws.model import HandwritingSynthesisModel, ModelConfig
from hws.tokenizer import StrokeTokenizer

CONFIG_PATH = Path(__file__).resolve().parent.parent / "datas" / "processed" / "config.json"
TOKENIZER_PATH = Path(__file__).resolve().parent.parent / "datas" / "processed" / "tokenizer.json"
ALPHABET_PATH = Path(__file__).resolve().parent.parent / "datas" / "processed" / "alphabet.json"


@pytest.fixture
def small_config() -> ModelConfig:
    return ModelConfig(
        char_vocab_size=20,
        n_writers=10,
        vocab_dx=32,
        vocab_dy=32,
        vocab_dt=16,
        vocab_pen=2,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=2,
        dim_feedforward=128,
        dropout=0.0,
        max_seq_len=200,
    )


def test_model_config_loading():
    if CONFIG_PATH.exists():
        cfg = ModelConfig.from_config_json(CONFIG_PATH)
        assert cfg.vocab_dx == 512
        assert cfg.vocab_dy == 512
        assert cfg.vocab_dt == 266
        assert cfg.vocab_pen == 2
        assert cfg.char_vocab_size == 78


def test_model_forward_shapes(small_config: ModelConfig):
    model = HandwritingSynthesisModel(small_config)
    b, l, t = 2, 5, 8

    chars = torch.randint(1, small_config.char_vocab_size, (b, l))
    char_mask = torch.ones((b, l), dtype=torch.bool)

    in_dx = torch.randint(0, small_config.vocab_dx, (b, t))
    in_dy = torch.randint(0, small_config.vocab_dy, (b, t))
    in_dt = torch.randint(0, small_config.vocab_dt, (b, t))
    in_pen = torch.randint(0, 2, (b, t))
    in_mask = torch.ones((b, t), dtype=torch.bool)
    writer = torch.tensor([1, 2], dtype=torch.long)

    logits = model(
        chars=chars,
        in_dx=in_dx,
        in_dy=in_dy,
        in_dt=in_dt,
        in_pen=in_pen,
        char_mask=char_mask,
        in_mask=in_mask,
        writer=writer,
    )

    assert logits["dx"].shape == (b, t, small_config.vocab_dx)
    assert logits["dy"].shape == (b, t, small_config.vocab_dy)
    assert logits["dt"].shape == (b, t, small_config.vocab_dt)
    assert logits["pen"].shape == (b, t, small_config.vocab_pen)


def test_model_loss_and_gradients(small_config: ModelConfig):
    model = HandwritingSynthesisModel(small_config)
    b, l, t = 2, 6, 10

    chars = torch.randint(1, small_config.char_vocab_size, (b, l))
    in_dx = torch.randint(0, small_config.vocab_dx, (b, t))
    in_dy = torch.randint(0, small_config.vocab_dy, (b, t))
    in_dt = torch.randint(0, small_config.vocab_dt, (b, t))
    in_pen = torch.randint(0, 2, (b, t))

    targets = {
        "target_dx": torch.randint(0, small_config.vocab_dx, (b, t)),
        "target_dy": torch.randint(0, small_config.vocab_dy, (b, t)),
        "target_dt": torch.randint(0, small_config.vocab_dt, (b, t)),
        "target_pen": torch.randint(0, 2, (b, t)),
    }
    # Set padded elements to -100
    targets["target_dx"][:, -2:] = -100
    targets["target_dy"][:, -2:] = -100
    targets["target_dt"][:, -2:] = -100
    targets["target_pen"][:, -2:] = -100

    writer = torch.tensor([1, 2], dtype=torch.long)
    logits = model(chars, in_dx, in_dy, in_dt, in_pen, writer=writer)
    loss_dict = model.compute_loss(logits, targets)

    assert "loss" in loss_dict
    assert torch.isfinite(loss_dict["loss"])
    assert loss_dict["loss"].item() > 0

    loss_dict["loss"].backward()
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"Parameter {name} has no gradient"


def test_model_causality(small_config: ModelConfig):
    """Ensure decoder prediction at step t does NOT depend on input at step t+k."""
    model = HandwritingSynthesisModel(small_config)
    model.eval()

    b, l, t = 1, 4, 8
    chars = torch.randint(1, small_config.char_vocab_size, (b, l))

    in_dx = torch.randint(0, small_config.vocab_dx, (b, t))
    in_dy = torch.randint(0, small_config.vocab_dy, (b, t))
    in_dt = torch.randint(0, small_config.vocab_dt, (b, t))
    in_pen = torch.randint(0, 2, (b, t))

    with torch.no_grad():
        out1 = model(chars, in_dx, in_dy, in_dt, in_pen)["dx"]

    # Modify the last 2 steps of input
    in_dx_mod = in_dx.clone()
    in_dx_mod[0, -2:] = (in_dx_mod[0, -2:] + 1) % small_config.vocab_dx

    with torch.no_grad():
        out2 = model(chars, in_dx_mod, in_dy, in_dt, in_pen)["dx"]

    # Outputs at steps 0 .. t-3 MUST be strictly identical
    assert torch.allclose(out1[:, :-2, :], out2[:, :-2, :], atol=1e-5)


@pytest.mark.skipif(
    not (CONFIG_PATH.exists() and TOKENIZER_PATH.exists() and ALPHABET_PATH.exists()),
    reason="Processed data artifacts not present",
)
def test_model_generate_smoke():
    config = ModelConfig.from_config_json(
        CONFIG_PATH,
        d_model=64,
        nhead=2,
        num_encoder_layers=1,
        num_decoder_layers=1,
        dim_feedforward=128,
    )
    model = HandwritingSynthesisModel(config)

    alphabet = Alphabet.load(ALPHABET_PATH)
    tokenizer = StrokeTokenizer.load(TOKENIZER_PATH)

    points = model.generate(
        text="test line",
        alphabet=alphabet,
        tokenizer=tokenizer,
        writer_id=0,
        max_steps=20,
        temperature=0.8,
    )

    assert isinstance(points, np.ndarray)
    assert points.shape[1] == 4
    assert len(points) > 1
    assert set(points[:, 2].tolist()).issubset({0.0, 1.0})
