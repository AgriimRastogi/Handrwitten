"""Configuration for the Continuous Guided-Attention Handwriting Synthesis Model."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class NewModelConfig:
    """Hyperparameters and architecture configuration for Continuous Transformer-MDN."""

    # Continuous sequence dimensions
    char_vocab_size: int = 78
    n_writers: int = 197
    input_dim: int = 3           # [dx, dy, pen_up] continuous stroke coordinates
    num_gaussians: int = 20      # M = 20 bivariate Gaussian mixture components

    # Transformer dimensions
    d_model: int = 256
    nhead: int = 4
    num_encoder_layers: int = 2
    num_decoder_layers: int = 4
    dim_feedforward: int = 1024
    dropout: float = 0.1
    max_seq_len: int = 1200

    # Alignment prior loss (Guided Monotonic Attention)
    guided_attention_weight: float = 0.1
    guided_attention_sigma: float = 0.2

    # Training parameters
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4

    @classmethod
    def from_config_json(cls, path: str | Path, **kwargs: Any) -> NewModelConfig:
        """Instantiate configuration from dataset config.json with custom overrides."""
        path = Path(path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls(
            char_vocab_size=int(data.get("char_vocab_size", 78)),
            n_writers=int(data.get("n_writers", 197)),
            **kwargs,
        )
