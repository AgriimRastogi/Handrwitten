"""Smoke test for the end-to-end training and evaluation pipeline (scripts/train.py)."""

import shutil
from pathlib import Path

import pytest
import torch

from scripts.train import main as train_main

DATA_DIR = Path(__file__).resolve().parent.parent / "datas" / "processed"
TRAIN_H5 = DATA_DIR / "train.h5"
VAL_H5 = DATA_DIR / "val.h5"
CONFIG_JSON = DATA_DIR / "config.json"


@pytest.mark.skipif(
    not (TRAIN_H5.exists() and VAL_H5.exists() and CONFIG_JSON.exists()),
    reason="Processed HDF5 datasets not present on disk",
)
def test_train_smoke_run(tmp_path: Path):
    ckpt_dir = tmp_path / "checkpoints"
    samples_dir = tmp_path / "samples"

    cli_args = [
        "--train-data", str(TRAIN_H5),
        "--val-data", str(VAL_H5),
        "--config", str(CONFIG_JSON),
        "--alphabet", str(DATA_DIR / "alphabet.json"),
        "--tokenizer", str(DATA_DIR / "tokenizer.json"),
        "--checkpoint-dir", str(ckpt_dir),
        "--samples-dir", str(samples_dir),
        "--batch-size", "4",
        "--device", "cpu",
        "--smoke-test",
    ]

    exit_code = train_main(cli_args)
    assert exit_code == 0

    # Verify best_model.pt checkpoint was created
    best_ckpt = ckpt_dir / "best_model.pt"
    data = torch.load(best_ckpt, map_location="cpu", weights_only=True)
    assert "model_state" in data
    assert "val_loss" in data
    assert "config" in data

    # Verify at least one SVG sample was generated
    svg_files = list(samples_dir.glob("*.svg"))
    assert len(svg_files) >= 1
    svg_content = svg_files[0].read_text(encoding="utf-8")
    assert "<svg" in svg_content
    assert "</svg>" in svg_content
    assert "<path" in svg_content
