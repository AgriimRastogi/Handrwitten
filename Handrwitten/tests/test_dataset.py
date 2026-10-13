"""Unit tests for PyTorch Dataset and DataLoader collation (hws.dataset)."""

from pathlib import Path

import pytest
import torch

from hws.dataset import StrokeDataset, collate_strokes, get_dataloader

VAL_H5 = Path(__file__).resolve().parent.parent / "datas" / "processed" / "val.h5"


@pytest.mark.skipif(not VAL_H5.exists(), reason="datas/processed/val.h5 not generated")
def test_dataset_metadata_and_length():
    dataset = StrokeDataset(VAL_H5, load_raw=True)
    assert len(dataset) > 0
    assert dataset.n_lines == len(dataset)
    assert "dx" in dataset.token_channels
    assert "dy" in dataset.token_channels
    assert "pen" in dataset.token_channels
    assert len(dataset.form_ids) == len(dataset)
    assert len(dataset.writer_indices) == len(dataset)
    dataset.close()


@pytest.mark.skipif(not VAL_H5.exists(), reason="datas/processed/val.h5 not generated")
def test_dataset_getitem_types_and_shapes():
    dataset = StrokeDataset(VAL_H5, load_raw=True)
    sample = dataset[0]

    assert isinstance(sample["chars"], torch.LongTensor)
    assert isinstance(sample["dx"], torch.LongTensor)
    assert isinstance(sample["dy"], torch.LongTensor)
    assert isinstance(sample["pen"], torch.LongTensor)
    assert isinstance(sample["raw"], torch.FloatTensor)

    n_points = sample["n_points"]
    n_chars = sample["n_chars"]

    assert sample["chars"].shape == (n_chars,)
    assert sample["dx"].shape == (n_points,)
    assert sample["dy"].shape == (n_points,)
    assert sample["pen"].shape == (n_points,)
    assert sample["raw"].shape == (n_points, 4)

    # Pen values must be binary: 0 or 1
    assert set(sample["pen"].tolist()).issubset({0, 1})
    dataset.close()


@pytest.mark.skipif(not VAL_H5.exists(), reason="datas/processed/val.h5 not generated")
def test_collate_strokes_padding_and_alignment():
    dataset = StrokeDataset(VAL_H5, load_raw=True)
    # Pick two samples of different lengths
    s1 = dataset[0]
    s2 = dataset[1]
    dataset.close()

    batch = collate_strokes([s1, s2])

    max_chars = max(s1["n_chars"], s2["n_chars"])
    max_strokes = max(s1["n_points"], s2["n_points"])

    assert batch["chars"].shape == (2, max_chars)
    assert batch["char_mask"].shape == (2, max_chars)
    assert batch["stroke_mask"].shape == (2, max_strokes)
    assert batch["dx"].shape == (2, max_strokes)

    # Masks must match lengths exactly
    assert batch["char_mask"][0].sum().item() == s1["n_chars"]
    assert batch["char_mask"][1].sum().item() == s2["n_chars"]
    assert batch["stroke_mask"][0].sum().item() == s1["n_points"]
    assert batch["stroke_mask"][1].sum().item() == s2["n_points"]

    # Teacher forcing alignment:
    # Inputs have length max_strokes - 1
    # Targets have length max_strokes - 1
    max_in = max_strokes - 1
    assert batch["in_dx"].shape == (2, max_in)
    assert batch["target_dx"].shape == (2, max_in)

    # Verify shift: in_dx at t is dx at t; target_dx at t is dx at t+1 (where valid)
    valid_len_0 = s1["n_points"] - 1
    assert torch.equal(batch["in_dx"][0, :valid_len_0], batch["dx"][0, :valid_len_0])
    assert torch.equal(batch["target_dx"][0, :valid_len_0], batch["dx"][0, 1 : valid_len_0 + 1])

    # Targets in padded positions must be -100 (CrossEntropyLoss ignore_index)
    if s1["n_points"] < max_strokes:
        assert (batch["target_dx"][0, valid_len_0:] == -100).all()
    if s2["n_points"] < max_strokes:
        assert (batch["target_dx"][1, (s2["n_points"] - 1):] == -100).all()


@pytest.mark.skipif(not VAL_H5.exists(), reason="datas/processed/val.h5 not generated")
def test_dataloader_iteration():
    loader = get_dataloader(VAL_H5, batch_size=8, shuffle=False, num_workers=0)
    batches_seen = 0
    for batch in loader:
        assert "chars" in batch
        assert "in_dx" in batch
        assert "target_dx" in batch
        assert batch["chars"].shape[0] <= 8
        batches_seen += 1
        if batches_seen >= 3:
            break
    assert batches_seen == 3
