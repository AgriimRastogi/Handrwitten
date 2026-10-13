"""PyTorch Dataset and DataLoader collation for packed HDF5 handwriting datasets.

This module bridges the offline packed datasets (train.h5, val.h5, test.h5)
to PyTorch neural network training.

KEY DESIGN DECISIONS
--------------------
1. Multi-worker fork safety:
   HDF5 file handles are not fork-safe across multiple DataLoader worker
   processes. `StrokeDataset` caches lightweight index arrays (offsets,
   char_offsets, writer_index, form_id -- ~100 KB total) in memory at
   initialisation, and lazily instantiates the `h5py.File` handle on first
   read within each worker process.

2. Zero zero-padding on disk, dynamic padding in collate:
   Variable-length sequences are stored ragged in HDF5 without dummy padding.
   `collate_strokes` dynamically pads sequences to the maximum length *in each
   mini-batch* rather than a dataset-wide maximum, saving memory and compute.

3. Autoregressive teacher-forcing alignment:
   For sequence length T, inputs are strokes 0..T-2 and targets are 1..T-1.
   Padded target tokens are set to -100 so `torch.nn.CrossEntropyLoss` ignores
   them automatically.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset

from hws.alphabet import PAD_INDEX


class StrokeDataset(Dataset):
    """PyTorch Dataset streaming lines from a packed HDF5 file.

    Parameters
    ----------
    path:
        Path to an HDF5 file produced by `scripts/prepare_data.py`.
    load_raw:
        If True, loads unquantized float32 offsets `(T, 4)` along with
        discrete tokens. Set False during pure tokenized training to save
        I/O and RAM.
    """

    def __init__(self, path: str | Path, load_raw: bool = True) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"HDF5 dataset not found: {self.path}")

        self.load_raw = load_raw

        # Read lightweight index and metadata once into memory.
        # Across 9,000 lines, these arrays occupy less than 200 KB in RAM.
        with h5py.File(self.path, "r") as handle:
            self.attrs = dict(handle.attrs)
            self.n_lines = int(self.attrs["n_lines"])
            self.offsets = np.asarray(handle["offsets"], dtype=np.int64)
            self.char_offsets = np.asarray(handle["char_offsets"], dtype=np.int64)
            self.writer_indices = np.asarray(handle["writer_index"], dtype=np.int32)
            self.line_indices = np.asarray(handle["line_index"], dtype=np.int16)

            # Form IDs (e.g. "a01-000u") stored as strings
            raw_form_ids = handle["form_id"]
            self.form_ids = [
                fid.decode("utf-8") if isinstance(fid, (bytes, np.bytes_)) else str(fid)
                for fid in raw_form_ids
            ]

            # Available token channels (e.g. ["dx", "dy", "dt", "pen"])
            token_group = handle["tokens"]
            self.token_channels = [c for c in token_group.keys()]

        # File handle is lazily opened per-process in `_get_file()`
        self._file: h5py.File | None = None

    def _get_file(self) -> h5py.File:
        """Lazily acquire process-local HDF5 file handle."""
        if self._file is None:
            self._file = h5py.File(self.path, "r", swmr=True)
        return self._file

    def close(self) -> None:
        """Explicitly close the open HDF5 handle if present."""
        if self._file is not None:
            try:
                self._file.close()
            except Exception:
                pass
            self._file = None

    def __del__(self) -> None:
        self.close()

    def __len__(self) -> int:
        return self.n_lines

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if idx < 0 or idx >= self.n_lines:
            raise IndexError(f"Index {idx} out of range [0, {self.n_lines})")

        p_start = self.offsets[idx]
        p_end = self.offsets[idx + 1]

        c_start = self.char_offsets[idx]
        c_end = self.char_offsets[idx + 1]

        handle = self._get_file()

        # Characters: int16 on disk -> int64 LongTensor for nn.Embedding
        chars_arr = np.asarray(handle["chars"][c_start:c_end], dtype=np.int64)
        chars = torch.from_numpy(chars_arr)

        # Stroke tokens: int16 on disk -> int64 LongTensor
        tokens_grp = handle["tokens"]
        sample: dict[str, Any] = {
            "chars": chars,
            "writer": int(self.writer_indices[idx]),
            "line_index": int(self.line_indices[idx]),
            "form_id": self.form_ids[idx],
            "n_points": int(p_end - p_start),
            "n_chars": int(c_end - c_start),
        }

        for channel in self.token_channels:
            channel_slice = np.asarray(tokens_grp[channel][p_start:p_end], dtype=np.int64)
            sample[channel] = torch.from_numpy(channel_slice)

        if self.load_raw and "raw" in handle:
            raw_slice = np.asarray(handle["raw"][p_start:p_end], dtype=np.float32)
            sample["raw"] = torch.from_numpy(raw_slice)
        else:
            sample["raw"] = None

        return sample


def collate_strokes(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate variable-length stroke sequences and text into a batched dictionary.

    Parameters
    ----------
    batch:
        List of single sample dictionaries returned by `StrokeDataset`.

    Returns
    -------
    dict with:
        `chars`: `(B, max_L)` padded character tokens (pad = 0)
        `char_lens`: `(B,)` sequence lengths of characters
        `char_mask`: `(B, max_L)` bool mask (True = real character, False = pad)
        `stroke_lens`: `(B,)` sequence lengths of strokes
        `stroke_mask`: `(B, max_T)` bool mask (True = real point, False = pad)
        `dx`, `dy`, `dt`, `pen`: `(B, max_T)` padded full stroke tokens
        `in_dx`, `in_dy`, `in_dt`, `in_pen`: `(B, max_T - 1)` teacher-forcing inputs (steps 0..T-2)
        `target_dx`, `target_dy`, `target_dt`, `target_pen`: `(B, max_T - 1)` prediction targets
            (steps 1..T-1) with padded positions set to -100 for CrossEntropyLoss
        `in_mask`: `(B, max_T - 1)` bool mask for the input steps
        `raw`: `(B, max_T, 4)` padded float offsets if loaded, else None
        `writer`: `(B,)` int64 tensor of writer IDs
        `form_ids`: list of str form IDs
        `line_indices`: list of int line indices
    """
    batch_size = len(batch)

    # ── 1. Characters ──────────────────────────────────────────────────────────
    chars_list = [item["chars"] for item in batch]
    char_lens = torch.tensor([item["n_chars"] for item in batch], dtype=torch.long)
    padded_chars = pad_sequence(chars_list, batch_first=True, padding_value=PAD_INDEX)
    max_char_len = padded_chars.shape[1]
    char_mask = torch.arange(max_char_len).unsqueeze(0) < char_lens.unsqueeze(1)

    # ── 2. Strokes ─────────────────────────────────────────────────────────────
    stroke_lens = torch.tensor([item["n_points"] for item in batch], dtype=torch.long)
    max_stroke_len = int(stroke_lens.max().item())
    stroke_mask = torch.arange(max_stroke_len).unsqueeze(0) < stroke_lens.unsqueeze(1)

    # Identify which token channels are present (e.g. dx, dy, dt, pen)
    token_keys = [k for k in ("dx", "dy", "dt", "pen") if k in batch[0]]

    collated: dict[str, Any] = {
        "chars": padded_chars,
        "char_lens": char_lens,
        "char_mask": char_mask,
        "stroke_lens": stroke_lens,
        "stroke_mask": stroke_mask,
        "writer": torch.tensor([item["writer"] for item in batch], dtype=torch.long),
        "form_ids": [item["form_id"] for item in batch],
        "line_indices": [item["line_index"] for item in batch],
    }

    # For autoregressive training, input has length (max_T - 1) and target has length (max_T - 1)
    max_in_len = max(0, max_stroke_len - 1)
    in_mask = (
        torch.arange(max_in_len).unsqueeze(0) < (stroke_lens - 1).clamp(min=0).unsqueeze(1)
        if max_in_len > 0
        else torch.zeros((batch_size, 0), dtype=torch.bool)
    )
    collated["in_mask"] = in_mask

    for key in token_keys:
        tensors = [item[key] for item in batch]
        padded = pad_sequence(tensors, batch_first=True, padding_value=0)
        collated[key] = padded

        if max_in_len > 0:
            # Inputs: steps 0 .. T-2
            collated[f"in_{key}"] = padded[:, :-1]
            # Targets: steps 1 .. T-1
            targets = padded[:, 1:].clone()
            # Mask out padding with -100 for CrossEntropyLoss ignore_index
            targets.masked_fill_(~in_mask, -100)
            collated[f"target_{key}"] = targets
        else:
            collated[f"in_{key}"] = torch.zeros((batch_size, 0), dtype=torch.long)
            collated[f"target_{key}"] = torch.zeros((batch_size, 0), dtype=torch.long)

    # ── 3. Raw Offsets (if loaded) ─────────────────────────────────────────────
    if batch[0]["raw"] is not None:
        raw_list = [item["raw"] for item in batch]
        padded_raw = pad_sequence(raw_list, batch_first=True, padding_value=0.0)
        collated["raw"] = padded_raw
    else:
        collated["raw"] = None

    return collated


def get_dataloader(
    path: str | Path,
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    load_raw: bool = False,
    drop_last: bool = False,
) -> DataLoader:
    """Create a DataLoader streaming batches from an HDF5 dataset.

    Parameters
    ----------
    path:
        Path to an HDF5 dataset file (e.g. `datas/processed/train.h5`).
    batch_size:
        Mini-batch size.
    shuffle:
        Whether to shuffle line order per epoch.
    num_workers:
        Number of background worker subprocesses for data loading.
        Multi-worker loading is fully supported via lazy file instantiation.
    pin_memory:
        Whether to pin memory for faster GPU page-locked transfers.
    load_raw:
        Whether to load continuous float offsets alongside tokens.
    drop_last:
        Whether to drop the last incomplete batch.
    """
    dataset = StrokeDataset(path, load_raw=load_raw)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_strokes,
        drop_last=drop_last,
    )
