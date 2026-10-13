"""Tests for stroke tokenisation.

    pytest -q tests/test_tokenizer.py

The two things most worth guarding here:

1. **The jump-mask off-by-one.** Pen-up jumps are the row *after* a pen_up
   flag, not the flagged row. Get it backwards and the jump bins get fitted on
   pen-down data, which is worse than not splitting at all — and it produces
   no error, just quietly bad output.

2. **Token ranges stay separated.** Move tokens must land below ``n_move`` and
   jump tokens above it. If they overlap, a single token id means two
   different displacements and decoding is ambiguous.
"""

from __future__ import annotations

import numpy as np
import pytest

from hws.iam import PEN_UP, TIME, X, Y
from hws.tokenizer import StrokeTokenizer, jump_mask


def make_offsets(
    n_strokes: int = 6,
    points_per_stroke: int = 30,
    jump_scale: float = 15.0,
    seed: int = 0,
) -> np.ndarray:
    """Offsets with a realistic bimodal step distribution.

    Small pen-down steps within strokes, big pen-up jumps between them — which
    is the structure the two-mode binning exists to handle. ``jump_scale``
    controls how separated the two modes are.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for stroke_index in range(n_strokes):
        for point_index in range(points_per_stroke):
            is_first_of_stroke = point_index == 0
            if is_first_of_stroke and stroke_index > 0:
                # Arriving at a new stroke = the airborne jump.
                dx, dy, dt = jump_scale + rng.normal(0, 2), rng.normal(0, 3), 0.25
            else:
                dx, dy, dt = rng.normal(0.6, 0.3), rng.normal(0, 0.5), 0.02
            pen_up = 1.0 if point_index == points_per_stroke - 1 else 0.0
            rows.append([dx, dy, pen_up, dt])

    offsets = np.asarray(rows, dtype=np.float32)
    offsets[0, [X, Y, TIME]] = 0.0  # first row has no predecessor
    return offsets


@pytest.fixture
def fitted() -> StrokeTokenizer:
    corpus = [make_offsets(seed=s) for s in range(20)]
    return StrokeTokenizer.fit(corpus, n_bins=64, n_jump_bins=64)


# ── jump mask ─────────────────────────────────────────────────────────────────


def test_jump_mask_marks_row_after_pen_up_not_the_flagged_row():
    """The critical off-by-one. See module docstring."""
    offsets = np.zeros((6, 4), dtype=np.float32)
    offsets[2, PEN_UP] = 1.0  # pen lifts after point 2

    mask = jump_mask(offsets)
    assert not mask[2], "the pen_up row itself is a pen-DOWN move"
    assert mask[3], "the row after pen_up is the airborne jump"
    assert mask.sum() == 1


def test_jump_mask_first_row_is_never_a_jump():
    offsets = np.zeros((4, 4), dtype=np.float32)
    offsets[:, PEN_UP] = 1.0
    assert not jump_mask(offsets)[0]


def test_jump_mask_handles_short_input():
    for n in (0, 1):
        assert jump_mask(np.zeros((n, 4), dtype=np.float32)).sum() == 0


def test_jump_mask_count_matches_stroke_count():
    """n strokes means n-1 jumps between them."""
    offsets = make_offsets(n_strokes=7, points_per_stroke=20)
    assert jump_mask(offsets).sum() == 6


# ── token ranges ──────────────────────────────────────────────────────────────


def test_move_and_jump_tokens_occupy_disjoint_ranges(fitted):
    offsets = make_offsets(seed=99)
    tokens = fitted.encode(offsets)
    is_jump = jump_mask(offsets)

    for channel in fitted.channels:
        n_move = fitted.bins[channel].n_move
        ids = tokens[channel]
        assert np.all(ids[~is_jump] < n_move), f"{channel}: move token in jump range"
        assert np.all(ids[is_jump] >= n_move), f"{channel}: jump token in move range"


def test_tokens_stay_inside_vocabulary(fitted):
    """Out-of-range ids would crash nn.Embedding on the first batch."""
    offsets = make_offsets(seed=7)
    tokens = fitted.encode(offsets)
    for channel in fitted.channels:
        assert tokens[channel].min() >= 0
        assert tokens[channel].max() < fitted.bins[channel].vocab_size
    assert set(np.unique(tokens["pen"])) <= {0, 1}


def test_extreme_values_clip_into_outermost_bins(fitted):
    """Unseen extremes must clip, not overflow the vocabulary.

    The outermost bins are deliberately unbounded so inference-time outliers
    have somewhere to go.
    """
    offsets = make_offsets(seed=1)
    offsets[5, X] = 1e6
    offsets[6, Y] = -1e6

    tokens = fitted.encode(offsets)
    for channel in ("dx", "dy"):
        assert tokens[channel].max() < fitted.bins[channel].vocab_size


def test_vocab_sizes_reports_every_head(fitted):
    sizes = fitted.vocab_sizes
    assert set(sizes) == set(fitted.channels) | {"pen"}
    assert sizes["pen"] == 2
    for channel in fitted.channels:
        assert sizes[channel] == fitted.bins[channel].vocab_size


# ── round trip ────────────────────────────────────────────────────────────────


def test_decode_recovers_pen_flags_exactly(fitted):
    """pen_up is binary, so it must round-trip with zero loss."""
    offsets = make_offsets(seed=3)
    decoded = fitted.decode(fitted.encode(offsets))
    assert np.array_equal(decoded[:, PEN_UP], (offsets[:, PEN_UP] > 0.5).astype(np.float32))


def test_decode_preserves_length(fitted):
    offsets = make_offsets(seed=4)
    assert fitted.decode(fitted.encode(offsets)).shape == offsets.shape


def test_roundtrip_is_approximately_correct(fitted):
    """Lossy by design, but each step should land in the right neighbourhood."""
    offsets = make_offsets(seed=5)
    decoded = fitted.decode(fitted.encode(offsets))
    for column in (X, Y):
        spread = offsets[:, column].std()
        assert np.abs(decoded[:, column] - offsets[:, column]).max() < spread


def test_two_mode_split_beats_single_mode_binning():
    """The empirical result that motivated the whole design.

    Simulates single-mode binning by fitting on data with the jump structure
    erased, then compares accumulated drift. The split should win clearly at
    equal total vocabulary.
    """
    corpus = [make_offsets(seed=s) for s in range(20)]
    held_out = make_offsets(seed=500)

    split = StrokeTokenizer.fit(corpus, n_bins=32, n_jump_bins=32)

    # Same total vocabulary (64), but all of it spent on one distribution:
    # flatten the pen_up flags so every row looks like a pen-down move.
    flattened = []
    for offsets in corpus:
        copy = offsets.copy()
        copy[:, PEN_UP] = 0.0
        flattened.append(copy)
    single = StrokeTokenizer.fit(flattened, n_bins=64, n_jump_bins=1)

    flat_held = held_out.copy()
    flat_held[:, PEN_UP] = 0.0

    split_drift = split.reconstruction_error(held_out)["drift_ratio"]
    single_drift = single.reconstruction_error(flat_held)["drift_ratio"]
    assert split_drift < single_drift


def test_reconstruction_error_reports_expected_keys(fitted):
    stats = fitted.reconstruction_error(make_offsets(seed=6))
    assert set(stats) == {"max_drift", "final_drift", "x_height", "drift_ratio"}
    assert stats["max_drift"] >= 0
    assert np.isfinite(stats["drift_ratio"])


def test_reconstruction_error_is_zero_for_a_perfect_fit():
    """Two distinct values, two bins per mode: quantisation must be exact."""
    offsets = np.zeros((8, 4), dtype=np.float32)
    offsets[:, X] = 1.0
    offsets[:, Y] = 0.0
    tokenizer = StrokeTokenizer.fit([offsets], n_bins=4, n_jump_bins=4, channels=("dx", "dy"))
    assert tokenizer.reconstruction_error(offsets)["max_drift"] < 1e-4


# ── channel selection ─────────────────────────────────────────────────────────


def test_can_fit_without_the_dt_channel():
    corpus = [make_offsets(seed=s) for s in range(5)]
    tokenizer = StrokeTokenizer.fit(corpus, n_bins=16, channels=("dx", "dy"))
    assert tokenizer.channels == ("dx", "dy")
    assert "dt" not in tokenizer.vocab_sizes


def test_decode_leaves_missing_channels_at_zero(fitted):
    """A model predicting only dx/dy/pen must still produce a drawable array."""
    offsets = make_offsets(seed=8)
    tokens = fitted.encode(offsets)
    tokens.pop("dt")

    decoded = fitted.decode(tokens)
    assert np.all(decoded[:, TIME] == 0.0)
    assert not np.all(decoded[:, X] == 0.0)


def test_fit_rejects_unknown_channel():
    with pytest.raises(ValueError, match="unknown channel"):
        StrokeTokenizer.fit([make_offsets()], channels=("dx", "nope"))


def test_fit_rejects_empty_corpus():
    with pytest.raises(ValueError, match="no non-empty"):
        StrokeTokenizer.fit([np.zeros((0, 4), dtype=np.float32)])


# ── persistence ───────────────────────────────────────────────────────────────


def test_save_load_roundtrip_produces_identical_tokens(fitted, tmp_path):
    """Bit-identical, not merely close.

    A tokenizer that reloads slightly differently silently invalidates every
    checkpoint trained against it.
    """
    path = tmp_path / "tokenizer.json"
    fitted.save(path)
    reloaded = StrokeTokenizer.load(path)

    assert reloaded.channels == fitted.channels
    assert reloaded.vocab_sizes == fitted.vocab_sizes

    offsets = make_offsets(seed=123)
    original_tokens = fitted.encode(offsets)
    reloaded_tokens = reloaded.encode(offsets)
    for channel in original_tokens:
        assert np.array_equal(original_tokens[channel], reloaded_tokens[channel])


def test_fit_is_deterministic():
    """Same corpus must give the same bins, or checkpoints stop matching."""
    corpus = [make_offsets(seed=s) for s in range(8)]
    a = StrokeTokenizer.fit(corpus, n_bins=32)
    b = StrokeTokenizer.fit(corpus, n_bins=32)
    for channel in a.channels:
        assert np.array_equal(
            a.bins[channel].representatives, b.bins[channel].representatives
        )


def test_describe_mentions_both_modes(fitted):
    text = fitted.describe()
    assert "move" in text and "jump" in text
    for channel in fitted.channels:
        assert channel in text


def test_channel_bins_rejects_inconsistent_lengths():
    from hws.tokenizer import ChannelBins

    with pytest.raises(ValueError, match="representatives"):
        StrokeTokenizer(
            bins={
                "dx": ChannelBins(
                    move_edges=np.array([0.0, 1.0], dtype=np.float32),
                    jump_edges=np.array([5.0], dtype=np.float32),
                    # Needs (2+1) + (1+1) = 5 representatives; give 4.
                    representatives=np.zeros(4, dtype=np.float32),
                    n_move=3,
                )
            }
        )
