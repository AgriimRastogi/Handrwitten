"""Tests for the invariants that break silently.

    pytest -q

These deliberately do NOT test "does it look like handwriting" — that is what
scripts/inspect_data.py and your eyes are for. What they cover is the class of
bug that produces plausible-looking output with wrong content: a round-trip
that loses a point, a pen_up flag in the wrong place, a resampler that quietly
truncates a long line.

Those are the bugs worth automating, because you cannot see them.
"""

from __future__ import annotations

import numpy as np
import pytest

from hws import preprocess as pp
from hws import render
from hws.alphabet import Alphabet, build_alphabet
from hws.iam import PEN_UP, TIME, X, Y


def make_line(n_strokes: int = 3, points_per_stroke: int = 20, seed: int = 0) -> np.ndarray:
    """Synthesise a plausible (N, 4) line: several left-to-right strokes."""
    rng = np.random.default_rng(seed)
    strokes = []
    x_cursor, t_cursor = 0.0, 0.0
    for _ in range(n_strokes):
        xs = x_cursor + np.cumsum(rng.uniform(0.5, 2.0, points_per_stroke))
        ys = np.sin(np.linspace(0, 3, points_per_stroke)) * 5 + rng.normal(0, 0.1, points_per_stroke)
        ts = t_cursor + np.cumsum(rng.uniform(0.01, 0.03, points_per_stroke))

        stroke = np.zeros((points_per_stroke, 4), dtype=np.float32)
        stroke[:, X] = xs
        stroke[:, Y] = ys
        stroke[:, TIME] = ts
        stroke[-1, PEN_UP] = 1.0
        strokes.append(stroke)

        x_cursor = xs[-1] + 3.0  # inter-word gap
        t_cursor = ts[-1] + 0.2
    return np.concatenate(strokes, axis=0)


# ── stroke splitting ──────────────────────────────────────────────────────────


def test_split_strokes_recovers_stroke_count():
    line = make_line(n_strokes=4)
    assert len(pp.split_strokes(line)) == 4


def test_split_join_roundtrip_is_exact():
    line = make_line(n_strokes=5)
    assert np.allclose(pp.join_strokes(pp.split_strokes(line)), line)


def test_join_strokes_enforces_pen_up_invariant():
    """Exactly one pen_up per stroke, always on the final point.

    If this drifts, two strokes merge into one and the renderer draws a line
    through the air between them — visible, but only if you happen to look at
    that sample.
    """
    line = make_line(n_strokes=3)
    line[:, PEN_UP] = 0.0  # wipe every flag
    fixed = pp.join_strokes(pp.split_strokes(line))
    # With no flags there is one "stroke", so one terminator at the very end.
    assert fixed[:, PEN_UP].sum() == 1
    assert fixed[-1, PEN_UP] == 1.0


def test_split_strokes_handles_empty():
    assert pp.split_strokes(np.zeros((0, 4), dtype=np.float32)) == []


# ── offsets round-trip ────────────────────────────────────────────────────────


def test_offsets_roundtrip_recovers_coordinates():
    """The single most important invariant in the codebase.

    Model output is offsets; everything drawable is coordinates. If this pair
    is lossy, every sample you ever render is subtly wrong and the loss curve
    will look perfectly healthy.
    """
    line = make_line(n_strokes=4)
    # coords_to_offsets is relative, so absolute position is not recoverable.
    # Zero the origin to compare shape rather than placement.
    line[:, [X, Y]] -= line[0, [X, Y]]
    line[:, TIME] -= line[0, TIME]

    recovered = pp.offsets_to_coords(pp.coords_to_offsets(line))
    assert np.allclose(recovered[:, [X, Y]], line[:, [X, Y]], atol=1e-4)
    assert np.allclose(recovered[:, TIME], line[:, TIME], atol=1e-4)
    assert np.array_equal(recovered[:, PEN_UP], line[:, PEN_UP])


def test_offsets_preserve_length():
    """Same length in both representations, so indices line up between them."""
    line = make_line()
    assert pp.coords_to_offsets(line).shape == line.shape


def test_first_offset_is_zero():
    """No previous point to move from, so the first delta must be zero."""
    offsets = pp.coords_to_offsets(make_line())
    assert offsets[0, X] == 0.0
    assert offsets[0, Y] == 0.0
    assert offsets[0, TIME] == 0.0


# ── slant removal ─────────────────────────────────────────────────────────────


def test_remove_slant_flattens_a_known_slope():
    """Impose a 20-degree slope, then check it comes back out."""
    n = 200
    line = np.zeros((n, 4), dtype=np.float32)
    line[:, X] = np.linspace(0, 100, n)
    line[:, Y] = np.tan(np.deg2rad(20)) * line[:, X]  # pure slope, no content
    line[:, TIME] = np.linspace(0, 1, n)
    line[-1, PEN_UP] = 1.0

    flattened = pp.remove_slant(line)
    design = np.stack([np.ones(n), flattened[:, X]], axis=1)
    residual_slope = np.linalg.lstsq(design, flattened[:, Y], rcond=None)[0][1]
    assert abs(residual_slope) < 1e-3


def test_remove_slant_handles_degenerate_input():
    for line in (np.zeros((0, 4), np.float32), np.zeros((1, 4), np.float32)):
        assert pp.remove_slant(line).shape == line.shape


# ── resampling ────────────────────────────────────────────────────────────────


def test_resample_by_step_preserves_stroke_count():
    """Resampling must never drop or merge a stroke.

    A lost stroke is a lost dot on an 'i' or a lost full stop — invisible in
    aggregate stats.
    """
    line = make_line(n_strokes=6, points_per_stroke=30)
    out = pp.resample_by_step(line, step_factor=2.0, max_points=None)
    assert len(pp.split_strokes(out)) == 6


def test_resample_by_step_actually_reduces_points():
    line = make_line(n_strokes=4, points_per_stroke=50)
    out = pp.resample_by_step(line, step_factor=2.0, max_points=None)
    # ~2x fewer, allowing for the +1-per-stroke endpoint cost.
    assert out.shape[0] < line.shape[0] * 0.75


def test_resample_by_step_respects_max_points():
    """The cap must hold by widening the step, never by truncating.

    Regression test: the first version scaled a predicted count and ignored the
    per-stroke endpoint cost, overshooting a 600-point cap by 32 points.
    """
    line = make_line(n_strokes=20, points_per_stroke=60)
    for cap in (80, 150, 300):
        out = pp.resample_by_step(line, step_factor=1.0, max_points=cap)
        assert out.shape[0] <= cap, f"cap {cap} exceeded: got {out.shape[0]}"
        # And nothing was cut off the end: stroke count survives.
        assert len(pp.split_strokes(out)) == 20


def test_resample_arclength_hits_target_approximately():
    line = make_line(n_strokes=5, points_per_stroke=40)
    out = pp.resample_arclength(line, n_points=100)
    assert 90 <= out.shape[0] <= 115


def test_resample_keeps_dots_as_single_points():
    """A zero-length stroke is a dot, not an error. It must survive."""
    dot = np.zeros((1, 4), dtype=np.float32)
    dot[0, PEN_UP] = 1.0
    line = np.concatenate([make_line(n_strokes=1), dot], axis=0)

    out = pp.resample_by_step(line, step_factor=2.0, max_points=None)
    assert len(pp.split_strokes(out)) == 2


def test_resample_preserves_time_ordering():
    """dt must never go negative, or velocity becomes nonsense.

    Checks the WHOLE line, not stroke by stroke. An earlier version of this
    test looped over strokes, which meant it could not see a violation at a
    stroke *boundary* — and boundary violations are exactly what occurs in the
    ~0.25% of IAM files with corrupt timestamps.
    """
    line = make_line(n_strokes=4, points_per_stroke=40)
    out = pp.resample_by_step(line, step_factor=2.0)
    assert np.all(np.diff(out[:, TIME]) >= -1e-6)


# ── scale normalisation ───────────────────────────────────────────────────────


def test_normalize_scale_sets_median_step():
    offsets = pp.coords_to_offsets(make_line(n_strokes=4, points_per_stroke=40))
    scaled, factor = pp.normalize_scale(offsets, target_median=1.0)

    lengths = np.linalg.norm(scaled[:, [X, Y]], axis=1)
    assert np.isclose(np.median(lengths[lengths > 0]), 1.0, atol=0.01)
    assert factor > 0


def test_normalize_scale_leaves_time_alone():
    """dt stays in real seconds; its physical meaning is the whole point."""
    offsets = pp.coords_to_offsets(make_line())
    scaled, _ = pp.normalize_scale(offsets)
    assert np.allclose(scaled[:, TIME], offsets[:, TIME])


# ── full pipeline ─────────────────────────────────────────────────────────────


def test_preprocess_line_runs_all_modes():
    line = make_line(n_strokes=5, points_per_stroke=40)
    for mode in ("step", "count", "none"):
        out = pp.preprocess_line(line, resample=mode)
        assert out.shape[1] == 4
        assert out.shape[0] > 0
        assert np.isfinite(out).all(), f"{mode} produced NaN or inf"


def test_preprocess_line_rejects_bad_mode():
    with pytest.raises(ValueError, match="resample must be"):
        pp.preprocess_line(make_line(), resample="bogus")


# ── alphabet ──────────────────────────────────────────────────────────────────


def test_alphabet_roundtrip():
    alpha, _ = build_alphabet(["hello world", "ABC xyz"])
    text = "hello world"
    assert alpha.decode(alpha.encode(text)) == text


def test_alphabet_never_emits_pad_index():
    """PAD is 0. A real character mapping to 0 is the 2013 model's silent bug:
    out-of-vocabulary characters became indistinguishable from padding."""
    alpha, _ = build_alphabet(["abcXYZ 123"])
    assert 0 not in alpha.encode("abcXYZ 123")


def test_alphabet_raises_on_unknown_character():
    alpha, _ = build_alphabet(["abc"])
    with pytest.raises(KeyError, match="not in the alphabet"):
        alpha.encode("abZ")


def test_alphabet_can_drop_unknown_characters():
    alpha, _ = build_alphabet(["abc"])
    assert alpha.decode(alpha.encode("aZbZc", errors="drop")) == "abc"


def test_alphabet_vocab_size_accounts_for_pad():
    alpha, _ = build_alphabet(["abc"])
    assert len(alpha) == 3
    assert alpha.vocab_size == 4  # + PAD


def test_alphabet_min_count_filters():
    alpha, counts = build_alphabet(["aaab"], min_count=2)
    assert "a" in alpha
    assert "b" not in alpha
    assert counts["b"] == 1  # reported even though dropped


def test_alphabet_is_deterministic():
    """Sorted by codepoint, so the same corpus always gives the same mapping.
    Otherwise old checkpoints silently mismatch their tokenizer."""
    a1, _ = build_alphabet(["zyx", "abc"])
    a2, _ = build_alphabet(["abc", "zyx"])
    assert a1.chars == a2.chars


def test_alphabet_save_load(tmp_path):
    alpha, _ = build_alphabet(["hello world 123"])
    path = tmp_path / "alphabet.json"
    alpha.save(path)
    assert Alphabet.load(path).chars == alpha.chars


def test_alphabet_rejects_duplicates():
    with pytest.raises(ValueError, match="duplicate"):
        Alphabet(chars=("a", "b", "a"))


# ── rendering ─────────────────────────────────────────────────────────────────


def test_path_data_starts_each_stroke_with_moveto():
    """One M per stroke, the rest L. An extra M splits a letter; a missing one
    connects two that should be separate."""
    line = make_line(n_strokes=4)
    d = render.path_data(line)
    assert d.count("M") == 4
    assert d.count("L") == line.shape[0] - 4


def test_path_data_flips_y_for_svg():
    """Our +y is up, SVG's is down. Getting this wrong mirrors the output."""
    line = make_line(n_strokes=1)
    flipped = render.path_data(line, y_flip=True)
    unflipped = render.path_data(line, y_flip=False)
    assert flipped != unflipped

    first_y_flipped = float(flipped.split()[0].split(",")[1])
    first_y_plain = float(unflipped.split()[0].split(",")[1])
    assert np.isclose(first_y_flipped, -first_y_plain, atol=1e-6)


def test_path_data_handles_empty():
    assert render.path_data(np.zeros((0, 4), dtype=np.float32)) == ""


def test_render_line_viewbox_contains_all_ink():
    """Regression guard against a hardcoded viewBox clipping the content."""
    from xml.etree import ElementTree

    line = make_line(n_strokes=3)
    root = ElementTree.fromstring(render.render_line(line, padding=5.0))
    min_x, min_y, width, height = (float(v) for v in root.attrib["viewBox"].split())

    bb_min_x, bb_min_y, bb_max_x, bb_max_y = render.bounding_box(line)
    assert min_x <= bb_min_x
    assert min_x + width >= bb_max_x
    # y is negated on the way into SVG space.
    assert min_y <= -bb_max_y
    assert min_y + height >= -bb_min_y


def test_render_page_is_valid_xml_and_places_every_line():
    from xml.etree import ElementTree

    lines = [make_line(n_strokes=2, seed=i) for i in range(5)]
    root = ElementTree.fromstring(render.render_page(lines, seed=0))
    groups = root.findall("{http://www.w3.org/2000/svg}g")
    assert len(groups) == 5


def test_render_page_is_reproducible_given_a_seed():
    lines = [make_line(n_strokes=2, seed=i) for i in range(3)]
    assert render.render_page(lines, seed=42) == render.render_page(lines, seed=42)
    assert render.render_page(lines, seed=1) != render.render_page(lines, seed=2)


def test_speeds_are_finite_and_positive():
    """Zero-dt points must not produce inf, which would poison stroke widths."""
    line = make_line(n_strokes=3)
    line[5, TIME] = line[4, TIME]  # duplicate timestamp
    s = render.speeds(line)
    assert s.shape[0] == line.shape[0]
    assert np.isfinite(s).all()
    assert (s >= 0).all()


def test_variable_width_emits_one_path_per_segment():
    line = make_line(n_strokes=3, points_per_stroke=10)
    svg = render.render_line(line, style=render.InkStyle(variable_width=True))
    # 3 strokes x 10 points = 30 points -> 30 - 3 = 27 segments.
    assert svg.count("<path") == 27
