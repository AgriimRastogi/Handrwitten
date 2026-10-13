"""Geometric preprocessing: raw pen coordinates -> model-ready arrays.

Everything here is pure NumPy and stateless. No file I/O, no knowledge of
IAM. That makes each function individually testable, which matters because
preprocessing bugs are the hardest kind to notice: the model still trains,
the loss still falls, and the output is just subtly wrong forever.

THE PIPELINE, AND WHY EACH STEP EXISTS
--------------------------------------
    raw sensor points
      |
      1. remove_slant     writers hold the page at an angle; that angle is
      |                   not information about *what* was written, so
      |                   removing it means the model does not have to spend
      |                   capacity modelling it.
      |
      2. denoise          the whiteboard sensor jitters by a pixel or two per
      |                   sample. High-frequency noise the model would
      |                   otherwise dutifully learn to reproduce.
      |
      3. resample         cut ~1200 points down to ~500 at even spatial
      |                   intervals. See the note below — this is the step
      |                   the 2013 implementation gets backwards.
      |
      4. to_offsets       absolute position -> per-step deltas. A model
      |                   predicting absolute coordinates would have to learn
      |                   "where on the page am I", which is layout, not
      |                   handwriting. Deltas are translation-invariant.
      |
      5. normalize_scale  divide by the median step length so a typical step
      |                   is ~1 unit. Keeps activations in a sane range and
      v                   makes the loss scale independent of sensor DPI.
    model input

ON RESAMPLING (the important one)
---------------------------------
IAM samples at a fixed *time* interval, so slow careful strokes get many
points and fast strokes get few. That is backwards for a generative model:
sequence length ends up encoding writing speed rather than shape, and the
model wastes its context window on the slow bits.

Resampling at even *arc length* instead gives every millimetre of ink the
same number of points. Sequences get shorter (~500 vs ~1200), which means
faster training, and attention that spans the whole line instead of a third
of it.

The 2013 reference implementation has an ``interpolate()`` that *up*samples
by a factor of 2 — moving in exactly the wrong direction. Worth knowing if
you compare code.

Note we interpolate the ``time`` channel too, so pen velocity survives
resampling. Velocity is what makes variable stroke width possible in the
renderer, and it is data the 2013 model throws away entirely.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from hws.iam import PEN_UP, TIME, X, Y

# Offset arrays keep the same column *order* as coordinate arrays, so the
# X/Y/PEN_UP/TIME constants still index the right thing. The meaning shifts
# from absolute to relative: [dx, dy, pen_up, dt].
DX, DY, DT = X, Y, TIME


# ── stroke splitting ──────────────────────────────────────────────────────────


def split_strokes(points: np.ndarray) -> list[np.ndarray]:
    """Split an ``(N, 4)`` point array into a list of individual strokes.

    A stroke is a run of points between pen-lifts. Since ``pen_up`` marks the
    *last* point of each stroke, the split indices are one past each flag.

    Almost every function here operates per-stroke rather than on the whole
    line, because smoothing or interpolating *across* a pen-lift would draw
    a line through the air between two disconnected marks.
    """
    if points.shape[0] == 0:
        return []

    # +1 so the pen_up point stays at the END of its own stroke.
    cut_at = np.where(points[:, PEN_UP] == 1.0)[0] + 1
    strokes = np.split(points, cut_at, axis=0)

    # np.split leaves a trailing empty array when the last point is pen_up
    # (which is the normal case), so filter those out.
    return [s for s in strokes if s.shape[0] > 0]


def join_strokes(strokes: list[np.ndarray]) -> np.ndarray:
    """Inverse of :func:`split_strokes`, re-asserting the pen_up invariant.

    Rebuilding the flags rather than trusting the inputs means a stroke that
    lost its terminator during processing gets fixed here instead of causing
    two strokes to silently merge into one later.
    """
    if not strokes:
        return np.zeros((0, 4), dtype=np.float32)

    fixed = []
    for stroke in strokes:
        stroke = stroke.copy()
        stroke[:, PEN_UP] = 0.0
        stroke[-1, PEN_UP] = 1.0
        fixed.append(stroke)

    return np.concatenate(fixed, axis=0).astype(np.float32)


# ── step 1: slant removal ─────────────────────────────────────────────────────


def remove_slant(points: np.ndarray) -> np.ndarray:
    """Rotate the line so its baseline is horizontal.

    Fits ``y = a + b*x`` across every point by least squares, treats
    ``theta = arctan(b)`` as the page tilt, and rotates by ``-theta``.

    This estimates the tilt from the *entire* line, so it corrects global
    page rotation but leaves per-character italic slant intact — which is
    what you want, since italic slant is part of a person's handwriting
    style and the model should learn it.

    Uses ``lstsq`` rather than explicitly inverting ``XᵀX``. Same answer,
    but numerically stable when x-values are nearly collinear (a very short
    line, for instance), where the explicit inverse can blow up.
    """
    if points.shape[0] < 2:
        return points.copy()

    out = points.copy()
    x = out[:, X]
    y = out[:, Y]

    # Design matrix [1, x] -> solving for [intercept, slope].
    design = np.stack([np.ones_like(x), x], axis=1)
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]

    theta = np.arctan(slope)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    # Right-multiplying row vectors by this matrix rotates by -theta:
    #   x' =  x·cos + y·sin
    #   y' = -x·sin + y·cos
    rotation = np.array([[cos_t, -sin_t], [sin_t, cos_t]], dtype=np.float32)
    out[:, [X, Y]] = out[:, [X, Y]] @ rotation

    # Centre vertically on the fitted line so y=0 is roughly the baseline.
    # A proper baseline estimate would use the bottom of the x-height band
    # (ignoring descenders like 'g' and 'p'); this is the cheap version and
    # is fine because normalize_scale removes the overall scale anyway.
    out[:, Y] -= intercept * cos_t

    return out


# ── step 2: denoising ─────────────────────────────────────────────────────────


def denoise(points: np.ndarray, window: int = 7, polyorder: int = 3) -> np.ndarray:
    """Smooth sensor jitter with a Savitzky-Golay filter, stroke by stroke.

    Savitzky-Golay fits a low-order polynomial to a sliding window, which
    preserves the sharp corners and cusps that make handwriting legible —
    a plain moving average would round them off and turn everything mushy.

    Parameters
    ----------
    window:
        Points per window. Must be odd; forced odd if you pass an even number.
        Larger = smoother, at the cost of detail.
    polyorder:
        Degree of the fitted polynomial. Must be < window.

    Strokes shorter than ``window`` are passed through untouched. That is not
    laziness: SciPy would either raise, or with ``mode="nearest"`` invent
    values from padding, and a 3-point stroke (the dot on an 'i') has no
    high-frequency content to remove anyway.
    """
    if window % 2 == 0:
        window += 1
    if polyorder >= window:
        raise ValueError(f"polyorder ({polyorder}) must be < window ({window})")

    smoothed = []
    for stroke in split_strokes(points):
        if stroke.shape[0] < window:
            smoothed.append(stroke)
            continue

        out = stroke.copy()
        # Only x and y. Smoothing pen_up would turn a binary flag into
        # fractional nonsense; smoothing time could make it non-monotonic,
        # which would produce negative dt and thus negative velocity.
        out[:, X] = savgol_filter(stroke[:, X], window, polyorder, mode="nearest")
        out[:, Y] = savgol_filter(stroke[:, Y], window, polyorder, mode="nearest")
        smoothed.append(out)

    return join_strokes(smoothed)


# ── step 3: arc-length resampling ─────────────────────────────────────────────


def _arc_length(stroke: np.ndarray) -> np.ndarray:
    """Cumulative distance travelled along a stroke. Shape ``(len(stroke),)``.

    ``result[0]`` is 0; ``result[-1]`` is the total length.
    """
    deltas = np.diff(stroke[:, [X, Y]], axis=0)
    step_lengths = np.linalg.norm(deltas, axis=1)
    return np.concatenate([[0.0], np.cumsum(step_lengths)])


def resample_arclength(points: np.ndarray, n_points: int = 500) -> np.ndarray:
    """Resample to ``≈n_points`` total, spaced evenly by distance travelled.

    **Fixed-count mode.** The budget is divided between strokes in proportion
    to their length, so a long stroke gets more points than a short one and
    pen-lift structure is preserved exactly (no stroke is ever dropped).

    ``≈`` because each stroke needs at least 2 points and rounding is
    per-stroke, so you land within a few percent of the target.

    TRADE-OFF YOU SHOULD UNDERSTAND BEFORE PICKING THIS
    ---------------------------------------------------
    A fixed count gives uniform sequence length, which is convenient. But it
    means *spatial resolution varies per line*: a physically short line and a
    long one both get ``n_points``, so the long one has points spaced further
    apart. The model then sees "one step" meaning different distances in
    different samples, which is exactly the inconsistency arc-length
    resampling was supposed to remove.

    Measured on this corpus (see scripts/inspect_data.py):

        raw points per line: min 445 | median 630 | p95 860 | max ~880

    So ``n_points=500`` compresses the median by only 1.26x and *upsamples*
    the shortest lines from 445 to 500 — moving in the wrong direction. If
    you want real compression, use ``n_points≈350``, or better, use
    :func:`resample_by_step` and get consistent spacing instead.

    Kept because uniform length genuinely simplifies some experiments, and
    because it is the honest baseline to compare the step-based version against.
    """
    strokes = split_strokes(points)
    if not strokes:
        return np.zeros((0, 4), dtype=np.float32)

    arc_lengths = [_arc_length(s) for s in strokes]
    totals = np.array([a[-1] for a in arc_lengths])
    grand_total = totals.sum()

    # Degenerate line: every stroke is a single point (or all points
    # coincide). Nothing to resample.
    if grand_total <= 0:
        return join_strokes([s[:1] for s in strokes])

    resampled = []
    for stroke, arc, total in zip(strokes, arc_lengths, totals):
        # A zero-length stroke is a dot — genuine handwriting (the dot on an
        # 'i', a full stop). Keep exactly one point; interpolating it would
        # divide by zero.
        if total <= 0 or stroke.shape[0] < 2:
            resampled.append(stroke[:1])
            continue

        n_i = max(2, int(round(n_points * total / grand_total)))
        resampled.append(_interp_stroke(stroke, arc, total, n_i))

    return join_strokes(resampled)


def median_spacing(points: np.ndarray) -> float:
    """Median distance between consecutive points, ignoring pen-up jumps and dupes.

    This is the natural unit of a line's sampling density. Measured on
    IAM-OnDB it is ~20 raw sensor units, but it varies from ~13 to ~28 between
    writers (different pen speeds, different sensor calibration), which is
    exactly why :func:`resample_by_step` expresses its step relative to this
    rather than as an absolute distance.
    """
    strokes = split_strokes(points)
    all_steps = []
    for stroke in strokes:
        if stroke.shape[0] < 2:
            continue
        # Per-stroke, so the big pen-up jumps between words never enter the
        # measurement — those are not sampling density, they are gaps.
        d = np.linalg.norm(np.diff(stroke[:, [X, Y]], axis=0), axis=1)
        all_steps.append(d[d > 0])

    if not all_steps:
        return 0.0
    combined = np.concatenate(all_steps)
    return float(np.median(combined)) if combined.size else 0.0


def resample_by_step(
    points: np.ndarray,
    step_factor: float = 2.0,
    max_points: int | None = 600,
) -> np.ndarray:
    """Resample to a consistent spatial resolution. **Recommended mode.**

    **Fixed-spacing mode.** Output length varies with how much ink the line
    contains, which is the point: every step means the same physical pen
    movement, so the model learns "how far does the pen travel per step" once
    rather than once per line length.

    You need padding for batching anyway (text lengths vary), so variable
    sequence length costs nothing you were not already paying.

    Parameters
    ----------
    step_factor:
        Target spacing as a **multiple of this line's own median spacing**
        (see :func:`median_spacing`). ``2.0`` roughly halves the point count,
        ``3.0`` roughly thirds it, ``1.0`` leaves density unchanged.

        Relative rather than absolute because the corpus is not uniform: raw
        median spacing ranges ~13-28 sensor units across writers, so any fixed
        absolute step would downsample some lines and *upsample* others. Made
        that exact mistake writing this function — an absolute ``step=2.0`` in
        sensor units inflated 580-point lines to 625 points.
    max_points:
        Hard cap, enforced by widening the step for that line rather than
        truncating. Truncation would silently cut words off the end of long
        lines: invisible in aggregate statistics, and obvious only if you
        happen to render the affected sample. ``None`` disables the cap.

    Returns
    -------
    ``(M, 4)`` coordinates, where M is close to
    ``len(points) / step_factor`` but never above ``max_points``.
    """
    strokes = split_strokes(points)
    if not strokes:
        return np.zeros((0, 4), dtype=np.float32)

    arc_lengths = [_arc_length(s) for s in strokes]
    totals = np.array([a[-1] for a in arc_lengths])
    grand_total = float(totals.sum())

    if grand_total <= 0:
        return join_strokes([s[:1] for s in strokes])

    spacing = median_spacing(points)
    if spacing <= 0:
        return join_strokes([s[:1] for s in strokes])

    target_step = step_factor * spacing

    def points_per_stroke(step: float) -> list[int]:
        """How many output points each stroke gets at this step size."""
        return [
            1 if (t <= 0 or s.shape[0] < 2) else max(2, int(round(float(t) / step)) + 1)
            for s, t in zip(strokes, totals)
        ]

    if max_points is not None:
        # A stroke of arc length L resampled at step s yields L/s + 1 points
        # (both endpoints included), so summing over strokes:
        #
        #     total_points = grand_total / s + n_strokes
        #
        # Solving total_points <= max_points gives the minimum step honouring
        # the cap. Deriving this algebraically rather than scaling a prediction
        # matters: the per-stroke "+1" is ~28 points on a typical IAM line, so
        # ignoring it overshoots by that much.
        n_strokes = len(strokes)
        budget = max_points - n_strokes

        if budget < n_strokes:
            # Cannot honour the cap: every stroke needs at least 2 points.
            # Emit the minimum and let the caller notice the length.
            return join_strokes(
                [
                    _interp_stroke(s, a, float(t), 2) if t > 0 and s.shape[0] >= 2 else s[:1]
                    for s, a, t in zip(strokes, arc_lengths, totals)
                ]
            )

        target_step = max(target_step, grand_total / budget)

        # The algebra above is exact for real-valued point counts, but each
        # stroke's count is rounded, and ~28 roundings of up to +0.5 each can
        # still push a few points over. (Measured: a 600 cap produced 601-606.)
        # Widen the step by the observed overshoot ratio until it fits — this
        # converges in one or two passes.
        for _ in range(8):
            total = sum(points_per_stroke(target_step))
            if total <= max_points:
                break
            target_step *= max(total / max_points, 1.0 + 1e-3)

    counts = points_per_stroke(target_step)
    resampled = [
        _interp_stroke(s, a, float(t), n) if n >= 2 else s[:1]
        for s, a, t, n in zip(strokes, arc_lengths, totals, counts)
    ]

    return join_strokes(resampled)


def _interp_stroke(
    stroke: np.ndarray, arc: np.ndarray, total: float, n_points: int
) -> np.ndarray:
    """Linearly resample one stroke to ``n_points`` evenly spaced by arc length.

    Shared by both resampling modes so the interpolation itself is defined
    once. ``arc`` is the cumulative-distance array from :func:`_arc_length`.
    """
    # Sample evenly along the *distance* axis, then look up what x/y/time
    # were at each of those distances.
    targets = np.linspace(0.0, total, n_points)

    out = np.zeros((n_points, 4), dtype=np.float32)
    out[:, X] = np.interp(targets, arc, stroke[:, X])
    out[:, Y] = np.interp(targets, arc, stroke[:, Y])
    # Interpolating time is what keeps velocity information alive: points
    # that were far apart in time stay far apart, so dt still tells you how
    # fast the pen was moving. This is the channel the 2013 model discards.
    out[:, TIME] = np.interp(targets, arc, stroke[:, TIME])
    # pen_up is rebuilt by join_strokes; interpolating a binary flag would
    # produce values like 0.4.
    return out


# ── steps 4 & 5: offsets and scale ────────────────────────────────────────────


def coords_to_offsets(points: np.ndarray) -> np.ndarray:
    """Absolute coordinates -> per-step deltas. ``(N, 4)`` -> ``(N, 4)``.

    Output columns are ``[dx, dy, pen_up, dt]``. Row *i* means "to get from
    point i-1 to point i, move by (dx, dy) over dt seconds; afterwards the
    pen is up iff pen_up".

    The first row is ``[0, 0, pen_up₀, 0]`` — there is no previous point to
    move from. Keeping it (rather than dropping it) means offsets and
    coordinates have the same length, so indices line up between the two
    representations and off-by-one bugs get much rarer.
    """
    if points.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    offsets = np.zeros_like(points, dtype=np.float32)
    offsets[1:, [X, Y]] = np.diff(points[:, [X, Y]], axis=0)
    offsets[1:, TIME] = np.diff(points[:, TIME])
    # pen_up is a property of the point, not of the movement, so it carries
    # over unchanged.
    offsets[:, PEN_UP] = points[:, PEN_UP]
    return offsets


def offsets_to_coords(offsets: np.ndarray) -> np.ndarray:
    """Inverse of :func:`coords_to_offsets` — cumulative sum of the deltas.

    This is the function that turns model output into something drawable, so
    it is on the critical path for every sample you ever look at. Round-trip
    it in a unit test.
    """
    if offsets.shape[0] == 0:
        return np.zeros((0, 4), dtype=np.float32)

    coords = np.zeros_like(offsets, dtype=np.float32)
    coords[:, [X, Y]] = np.cumsum(offsets[:, [X, Y]], axis=0)
    coords[:, TIME] = np.cumsum(offsets[:, TIME])
    coords[:, PEN_UP] = offsets[:, PEN_UP]
    return coords


def normalize_scale(offsets: np.ndarray, target_median: float = 1.0) -> tuple[np.ndarray, float]:
    """Scale dx/dy so the median step length is ``target_median``.

    Returns ``(scaled_offsets, scale_factor)``. Keep the factor if you want
    to undo this later; the renderer does not need it, since it fits output
    to a bounding box regardless.

    Median rather than mean because the mean is dragged around by the long
    pen-up jumps between words, which are not representative of normal
    writing motion.

    ``dt`` is deliberately left in real seconds. It is already in sane units
    and its physical meaning — velocity — is only useful un-normalised.
    """
    if offsets.shape[0] == 0:
        return offsets.copy(), 1.0

    step_lengths = np.linalg.norm(offsets[:, [X, Y]], axis=1)
    # Ignore zero-length steps (duplicate points) so they don't drag the
    # median toward zero.
    nonzero = step_lengths[step_lengths > 0]
    median = float(np.median(nonzero)) if nonzero.size else 0.0

    if median <= 0:
        return offsets.copy(), 1.0

    scale = target_median / median
    out = offsets.copy()
    out[:, [X, Y]] *= scale
    return out, scale


# ── the whole pipeline ────────────────────────────────────────────────────────


def preprocess_line(
    points: np.ndarray,
    resample: str = "step",
    step_factor: float = 2.0,
    n_points: int = 500,
    max_points: int | None = 600,
    denoise_window: int = 7,
    do_remove_slant: bool = True,
    do_denoise: bool = True,
) -> np.ndarray:
    """Run the full pipeline: raw sensor points -> normalised offsets.

    Every step has a switch so you can ablate them. Do actually run that
    ablation — render the same line with each step disabled in turn and look
    at the results. It takes ten minutes and it is the difference between
    knowing your pipeline works and hoping it does.

    Parameters
    ----------
    points:
        ``(N, 4)`` raw array from :func:`hws.iam.parse_stroke_xml`.
    resample:
        ``"step"``  — consistent spatial resolution via :func:`resample_by_step`.
                      Default, and the one to prefer.
        ``"count"`` — fixed length via :func:`resample_arclength`.
        ``"none"``  — keep the sensor's native sampling.
    step_factor, max_points:
        Passed to :func:`resample_by_step` when ``resample="step"``.
    n_points:
        Passed to :func:`resample_arclength` when ``resample="count"``.

    Returns
    -------
    ``(M, 4)`` offsets ``[dx, dy, pen_up, dt]``, ready for tokenisation.
    """
    if resample not in {"step", "count", "none"}:
        raise ValueError(f"resample must be 'step', 'count' or 'none', got {resample!r}")

    out = points
    if do_remove_slant:
        out = remove_slant(out)
    if do_denoise:
        out = denoise(out, window=denoise_window)

    # Resampling runs on raw-scale coordinates, before normalize_scale. That is
    # fine because step_factor is relative to the line's own median spacing, so
    # it is scale-invariant either way.
    if resample == "step":
        out = resample_by_step(out, step_factor=step_factor, max_points=max_points)
    elif resample == "count":
        out = resample_arclength(out, n_points=n_points)

    out = coords_to_offsets(out)
    out, _ = normalize_scale(out)
    return out


# ── training-time augmentation ────────────────────────────────────────────────
# These are NOT part of the deterministic pipeline above. They are applied
# randomly per-epoch during training to multiply the effective size of a
# corpus that only has ~13k lines. Keep them separate: augmentation belongs
# in the Dataset's __getitem__, not in data preparation, so that every epoch
# sees different perturbations rather than one frozen set.


def augment_skew(points: np.ndarray, degrees: float) -> np.ndarray:
    """Shear horizontally — simulates a different italic slant.

    Positive degrees lean right. IAM writers span roughly ±15°, so sampling
    in ``[-8, 8]`` on top of that stays plausible.
    """
    theta = np.deg2rad(degrees)
    out = points.copy()
    shear = np.array([[1.0, 0.0], [np.tan(theta), 1.0]], dtype=np.float32)
    out[:, [X, Y]] = out[:, [X, Y]] @ shear
    return out


def augment_stretch(points: np.ndarray, x_factor: float, y_factor: float) -> np.ndarray:
    """Scale the axes independently — simulates wider/narrower, taller/shorter hands.

    Try ``x_factor, y_factor ~ U(0.9, 1.1)`` independently. Anything more
    aggressive starts producing letter proportions no human would write,
    which teaches the model to generate them.
    """
    out = points.copy()
    out[:, X] *= x_factor
    out[:, Y] *= y_factor
    return out


def augment_jitter(points: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """Add Gaussian noise to every point.

    Deliberately applied *after* denoising, so this is controlled noise
    replacing uncontrolled sensor noise. Keep ``scale`` small (~0.02 of
    median step length); too much and you are training the model to be shaky.

    Note the first point is left alone, so the line's starting position
    stays put.
    """
    out = points.copy()
    out[1:, [X, Y]] += rng.normal(0.0, scale, size=out[1:, [X, Y]].shape)
    return out
