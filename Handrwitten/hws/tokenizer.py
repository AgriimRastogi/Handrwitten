"""Stroke discretisation: continuous pen offsets <-> integer tokens.

WHY DISCRETISE AT ALL
---------------------
Graves' 2013 model predicts a Mixture Density Network: 20 Gaussians per step,
trained by maximising likelihood. MDNs work, and they produce genuinely smooth
curves, but they are miserable to train — the loss involves a log-sum-exp over
mixture components and goes NaN when a component's variance collapses, which it
will. Debugging that as your first generative model is a bad time.

Binning the offsets and predicting a softmax over bins turns the whole thing
into classification. Cross-entropy cannot go NaN. You get a working model in
days instead of weeks, and you can add an MDN head later and compare — only
the model's output head would change, not this file.

QUANTILE BINS, NOT UNIFORM BINS
-------------------------------
Pen offsets are heavy-tailed. Almost every step is small (the pen tracing a
letter) but a few are large (jumping between words with the pen up). Uniform
bins would spend most of their resolution on empty space, leaving the dense
region — where all the shape information lives — crudely quantised. Quantile
bins put boundaries where the data actually is.

TWO BIN SETS: PEN-DOWN MOVES vs PEN-UP JUMPS
--------------------------------------------
This is the part that was arrived at empirically, and it matters.

A single quantile binning over all offsets fails in a specific, visible way.
Rendering round-trips at 256 bins showed inter-word spacing collapsing —
"hear this" became "hearthis", "behind the" became "behindthe" — and at 1024
bins one sample grew a stray stroke slashing through a word. Raising the count
to 4096 fixed it, but that is treating a symptom.

The cause: pen-up jumps are only ~8% of steps but are 5-20x longer than
pen-down steps. Quantile binning allocates bins by *density*, so those long
jumps all fall into a handful of very wide tail bins and decode to badly wrong
values. A mis-decoded pen-down step is a slightly lumpy letter; a mis-decoded
pen-up jump puts the next word in the wrong place.

So each channel gets **two** bin sets, fitted separately on pen-down and
pen-up steps, and the token vocabulary is their concatenation:

    token  0            .. n_move-1          -> a pen-down step
    token  n_move       .. n_move+n_jump-1   -> a pen-up jump

One output head, unambiguous tokens, and decoding needs no extra context. The
model learns on its own that a pen-up flag is followed by a jump-range token,
because that is what the training data does.

Result: 256+256 bins now beats a single set of 4096, at a sixteenth of the
vocabulary.

THE ERROR METRIC THAT MATTERS
-----------------------------
Per-step quantisation error is nearly meaningless. Offsets get integrated by
`cumsum` to recover positions, so errors **accumulate**. Measured on this
corpus the accumulation is linear, not sqrt(N) — halving the bin width halves
the drift — which means the errors are systematically signed rather than
independent.

:meth:`StrokeTokenizer.reconstruction_error` therefore reports drift in
*position* space after a full encode -> decode -> cumsum round trip, relative
to the line's own x-height. Calibrated by rendering: below ~0.05 is invisible,
~0.2 shows visible spacing damage, above ~0.4 is clearly broken. Check it
before you train.

Bin representatives are the **mean of the training values in each bin**, not
the midpoint. The mean is the L2-optimal representative and it matters most in
the wide tail bins.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from hws.iam import PEN_UP, TIME, X, Y

logger = logging.getLogger(__name__)

# Channel names in the order a model would predict them. A tuple so the
# ordering is fixed: if it changed, saved checkpoints would map heads to the
# wrong channels without any error.
CONTINUOUS_CHANNELS = ("dx", "dy", "dt")
_CHANNEL_COLUMN = {"dx": X, "dy": Y, "dt": TIME}


def jump_mask(offsets: np.ndarray) -> np.ndarray:
    """Boolean mask: which rows are pen-up jumps rather than pen-down moves?

    Recall the convention from :func:`hws.preprocess.coords_to_offsets`: row
    *i* is the movement *arriving at* point i, and ``pen_up[i]`` means the pen
    lifts *after* point i. So the airborne movement is the row **following** a
    pen_up flag, not the flagged row itself.

    Off-by-one here would fit the jump bins on pen-down data and vice versa,
    which is worse than not splitting at all.
    """
    mask = np.zeros(offsets.shape[0], dtype=bool)
    if offsets.shape[0] > 1:
        mask[1:] = offsets[:-1, PEN_UP] > 0.5
    return mask


def _representatives_for(
    values: np.ndarray, edges: np.ndarray, fallback: np.ndarray
) -> np.ndarray:
    """Mean of the training values in each bin; ``fallback[i]`` where a bin is empty.

    The mean is the L2-optimal representative for a bin. Empty bins are real
    with uniform binning (the sparse tail has gaps), and they need a sensible
    value because a freshly-initialised model can emit any token.
    """
    n_bins = len(edges) + 1
    indices = np.searchsorted(edges, values, side="right")

    counts = np.bincount(indices, minlength=n_bins)
    sums = np.bincount(indices, weights=values.astype(np.float64), minlength=n_bins)

    representatives = np.array(fallback, dtype=np.float32, copy=True)
    nonempty = counts > 0
    representatives[nonempty] = (sums[nonempty] / counts[nonempty]).astype(np.float32)
    return representatives


def _fit_bins_quantile(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Quantile bins: equal *mass* per bin.

    Right for the pen-down channel, whose values are tightly bounded — arc-length
    resampling makes every pen-down step nearly the same length by construction,
    so there is no heavy tail to worry about and equal-mass bins give excellent
    resolution exactly where the data is.

    Wrong for pen-up jumps; see :func:`_fit_bins_asinh`.

    The bin count can come out below ``n_bins``: if the data has a spike at one
    value (offsets contain many exact zeros), multiple quantiles land on the same
    number and those bins collapse. Deduplicating beats keeping empty bins.
    """
    quantiles = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    edges = np.unique(np.quantile(values, quantiles).astype(np.float32))

    # Bin midpoints as the empty-bin fallback; with quantile bins essentially
    # nothing is empty, so this is belt-and-braces.
    padded = np.concatenate([[values.min()], edges, [values.max()]])
    midpoints = ((padded[:-1] + padded[1:]) / 2).astype(np.float32)

    return edges, _representatives_for(values, edges, midpoints)


def _fit_bins_asinh(values: np.ndarray, n_bins: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform bins in ``asinh`` space: equal *ratio* per bin in the tail.

    This exists because quantile binning fails on pen-up jumps, and it fails in
    a way that is worth understanding.

    Jump magnitudes on this corpus run: median 3, p99 20, p99.9 37, max 77.
    Quantile bins allocate by mass, so the outermost bin holds the top 0.4% of
    jumps — an unbounded *range* — and decodes all of them to their mean.
    Measured consequence: a true jump of dx = -47.08 decoded to -17.56, a
    single-step error of 29.5 units that alone accounted for 85% of that line's
    total reconstruction drift. 97% of all error was in jump steps.

    Note that log/asinh-transforming before *quantile* binning changes nothing:
    quantiles are invariant under monotonic transforms, so the bin edges come
    out identical. The allocation strategy itself has to change.

    Uniform spacing in ``asinh`` space does that. ``asinh`` is linear near zero
    and logarithmic for large magnitudes, and unlike ``log`` it is defined at
    and across zero, so signed values need no special handling. Bins are dense
    where jumps are common and widen smoothly into the tail: at the extreme the
    bin is ~3 units wide instead of ~60, cutting worst-case error by ~20x.
    """
    transformed = np.arcsinh(values.astype(np.float64))
    lo, hi = float(transformed.min()), float(transformed.max())

    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        # Degenerate: all values identical. One bin, exact representative.
        return np.zeros(0, dtype=np.float32), np.array([values.mean()], dtype=np.float32)

    # Widen slightly so the observed extremes sit inside a bin rather than on
    # its boundary, and so a marginally larger value at inference still lands
    # in a narrow bin instead of clipping.
    span = hi - lo
    lo -= 0.02 * span
    hi += 0.02 * span

    boundaries = np.linspace(lo, hi, n_bins + 1)
    edges = np.sinh(boundaries[1:-1]).astype(np.float32)

    # Empty-bin fallback: the bin's midpoint mapped back from asinh space. Much
    # better than an edge value, and the sparse tail genuinely has empty bins.
    #
    # Clamped to the observed range because the 2% widening above pushes the
    # outermost midpoints past the real min/max. Unclamped, the dt channel got a
    # bin representative of -0.106 seconds -- a negative duration, which is
    # physically impossible and would decode to a negative pen velocity. The
    # model would rarely emit it (no training mass), but a representative should
    # never lie outside the data it represents.
    midpoints = np.clip(
        np.sinh((boundaries[:-1] + boundaries[1:]) / 2),
        values.min(),
        values.max(),
    ).astype(np.float32)

    # Deduplicate defensively: sinh of distinct values is distinct, but float32
    # rounding can collide for very close boundaries.
    unique_edges, keep = np.unique(edges, return_index=True)
    if len(unique_edges) != len(edges):
        edges = unique_edges
        midpoints = np.concatenate([midpoints[keep], midpoints[-1:]])[: len(edges) + 1]

    return edges, _representatives_for(values, edges, midpoints)


def _fit_bins(
    values: np.ndarray, n_bins: int, strategy: str = "quantile"
) -> tuple[np.ndarray, np.ndarray]:
    """Dispatch to a binning strategy.

    Returns ``(interior_edges, representatives)`` with
    ``len(representatives) == len(interior_edges) + 1``. Interior edges only:
    n bins need n-1 boundaries, and the outermost bins stay unbounded so
    extreme values at inference clip into them rather than falling outside the
    vocabulary.
    """
    if values.size == 0:
        raise ValueError("cannot fit bins on an empty array")
    if strategy == "quantile":
        return _fit_bins_quantile(values, n_bins)
    if strategy == "asinh":
        return _fit_bins_asinh(values, n_bins)
    raise ValueError(f"unknown binning strategy {strategy!r}")


def _fit_bins_or_trivial(
    values: np.ndarray, n_bins: int, strategy: str = "quantile"
) -> tuple[np.ndarray, np.ndarray]:
    """Like :func:`_fit_bins`, but degrades to a single zero bin when empty.

    A mode can legitimately have no samples: a corpus of single-stroke lines has
    no pen-up jumps at all. Rather than crash, emit one bin that decodes to zero
    displacement — "never saw a jump, so a jump means no movement".
    """
    if values.size == 0:
        return (
            np.zeros(0, dtype=np.float32),      # no interior edges -> 1 bin
            np.zeros(1, dtype=np.float32),      # which decodes to 0.0
        )
    return _fit_bins(values, n_bins, strategy)


@dataclass(frozen=True)
class ChannelBins:
    """Bin definitions for one continuous channel, both modes.

    ``representatives`` is the concatenation ``[move_reps, jump_reps]``, so a
    token index decodes with a single array lookup and no mode argument.
    """

    move_edges: np.ndarray
    jump_edges: np.ndarray
    representatives: np.ndarray
    n_move: int

    @property
    def n_jump(self) -> int:
        return len(self.representatives) - self.n_move

    @property
    def vocab_size(self) -> int:
        return len(self.representatives)


@dataclass(frozen=True)
class StrokeTokenizer:
    """Frozen mapping between continuous offsets and integer tokens.

    Like :class:`hws.alphabet.Alphabet`, fit this once on the training split
    and save it. Refitting after training invalidates every checkpoint
    silently — the model keeps emitting token 137, that token just means a
    different displacement now.

    Fit on the **training split only**; fitting over the whole corpus leaks the
    validation distribution into preprocessing.
    """

    bins: dict[str, ChannelBins]

    def __post_init__(self) -> None:
        """Validate that edges and representatives agree.

        n bins need n-1 interior edges, so representatives must total
        ``(len(move_edges) + 1) + (len(jump_edges) + 1)`` and ``n_move`` must be
        ``len(move_edges) + 1``. A mismatch means decode() indexes into the
        wrong mode's range — silently, producing plausible-looking output with
        wrong displacements.
        """
        for channel, cb in self.bins.items():
            expected_move = len(cb.move_edges) + 1
            expected_total = expected_move + len(cb.jump_edges) + 1

            if cb.n_move != expected_move:
                raise ValueError(
                    f"channel {channel!r}: n_move={cb.n_move} but "
                    f"{len(cb.move_edges)} move edges imply {expected_move}"
                )
            if len(cb.representatives) != expected_total:
                raise ValueError(
                    f"channel {channel!r}: {len(cb.representatives)} representatives "
                    f"but edges imply {expected_total}"
                )

    @property
    def channels(self) -> tuple[str, ...]:
        return tuple(c for c in CONTINUOUS_CHANNELS if c in self.bins)

    @property
    def vocab_sizes(self) -> dict[str, int]:
        """Head sizes for the model, including binary ``pen``.

        Pass this straight to the model constructor so heads are sized from the
        tokenizer rather than a hardcoded constant that can drift out of sync.
        """
        sizes = {c: self.bins[c].vocab_size for c in self.channels}
        sizes["pen"] = 2
        return sizes

    # ── fitting ───────────────────────────────────────────────────────────────

    @classmethod
    def fit(
        cls,
        offsets: Iterable[np.ndarray],
        n_bins: int = 256,
        n_jump_bins: int | None = None,
        channels: Sequence[str] = ("dx", "dy", "dt"),
        move_strategy: str = "quantile",
        jump_strategy: str = "asinh",
        max_samples: int = 4_000_000,
    ) -> "StrokeTokenizer":
        """Fit bins on a collection of offset arrays.

        Parameters
        ----------
        offsets:
            Preprocessed ``(N, 4)`` arrays from
            :func:`hws.preprocess.preprocess_line`. **Training split only.**
        n_bins:
            Bins for pen-down moves. 256 is comfortable: arc-length resampling
            makes pen-down steps nearly uniform in length, so this channel is
            easy — measured move error is ~3% of total reconstruction error.
        n_jump_bins:
            Bins for pen-up jumps. Defaults to ``n_bins``. Jumps are only ~8% of
            steps but account for ~97% of reconstruction error, so this is the
            knob that matters.
        move_strategy, jump_strategy:
            ``"quantile"`` (equal mass per bin) or ``"asinh"`` (uniform in
            ``asinh`` space, so bins widen smoothly into the tail). The defaults
            are empirical: quantile for the tightly-bounded move channel, asinh
            for the heavy-tailed jump channel. Exposed for ablation — set both
            to ``"quantile"`` to reproduce the failure mode described in
            :func:`_fit_bins_asinh`.
        channels:
            Which continuous channels to bin. Include ``"dt"`` for the model to
            generate pen *timing* as well as geometry — that is what enables
            velocity-driven stroke width, and it is information the 2013 model
            discards. Costs one extra output head.
        max_samples:
            Subsample cap per channel. Quantiles over 4M points are already far
            more precise than any bin count here can express.
        """
        for channel in channels:
            if channel not in CONTINUOUS_CHANNELS:
                raise ValueError(
                    f"unknown channel {channel!r}; expected one of {CONTINUOUS_CHANNELS}"
                )
        n_jump_bins = n_jump_bins if n_jump_bins is not None else n_bins

        arrays = [o for o in offsets if o.shape[0] > 0]
        if not arrays:
            raise ValueError("no non-empty offset arrays to fit on")

        stacked = np.concatenate(arrays, axis=0)
        is_jump = np.concatenate([jump_mask(o) for o in arrays])
        logger.info(
            "Fitting tokenizer on %d points (%d pen-down, %d pen-up jumps, %.1f%% jumps)",
            stacked.shape[0],
            int((~is_jump).sum()),
            int(is_jump.sum()),
            100.0 * is_jump.mean(),
        )

        # Fixed seed: bin edges must be reproducible across runs.
        rng = np.random.default_rng(0)
        if stacked.shape[0] > max_samples:
            keep = rng.choice(stacked.shape[0], max_samples, replace=False)
            stacked, is_jump = stacked[keep], is_jump[keep]

        bins: dict[str, ChannelBins] = {}
        for channel in channels:
            column = stacked[:, _CHANNEL_COLUMN[channel]]

            move_edges, move_reps = _fit_bins_or_trivial(
                column[~is_jump], n_bins, move_strategy
            )
            jump_edges, jump_reps = _fit_bins_or_trivial(
                column[is_jump], n_jump_bins, jump_strategy
            )

            bins[channel] = ChannelBins(
                move_edges=move_edges,
                jump_edges=jump_edges,
                representatives=np.concatenate([move_reps, jump_reps]),
                n_move=len(move_reps),
            )

            if len(move_reps) < n_bins or len(jump_reps) < n_jump_bins:
                logger.info(
                    "channel %r collapsed bins (move %d/%d, jump %d/%d): the value "
                    "distribution has a spike, most likely exact zeros.",
                    channel, len(move_reps), n_bins, len(jump_reps), n_jump_bins,
                )

        return cls(bins=bins)

    # ── encode / decode ───────────────────────────────────────────────────────

    def encode(self, offsets: np.ndarray) -> dict[str, np.ndarray]:
        """Offsets -> integer token arrays, one per channel plus ``pen``.

        Returns int16 arrays. int16 rather than int64 because the vocabularies
        fit comfortably and it quarters the dataset size on disk; cast to int64
        in the Dataset's ``__getitem__``, which is what ``nn.Embedding`` and
        ``cross_entropy`` want.
        """
        is_jump = jump_mask(offsets)
        tokens: dict[str, np.ndarray] = {}

        for channel in self.channels:
            channel_bins = self.bins[channel]
            column = offsets[:, _CHANNEL_COLUMN[channel]]

            # side="right" so a value exactly on an edge goes to the upper bin,
            # matching the convention used when fitting representatives.
            move_idx = np.searchsorted(channel_bins.move_edges, column, side="right")
            jump_idx = np.searchsorted(channel_bins.jump_edges, column, side="right")

            # Jump tokens live above the move range, so every token id has
            # exactly one meaning and decode needs no mode argument.
            combined = np.where(is_jump, jump_idx + channel_bins.n_move, move_idx)
            tokens[channel] = combined.astype(np.int16)

        # pen_up is already binary; no binning, just a type change.
        tokens["pen"] = (offsets[:, PEN_UP] > 0.5).astype(np.int16)
        return tokens

    def decode(self, tokens: dict[str, np.ndarray]) -> np.ndarray:
        """Integer tokens -> ``(N, 4)`` offsets. Inverse of :meth:`encode`.

        A single lookup per channel: the token id already encodes whether it was
        a move or a jump. Channels absent from ``tokens`` come back as zeros, so
        a model predicting only dx/dy/pen still yields a drawable array (dt = 0
        just disables velocity-based stroke width).
        """
        length = len(next(iter(tokens.values())))
        offsets = np.zeros((length, 4), dtype=np.float32)

        for channel in self.channels:
            if channel not in tokens:
                continue
            reps = self.bins[channel].representatives
            indices = np.asarray(tokens[channel], dtype=np.int64)
            # Clip rather than raise: a freshly-initialised model emits nonsense
            # indices, and crashing on the first sample render is unhelpful when
            # you are trying to watch it learn.
            offsets[:, _CHANNEL_COLUMN[channel]] = reps[np.clip(indices, 0, len(reps) - 1)]

        if "pen" in tokens:
            offsets[:, PEN_UP] = np.asarray(tokens["pen"], dtype=np.float32)

        return offsets

    # ── quality check ─────────────────────────────────────────────────────────

    def reconstruction_error(self, offsets: np.ndarray) -> dict[str, float]:
        """Measure round-trip error in *position* space, after integration.

        This is the number that predicts whether decoded output looks right.
        Per-step error does not, because `cumsum` amplifies it.

        Returns
        -------
        dict with:
            ``max_drift``     worst position error, normalised units
            ``final_drift``   error at the last point (total accumulated)
            ``x_height``      the line's vertical ink extent, for scale
            ``drift_ratio``   ``max_drift / x_height`` -- **the one to read**

        Calibrated by rendering round-trips: below ~0.05 invisible, ~0.2 shows
        visible inter-word spacing damage, above ~0.4 clearly broken. If it is
        too high, raise ``n_jump_bins`` first — jump error is what displaces
        whole words — then ``n_bins``.
        """
        decoded = self.decode(self.encode(offsets))

        true_xy = np.cumsum(offsets[:, [X, Y]], axis=0)
        decoded_xy = np.cumsum(decoded[:, [X, Y]], axis=0)
        drift = np.linalg.norm(true_xy - decoded_xy, axis=1)

        x_height = float(true_xy[:, 1].max() - true_xy[:, 1].min()) if len(true_xy) else 0.0
        max_drift = float(drift.max()) if drift.size else 0.0

        return {
            "max_drift": max_drift,
            "final_drift": float(drift[-1]) if drift.size else 0.0,
            "x_height": x_height,
            "drift_ratio": max_drift / x_height if x_height > 0 else float("inf"),
        }

    # ── persistence ───────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Write to JSON alongside the alphabet and the packed dataset.

        JSON rather than .npy so you can open the file and see what token 0
        actually means. The arrays are small, so readability costs nothing.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "channels": list(self.channels),
            "vocab_sizes": self.vocab_sizes,
            "bins": {
                c: {
                    "move_edges": self.bins[c].move_edges.tolist(),
                    "jump_edges": self.bins[c].jump_edges.tolist(),
                    "representatives": self.bins[c].representatives.tolist(),
                    "n_move": self.bins[c].n_move,
                }
                for c in self.channels
            },
        }
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "StrokeTokenizer":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            bins={
                channel: ChannelBins(
                    move_edges=np.asarray(spec["move_edges"], dtype=np.float32),
                    jump_edges=np.asarray(spec["jump_edges"], dtype=np.float32),
                    representatives=np.asarray(spec["representatives"], dtype=np.float32),
                    n_move=int(spec["n_move"]),
                )
                for channel, spec in payload["bins"].items()
            }
        )

    def describe(self) -> str:
        """Human-readable summary, for logs and for sanity-checking a fit."""

        def span(reps: np.ndarray) -> str:
            # A mode can legitimately have one bin (all values identical, or no
            # samples at all), and np.diff of a single element is empty.
            if len(reps) < 2:
                value = reps[0] if len(reps) else 0.0
                return f"single bin at {value:+.3f}"
            return (
                f"range [{reps[0]:+.3f}, {reps[-1]:+.3f}]  "
                f"finest step {np.min(np.diff(reps)):.5f}"
            )

        lines = [f"StrokeTokenizer  channels={list(self.channels)}"]
        for channel in self.channels:
            cb = self.bins[channel]
            lines.append(
                f"  {channel}: {cb.vocab_size} tokens "
                f"({cb.n_move} move + {cb.n_jump} jump)"
            )
            lines.append(f"       move {span(cb.representatives[: cb.n_move])}")
            lines.append(f"       jump {span(cb.representatives[cb.n_move :])}")
        lines.append("  pen: 2 tokens (binary)")
        return "\n".join(lines)
