"""Rendering: pen coordinates -> SVG.

THIS FILE IS SHARED INFRASTRUCTURE. TREAT IT AS SUCH.
-----------------------------------------------------
Three separate things will render strokes: your training-time sample dumps,
the inference API, and the app. If any of them has its own copy of this
logic, they will drift, and you will spend days chasing a "model bug" that
is really a rendering bug — the model output was fine, one renderer just
flipped an axis.

So: one implementation, imported everywhere. If the app ends up needing a
TypeScript port, port it as a direct translation and add a test that both
produce the same path string for the same input.

(For a concrete example of what goes wrong without this discipline, the
calligrapher-ai desktop app has layout code that computes each chunk's
rendered width and then never advances the x-cursor with it, so multi-chunk
text renders every chunk on top of the previous one. The generation was
fine; the compositing silently lost a line.)

COORDINATE SYSTEMS — THE ONE THING TO GET RIGHT
----------------------------------------------
    Our arrays:  +y is UP    (set in hws.iam.parse_stroke_xml)
    SVG:         +y is DOWN

So every function here negates y on the way out. It happens in exactly one
place — :func:`path_data` — and nowhere else. If your handwriting renders
upside down, that is the line to look at.

WHY VECTORS AND NOT PIXELS
--------------------------
Output is SVG path data: a few KB per line, scales to any resolution, edits
in Illustrator/Inkscape, and drives a pen plotter directly. The image-based
handwriting models (HWT, VATr, diffusion-on-pixels) produce word bitmaps
that then have to be stitched with matching baselines and blended
backgrounds — a much harder problem that we avoid entirely by generating
strokes instead of pictures.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from hws.iam import PEN_UP, TIME, X, Y
from hws.preprocess import split_strokes

# Coordinates are rounded to this many decimals in the emitted path data.
# Two is well under the width of a rendered pixel at any sane zoom, and it
# roughly halves file size versus full float repr.
_COORD_PRECISION = 2


@dataclass
class InkStyle:
    """How the ink looks. Purely presentational — never affects the model.

    This is the "programmatic realism" layer: the parameters here are what
    turn correct-but-sterile stroke data into something that reads as
    handwriting on paper. Tune them freely; none of it requires retraining.
    """

    color: str = "#1a1a2e"
    """Not pure black. Real ink is never #000000, and the difference is
    surprisingly visible — pure black reads as 'computer'."""

    width: float = 1.6
    """Base stroke width in SVG units."""

    variable_width: bool = False
    """Modulate width by pen speed. Requires the ``time`` channel to have
    survived preprocessing (it does, if you used resample_arclength)."""

    width_contrast: float = 0.45
    """How strongly speed affects width, in [0, 1). 0 = constant width.
    0.45 is a visible-but-not-cartoonish default."""

    linecap: str = "round"
    """``round`` matches a ballpoint. ``butt`` looks like a technical pen."""

    background: str | None = "#fdfdf8"
    """Slight warm off-white — paper, not screen. ``None`` for transparent."""

    bezier: bool = True
    """Use C1-continuous Catmull-Rom cubic Bézier splines for calligraphic ink fluidity."""


# ── geometry helpers ──────────────────────────────────────────────────────────


def bounding_box(coords: np.ndarray) -> tuple[float, float, float, float]:
    """Return ``(min_x, min_y, max_x, max_y)`` in *our* coordinate space (+y up).

    Computed from the data rather than assumed. The alternative — hardcoding
    a viewBox — is how you end up with content rendered outside the visible
    area and no idea why the file looks empty.
    """
    if coords.shape[0] == 0:
        return 0.0, 0.0, 0.0, 0.0
    return (
        float(coords[:, X].min()),
        float(coords[:, Y].min()),
        float(coords[:, X].max()),
        float(coords[:, Y].max()),
    )


def speeds(coords: np.ndarray) -> np.ndarray:
    """Per-point pen speed in units/second. Shape ``(len(coords),)``.

    This is why we bothered keeping the ``time`` channel through parsing and
    resampling. Real pens lay down less ink when moving fast, so speed maps
    to stroke width and you get pressure variation for free — without
    modelling pressure at all.

    Points where dt is zero or negative fall back to the median speed rather
    than producing an infinity.
    """
    if coords.shape[0] < 2:
        return np.ones(coords.shape[0], dtype=np.float32)

    deltas = np.diff(coords[:, [X, Y]], axis=0)
    distances = np.linalg.norm(deltas, axis=1)
    dts = np.diff(coords[:, TIME])

    valid = dts > 0
    out = np.zeros(len(distances), dtype=np.float32)
    out[valid] = distances[valid] / dts[valid]

    fallback = float(np.median(out[valid])) if valid.any() else 1.0
    out[~valid] = fallback

    # Prepend so the array lines up with coords: point 0 inherits point 1's
    # speed, since there is no earlier point to measure against.
    return np.concatenate([[out[0]], out]).astype(np.float32)


# ── path generation ───────────────────────────────────────────────────────────


def path_data(coords: np.ndarray, y_flip: bool = True, bezier: bool = False) -> str:
    """Convert coordinates to an SVG ``d`` attribute string.

    ``M`` starts each stroke (pen down), ``L`` (or ``C`` when bezier=True) continues it.
    When ``bezier=True``, strokes are smoothed with Catmull-Rom cubic
    Bézier splines, giving calligraphic fluidity and eliminating polygonal
    hand tremors. Pen-up gaps produce no output at all.

    Parameters
    ----------
    coords:
        Array of shape ``(N, 4)``: ``[x, y, pen_up, time]``.
    y_flip:
        Negate y for SVG's downward axis. Leave this ``True`` unless you are
        writing a test that wants raw values.
    bezier:
        If True, interpolate stroke points with C1-continuous cubic Bézier curves.

    Returns
    -------
    A ``d`` string, or ``""`` for empty input (a valid, empty path).
    """
    parts: list[str] = []
    sign = -1.0 if y_flip else 1.0

    for stroke in split_strokes(coords):
        n = stroke.shape[0]
        if n == 0:
            continue
        if n == 1 or not bezier:
            for i, point in enumerate(stroke):
                command = "M" if i == 0 else "L"
                x = round(float(point[X]), _COORD_PRECISION)
                y = round(sign * float(point[Y]), _COORD_PRECISION)
                parts.append(f"{command}{x},{y}")
            continue

        if n == 2:
            x0 = round(float(stroke[0, X]), _COORD_PRECISION)
            y0 = round(sign * float(stroke[0, Y]), _COORD_PRECISION)
            x1 = round(float(stroke[1, X]), _COORD_PRECISION)
            y1 = round(sign * float(stroke[1, Y]), _COORD_PRECISION)
            parts.append(f"M{x0},{y0} L{x1},{y1}")
            continue

        # Catmull-Rom to Cubic Bézier spline
        pts = np.vstack([[stroke[0]], stroke, [stroke[-1]]])
        x0 = round(float(stroke[0, X]), _COORD_PRECISION)
        y0 = round(sign * float(stroke[0, Y]), _COORD_PRECISION)
        parts.append(f"M{x0},{y0}")

        for i in range(1, n):
            p0 = pts[i - 1]
            p1 = pts[i]
            p2 = pts[i + 1]
            p3 = pts[i + 2]

            c1_x = round(float(p1[X] + (p2[X] - p0[X]) / 6.0), _COORD_PRECISION)
            c1_y = round(sign * float(p1[Y] + (p2[Y] - p0[Y]) / 6.0), _COORD_PRECISION)
            c2_x = round(float(p2[X] - (p3[X] - p1[X]) / 6.0), _COORD_PRECISION)
            c2_y = round(sign * float(p2[Y] - (p3[Y] - p1[Y]) / 6.0), _COORD_PRECISION)
            p2_x = round(float(p2[X]), _COORD_PRECISION)
            p2_y = round(sign * float(p2[Y]), _COORD_PRECISION)

            parts.append(f"C{c1_x},{c1_y} {c2_x},{c2_y} {p2_x},{p2_y}")

    return " ".join(parts)


def _variable_width_paths(coords: np.ndarray, style: InkStyle) -> list[tuple[str, float]]:
    """Split into one short path per segment so each can carry its own width.

    Returns ``[(d, width), ...]``.

    More verbose output than a single path — roughly one path element per
    point — but it is the simplest way to get variable width in plain SVG
    with no filters or outline math. If file size becomes a problem, the next
    step is fitting a filled polygon around the stroke centreline.
    """
    pen_speeds = speeds(coords)

    # Normalise speed to [0, 1] using percentiles rather than min/max, so a
    # single anomalous fast segment cannot compress everything else into a
    # narrow band.
    lo, hi = np.percentile(pen_speeds, [5, 95])
    if hi <= lo:
        normalised = np.full_like(pen_speeds, 0.5)
    else:
        normalised = np.clip((pen_speeds - lo) / (hi - lo), 0.0, 1.0)

    # Fast -> thin. Hence (1 - normalised).
    widths = style.width * (1.0 - style.width_contrast * normalised)

    out: list[tuple[str, float]] = []
    offset = 0
    for stroke in split_strokes(coords):
        n = stroke.shape[0]
        for i in range(n - 1):
            a, b = stroke[i], stroke[i + 1]
            d = (
                f"M{round(float(a[X]), _COORD_PRECISION)},"
                f"{round(-float(a[Y]), _COORD_PRECISION)} "
                f"L{round(float(b[X]), _COORD_PRECISION)},"
                f"{round(-float(b[Y]), _COORD_PRECISION)}"
            )
            out.append((d, float(widths[offset + i])))
        offset += n

    return out


# ── SVG documents ─────────────────────────────────────────────────────────────


def _svg_document(body: str, view_box: tuple[float, float, float, float], background: str | None) -> str:
    """Wrap path elements in an SVG root with an explicit viewBox.

    ``width``/``height`` are omitted deliberately so the SVG scales to
    whatever container it lands in — a React Native ``<Svg>``, a browser, a
    PDF. Setting them in pixels here would fight every consumer.
    """
    min_x, min_y, width, height = view_box
    rect = (
        f'<rect x="{min_x:.2f}" y="{min_y:.2f}" width="{width:.2f}" '
        f'height="{height:.2f}" fill="{background}"/>'
        if background
        else ""
    )
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        f'<svg xmlns="http://www.w3.org/2000/svg" '
        f'viewBox="{min_x:.2f} {min_y:.2f} {width:.2f} {height:.2f}">\n'
        f"{rect}\n{body}\n</svg>\n"
    )


def render_line(
    coords: np.ndarray,
    style: InkStyle | None = None,
    padding: float = 8.0,
) -> str:
    """Render one line of handwriting as a standalone SVG document.

    The viewBox is fitted to the actual ink plus ``padding``, so the output
    is never clipped and never mostly-empty regardless of where in coordinate
    space the strokes happen to sit.
    """
    style = style or InkStyle()

    if coords.shape[0] == 0:
        return _svg_document("", (0.0, 0.0, 1.0, 1.0), style.background)

    min_x, min_y, max_x, max_y = bounding_box(coords)

    # Flip y for SVG, then pad. Note min/max swap under negation: what was
    # the topmost point (max y, +y up) becomes the smallest SVG y.
    svg_min_y = -max_y - padding
    svg_min_x = min_x - padding
    width = (max_x - min_x) + 2 * padding
    height = (max_y - min_y) + 2 * padding

    if style.variable_width:
        body = "\n".join(
            f'<path d="{d}" stroke="{style.color}" stroke-width="{w:.2f}" '
            f'stroke-linecap="{style.linecap}" fill="none"/>'
            for d, w in _variable_width_paths(coords, style)
        )
    else:
        body = (
            f'<path d="{path_data(coords, bezier=style.bezier)}" stroke="{style.color}" '
            f'stroke-width="{style.width}" stroke-linecap="{style.linecap}" '
            f'stroke-linejoin="round" fill="none"/>'
        )

    return _svg_document(body, (svg_min_x, svg_min_y, width, height), style.background)


@dataclass
class PageLayout:
    """Document geometry for composing many lines into a page.

    This is the answer to "how do I make it look like an actual assignment":
    you do not stitch images, you place stroke lines at y-offsets. All of it
    is deterministic code, none of it involves the model, and every value
    here is yours to tune.

    Defaults are loosely A4-ish assuming preprocessed coordinates (median
    step length 1.0), which puts x-height at roughly 10 units.
    """

    width: float = 620.0
    height: float = 877.0
    margin_left: float = 60.0
    margin_top: float = 70.0
    margin_right: float = 45.0
    line_height: float = 42.0

    ruled: bool = False
    """Draw ruled lines. Convincing for exercise-book output."""

    rule_color: str = "#c9d6e3"
    margin_rule: bool = False
    """Draw the vertical red margin line found in Indian school notebooks."""

    margin_rule_color: str = "#e8a0a0"

    jitter_x: float = 2.5
    """Random horizontal wobble per line, in SVG units. Nobody starts every
    line at exactly the same x, and identical left edges read as machine
    output more strongly than almost anything else."""

    jitter_baseline: float = 1.5
    """Random vertical wobble per line."""

    jitter_rotation: float = 0.35
    """Random per-line rotation in degrees. Small — this compounds visually."""


def render_page(
    lines: list[np.ndarray],
    layout: PageLayout | None = None,
    style: InkStyle | None = None,
    seed: int | None = None,
) -> str:
    """Compose several lines into a single page SVG.

    Parameters
    ----------
    lines:
        One coordinate array per line, each already preprocessed (so they
        share a common scale — that is what ``normalize_scale`` buys you, and
        it is why no per-line rescaling happens here).
    seed:
        Seeds the layout jitter. Pass a fixed value when you want the same
        page twice (comparing two models, say); pass ``None`` for variety.

    Returns
    -------
    An SVG document string. Overflowing lines are *not* silently dropped —
    they are still emitted, just positioned below the page bottom, so a
    layout mistake is visible rather than invisible.
    """
    layout = layout or PageLayout()
    style = style or InkStyle()
    rng = np.random.default_rng(seed)

    parts: list[str] = []

    if style.background:
        parts.append(
            f'<rect x="0" y="0" width="{layout.width}" height="{layout.height}" '
            f'fill="{style.background}"/>'
        )

    # ── paper furniture, drawn first so ink sits on top ──
    if layout.ruled:
        y = layout.margin_top
        while y < layout.height - 20:
            parts.append(
                f'<line x1="{layout.margin_left - 10:.1f}" y1="{y:.1f}" '
                f'x2="{layout.width - layout.margin_right:.1f}" y2="{y:.1f}" '
                f'stroke="{layout.rule_color}" stroke-width="0.7"/>'
            )
            y += layout.line_height

    if layout.margin_rule:
        parts.append(
            f'<line x1="{layout.margin_left - 14:.1f}" y1="0" '
            f'x2="{layout.margin_left - 14:.1f}" y2="{layout.height}" '
            f'stroke="{layout.margin_rule_color}" stroke-width="1.1"/>'
        )

    # ── the handwriting ──
    for i, coords in enumerate(lines):
        if coords.shape[0] == 0:
            continue

        baseline_y = layout.margin_top + i * layout.line_height
        min_x, _, _, _ = bounding_box(coords)

        # Shift so the line starts at the left margin. Without this, lines
        # inherit whatever absolute x the model happened to start at and the
        # left edge of the page looks ragged in a way that reads as broken
        # rather than human.
        dx = layout.margin_left - min_x + float(rng.normal(0.0, layout.jitter_x))
        dy = baseline_y + float(rng.normal(0.0, layout.jitter_baseline))
        rotation = float(rng.normal(0.0, layout.jitter_rotation))

        if style.variable_width:
            inner = "\n".join(
                f'<path d="{d}" stroke="{style.color}" stroke-width="{w:.2f}" '
                f'stroke-linecap="{style.linecap}" fill="none"/>'
                for d, w in _variable_width_paths(coords, style)
            )
        else:
            inner = (
                f'<path d="{path_data(coords, bezier=style.bezier)}" stroke="{style.color}" '
                f'stroke-width="{style.width}" stroke-linecap="{style.linecap}" '
                f'stroke-linejoin="round" fill="none"/>'
            )

        # rotate() after translate() pivots about the line's own start point,
        # which is what a hand does. Rotating about the page origin would
        # swing the line across the page.
        parts.append(
            f'<g transform="translate({dx:.2f},{dy:.2f}) rotate({rotation:.2f})">'
            f"{inner}</g>"
        )

    return _svg_document(
        "\n".join(parts),
        (0.0, 0.0, layout.width, layout.height),
        background=None,  # already drawn above, under the ruling
    )


def save_svg(svg: str, path: str | Path) -> Path:
    """Write an SVG string to disk, creating parent directories as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(svg, encoding="utf-8")
    return path
