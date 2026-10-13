"""Look at your data. Run this FIRST, before writing a single line of model code.

    python scripts/inspect_data.py
    # then open outputs/inspect/index.html

WHY THIS IS THE MOST IMPORTANT SCRIPT IN THE REPO
-------------------------------------------------
If your ground truth renders wrong, nothing downstream can possibly work —
and every symptom will point at the model. You will tune learning rates for
a week to fix a bug that was a flipped y-axis in the parser.

So before any training: render real data back to SVG and *look at it*. If
these samples are legible handwriting, your parser, coordinate conventions,
preprocessing and renderer are all correct, and any later bug is genuinely
in the model. That is an enormously valuable thing to know for certain.

WHAT IT PRINTS
--------------
1. Corpus statistics — how many lines survived indexing, and what got dropped
2. Writer-ID coverage — how much of the corpus can be style-conditioned
3. Character inventory — every character and its frequency
4. Sequence-length distribution — before and after resampling
5. An HTML contact sheet of rendered samples, raw vs preprocessed
6. Optionally a composed document page (``--page``)

Read the numbers. They tell you things about IAM-OnDB that no paper does.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np

# Make `hws` importable when running this file directly. Python puts the
# *script's* directory on sys.path, not the repo root, so `import hws` would
# otherwise fail. Unnecessary if you ran `pip install -e .`, harmless if you did.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hws import alphabet as alpha_mod  # noqa: E402
from hws import iam, preprocess, render  # noqa: E402

# Defaults match the layout produced by extracting the three IAM tarballs
# into datas/raw/iam-ondb/.
DEFAULT_DATA_ROOT = Path("datas/raw/iam-ondb")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the IAM-OnDB corpus and render samples to SVG.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT,
                        help="Directory containing ascii/, lineStrokes/, original/")
    parser.add_argument("--out", type=Path, default=Path("outputs/inspect"),
                        help="Where to write the SVGs and index.html")
    parser.add_argument("-n", "--n-samples", type=int, default=16,
                        help="How many random lines to render")
    parser.add_argument("--resample", choices=["step", "count", "none"], default="step",
                        help="Resampling mode (see hws.preprocess.preprocess_line)")
    parser.add_argument("--step-factor", type=float, default=2.0,
                        help="Spacing as a multiple of the line's median spacing "
                             "(step mode). 2.0 ~ halves the point count.")
    parser.add_argument("--points", type=int, default=500,
                        help="Target point count, when --resample count")
    parser.add_argument("--max-points", type=int, default=600,
                        help="Hard cap on points per line (step mode)")
    parser.add_argument("--seed", type=int, default=0,
                        help="Sampling seed; change it to see different lines")
    parser.add_argument("--min-count", type=int, default=1,
                        help="Alphabet frequency cutoff, for the report only")
    parser.add_argument("--page", action="store_true",
                        help="Also compose the samples into a ruled document page")
    parser.add_argument("--variable-width", action="store_true",
                        help="Modulate stroke width by pen speed (uses the time channel)")
    parser.add_argument("--verbose", action="store_true",
                        help="Show per-file skip reasons")
    return parser.parse_args()


def describe_distribution(name: str, values: np.ndarray) -> str:
    """Percentiles, not just mean/std.

    Handwriting length distributions are skewed, so a mean hides the tail —
    and the tail is what determines the sequence length you must pad to, and
    therefore your memory footprint.
    """
    if values.size == 0:
        return f"{name}: (no data)"
    pcts = np.percentile(values, [0, 5, 50, 95, 99, 100])
    return (
        f"{name}:\n"
        f"  min {pcts[0]:.0f} | p5 {pcts[1]:.0f} | median {pcts[2]:.0f} | "
        f"p95 {pcts[3]:.0f} | p99 {pcts[4]:.0f} | max {pcts[5]:.0f}\n"
        f"  mean {values.mean():.1f}, std {values.std():.1f}, n={values.size}"
    )


def build_html(cards: list[dict], out_dir: Path, page_svg: str | None) -> Path:
    """Write a contact sheet so you can scan all samples in one screen.

    SVGs are inlined rather than referenced with <img>, so the HTML is a
    single self-contained file you can move around or send to someone.
    """
    css = """
    body { font-family: ui-sans-serif, system-ui, sans-serif; background:#14161a;
           color:#e6e6e6; margin:0; padding:24px; }
    h1 { font-size:20px; margin:0 0 4px; }
    .sub { color:#8b93a1; font-size:13px; margin-bottom:24px; }
    .card { background:#1c1f26; border:1px solid #2a2f3a; border-radius:10px;
            padding:14px 16px; margin-bottom:16px; }
    .text { font-size:14px; color:#f0f0f0; margin-bottom:4px; }
    .meta { font-size:12px; color:#7f8796; margin-bottom:10px;
            font-family: ui-monospace, monospace; }
    .pair { display:grid; grid-template-columns:1fr 1fr; gap:14px; }
    .panel { background:#fdfdf8; border-radius:6px; padding:6px; overflow:hidden; }
    .panel svg { width:100%; height:auto; display:block; }
    .label { font-size:11px; color:#8b93a1; text-transform:uppercase;
             letter-spacing:.06em; margin-bottom:4px; }
    .page { background:#fdfdf8; border-radius:6px; padding:10px; max-width:660px; }
    .page svg { width:100%; height:auto; display:block; }
    """

    parts = [
        "<!DOCTYPE html><html><head><meta charset='utf-8'>",
        "<title>IAM-OnDB inspection</title>",
        f"<style>{css}</style></head><body>",
        "<h1>IAM-OnDB inspection</h1>",
        "<div class='sub'>Left: raw sensor points, as parsed. "
        "Right: after slant removal, denoising and arc-length resampling. "
        "If the right column is legible handwriting, the whole pipeline is correct.</div>",
    ]

    for card in cards:
        parts.append("<div class='card'>")
        parts.append(f"<div class='text'>{card['text']}</div>")
        parts.append(f"<div class='meta'>{card['meta']}</div>")
        parts.append("<div class='pair'>")
        parts.append(
            f"<div><div class='label'>raw &mdash; {card['n_raw']} pts</div>"
            f"<div class='panel'>{card['svg_raw']}</div></div>"
        )
        parts.append(
            f"<div><div class='label'>processed &mdash; {card['n_proc']} pts</div>"
            f"<div class='panel'>{card['svg_proc']}</div></div>"
        )
        parts.append("</div></div>")

    if page_svg:
        parts.append("<h1 style='margin-top:32px'>Composed page</h1>")
        parts.append(
            "<div class='sub'>Real lines placed at y-offsets with per-line jitter. "
            "This is what document assembly looks like &mdash; no image stitching.</div>"
        )
        parts.append(f"<div class='page'>{page_svg}</div>")

    parts.append("</body></html>")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "index.html"
    # XML declarations are illegal mid-HTML; strip them from the inlined SVGs.
    html = "\n".join(parts).replace('<?xml version="1.0" encoding="UTF-8"?>', "")
    path.write_text(html, encoding="utf-8")
    return path


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    ascii_root = args.data_root / "ascii"
    strokes_root = args.data_root / "lineStrokes"
    original_root = args.data_root / "original"

    for path, name in [(ascii_root, "ascii"), (strokes_root, "lineStrokes")]:
        if not path.is_dir():
            print(f"ERROR: {name} not found at {path}", file=sys.stderr)
            print(f"Pass --data-root, or extract the IAM tarballs there.", file=sys.stderr)
            return 1

    # ── 1. index the corpus (fast: no stroke XML parsed yet) ──────────────────
    print("=" * 72)
    print("INDEXING CORPUS")
    print("=" * 72)

    stats = iam.ParseStats()
    refs = list(
        iam.iter_line_refs(
            ascii_root,
            strokes_root,
            original_root if original_root.is_dir() else None,
            stats=stats,
        )
    )
    print(stats.summary())

    if not refs:
        print("\nERROR: no lines indexed. Check --data-root.", file=sys.stderr)
        return 1

    # ── 2. writer coverage ───────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("WRITER COVERAGE")
    print("=" * 72)

    with_writer = [r for r in refs if r.writer_id is not None]
    writers = {r.writer_id for r in with_writer}
    pct = 100.0 * len(with_writer) / len(refs)
    print(f"lines with a writer ID : {len(with_writer)} / {len(refs)}  ({pct:.1f}%)")
    print(f"distinct writers       : {len(writers)}")

    if with_writer:
        per_writer = np.array(
            [sum(1 for r in with_writer if r.writer_id == w) for w in sorted(writers)]
        )
        print(describe_distribution("lines per writer", per_writer))
    if pct < 100:
        # ASCII only in console output: the default Windows terminal codepage
        # is cp1252 and turns em-dashes into mojibake.
        print(
            f"\nNOTE: {len(refs) - len(with_writer)} lines have no writer ID. Decide "
            "explicitly whether to\n      drop them or give them a 'writer unknown' "
            "embedding - do not let it default."
        )

    # ── 3. character inventory ───────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("CHARACTER INVENTORY")
    print("=" * 72)

    texts = [r.text for r in refs]
    _, counts = alpha_mod.build_alphabet(texts, min_count=args.min_count)
    print(alpha_mod.format_inventory(counts, min_count=args.min_count))

    char_lengths = np.array([len(t) for t in texts])
    print("\n" + describe_distribution("characters per line", char_lengths))

    # ── 4. sample, preprocess, render ────────────────────────────────────────
    print("\n" + "=" * 72)
    print(f"RENDERING {args.n_samples} SAMPLES")
    print("=" * 72)

    rng = np.random.default_rng(args.seed)
    chosen_idx = rng.choice(len(refs), size=min(args.n_samples, len(refs)), replace=False)

    style = render.InkStyle(variable_width=args.variable_width)
    cards: list[dict] = []
    processed_lines: list[np.ndarray] = []
    raw_counts: list[int] = []
    proc_counts: list[int] = []

    for i in chosen_idx:
        ref = refs[int(i)]
        try:
            line = ref.load()
        except Exception as exc:  # noqa: BLE001 - inspection tool, report and continue
            print(f"  skip {ref.stroke_path.name}: {exc}")
            continue

        # Full pipeline -> offsets. Convert back to coordinates to draw, which
        # incidentally round-trips coords_to_offsets/offsets_to_coords: if that
        # pair were buggy, these renders would be visibly wrong.
        offsets = preprocess.preprocess_line(
            line.points,
            resample=args.resample,
            step_factor=args.step_factor,
            n_points=args.points,
            max_points=args.max_points,
        )
        coords = preprocess.offsets_to_coords(offsets)

        raw_counts.append(line.n_points)
        proc_counts.append(coords.shape[0])
        processed_lines.append(coords)

        writer = ref.writer_id if ref.writer_id is not None else "unknown"
        cards.append(
            {
                "text": line.text,
                "meta": (
                    f"{ref.form_id}-{ref.line_index:02d} | writer {writer} | "
                    f"{line.n_strokes} strokes | {line.duration:.1f}s"
                ),
                "n_raw": line.n_points,
                "n_proc": coords.shape[0],
                "svg_raw": render.render_line(line.points, style=style),
                "svg_proc": render.render_line(coords, style=style),
            }
        )

    if not cards:
        print("ERROR: every sampled line failed to parse.", file=sys.stderr)
        return 1

    raw_arr, proc_arr = np.array(raw_counts), np.array(proc_counts)
    print(describe_distribution("points per line, RAW", raw_arr))
    print(describe_distribution("points per line, RESAMPLED", proc_arr))
    print(f"\ncompression: {raw_arr.mean() / max(proc_arr.mean(), 1):.2f}x fewer points")

    # ── 5. optional page composition ─────────────────────────────────────────
    page_svg = None
    if args.page:
        layout = render.PageLayout(ruled=True, margin_rule=True)
        page_svg = render.render_page(
            processed_lines[:18], layout=layout, style=style, seed=args.seed
        )
        render.save_svg(page_svg, args.out / "page.svg")
        print(f"\nwrote {args.out / 'page.svg'}")

    # ── 6. contact sheet ─────────────────────────────────────────────────────
    html_path = build_html(cards, args.out, page_svg)
    print("\n" + "=" * 72)
    print(f"Wrote {len(cards)} samples -> {html_path}")
    print("=" * 72)
    print("\nOpen that file and check the RIGHT column of each pair.")
    print("Legible handwriting => parser, conventions, preprocessing, renderer all OK.")
    print("Upside down        => y-axis sign in hws.render.path_data")
    print("Scribbled mess     => resampling too aggressive; try --points 800")
    print("Lines through air  => pen_up handling in hws.preprocess.split_strokes")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
