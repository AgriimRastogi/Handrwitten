"""Pack the corpus into HDF5 datasets ready for training.

    python scripts/prepare_data.py
    # writes datas/processed/{train,val,test}.h5 + alphabet.json + tokenizer.json

WHAT THIS DOES, IN ORDER
------------------------
1. Parse and preprocess every line (the slow part, ~2 min)
2. Split **by writer**, so no person appears in two splits
3. Fit the alphabet on the **training split only**
4. Fit the stroke tokenizer on the **training split only**
5. Encode and write each split
6. Report tokenizer reconstruction error on val -- the quality gate

WHY STEPS 3 AND 4 SAY "TRAINING SPLIT ONLY"
-------------------------------------------
Fitting either one over the whole corpus leaks the validation distribution into
your preprocessing. For the tokenizer this is not academic: bin edges are
quantiles of the data, so fitting on val means the bins are tuned to values the
model is supposed to be evaluated on. The leak is small but it is free to avoid.

Consequence: a val/test line can contain a character the training split never
had (some appear once in the entire corpus). Those lines are dropped and
counted rather than silently mapped to padding.

STORAGE FORMAT
--------------
Ragged, not padded. Sequences are concatenated into one flat array per channel,
with an offsets index giving each line's slice:

    tokens/dx    int16   [total_points]        line i = [off[i] : off[i+1]]
    offsets      int64   [n_lines + 1]
    chars        int16   [total_chars]
    char_offsets int64   [n_lines + 1]
    raw          float32 [total_points, 4]     un-tokenised offsets
    writer_index int32   [n_lines]             contiguous, 0 = unknown
    ...

Padding to a fixed length would waste ~40% of the file (p95 length is well
above the median) and, worse, bake a maximum length into the dataset. Ragged
storage lets the DataLoader's collate function decide padding per batch, which
is where that decision belongs.

``raw`` holds the pre-tokenisation float offsets. It roughly doubles file size
(~100 MB total, trivial) and means you can try an MDN output head later without
re-running any of this.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hws import iam, preprocess  # noqa: E402
from hws.alphabet import Alphabet, build_alphabet, format_inventory  # noqa: E402
from hws.tokenizer import StrokeTokenizer  # noqa: E402

logger = logging.getLogger("prepare_data")

DEFAULT_DATA_ROOT = Path("datas/raw/iam-ondb")
DEFAULT_OUT = Path("datas/processed")

# Reserved embedding index for "we do not know who wrote this".
WRITER_UNKNOWN = 0


@dataclass
class Record:
    """One preprocessed line, held in memory between the fit and write passes."""

    form_id: str
    line_index: int
    text: str
    writer_id: int | None
    offsets: np.ndarray  # (N, 4) float32 [dx, dy, pen_up, dt]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preprocess and pack IAM-OnDB into HDF5 for training.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)

    group = parser.add_argument_group("preprocessing")
    group.add_argument("--resample", choices=["step", "count", "none"], default="step")
    group.add_argument("--step-factor", type=float, default=2.0,
                       help="Spacing as a multiple of each line's median spacing")
    group.add_argument("--points", type=int, default=500,
                       help="Target point count, when --resample count")
    group.add_argument("--max-points", type=int, default=600,
                       help="Hard cap on points per line")

    group = parser.add_argument_group("tokenizer")
    group.add_argument("--bins", type=int, default=256,
                       help="Bins for pen-down moves, per channel")
    group.add_argument("--jump-bins", type=int, default=256,
                       help="Bins for pen-up jumps, per channel")
    group.add_argument("--no-dt", action="store_true",
                       help="Skip the dt channel (drops velocity modelling)")
    group.add_argument("--move-strategy", choices=["quantile", "asinh"], default="quantile",
                       help="Binning for pen-down moves")
    group.add_argument("--jump-strategy", choices=["quantile", "asinh"], default="asinh",
                       help="Binning for pen-up jumps. 'quantile' reproduces the "
                            "heavy-tail failure documented in hws/tokenizer.py")

    group = parser.add_argument_group("splitting")
    group.add_argument("--val-frac", type=float, default=0.1)
    group.add_argument("--test-frac", type=float, default=0.1)
    group.add_argument("--split-seed", type=int, default=0)
    group.add_argument("--unknown-writers", choices=["drop", "train-only"], default="drop",
                       help="What to do with the ~11%% of lines lacking a writer ID. "
                            "'drop' excludes them. 'train-only' keeps them in train "
                            "with writer index 0, never in val/test.")

    group = parser.add_argument_group("misc")
    group.add_argument("--min-char-count", type=int, default=1,
                       help="Drop alphabet characters rarer than this")
    group.add_argument("--min-points", type=int, default=20,
                       help="Discard lines shorter than this after preprocessing")
    group.add_argument("--limit", type=int, default=None,
                       help="Only process the first N lines (smoke test)")
    group.add_argument("--verbose", action="store_true")
    return parser.parse_args()


# ── stage 1: parse and preprocess ─────────────────────────────────────────────


def load_records(args: argparse.Namespace) -> tuple[list[Record], iam.ParseStats]:
    """Parse and preprocess the whole corpus into memory.

    Holding it all in RAM is fine: ~4M points x 4 float32 is about 65 MB. Doing
    it in one pass matters because we need the training split identified before
    we can fit the alphabet and tokenizer, and re-parsing 12k XML files to avoid
    65 MB of RAM would be a poor trade.
    """
    ascii_root = args.data_root / "ascii"
    strokes_root = args.data_root / "lineStrokes"
    original_root = args.data_root / "original"

    if not original_root.is_dir():
        logger.warning(
            "original/ missing -> no writer IDs -> a writer-disjoint split is "
            "impossible. Extract original-all.tar first."
        )

    stats = iam.ParseStats()
    records: list[Record] = []
    n_too_short = 0

    iterator = iam.iter_raw_lines(
        ascii_root,
        strokes_root,
        original_root if original_root.is_dir() else None,
        stats=stats,
        require_monotonic_time=True,
    )

    for line in tqdm(iterator, desc="preprocessing", unit="line"):
        if args.limit is not None and len(records) >= args.limit:
            break

        offsets = preprocess.preprocess_line(
            line.points,
            resample=args.resample,
            step_factor=args.step_factor,
            n_points=args.points,
            max_points=args.max_points,
        )

        # A line this short is a fragment, not handwriting. Cheaper to drop it
        # here than to have it produce a degenerate training example.
        if offsets.shape[0] < args.min_points:
            n_too_short += 1
            continue

        records.append(
            Record(
                form_id=line.form_id,
                line_index=line.line_index,
                text=line.text,
                writer_id=line.writer_id,
                offsets=offsets,
            )
        )

    if n_too_short:
        logger.info("Dropped %d lines shorter than %d points", n_too_short, args.min_points)

    return records, stats


# ── stage 2: split by writer ──────────────────────────────────────────────────


def split_records(
    records: list[Record], args: argparse.Namespace
) -> tuple[dict[str, list[Record]], dict[int, int]]:
    """Split writer-disjointly and build a contiguous writer index mapping.

    Returns ``(splits, writer_index)`` where ``writer_index`` maps raw IAM
    writer IDs (values like 10026) to contiguous embedding indices starting at
    1. Index 0 is reserved for "unknown".

    Raw IDs cannot be used directly: ``nn.Embedding`` needs indices in
    ``[0, n_writers)``, and an embedding table sized for 10026+ would be mostly
    dead rows.
    """
    known = [r for r in records if r.writer_id is not None]
    unknown = [r for r in records if r.writer_id is None]

    logger.info(
        "%d lines with a writer ID, %d without (%.1f%%)",
        len(known), len(unknown), 100.0 * len(unknown) / max(len(records), 1),
    )

    if not known:
        raise SystemExit(
            "No lines have writer IDs, so no writer-disjoint split is possible.\n"
            "Extract original-all.tar into <data-root>/original/ and re-run."
        )

    splits = iam.split_by_writer(
        known, val_frac=args.val_frac, test_frac=args.test_frac, seed=args.split_seed
    )

    if unknown:
        if args.unknown_writers == "train-only":
            # Train only, never val/test: "generalises to an unseen writer" is
            # the claim being evaluated, and a line whose writer is unknown
            # cannot support or refute it.
            splits["train"].extend(unknown)
            logger.info("Added %d unknown-writer lines to train only", len(unknown))
        else:
            logger.info("Dropped %d unknown-writer lines", len(unknown))

    # Sorted so the mapping is reproducible across runs.
    writer_ids = sorted({r.writer_id for r in known})
    writer_index = {wid: i + 1 for i, wid in enumerate(writer_ids)}
    logger.info(
        "Writer embedding table: %d rows (index 0 = unknown, 1..%d = writers)",
        len(writer_index) + 1, len(writer_index),
    )
    return splits, writer_index


# ── stage 3: filter on alphabet coverage ──────────────────────────────────────


def filter_by_alphabet(
    records: list[Record], alphabet: Alphabet, split_name: str
) -> list[Record]:
    """Drop lines containing characters absent from the training alphabet.

    Necessary because the alphabet is fitted on train only and some characters
    occur once in the whole corpus — `&` and `*` appear exactly once each. If
    that one occurrence lands in val, the alphabet has no index for it.

    Dropping the line is the honest option. Mapping the character to padding
    (what the 2013 implementation does) trains the model to treat a real glyph
    as nothing at all.
    """
    kept, dropped_chars = [], set()
    for record in records:
        missing = {c for c in record.text if c not in alphabet}
        if missing:
            dropped_chars |= missing
            continue
        kept.append(record)

    n_dropped = len(records) - len(kept)
    if n_dropped:
        logger.info(
            "%s: dropped %d lines containing out-of-alphabet characters %s",
            split_name, n_dropped, sorted(dropped_chars),
        )
    return kept


# ── stage 4: write HDF5 ───────────────────────────────────────────────────────


def write_split(
    path: Path,
    records: list[Record],
    alphabet: Alphabet,
    tokenizer: StrokeTokenizer,
    writer_index: dict[int, int],
) -> dict:
    """Write one split to HDF5 in ragged form. Returns a stats dict."""
    path.parent.mkdir(parents=True, exist_ok=True)

    token_channels = list(tokenizer.channels) + ["pen"]
    per_channel: dict[str, list[np.ndarray]] = {c: [] for c in token_channels}
    raw_chunks: list[np.ndarray] = []
    char_chunks: list[np.ndarray] = []

    point_offsets = [0]
    char_offsets = [0]
    writer_indices, line_indices, form_ids = [], [], []

    for record in records:
        tokens = tokenizer.encode(record.offsets)
        for channel in token_channels:
            per_channel[channel].append(tokens[channel])
        raw_chunks.append(record.offsets)

        encoded_text = np.asarray(alphabet.encode(record.text), dtype=np.int16)
        char_chunks.append(encoded_text)

        point_offsets.append(point_offsets[-1] + record.offsets.shape[0])
        char_offsets.append(char_offsets[-1] + encoded_text.shape[0])

        writer_indices.append(
            WRITER_UNKNOWN if record.writer_id is None else writer_index[record.writer_id]
        )
        line_indices.append(record.line_index)
        form_ids.append(record.form_id)

    with h5py.File(path, "w") as handle:
        token_group = handle.create_group("tokens")
        for channel in token_channels:
            token_group.create_dataset(
                channel,
                data=np.concatenate(per_channel[channel]),
                compression="gzip",
                compression_opts=4,
            )

        handle.create_dataset("raw", data=np.concatenate(raw_chunks, axis=0),
                              compression="gzip", compression_opts=4)
        handle.create_dataset("offsets", data=np.asarray(point_offsets, dtype=np.int64))
        handle.create_dataset("chars", data=np.concatenate(char_chunks),
                              compression="gzip", compression_opts=4)
        handle.create_dataset("char_offsets", data=np.asarray(char_offsets, dtype=np.int64))
        handle.create_dataset("writer_index", data=np.asarray(writer_indices, dtype=np.int32))
        handle.create_dataset("line_index", data=np.asarray(line_indices, dtype=np.int16))
        handle.create_dataset("form_id", data=np.asarray(form_ids, dtype=h5py.string_dtype()))

        # Attributes so a loader can sanity-check itself against the file rather
        # than trusting a config that may have drifted.
        handle.attrs["n_lines"] = len(records)
        handle.attrs["n_points"] = point_offsets[-1]
        handle.attrs["n_chars"] = char_offsets[-1]
        handle.attrs["char_vocab_size"] = alphabet.vocab_size
        handle.attrs["n_writers"] = len(writer_index) + 1
        handle.attrs["token_channels"] = token_channels
        for channel, size in tokenizer.vocab_sizes.items():
            handle.attrs[f"vocab_{channel}"] = size

    lengths = np.diff(point_offsets)
    return {
        "n_lines": len(records),
        "n_points": int(point_offsets[-1]),
        "n_writers": len({w for w in writer_indices if w != WRITER_UNKNOWN}),
        "median_len": int(np.median(lengths)) if lengths.size else 0,
        "max_len": int(lengths.max()) if lengths.size else 0,
        "file_mb": path.stat().st_size / 1e6,
    }


# ── main ──────────────────────────────────────────────────────────────────────


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not (args.data_root / "ascii").is_dir():
        print(f"ERROR: no ascii/ under {args.data_root}", file=sys.stderr)
        return 1

    # ── 1. parse + preprocess ────────────────────────────────────────────────
    print("=" * 72)
    print("PARSING + PREPROCESSING")
    print("=" * 72)
    records, stats = load_records(args)
    print(stats.summary())
    if not records:
        print("ERROR: no usable lines.", file=sys.stderr)
        return 1
    print(f"\nusable lines: {len(records)}")

    # ── 2. split ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("WRITER-DISJOINT SPLIT")
    print("=" * 72)
    splits, writer_index = split_records(records, args)

    # ── 3. alphabet, from train only ─────────────────────────────────────────
    print("\n" + "=" * 72)
    print("ALPHABET (fitted on train only)")
    print("=" * 72)
    alphabet, counts = build_alphabet(
        (r.text for r in splits["train"]), min_count=args.min_char_count
    )
    print(format_inventory(counts, min_count=args.min_char_count))
    print(f"\nvocab_size (incl. PAD): {alphabet.vocab_size}")

    for name in ("train", "val", "test"):
        splits[name] = filter_by_alphabet(splits[name], alphabet, name)

    # ── 4. tokenizer, from train only ────────────────────────────────────────
    print("\n" + "=" * 72)
    print("STROKE TOKENIZER (fitted on train only)")
    print("=" * 72)
    channels = ("dx", "dy") if args.no_dt else ("dx", "dy", "dt")
    tokenizer = StrokeTokenizer.fit(
        (r.offsets for r in splits["train"]),
        n_bins=args.bins,
        n_jump_bins=args.jump_bins,
        channels=channels,
        move_strategy=args.move_strategy,
        jump_strategy=args.jump_strategy,
    )
    print(tokenizer.describe())

    # ── 5. quality gate: drift measured on val ───────────────────────────────
    print("\n" + "=" * 72)
    print("RECONSTRUCTION QUALITY (val split)")
    print("=" * 72)
    ratios = np.array(
        [tokenizer.reconstruction_error(r.offsets)["drift_ratio"] for r in splits["val"]]
    )
    print(
        f"drift_ratio  median {np.median(ratios):.4f} | "
        f"p95 {np.percentile(ratios, 95):.4f} | max {ratios.max():.4f}"
    )
    print("(calibrated by rendering: <0.05 invisible, ~0.2 visible spacing damage,")
    print(" >0.4 clearly broken. Raise --jump-bins first if this is too high.)")
    if np.median(ratios) > 0.1:
        print("\nWARNING: median drift above 0.1. Consider --bins 512 --jump-bins 512.")

    # ── 6. write ─────────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("WRITING")
    print("=" * 72)
    args.out.mkdir(parents=True, exist_ok=True)

    alphabet.save(args.out / "alphabet.json")
    tokenizer.save(args.out / "tokenizer.json")
    (args.out / "writer_index.json").write_text(
        json.dumps(
            {"unknown_index": WRITER_UNKNOWN,
             "writer_to_index": {str(k): v for k, v in writer_index.items()}},
            indent=2,
        ),
        encoding="utf-8",
    )

    summary = {}
    for name in ("train", "val", "test"):
        if not splits[name]:
            logger.warning("split %r is empty, skipping", name)
            continue
        info = write_split(
            args.out / f"{name}.h5", splits[name], alphabet, tokenizer, writer_index
        )
        summary[name] = info
        print(
            f"{name:5}  {info['n_lines']:6d} lines  {info['n_points']:9d} points  "
            f"{info['n_writers']:4d} writers  median len {info['median_len']:4d}  "
            f"max {info['max_len']:4d}  {info['file_mb']:6.1f} MB"
        )

    config = {
        "resample": args.resample,
        "step_factor": args.step_factor,
        "max_points": args.max_points,
        "bins": args.bins,
        "jump_bins": args.jump_bins,
        "move_strategy": args.move_strategy,
        "jump_strategy": args.jump_strategy,
        "channels": list(channels),
        "unknown_writers": args.unknown_writers,
        "split_seed": args.split_seed,
        "char_vocab_size": alphabet.vocab_size,
        "n_writers": len(writer_index) + 1,
        "vocab_sizes": tokenizer.vocab_sizes,
        "drift_ratio_median": float(np.median(ratios)),
        "splits": summary,
    }
    (args.out / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print("\n" + "=" * 72)
    print(f"Done -> {args.out}")
    print("=" * 72)
    print("  train.h5 / val.h5 / test.h5   ragged token arrays + raw float offsets")
    print("  alphabet.json                 frozen char mapping")
    print("  tokenizer.json                frozen stroke bins")
    print("  writer_index.json             raw writer ID -> embedding index")
    print("  config.json                   every setting used, for reproducibility")
    print("\nModel sizing, straight from the data:")
    print(f"  char embedding : {alphabet.vocab_size}")
    print(f"  writer embedding: {len(writer_index) + 1}")
    for channel, size in tokenizer.vocab_sizes.items():
        print(f"  {channel} head       : {size}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
