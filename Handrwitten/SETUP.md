# Setup & reading guide

## Install

```bash
cd "AR-Handwriting Synthesis/Handrwitten"
pip install -r requirements.txt
pip install -e .            # makes `import hws` work from anywhere
```

Verify:

```bash
pytest -q                   # expect 36 passed
```

## Look at the data (do this first)

```bash
python scripts/inspect_data.py -n 24 --page
# then open outputs/inspect/index.html
```

Check the **right-hand** column of each pair. Legible handwriting means the
parser, coordinate conventions, preprocessing and renderer are all correct, and
any bug you hit later is genuinely in the model. That certainty is worth the
five minutes.

Useful variants:

```bash
python scripts/inspect_data.py --step-factor 3.0     # more aggressive resampling
python scripts/inspect_data.py --resample none       # native sensor sampling
python scripts/inspect_data.py --variable-width      # pen-speed stroke width
python scripts/inspect_data.py --seed 5              # different random lines
```

## PyTorch (Phase 1, not needed yet)

Your GPU is an **RTX 3050 6 GB Laptop, compute capability 8.6 (Ampere)**. That
gets you bf16, TF32 and FlashAttention-2 — everything the plan needs.

Install with a CUDA build, not the default CPU wheel:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu126
```

Check https://pytorch.org for the current CUDA index if that one 404s, then:

```python
import torch
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))
print(torch.cuda.get_device_capability(0))   # expect (8, 6)
```

Note it is a *laptop* GPU, so sustained clocks drop on long runs — expect ~65%
of desktop 3050 throughput. Fine for this model. Keep Kaggle (30 h/week, 2×T4
16 GB) in reserve for final long runs.

## Data layout

The code expects:

```
datas/raw/iam-ondb/
├── ascii/          transcriptions      (from ascii-all.tar)
├── lineStrokes/    pen trajectories    (from lineStrokes-all.tar)
└── original/       writer IDs          (from original-all.tar)
```

Override with `--data-root`. `datas/archives/` holds the source tarballs and can
be deleted to reclaim ~426 MB once you trust the extraction.

**Licence:** IAM-OnDB is free for non-commercial research only. Fine for
coursework. If the app ever ships commercially you need your own data.

## Build the training dataset

```bash
python scripts/prepare_data.py
# writes datas/processed/{train,val,test}.h5 + alphabet.json + tokenizer.json
```

Takes ~2 minutes. Read the `RECONSTRUCTION QUALITY` block it prints — that is the
gate on whether tokenisation is losing information you care about.

Ablate the tokenizer design if you want to see why it is built this way:

```bash
# reproduce the failure mode the asinh jump bins exist to fix
python scripts/prepare_data.py --bins 256 --jump-bins 256 --out /tmp/quantile-only
```

## Reading order

The modules are written to be read in this order — each only depends on the ones
above it.

| # | File | What to take from it |
|---|------|----------------------|
| 1 | `hws/iam.py` | The data format, and why only the `CSR:` block is usable. Read the module docstring first. |
| 2 | `hws/preprocess.py` | The five pipeline steps and why each exists. The resampling docstrings explain a real trade-off you have to decide. |
| 3 | `hws/render.py` | Coordinate systems (+y up vs SVG's +y down), and how a document page is composed from lines. |
| 4 | `hws/alphabet.py` | Why the character mapping must be frozen, and the padding-index bug in the 2013 implementation. |
| 5 | `hws/tokenizer.py` | Why strokes are discretised instead of using an MDN, and the two empirical findings that shaped it (two-mode bins, asinh tails). The most interesting file here. |
| 6 | `scripts/inspect_data.py` | How the pieces compose. |
| 7 | `scripts/prepare_data.py` | The full pipeline end to end, including what "fit on train only" costs. |
| 8 | `tests/` | The invariants that break *silently*. Worth reading as documentation. |

## Measured facts about this corpus

From the scripts, so you don't have to rediscover them:

| | |
|---|---|
| Lines indexed | 12,178 (from 1,907 forms) |
| Dropped | 177 forms with no strokes, 2 with a line-count mismatch |
| Corrupt timestamps | ~0.25% of lines (one has a −746 s step) |
| Writer coverage | 10,859 / 12,178 = **89.2%** |
| Distinct writers | **197** |
| Lines per writer | median 55, min 7, max 78 |
| Distinct characters | **81** in the corpus, 77 + PAD in the train split |
| Characters per line | median 29, p95 41, max 64 |
| Raw points per line | median ~620, min 445, p95 857 |
| Strokes per line | median 28, max 46 |
| Median point spacing | ~20 sensor units (13–28 across writers) |
| After `--step-factor 2.0` | median 342 points, **1.84× compression** |
| Pen-up jumps | ~8% of steps, but **~97% of tokenisation error** |
| Jump magnitude `\|dx\|` | median 3, p99 20, p99.9 37, **max 77** |

Three consequences worth knowing:

- **81 characters, not the 75 Graves used.** The ones he omitted really do occur:
  `Q`(19), `Z`(7), `X`(4), `+`(8), `/`(6), `[`(3), `]`(3), `*`(1), `&`(1). His
  implementation mapped them to index 0 — the same slot as padding.
- **1,319 lines have no writer ID.** Default is to drop them; use
  `--unknown-writers train-only` to keep them out of val/test instead.
  `split_by_writer()` refuses to run without writer IDs rather than silently
  falling back to a line-level split, which would inflate scores invisibly.
- **Pen-up jumps dominate tokenisation error.** Hence two bin sets per channel,
  with `asinh`-uniform bins for jumps. See `hws/tokenizer.py`.

## Final dataset

```
datas/processed/
├── train.h5            8,735 lines | 3,008,689 points | 156 writers | 32.9 MB
├── val.h5              1,028 lines |   357,427 points |  20 writers |  4.0 MB
├── test.h5             1,093 lines |   364,936 points |  20 writers |  4.1 MB
├── alphabet.json       frozen char mapping (78 incl. PAD)
├── tokenizer.json      frozen stroke bins
├── writer_index.json   raw writer ID -> embedding index (197 incl. unknown)
└── config.json         every setting used
```

Reconstruction drift on val: **median 0.032, p95 0.069, max 0.170** — every line
inside the "invisible" band.

Model sizing, straight from the data:

| Component | Size |
|---|---|
| char embedding | 78 |
| writer embedding | 197 |
| `dx` head | 512 (256 move + 256 jump) |
| `dy` head | 512 |
| `dt` head | 266 |
| `pen` head | 2 |

## Status

**Phase 0 complete** — parsing, preprocessing, rendering, tokenisation, packing,
58 tests, and visual verification against real data. Any bug from here is in the
model, not the data.

**Next** — the LSTM baseline (`hws/models/baseline_lstm.py`), then a
`torch.utils.data.Dataset` over these HDF5 files, then training.

