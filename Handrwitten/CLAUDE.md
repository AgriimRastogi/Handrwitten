# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project is

A from-scratch stroke-level handwriting synthesis model, trained on IAM-OnDB, with the
goal of **beating** the 2013 reference implementation
([sjvasquez/handwriting-synthesis](https://github.com/sjvasquez/handwriting-synthesis),
Graves' *Generating Sequences with RNNs*) rather than reusing it. Output is **vector
strokes, not raster images** — pen-offset sequences rendered to SVG — because the target
is composable handwritten documents (and eventually a React Native app), where bitmap
word-stitching is the wrong tool.

`Readme.md` and `GLA_PROJECT_SYNOPSIS_REPORT.pdf` are **stale and describe a different
project** (a Rust/Sciter desktop app wrapping a prebuilt DLL). Do not treat them as
specifications for this codebase. `SETUP.md` is the accurate status document.

Current state: **Phase 0 (data pipeline) complete and verified.** Phase 1 is a PyTorch
`Dataset` over the HDF5 files, then a Graves-style LSTM baseline, then a Transformer
decoder with cross-attention to replace the hand-rolled Gaussian soft-window.

Note most of the tree is **not yet committed** — `git ls-files` shows only the two stale
docs.

## Commands

```bash
pip install -r requirements.txt
pip install -e .                    # makes `import hws` work; without it scripts fall
                                    # back to a sys.path insert

pytest -q                           # 58 tests
pytest -q tests/test_tokenizer.py   # one file
pytest -q -k jump_mask              # by name
pytest -q "tests/test_tokenizer.py::test_move_and_jump_tokens_occupy_disjoint_ranges"

python scripts/inspect_data.py -n 24 --page   # -> outputs/inspect/index.html
python scripts/prepare_data.py                # -> datas/processed/*.h5  (~2 min)
```

`inspect_data.py` renders raw vs preprocessed samples side by side. Run it after any
change to parsing, preprocessing or rendering — the pipeline was originally validated by
eye, and that is still the only check that catches a mirrored axis or a stroke drawn
through the air.

Useful flags: `--resample {step,count,none}`, `--step-factor`, `--variable-width`,
`--seed`. On `prepare_data.py`: `--limit N` (smoke test), `--bins`, `--jump-bins`,
`--unknown-writers {drop,train-only}`, `--no-dt`.

To see why the tokenizer is built the way it is, reproduce the failure it fixes:

```bash
python scripts/prepare_data.py --limit 2000 --jump-strategy quantile --out /tmp/ablate
# compare the RECONSTRUCTION QUALITY block against the asinh default
```

## Architecture

Strictly layered; the dependency arrow only points one way. `render` and `preprocess`
know nothing about IAM; `iam` knows nothing about the model. Swapping the LSTM for a
Transformer should require touching nothing in `hws/` except new model files.

```
hws/iam.py         raw IAM files  -> RawLine / LineRef
hws/preprocess.py  raw points     -> normalised offsets       (pure NumPy, stateless)
hws/tokenizer.py   offsets        <-> integer tokens
hws/alphabet.py    text           <-> integer indices
hws/render.py      coordinates    -> SVG
scripts/           inspect_data.py (look), prepare_data.py (pack)
```

### Array layout and coordinate conventions

Point and offset arrays are both `(N, 4)` and **share the same column order**, so the
constants `X, Y, PEN_UP, TIME` from `hws.iam` index correctly in both:

| | coordinates | offsets |
|---|---|---|
| `X` | x | dx |
| `Y` | y | dy |
| `PEN_UP` | pen lifts after this point | same (carried through) |
| `TIME` | seconds since line start | dt |

Always use the named constants. `points[:, 2]` at a call site three files away is
unreadable, and swapping two of them is a bug you will not see.

**`+y is up` everywhere in this codebase.** IAM records y growing downward (sensor at
`top_left`); `parse_stroke_xml` negates it at parse time. SVG's y also grows downward, so
the flip back happens in exactly one place — `render.path_data(y_flip=True)`. Nowhere
else. If output renders mirrored vertically, that function is the only suspect.

### Two non-obvious semantics

**`pen_up[i]` means the pen lifts *after* point i.** So the airborne movement between
strokes is row `i+1`, not row `i`. `preprocess.split_strokes` splits at
`where(pen_up)+1`, and `tokenizer.jump_mask` marks `offsets[:-1, PEN_UP] > 0.5` shifted by
one. Getting this off by one silently fits the jump bins on pen-down data, which is worse
than not splitting at all.

**Token vocabularies are concatenated, not mode-switched.** Each continuous channel has
two bin sets — pen-down moves and pen-up jumps — laid out as one vocabulary:

```
token <  n_move   -> a pen-down step
token >= n_move   -> a pen-up jump
```

So a model has **one head per channel**, every token id has exactly one meaning, and
`decode()` needs no mode argument. This exists because pen-up jumps are ~8% of steps but
accounted for **~97% of reconstruction error** with a single bin set: word spacing visibly
collapsed ("hear this" → "hearthis"). Jumps additionally use uniform-in-`asinh` bins
rather than quantile bins, because quantile binning gives the heavy tail one bin over an
unbounded range — and note that log/asinh-transforming before *quantile* binning changes
nothing, since quantiles are invariant under monotonic transforms. See the
`hws/tokenizer.py` module docstring; it records the measurements.

### Frozen artifacts

`datas/processed/alphabet.json` and `tokenizer.json` define the integer mappings a
checkpoint was trained against. **Refitting either one silently invalidates every
checkpoint** — the model keeps emitting token 137, that token just means a different
displacement now. Both are fitted on the **training split only** (fitting over the whole
corpus leaks the val distribution into preprocessing), and both are `@dataclass(frozen=True)`
for that reason. Size model heads from `tokenizer.vocab_sizes` and `alphabet.vocab_size`,
never from hardcoded constants.

### Deliberate hard failures — do not work around these

- `split_by_writer()` **raises** if any line lacks a `writer_id`, rather than falling back
  to a line-level split. A line-level split puts the same person's hand in train and test,
  so validation measures memorisation of 221 specific hands instead of generalisation —
  and it does so invisibly, with better-looking numbers.
- `Alphabet.encode()` **raises** on an unknown character unless you pass `errors="drop"`.
  The 2013 implementation mapped out-of-vocabulary characters to index 0, which is also
  the padding index, so it trained the model to treat `Q` and "nothing at all" as
  identical.

### Data quirks that cost time to rediscover

- IAM ascii files have an `OCR:` block (the printed prompt) and a `CSR:` block (what the
  writer actually wrote, typos preserved — `nomnating`). **Only `CSR:` matches the
  strokes.** Training against `OCR:` teaches the model that a stroke sequence spells a
  word it does not spell.
- Writer IDs live **only** in `original/*/strokes*.xml`, in `<General><Form writerID=...>`
  at line ~3 of a ~300 KB file. `load_writer_ids` uses `iterparse` with an early break for
  this reason; a full `ElementTree.parse` over 1,734 files takes 61 s instead of 6.6 s.
- ~0.25% of lines have **corrupt timestamps** (one has a −746 s step, with violations
  *inside* strokes). `iter_raw_lines(require_monotonic_time=True)` filters them by default
  because a negative dt makes pen velocity meaningless.
- A form whose stroke-file count disagrees with its CSR line count is dropped **whole** —
  a partial pairing is wrong for every line after the mismatch, and there is no way to
  tell which line is missing.

`SETUP.md` has the full measured corpus statistics (12,178 lines, 197 writers, 81 distinct
characters vs Graves' 75, jump magnitudes, etc.). Prefer reading those to re-deriving them.

### HDF5 layout

Ragged, not padded: sequences are concatenated per channel with an offsets index, so line
`i` is `tokens/dx[offsets[i]:offsets[i+1]]`. Padding to a fixed length would waste ~40% of
the file and bake a maximum length into the dataset; per-batch padding belongs in the
DataLoader's collate function. Each file also stores `raw` — the pre-tokenisation float
offsets — so an MDN output head can be tried later without re-running preprocessing.

Tokens are `int16` on disk; cast to `int64` in `__getitem__` for `nn.Embedding` and
`cross_entropy`.

## Conventions

- **`hws/render.py` is the single source of truth for strokes → SVG.** Training-time sample
  dumps, the inference API and the app must all import it. A second copy will drift, and
  you will spend days debugging a "model bug" that is a renderer bug. (The
  `calligrapher-ai` app in this workspace demonstrates the failure: its layout code
  computes each chunk's width and then never advances the x-cursor, so chunks stack.)
- **Console output must be ASCII.** The default Windows terminal is cp1252 and turns
  em-dashes into mojibake. Docstrings and comments may use Unicode freely.
- **Measure, don't assume, when a default depends on the data.** Every wrong default so far
  came from a plausible guess: an absolute resampling step assumed ~1 unit spacing when it
  is ~20 and varies 13–28 across writers; 256 tokenizer bins were assumed sufficient. Both
  were caught by rendering output and looking at it.
- **Quantisation error accumulates through `cumsum`**, so per-step error is close to
  meaningless. `StrokeTokenizer.reconstruction_error` reports drift in *position* space
  relative to the line's x-height. Calibrated by rendering: `<0.05` invisible, `~0.2`
  visible spacing damage, `>0.4` broken. Current val: median 0.032, p95 0.069, max 0.170.
- Tests target invariants that **fail silently** — round-trips, the `jump_mask` off-by-one,
  token range separation, cap enforcement — not "does it look like handwriting". That
  judgement is `inspect_data.py` plus your eyes. Determinism tests exist because
  non-reproducible bin edges or alphabet ordering break checkpoints without any error.
- `datas/` is gitignored. IAM-OnDB is licensed for **non-commercial research only**, which
  matters if the app is ever sold.

## Environment

Python 3.13, NumPy 2.x (avoid APIs removed in 2.0: `np.bool`, `ndarray.tostring`). GPU is
an **RTX 3050 6 GB Laptop, compute 8.6** — bf16, TF32 and FlashAttention-2 are available;
sustained clocks throttle on long runs. `torch` is intentionally absent from
`requirements.txt` because it needs a CUDA-specific index URL; see `SETUP.md`.
