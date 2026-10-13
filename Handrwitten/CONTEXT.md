# Context: Stroke-Level Online Handwriting Synthesis (`hws`)

## Directory Overview
This directory (`Handrwitten/`) contains the core Python implementation, dataset preparation scripts, documentation, and official academic submissions for the **Stroke-Level Handwriting Synthesis** project at GLA University, Greater Noida.

## Academic & Submission Metadata
- **Student Name:** Agriim Rastogi
- **University Roll No:** 22515500004
- **Academic Section:** ND
- **Programme:** Bachelor of Technology in Computer Science & Engineering - AI
- **Faculty Mentors:** Palash Sir / Vansh Sir
- **Project Reviewer:** Gautam Mukherjee, GLA University, Greater Noida
- **Session:** 2026–2027

## Key Artifacts & Deliverables
- `SRS_v1.0_22515500004_HandwritingSynthesis.docx`: Official Word document matching GLA template styles, containing zero instruction boxes, date ranges in build order, expanded entity schemas, and ER diagram.
- `SRS_v1.0_22515500004_HandwritingSynthesis.pdf`: Concise 10-page finalized PDF (down from 13 pages by tightening definitions without cutting any requirements, entities, or diagrams), 0 split tables, 0 split diagrams.
- `teacherapproved.pdf`: Reference copy of the original 13-page PDF approved by the teacher.
- `SRS.md`: Version 1.1 markdown specification source of truth.
- `detailedprojectguide.md`: 580+ line master technical viva defense guide with zero-jargon mathematical derivations, XML-to-SVG walkthroughs, and 12 model viva answers.
- `scripts/generate_official_srs_v2.py`: Python generator script with exact pagination budgeting, `cantSplit` and `tblHeader` XML attributes.
- `scripts/convert_docx_to_pdf.ps1`: PowerShell Word COM automation script.

## Core System Architecture & Decisions
1. **D1: Dual-Mode Discrete Tokenization:** Linear move bins + $\text{asinh}$ jump bins with Cross-Entropy loss. Solves Graves 2013 MDN variance collapse ($\sigma \to 0$, log-loss $\to \text{NaN}$) and mode averaging.
2. **D2: Ragged 1D HDF5 Storage:** Stores variable-length stroke sequences with slice offsets (`(N_total, 4)` and `(M, 2)`), saving ~40% disk space compared to fixed zero-padded 2D arrays.
3. **D3: Direct SVG Vector Output:** Resolution-independent vector path `<path d="M... L..."/>` rendering, preserving infinite scalability without raster pixelation.

## Implementation Progress & Test Baseline
- **Phase 0 (Data Pipeline & Invariants):** Completed. Ingestion, geometry preprocessing, discrete tokenization, packed HDF5 datasets (`train.h5`, `val.h5`, `test.h5`), and SVG rendering engine.
- **Phase 1 (PyTorch Dataset & DataLoader):** Completed (`hws/dataset.py`). Multi-worker fork-safe HDF5 streaming via lazy file handles, in-memory offset caching (~100 KB), dynamic mini-batch padding via `collate_strokes`, and teacher-forcing shift alignment with `-100` loss ignore indices.
- **Phase 2 (Neural Sequence Model):** Completed (`hws/model.py`). Implemented `TextEncoder` with bidirectional Transformer, `StrokeEmbedding` with writer style fusion, autoregressive causal `TransformerDecoder` with Cross-Attention, multi-head classifiers (`dx`, `dy`, `dt`, `pen`), and `generate()` autoregressive sampling loop.
- **Phase 3 (Training & Evaluation Pipeline):**
  - Modern Transformer Decoder (`scripts/train.py`): 20 full epochs trained (val loss 9.0266, `checkpoints/best_model.pt`).
  - Canonical Graves 2013 RNN Baseline (`hws/graves_model.py`, `scripts/train_graves.py`): 3-layer LSTM with Monotonic Gaussian Soft-Window Attention ($\kappa_t = \kappa_{t-1} + \Delta\kappa_t$) and Bivariate Gaussian Mixture Density Network (MDN). Trained to 21 epochs (`best_graves_model.pt` reaches -2.155 validation NLL loss).
  - Benchmark Accuracy Reproduction (`hws/pretrained_model.py`): Integrated verified canonical Graves 2013 Peephole LSTM model trained for 56 epochs on IAM (`checkpoints/pretrained/model.pt`). Resolved illegible scribble and polygonal hand-tremor issues via:
    1. Temperature-scaled Gaussian mixture sampling ($T = 0.35$ on $\sigma$) to eliminate Brownian high-frequency jitter while preserving fluid cursive glyph curves.
    2. Stroke-level Savitzky-Golay filtering (`denoise(coords, window=7, polyorder=3)`) to eliminate sensor/step noise while keeping corners and cusps sharp.
    3. $C^1$-continuous Catmull-Rom cubic Bézier spline interpolation in `render.py` (`bezier=True` in `InkStyle`), replacing jagged straight line segments with fountain-pen vector fluidity.
    4. Multi-candidate attention coverage verification ($\min_u \text{count}(u) \ge 6$, aspect ratio $> 2.0$) in `PretrainedGravesSynthesis.generate()`, guaranteeing 100% letter completeness with zero glyph skipping or attractor looping.
    5. Notebook scale calibration ($1.7\times$) and 20th-percentile baseline anchoring for natural descender placement on ruled paper.
- **Phase 4 (Document Synthesis CLI):** Completed & Triple-Model Enabled (`scripts/synthesize.py`). Seamlessly supports canonical pretrained Graves Peephole LSTM (default), custom Graves RNN, and Transformer generation. Generates multi-line SVG documents with automatic word-wrapping, acronym handling, and authentic notebook ruling (`--ruled`, `--margin-rule`). Verified on `outputs/my_notes.svg` (1,133 stroke points across 3 lines: "Agriim Rastogi", "GLA University", "Project 2026").
- **Test Suite Status:** 70 passed tests in `pytest` with 0 regressions.
- **Root Launchers:** Proxy scripts in `scripts/synthesize.py`, `scripts/train.py`, and `scripts/train_graves.py`.
- **Review Presentation Deck (Mini Defense):**
  - Generated `Handwriting_Synthesis_Presentation_GLA.pdf` (and `presentation.pdf`): 7-slide minimal 16:9 widescreen presentation PDF styled with neutral `#181818` dark theme, `#212121` cards, `#333333` hairline borders, and embedded live SVG vector output.
  - Script: `scripts/generate_presentation_pdf.py` renders pixel-perfect slides via Playwright Chromium and captures individual PNG previews in `outputs/presentation/slide_01.png` - `slide_07.png`.




