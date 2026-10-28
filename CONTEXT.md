# Context: AR-Handwriting Synthesis Workspace

## Workspace Overview
Root workspace directory for Agriim Rastogi's B.Tech CSE (AI) Capstone Project at GLA University, Greater Noida: **Stroke-Level Handwriting Synthesis: A Vector-Output Generative Model for Composable Handwritten Documents**.

## Active Submission Files
- `SRS_v1.0_22515500004_HandwritingSynthesis.pdf`: Concise 10-page finalized PDF (down from 13 pages by tightening definitions without cutting any requirements, entities, or diagrams), 0 split tables, 0 split diagrams.
- `teacherapproved.pdf`: Reference copy of the original 13-page PDF approved by the teacher.
- `SRS_v1.0_22515500004_HandwritingSynthesis.docx`: Source Word document styled to GLA ML Projects template standards.
- `SRS_Template_v1.0_ML_Projects-2.docx`: Reference university template.
- `issues.md`: Detailed technical report on Graves LSTM failure modes (attractor loops, stalls, vertical drift), attempted mitigations, and alternative generative architectures (Mamba, Diffusion, Duration TTS-style models).
- `Handrwitten/`: Core codebase, scripts, tests, datasets, and guides.

## Review & Compliance Checklist
- [x] Cover Page updated with Version 1.1, Agriim Rastogi (Roll: 22515500004, Sec: ND), Mentors Palash Sir / Vansh Sir, Reviewer Gautam Mukherjee.
- [x] Revision history contains Version 1.0 (05-09-2026) and Version 1.1 (11-09-2026).
- [x] Section 3 empirical dataset statistics (12,178 lines, 1,907 forms, 3.73M points, 92.1%/7.9% move/jump balance).
- [x] Section 4 Architecture: Figure 1A (Offline) and Figure 1B (Online) intact on Page 5.
- [x] Section 4.1 System Components Table: 7 modules intact on Page 6.
- [x] Section 5 & 6: Functional (FR-1.1 to 2.5) and Non-Functional (NFR-1 to 4) requirements intact on Page 7.
- [x] Section 7: Build Plan & Sprints (Sprint I to Final Viva with explicit date ranges) on Page 8.
- [x] Section 8: Five fully expanded entity schemas (Pages 9-11) and complete ASCII ER Diagram (Page 12).
- [x] Section 9 & 10: Decisions D1-D3 and Acceptance Criteria AC-1 to AC-4 on Page 13.
- [x] Section 11 & Sign-Off: Out of scope items and institutional sign-off block on Page 14.
- [x] Zero unicode replacement characters, zero split tables, zero split diagrams.

## Execution Environment & Training
- Root launchers `scripts/synthesize.py` and `scripts/train.py` use `runpy` to execute core logic cleanly from root.
- PyTorch 2.14.0+cu126 enabled on NVIDIA GeForce RTX 3050 6GB Laptop GPU with AMP mixed precision.
- Unit test suite: 70/70 tests passing.
- **Model Training & Accuracy Baseline Reproduction:**
  - Modern Transformer Decoder: 20 epochs trained (`checkpoints/best_model.pt`).
  - Canonical Graves 2013 RNN Baseline: Reproduced and trained to 21 epochs (`best_graves_model.pt` reaching -2.155 NLL loss).
  - Pretrained Canonical Graves Benchmark: Integrated verified 56-epoch IAM Peephole LSTM checkpoint (`checkpoints/pretrained/model.pt`).
  - **High-Fidelity Calligraphic Rendering & Accuracy Matching:**
    - Solved irregular hand tremors and polygonal jitter via temperature scaling ($T = 0.35$) on mixture standard deviations and stroke-level Savitzky-Golay filtering (`denoise(coords, window=7, polyorder=3)`).
    - Integrated $C^1$-continuous Catmull-Rom cubic Bézier spline interpolation (`C` path commands) in `render.py`, turning raw polylines into calligraphic fountain-pen ink strokes.
    - Implemented attention-verified candidate selection in `PretrainedGravesSynthesis.generate()`, tracking character soft-window alignment ($\min_u \text{count}(u) \ge 6$) and aspect ratio ($> 2.0$) to guarantee complete letter coverage with zero glyph skipping or attractor looping.
    - Calibrated notebook geometry with 20th-percentile baseline anchoring and $1.7\times$ scale fitting onto 42-48 unit ruled notebook lines.
- **Document Synthesis CLI:** `scripts/synthesize.py` supports canonical Graves Peephole LSTM (default), custom Graves RNN, and Transformer. Generates multi-line SVG documents with automatic word-wrapping, acronym handling, and authentic notebook ruling (`--ruled`, `--margin-rule`). Verified on `outputs/my_notes.svg` (1,133 stroke points across 3 lines: "Agriim Rastogi", "GLA University", "Project 2026").
- Unit test suite: 70/70 tests passing with 0 regressions.
- **Review Presentation Deck (Mini Defense):**
  - Generated `Handwriting_Synthesis_Presentation_GLA.pdf` (and `presentation.pdf`): 7-slide minimal 16:9 widescreen presentation PDF styled with neutral `#181818` dark theme, `#212121` cards, `#333333` hairline borders, and embedded live SVG vector output.
  - Script: `scripts/generate_presentation_pdf.py` renders pixel-perfect slides via Playwright Chromium and captures individual PNG previews in `outputs/presentation/slide_01.png` - `slide_07.png`.
- **5-Slide Focused Presentation Deck (v2):**
  - Generated `Handwriting_Synthesis_Presentation_v2.pdf` (and `outputs/presentation_v2/Handwriting_Synthesis_v2.pdf`): 5-slide streamlined 16:9 presentation deck following exact defense structure:
    1. Problem Statement & Project Description (Static fonts vs. online stroke synthesis, candidate details)
    2. Real-World Applications (Featured Swiggy/Zomato personalized customer letters, accessibility, OCR training data)
    3. Verified Synthesis Outputs & Quality Engineering (Embedded live ruled notebook SVG, candidate signature, institution heading, sentence synthesis, temperature scaling & Catmull-Rom splines)
    4. Dual-Track Neural Sequence Architecture (Canonical Graves 2013 RNN vs. Autoregressive Transformer with monotonic soft-window math & pipeline flow)
    5. Key References & Future Scope (Academic citations, primary future milestone: cross-platform mobile app with OCR Document → Text → Handwritten Document pipeline, style transfer, multi-script)
  - Generator: `scripts/generate_presentation_v2.py` with standalone HTML template `outputs/presentation_v2/presentation_v2.html` and PNG slide previews `slide_01.png` - `slide_05.png`.




