# Stroke-Level Online Handwriting Synthesis (`hws`)

<p align="center">
  <strong>A Vector-Output Generative Sequence Architecture for Composable Handwritten Documents</strong>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.14%2Bcu126-EE4C2C?logo=pytorch&logoColor=white" alt="PyTorch" />
  <img src="https://img.shields.io/badge/Tests-70%2F70%20Passed%20(100%25)-brightgreen" alt="Tests" />
  <img src="https://img.shields.io/badge/Output-Resolution--Independent%20SVG-00C7B7?logo=svg&logoColor=white" alt="SVG" />
  <img src="https://img.shields.io/badge/Dataset-IAM--OnDB%20(12.1k%20Lines)-blue" alt="IAM-OnDB" />
  <img src="https://img.shields.io/badge/Academic%20Session-2026--2027-purple" alt="Session" />
</p>

---

## 📌 Executive Overview

**Stroke-Level Online Handwriting Synthesis (`hws`)** is a generative deep learning pipeline designed to synthesize authentic, resolution-independent vector handwriting directly from arbitrary text strings. 

Instead of generating fixed-size 2D pixel grids (like raster GANs or Diffusion models) or repeating static glyph outlines (like TTF/OTF fonts), this system models handwriting in its fundamental physical form: **a continuous time-series trajectory of pen coordinates $(\Delta x_t, \Delta y_t, \Delta t)$ and binary pen-up/pen-down state transitions ($p_t$)**.

The synthesized trajectory is dynamically rendered into scalable vector graphics (**SVG**) using **$C^1$-continuous Catmull-Rom cubic Bézier splines**, producing fountain-pen calligraphy aligned naturally to authentic ruled notebook geometry.

---

## 🎓 Academic Credentials

| Attribute | Details |
| :--- | :--- |
| **Candidate** | **Agriim Rastogi** |
| **Roll Number** | `22515500004` |
| **Academic Section** | Section `ND` |
| **Programme** | Bachelor of Technology in Computer Science & Engineering (Artificial Intelligence) |
| **Institution** | GLA University, Greater Noida Campus |
| **Faculty Mentors** | Palash Sir / Vansh Sir |
| **Project Reviewer** | Gautam Mukherjee |
| **Session** | 2026–2027 |
| **Official Documentation** | [10-Page Approved University SRS (PDF)](SRS_v1.0_22515500004_HandwritingSynthesis.pdf) • [Review Presentation Deck (PDF)](Handwriting_Synthesis_Presentation_GLA.pdf) |

---

## 🔍 Why Online Vector Synthesis?

| Parameter | Static Digital Fonts (`.ttf` / `.otf`) | 2D Raster Models (Diffusion / GANs) | **Stroke-Level Vector (`hws`) [Ours]** |
| :--- | :--- | :--- | :--- |
| **Representation** | Fixed geometric glyph contours | Fixed $256 \times 256$ or $512 \times 512$ pixel grid | **Continuous pen trajectory $(\Delta x, \Delta y, \Delta t, p)$** |
| **Cursive Ligatures** | ❌ None (letters sit side-by-side) | ⚠️ Blurry pixel transitions | **✅ Organic physical connecting loops** |
| **Scalability** | ✅ Vector outline | ❌ Pixelates and blurs when zoomed | **✅ Resolution-independent SVG paths (`<path d="M... C..."/>`)** |
| **Inference Latency** | Instant (lookup table) | Slow ($5\text{s} - 30\text{s}$ multi-step denoising) | **Fast ($< 3.2\text{s}$ for a full multi-line notebook page)** |
| **Editability** | Non-editable | Non-editable baked bitmap | **Individual strokes can be recolored, resized, and animated** |

---

## ⚙️ Core Technical Innovations

### 1. Decision 1 (D1): Dual-Mode Discrete Tokenization
Continuous Mixture Density Networks (MDNs) predicting bivariate Gaussian distributions often suffer from **variance collapse** ($\sigma \to 0$, causing log-likelihood loss to blow up to $\text{NaN}$) and **mode averaging** (drawing straight lines between alternative cursive ligatures). We partition velocity into two distinct regimes:
* **Moves (Pen Down, 92.1% of points):** Uniform linear spatial bins providing fine-grained precision for cursive letterforms.
* **Jumps (Pen Up, 7.9% of points):** Heavy-tailed inverse hyperbolic sine ($\text{asinh}$) non-linear bins:
  $$k = \operatorname{sgn}(\Delta x) \cdot \left\lfloor \operatorname{asinh}\left(\frac{|\Delta x|}{s}\right) \cdot N \right\rfloor$$
* **Result:** Formulates trajectory generation as bounded, numerically stable **Cross-Entropy classification**.

### 2. Spatial Arc-Length Resampling ($ds = 2.0$)
Digitizer hardware captures stroke coordinates at non-uniform temporal intervals, causing dense clusters during slow writing and sparse gaps during rapid strokes.
* Normalizes the trajectory to a uniform spatial step $ds = 2.0$ units.
* Achieves a **$1.84\times$ sequence length compression** (from ~620 down to ~342 points per line) with zero perceptual degradation.

### 3. Decision 2 (D2): Ragged 1D HDF5 Dataset Storage
Instead of padding variable-length sentences into a giant 2D matrix with thousands of empty zeros:
* Packs all 3.73 million coordinates contiguously in a flat 1D array `(N_total, 4)`.
* Uses a lightweight slice lookup table `(M, 2)` where row $i$ points to `[start_idx : end_idx]`.
* **Saves ~40% disk storage** and enables zero-copy lazy streaming into PyTorch DataLoader workers.

### 4. Visual Quality Breakthrough (Eliminating Brownian Tremors)
Raw autoregressive stroke sampling suffers from high-frequency jitter resembling muscle tremors. We resolved this through 4 integrated techniques:
1. **Temperature-Scaled Mixture Covariance ($T = 0.35$):** Scaled covariance $\sigma_{\text{scaled}} = \sigma \cdot T^{1.2}$ concentrates generation around letterform modal skeletons.
2. **$C^1$-Continuous Catmull-Rom Bézier Splines:** Converts jagged segmented polylines (`L`) into smooth cubic Bézier curves (`C` commands in SVG).
3. **Savitzky-Golay Denoising (`window=7, polyorder=3`):** Strips sensor step noise while strictly preserving sharp angular stroke cusps ('v', 'w', 't').
4. **Attention-Verified Sampling:** Enforces character soft-window dwell coverage ($\min_u \text{count}(u) \ge 6$) and aspect ratio ($> 2.0$) to guarantee zero dropped characters or attractor stalls.

---

## 🧠 Dual-Track Neural Architecture

The codebase supports two distinct generative modeling paradigms:

```
Arbitrary Text Sequence c ──► Soft-Window Monotonic Attention (κ_t) ──► Recurrent / Transformer State ──► MDN / Discrete Heads ──► Catmull-Rom SVG Engine
```

### Track A: Canonical Alex Graves (2013) Peephole LSTM (Baseline)
* **Architecture:** 3-layer recurrent network with cell-to-gate peephole connections.
* **Monotonic Soft-Window Attention:**
  $$\kappa_t^k = \kappa_{t-1}^k + \exp(\hat{\kappa}_t^k), \quad w_t(u) = \sum_{k=1}^K \alpha_t^k \exp\left(-\beta_t^k (\kappa_t^k - u)^2\right)$$
  *(Monotonicity constraint guarantees the attention window only moves forward across text).*
* **Output Head:** 20-component Bivariate Gaussian Mixture Density Network (MDN) + Bernoulli pen lift.
* **Benchmark Status:** Trained on IAM-OnDB to **-2.155 NLL validation loss**, fully matching canonical published benchmarks.

### Track B: Modern Autoregressive Transformer Decoder (Custom)
* **Architecture:** Bidirectional Text Encoder + Causal Transformer Decoder with Multi-Head Cross-Attention.
* **Style Conditioning:** Incorporates learned writer style embeddings ($\mathbf{e}_{\text{style}}$) allowing distinct personal aesthetics across lines.
* **Training Status:** 20 full epochs trained (`checkpoints/best_model.pt`).

---

## 📊 Dataset: IAM Online Handwriting Database (IAM-OnDB)

* **Lines:** 12,178 handwritten lines
* **Forms:** 1,907 forms from ~500 distinct human writers
* **Total Points:** 3,733,404 raw temporal coordinate waypoints
* **Velocity Distribution:** 92.1% moves (on-paper writing) vs. 7.9% airborne jumps (pen lift transitions)
* **Splits:** Writer-disjoint Train, Validation, and Test partitions preventing data leakage

---

## 🚀 Quickstart & Usage

### 1. Installation
Clone the repository and install the dependencies:
```bash
git clone https://github.com/AgriimRastogi/Handrwitten.git
cd Handrwitten
pip install -r Handrwitten/requirements.txt
```

### 2. Synthesize Handwritten Documents
Generate multi-line handwritten vector documents with authentic notebook rulings:
```bash
# Basic single-line synthesis
python scripts/synthesize.py --text "Stroke-Level Online Handwriting Synthesis" --out outputs/sample.svg

# Multi-line ruled notebook page synthesis
python scripts/synthesize.py --text "GLA University Greater Noida\nDeep Recurrent Neural Networks\nCapstone Project 2026" --ruled --margin-rule --out outputs/notebook.svg
```

### 3. Run Unit Test Suite
Verify that all 70 tests pass across geometry, tokenization, dataset streaming, and rendering:
```bash
$env:PYTHONPATH="."  # On PowerShell (or export PYTHONPATH=. on Bash)
pytest Handrwitten/tests
# ======================= 70 passed in 10.60s =======================
```

### 4. Generate Academic Presentation Deck
Compile the pixel-perfect 16:9 widescreen presentation PDF:
```bash
python scripts/generate_presentation_pdf.py
# Outputs: Handwriting_Synthesis_Presentation_GLA.pdf & outputs/presentation/slide_*.png
```

---

## 📂 Repository Structure

```
AR-Handwriting Synthesis/
├── .gitignore                                # Excludes large datasets, checkpoints, and caches
├── CONTEXT.md                                # Workspace status & execution context
├── README.md                                 # Master project documentation
├── SRS_v1.0_22515500004_HandwritingSynthesis.pdf # Official 10-page signed GLA University SRS
├── Handwriting_Synthesis_Presentation_GLA.pdf # 7-slide minimal 16:9 review presentation PDF
├── sample_preview.svg                        # Live calligraphic SVG vector sample
│
├── Handrwitten/                              # Core engineering codebase
│   ├── hws/                                  # Core Python library
│   │   ├── alphabet.py                       # Vocabulary and character mappings
│   │   ├── dataset.py                        # Fork-safe HDF5 PyTorch DataLoader
│   │   ├── graves_model.py                   # Canonical Graves 2013 Peephole LSTM & Attention
│   │   ├── iam.py                            # IAM-OnDB XML parsing and constants
│   │   ├── model.py                          # Modern Transformer Decoder with Cross-Attention
│   │   ├── preprocess.py                     # Arc-length resampling & normalization
│   │   ├── pretrained_model.py               # Pretrained Graves synthesis & attention verification
│   │   ├── render.py                         # C¹ Catmull-Rom cubic Bézier SVG vector renderer
│   │   └── tokenizer.py                      # Dual-Mode discrete stroke tokenizer
│   ├── tests/                                # 70 unit tests (geometry, tokenization, loaders, smoke)
│   ├── pyproject.toml                        # Project metadata and test configuration
│   └── requirements.txt                      # Python dependencies
│
└── scripts/                                  # Root executable runners
    ├── generate_presentation_pdf.py          # Playwright-based presentation PDF generator
    ├── generate_showcase.py                  # High-resolution showcase generator
    ├── synthesize.py                         # CLI document synthesis interface
    ├── train.py                              # Transformer training runner
    └── train_graves.py                       # Graves LSTM training runner
```

---

## 🗺️ Project Roadmap & Milestones

- [x] **Phase 0 — Data Pipeline & Geometry:** Ingestion of 12,178 IAM-OnDB lines, spatial resampling, and ragged 1D HDF5 storage.
- [x] **Phase 1 — PyTorch Streaming DataLoader:** Multi-worker lazy streaming with offset cache.
- [x] **Phase 2 — Dual-Mode Tokenization:** Non-linear $\text{asinh}$ jumps + linear moves with Cross-Entropy loss.
- [x] **Phase 3 — Baseline Accuracy Reproduction:** Canonical Graves 2013 Peephole LSTM (-2.155 NLL) + 20-epoch Transformer.
- [x] **Phase 4 — High-Fidelity Calligraphy:** Temperature scaling ($T=0.35$), $C^1$ Catmull-Rom splines, and Savitzky-Golay filtering.
- [x] **Phase 5 — Academic Compliance:** 10-page GLA SRS approved and 7-slide minimal dark-mode review deck generated.
- [ ] **Phase 6 — Real-Time Stylus Web Canvas:** In-browser touch/stylus capture for native online stroke recording.
- [ ] **Phase 7 — Controllable Style Knobs:** Parametric sliders for Slant ($\theta$), Pen Pressure, and Word Spacing.
- [ ] **Phase 8 — Quantitative Recognition Benchmark:** Character Error Rate (CER) evaluation using Microsoft TrOCR.
- [ ] **Phase 9 — Final Capstone Thesis & Viva Voce:** Comprehensive thesis submission and oral defense.

---

## 📜 License & Citation

Developed by **Agriim Rastogi** for the B.Tech CSE (AI) Capstone Project at GLA University, Greater Noida.  
Released under the [MIT License](LICENSE).
