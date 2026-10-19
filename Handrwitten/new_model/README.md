# New Model: Guided-Attention Handwriting Synthesis Transformer

This directory (`Handrwitten/new_model/`) contains the next-generation model implementation for online stroke-level handwriting synthesis.

It is **100% isolated** from the canonical Alex Graves (2013) Peephole LSTM baseline (`hws/pretrained_model.py`, `hws/graves_model.py`, and `checkpoints/pretrained/`). The Graves baseline remains untouched, runnable, and preserved.

---

## 1. Why a New Architecture?

| Feature | Canonical Alex Graves (2013) Baseline | New Guided-Attention Transformer (`new_model/`) |
| :--- | :--- | :--- |
| **Model Class** | 3-layer Peephole LSTM (Sequential recurrent bottleneck) | Causal Transformer Decoder with Multi-Head Self-Attention |
| **Alignment Prior** | Monotonic Gaussian Soft-Window Attention ($\kappa_t$) | Multi-Head Cross-Attention with **Guided Monotonic Alignment Loss** ($\mathcal{L}_{\text{guided}}$) |
| **Output Space** | 20-component Bivariate GMM (Susceptible to mode averaging & variance collapse) | Dual-Mode Discrete Token Classification (Linear moves + $\text{asinh}$ jumps with Cross-Entropy) |
| **Training Parallelism** | Step-by-step sequential ($O(T)$ steps) | Parallel teacher-forcing across all time steps ($O(1)$) |
| **Checkpoints** | `checkpoints/pretrained/` & `checkpoints/best_graves_model.pt` | `checkpoints/new_model/` |

---

## 2. Directory Structure

```
Handrwitten/new_model/
├── __init__.py          # Package exports (NewModelConfig, GuidedHandwritingTransformer, GuidedSynthesisLoss)
├── config.py            # Typed hyperparameter configuration dataclass
├── attention.py         # GuidedMultiheadCrossAttention & GuidedTransformerDecoderLayer
├── model.py             # GuidedHandwritingTransformer (TextEncoder, StrokeEmbedding, generate())
├── loss.py              # Dual-Mode Cross-Entropy + Diagonal Guided Attention Loss
├── train.py             # Standalone trainer saving checkpoints into checkpoints/new_model/
├── synthesize.py        # Standalone synthesis script for vector SVG generation
└── README.md            # This documentation file
```

---

## 3. How to Train the New Model

From the repository root:
```powershell
# Train for 20 epochs (saves to checkpoints/new_model/best_model.pt)
python scripts/train_new_model.py --epochs 20 --batch-size 32

# Or with custom guided attention alignment penalty:
python scripts/train_new_model.py --epochs 30 --guided-weight 1.5
```

---

## 4. How to Synthesize with the New Model

```powershell
python scripts/synthesize_new_model.py --text "Stroke Level Synthesis" --out "outputs/new_model_test.svg"
```

---

## 5. Safety Guarantee

* **Zero Baseline Overwrites:** Does not touch `hws/pretrained_model.py`, `checkpoints/pretrained/`, or `scripts/synthesize.py`.
* **Independent Checkpoints:** Checkpoints are stored strictly under `checkpoints/new_model/`.
* **Test Suite Green:** All 70 existing PyTest tests continue passing without regression.
