# Technical Report: Architectural Challenges, Failure Modes, and Alternative Paradigms in Online Handwriting Synthesis

---

## 1. Executive Objective

The objective of this project is to synthesize high-fidelity, legitimate human-quality online handwriting as continuous vector stroke sequences $(\Delta x_t, \Delta y_t, p_t)$ conditioned on arbitrary text strings.

The primary architectural benchmark is the seminal model proposed by **Alex Graves (2013)**: a 3-layer recurrent neural network with Peephole Long Short-Term Memory (LSTM) cells, continuous Monotonic Gaussian Soft-Window Attention, and a Bivariate Gaussian Mixture Density Network (MDN) output head.

While the Graves baseline serves as the historic gold standard for realistic cursive motor dynamics, it suffers from severe systemic failure modes (attractor loops, attention stalls, vertical drift, and runaway deflection). Our objective is two-fold:
1. **Analyze and resolve the fundamental failure modes** of the recurrent Graves LSTM architecture.
2. **Evaluate modern alternative generative architectures** (e.g., Transformers, State Space Models / Mamba, Diffusion / Flow Matching, and Non-Autoregressive Duration-based Models) to determine whether a more suitable paradigm exists for online stroke synthesis.

---

## 2. Why the Graves LSTM Architecture Was the Historical Standard

Online handwriting differs fundamentally from offline handwriting images or discrete text generation:

$$\mathbf{x}_t = (\Delta x_t, \Delta y_t, p_t) \in \mathbb{R}^2 \times \{0, 1\}$$

1. **Continuous Physical Inductive Bias:**
   - Physical handwriting is produced by a mechanical motor system governed by pen-tip inertia, muscle acceleration, and surface friction.
   - The hidden state and memory cell of an LSTM:
     $$c_t = f_t \odot c_{t-1} + i_t \odot \tilde{c}_t$$
     $$h_t = o_t \odot \tanh(c_t)$$
     act as an internal physical momentum reservoir. The recurrence naturally preserves smooth stroke velocity and curvature continuity ($C^1$ continuity) without abrupt pen jumps.
2. **Monotonic Gaussian Soft-Window Attention:**
   - Instead of standard unconstrained softmax attention across the entire sequence, Graves introduced a monotonic Gaussian window:
     $$\phi(t, u) = \sum_{k=1}^K \alpha_t^k \exp\left(-\beta_t^k (\kappa_t^k - u)^2\right)$$
     $$\kappa_t^k = \kappa_{t-1}^k + \exp(k_t^k)$$
   - Because $\exp(k_t^k) > 0$, the attention center $\kappa_t$ is mathematically guaranteed to move only forward along the character sequence $u$, preventing the pen from jumping backward in text.

---

## 3. Core Issues and Failure Modes in the Graves LSTM Architecture

Despite its physical plausibility, the Graves recurrent framework exhibits five critical structural failures:

### Issue 1: Attractor Basins & The "Barcode Effect" on Repetitive Characters
- **Symptom:** When given repeated character sequences such as `"llllllll"`, `"ee"`, `"0000"`, or double letters (`"ll"` in `"GLA"`), the model frequently enters an infinite loop, drawing dozens or hundreds of overlapping cursive loops at the same coordinate without advancing.
- **Root Cause:**
  - In recurrent state space, repetitive identical character embeddings cause the recurrent hidden state $h_t$ to enter a limit-cycle attractor orbit.
  - The network's attention advancement output $\hat{k}_t$ collapses:
    $$\Delta\kappa_t = \exp(\hat{k}_t) \to 0$$
  - The attention window center $\kappa_t$ freezes on the current character. Because the RNN receives the exact same character conditioning and stroke history, it is trapped in the attractor basin until the maximum step limit (e.g., 1,200 steps) terminates the process, leaving a dense, black "barcode" ink smudge.

### Issue 2: Attention Stalling on Out-of-Distribution Glyphs and Capitals
- **Symptom:** On unfamiliar characters, punctuation, or capital letters (e.g., capital `'G'`, `'Z'`, uppercase sequences), the attention mechanism stalls at step 0 ($\kappa_t \approx 0$), failing to emit strokes or looping indefinitely.
- **Root Cause:**
  - Capital letters occur far less frequently in the IAM dataset than lowercase characters.
  - If the initial hidden state cannot produce a strong enough $\exp(\hat{k}_t)$ to cross the character's attention threshold, the Gaussian window fails to activate the character's downstream ligature dynamics.

### Issue 3: Cumulative Vertical and Baseline Drift
- **Symptom:** As the pen writes across a line, sentences frequently drift downward or upward diagonally across the page, failing to stay aligned with notebook ruling lines.
- **Root Cause:**
  - The model operates strictly autoregressively on differential offsets $(\Delta x_t, \Delta y_t)$.
  - Any small bias $\epsilon_y$ in $\Delta y_t$ accumulates monotonically over $T$ steps:
    $$y_T = y_0 + \sum_{t=1}^T (\Delta y_t + \epsilon_y) = y_0 + \sum_{t=1}^T \Delta y_t + T \cdot \epsilon_y$$
  - Without a global canvas reference frame or absolute vertical anchoring, open-loop integration invariably drifts over long sequences.

### Issue 4: Catastrophic Trajectory Deflection / Wild Arcs
- **Symptom:** Occasionally, sampling from the bivariate Gaussian mixture emits an outlier offset, causing the pen to shoot violently across the canvas in a giant elliptical arc before returning or crashing.
- **Root Cause:**
  - In low neatness bias regimes ($b \approx 0$), the covariance matrix $\Sigma = \begin{bmatrix} \sigma_x^2 & \rho \sigma_x \sigma_y \\ \rho \sigma_x \sigma_y & \sigma_y^2 \end{bmatrix}$ occasionally samples from heavy tails.
  - Because each step conditions the next input $x_{t+1}$, a single extreme displacement throws the LSTM hidden state into an unmapped region of its activation space, triggering runaway acceleration.

### Issue 5: Mixture Density Network (MDN) Training Instabilities
- **Symptom:** Training an MDN from scratch is notoriously delicate. Loss surfaces exhibit steep cliffs and numerical instability.
- **Root Cause:**
  - Negative Log-Likelihood (NLL) of Gaussian mixtures:
    $$\mathcal{L}_{\text{MDN}} = -\log \sum_{j=1}^M \pi_j \mathcal{N}(\Delta x_t, \Delta y_t \mid \mu_j, \Sigma_j)$$
  - If a mixture component collapses toward a single data point ($\sigma_j \to 0$), the likelihood tends to infinity and loss collapses to $-\infty$ or `NaN`.
  - Conversely, mode averaging can cause blurred, indecisive strokes.

---

## 4. Attempted Mitigations and Why Naive Alternatives Failed

### 1. Dwell-Governed Monotonic Heuristic (Inference Patch)
- **Implementation:** Added an attention dwell counter in inference: if attention lingered on character $u$ for $> 35$ steps, the Gaussian centers were manually incremented:
  $$\kappa \leftarrow \kappa + 0.8$$
- **Outcome:** Successfully resolved the barcode loop on `"llllllll"` (completing in 196 steps) and eliminated stalls on `'G'`.
- **Limitation:** This is an external rule-based heuristic applied during inference. It is not an inherent learned property of the neural network. If `max_dwell` is set too low ($< 25$), complex glyphs are clipped prematurely; if set too high ($> 45$), unnatural looping artifacts re-emerge.

### 2. Discrete Tokenization (512 Spatial Bins + Cross-Entropy)
- **Concept:** Discretized continuous coordinate space into quantized displacement tokens to eliminate MDN variance collapse.
- **Outcome:** Completely eliminated `NaN` losses, but generated severe staircase artifacts. Human cursive relies on sub-millimeter motor smoothness; discrete binning stripped away micro-dynamics and natural ligatures.

### 3. Autoregressive Continuous Transformer-MDN
- **Concept:** Replaced recurrent LSTM with a causal multi-head self-attention and cross-attention Transformer decoder with diagonal guided attention loss ($\mathcal{L}_{\text{guided}}$).
- **Outcome:** The training loss dropped rapidly to $-1.37$ (validation loss $0.0027$), and monotonic attention alignment was enforced mathematically.
- **Visual Failure:** Despite excellent numerical losses, the generated strokes were visually broken into detached, Morse-code dashes with flickering pen states ($e_t \approx 0.5$).
- **The Core Insight:** Pure self-attention computes dot-product similarities across all prior tokens without an explicit differential physics prior. It lacks the continuous recurrent momentum ($c_t = f_t c_{t-1} + i_t \tilde{c}_t$) of an LSTM. When predicting $\Delta x, \Delta y$ independently at each step via attention queries, the pen lacks physical inertia, causing high-frequency spatial jitter.

---

## 5. Architectural Comparison: Is There a Better Generative Paradigm?

| Model Paradigm | Sequential Physical Inertia | Monotonic Alignment Stability | Training Stability | Parallel Training | Inference Speed | Suitability for Stroke Synthesis |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Graves LSTM (2013)** | **Very High** (Continuous cell memory) | **Poor** (Prone to attractor loops/stalls) | **Low** (MDN variance collapse) | **No** (Strictly sequential $O(T)$) | Fast ($O(1)$ recurrent step) | Strong calligraphic ink; weak attention reliability |
| **Vanilla Causal Transformer** | **Poor** (No recurrent momentum; choppy) | **Moderate** (Requires heavy guidance) | **Moderate** (Self-attention stability) | **Yes** ($O(1)$ parallel training) | Slow ($O(T)$ auto-regressive KV-cache) | Poor (Detached dashes without recurrence) |
| **State Space Models (Mamba / S6)** | **High** (Continuous differential hidden state) | **High** (Can be constrained monotonically) | **High** (Normalized state-space transitions) | **Yes** (Associative parallel scan) | **Very Fast** ($O(1)$ recurrent inference) | **Very High (Promising modern alternative)** |
| **Stroke Diffusion / Flow Matching** | **Very High** (Denoised globally as a continuous curve) | **Absolute** (Conditioned on full aligned text) | **High** (Standard MSE score matching) | **Yes** (Non-autoregressive) | Moderate (Requires 20–50 sampling steps) | **State-of-the-art visual quality** |
| **Non-Autoregressive Duration TTS-Style** | **High** (Interpolated spline trajectories) | **Absolute** (Explicit duration predictor) | **Very High** (Explicit alignment / CTC) | **Yes** (Fully parallel feed-forward) | **Instantaneous** (Single feed-forward pass) | **Excellent for stable, ruled documents** |

---

## 6. Most Promising Alternative Directions

### Paradigm A: Stroke Diffusion Models / Continuous Trajectory Flow Matching
- **How it works:** Rather than predicting point $t$ from point $t-1$ autoregressively (which causes drift and looping), a Diffusion model or Flow Matching network (similar to Diff-Handwriting or modern voice models) generates the entire stroke trajectory $X \in \mathbb{R}^{T \times 3}$ simultaneously from Gaussian noise, conditioned on text embeddings.
- **Why it solves our issues:**
  - **No Attractor Loops:** Global trajectory denoising cannot get stuck in a single-step recurrent attractor loop.
  - **Zero Baseline Drift:** The global receptive field sees the entire line canvas at once, naturally respecting horizontal ruled baselines.
  - **No Runaway Arcs:** Global score matching prevents individual coordinate explosion.

### Paradigm B: Selective State Space Models (Mamba / S6)
- **How it works:** Replaces both the LSTM gating and the Transformer attention with continuous-time state space equations:
  $$h'(t) = \mathbf{A} h(t) + \mathbf{B} x(t)$$
  $$y(t) = \mathbf{C} h(t) + \mathbf{D} x(t)$$
- **Why it solves our issues:**
  - Preserves the continuous differential physical momentum of the LSTM.
  - Avoids quadratic attention memory while supporting parallel training via associative scans.
  - Time-varying $\mathbf{B}$ and $\mathbf{C}$ parameters allow selective filtering of repetitive tokens without falling into limit cycles.

### Paradigm C: Explicit Monotonic Alignment Duration Predictor (TTS Style)
- **How it works:** Inspired by modern Speech Synthesis (FastSpeech 2, VITS):
  1. A Text Encoder processes the string.
  2. A **Duration Predictor** explicitly predicts how many stroke points each character should receive (e.g., `'l'` receives 22 points, `'G'` receives 38 points).
  3. Character representations are expanded according to predicted durations.
  4. A Convolutional/Recurrent decoder generates coordinates for the pre-allocated length in a single forward pass.
- **Why it solves our issues:**
  - **Mathematically impossible to enter an attractor loop or stall:** Every character receives an explicit, finite duration budget.
  - Total length is deterministic and strictly bounded.

---

## 7. Concrete Roadmap & Next Actions

1. **Short-Term (Baseline Consolidation):**
   - Retain the dwell-governed Graves baseline (`checkpoints/pretrained/`) as the validated, working production engine for official academic submission and document rendering (`outputs/final_showcase.svg`).
   - Document all empirical dwell boundaries (`max_dwell=35`, neatness bias $b=1.0$) in project defense documentation.
2. **Medium-Term (Architectural Innovation):**
   - Implement an explicit **Duration-Predictor Alignment Module** or evaluate a **Mamba/S6 continuous motor block** to prove a superior next-generation architecture that natively eliminates the Graves failure modes without heuristic inference patches.
