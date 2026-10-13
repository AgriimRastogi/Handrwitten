# Stroke-Level Handwriting Synthesis
### A Vector-Output Generative Model for Composable Handwritten Documents

**Student Name:** Agriim Rastogi
**Enrollment / Roll No.:** 22515500004
**Programme / Branch:** BTECH CSE AIML 2nd year
**Section:** ND
**Academic Session:** 2026–27
**Date:** 25 August 2026

---

## 1. Project Title

**Stroke-Level Handwriting Synthesis: A Vector-Output Generative Model for
Composable Handwritten Documents**, benchmarked against and aiming to improve
upon the seminal recurrent-network approach of Graves (2013).

---

## 2. Problem Statement

**Statement.** Design and implement a machine-learning system that *generates*
realistic human handwriting from arbitrary input text, producing the output as
**vector pen-strokes** (a sequence of pen movements) rather than as a raster
image of pixels, so that the result is legible, stylistically coherent, and can
be composed into complete, infinitely scalable handwritten documents.

**Description.** Handwriting carries individuality and a human quality that
uniform digital fonts cannot reproduce. Automatically synthesising handwriting
has real applications: personalised note and letter generation, assistive tools
for users who cannot write by hand, synthetic training data to improve
handwriting-recognition (OCR/HTR) systems, custom handwriting "fonts," and
educational tools for teaching letter formation.

Existing solutions fall into two families. **Image-based** methods (GANs,
diffusion models) generate pixels; their output is fixed-resolution, hard to
edit or rescale, and awkward to stitch into flowing multi-line documents.
**Trajectory-based (online)** methods instead model the *motion of the pen* over
time — the sequence of small displacements and pen-up/pen-down events that a
person actually produces. The seminal work here is Graves (2013), *Generating
Sequences With Recurrent Neural Networks*, which conditions a recurrent network
on text through a Gaussian "soft-window" attention mechanism.

The problem this project addresses is threefold. First, the pen trajectory is a
noisy, variable-length, real-valued signal that must be represented in a form a
neural network can learn from without losing legibility. Second, the widely used
2013 reference approach predates modern sequence architectures and is more often
*reused* than *improved upon*. Third, most public results are raster images,
whereas a genuinely useful system should emit **editable vector output** that
composes into documents. This project therefore builds a handwriting-synthesis
pipeline **from scratch**, targets a measurable improvement over the 2013
baseline, and produces **SVG vector strokes** as its native output.

---

## 3. Objectives

1. **Build a reproducible data pipeline** over the IAM-OnDB online-handwriting
   corpus: parse the raw sensor files, clean corrupt records, normalise
   coordinates, and pack the result into an efficient training format.
2. **Represent pen trajectories as discrete token sequences** that preserve
   legibility, keeping reconstruction error below the threshold at which
   spacing damage becomes visible to the eye.
3. **Implement a Graves-style LSTM baseline** with Gaussian soft-window
   attention that conditions the generated strokes on the input text.
4. **Design and train a Transformer decoder** with cross-attention to the text
   as an improved model, and quantitatively compare it against the baseline.
5. **Support writer-style conditioning** through learned per-writer embeddings,
   so the same text can be rendered in different hands.
6. **Render generated strokes to scalable SVG** and compose individual lines
   into complete, multi-line document pages.
7. **Evaluate rigorously**, both quantitatively (position-space reconstruction
   and generation error) and qualitatively (legibility and stylistic realism).

---

## 4. Scope

**In scope.**

- Online (trajectory-level) handwriting synthesis for **English text in Latin
  script**, learned from the IAM-OnDB corpus (≈12,178 line samples written by
  197 distinct writers).
- **Text-conditioned** generation, with **optional writer-style** conditioning.
- **Vector (SVG) output** and composition of lines into document pages.
- Use of the dataset strictly for **non-commercial academic research**, in
  compliance with the IAM-OnDB licence.

**Out of scope.**

- Offline/image-based handwriting recognition, or generation from photographs of
  handwriting.
- Non-Latin scripts and multilingual generation.
- A fully productionised mobile application. (A React Native app is a possible
  long-term extension, not a deliverable of this project.)
- Any commercial deployment, which the dataset licence does not permit.

---

## 5. Proposed Methodology

**Data preparation.** From IAM-OnDB, only the transcription that matches the
actual pen strokes is used. Records with corrupt timestamps or mismatched line
counts are discarded. The corpus is split **by writer**, so that no individual's
handwriting appears in both the training and evaluation sets — this prevents the
model from being scored on memorised hands instead of genuine generalisation.

**Representation.** Each pen trajectory is converted into per-step offsets —
change in x, change in y, elapsed time, and a pen-up/pen-down flag. These
continuous values are then **discretised (tokenised)** into integer tokens so
the problem becomes next-token prediction. Because the large "jumps" the pen
makes between words are rare (~8% of steps) but dominate reconstruction error,
they are given a **separate set of bins** from ordinary in-word movements, using
a transform that allocates resolution to the heavy tail. This design keeps the
median position reconstruction error at ≈0.032, well inside the band where
distortion is imperceptible.

**Modelling.** Two models are built and compared:
(a) an **LSTM with Gaussian soft-window attention**, reproducing the 2013
baseline; and
(b) a **Transformer decoder with cross-attention** to the encoded text, as the
proposed improvement. Both are trained autoregressively with a separate
prediction head per output channel.

**Training and rendering.** Training uses PyTorch with mixed-precision
(bf16/TF32) on an RTX 3050 GPU, with a cloud T4 GPU held in reserve for longer
runs. Generated token sequences are decoded back into pen offsets and rendered
to SVG through a single, shared rendering module.

---

## 6. Expected Outcomes

- A trained model that generates **legible, novel handwriting** from arbitrary
  English text.
- A **measurable quantitative improvement** of the Transformer model over the
  LSTM baseline (and hence over the 2013 reference approach).
- A **modular, well-tested codebase** and a reusable dataset pipeline.
- **SVG document output** demonstrating that generated handwriting composes
  cleanly into scalable, multi-line pages.

---

## 7. Tools and Technologies

- **Language:** Python 3.13
- **Numerical / data:** NumPy, SciPy, h5py (HDF5 storage), tqdm
- **Deep learning:** PyTorch (CUDA build)
- **Testing / reproducibility:** pytest (58 automated tests currently passing on
  the data pipeline)
- **Output:** Scalable Vector Graphics (SVG)
- **Dataset:** IAM-OnDB online handwriting database
- **Hardware:** NVIDIA RTX 3050 (6 GB, Ampere) laptop GPU; cloud T4 GPU for
  extended training

---

## 8. Justification Against Course Components

The project is designed to exercise the full range of components expected of a
2nd-year AIML project. (The mapping below should be read against the specific
evaluation rubric; each listed course component is directly addressed.)

- **Application of AI/ML concepts** — sequence modelling, LSTMs, attention
  mechanisms, and Transformer decoders form the core of the work.
- **Mathematical foundations** — autoregressive probabilistic modelling,
  quantisation and its error analysis, and coordinate geometry are all applied
  and reasoned about explicitly.
- **Data engineering** — a real, noisy dataset is parsed, cleaned, normalised,
  split without leakage, and stored efficiently, demonstrating practical data
  handling rather than working on a pre-cleaned toy set.
- **Software engineering** — a strictly layered, single-responsibility
  architecture backed by 58 automated tests shows disciplined, maintainable
  design and reproducibility.
- **Experimentation and evaluation** — a baseline is implemented specifically so
  the improved model can be compared against it, using both quantitative metrics
  and qualitative inspection.
- **Research orientation and novelty** — the project engages directly with a
  landmark 2013 paper, aims to improve on it with a modern architecture, and
  chooses vector rather than raster output.
- **Practical relevance** — the results apply to assistive technology,
  personalisation, and generating synthetic training data for handwriting
  recognition.

---

## 9. References

1. A. Graves, "Generating Sequences With Recurrent Neural Networks,"
   *arXiv:1308.0850*, 2013.
2. M. Liwicki and H. Bunke, "IAM-OnDB — an On-Line English Sentence Database
   Acquired from Handwritten Text on a Whiteboard," *Proc. ICDAR*, 2005.
3. A. Vaswani et al., "Attention Is All You Need," *Advances in Neural
   Information Processing Systems (NeurIPS)*, 2017.

---

**Submitted by:** Agriim Rastogi — 22515500004
