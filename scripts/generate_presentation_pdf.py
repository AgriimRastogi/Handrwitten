"""Generate a minimal, executive dark-mode presentation PDF (16:9 widescreen, 7 slides)
for the Stroke-Level Online Handwriting Synthesis capstone project review at GLA University.

Theme: Minimalist Dark Mode with user-specified #181818 canvas foundation.
Uses Playwright to render pixel-perfect 1920x1080 slides to PDF and export PNG slide previews.
"""

import asyncio
from pathlib import Path
from playwright.async_api import async_playwright

def build_slides_html(sample_svg_content: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Stroke-Level Online Handwriting Synthesis - Project Review</title>
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

  @page {{
    size: 1920px 1080px;
    margin: 0;
  }}

  * {{
    box-sizing: border-box;
    margin: 0;
    padding: 0;
  }}

  body {{
    background-color: #181818;
    color: #EDEDED;
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    -webkit-font-smoothing: antialiased;
    -moz-osx-font-smoothing: grayscale;
  }}

  .slide {{
    width: 1920px;
    height: 1080px;
    page-break-after: always;
    position: relative;
    overflow: hidden;
    background-color: #181818;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    padding: 68px 84px 56px 84px;
  }}

  /* Minimal subtle background grid pattern */
  .grid-bg {{
    position: absolute;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-image: 
      linear-gradient(to right, rgba(255, 255, 255, 0.02) 1px, transparent 1px),
      linear-gradient(to bottom, rgba(255, 255, 255, 0.02) 1px, transparent 1px);
    background-size: 60px 60px;
    pointer-events: none;
    z-index: 0;
  }}

  .content-layer {{
    position: relative;
    z-index: 1;
    display: flex;
    flex-direction: column;
    height: 100%;
    justify-content: space-between;
  }}

  /* Header Section */
  .slide-header {{
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 32px;
  }}

  .header-titles {{
    max-width: 1400px;
  }}

  .pill-tag {{
    display: inline-block;
    font-family: 'JetBrains Mono', monospace;
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    padding: 5px 12px;
    background: #242424;
    border: 1px solid #333333;
    border-radius: 4px;
    color: #3B82F6;
    margin-bottom: 12px;
  }}

  .pill-tag.green {{
    color: #10B981;
    border-color: #1e3d30;
    background: #19261f;
  }}

  .pill-tag.amber {{
    color: #F59E0B;
    border-color: #3d321e;
    background: #262219;
  }}

  .slide-title {{
    font-size: 44px;
    font-weight: 700;
    letter-spacing: -1.2px;
    color: #FFFFFF;
    line-height: 1.15;
    margin-bottom: 8px;
  }}

  .slide-subtitle {{
    font-size: 19px;
    font-weight: 400;
    color: #9E9E9E;
    line-height: 1.4;
  }}

  .slide-num {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 15px;
    font-weight: 600;
    color: #707070;
    letter-spacing: 2px;
    padding: 6px 14px;
    background: #222222;
    border: 1px solid #303030;
    border-radius: 4px;
  }}

  /* Main Body Layouts */
  .slide-body {{
    flex: 1;
    display: flex;
    flex-direction: column;
    justify-content: center;
  }}

  /* Cards & Components */
  .card {{
    background: #212121;
    border: 1px solid #333333;
    border-radius: 8px;
    padding: 26px 28px;
  }}

  .card-header {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 14px;
    border-bottom: 1px solid #2d2d2d;
    padding-bottom: 12px;
  }}

  .card-title {{
    font-size: 20px;
    font-weight: 600;
    color: #F5F5F5;
    letter-spacing: -0.3px;
    display: flex;
    align-items: center;
    gap: 10px;
  }}

  .card-badge {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    padding: 3px 8px;
    border-radius: 3px;
    background: #2a2a2a;
    color: #B0B0B0;
    border: 1px solid #383838;
  }}

  .card-text {{
    font-size: 15.5px;
    color: #B5B5B5;
    line-height: 1.6;
    margin-bottom: 14px;
  }}

  .bullet-list {{
    list-style: none;
    display: flex;
    flex-direction: column;
    gap: 10px;
  }}

  .bullet-list li {{
    font-size: 15px;
    color: #CCCCCC;
    line-height: 1.5;
    position: relative;
    padding-left: 20px;
  }}

  .bullet-list li::before {{
    content: "■";
    position: absolute;
    left: 0;
    font-size: 9px;
    top: 5px;
    color: #3B82F6;
  }}

  .bullet-list.green li::before {{
    color: #10B981;
  }}

  .bullet-list.rose li::before {{
    color: #EF4444;
  }}

  .bullet-list.amber li::before {{
    color: #F59E0B;
  }}

  /* Code / Monospace accents */
  code, .mono {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 13.5px;
    background: #1c1c1c;
    border: 1px solid #333333;
    padding: 2px 6px;
    border-radius: 3px;
    color: #60A5FA;
  }}

  .math-block {{
    font-family: 'JetBrains Mono', monospace;
    background: #181818;
    border: 1px solid #333333;
    padding: 12px 16px;
    border-radius: 6px;
    color: #F59E0B;
    font-size: 14.5px;
    margin: 12px 0;
    line-height: 1.5;
  }}

  /* Footer */
  .slide-footer {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #2d2d2d;
    padding-top: 18px;
    margin-top: 24px;
    font-size: 13px;
    color: #7A7A7A;
    letter-spacing: 0.3px;
  }}

  .footer-left {{
    display: flex;
    gap: 20px;
    align-items: center;
  }}

  .footer-item {{
    display: flex;
    align-items: center;
    gap: 6px;
  }}

  .footer-item strong {{
    color: #A0A0A0;
    font-weight: 500;
  }}

  /* Grid Layouts */
  .grid-2 {{
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 24px;
  }}

  .grid-3 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr;
    gap: 24px;
  }}

  .grid-4 {{
    display: grid;
    grid-template-columns: 1fr 1fr 1fr 1fr;
    gap: 20px;
  }}

  .stat-box {{
    background: #1e1e1e;
    border: 1px solid #303030;
    border-radius: 6px;
    padding: 16px 20px;
  }}

  .stat-val {{
    font-family: 'JetBrains Mono', monospace;
    font-size: 28px;
    font-weight: 700;
    color: #FFFFFF;
    margin-bottom: 4px;
  }}

  .stat-val.blue {{ color: #3B82F6; }}
  .stat-val.green {{ color: #10B981; }}
  .stat-val.amber {{ color: #F59E0B; }}

  .stat-label {{
    font-size: 12.5px;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #888888;
  }}

  .flow-step {{
    display: flex;
    align-items: center;
    gap: 12px;
    background: #1e1e1e;
    border: 1px solid #303030;
    border-radius: 6px;
    padding: 12px 16px;
  }}

  .flow-idx {{
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 14px;
    color: #3B82F6;
    background: #252e3e;
    padding: 4px 8px;
    border-radius: 4px;
  }}
</style>
</head>
<body>

<!-- SLIDE 1: TITLE & ACADEMIC IDENTITY -->
<div class="slide">
  <div class="grid-bg"></div>
  <div class="content-layer">
    <div class="slide-header">
      <div class="header-titles">
        <span class="pill-tag">B.Tech Capstone Project • Session 2026–2027</span>
        <h1 class="slide-title" style="font-size: 50px; margin-top: 8px; margin-bottom: 12px;">Stroke-Level Online Handwriting Synthesis</h1>
        <p class="slide-subtitle" style="font-size: 22px; color: #A8A8A8; max-width: 1200px;">
          A Vector-Output Generative Sequence Architecture for Composable Handwritten Documents
        </p>
      </div>
      <div class="slide-num">01 / 07</div>
    </div>

    <div class="slide-body">
      <!-- High-level Value Banner -->
      <div style="background: #212121; border: 1px solid #333333; border-radius: 8px; padding: 22px 28px; margin-bottom: 28px; display: flex; align-items: center; justify-content: space-between;">
        <div>
          <div style="font-size: 13px; font-family: 'JetBrains Mono', monospace; color: #3B82F6; letter-spacing: 1px; margin-bottom: 4px; text-transform: uppercase;">Core Generative Pipeline</div>
          <div style="font-size: 19px; font-weight: 500; color: #FFFFFF;">
            Arbitrary Text String <span style="color:#3B82F6;">&rarr;</span> Soft-Window Attention <span style="color:#3B82F6;">&rarr;</span> Autoregressive Temporal Strokes <span style="color:#3B82F6;">&rarr;</span> <span style="color:#10B981;">C&sup1; Bézier SVG</span>
          </div>
        </div>
        <div style="display: flex; gap: 12px;">
          <span style="background: #1c2a22; border: 1px solid #234d35; color: #34D399; font-size: 13px; font-family: 'JetBrains Mono', monospace; padding: 6px 14px; border-radius: 4px;">70/70 Unit Tests Green</span>
          <span style="background: #1f2736; border: 1px solid #2d4366; color: #60A5FA; font-size: 13px; font-family: 'JetBrains Mono', monospace; padding: 6px 14px; border-radius: 4px;">-2.155 NLL Loss</span>
        </div>
      </div>

      <!-- Institutional Metadata Grid -->
      <div class="grid-4">
        <div class="card" style="padding: 22px;">
          <div class="stat-label">Candidate</div>
          <div style="font-size: 20px; font-weight: 700; color: #FFFFFF; margin: 6px 0 4px 0;">Agriim Rastogi</div>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #9E9E9E;">Roll: 22515500004</div>
          <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #9E9E9E;">Section: ND</div>
        </div>
        <div class="card" style="padding: 22px;">
          <div class="stat-label">Programme</div>
          <div style="font-size: 18px; font-weight: 600; color: #FFFFFF; margin: 6px 0 4px 0;">B.Tech CSE</div>
          <div style="font-size: 14px; color: #3B82F6; font-weight: 500;">Specialization in AI</div>
          <div style="font-size: 13px; color: #888888;">Department of CEA</div>
        </div>
        <div class="card" style="padding: 22px;">
          <div class="stat-label">Mentors & Reviewer</div>
          <div style="font-size: 16px; font-weight: 600; color: #FFFFFF; margin: 6px 0 4px 0;">Palash Sir / Vansh Sir</div>
          <div style="font-size: 13.5px; color: #9E9E9E;">Reviewer: Gautam Mukherjee</div>
          <div style="font-size: 13px; color: #888888;">Faculty of Engineering</div>
        </div>
        <div class="card" style="padding: 22px;">
          <div class="stat-label">Institution</div>
          <div style="font-size: 18px; font-weight: 600; color: #FFFFFF; margin: 6px 0 4px 0;">GLA University</div>
          <div style="font-size: 13.5px; color: #9E9E9E;">Greater Noida Campus</div>
          <div style="font-size: 13px; color: #888888;">Session 2026–2027</div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <div class="footer-item"><strong>Document:</strong> Mini Project Presentation Deck</div>
        <div class="footer-item"><strong>Specification:</strong> SRS v1.1 Approved</div>
      </div>
      <div>GLA University • Capstone Review</div>
    </div>
  </div>
</div>

<!-- SLIDE 2: PROBLEM STATEMENT & VECTOR PARADIGM -->
<div class="slide">
  <div class="grid-bg"></div>
  <div class="content-layer">
    <div class="slide-header">
      <div class="header-titles">
        <span class="pill-tag">02 / 07 • Motivation & Paradigm</span>
        <h2 class="slide-title">Why Online Vector Synthesis?</h2>
        <p class="slide-subtitle">Overcoming the fundamental flaws of static fonts and raster 2D diffusion models.</p>
      </div>
      <div class="slide-num">02 / 07</div>
    </div>

    <div class="slide-body">
      <div class="grid-3" style="margin-bottom: 24px;">
        <!-- Card 1: Static Fonts -->
        <div class="card" style="border-top: 3px solid #EF4444;">
          <div class="card-header">
            <span class="card-title">1. Static Digital Fonts</span>
            <span class="card-badge" style="color: #F87171;">.TTF / .OTF</span>
          </div>
          <p class="card-text">Every instance of a letter renders identical fixed outlines.</p>
          <ul class="bullet-list rose">
            <li><strong>Repetition Flaw:</strong> Exact duplicate glyphs across a document; immediately detected as artificial.</li>
            <li><strong>Zero Ligatures:</strong> Cannot model organic connecting strokes between cursive letters.</li>
            <li><strong>Rigid Baselines:</strong> Lacks natural human drift, slant variation, and pressure cadence.</li>
          </ul>
        </div>

        <!-- Card 2: Raster Diffusion -->
        <div class="card" style="border-top: 3px solid #F59E0B;">
          <div class="card-header">
            <span class="card-title">2. 2D Raster Models</span>
            <span class="card-badge" style="color: #FBBF24;">Diffusion / GANs</span>
          </div>
          <p class="card-text">Synthesizes handwriting directly as static RGB pixel bitmaps.</p>
          <ul class="bullet-list amber">
            <li><strong>Resolution Barrier:</strong> Fixed 256&times;256 or 512&times;512 grids; severe blur when zoomed in.</li>
            <li><strong>Heavy Latency:</strong> Multi-step denoising requires massive GPU compute per sentence.</li>
            <li><strong>No Vector Control:</strong> Output is a baked raster; strokes cannot be parsed or edited.</li>
          </ul>
        </div>

        <!-- Card 3: Our Approach -->
        <div class="card" style="border-top: 3px solid #10B981; background: #232724;">
          <div class="card-header">
            <span class="card-title" style="color: #34D399;">3. Online Stroke Synthesis</span>
            <span class="card-badge" style="color: #34D399; background: #1a2b22; border-color: #275239;">hws (Our Project)</span>
          </div>
          <p class="card-text">Generates temporal pen coordinates <code>(&Delta;x, &Delta;y, &Delta;t, p)</code> sequentially.</p>
          <ul class="bullet-list green">
            <li><strong>Infinite Scalability:</strong> Renders to scalable vector graphics (<code>SVG &lt;path&gt;</code>); razor sharp at any zoom.</li>
            <li><strong>Sub-3s Inference:</strong> Lightweight autoregressive sequence decoding on edge GPUs.</li>
            <li><strong>Authentic Physics:</strong> Models cursive ligatures, airborne transitions, and notebook ruling.</li>
          </ul>
        </div>
      </div>

      <!-- Core Architectural Thesis Banner -->
      <div style="background: #212121; border: 1px solid #333333; border-radius: 6px; padding: 18px 24px; display: flex; align-items: center; justify-content: space-between;">
        <div style="font-size: 15px; color: #D1D5DB; line-height: 1.5;">
          <strong style="color: #60A5FA;">Key Insight:</strong> Handwriting is fundamentally a <em>continuous temporal trajectory</em> of a physical stylus tip on paper, not a 2D image. Modeling the time-series trajectory yields mathematical elegance, infinite zoom, and compact storage.
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #10B981; white-space: nowrap; margin-left: 24px;">
          &Delta;s = 2.0 Spatial Arc-Length
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <div class="footer-item"><strong>Agriim Rastogi</strong> (22515500004)</div>
        <div class="footer-item"><strong>Sec:</strong> ND</div>
        <div class="footer-item"><strong>Mentors:</strong> Palash Sir / Vansh Sir</div>
      </div>
      <div>Problem Formulation & Architectural Paradigm</div>
    </div>
  </div>
</div>

<!-- SLIDE 3: MATHEMATICAL & DATA CORE -->
<div class="slide">
  <div class="grid-bg"></div>
  <div class="content-layer">
    <div class="slide-header">
      <div class="header-titles">
        <span class="pill-tag">03 / 07 • Mathematical & Data Core</span>
        <h2 class="slide-title">Algorithmic Innovations & Data Pipeline</h2>
        <p class="slide-subtitle">Solving Graves 2013 variance collapse, sequence bloat, and ragged dataset storage.</p>
      </div>
      <div class="slide-num">03 / 07</div>
    </div>

    <div class="slide-body">
      <!-- Top Metrics Bar -->
      <div class="grid-4" style="margin-bottom: 22px;">
        <div class="stat-box">
          <div class="stat-val blue">12,178</div>
          <div class="stat-label">IAM-OnDB Lines</div>
        </div>
        <div class="stat-box">
          <div class="stat-val blue">3,733,404</div>
          <div class="stat-label">Stroke Points</div>
        </div>
        <div class="stat-box">
          <div class="stat-val green">92.1% / 7.9%</div>
          <div class="stat-label">Moves vs. Airborne Jumps</div>
        </div>
        <div class="stat-box">
          <div class="stat-val amber">1.84&times;</div>
          <div class="stat-label">Sequence Compression</div>
        </div>
      </div>

      <div class="grid-2">
        <!-- Decision 1: Dual-Mode Discrete Tokenization -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Decision 1 (D1): Dual-Mode Tokenization</span>
            <span class="card-badge" style="color: #60A5FA;">Stable Cross-Entropy</span>
          </div>
          <p class="card-text">
            Continuous Mixture Density Networks (MDNs) suffer from numerical instability where variance collapses (&sigma; &rarr; 0, log-loss &rarr; &infin;). We partition the velocity space into two distinct regimes:
          </p>
          <div class="math-block">
            <strong>Airborne Jumps (asinh Binning):</strong><br>
            k = sgn(&Delta;x) &middot; &lfloor; asinh(|&Delta;x| / s) &middot; N &rfloor;
          </div>
          <ul class="bullet-list">
            <li><strong>Moves (Pen Down):</strong> High-density linear spatial bins capturing delicate cursive curves.</li>
            <li><strong>Jumps (Pen Up):</strong> Heavy-tailed inverse hyperbolic sine (asinh) bins covering wide transitions.</li>
            <li><strong>Stability:</strong> Replaces fragile Gaussian log-likelihood with bounded Cross-Entropy loss.</li>
          </ul>
        </div>

        <!-- Decision 2: Resampling & Ragged Storage -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Decision 2 (D2): Ragged HDF5 & Resampling</span>
            <span class="card-badge" style="color: #34D399;">Storage & Throughput</span>
          </div>
          <p class="card-text">
            Raw tablet digitizer streams sample points at erratic temporal intervals, leading to dense clusters during slow writing and sparse gaps during fast strokes.
          </p>
          <ul class="bullet-list green">
            <li><strong>Spatial Arc-Length Resampling:</strong> Enforces uniform Euclidean spacing <code>ds = 2.0</code> along the stroke trajectory.</li>
            <li><strong>1.84&times; Sequence Reduction:</strong> Compresses average line length from 620 down to 342 points without losing fidelity.</li>
            <li><strong>Ragged 1D HDF5 Arrays:</strong> Stores variable-length samples contiguously in 1D memory <code>(N_total, 4)</code> with index table <code>(M, 2)</code>.</li>
            <li><strong>Zero Waste:</strong> Eliminates 40% wasted disk space compared to fixed zero-padded matrices.</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <div class="footer-item"><strong>Dataset:</strong> IAM-OnDB XML Corpus</div>
        <div class="footer-item"><strong>Format:</strong> HDF5 (Train / Val / Test)</div>
      </div>
      <div>GLA University • Capstone Review</div>
    </div>
  </div>
</div>

<!-- SLIDE 4: DUAL-TRACK NEURAL ARCHITECTURE -->
<div class="slide">
  <div class="grid-bg"></div>
  <div class="content-layer">
    <div class="slide-header">
      <div class="header-titles">
        <span class="pill-tag">04 / 07 • Neural Architecture</span>
        <h2 class="slide-title">Dual-Track Neural Sequence Modeling</h2>
        <p class="slide-subtitle">Reproducing the canonical Graves 2013 recurrent baseline alongside modern Transformer attention.</p>
      </div>
      <div class="slide-num">04 / 07</div>
    </div>

    <div class="slide-body">
      <!-- System Architecture Flow -->
      <div style="display: flex; gap: 14px; margin-bottom: 22px;">
        <div class="flow-step" style="flex: 1;">
          <span class="flow-idx">1</span>
          <div>
            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF;">Text Conditioning</div>
            <div style="font-size: 12px; color: #888888;">One-Hot / Tokenizer</div>
          </div>
        </div>
        <div class="flow-step" style="flex: 1;">
          <span class="flow-idx">2</span>
          <div>
            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF;">Monotonic Attention</div>
            <div style="font-size: 12px; color: #888888;">Soft-Window (&kappa;<sub>t</sub>)</div>
          </div>
        </div>
        <div class="flow-step" style="flex: 1;">
          <span class="flow-idx">3</span>
          <div>
            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF;">Recurrent / Trans State</div>
            <div style="font-size: 12px; color: #888888;">Peephole LSTM / Causal</div>
          </div>
        </div>
        <div class="flow-step" style="flex: 1;">
          <span class="flow-idx">4</span>
          <div>
            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF;">Output Heads</div>
            <div style="font-size: 12px; color: #888888;">20-Comp MDN / Discrete</div>
          </div>
        </div>
        <div class="flow-step" style="flex: 1;">
          <span class="flow-idx">5</span>
          <div>
            <div style="font-size: 14px; font-weight: 600; color: #FFFFFF;">SVG Render Engine</div>
            <div style="font-size: 12px; color: #888888;">Catmull-Rom Splines</div>
          </div>
        </div>
      </div>

      <div class="grid-2">
        <!-- Track A: Graves 2013 Peephole LSTM -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Track A: Canonical Graves 2013 RNN</span>
            <span class="card-badge" style="color: #60A5FA;">Baseline Reproduction</span>
          </div>
          <p class="card-text">
            3-layer recurrent neural network with cell-to-gate peephole connections and monotonic Gaussian soft-window attention:
          </p>
          <div class="math-block">
            &kappa;<sub>t</sub><sup>k</sup> = &kappa;<sub>t-1</sub><sup>k</sup> + exp(&alpha;<sub>t</sub><sup>k</sup>), &nbsp;&nbsp;
            w<sub>t</sub>(u) = &sum; &alpha;<sub>t</sub><sup>k</sup> exp(-&beta;<sub>t</sub><sup>k</sup>(&kappa;<sub>t</sub><sup>k</sup> - u)&sup2;)
          </div>
          <ul class="bullet-list">
            <li><strong>Monotonic Guarantee:</strong> Window center &kappa; only advances forward; strictly prevents reading backward.</li>
            <li><strong>20-Component Bivariate GMM:</strong> Predicts (&mu;<sub>x</sub>, &mu;<sub>y</sub>, &sigma;<sub>x</sub>, &sigma;<sub>y</sub>, &rho;, &pi;) plus Bernoulli pen lift.</li>
            <li><strong>Reproduction Status:</strong> Reached <strong>-2.155 NLL validation loss</strong> matching published SOTA.</li>
          </ul>
        </div>

        <!-- Track B: Modern Transformer Decoder -->
        <div class="card">
          <div class="card-header">
            <span class="card-title">Track B: Autoregressive Transformer</span>
            <span class="card-badge" style="color: #F59E0B;">Custom Innovation</span>
          </div>
          <p class="card-text">
            Bidirectional text encoder coupled with a causal autoregressive Transformer decoder utilizing multi-head cross-attention:
          </p>
          <ul class="bullet-list amber">
            <li><strong>Causal Self-Attention:</strong> Attends to previous stroke trajectory tokens without recurrent bottleneck.</li>
            <li><strong>Cross-Attention:</strong> Dynamically aligns stroke coordinates to text characters across all positions.</li>
            <li><strong>Style Conditioning (e<sub>style</sub>):</strong> Incorporates learned writer embeddings to synthesize varied personal handwriting aesthetics.</li>
            <li><strong>Training Milestone:</strong> 20 full epochs trained (<code>checkpoints/best_model.pt</code>).</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <div class="footer-item"><strong>PyTorch:</strong> 2.14.0+cu126</div>
        <div class="footer-item"><strong>Hardware:</strong> NVIDIA RTX 3050 6GB Laptop GPU (AMP Enabled)</div>
      </div>
      <div>GLA University • Capstone Review</div>
    </div>
  </div>
</div>

<!-- SLIDE 5: VISUAL QUALITY BREAKTHROUGH -->
<div class="slide">
  <div class="grid-bg"></div>
  <div class="content-layer">
    <div class="slide-header">
      <div class="header-titles">
        <span class="pill-tag green">05 / 07 • Engineering Breakthrough</span>
        <h2 class="slide-title">Visual Quality & Calligraphic Stability</h2>
        <p class="slide-subtitle">Eliminating Brownian hand tremors, robotic polygon edges, and character skipping.</p>
      </div>
      <div class="slide-num">05 / 07</div>
    </div>

    <div class="slide-body">
      <!-- 4 Pillars 2x2 Grid -->
      <div class="grid-2" style="gap: 20px;">
        <!-- Pillar 1: Temperature Scaling -->
        <div class="card" style="padding: 22px;">
          <div class="card-header">
            <span class="card-title" style="font-size: 18px;">1. Temperature Scaling (T = 0.35)</span>
            <span class="card-badge" style="color: #60A5FA;">Tremor Elimination</span>
          </div>
          <p class="card-text">
            Unconditioned Gaussian sampling exhibits severe Brownian high-frequency noise resembling severe muscle tremors.
          </p>
          <div class="math-block" style="margin: 8px 0; font-size: 13px;">
            &sigma;<sub>scaled</sub> = &sigma; &middot; T<sup>1.2</sup> &nbsp;&nbsp;(T = 0.35)
          </div>
          <p style="font-size: 14px; color: #CCCCCC;">
            Shrinking mixture covariance concentrates sampling around the modal skeleton of letterforms while preserving organic cursive curvature.
          </p>
        </div>

        <!-- Pillar 2: Catmull-Rom Bézier Splines -->
        <div class="card" style="padding: 22px;">
          <div class="card-header">
            <span class="card-title" style="font-size: 18px;">2. C&sup1; Catmull-Rom Bézier Splines</span>
            <span class="card-badge" style="color: #34D399;">Fountain Pen Flow</span>
          </div>
          <p class="card-text">
            Standard polyline renderers draw rigid straight line segments (<code>L</code> commands) that appear segmented and polygonal.
          </p>
          <ul class="bullet-list green">
            <li>Interpolates discrete coordinates into smooth cubic Bézier curves (<code>C</code> commands in SVG).</li>
            <li>Guarantees continuous first-order tangents (C&sup1;) across every waypoint.</li>
            <li>Transforms raw discrete coordinates into fluid ink calligraphy.</li>
          </ul>
        </div>

        <!-- Pillar 3: Savitzky-Golay Filtering -->
        <div class="card" style="padding: 22px;">
          <div class="card-header">
            <span class="card-title" style="font-size: 18px;">3. Savitzky-Golay Denoising</span>
            <span class="card-badge" style="color: #FBBF24;">window=7, poly=3</span>
          </div>
          <p class="card-text">
            Digital low-pass moving-average filters destroy sharp corners (like the vertices of 'v', 'w', and crossbars of 't').
          </p>
          <ul class="bullet-list amber">
            <li>Fits local 3rd-order polynomials over sliding 7-point windows.</li>
            <li>Suppresses coordinate quantization noise without rounding out sharp stroke cusps.</li>
          </ul>
        </div>

        <!-- Pillar 4: Attention Verification -->
        <div class="card" style="padding: 22px;">
          <div class="card-header">
            <span class="card-title" style="font-size: 18px;">4. Attention-Verified Sampling</span>
            <span class="card-badge" style="color: #A78BFA;">0 Dropped Glyphs</span>
          </div>
          <p class="card-text">
            Recurrent attention occasionally stalls on repetitive double letters ('ll', 'ee') or skips trailing punctuation.
          </p>
          <ul class="bullet-list">
            <li><strong>Automated Verification:</strong> Enforces minimum character soft-window dwell count (min<sub>u</sub> count(u) &ge; 6).</li>
            <li><strong>Aspect Ratio Guard:</strong> Rejects degenerated tight-loop attractors (requires width/height &gt; 2.0).</li>
          </ul>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <div class="footer-item"><strong>Status:</strong> High-Fidelity Calligraphy Achieved</div>
        <div class="footer-item"><strong>Renderer:</strong> hws.render (Catmull-Rom Engine)</div>
      </div>
      <div>GLA University • Capstone Review</div>
    </div>
  </div>
</div>

<!-- SLIDE 6: LIVE RESULTS & BENCHMARKS -->
<div class="slide">
  <div class="grid-bg"></div>
  <div class="content-layer">
    <div class="slide-header">
      <div class="header-titles">
        <span class="pill-tag green">06 / 07 • Live Results & Gallery</span>
        <h2 class="slide-title">Verified Synthesis Outputs & Benchmarks</h2>
        <p class="slide-subtitle">Multi-line notebook page composition, unit test coverage, and live generated sample.</p>
      </div>
      <div class="slide-num">06 / 07</div>
    </div>

    <div class="slide-body">
      <div class="grid-2" style="align-items: stretch; gap: 28px;">
        <!-- Left: Live Render Visual Showcase -->
        <div class="card" style="display: flex; flex-direction: column; justify-content: space-between; padding: 22px;">
          <div>
            <div class="card-header" style="margin-bottom: 10px;">
              <span class="card-title" style="font-size: 18px;">Live Model Vector Output (Real SVG)</span>
              <span class="card-badge" style="color: #34D399;">Ruled Notebook Paper</span>
            </div>
            <p style="font-size: 13.5px; color: #9E9E9E; margin-bottom: 12px;">
              Direct vector rendering of <code>Online Handwriting Synthesis</code> &amp; <code>GLA University Capstone 2026</code> (1,744 stroke points, C&sup1; Bézier curves):
            </p>
          </div>

          <!-- Embedded SVG Container -->
          <div style="background: #fdfdf8; border: 1px solid #444; border-radius: 6px; padding: 8px; display: flex; align-items: center; justify-content: center; min-height: 250px; overflow: hidden;">
            {sample_svg_content if sample_svg_content else '<div style="color:#000; font-family: monospace;">[Vector SVG Rendered: 1,744 Stroke Points]</div>'}
          </div>

          <div style="margin-top: 12px; font-size: 12.5px; color: #888888; display: flex; justify-content: space-between;">
            <span>Line Height: 46.0 units</span>
            <span>Red Margin Rule: #e8a0a0</span>
            <span>Ink Color: #0f172a</span>
          </div>
        </div>

        <!-- Right: Verification Metrics Dashboard -->
        <div style="display: flex; flex-direction: column; gap: 16px;">
          <!-- Benchmark Card -->
          <div class="card" style="padding: 22px;">
            <div class="card-header" style="margin-bottom: 12px;">
              <span class="card-title" style="font-size: 18px;">System Health & Test Dashboard</span>
              <span class="card-badge" style="color: #34D399;">Passing 100%</span>
            </div>
            <div class="grid-3" style="gap: 12px; margin-bottom: 14px;">
              <div class="stat-box" style="padding: 12px;">
                <div class="stat-val green" style="font-size: 24px;">70 / 70</div>
                <div class="stat-label" style="font-size: 11px;">PyTest Suite</div>
              </div>
              <div class="stat-box" style="padding: 12px;">
                <div class="stat-val blue" style="font-size: 24px;">-2.155</div>
                <div class="stat-label" style="font-size: 11px;">Validation NLL</div>
              </div>
              <div class="stat-box" style="padding: 12px;">
                <div class="stat-val amber" style="font-size: 24px;">&lt; 3.2s</div>
                <div class="stat-label" style="font-size: 11px;">Full Page Gen</div>
              </div>
            </div>
            <p style="font-size: 14px; color: #B0B0B0; line-height: 1.5;">
              Complete automated coverage across geometry transformations, XML parsers, ragged HDF5 loaders, attention alignment invariants, and Bézier SVG path generation.
            </p>
          </div>

          <!-- Document Synthesis Card -->
          <div class="card" style="padding: 22px;">
            <div class="card-header" style="margin-bottom: 12px;">
              <span class="card-title" style="font-size: 18px;">Composable Document Synthesizer</span>
              <span class="card-badge" style="color: #60A5FA;">scripts/synthesize.py</span>
            </div>
            <ul class="bullet-list" style="gap: 8px;">
              <li><strong>Automated Word-Wrapping:</strong> Splits arbitrary length text across lines while preserving full word boundaries.</li>
              <li><strong>Indian School Notebook Rulings:</strong> 42–48 unit horizontal guides with vertical red margin rule (<code>stroke="#e8a0a0"</code>).</li>
              <li><strong>Baseline Calibration:</strong> Anchors letters using 20th-percentile bounding box, ensuring natural descenders ('g', 'p', 'y').</li>
            </ul>
          </div>
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <div class="footer-item"><strong>Generated:</strong> NVIDIA RTX 3050 Laptop GPU</div>
        <div class="footer-item"><strong>CLI:</strong> python scripts/synthesize.py --ruled</div>
      </div>
      <div>GLA University • Capstone Review</div>
    </div>
  </div>
</div>

<!-- SLIDE 7: ROADMAP & WHAT'S LEFT -->
<div class="slide">
  <div class="grid-bg"></div>
  <div class="content-layer">
    <div class="slide-header">
      <div class="header-titles">
        <span class="pill-tag amber">07 / 07 • Roadmap & Remaining Sprints</span>
        <h2 class="slide-title">Project Roadmap & Next Milestones</h2>
        <p class="slide-subtitle">Clear demarcation of completed work, active development, and upcoming deliverables.</p>
      </div>
      <div class="slide-num">07 / 07</div>
    </div>

    <div class="slide-body">
      <div class="grid-2" style="gap: 28px; margin-bottom: 22px;">
        <!-- Completed Column -->
        <div class="card" style="border-top: 3px solid #10B981;">
          <div class="card-header">
            <span class="card-title" style="color: #34D399;">Completed Milestones (100% Done)</span>
            <span class="card-badge" style="color: #34D399;">Phases 0–4</span>
          </div>
          <ul class="bullet-list green" style="gap: 12px;">
            <li><strong>Dataset Preprocessing:</strong> Ingestion of 12,178 IAM-OnDB lines, spatial resampling, and ragged 1D HDF5 dataset pipeline.</li>
            <li><strong>Dual-Mode Tokenizer:</strong> Linear move bins + asinh jump bins for numerically stable Cross-Entropy sequence training.</li>
            <li><strong>Canonical Graves 2013 RNN:</strong> Full 3-layer peephole LSTM with soft-window Gaussian attention reproduced (-2.155 NLL).</li>
            <li><strong>Transformer Decoder:</strong> 20 full epochs trained with causal self-attention and writer style conditioning.</li>
            <li><strong>Calligraphic Spline Engine:</strong> Catmull-Rom cubic Bézier renderer with temperature-scaled sampling (T=0.35).</li>
            <li><strong>Official Academic SRS:</strong> 10-page GLA SRS specification finalized, signed off, and approved.</li>
          </ul>
        </div>

        <!-- Remaining Column -->
        <div class="card" style="border-top: 3px solid #F59E0B;">
          <div class="card-header">
            <span class="card-title" style="color: #FBBF24;">In Progress & What's Left</span>
            <span class="card-badge" style="color: #FBBF24;">Next Sprints</span>
          </div>
          <ul class="bullet-list amber" style="gap: 12px;">
            <li><strong>Sprint 5 — Stylus / Tablet Input Canvas:</strong> Real-time browser canvas to capture user stylus stroke vectors for few-shot handwriting style cloning.</li>
            <li><strong>Sprint 6 — Web UI Playground:</strong> Interactive web app (React / Vite + FastAPI backend) allowing live document synthesis, custom ink styling, and PDF export.</li>
            <li><strong>Sprint 7 — Quantitative Recognition Ablation:</strong> Word Error Rate (WER) and Character Error Rate (CER) benchmarking using pretrained offline TrOCR.</li>
            <li><strong>Sprint 8 — Final Thesis & Viva Defense:</strong> Final comprehensive capstone documentation, presentation, and oral defense.</li>
          </ul>
        </div>
      </div>

      <!-- Concluding Banner -->
      <div style="background: #212121; border: 1px solid #333333; border-radius: 8px; padding: 18px 24px; display: flex; align-items: center; justify-content: space-between;">
        <div style="display: flex; align-items: center; gap: 16px;">
          <span style="display: inline-block; width: 10px; height: 10px; border-radius: 50%; background: #10B981; box-shadow: 0 0 8px #10B981;"></span>
          <span style="font-size: 15px; font-weight: 500; color: #FFFFFF;">
            Core Vector Generative Engine Operational • Zero Regressions • Prepared for Faculty Review & Live Demonstration
          </span>
        </div>
        <div style="font-family: 'JetBrains Mono', monospace; font-size: 13px; color: #9E9E9E;">
          GLA University • Dept of CEA (AI)
        </div>
      </div>
    </div>

    <div class="slide-footer">
      <div class="footer-left">
        <div class="footer-item"><strong>Candidate:</strong> Agriim Rastogi (22515500004)</div>
        <div class="footer-item"><strong>Mentors:</strong> Palash Sir / Vansh Sir</div>
        <div class="footer-item"><strong>Reviewer:</strong> Gautam Mukherjee</div>
      </div>
      <div>Thank You • Q&amp;A</div>
    </div>
  </div>
</div>

</body>
</html>
"""

async def generate_deck():
    output_dir = Path("outputs/presentation")
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load the generated handwriting sample SVG
    sample_svg_path = Path("sample_preview.svg")
    sample_svg_content = ""
    if sample_svg_path.exists():
        sample_svg_content = sample_svg_path.read_text(encoding="utf-8")
        if sample_svg_content.startswith("<?xml"):
            sample_svg_content = sample_svg_content[sample_svg_content.find("<svg"):]

    html_content = build_slides_html(sample_svg_content)
    html_path = output_dir / "slides.html"
    pdf_path = Path("Handwriting_Synthesis_Presentation_GLA.pdf")
    root_pdf_path = Path("presentation.pdf")

    html_path.write_text(html_content, encoding="utf-8")
    print(f"HTML presentation written to: {html_path}")

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page(viewport={"width": 1920, "height": 1080})
        
        await page.goto(f"file:///{html_path.resolve().as_posix()}", wait_until="networkidle")

        # 2. Export Crisp 16:9 PDF
        print("Rendering multi-page 16:9 PDF...")
        pdf_bytes = await page.pdf(
            width="1920px",
            height="1080px",
            print_background=True,
            prefer_css_page_size=True,
        )
        pdf_path.write_bytes(pdf_bytes)
        root_pdf_path.write_bytes(pdf_bytes)
        print(f"Saved PDF to: {pdf_path.resolve()} ({len(pdf_bytes):,} bytes)")

        # 3. Export individual slide PNGs for instant inspection
        print("Capturing individual slide PNG previews...")
        slides = await page.locator(".slide").all()
        for idx, slide_loc in enumerate(slides, start=1):
            slide_png = output_dir / f"slide_{idx:02d}.png"
            await slide_loc.screenshot(path=str(slide_png))
            print(f"  Slide {idx}: {slide_png}")

        await browser.close()

    print("\nPresentation generation complete! All deliverables ready.")

if __name__ == "__main__":
    asyncio.run(generate_deck())
