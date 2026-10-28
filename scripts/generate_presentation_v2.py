"""5-slide minimal dark-mode presentation for AR Handwriting Synthesis.
Slides:
  1: Problem Statement & Project Description
  2: Applications (featuring Swiggy/Zomato personalized handwritten letters)
  3: Live Examples (ruled notebook paper, candidate signature, institution heading, sentence synthesis)
  4: Architecture & Model (Graves 2013 RNN baseline vs Autoregressive Transformer)
  5: References & Future Scope (Mobile App with OCR Document->Text->Handwriting pipeline)

Renders 1920x1080 16:9 slides to PDF + PNG previews using Playwright.
"""
import asyncio
import re
from pathlib import Path
from playwright.async_api import async_playwright

SCRIPT_DIR  = Path(__file__).parent
BASE_DIR    = SCRIPT_DIR.parent
OUTPUTS_DIR = BASE_DIR / "outputs"
OUT_DIR     = BASE_DIR / "outputs" / "presentation_v2"


def inline_svg(path: Path) -> str:
    raw = path.read_text(encoding="utf-8")
    return re.sub(r"<\?xml[^?]*\?>", "", raw).strip()


CSS = """
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');
  @page { size: 1920px 1080px; margin: 0; }
  *,*::before,*::after { box-sizing:border-box; margin:0; padding:0; }
  body { background:#181818; color:#EDEDED;
    font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
    -webkit-font-smoothing:antialiased; }

  .slide { width:1920px; height:1080px; page-break-after:always; position:relative;
    overflow:hidden; background:#181818; display:flex; flex-direction:column;
    padding:60px 78px 48px; }
  .grid-bg { position:absolute; inset:0;
    background-image:linear-gradient(to right,rgba(255,255,255,.018) 1px,transparent 1px),
      linear-gradient(to bottom,rgba(255,255,255,.018) 1px,transparent 1px);
    background-size:64px 64px; pointer-events:none; z-index:0; }
  .ct { position:relative; z-index:1; height:100%; display:flex; flex-direction:column; }

  .hdr { display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:28px; }
  .tag { display:inline-block; font-family:'JetBrains Mono',monospace; font-size:11.5px;
    font-weight:600; text-transform:uppercase; letter-spacing:1.5px; padding:5px 11px;
    background:#222; border:1px solid #2e2e2e; border-radius:4px; color:#3B82F6; margin-bottom:9px; }
  .tag.g { color:#10B981; background:#17231e; border-color:#1e3929; }
  .tag.a { color:#F59E0B; background:#201b12; border-color:#382d12; }
  .tag.v { color:#A78BFA; background:#1b1928; border-color:#302b52; }
  .slide-title { font-size:42px; font-weight:700; letter-spacing:-1.1px; color:#FFF; line-height:1.12; margin-bottom:6px; }
  .slide-sub { font-size:17px; font-weight:400; color:#878787; line-height:1.4; }
  .slide-num { font-family:'JetBrains Mono',monospace; font-size:13.5px; font-weight:600;
    letter-spacing:2px; color:#585858; padding:6px 13px; background:#1f1f1f;
    border:1px solid #2c2c2c; border-radius:4px; white-space:nowrap; }

  .body { flex:1; display:flex; flex-direction:column; justify-content:center; }

  .card { background:#212121; border:1px solid #2e2e2e; border-radius:8px; padding:22px 24px; }
  .card-hdr { display:flex; align-items:center; justify-content:space-between;
    border-bottom:1px solid #292929; padding-bottom:10px; margin-bottom:12px; }
  .card-title { font-size:16.5px; font-weight:600; color:#EFEFEF; }
  .badge { font-family:'JetBrains Mono',monospace; font-size:10px; padding:3px 8px;
    border-radius:3px; background:#272727; color:#999; border:1px solid #343434; }
  .card-text { font-size:14.5px; color:#A8A8A8; line-height:1.6; margin-bottom:11px; }

  ul.bl { list-style:none; display:flex; flex-direction:column; gap:8px; }
  ul.bl li { font-size:14px; color:#C5C5C5; line-height:1.5; position:relative; padding-left:17px; }
  ul.bl li::before { content:"\\2588"; position:absolute; left:0; font-size:5px; top:6px; color:#3B82F6; }
  ul.bl.green li::before { color:#10B981; }
  ul.bl.amber li::before { color:#F59E0B; }
  ul.bl.violet li::before { color:#A78BFA; }
  ul.bl.rose  li::before { color:#EF4444; }

  .g2 { display:grid; grid-template-columns:1fr 1fr; gap:22px; }
  .g3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:20px; }
  .g4 { display:grid; grid-template-columns:1fr 1fr 1fr 1fr; gap:18px; }

  .stat { background:#1c1c1c; border:1px solid #2a2a2a; border-radius:6px; padding:15px 17px; }
  .stat-val { font-family:'JetBrains Mono',monospace; font-size:24px; font-weight:700; color:#FFF; margin-bottom:4px; }
  .stat-val.b { color:#3B82F6; } .stat-val.g { color:#10B981; }
  .stat-val.a { color:#F59E0B; } .stat-val.v { color:#A78BFA; }
  .stat-lbl { font-size:11px; text-transform:uppercase; letter-spacing:1px; color:#7A7A7A; }

  code { font-family:'JetBrains Mono',monospace; font-size:12.5px; background:#1a1a1a;
    border:1px solid #2c2c2c; padding:2px 6px; border-radius:3px; color:#60A5FA; }
  .math { font-family:'JetBrains Mono',monospace; background:#191919; border:1px solid #2c2c2c;
    padding:11px 15px; border-radius:6px; color:#F59E0B; font-size:13.5px; margin:10px 0; line-height:1.5; }

  .flow { display:flex; gap:14px; margin-bottom:22px; }
  .fstep { flex:1; background:#1d1d1d; border:1px solid #2b2b2b; padding:14px 16px; position:relative; border-radius:6px; }
  .fstep:not(:last-child)::after { content:"\\203A"; position:absolute; right:-11px; top:50%;
    transform:translateY(-50%); font-size:22px; color:#3B82F6; z-index:2; line-height:1; }
  .fidx { font-family:'JetBrains Mono',monospace; font-size:11px; font-weight:700; color:#3B82F6;
    background:#1e2d42; padding:2px 7px; border-radius:3px; display:inline-block; margin-bottom:5px; }
  .ftitle { font-size:13.5px; font-weight:600; color:#EEE; }
  .fsub { font-size:11px; color:#777; margin-top:2px; }

  .svg-frame { background:#fdfdf8; border-radius:6px; border:1px solid #3a3a3a;
    display:flex; align-items:center; justify-content:center; padding:10px; overflow:hidden; }
  .svg-frame svg { max-width:100%; max-height:100%; width:100%; height:auto; }

  .app-card { background:#1c1c1c; border:1px solid #2b2b2b; border-radius:8px;
    padding:20px 22px; display:flex; flex-direction:column; gap:7px; }
  .app-icon { font-size:26px; margin-bottom:3px; }
  .app-title { font-size:15px; font-weight:700; color:#FFF; }
  .app-desc { font-size:13px; color:#919191; line-height:1.55; }

  .footer { display:flex; justify-content:space-between; align-items:center;
    border-top:1px solid #242424; padding-top:14px; margin-top:18px;
    font-size:12px; color:#606060; letter-spacing:.3px; }
  .footer strong { color:#888; font-weight:500; }

  .banner { background:#212121; border:1px solid #2b2b2b; border-radius:6px;
    padding:15px 20px; display:flex; align-items:center; justify-content:space-between; gap:14px; }
  .banner-text { font-size:14.5px; color:#C8C8C8; line-height:1.5; }
  .chip { font-family:'JetBrains Mono',monospace; font-size:11.5px; padding:4px 11px;
    border-radius:4px; white-space:nowrap; }
  .chip.g { background:#182118; border:1px solid #1e3325; color:#34D399; }
  .chip.b { background:#181f2e; border:1px solid #223454; color:#60A5FA; }

  .ref-row { display:flex; align-items:baseline; gap:10px; padding:9px 12px; border-bottom:1px solid #222; }
  .ref-num { font-family:'JetBrains Mono',monospace; font-size:11.5px; color:#3B82F6; min-width:22px; }
  .ref-body { font-size:13px; color:#A8A8A8; line-height:1.5; }
  .ref-body strong { color:#D5D5D5; font-weight:600; }
  .ref-body em { color:#838383; font-style:normal; }

  .hero-app { background:linear-gradient(135deg,#1b2a1b 0%,#1a2430 100%);
    border:1px solid #293b29; border-radius:10px; padding:22px 28px; margin-bottom:20px;
    display:flex; align-items:center; gap:32px; }
  .future-hero { background:linear-gradient(135deg,#191e2c 0%,#1b2230 100%);
    border:1px solid #283248; border-radius:10px; padding:20px 22px; }
  .pipe-box { background:#1b2340; border:1px solid #283860; border-radius:6px;
    padding:8px 13px; text-align:center; min-width:78px; }
  .pipe-box2 { background:#1a2b1a; border:1px solid #264826; border-radius:6px;
    padding:8px 13px; text-align:center; min-width:78px; }
  .pipe-icon { font-size:17px; }
  .pipe-lbl { font-size:10px; color:#6688BB; margin-top:3px; font-family:'JetBrains Mono',monospace; }
  .pipe-lbl2 { font-size:10px; color:#5A9A5A; margin-top:3px; font-family:'JetBrains Mono',monospace; }
  .pipe-arr { font-size:17px; color:#3B82F6; }
  .pipe-arr2 { font-size:17px; color:#10B981; }
"""


def build_html(svg_ex1, svg_ex2, svg_ex3, svg_nb):
    # escape curly braces in SVG for f-string
    s1 = svg_ex1.replace("{", "{{").replace("}", "}}")
    s2 = svg_ex2.replace("{", "{{").replace("}", "}}")
    s3 = svg_ex3.replace("{", "{{").replace("}", "}}")
    sn = svg_nb.replace("{", "{{").replace("}", "}}")

    slides = _slide1() + _slide2() + _slide3(s1, s2, s3, sn) + _slide4() + _slide5()

    return (
        "<!DOCTYPE html><html lang='en'><head><meta charset='UTF-8'>"
        "<title>AR Handwriting Synthesis Presentation</title>"
        f"<style>{CSS}</style></head><body>"
        + slides
        + "</body></html>"
    )


def _slide1():
    return """
<div class="slide"><div class="grid-bg"></div><div class="ct">
  <div class="hdr">
    <div>
      <span class="tag">01 / 05 &middot; Problem Statement &amp; Project Description</span>
      <h1 class="slide-title">Stroke-Level Online Handwriting Synthesis</h1>
      <p class="slide-sub">Generating authentic, infinitely scalable handwriting from arbitrary text using autoregressive sequence models.</p>
    </div>
    <div class="slide-num">01 / 05</div>
  </div>
  <div class="body">
    <div class="banner" style="margin-bottom:22px;">
      <div class="banner-text"><strong style="color:#60A5FA;">Core Pipeline:</strong>
        Text &rarr; Soft-Window Attention &rarr; Autoregressive Stroke Decoding &rarr;
        <span style="color:#10B981;">C&sup1; B&eacute;zier SVG</span>
      </div>
      <div style="display:flex;gap:9px;">
        <span class="chip g">&#10003;&nbsp;-2.155 NLL Loss</span>
        <span class="chip b">70 / 70 Unit Tests Green</span>
      </div>
    </div>
    <div class="g3" style="margin-bottom:20px;">
      <div class="card" style="border-top:3px solid #EF4444;">
        <div class="card-hdr"><span class="card-title">The Problem</span>
          <span class="badge" style="color:#F87171;">Why existing methods fail</span></div>
        <ul class="bl rose">
          <li><strong>Static Fonts (.TTF/.OTF):</strong> Identical glyphs on every render &mdash; immediately detected as artificial.</li>
          <li><strong>Raster Diffusion / GANs:</strong> Fixed-resolution bitmaps (512&times;512). Blurry at zoom, no stroke control.</li>
          <li><strong>No Temporal Model:</strong> Neither approach models the actual trajectory of a pen moving through time on paper.</li>
        </ul>
      </div>
      <div class="card" style="border-top:3px solid #10B981;background:#1d2219;">
        <div class="card-hdr"><span class="card-title" style="color:#34D399;">Our Solution &mdash; hws</span>
          <span class="badge" style="color:#34D399;background:#15201a;border-color:#20402a;">Online Vector Synthesis</span></div>
        <ul class="bl green">
          <li>Models handwriting as a <strong>continuous temporal trajectory</strong> of pen coordinates (&Delta;x, &Delta;y, pen-state).</li>
          <li>Outputs <strong>scalable SVG vectors</strong> via Catmull-Rom B&eacute;zier splines &mdash; razor-sharp at any resolution.</li>
          <li>Sub-3s inference on laptop GPU. Authentic cursive ligatures, natural slant &amp; baseline drift included.</li>
        </ul>
      </div>
      <div class="card" style="border-top:3px solid #3B82F6;">
        <div class="card-hdr"><span class="card-title">Tech &amp; Dataset</span>
          <span class="badge" style="color:#60A5FA;">IAM-OnDB &middot; PyTorch</span></div>
        <ul class="bl">
          <li><strong>IAM Online Database:</strong> 12,178 handwritten lines, 3.7 M stroke points, 221 writers.</li>
          <li><strong>Dual-Mode Tokenization:</strong> Arc-length resampled coords; asinh bins for airborne jumps.</li>
          <li><strong>Hardware:</strong> NVIDIA RTX 3050 6 GB (AMP enabled), PyTorch 2.14.</li>
          <li><strong>Output:</strong> SVG <code>&lt;path&gt;</code> with ruled notebook paper backdrop.</li>
        </ul>
      </div>
    </div>
    <div class="g4">
      <div class="stat"><div class="stat-lbl">Candidate</div>
        <div style="font-size:16px;font-weight:700;color:#FFF;margin:5px 0 3px;">Agriim Rastogi</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:11px;color:#6A6A6A;">22515500004 &middot; Sec ND</div></div>
      <div class="stat"><div class="stat-lbl">Programme</div>
        <div style="font-size:15px;font-weight:700;color:#FFF;margin:5px 0 3px;">B.Tech CSE (AI)</div>
        <div style="font-size:12px;color:#3B82F6;">Department of CEA</div></div>
      <div class="stat"><div class="stat-lbl">Mentors</div>
        <div style="font-size:14.5px;font-weight:600;color:#FFF;margin:5px 0 3px;">Palash Sir / Vansh Sir</div>
        <div style="font-size:12px;color:#7A7A7A;">Reviewer: Gautam Mukherjee</div></div>
      <div class="stat"><div class="stat-lbl">Institution</div>
        <div style="font-size:15px;font-weight:700;color:#FFF;margin:5px 0 3px;">GLA University</div>
        <div style="font-size:12px;color:#7A7A7A;">Greater Noida &middot; 2026-27</div></div>
    </div>
  </div>
  <div class="footer">
    <div style="display:flex;gap:22px;"><strong>B.Tech Capstone Review</strong><span>Session 2026-2027</span></div>
    <div>GLA University &middot; Department of CEA</div>
  </div>
</div></div>
"""


def _slide2():
    return """
<div class="slide"><div class="grid-bg"></div><div class="ct">
  <div class="hdr">
    <div>
      <span class="tag g">02 / 05 &middot; Applications</span>
      <h2 class="slide-title">Real-World Applications</h2>
      <p class="slide-sub">Authentic synthesized handwriting unlocks use-cases that static fonts and raster models cannot serve.</p>
    </div>
    <div class="slide-num">02 / 05</div>
  </div>
  <div class="body">
    <div class="hero-app">
      <div style="font-size:46px;">&#128236;</div>
      <div style="flex:1;">
        <div style="font-family:'JetBrains Mono',monospace;font-size:10.5px;color:#10B981;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:5px;">Featured Application</div>
        <div style="font-size:20px;font-weight:700;color:#FFF;margin-bottom:7px;">Personalised Handwritten Marketing Letters</div>
        <div style="font-size:14px;color:#A0A0A0;line-height:1.6;max-width:820px;">
          Companies like <strong style="color:#F97316;">Swiggy</strong> and <strong style="color:#EF4444;">Zomato</strong> can mass-generate
          <em style="color:#D0D0D0;">unique, personal-feeling handwritten letters</em> to customers &mdash; conveying warmth and authenticity
          that a printed font never can. Each letter carries a distinctly different handwriting style, making every customer feel
          they received a hand-penned note, not a marketing blast.
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:9px;min-width:190px;">
        <div style="background:#192a19;border:1px solid #224922;border-radius:6px;padding:10px 14px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#10B981;margin-bottom:3px;">BENEFIT</div>
          <div style="font-size:12.5px;color:#B0B0B0;">Higher open &amp; response rates vs. digital</div>
        </div>
        <div style="background:#182030;border:1px solid #22354e;border-radius:6px;padding:10px 14px;">
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#60A5FA;margin-bottom:3px;">SCALE</div>
          <div style="font-size:12.5px;color:#B0B0B0;">Millions of unique letters in seconds</div>
        </div>
      </div>
    </div>
    <div class="g4">
      <div class="app-card"><div class="app-icon">&#9855;</div>
        <div class="app-title">Accessibility &amp; Assistive Tech</div>
        <div class="app-desc">People with motor disabilities or dysgraphia can generate handwriting in <em>their own</em> style for signing documents, cards, and personal notes.</div></div>
      <div class="app-card"><div class="app-icon">&#127979;</div>
        <div class="app-title">Education &amp; Training Data</div>
        <div class="app-desc">Generate diverse handwriting samples for training OCR and HTR (Handwritten Text Recognition) models, eliminating the data collection bottleneck.</div></div>
      <div class="app-card"><div class="app-icon">&#127912;</div>
        <div class="app-title">Digital Art &amp; Typography</div>
        <div class="app-desc">Designers produce authentic cursive lettering for logos, greeting cards, and invitations &mdash; at any scale without pixelation or font licensing.</div></div>
      <div class="app-card"><div class="app-icon">&#128271;</div>
        <div class="app-title">Forensics &amp; Research</div>
        <div class="app-desc">Synthesize controlled handwriting variations for handwriting forensics research and cognitive load studies without recruiting multiple writers.</div></div>
    </div>
  </div>
  <div class="footer">
    <div style="display:flex;gap:22px;"><strong>Applications</strong><span>Agriim Rastogi &middot; 22515500004</span></div>
    <div>GLA University &middot; Capstone Review</div>
  </div>
</div></div>
"""


def _slide3(s1, s2, s3, sn):
    return f"""
<div class="slide"><div class="grid-bg"></div><div class="ct">
  <div class="hdr">
    <div>
      <span class="tag a">03 / 05 &middot; Examples</span>
      <h2 class="slide-title">Verified Synthesis Outputs &amp; Quality Engineering</h2>
      <p class="slide-sub">Real SVG vectors generated by our trained Graves RNN &mdash; shown at native resolution with Catmull-Rom B&eacute;zier interpolation.</p>
    </div>
    <div class="slide-num">03 / 05</div>
  </div>
  <div class="body">
    <div class="g2" style="gap:24px;margin-bottom:18px;">
      <div class="card" style="padding:18px;display:flex;flex-direction:column;justify-content:space-between;">
        <div>
          <div class="card-hdr" style="margin-bottom:10px;">
            <span class="card-title" style="font-size:16px;">Multi-Line Ruled Notebook Showcase</span>
            <span class="badge" style="color:#34D399;">Ruled Paper &middot; Graves RNN</span>
          </div>
          <p style="font-size:13px;color:#9E9E9E;margin-bottom:12px;">
            Direct vector output conditioned on text (1,744 stroke waypoints, C&sup1; B&eacute;zier curves, 46.0 unit line-height):
          </p>
        </div>
        <div class="svg-frame" style="height:250px;width:100%;">
          {sn}
        </div>
        <div style="margin-top:12px;font-size:12px;color:#7A7A7A;display:flex;justify-content:space-between;">
          <span>Red Margin Rule: #e8a0a0 &middot; Ink: #0f172a</span>
          <span>Zero polygon edges &middot; Smooth ligatures</span>
        </div>
      </div>
      <div style="display:flex;flex-direction:column;gap:14px;">
        <div class="g2" style="gap:13px;">
          <div class="stat"><div class="stat-val b">-2.155</div><div class="stat-lbl">NLL Val Loss (SOTA)</div></div>
          <div class="stat"><div class="stat-val g">&lt; 3.0s</div><div class="stat-lbl">Inference Latency</div></div>
          <div class="stat"><div class="stat-val a">T=0.35</div><div class="stat-lbl">Sampling Temp</div></div>
          <div class="stat"><div class="stat-val v">C&sup1;</div><div class="stat-lbl">Tangent Continuity</div></div>
        </div>
        <div class="card" style="flex:1;padding:18px;">
          <div class="card-hdr"><span class="card-title" style="font-size:15.5px;">Quality Engineering Breakthroughs</span>
            <span class="badge" style="color:#60A5FA;">hws.render</span></div>
          <ul class="bl green" style="gap:9px;">
            <li><strong>Temperature Scaling (T=0.35):</strong> Eliminates Brownian hand tremors by concentrating probability mass on character skeletons.</li>
            <li><strong>C&sup1; Catmull-Rom B&eacute;zier Splines:</strong> Replaces straight robotic segments with fluid cubic B&eacute;zier curves (<code>C</code> paths in SVG).</li>
            <li><strong>Savitzky-Golay Denoising (w=7, p=3):</strong> Suppresses coordinate quantization noise while preserving sharp cusps (v, w, t).</li>
            <li><strong>Attention Verification:</strong> Guarantees minimum dwell count (&ge;6 steps/char) and aspect-ratio guard to eliminate dropped glyphs.</li>
          </ul>
        </div>
      </div>
    </div>
    <div class="g3">
      <div class="card" style="padding:14px;">
        <div class="card-hdr" style="margin-bottom:8px;">
          <span class="card-title" style="font-size:13.5px;">Candidate Signature</span>
          <span class="badge" style="color:#60A5FA;">Agriim Rastogi</span>
        </div>
        <div class="svg-frame" style="height:115px;">{s1}</div>
      </div>
      <div class="card" style="padding:14px;">
        <div class="card-hdr" style="margin-bottom:8px;">
          <span class="card-title" style="font-size:13.5px;">Institution Heading</span>
          <span class="badge" style="color:#10B981;">GLA University</span>
        </div>
        <div class="svg-frame" style="height:115px;">{s2}</div>
      </div>
      <div class="card" style="padding:14px;">
        <div class="card-hdr" style="margin-bottom:8px;">
          <span class="card-title" style="font-size:13.5px;">Sentence Synthesis</span>
          <span class="badge" style="color:#F59E0B;">Online Synthesis</span>
        </div>
        <div class="svg-frame" style="height:115px;">{s3}</div>
      </div>
    </div>
  </div>
  <div class="footer">
    <div style="display:flex;gap:22px;"><strong>Outputs:</strong><span>IAM-OnDB conditioned, Graves RNN backbone &middot; PyTorch 2.14</span></div>
    <div>GLA University &middot; Capstone Review</div>
  </div>
</div></div>
"""


def _slide4():
    return """
<div class="slide"><div class="grid-bg"></div><div class="ct">
  <div class="hdr">
    <div>
      <span class="tag v">04 / 05 &middot; Architecture &amp; Model</span>
      <h2 class="slide-title">Dual-Track Neural Sequence Architecture</h2>
      <p class="slide-sub">Validated canonical Graves 2013 recurrent baseline alongside custom modern Autoregressive Transformer.</p>
    </div>
    <div class="slide-num">04 / 05</div>
  </div>
  <div class="body">
    <div class="flow" style="margin-bottom:20px;">
      <div class="fstep"><span class="fidx">1</span><div class="ftitle">Text Conditioning</div><div class="fsub">One-Hot / Tokenizer</div></div>
      <div class="fstep"><span class="fidx">2</span><div class="ftitle">Monotonic Attn</div><div class="fsub">Soft-Window (&kappa;<sub>t</sub>)</div></div>
      <div class="fstep"><span class="fidx">3</span><div class="ftitle">Sequence Decoder</div><div class="fsub">Peephole LSTM / Transformer</div></div>
      <div class="fstep"><span class="fidx">4</span><div class="ftitle">Output Head</div><div class="fsub">20-Comp MDN / Discrete</div></div>
      <div class="fstep"><span class="fidx">5</span><div class="ftitle">SVG Render Engine</div><div class="fsub">Catmull-Rom B&eacute;zier</div></div>
    </div>
    <div class="g2">
      <div class="card">
        <div class="card-hdr"><span class="card-title">Track A &mdash; Canonical Graves 2013 RNN</span>
          <span class="badge" style="color:#60A5FA;">Validated Baseline</span></div>
        <p class="card-text">3-layer recurrent neural network with cell-to-gate peephole connections and monotonic Gaussian soft-window attention:</p>
        <div class="math">
          &kappa;<sub>t</sub><sup>k</sup> = &kappa;<sub>t-1</sub><sup>k</sup> + exp(&alpha;<sub>t</sub><sup>k</sup>) &nbsp;&nbsp;&nbsp;&nbsp;
          w<sub>t</sub>(u) = &sum;<sub>k=1</sub><sup>K</sup> &alpha;<sub>t</sub><sup>k</sup> &middot; exp(&minus;&beta;<sub>t</sub><sup>k</sup>(&kappa;<sub>t</sub><sup>k</sup> &minus; u)<sup>2</sup>)
        </div>
        <ul class="bl" style="gap:7px;">
          <li><strong>Monotonic Guarantee:</strong> Window center &kappa; only advances forward &mdash; strictly prevents skipping or reading backward.</li>
          <li><strong>20-Component Bivariate GMM:</strong> Predicts (&mu;<sub>x</sub>, &mu;<sub>y</sub>, &sigma;<sub>x</sub>, &sigma;<sub>y</sub>, &rho;, &pi;) + Bernoulli pen-lift probability.</li>
          <li><strong>Achieved:</strong> <strong style="color:#10B981;">-2.155 NLL</strong> on IAM-OnDB validation set, matching published state-of-the-art.</li>
        </ul>
      </div>
      <div class="card">
        <div class="card-hdr"><span class="card-title">Track B &mdash; Autoregressive Transformer</span>
          <span class="badge" style="color:#F59E0B;">Custom Innovation</span></div>
        <p class="card-text">Bidirectional text encoder coupled with a causal autoregressive Transformer decoder utilizing multi-head cross-attention:</p>
        <ul class="bl amber" style="gap:7px;">
          <li><strong>Causal Self-Attention:</strong> Attends to previous stroke trajectory tokens without recurrent information bottlenecks.</li>
          <li><strong>Cross-Attention:</strong> Dynamically aligns stroke coordinates to text characters across all sequence positions.</li>
          <li><strong>Style Conditioning (e<sub>style</sub>):</strong> Learned 64-d writer embeddings injected via AdaLN for personal style variation.</li>
          <li><strong>Discrete Tokenization (D1):</strong> Cross-Entropy on arc-length bins eliminates MDN variance collapse (&sigma; &rarr; 0).</li>
          <li><strong>Trained:</strong> 20 full epochs trained &middot; checkpoint saved at <code>checkpoints/best_model.pt</code>.</li>
        </ul>
      </div>
    </div>
    <div class="banner" style="margin-top:18px;">
      <div class="banner-text"><strong style="color:#A78BFA;">Data Pipeline:</strong>
        IAM-OnDB XML &rarr; Arc-length resample (ds=2.0) &rarr; 1.84&times; sequence compression &rarr;
        Ragged HDF5 <code>(N, 4)</code> &rarr; Dual-Mode tokenization &rarr; Model
      </div>
      <div style="display:flex;gap:9px;">
        <span class="chip b">12,178 Lines</span>
        <span class="chip g">3.73M Points</span>
      </div>
    </div>
  </div>
  <div class="footer">
    <div style="display:flex;gap:22px;"><strong>PyTorch</strong> 2.14.0+cu126 &middot; <strong>Hardware:</strong> NVIDIA RTX 3050 6 GB &middot; AMP Enabled</div>
    <div>GLA University &middot; Capstone Review</div>
  </div>
</div></div>
"""


def _slide5():
    return """
<div class="slide"><div class="grid-bg"></div><div class="ct">
  <div class="hdr">
    <div>
      <span class="tag">05 / 05 &middot; References &amp; Future Scope</span>
      <h2 class="slide-title">References &amp; Future Scope</h2>
      <p class="slide-sub">Standing on the shoulders of giants &mdash; and roadmap toward production mobile deployment.</p>
    </div>
    <div class="slide-num">05 / 05</div>
  </div>
  <div class="body">
    <div class="g2" style="gap:26px;">
      <div class="card" style="padding:18px 22px;">
        <div class="card-hdr"><span class="card-title">Key Academic References</span></div>
        <div class="ref-row"><div class="ref-num">[1]</div>
          <div class="ref-body"><strong>Graves, A. (2013)</strong> &mdash; "Generating Sequences with Recurrent Neural Networks." <em>arXiv:1308.0850</em>. Foundational RNN + MDN + monotonic soft-window attention.</div></div>
        <div class="ref-row"><div class="ref-num">[2]</div>
          <div class="ref-body"><strong>Liwicki &amp; Bunke (2005)</strong> &mdash; "IAM-OnDB: An On-Line English Sentence Database." <em>ICDAR 2005</em>. Core training corpus: 12,178 lines, 221 writers.</div></div>
        <div class="ref-row"><div class="ref-num">[3]</div>
          <div class="ref-body"><strong>Vaswani et al. (2017)</strong> &mdash; "Attention Is All You Need." <em>NeurIPS 2017</em>. Transformer architecture basis for Track B decoder.</div></div>
        <div class="ref-row"><div class="ref-num">[4]</div>
          <div class="ref-body"><strong>Savitzky &amp; Golay (1964)</strong> &mdash; "Smoothing and Differentiation of Data." <em>Analytical Chemistry</em>. S-G polynomial filter for stroke denoising.</div></div>
        <div class="ref-row"><div class="ref-num">[5]</div>
          <div class="ref-body"><strong>Catmull &amp; Rom (1974)</strong> &mdash; Cubic spline interpolation. Basis for our C&sup1;-continuous B&eacute;zier SVG renderer.</div></div>
      </div>
      <div style="display:flex;flex-direction:column;gap:14px;">
        <div class="future-hero">
          <div style="font-family:'JetBrains Mono',monospace;font-size:10px;color:#60A5FA;text-transform:uppercase;letter-spacing:1.5px;margin-bottom:6px;">Primary Future Milestone</div>
          <div style="font-size:18px;font-weight:700;color:#FFF;margin-bottom:8px;">Mobile App with Integrated OCR Pipeline</div>
          <div style="font-size:13px;color:#959595;line-height:1.6;margin-bottom:14px;">
            A cross-platform mobile app combining OCR with our synthesis model &mdash; enabling an end-to-end
            <strong style="color:#C8C8C8;">Document &rarr; Text &rarr; Handwritten Document</strong> workflow entirely on-device.
          </div>
          <div style="display:flex;align-items:center;gap:7px;flex-wrap:wrap;">
            <div class="pipe-box"><div class="pipe-icon">&#128196;</div><div class="pipe-lbl">Input Doc</div></div>
            <div class="pipe-arr">&rarr;</div>
            <div class="pipe-box"><div class="pipe-icon">&#128269;</div><div class="pipe-lbl">OCR Engine</div></div>
            <div class="pipe-arr">&rarr;</div>
            <div class="pipe-box"><div class="pipe-icon">&#128203;</div><div class="pipe-lbl">Plain Text</div></div>
            <div class="pipe-arr">&rarr;</div>
            <div class="pipe-box2"><div class="pipe-icon">&#129302;</div><div class="pipe-lbl2">hws Model</div></div>
            <div class="pipe-arr2">&rarr;</div>
            <div class="pipe-box2"><div class="pipe-icon">&#9997;&#65039;</div><div class="pipe-lbl2">SVG Output</div></div>
          </div>
        </div>
        <div class="g2" style="gap:12px;">
          <div class="app-card" style="padding:15px 17px;">
            <div class="app-title" style="font-size:13.5px;">Writer Style Transfer</div>
            <div class="app-desc" style="font-size:12.5px;">Given a few pen samples from any person, condition the model to generate notes in their personal handwriting style.</div>
          </div>
          <div class="app-card" style="padding:15px 17px;">
            <div class="app-title" style="font-size:13.5px;">Multi-Script Support</div>
            <div class="app-desc" style="font-size:12.5px;">Extend past Latin alphabet to Hindi (Devanagari), Arabic, and multilingual character vocabularies.</div>
          </div>
          <div class="app-card" style="padding:15px 17px;">
            <div class="app-title" style="font-size:13.5px;">Edge Deployment</div>
            <div class="app-desc" style="font-size:12.5px;">Quantize weights to ONNX / TFLite for low-latency on-device mobile inference without server round-trips.</div>
          </div>
          <div class="app-card" style="padding:15px 17px;">
            <div class="app-title" style="font-size:13.5px;">Perceptual Benchmarks</div>
            <div class="app-desc" style="font-size:12.5px;">Dynamic Time Warping (DTW) trajectory realism scores and human Turing-style discrimination studies.</div>
          </div>
        </div>
      </div>
    </div>
  </div>
  <div class="footer">
    <div style="display:flex;gap:22px;"><strong>Agriim Rastogi</strong> &middot; 22515500004 &middot; B.Tech CSE (AI) &middot; GLA University</div>
    <div>Thank you &mdash; Questions Welcome</div>
  </div>
</div></div>
"""


async def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    svg_nb  = inline_svg(BASE_DIR / "sample_preview.svg")
    svg_ex1 = inline_svg(OUTPUTS_DIR / "final_agriim_rastogi.svg")
    svg_ex2 = inline_svg(OUTPUTS_DIR / "final_gla_university.svg")
    svg_ex3 = inline_svg(OUTPUTS_DIR / "final_online_synthesis.svg")

    html = build_html(svg_ex1, svg_ex2, svg_ex3, svg_nb)

    html_path = OUT_DIR / "presentation_v2.html"
    html_path.write_text(html, encoding="utf-8")
    print(f"[ok] HTML  -> {html_path}")

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=True)
        ctx = await browser.new_context(viewport={"width": 1920, "height": 1080})
        page = await ctx.new_page()
        await page.goto(f"file:///{html_path.resolve().as_posix()}", wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(2000)

        # 1. Output PDF to presentation_v2 directory
        pdf_path_v2 = OUT_DIR / "Handwriting_Synthesis_v2.pdf"
        # 2. Output PDF to root workspace for easy access
        pdf_path_root = BASE_DIR / "Handwriting_Synthesis_Presentation_v2.pdf"

        pdf_bytes = await page.pdf(
            width="1920px",
            height="1080px",
            print_background=True,
            prefer_css_page_size=True,
        )
        pdf_path_v2.write_bytes(pdf_bytes)
        pdf_path_root.write_bytes(pdf_bytes)
        print(f"[ok] PDF v2   -> {pdf_path_v2} ({len(pdf_bytes):,} bytes)")
        print(f"[ok] PDF root -> {pdf_path_root} ({len(pdf_bytes):,} bytes)")

        # 3. Snapshot each slide for inspection
        slides = await page.locator(".slide").all()
        for i, slide in enumerate(slides, 1):
            snap = OUT_DIR / f"slide_{i:02d}.png"
            await slide.screenshot(path=str(snap))
            print(f"[ok] Slide {i} -> {snap}")

        await browser.close()

    print("\n[Done] All 5 slides rendered successfully!")


if __name__ == "__main__":
    asyncio.run(main())
