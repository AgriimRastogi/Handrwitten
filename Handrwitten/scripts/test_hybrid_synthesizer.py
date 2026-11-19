"""Hybrid Stroke Synthesizer: Seamless Cursive + Glyph Trajectory Fusion."""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "Handrwitten"))

import numpy as np
import torch
import re
from hws.pretrained_model import PretrainedGravesSynthesis, normalize_scale, offsets_to_coords, denoise
from hws.render import render_line, render_page, PageLayout, InkStyle
from playwright.sync_api import sync_playwright


# Geometric stroke templates for rare/unsupported symbols that fail in pure cursive RNNs
# Stored as relative (dx, dy, pen_up) normalized to standard character height ~20.0
GLYPH_TEMPLATES = {
    '/': [
        # Upward-right diagonal slash: start bottom-left, draw to top-right, pen up
        np.array([
            [0.0, -8.0, 0.0, 0.02],
            [1.5, -4.0, 0.0, 0.02],
            [3.0, 0.0, 0.0, 0.02],
            [4.5, 4.0, 0.0, 0.02],
            [6.0, 8.0, 0.0, 0.02],
            [7.5, 12.0, 1.0, 0.02],  # pen up
        ], dtype=np.float32)
    ],
    '\\': [
        # Downward-right diagonal backslash
        np.array([
            [0.0, 12.0, 0.0, 0.02],
            [1.5, 8.0, 0.0, 0.02],
            [3.0, 4.0, 0.0, 0.02],
            [4.5, 0.0, 0.0, 0.02],
            [6.0, -4.0, 0.0, 0.02],
            [7.5, -8.0, 1.0, 0.02],
        ], dtype=np.float32)
    ],
    '-': [
        # Horizontal dash
        np.array([
            [0.0, 2.0, 0.0, 0.02],
            [2.5, 2.0, 0.0, 0.02],
            [5.0, 2.0, 0.0, 0.02],
            [7.5, 2.0, 1.0, 0.02],
        ], dtype=np.float32)
    ],
    '|': [
        # Vertical pipe
        np.array([
            [0.0, 12.0, 0.0, 0.02],
            [0.0, 6.0, 0.0, 0.02],
            [0.0, 0.0, 0.0, 0.02],
            [0.0, -6.0, 1.0, 0.02],
        ], dtype=np.float32)
    ]
}


def add_motor_jitter(stroke: np.ndarray, jitter_std: float = 0.25) -> np.ndarray:
    """Add subtle physiological tremor and velocity modulation to synthetic strokes."""
    res = stroke.copy()
    n = len(res)
    noise = np.random.normal(0, jitter_std, size=(n, 2))
    # Keep endpoints stable, add slight organic curve in middle
    weight = np.sin(np.linspace(0, np.pi, n))[:, None]
    res[:, :2] += noise * weight
    return res


def synthesize_hybrid(model: PretrainedGravesSynthesis, text: str, seed: int = 42, bias: float = 1.0) -> np.ndarray:
    """Synthesize text by fusing robust Graves cursive generation with clean symbol trajectory injection."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Check if text contains isolated repeated punctuation (e.g. ///// or -----)
    # or pure symbol sequences
    if re.fullmatch(r'[/\\|\-]+', text.strip()):
        # Pure symbol sequence like "/////"
        strokes = []
        cur_x = 0.0
        for ch in text.strip():
            tpl = GLYPH_TEMPLATES.get(ch, GLYPH_TEMPLATES['/'])[0]
            stk = add_motor_jitter(tpl)
            stk_shifted = stk.copy()
            stk_shifted[:, 0] += cur_x
            strokes.append(stk_shifted)
            cur_x += (stk[:, 0].max() - stk[:, 0].min()) + 6.0  # character spacing

        all_pts = np.vstack(strokes)
        return all_pts

    # Otherwise, use our adaptive dwell-governed Graves engine
    # Normalizing repeated punctuation with breathing spaces
    def space_repeats(match):
        return " " + " ".join(list(match.group(0))) + " "

    norm_text = re.sub(r'([/\\#\-\*\=\_\+])\1+', space_repeats, text)
    norm_text = re.sub(r'\s+', ' ', norm_text).strip()

    return model.generate(norm_text, bias=bias, seed=seed)


def main():
    model_path = REPO_ROOT / "Handrwitten" / "checkpoints" / "pretrained"
    model = PretrainedGravesSynthesis(model_path, bias=1.0)

    test_cases = [
        ("user_case_1_H", "hello world HHHHHHHHH"),
        ("user_case_2_slash", "/////"),
        ("stress_barcode_l", "llllllll"),
        ("stress_gla_dates", "GLA University - 2026 / 2027"),
        ("stress_project_title", "Stroke-Level Handwriting Synthesis"),
    ]

    out_dir = REPO_ROOT / "outputs" / "cool_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    style = InkStyle(color="#0f172a", width=1.6, bezier=True)

    print("Synthesizing showcase results...")
    results = {}
    for name, text in test_cases:
        coords = synthesize_hybrid(model, text, seed=42)
        svg = render_line(coords, style=style)
        svg_file = out_dir / f"{name}.svg"
        svg_file.write_text(svg, encoding="utf-8")
        results[name] = svg_file
        print(f"  {name:25s}: {len(coords)} points -> {svg_file.name}")

    print("Capturing PNGs via Playwright...")
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 900, "height": 300})
        for name, svg_file in results.items():
            page.goto(f"file:///{svg_file.resolve().as_posix()}")
            png_file = out_dir / f"{name}.png"
            page.screenshot(path=str(png_file))
            print(f"  Rendered: {png_file.name}")
        browser.close()

    print("All cool results successfully generated!")


if __name__ == "__main__":
    main()
