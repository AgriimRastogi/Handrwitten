"""Continuous Handwriting Synthesis CLI using Continuous Guided-Attention Transformer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from hws.alphabet import Alphabet
from hws.preprocess import denoise, normalize_scale
from hws.render import InkStyle, PageLayout, render_page
from new_model.config import NewModelConfig
from new_model.model import ContinuousGuidedTransformer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Synthesize handwriting with Continuous Guided Transformer-MDN")
    parser.add_argument("--text", type=str, default="Online Handwriting Synthesis", help="Text to synthesize")
    parser.add_argument("--checkpoint", type=str, default=str(root_dir / "checkpoints" / "new_model" / "best_model.pt"), help="Path to checkpoint")
    parser.add_argument("--config", type=str, default=str(root_dir / "datas" / "processed" / "config.json"), help="Path to config.json")
    parser.add_argument("--alphabet", type=str, default=str(root_dir / "datas" / "processed" / "alphabet.json"), help="Path to alphabet.json")
    parser.add_argument("--out", type=str, default="outputs/continuous_sample.svg", help="Output SVG path")
    parser.add_argument("--writer", type=int, default=0, help="Writer style ID (0-196)")
    parser.add_argument("--bias", type=float, default=1.0, help="Neatness bias (>= 0.0)")
    parser.add_argument("--max-steps", type=int, default=800, help="Max stroke steps")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        print(f"Error: Checkpoint not found at {ckpt_path}")
        print("Please train the continuous model first using 'python scripts/train_new_model.py'")
        sys.exit(1)

    # 1. Resolve paths
    config_path = Path(args.config)
    if not config_path.is_file() and (root_dir / "datas" / "processed" / "config.json").is_file():
        config_path = root_dir / "datas" / "processed" / "config.json"

    alphabet_path = Path(args.alphabet)
    if not alphabet_path.is_file() and (root_dir / "datas" / "processed" / "alphabet.json").is_file():
        alphabet_path = root_dir / "datas" / "processed" / "alphabet.json"

    alphabet = Alphabet.load(alphabet_path)

    # 2. Load model
    config = NewModelConfig.from_config_json(config_path)
    model = ContinuousGuidedTransformer(config).to(device)

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # 3. Tokenize input text
    char_ids = alphabet.encode(args.text)
    char_tensor = torch.tensor([char_ids], dtype=torch.long, device=device)
    writer_tensor = torch.tensor([args.writer], dtype=torch.long, device=device)

    print(f"Synthesizing: '{args.text}' (Style: {args.writer}, Bias: {args.bias})")
    result = model.generate(
        chars=char_tensor,
        writer=writer_tensor,
        max_steps=args.max_steps,
        bias=args.bias,
    )

    coords = result["coords"]
    if len(coords) == 0:
        print("Warning: Model emitted 0 coordinates.")
        return

    # Denoise smooth continuous curves
    if coords.shape[0] >= 7:
        coords = denoise(coords, window=7, polyorder=3)

    # Level baseline drift: subtract fitted linear regression slope
    if coords.shape[0] >= 10:
        x_span = coords[:, 0].max() - coords[:, 0].min()
        if x_span > 1.0:
            poly = np.polyfit(coords[:, 0], coords[:, 1], deg=1)
            coords[:, 1] -= poly[0] * coords[:, 0]

        # Shift baseline to zero
        base = float(np.percentile(coords[:, 1], 15))
        coords[:, 1] -= base

        # Scale to fit standard ruled notebook height (~22 units)
        h_span = coords[:, 1].max() - coords[:, 1].min()
        if h_span > 2.0:
            scale = min(2.2, max(1.0, 22.0 / h_span))
            coords[:, :2] *= scale

    print(f"Generated {len(coords)} continuous stroke coordinates.")

    # 4. Render SVG
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    layout = PageLayout(ruled=True, margin_rule=True, line_height=46.0, margin_top=65.0, margin_left=65.0)
    style = InkStyle(color="#0f172a", width=1.4, bezier=True)
    svg = render_page([coords], layout=layout, style=style, seed=42)
    out_path.write_text(svg, encoding="utf-8")
    print(f"Saved SVG to: {out_path.resolve()}")


if __name__ == "__main__":
    main()
