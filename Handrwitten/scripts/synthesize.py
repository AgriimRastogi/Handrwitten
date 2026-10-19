"""Standalone synthesis CLI & document composer.

Synthesize arbitrary text into clean, vector-rendered SVG handwriting documents:

    python scripts/synthesize.py --text "Agriim Rastogi\nGLA University\nProject 2026" --ruled --margin-rule
    python scripts/synthesize.py --file my_notes.txt --ruled --out outputs/notebook.svg

Supports multi-line formatting, automatic word-wrapping, writer style conditioning,
and notebook paper layout (blue horizontal lines + red Indian school margin).
"""

from __future__ import annotations

import argparse
import logging
import sys
import textwrap
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hws.alphabet import Alphabet
from hws.graves_model import GravesConfig, GravesHandwritingModel
from hws.model import HandwritingSynthesisModel, ModelConfig
from hws.pretrained_model import PretrainedGravesSynthesis
from hws.render import InkStyle, PageLayout, render_line, render_page
from hws.tokenizer import StrokeTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("synthesize")


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Synthesize text into realistic multi-line SVG handwriting.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--text", type=str, help="Text to synthesize (use '\\n' for new lines)")
    group.add_argument("--file", type=Path, help="Text file containing content to synthesize")

    parser.add_argument("--checkpoint", type=Path, default=None,
                        help="Path to trained model checkpoint (.pt). Defaults to best_graves_model.pt if present, else best_model.pt")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "datas" / "processed" / "config.json")
    parser.add_argument("--alphabet", type=Path, default=REPO_ROOT / "datas" / "processed" / "alphabet.json")
    parser.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "datas" / "processed" / "tokenizer.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "synthesized_document.svg",
                        help="Output SVG file path")

    # Presentation & Layout
    parser.add_argument("--ruled", action="store_true", help="Draw blue horizontal notebook ruling lines")
    parser.add_argument("--margin-rule", action="store_true", help="Draw vertical red left margin line")
    parser.add_argument("--color", type=str, default="#1a1a2e", help="Ink color (hex or CSS color)")
    parser.add_argument("--width", type=float, default=1.6, help="Pen stroke width")
    parser.add_argument("--line-width-chars", type=int, default=42, help="Wrap lines exceeding this character count")

    # Generation Parameters
    parser.add_argument("--seed", type=int, default=None, help="Random seed for generation")
    parser.add_argument("--temp", type=float, default=0.35, help="Sampling temperature for Graves model (lower = neater, higher = more variation)")
    parser.add_argument("--style", type=int, default=0, help="Writer style ID index (0 for default)")
    parser.add_argument("--bias", type=float, default=1.0, help="Graves model sampling neatness bias (b >= 0)")
    parser.add_argument("--char-step-scale", type=float, default=0.10, help="Graves model attention advancement scale per step")
    parser.add_argument("--temperature", type=float, default=0.8, help="Transformer sampling temperature")
    parser.add_argument("--top-k", type=int, default=50, help="Top-k filtering threshold")
    parser.add_argument("--top-p", type=float, default=0.9, help="Top-p nucleus threshold")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])

    return parser.parse_args(args)


def wrap_text_to_lines(raw_text: str, max_chars: int = 42) -> list[str]:
    """Split text by explicit newlines and wrap overly long lines at word boundaries."""
    raw_lines = raw_text.replace("\\n", "\n").splitlines()
    wrapped_lines: list[str] = []

    for line in raw_lines:
        line = line.strip()
        if not line:
            wrapped_lines.append("")
            continue

        if len(line) <= max_chars:
            wrapped_lines.append(line)
        else:
            wrapped_lines.extend(textwrap.wrap(line, width=max_chars))

    return wrapped_lines


def load_model(
    checkpoint_path: Path | None,
    config_path: Path,
    device: torch.device,
    bias: float = 0.0,
) -> tuple[nn.Module, str]:
    """Load model architecture and weights from checkpoint or initialize fallback."""
    pretrained_dir = REPO_ROOT / "checkpoints" / "pretrained"

    # 1. Check for Pretrained Canonical Graves Benchmark
    if checkpoint_path is None and (pretrained_dir / "model.pt").is_file():
        logger.info("Loading canonical Graves pretrained benchmark from %s", pretrained_dir)
        model = PretrainedGravesSynthesis(pretrained_dir, device=device, bias=bias)
        return model, "pretrained"

    if checkpoint_path is not None and ("pretrained" in str(checkpoint_path).lower() or checkpoint_path == pretrained_dir):
        target_dir = checkpoint_path if checkpoint_path.is_dir() else checkpoint_path.parent
        logger.info("Loading canonical Graves pretrained benchmark from %s", target_dir)
        model = PretrainedGravesSynthesis(target_dir, device=device, bias=bias)
        return model, "pretrained"

    # Resolve default checkpoint path if not provided
    if checkpoint_path is None:
        graves_ckpt = REPO_ROOT / "checkpoints" / "best_graves_model.pt"
        transformer_ckpt = REPO_ROOT / "checkpoints" / "best_model.pt"
        if graves_ckpt.is_file():
            checkpoint_path = graves_ckpt
        elif transformer_ckpt.is_file():
            checkpoint_path = transformer_ckpt
        else:
            checkpoint_path = graves_ckpt

    if checkpoint_path.is_file():
        logger.info("Loading model weights from %s", checkpoint_path)
        data = torch.load(checkpoint_path, map_location=device, weights_only=False)
        state_dict = data.get("model_state", data)
        config_data = data.get("config", {})

        # Detect Graves RNN model (has lstm_1 weights or GravesConfig keys)
        is_graves = (
            "lstm_1.weight_ih" in state_dict
            or (isinstance(config_data, dict) and "num_window_components" in config_data)
            or (isinstance(config_data, dict) and "hidden_size" in config_data)
            or "graves" in checkpoint_path.name.lower()
        )

        if is_graves:
            if isinstance(config_data, dict) and "hidden_size" in config_data:
                g_config = GravesConfig(**config_data)
            else:
                g_config = GravesConfig()
            model = GravesHandwritingModel(g_config).to(device)
            model.load_state_dict(state_dict)
            model.eval()
            return model, "graves"

        # Otherwise Transformer model
        if isinstance(config_data, dict) and "d_model" in config_data:
            t_config = ModelConfig(**config_data)
        elif isinstance(config_data, ModelConfig):
            t_config = config_data
        else:
            t_config = ModelConfig.from_config_json(config_path)

        model = HandwritingSynthesisModel(t_config).to(device)
        model.load_state_dict(state_dict)
        model.eval()
        return model, "transformer"

    logger.warning("Checkpoint '%s' not found. Using initialized Graves model.", checkpoint_path)
    model = GravesHandwritingModel().to(device)
    model.eval()
    return model, "graves"


def main(cli_args: Sequence[str] | None = None) -> int:
    args = parse_args(cli_args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 1. Resolve Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # 2. Extract Text
    if args.file:
        if not args.file.is_file():
            logger.error("Input file not found: %s", args.file)
            return 1
        raw_text = args.file.read_text(encoding="utf-8")
    else:
        raw_text = args.text or ""

    lines_text = wrap_text_to_lines(raw_text, max_chars=args.line_width_chars)
    if not lines_text:
        logger.error("No valid text provided to synthesize.")
        return 1

    # 3. Load Alphabet and Tokenizer
    if not (args.alphabet.exists() and args.tokenizer.exists() and args.config.exists()):
        logger.error("Data artifacts missing in %s. Run scripts/prepare_data.py first.", args.config.parent)
        return 1

    alphabet = Alphabet.load(args.alphabet) if args.alphabet.exists() else None
    tokenizer = StrokeTokenizer.load(args.tokenizer) if args.tokenizer.exists() else None
    model, model_type = load_model(args.checkpoint, args.config, device, bias=args.bias)
    print(f"Active model type: {model_type.upper()}")

    # 4. Synthesize Each Line
    synthesized_lines: list[np.ndarray] = []
    total_points = 0
    print(f"Synthesizing {len(lines_text)} line(s)...")

    for i, line_str in enumerate(lines_text, start=1):
        if not line_str.strip():
            # Blank line represented as empty array
            synthesized_lines.append(np.zeros((0, 4), dtype=np.float32))
            continue

        try:
            if model_type == "pretrained":
                points = model.generate(
                    text=line_str,
                    bias=args.bias,
                    max_steps=1200,
                    seed=args.seed,
                    temp=args.temp,
                )
            elif model_type == "graves":
                points = model.generate(
                    text=line_str,
                    alphabet=alphabet,
                    max_steps=1200,
                    bias=args.bias,
                    device=device,
                    char_step_scale=args.char_step_scale,
                )
            else:
                points = model.generate(
                    text=line_str,
                    alphabet=alphabet,
                    tokenizer=tokenizer,
                    writer_id=args.style,
                    max_steps=600,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=device,
                )
            synthesized_lines.append(points)
            total_points += len(points)
            print(f"  Line {i:2d}/{len(lines_text)}: '{line_str}' -> {len(points)} points")
        except Exception as exc:
            logger.warning("Line %d generation error: %s", i, exc)
            synthesized_lines.append(np.zeros((0, 4), dtype=np.float32))

    # 5. Render Page Layout
    ink_style = InkStyle(color=args.color, width=args.width, bezier=True)
    page_h = max(877.0, 100.0 + len(synthesized_lines) * 52.0) if args.ruled else max(240.0, 90.0 + len(synthesized_lines) * 52.0)
    layout = PageLayout(
        ruled=args.ruled,
        margin_rule=args.margin_rule,
        line_height=52.0,
        height=page_h,
    )

    svg_content = render_page(synthesized_lines, layout=layout, style=ink_style)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(svg_content, encoding="utf-8")

    file_size_kb = args.out.stat().st_size / 1024.0
    print("-" * 65)
    print(f"Saved: {args.out} ({file_size_kb:.1f} KB, {total_points} total points)")
    print(f"Open in browser to view: file:///{args.out.resolve().as_posix()}")
    print("-" * 65)
    return 0


if __name__ == "__main__":
    sys.exit(main())
