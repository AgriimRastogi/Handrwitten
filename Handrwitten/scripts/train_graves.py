"""Training loop and evaluation pipeline for Alex Graves 2013 RNN model.

Usage:
    python scripts/train_graves.py --epochs 25 --batch-size 32
    # Quick smoke test:
    python scripts/train_graves.py --smoke-test
"""

from __future__ import annotations

import argparse
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from hws.alphabet import Alphabet
from hws.dataset import get_dataloader
from hws.graves_model import GravesConfig, GravesHandwritingModel
from hws.render import InkStyle, render_line

logger = logging.getLogger("train_graves")

DEFAULT_PROMPTS = [
    "GLA University",
    "Agriim Rastogi",
    "Handwriting synthesis",
    "Project 2026",
]


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Graves (2013) 3-layer LSTM with Monotonic Gaussian Attention & MDN.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-data", type=Path, default=REPO_ROOT / "datas" / "processed" / "train.h5")
    parser.add_argument("--val-data", type=Path, default=REPO_ROOT / "datas" / "processed" / "val.h5")
    parser.add_argument("--alphabet", type=Path, default=REPO_ROOT / "datas" / "processed" / "alphabet.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--samples-dir", type=Path, default=REPO_ROOT / "outputs" / "graves_samples")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--lr", type=float, default=4e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-steps", type=int, default=150)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=1)
    parser.add_argument("--bias", type=float, default=1.0, help="Sampling neatness bias (b >= 0)")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 train batches + 1 val batch and exit")
    parser.add_argument("--resume", type=Path, default=None)

    return parser.parse_args(args)


def get_lr_scheduler(optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int) -> LambdaLR:
    """Linear warmup followed by cosine annealing to 0.1 * lr."""
    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(0.1, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: GravesHandwritingModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    device: torch.device,
    grad_clip: float = 5.0,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Execute one training epoch across minibatches."""
    model.train()
    total_loss = 0.0
    total_xy = 0.0
    total_pen = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        # Raw continuous strokes: (B, T, 4) -> [dx, dy, pen_up, dt]
        raw = batch["raw"].to(device)
        chars = batch["chars"].to(device)
        char_mask = batch["char_mask"].to(device)
        stroke_mask = batch["stroke_mask"].to(device)

        # Teacher-forcing inputs: steps 0 .. T-2, targets: steps 1 .. T-1
        in_strokes = raw[:, :-1, :3]
        target_strokes = raw[:, 1:, :3]
        in_mask = stroke_mask[:, 1:]

        optimizer.zero_grad(set_to_none=True)

        predictions = model(in_strokes, chars, char_mask)
        losses = model.compute_loss(predictions, target_strokes, in_mask)
        loss = losses["loss"]

        loss.backward()

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        optimizer.step()
        if scheduler is not None:
            scheduler.step()

        total_loss += float(loss.item())
        total_xy += float(losses["loss_xy"].item())
        total_pen += float(losses["loss_pen"].item())
        n_batches += 1

        if limit_batches is None and ((batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader)):
            avg_loss = total_loss / n_batches
            print(
                f"  [Batch {batch_idx + 1:3d}/{len(dataloader):3d}] "
                f"loss: {avg_loss:6.3f} (xy: {losses['loss_xy'].item():.2f}, pen: {losses['loss_pen'].item():.2f})",
                flush=True,
            )

    denom = max(1, n_batches)
    return {
        "loss": total_loss / denom,
        "loss_xy": total_xy / denom,
        "loss_pen": total_pen / denom,
    }


@torch.no_grad()
def evaluate(
    model: GravesHandwritingModel,
    dataloader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Compute validation loss on unseen validation lines."""
    model.eval()
    total_loss = 0.0
    total_xy = 0.0
    total_pen = 0.0
    n_batches = 0

    for batch_idx, batch in enumerate(dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        raw = batch["raw"].to(device)
        chars = batch["chars"].to(device)
        char_mask = batch["char_mask"].to(device)
        stroke_mask = batch["stroke_mask"].to(device)

        in_strokes = raw[:, :-1, :3]
        target_strokes = raw[:, 1:, :3]
        in_mask = stroke_mask[:, 1:]

        predictions = model(in_strokes, chars, char_mask)
        losses = model.compute_loss(predictions, target_strokes, in_mask)

        total_loss += float(losses["loss"].item())
        total_xy += float(losses["loss_xy"].item())
        total_pen += float(losses["loss_pen"].item())
        n_batches += 1

    denom = max(1, n_batches)
    return {
        "loss": total_loss / denom,
        "loss_xy": total_xy / denom,
        "loss_pen": total_pen / denom,
    }


def sample_and_save(
    model: GravesHandwritingModel,
    alphabet: Alphabet,
    prompts: list[str],
    out_dir: Path,
    epoch: int,
    device: torch.device,
    bias: float = 1.0,
) -> list[Path]:
    """Generate handwriting strokes and render them to SVG files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for idx, prompt in enumerate(prompts):
        try:
            points = model.generate(
                text=prompt,
                alphabet=alphabet,
                max_steps=700,
                bias=bias,
                device=device,
            )
            if len(points) < 2:
                continue

            svg_str = render_line(points, style=InkStyle(width=1.8))
            file_name = f"graves_epoch_{epoch:03d}_sample_{idx:02d}.svg"
            save_path = out_dir / file_name
            save_path.write_text(svg_str, encoding="utf-8")
            saved_paths.append(save_path)
        except Exception as exc:
            logger.warning("Failed to synthesize prompt '%s': %s", prompt, exc)

    return saved_paths


def main(cli_args: Sequence[str] | None = None) -> int:
    args = parse_args(cli_args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 1. Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 2. Vocabulary & Data
    alphabet = Alphabet.load(args.alphabet)
    config = GravesConfig(char_vocab_size=alphabet.vocab_size)

    logger.info("Initializing datasets with raw continuous coordinates...")
    train_loader = get_dataloader(
        args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        load_raw=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = get_dataloader(
        args.val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        load_raw=True,
        pin_memory=False,
    )

    # 3. Model
    model = GravesHandwritingModel(config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Graves RNN Model instantiated: {total_params / 1e6:.2f}M parameters (3-layer LSTM + Monotonic Window + Bivariate MDN)")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999),
    )

    epochs = 1 if args.smoke_test else args.epochs
    total_steps = len(train_loader) * epochs
    warmup_steps = min(args.warmup_steps, max(1, total_steps // 10))
    scheduler = get_lr_scheduler(optimizer, warmup_steps=warmup_steps, total_steps=total_steps)

    start_epoch = 1
    best_val_loss = float("inf")
    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.samples_dir.mkdir(parents=True, exist_ok=True)

    if args.resume and args.resume.exists():
        logger.info("Resuming from checkpoint %s", args.resume)
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", float("inf"))

    limit_batches = 2 if args.smoke_test else None
    print("-" * 65)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'MDN (xy)':^9} | {'Pen Lift':^9} | {'Time':^7}")
    print("-" * 65)

    for epoch in range(start_epoch, epochs + 1):
        t0 = time.time()
        train_stats = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            grad_clip=args.grad_clip,
            limit_batches=limit_batches,
        )

        val_stats = evaluate(
            model=model,
            dataloader=val_loader,
            device=device,
            limit_batches=limit_batches,
        )
        elapsed = time.time() - t0

        print(
            f"{epoch:5d}/{epochs} | "
            f"{train_stats['loss']:10.4f} | "
            f"{val_stats['loss']:10.4f} | "
            f"{val_stats['loss_xy']:9.4f} | "
            f"{val_stats['loss_pen']:9.4f} | "
            f"{elapsed:6.1f}s"
        )

        # Periodic SVG sampling
        if epoch % args.sample_every == 0 or args.smoke_test:
            sample_and_save(
                model=model,
                alphabet=alphabet,
                prompts=DEFAULT_PROMPTS if not args.smoke_test else DEFAULT_PROMPTS[:1],
                out_dir=args.samples_dir,
                epoch=epoch,
                device=device,
                bias=args.bias,
            )

        # Checkpointing
        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_stats["loss"],
            "best_val_loss": best_val_loss,
            "config": asdict(config),
        }

        torch.save(checkpoint_payload, args.checkpoint_dir / "last_graves_model.pt")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            checkpoint_payload["best_val_loss"] = best_val_loss
            torch.save(checkpoint_payload, args.checkpoint_dir / "best_graves_model.pt")

    print("-" * 65)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
