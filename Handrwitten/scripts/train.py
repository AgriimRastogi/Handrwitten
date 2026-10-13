"""Training loop and evaluation pipeline for stroke-level handwriting synthesis.

    python scripts/train.py
    # or quick verification:
    python scripts/train.py --smoke-test

This script trains `HandwritingSynthesisModel` on the packed HDF5 dataset,
evaluates validation cross-entropy loss across all 4 discrete token channels,
saves best/last checkpoints, and generates visual SVG samples per epoch so you
can watch the model learn to write.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hws.alphabet import Alphabet
from hws.dataset import StrokeDataset, collate_strokes, get_dataloader
from hws.model import HandwritingSynthesisModel, ModelConfig
from hws.render import InkStyle, render_line
from hws.tokenizer import StrokeTokenizer

REPO_ROOT = Path(__file__).resolve().parent.parent
logger = logging.getLogger("train")

DEFAULT_PROMPTS = [
    "GLA University",
    "Handwriting synthesis",
    "The quick brown fox jumps",
]


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train stroke-level discrete handwriting synthesis model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--train-data", type=Path, default=REPO_ROOT / "datas" / "processed" / "train.h5")
    parser.add_argument("--val-data", type=Path, default=REPO_ROOT / "datas" / "processed" / "val.h5")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "datas" / "processed" / "config.json")
    parser.add_argument("--alphabet", type=Path, default=REPO_ROOT / "datas" / "processed" / "alphabet.json")
    parser.add_argument("--tokenizer", type=Path, default=REPO_ROOT / "datas" / "processed" / "tokenizer.json")
    parser.add_argument("--checkpoint-dir", type=Path, default=REPO_ROOT / "checkpoints")
    parser.add_argument("--samples-dir", type=Path, default=REPO_ROOT / "outputs" / "samples")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-2)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-every", type=int, default=1, help="Generate SVG samples every N epochs")
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--smoke-test", action="store_true", help="Run 2 train batches + 1 val batch and exit")
    parser.add_argument("--resume", type=Path, default=None, help="Path to checkpoint to resume from")

    return parser.parse_args(args)


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Linear warmup followed by cosine annealing to 0.05 * base_lr."""

    def lr_lambda(current_step: int) -> float:
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(
            max(1, total_steps - warmup_steps)
        )
        return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))

    return LambdaLR(optimizer, lr_lambda)


def train_one_epoch(
    model: HandwritingSynthesisModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: LambdaLR | None,
    device: torch.device,
    grad_clip: float = 1.0,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Execute one training epoch across all minibatches."""
    model.train()
    total_loss = 0.0
    total_dx = 0.0
    total_dy = 0.0
    total_dt = 0.0
    total_pen = 0.0
    n_batches = 0

    use_cuda = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_cuda)

    for batch_idx, batch in enumerate(dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        chars = batch["chars"].to(device)
        char_mask = batch["char_mask"].to(device)
        in_dx = batch["in_dx"].to(device)
        in_dy = batch["in_dy"].to(device)
        in_dt = batch["in_dt"].to(device)
        in_pen = batch["in_pen"].to(device)
        in_mask = batch["in_mask"].to(device)
        writer = batch["writer"].to(device)

        targets = {
            "target_dx": batch["target_dx"].to(device),
            "target_dy": batch["target_dy"].to(device),
            "target_dt": batch["target_dt"].to(device),
            "target_pen": batch["target_pen"].to(device),
        }

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            logits = model(
                chars=chars,
                in_dx=in_dx,
                in_dy=in_dy,
                in_dt=in_dt,
                in_pen=in_pen,
                char_mask=char_mask,
                in_mask=in_mask,
                writer=writer,
            )
            losses = model.compute_loss(logits, targets)
            loss = losses["loss"]

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)

        scale_before = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        scale_after = scaler.get_scale()

        # Only step the scheduler if the optimizer actually stepped (no inf/nan scale rollback)
        if scheduler is not None and (scale_before <= scale_after or not use_cuda):
            scheduler.step()

        total_loss += float(loss.item())
        total_dx += float(losses["loss_dx"].item())
        total_dy += float(losses["loss_dy"].item())
        total_dt += float(losses["loss_dt"].item())
        total_pen += float(losses["loss_pen"].item())
        n_batches += 1

        if limit_batches is None and ((batch_idx + 1) % 50 == 0 or (batch_idx + 1) == len(dataloader)):
            avg_so_far = total_loss / n_batches
            print(
                f"  [Batch {batch_idx + 1:3d}/{len(dataloader):3d}] loss: {avg_so_far:6.3f} "
                f"(dx: {losses['loss_dx'].item():.2f}, dy: {losses['loss_dy'].item():.2f}, "
                f"dt: {losses['loss_dt'].item():.2f}, pen: {losses['loss_pen'].item():.2f})",
                flush=True,
            )

    denom = max(1, n_batches)
    return {
        "loss": total_loss / denom,
        "loss_dx": total_dx / denom,
        "loss_dy": total_dy / denom,
        "loss_dt": total_dt / denom,
        "loss_pen": total_pen / denom,
    }


@torch.no_grad()
def evaluate(
    model: HandwritingSynthesisModel,
    dataloader: DataLoader,
    device: torch.device,
    limit_batches: int | None = None,
) -> dict[str, float]:
    """Compute validation loss across all discrete channels."""
    model.eval()
    total_loss = 0.0
    total_dx = 0.0
    total_dy = 0.0
    total_dt = 0.0
    total_pen = 0.0
    n_batches = 0

    use_cuda = device.type == "cuda"

    for batch_idx, batch in enumerate(dataloader):
        if limit_batches is not None and batch_idx >= limit_batches:
            break

        chars = batch["chars"].to(device)
        char_mask = batch["char_mask"].to(device)
        in_dx = batch["in_dx"].to(device)
        in_dy = batch["in_dy"].to(device)
        in_dt = batch["in_dt"].to(device)
        in_pen = batch["in_pen"].to(device)
        in_mask = batch["in_mask"].to(device)
        writer = batch["writer"].to(device)

        targets = {
            "target_dx": batch["target_dx"].to(device),
            "target_dy": batch["target_dy"].to(device),
            "target_dt": batch["target_dt"].to(device),
            "target_pen": batch["target_pen"].to(device),
        }

        with torch.amp.autocast(device_type=device.type, enabled=use_cuda):
            logits = model(
                chars=chars,
                in_dx=in_dx,
                in_dy=in_dy,
                in_dt=in_dt,
                in_pen=in_pen,
                char_mask=char_mask,
                in_mask=in_mask,
                writer=writer,
            )
            losses = model.compute_loss(logits, targets)

        total_loss += float(losses["loss"].item())
        total_dx += float(losses["loss_dx"].item())
        total_dy += float(losses["loss_dy"].item())
        total_dt += float(losses["loss_dt"].item())
        total_pen += float(losses["loss_pen"].item())
        n_batches += 1

    denom = max(1, n_batches)
    return {
        "loss": total_loss / denom,
        "loss_dx": total_dx / denom,
        "loss_dy": total_dy / denom,
        "loss_dt": total_dt / denom,
        "loss_pen": total_pen / denom,
    }


def sample_and_save(
    model: HandwritingSynthesisModel,
    alphabet: Alphabet,
    tokenizer: StrokeTokenizer,
    prompts: list[str],
    out_dir: Path,
    epoch: int,
    device: torch.device,
    temperature: float = 0.8,
) -> list[Path]:
    """Generate handwriting strokes and render them to SVG files."""
    out_dir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []

    for idx, prompt in enumerate(prompts):
        try:
            points = model.generate(
                text=prompt,
                alphabet=alphabet,
                tokenizer=tokenizer,
                writer_id=0,
                max_steps=500,
                temperature=temperature,
                device=device,
            )
            if len(points) < 2:
                continue

            svg_str = render_line(points, style=InkStyle(width=2.0))
            file_name = f"epoch_{epoch:03d}_sample_{idx:02d}.svg"
            save_path = out_dir / file_name
            save_path.write_text(svg_str, encoding="utf-8")
            saved_paths.append(save_path)
        except Exception as exc:
            logger.warning("Failed to synthesize prompt '%s': %s", prompt, exc)

    return saved_paths


def main(cli_args: Sequence[str] | None = None) -> int:
    args = parse_args(cli_args)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    # 1. Device resolution
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Device: {device} ({torch.cuda.get_device_name(0) if device.type == 'cuda' else 'CPU'})")

    # 2. Load configurations and vocabularies
    if not args.config.exists():
        logger.error("Configuration file not found at %s. Run scripts/prepare_data.py first.", args.config)
        return 1

    model_config = ModelConfig.from_config_json(args.config)
    alphabet = Alphabet.load(args.alphabet)
    tokenizer = StrokeTokenizer.load(args.tokenizer)

    # 3. Datasets & Loaders
    logger.info("Initializing datasets...")
    train_loader = get_dataloader(
        args.train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = get_dataloader(
        args.val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=False,
    )

    # 4. Model instantiation
    model = HandwritingSynthesisModel(model_config).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model instantiated: {total_params / 1e6:.2f}M parameters")

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.98),
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

    # 5. Training loop
    limit_batches = 2 if args.smoke_test else None
    print("-" * 75)
    print(f"{'Epoch':^7} | {'Train Loss':^10} | {'Val Loss':^10} | {'dx':^7} | {'dy':^7} | {'dt':^7} | {'pen':^7} | {'Time':^7}")
    print("-" * 75)

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
            f"{val_stats['loss_dx']:7.3f} | "
            f"{val_stats['loss_dy']:7.3f} | "
            f"{val_stats['loss_dt']:7.3f} | "
            f"{val_stats['loss_pen']:7.3f} | "
            f"{elapsed:6.1f}s"
        )

        # Periodic SVG sampling
        if epoch % args.sample_every == 0 or args.smoke_test:
            saved_svgs = sample_and_save(
                model=model,
                alphabet=alphabet,
                tokenizer=tokenizer,
                prompts=DEFAULT_PROMPTS if not args.smoke_test else DEFAULT_PROMPTS[:1],
                out_dir=args.samples_dir,
                epoch=epoch,
                device=device,
                temperature=args.temperature,
            )
            if saved_svgs:
                logger.debug("Rendered %d SVG samples for epoch %d", len(saved_svgs), epoch)

        # Checkpointing
        checkpoint_payload = {
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "val_loss": val_stats["loss"],
            "best_val_loss": best_val_loss,
            "config": asdict(model_config),
        }

        torch.save(checkpoint_payload, args.checkpoint_dir / "last_model.pt")

        if val_stats["loss"] < best_val_loss:
            best_val_loss = val_stats["loss"]
            checkpoint_payload["best_val_loss"] = best_val_loss
            torch.save(checkpoint_payload, args.checkpoint_dir / "best_model.pt")

    print("-" * 75)
    print(f"Training complete! Best validation loss: {best_val_loss:.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
