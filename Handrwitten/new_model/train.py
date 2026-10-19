"""Standalone training script for the Continuous Guided-Attention Handwriting Synthesis Model.

Saves checkpoints exclusively into checkpoints/new_model/ without touching the Graves baseline.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Ensure Handrwitten is on path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from hws.dataset import get_dataloader
from new_model.config import NewModelConfig
from new_model.loss import ContinuousGuidedSynthesisLoss
from new_model.model import ContinuousGuidedTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Continuous Guided-Attention Handwriting Transformer")
    parser.add_argument("--data-dir", type=str, default=str(root_dir / "datas"), help="Path to processed HDF5 datasets")
    parser.add_argument("--checkpoint-dir", type=str, default=str(root_dir / "checkpoints" / "new_model"), help="Output directory for new model checkpoints")
    parser.add_argument("--epochs", type=int, default=20, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--guided-weight", type=float, default=1.0, help="Weight for monotonic alignment loss")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader subprocess workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Target compute device")
    return parser.parse_args()


def train_one_epoch(
    model: ContinuousGuidedTransformer,
    loader: torch.utils.data.DataLoader,
    criterion: ContinuousGuidedSynthesisLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool,
    scaler: torch.amp.GradScaler | None,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    total_xy = 0.0
    total_pen = 0.0
    total_guided = 0.0
    batches = 0

    for batch in loader:
        chars = batch["chars"].to(device, non_blocking=True)
        char_lens = batch["char_lens"].to(device, non_blocking=True)
        char_mask = batch["char_mask"].to(device, non_blocking=True)

        raw = batch["raw"].to(device, non_blocking=True)  # (B, max_T, 4)
        if raw.size(1) <= 1:
            continue

        in_strokes = raw[:, :-1, :3]
        target_strokes = raw[:, 1:, :3]
        in_mask = batch["in_mask"].to(device, non_blocking=True)
        stroke_lens = (batch["stroke_lens"] - 1).clamp(min=1).to(device, non_blocking=True)
        writer = batch["writer"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with torch.amp.autocast("cuda"):
                preds = model(
                    chars=chars,
                    strokes=in_strokes,
                    char_mask=char_mask,
                    stroke_mask=in_mask,
                    writer=writer,
                )
                loss_dict = criterion(
                    predictions=preds,
                    targets=target_strokes,
                    mask=in_mask,
                    attn_weights=preds["attn_weights"],
                    stroke_lens=stroke_lens,
                    char_lens=char_lens,
                )
            assert scaler is not None
            scaler.scale(loss_dict["total_loss"]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(
                chars=chars,
                strokes=in_strokes,
                char_mask=char_mask,
                stroke_mask=in_mask,
                writer=writer,
            )
            loss_dict = criterion(
                predictions=preds,
                targets=target_strokes,
                mask=in_mask,
                attn_weights=preds["attn_weights"],
                stroke_lens=stroke_lens,
                char_lens=char_lens,
            )
            loss_dict["total_loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += float(loss_dict["total_loss"].item())
        total_xy += float(loss_dict["loss_xy"].item())
        total_pen += float(loss_dict["loss_pen"].item())
        total_guided += float(loss_dict["loss_guided"].item())
        batches += 1

    return {
        "loss": total_loss / max(batches, 1),
        "xy_loss": total_xy / max(batches, 1),
        "pen_loss": total_pen / max(batches, 1),
        "guided_loss": total_guided / max(batches, 1),
    }


@torch.no_grad()
def evaluate(
    model: ContinuousGuidedTransformer,
    loader: torch.utils.data.DataLoader,
    criterion: ContinuousGuidedSynthesisLoss,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_xy = 0.0
    total_pen = 0.0
    total_guided = 0.0
    batches = 0

    for batch in loader:
        chars = batch["chars"].to(device)
        char_lens = batch["char_lens"].to(device)
        char_mask = batch["char_mask"].to(device)

        raw = batch["raw"].to(device)
        if raw.size(1) <= 1:
            continue

        in_strokes = raw[:, :-1, :3]
        target_strokes = raw[:, 1:, :3]
        in_mask = batch["in_mask"].to(device)
        stroke_lens = (batch["stroke_lens"] - 1).clamp(min=1).to(device)
        writer = batch["writer"].to(device)

        preds = model(
            chars=chars,
            strokes=in_strokes,
            char_mask=char_mask,
            stroke_mask=in_mask,
            writer=writer,
        )
        loss_dict = criterion(
            predictions=preds,
            targets=target_strokes,
            mask=in_mask,
            attn_weights=preds["attn_weights"],
            stroke_lens=stroke_lens,
            char_lens=char_lens,
        )

        total_loss += float(loss_dict["total_loss"].item())
        total_xy += float(loss_dict["loss_xy"].item())
        total_pen += float(loss_dict["loss_pen"].item())
        total_guided += float(loss_dict["loss_guided"].item())
        batches += 1

    return {
        "loss": total_loss / max(batches, 1),
        "xy_loss": total_xy / max(batches, 1),
        "pen_loss": total_pen / max(batches, 1),
        "guided_loss": total_guided / max(batches, 1),
    }


def main():
    args = parse_args()
    data_dir = Path(args.data_dir)
    ckpt_dir = Path(args.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    if not (data_dir / "config.json").is_file() and (data_dir / "processed" / "config.json").is_file():
        data_dir = data_dir / "processed"

    config_path = data_dir / "config.json"
    if not config_path.is_file():
        logger.error(f"Config file not found at {config_path}")
        sys.exit(1)

    config = NewModelConfig.from_config_json(
        config_path,
        guided_attention_weight=args.guided_weight,
        learning_rate=args.lr,
    )

    device = torch.device(args.device)
    logger.info(f"Using device: {device} (CUDA: {torch.cuda.is_available()})")

    # Load datasets with load_raw=True
    train_loader = get_dataloader(
        data_dir / "train.h5",
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        load_raw=True,
    )
    val_loader = get_dataloader(
        data_dir / "val.h5",
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        load_raw=True,
    )

    model = ContinuousGuidedTransformer(config).to(device)
    criterion = ContinuousGuidedSynthesisLoss(guided_weight=args.guided_weight).to(device)

    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=config.weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    use_amp = (device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    logger.info("=" * 60)
    logger.info(f"Training Continuous Guided-Attention Transformer ({args.epochs} epochs)")
    logger.info(f"Checkpoints will be saved to: {ckpt_dir.resolve()}")
    logger.info("=" * 60)

    best_val_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, use_amp, scaler)
        val_metrics = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        logger.info(
            f"Epoch {epoch:02d}/{args.epochs:02d} [{elapsed:.1f}s] "
            f"Train Loss: {train_metrics['loss']:.4f} (XY: {train_metrics['xy_loss']:.4f}, Pen: {train_metrics['pen_loss']:.4f}, Guided: {train_metrics['guided_loss']:.4f}) | "
            f"Val Loss: {val_metrics['loss']:.4f} (XY: {val_metrics['xy_loss']:.4f}, Pen: {val_metrics['pen_loss']:.4f}, Guided: {val_metrics['guided_loss']:.4f})"
        )

        # Save last checkpoint
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "config": config.__dict__,
            "val_loss": val_metrics["loss"],
        }, ckpt_dir / "last_model.pt")

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "config": config.__dict__,
                "val_loss": best_val_loss,
            }, ckpt_dir / "best_model.pt")
            logger.info(f"  --> Saved new best model (Val Loss: {best_val_loss:.4f})")

    logger.info("Training complete!")


if __name__ == "__main__":
    main()
