"""
train.py — Two-Phase Fine-Tuning Training Loop
================================================
Author: Samuel Appiah Kubi
Copernicus Master's in Digital Earth, Paris Lodron University Salzburg

Phase 1 (epochs 1 → phase1_epochs):
  Backbone frozen. Only classification head trained with high LR.

Phase 2 (phase1_epochs → total_epochs):
  Unfreeze top backbone layers. Train end-to-end with very low LR.

Features:
  - Early stopping with configurable patience
  - Checkpoint saving (best + last)
  - TensorBoard-compatible loss logging (optional)
  - Reproducible seeds
  - Class-weighted loss
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import load_config, set_seeds, timer, save_checkpoint, load_checkpoint
from scripts.dataset import build_dataloaders
from scripts.model import build_model, get_optimizer, get_scheduler, get_loss

logger = logging.getLogger("crop_mapping.train")

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Training step
# =============================================================================

def train_one_epoch(
    model: "nn.Module",
    loader,
    optimizer,
    criterion,
    device: str,
    scaler=None,
) -> float:
    """Run one training epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        if scaler is not None:  # AMP (mixed precision)
            from torch.cuda.amp import autocast
            with autocast():
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


# =============================================================================
# Validation step
# =============================================================================

@torch.no_grad()
def evaluate(
    model: "nn.Module",
    loader,
    criterion,
    device: str,
) -> Tuple[float, float]:
    """Run evaluation. Returns (avg_loss, accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item()

        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)

    return total_loss / max(len(loader), 1), correct / max(total, 1)


# =============================================================================
# Early stopping
# =============================================================================

class EarlyStopping:
    """Stop training if validation loss does not improve for `patience` epochs."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0
        self.should_stop = False

    def __call__(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True
        return self.should_stop


# =============================================================================
# Main training loop
# =============================================================================

@timer
def train(
    cfg: Dict,
    resume: bool = False,
) -> Dict:
    """
    Full two-phase training loop.

    Returns
    -------
    history : dict with train_loss, val_loss, val_acc per epoch
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Run: pip install torch")

    set_seeds(cfg["training"].get("seed", 42))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Device: {device}")
    if device == "cuda":
        logger.info(f"  GPU: {torch.cuda.get_device_name(0)}  "
                    f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    models_dir = cfg["paths"]["models"]
    os.makedirs(models_dir, exist_ok=True)
    best_ckpt_path = os.path.join(models_dir, "best_model.pth")
    last_ckpt_path = os.path.join(models_dir, "last_model.pth")
    history_path = os.path.join(models_dir, "training_history.json")

    # ── Build dataloaders ─────────────────────────────────────────────────────
    logger.info("Building dataloaders…")
    train_loader, val_loader, _, normaliser = build_dataloaders(cfg)

    # ── Build model ───────────────────────────────────────────────────────────
    logger.info("Building model…")
    model = build_model(cfg).to(device)

    # Class weights from training dataset
    train_ds = train_loader.dataset
    class_weights = train_ds.class_weights.to(device)

    criterion = get_loss(cfg, class_weights)
    optimizer = get_optimizer(model, cfg)
    scheduler = get_scheduler(optimizer, cfg)

    # Mixed precision
    use_amp = device == "cuda"
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    # ── Resume from checkpoint ────────────────────────────────────────────────
    start_epoch = 0
    history = {"train_loss": [], "val_loss": [], "val_acc": [], "phase": []}
    best_val_loss = float("inf")
    early_stop = EarlyStopping(patience=cfg["training"].get("patience", 10))

    if resume and os.path.exists(last_ckpt_path):
        ckpt = torch.load(last_ckpt_path, map_location=device)
        model.load_state_dict(ckpt["model_state"])
        optimizer.load_state_dict(ckpt["optimizer_state"])
        start_epoch = ckpt.get("epoch", 0) + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))
        if os.path.exists(history_path):
            with open(history_path) as f:
                history = json.load(f)
        logger.info(f"Resumed from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    # ── Phase split ───────────────────────────────────────────────────────────
    total_epochs = cfg["training"]["epochs"]
    # Phase 2 starts after 60% of total epochs
    phase2_start = int(total_epochs * 0.6)
    phase2_started = start_epoch >= phase2_start

    logger.info(
        f"Training: {total_epochs} epochs  "
        f"Phase 1: 0–{phase2_start}  Phase 2: {phase2_start}–{total_epochs}"
    )

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, total_epochs):
        # Transition to Phase 2
        if epoch == phase2_start and not phase2_started:
            logger.info(f"\n{'='*50}")
            logger.info(f"  PHASE 2 START (epoch {epoch}): unfreezing top backbone layers")
            logger.info(f"{'='*50}")
            if hasattr(model, "unfreeze_for_phase2"):
                model.unfreeze_for_phase2(n_transformer_blocks=2)
            elif hasattr(model, "_fallback") and hasattr(model._fallback, "unfreeze_top_layers"):
                model._fallback.unfreeze_top_layers(2)
            # Rebuild optimizer with phase 2 LR
            optimizer = get_optimizer(model, cfg)
            scheduler = get_scheduler(optimizer, cfg)
            phase2_started = True

        phase = "2" if epoch >= phase2_start else "1"

        # Train
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device, scaler)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        # Scheduler step
        if scheduler:
            if hasattr(scheduler, "step"):
                try:
                    scheduler.step(val_loss)   # ReduceLROnPlateau
                except TypeError:
                    scheduler.step()           # CosineAnnealing

        # History
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["val_acc"].append(float(val_acc))
        history["phase"].append(phase)

        # Log
        current_lr = optimizer.param_groups[0]["lr"]
        logger.info(
            f"Epoch {epoch+1:03d}/{total_epochs} [Phase {phase}] "
            f"train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3f}  lr={current_lr:.2e}"
        )

        # Save best checkpoint
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_acc": val_acc,
                    "best_val_loss": best_val_loss,
                    "config": cfg,
                },
                best_ckpt_path,
            )
            logger.info(f"  ✓ Best model saved (val_loss={best_val_loss:.4f})")

        # Save last checkpoint (every epoch)
        torch.save(
            {
                "epoch": epoch,
                "model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "best_val_loss": best_val_loss,
            },
            last_ckpt_path,
        )

        # Save history
        with open(history_path, "w") as f:
            json.dump(history, f, indent=2)

        # Early stopping
        if early_stop(val_loss):
            logger.info(f"Early stopping triggered at epoch {epoch+1}")
            break

    logger.info(f"\n✓ Training complete. Best val_loss={best_val_loss:.4f}")
    return history


# =============================================================================
# Training curves visualisation
# =============================================================================

def plot_training_history(history: Dict, save_path: Optional[str] = None):
    """Plot training and validation loss/accuracy curves."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping plot")
        return

    epochs = list(range(1, len(history["train_loss"]) + 1))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], label="Train Loss", color="royalblue")
    ax1.plot(epochs, history["val_loss"], label="Val Loss", color="tomato")
    # Mark phase transition
    phase2_start = next((i for i, p in enumerate(history["phase"]) if p == "2"), None)
    if phase2_start:
        ax1.axvline(x=phase2_start + 1, color="grey", linestyle="--", label="Phase 2 start")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("Training vs Validation Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["val_acc"], label="Val Accuracy", color="seagreen")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy")
    ax2.set_title("Validation Accuracy")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Training curves saved: {save_path}")
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

def run(config_path: str = "configs/config.yaml", resume: bool = False):
    cfg = load_config(config_path)
    history = train(cfg, resume=resume)
    save_path = os.path.join(cfg["paths"]["reports"], "training_curves.png")
    plot_training_history(history, save_path)
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train crop classification model")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--resume", action="store_true", help="Resume from last checkpoint")
    args = parser.parse_args()
    run(args.config, args.resume)
