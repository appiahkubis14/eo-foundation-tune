"""
evaluate.py — Model Evaluation, Baseline Comparison, and Visualisation
=======================================================================
Author: Samuel Appiah Kubi
Copernicus Master's in Digital Earth, Paris Lodron University Salzburg

Evaluates:
  1. Fine-tuned foundation model (best checkpoint)
  2. Random Forest baseline (spectral features)
  3. U-Net trained from scratch baseline

Produces:
  - Per-class metrics table (precision, recall, F1, IoU)
  - Confusion matrix heatmap
  - Comparison table (foundation model vs baselines)
  - Saves all results as JSON + CSV
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import (
    load_config, timer, compute_metrics, print_metrics_table,
    load_geotiff,
)
from scripts.dataset import build_dataloaders, build_patches_from_rasters
from scripts.model import build_model

logger = logging.getLogger("crop_mapping.evaluate")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Foundation model inference
# =============================================================================

@torch.no_grad()
def run_inference(
    model: "torch.nn.Module",
    loader,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a DataLoader.

    Returns
    -------
    y_true, y_pred : np.ndarray  shape (N,)
    """
    model.eval()
    all_true, all_pred = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        logits = model(x)
        preds = logits.argmax(dim=1).cpu().numpy()
        all_true.extend(y.numpy())
        all_pred.extend(preds)
    return np.array(all_true), np.array(all_pred)


def evaluate_foundation_model(
    cfg: Dict,
    split: str = "test",
) -> Dict:
    """Load best checkpoint and evaluate on the specified split."""
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    best_ckpt = os.path.join(cfg["paths"]["models"], "best_model.pth")

    if not os.path.exists(best_ckpt):
        logger.warning(f"No checkpoint at {best_ckpt}. Skipping foundation model evaluation.")
        return {}

    ckpt = torch.load(best_ckpt, map_location=device)
    model = build_model(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])
    logger.info(f"Loaded checkpoint from epoch {ckpt.get('epoch', '?')}")

    _, val_loader, test_loader, _ = build_dataloaders(cfg, fit_normaliser=False)
    loader = test_loader if split == "test" else val_loader

    y_true, y_pred = run_inference(model, loader, device)
    class_names = cfg["classes"]["names"]
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["model"] = "Foundation Model (fine-tuned)"
    metrics["split"] = split

    print_metrics_table(metrics, class_names)
    return metrics


# =============================================================================
# Random Forest baseline
# =============================================================================

@timer
def train_random_forest(
    cfg: Dict,
) -> Tuple[object, np.ndarray, np.ndarray]:
    """
    Train a Random Forest on flattened spectral features.
    Returns trained model + test predictions.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import StandardScaler

    processed_dir = cfg["paths"]["processed"]
    labels_dir = cfg["paths"]["labels"]
    composite_path = os.path.join(processed_dir, "s2_composite.tif")

    # Extract patches for all splits
    all_patches, all_labels = [], []
    for split in ("train", "val", "test"):
        label_path = os.path.join(labels_dir, f"{split}_label_raster.tif")
        if not os.path.exists(label_path):
            label_path = os.path.join(labels_dir, "label_raster.tif")
        p, l = build_patches_from_rasters(composite_path, label_path, cfg)
        all_patches.append(p)
        all_labels.append(l)

    train_p, val_p, test_p = all_patches
    train_l, val_l, test_l = all_labels

    # Flatten patches: (N, C, H, W) → (N, C) using mean per band
    def flatten(patches):
        """Mean and std of each channel across the patch."""
        mean = patches.mean(axis=(-2, -1))   # (N, C)
        std = patches.std(axis=(-2, -1))      # (N, C)
        return np.concatenate([mean, std], axis=1)  # (N, 2C)

    X_train = flatten(train_p[train_l >= 0])
    y_train = train_l[train_l >= 0]
    X_test = flatten(test_p[test_l >= 0])
    y_test = test_l[test_l >= 0]

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    rf_cfg = cfg.get("baselines", {}).get("random_forest", {})
    rf = RandomForestClassifier(
        n_estimators=rf_cfg.get("n_estimators", 200),
        max_depth=rf_cfg.get("max_depth", 15),
        n_jobs=rf_cfg.get("n_jobs", -1),
        random_state=42,
        class_weight="balanced",
    )
    logger.info(f"Training Random Forest ({rf.n_estimators} trees)…")
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    return rf, y_test, y_pred


def evaluate_random_forest(cfg: Dict) -> Dict:
    """Train RF and compute metrics on test set."""
    rf, y_true, y_pred = train_random_forest(cfg)
    class_names = cfg["classes"]["names"]
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["model"] = "Random Forest (baseline)"
    metrics["split"] = "test"
    print_metrics_table(metrics, class_names)
    return metrics


# =============================================================================
# Lightweight U-Net baseline
# =============================================================================

class UNetBlock(torch.nn.Module if TORCH_AVAILABLE else object):
    """Encoder block: Conv → BN → ReLU → Conv → BN → ReLU."""
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_ch, out_ch, 3, padding=1),
            torch.nn.BatchNorm2d(out_ch),
            torch.nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class SimpleUNet(torch.nn.Module if TORCH_AVAILABLE else object):
    """
    Lightweight U-Net for patch classification (trained from scratch).
    Outputs a single class label per patch (global average pooling after encoder).
    """
    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        import torch.nn as nn

        self.enc1 = UNetBlock(in_channels, 32)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = UNetBlock(32, 64)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = UNetBlock(64, 128)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.pool1(self.enc1(x))
        x = self.pool2(self.enc2(x))
        x = self.enc3(x)
        x = self.gap(x)
        return self.head(x)


@timer
def train_unet_baseline(cfg: Dict) -> Dict:
    """Train U-Net from scratch and evaluate."""
    if not TORCH_AVAILABLE:
        return {"model": "U-Net (scratch)", "error": "PyTorch unavailable"}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, val_loader, test_loader, _ = build_dataloaders(cfg)

    model = SimpleUNet(
        in_channels=cfg["patches"]["bands"],
        num_classes=cfg["classes"]["num_classes"],
    ).to(device)

    unet_cfg = cfg.get("baselines", {}).get("unet_scratch", {})
    optimizer = torch.optim.Adam(model.parameters(), lr=unet_cfg.get("lr", 1e-3))
    criterion = torch.nn.CrossEntropyLoss()
    epochs = unet_cfg.get("epochs", 30)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                val_loss += criterion(model(x), y).item()
        val_loss /= len(val_loader)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0:
            logger.info(f"  U-Net epoch {epoch+1}/{epochs}  val_loss={val_loss:.4f}")

    model.load_state_dict(best_state)
    y_true, y_pred = run_inference(model, test_loader, device)
    class_names = cfg["classes"]["names"]
    metrics = compute_metrics(y_true, y_pred, class_names)
    metrics["model"] = "U-Net (scratch baseline)"
    metrics["split"] = "test"
    print_metrics_table(metrics, class_names)
    return metrics


# =============================================================================
# Comparison table
# =============================================================================

def compare_models(metrics_list: List[Dict], save_path: str):
    """Create a comparison table and save as CSV + print to console."""
    import pandas as pd

    rows = []
    for m in metrics_list:
        if not m:
            continue
        rows.append({
            "Model": m.get("model", "?"),
            "Accuracy": f"{m.get('accuracy', 0):.4f}",
            "Precision": f"{m.get('precision_macro', 0):.4f}",
            "Recall": f"{m.get('recall_macro', 0):.4f}",
            "F1 (macro)": f"{m.get('f1_macro', 0):.4f}",
            "IoU (macro)": f"{m.get('iou_macro', 0):.4f}",
        })

    df = pd.DataFrame(rows)
    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)
    print(df.to_string(index=False))
    print("=" * 70 + "\n")

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Comparison table saved: {save_path}")
    return df


# =============================================================================
# Confusion matrix
# =============================================================================

def plot_confusion_matrix(
    metrics: Dict,
    class_names: List[str],
    model_name: str = "Model",
    save_path: Optional[str] = None,
):
    """Plot a normalised confusion matrix heatmap."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        logger.warning("matplotlib/seaborn not installed; skipping confusion matrix plot")
        return

    cm = np.array(metrics["confusion_matrix"])
    # Normalise
    row_sums = cm.sum(axis=1, keepdims=True)
    cm_norm = cm / np.maximum(row_sums, 1)

    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(
        cm_norm, annot=True, fmt=".2f", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        ax=ax, vmin=0, vmax=1,
    )
    ax.set_ylabel("True Label", fontsize=11)
    ax.set_xlabel("Predicted Label", fontsize=11)
    ax.set_title(f"Confusion Matrix — {model_name}", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Confusion matrix saved: {save_path}")
    plt.show()


# =============================================================================
# Entry point
# =============================================================================

def run(config_path: str = "configs/config.yaml", skip_baselines: bool = False):
    """Full evaluation pipeline."""
    cfg = load_config(config_path)
    reports_dir = cfg["paths"]["reports"]
    os.makedirs(reports_dir, exist_ok=True)
    class_names = cfg["classes"]["names"]

    all_metrics = []

    # 1. Foundation model
    logger.info("\n── Foundation Model Evaluation ──────────────────────────")
    fm_metrics = evaluate_foundation_model(cfg, split="test")
    if fm_metrics:
        all_metrics.append(fm_metrics)
        with open(os.path.join(reports_dir, "metrics_foundation_model.json"), "w") as f:
            json.dump(fm_metrics, f, indent=2)
        plot_confusion_matrix(
            fm_metrics, class_names, "Foundation Model",
            save_path=os.path.join(reports_dir, "confusion_matrix_fm.png"),
        )

    if not skip_baselines:
        # 2. Random Forest
        logger.info("\n── Random Forest Baseline ──────────────────────────────")
        try:
            rf_metrics = evaluate_random_forest(cfg)
            all_metrics.append(rf_metrics)
            with open(os.path.join(reports_dir, "metrics_random_forest.json"), "w") as f:
                json.dump(rf_metrics, f, indent=2)
            plot_confusion_matrix(
                rf_metrics, class_names, "Random Forest",
                save_path=os.path.join(reports_dir, "confusion_matrix_rf.png"),
            )
        except Exception as e:
            logger.warning(f"Random Forest baseline failed: {e}")

        # 3. U-Net from scratch
        logger.info("\n── U-Net Baseline ──────────────────────────────────────")
        try:
            unet_metrics = train_unet_baseline(cfg)
            all_metrics.append(unet_metrics)
            with open(os.path.join(reports_dir, "metrics_unet.json"), "w") as f:
                json.dump(unet_metrics, f, indent=2)
        except Exception as e:
            logger.warning(f"U-Net baseline failed: {e}")

    # Comparison table
    compare_models(all_metrics, save_path=os.path.join(reports_dir, "model_comparison.csv"))

    logger.info("✓ Evaluation complete.")
    return all_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate crop classification models")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--skip_baselines", action="store_true",
                        help="Skip baseline model training (faster)")
    args = parser.parse_args()
    run(args.config, args.skip_baselines)
