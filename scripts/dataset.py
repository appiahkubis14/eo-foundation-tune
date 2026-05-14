"""
dataset.py — PyTorch Dataset and DataLoader for Crop Mapping
============================================================
Author: Samuel Appiah Kubi
Copernicus Master's in Digital Earth, Paris Lodron University Salzburg

CropPatchDataset
  Loads patches of Sentinel-2 time series + crop type labels.
  Input  : (C, H, W) or (T, C, H, W) array per patch
  Target : integer class label

Augmentations
  random flip, 90°/180°/270° rotation, brightness/contrast jitter
"""

import os
import sys
import logging
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import (
    load_config, extract_patches,
    extract_patch_labels, load_geotiff, ChannelwiseNorm,
)

logger = logging.getLogger("crop_mapping.dataset")


try:
    import torch
    from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed. Dataset classes unavailable.")


# =============================================================================
# Augmentations (numpy, applied before tensor conversion)
# =============================================================================

class RandomFlip:
    """Random horizontal and/or vertical flip."""
    def __init__(self, p: float = 0.5):
        self.p = p

    def __call__(self, x: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            x = np.flip(x, axis=-1).copy()   # horizontal
        if np.random.rand() < self.p:
            x = np.flip(x, axis=-2).copy()   # vertical
        return x


class RandomRotation90:
    """Rotate by 0 / 90 / 180 / 270 degrees uniformly."""
    def __call__(self, x: np.ndarray) -> np.ndarray:
        k = np.random.randint(0, 4)
        return np.rot90(x, k=k, axes=(-2, -1)).copy()


class BrightnessContrastJitter:
    """Apply random brightness + contrast to each channel independently."""
    def __init__(self, brightness: float = 0.2, contrast: float = 0.15):
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, x: np.ndarray) -> np.ndarray:
        # Brightness
        alpha = 1.0 + np.random.uniform(-self.brightness, self.brightness)
        x = x * alpha
        # Contrast
        beta = np.random.uniform(-self.contrast, self.contrast)
        mean = x.mean(axis=(-2, -1), keepdims=True)
        x = (x - mean) * (1.0 + beta) + mean
        return x.astype(np.float32)


class Compose:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: np.ndarray) -> np.ndarray:
        for t in self.transforms:
            x = t(x)
        return x


def get_augmentations(cfg: Dict, mode: str = "train") -> Optional[Compose]:
    """Build augmentation pipeline from config."""
    if mode != "train":
        return None  # no augmentation at val/test time
    aug_cfg = cfg["training"].get("augmentation", {})
    transforms = []
    if aug_cfg.get("random_flip", True):
        transforms.append(RandomFlip(p=0.5))
    if aug_cfg.get("random_rotation"):
        transforms.append(RandomRotation90())
    brightness = aug_cfg.get("brightness_jitter", 0.2)
    contrast = aug_cfg.get("contrast_jitter", 0.15)
    if brightness > 0 or contrast > 0:
        transforms.append(BrightnessContrastJitter(brightness, contrast))
    return Compose(transforms) if transforms else None


# =============================================================================
# Dataset
# =============================================================================

if TORCH_AVAILABLE:

    class CropPatchDataset(Dataset):
        """
        Dataset of (patch, label) pairs for crop type classification.

        Each item:
          x : torch.FloatTensor  shape (C, patch_size, patch_size)
              or (T, C, patch_size, patch_size) for temporal
          y : torch.LongTensor   scalar class label

        Parameters
        ----------
        patches      : np.ndarray  shape (N, C, H, W)
        labels       : np.ndarray  shape (N,)  – integer, -1 = ignored
        augmentations: Compose or None
        normaliser   : ChannelwiseNorm or None  (already fitted on train set)
        """

        def __init__(
            self,
            patches: np.ndarray,
            labels: np.ndarray,
            augmentations=None,
            normaliser: Optional[ChannelwiseNorm] = None,
        ):
            # Filter out ignored patches (label == -1)
            valid_mask = labels >= 0
            self.patches = patches[valid_mask].astype(np.float32)
            self.labels = labels[valid_mask].astype(np.int64)
            self.augmentations = augmentations
            self.normaliser = normaliser

            logger.info(
                f"Dataset: {len(self.patches)} valid patches "
                f"(discarded {(~valid_mask).sum()} unlabelled)"
            )
            self._log_class_distribution()

        def _log_class_distribution(self):
            unique, counts = np.unique(self.labels, return_counts=True)
            logger.info(f"  Class distribution: { {int(u): int(c) for u, c in zip(unique, counts)} }")

        def __len__(self) -> int:
            return len(self.patches)

        def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
            x = self.patches[idx].copy()
            if self.augmentations:
                x = self.augmentations(x)
            if self.normaliser:
                x = self.normaliser.transform(x[np.newaxis])[0]
            x_tensor = torch.from_numpy(x)
            y_tensor = torch.tensor(self.labels[idx], dtype=torch.long)
            return x_tensor, y_tensor

        @property
        def class_weights(self) -> torch.Tensor:
            """Inverse-frequency class weights for WeightedRandomSampler / loss."""
            unique, counts = np.unique(self.labels, return_counts=True)
            n_classes = int(unique.max()) + 1
            freq = np.zeros(n_classes)
            for u, c in zip(unique, counts):
                freq[u] = c
            freq = np.where(freq == 0, 1e-6, freq)
            weights = 1.0 / freq
            weights = weights / weights.sum()
            return torch.tensor(weights, dtype=torch.float32)

        def get_sample_weights(self) -> np.ndarray:
            """Per-sample weights for WeightedRandomSampler."""
            class_w = self.class_weights.numpy()
            return class_w[self.labels]


# =============================================================================
# Patch builder
# =============================================================================

def build_patches_from_rasters(
    image_path: str,
    label_raster_path: str,
    cfg: Dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract patches from image + label rasters.

    Returns
    -------
    patches : np.ndarray  shape (N, C, patch_size, patch_size)
    labels  : np.ndarray  shape (N,)
    """
    patch_size = cfg["patches"]["size"]
    stride = cfg["patches"].get("stride", patch_size // 2)

    # Load image
    image, _ = load_geotiff(image_path)   # (C, H, W)
    image = np.nan_to_num(image, nan=0.0)

    # Load label raster
    label_arr, _ = load_geotiff(label_raster_path)  # (1, H, W)
    label_arr = label_arr[0].astype(np.int64)        # (H, W)

    # Extract patches
    patches, positions = extract_patches(image, patch_size=patch_size, stride=stride)

    # Extract labels (majority class per patch)
    labels = extract_patch_labels(
        label_arr,
        positions,
        patch_size=patch_size,
        background=255,
    )

    return patches, labels


# =============================================================================
# DataLoader factory
# =============================================================================

def build_dataloaders(
    cfg: Dict,
    normaliser: Optional[ChannelwiseNorm] = None,
    fit_normaliser: bool = True,
) -> Tuple["DataLoader", "DataLoader", "DataLoader", ChannelwiseNorm]:
    """
    Build train/val/test DataLoaders from config.

    Returns
    -------
    train_loader, val_loader, test_loader, normaliser
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required. Run: pip install torch")

    labels_dir = cfg["paths"]["labels"]
    processed_dir = cfg["paths"]["processed"]
    batch_size = cfg["training"]["batch_size"]

    composite = os.path.join(processed_dir, "s2_composite.tif")
    splits = {}
    for split in ("train", "val", "test"):
        label_path = os.path.join(labels_dir, f"{split}_label_raster.tif")
        if not os.path.exists(label_path):
            logger.warning(f"Label raster not found: {label_path}. Creating synthetic.")
            label_path = os.path.join(labels_dir, "label_raster.tif")  # fallback
        patches, labels = build_patches_from_rasters(composite, label_path, cfg)
        splits[split] = (patches, labels)

    # Fit normaliser on training data
    train_patches = splits["train"][0]
    if normaliser is None:
        normaliser = ChannelwiseNorm()
    if fit_normaliser and len(train_patches) > 0:
        normaliser.fit(train_patches)
        norm_path = os.path.join(cfg["paths"]["models"], "normaliser.npz")
        os.makedirs(cfg["paths"]["models"], exist_ok=True)
        normaliser.save(norm_path)
        logger.info(f"Normaliser saved to {norm_path}")

    datasets = {}
    for split in ("train", "val", "test"):
        patches, labels = splits[split]
        aug = get_augmentations(cfg, mode=split)
        datasets[split] = CropPatchDataset(patches, labels, aug, normaliser)

    # Weighted sampler for training (handles class imbalance)
    train_ds = datasets["train"]
    sample_weights = train_ds.get_sample_weights()
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size,
        sampler=sampler, num_workers=2, pin_memory=True,
    )
    val_loader = DataLoader(
        datasets["val"], batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )
    test_loader = DataLoader(
        datasets["test"], batch_size=batch_size,
        shuffle=False, num_workers=2, pin_memory=True,
    )

    logger.info(
        f"DataLoaders — train: {len(train_ds)}, val: {len(datasets['val'])}, "
        f"test: {len(datasets['test'])} patches"
    )
    return train_loader, val_loader, test_loader, normaliser


# =============================================================================
# Visualisation helper
# =============================================================================

def visualise_batch(
    patches: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    n_samples: int = 8,
    save_path: Optional[str] = None,
):
    """Plot a grid of patches with their class labels."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        logger.warning("matplotlib not installed; skipping visualisation")
        return

    n = min(n_samples, len(patches))
    fig, axes = plt.subplots(2, n // 2, figsize=(3 * (n // 2), 7))
    axes = axes.flatten()

    for i in range(n):
        ax = axes[i]
        patch = patches[i]
        # Show RGB (B4, B3, B2 → indices 2, 1, 0 after 5-band stack)
        if patch.shape[0] >= 3:
            rgb = np.stack([patch[2], patch[1], patch[0]], axis=-1)  # R, G, B
            # Percentile stretch
            lo, hi = np.percentile(rgb, [2, 98])
            rgb = np.clip((rgb - lo) / (hi - lo + 1e-8), 0, 1)
            ax.imshow(rgb)
        else:
            ax.imshow(patch[0], cmap="viridis")

        label = labels[i]
        class_label = class_names[label] if label < len(class_names) else f"class_{label}"
        ax.set_title(class_label, fontsize=9)
        ax.axis("off")

    for ax in axes[n:]:
        ax.axis("off")

    plt.suptitle("Sample Patches by Crop Class", fontsize=12, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Batch visualisation saved: {save_path}")
    plt.show()
