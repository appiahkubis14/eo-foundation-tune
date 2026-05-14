"""
utils.py — Shared utilities for Project 1: Foundation Model Fine-Tuning
Author: Samuel Appiah Kubi
Copernicus Master's in Digital Earth, Paris Lodron University Salzburg
"""

import os
import json
import logging
import time
import random
from pathlib import Path
from functools import wraps
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import yaml


# =============================================================================
# Logging
# =============================================================================

def setup_logging(log_dir: str = "data/logs", level: int = logging.INFO) -> logging.Logger:
    """Configure logging to console and rotating file."""
    os.makedirs(log_dir, exist_ok=True)
    log_path = Path(log_dir) / "project1.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"

    logging.basicConfig(
        level=level,
        format=fmt,
        datefmt=datefmt,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, mode="a"),
        ],
    )
    return logging.getLogger("crop_mapping")


logger = setup_logging()


# =============================================================================
# Config
# =============================================================================

def load_config(config_path: str = "configs/config.yaml") -> Dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    logger.info(f"Config loaded from {config_path}")
    return cfg


# =============================================================================
# Reproducibility
# =============================================================================

def set_seeds(seed: int = 42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass
    logger.info(f"Seeds set to {seed}")


# =============================================================================
# Timing
# =============================================================================

def timer(func):
    """Decorator: log execution time of any function."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - start
        logger.info(f"{func.__name__} completed in {elapsed:.1f}s")
        return result
    return wrapper


# =============================================================================
# Checkpoint helpers
# =============================================================================

def save_checkpoint(data: Dict, path: str):
    """Save a JSON checkpoint."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.debug(f"Checkpoint saved to {path}")


def load_checkpoint(path: str) -> Optional[Dict]:
    """Load a JSON checkpoint, return None if missing."""
    if os.path.exists(path):
        with open(path, "r") as f:
            data = json.load(f)
        logger.info(f"Checkpoint loaded from {path}")
        return data
    return None


# =============================================================================
# Spectral Indices
# =============================================================================

def compute_ndvi(red: np.ndarray, nir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """NDVI = (NIR - Red) / (NIR + Red). Clips to [-1, 1]."""
    ndvi = (nir - red) / (nir + red + eps)
    return np.clip(ndvi, -1.0, 1.0).astype(np.float32)


def compute_ndwi(green: np.ndarray, nir: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """NDWI = (Green - NIR) / (Green + NIR). Clips to [-1, 1]."""
    ndwi = (green - nir) / (green + nir + eps)
    return np.clip(ndwi, -1.0, 1.0).astype(np.float32)


def compute_evi(
    blue: np.ndarray,
    red: np.ndarray,
    nir: np.ndarray,
    eps: float = 1e-8,
) -> np.ndarray:
    """EVI = 2.5 * (NIR - Red) / (NIR + 6*Red - 7.5*Blue + 1). Clips to [-1, 1]."""
    evi = 2.5 * (nir - red) / (nir + 6 * red - 7.5 * blue + 1.0 + eps)
    return np.clip(evi, -1.0, 1.0).astype(np.float32)


def add_spectral_indices(
    stack: np.ndarray,
    band_names: List[str],
) -> Tuple[np.ndarray, List[str]]:
    """
    Append NDVI, NDWI, EVI to a band stack.

    Parameters
    ----------
    stack : np.ndarray  shape (C, H, W)  — spectral bands
    band_names : list[str]               — e.g. ['B02', 'B03', 'B04', 'B08', 'B11']

    Returns
    -------
    augmented_stack : np.ndarray  shape (C+3, H, W)
    new_band_names  : list[str]
    """
    idx = {b: i for i, b in enumerate(band_names)}
    blue = stack[idx["B02"]] / 10000.0
    green = stack[idx["B03"]] / 10000.0
    red = stack[idx["B04"]] / 10000.0
    nir = stack[idx["B08"]] / 10000.0

    ndvi = compute_ndvi(red, nir)
    ndwi = compute_ndwi(green, nir)
    evi = compute_evi(blue, red, nir)

    augmented = np.concatenate(
        [stack / 10000.0, ndvi[np.newaxis], ndwi[np.newaxis], evi[np.newaxis]], axis=0
    )
    new_names = band_names + ["NDVI", "NDWI", "EVI"]
    return augmented.astype(np.float32), new_names


# =============================================================================
# SCL Cloud masking
# =============================================================================

def make_cloud_mask(scl: np.ndarray, mask_classes: List[int] = None) -> np.ndarray:
    """
    Build a boolean cloud mask from the Sentinel-2 SCL band.
    True = cloudy / invalid pixel.
    """
    if mask_classes is None:
        mask_classes = [0, 1, 3, 8, 9, 10]  # nodata, saturated, cloud shadow, clouds
    mask = np.zeros_like(scl, dtype=bool)
    for cls in mask_classes:
        mask |= (scl == cls)
    return mask


def apply_cloud_mask(
    stack: np.ndarray, mask: np.ndarray, fill_value: float = np.nan
) -> np.ndarray:
    """Apply cloud mask to a (C, H, W) stack, setting masked pixels to fill_value."""
    out = stack.astype(np.float32).copy()
    out[:, mask] = fill_value
    return out


# =============================================================================
# Temporal compositing
# =============================================================================

def temporal_median(
    scenes: List[np.ndarray],
) -> np.ndarray:
    """
    Compute pixel-wise median across a list of (C, H, W) arrays.
    NaN values (cloud-masked) are ignored.

    Returns
    -------
    composite : np.ndarray  shape (C, H, W)
    """
    stack = np.stack(scenes, axis=0)  # (T, C, H, W)
    composite = np.nanmedian(stack, axis=0)
    return composite.astype(np.float32)


# =============================================================================
# Patch extraction
# =============================================================================

def extract_patches(
    data: np.ndarray,
    patch_size: int = 32,
    stride: int = 16,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Extract overlapping patches from a (C, H, W) array.

    Returns
    -------
    patches   : np.ndarray  shape (N, C, patch_size, patch_size)
    positions : list of (row, col) top-left corners
    """
    C, H, W = data.shape
    patches, positions = [], []

    for r in range(0, H - patch_size + 1, stride):
        for c in range(0, W - patch_size + 1, stride):
            patch = data[:, r : r + patch_size, c : c + patch_size]
            # Skip patches that are mostly NaN (>50%)
            if np.isnan(patch).mean() > 0.5:
                continue
            # Replace remaining NaN with 0
            patch = np.nan_to_num(patch, nan=0.0)
            patches.append(patch)
            positions.append((r, c))

    return np.array(patches, dtype=np.float32), positions


def reconstruct_from_patches(
    patches: np.ndarray,
    positions: List[Tuple[int, int]],
    output_shape: Tuple[int, int],
    patch_size: int = 32,
) -> np.ndarray:
    """
    Reconstruct a map from patches by averaging overlapping regions.

    Parameters
    ----------
    patches      : np.ndarray  shape (N, num_classes)  — predicted probabilities
    positions    : list of (row, col)
    output_shape : (H, W)
    patch_size   : int

    Returns
    -------
    result : np.ndarray  shape (num_classes, H, W)
    """
    num_classes = patches.shape[1]
    H, W = output_shape
    accumulator = np.zeros((num_classes, H, W), dtype=np.float64)
    count = np.zeros((H, W), dtype=np.float64)

    for probs, (r, c) in zip(patches, positions):
        for cls in range(num_classes):
            accumulator[cls, r : r + patch_size, c : c + patch_size] += probs[cls]
        count[r : r + patch_size, c : c + patch_size] += 1.0

    count = np.maximum(count, 1.0)
    result = accumulator / count[np.newaxis]
    return result.astype(np.float32)


# =============================================================================
# Spatial split helpers
# =============================================================================

def spatial_train_val_test_split(
    labels_gdf,
    n_folds: int = 3,
    test_fold: int = 2,
    epsg: int = 32630,
) -> Tuple:
    """
    Split a GeoDataFrame of field polygons into train/val/test by spatial fold.
    Fold assignment is based on grid quadrant of polygon centroid.

    Returns
    -------
    train_gdf, val_gdf, test_gdf
    """
    import geopandas as gpd

    gdf = labels_gdf.to_crs(epsg=epsg)
    bounds = gdf.total_bounds  # (minx, miny, maxx, maxy)
    mid_x = (bounds[0] + bounds[2]) / 2
    mid_y = (bounds[1] + bounds[3]) / 2

    def assign_fold(geom):
        cx, cy = geom.centroid.x, geom.centroid.y
        if cx < mid_x and cy >= mid_y:
            return 0  # NW
        elif cx >= mid_x and cy >= mid_y:
            return 1  # NE
        else:
            return 2  # South half

    gdf["fold"] = gdf["geometry"].apply(assign_fold)

    val_fold = (test_fold - 1) % n_folds
    test_gdf = gdf[gdf["fold"] == test_fold]
    val_gdf = gdf[gdf["fold"] == val_fold]
    train_gdf = gdf[~gdf["fold"].isin([test_fold, val_fold])]

    logger.info(
        f"Spatial split — train: {len(train_gdf)}, val: {len(val_gdf)}, test: {len(test_gdf)}"
    )
    return train_gdf, val_gdf, test_gdf


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str],
) -> Dict[str, Any]:
    """
    Compute per-class and macro-averaged classification metrics.

    Parameters
    ----------
    y_true, y_pred : 1-D arrays of integer class labels
    class_names    : list of class names

    Returns
    -------
    metrics_dict : dict with keys accuracy, precision, recall, f1, iou (per-class + macro)
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
        confusion_matrix,
        jaccard_score,
    )

    metrics = {}
    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["precision_macro"] = float(precision_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["recall_macro"] = float(recall_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["f1_macro"] = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    metrics["iou_macro"] = float(jaccard_score(y_true, y_pred, average="macro", zero_division=0))

    # Per-class
    precision_pc = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_pc = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_pc = f1_score(y_true, y_pred, average=None, zero_division=0)
    iou_pc = jaccard_score(y_true, y_pred, average=None, zero_division=0)

    metrics["per_class"] = {}
    for i, name in enumerate(class_names):
        if i < len(precision_pc):
            metrics["per_class"][name] = {
                "precision": float(precision_pc[i]),
                "recall": float(recall_pc[i]),
                "f1": float(f1_pc[i]),
                "iou": float(iou_pc[i]),
            }

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    metrics["confusion_matrix"] = cm.tolist()

    return metrics


def print_metrics_table(metrics: Dict, class_names: List[str]):
    """Pretty-print per-class metrics table."""
    header = f"{'Class':<12} {'Precision':>10} {'Recall':>10} {'F1':>10} {'IoU':>10}"
    print("\n" + "=" * 55)
    print(header)
    print("-" * 55)
    for name in class_names:
        if name in metrics.get("per_class", {}):
            m = metrics["per_class"][name]
            print(
                f"{name:<12} {m['precision']:>10.3f} {m['recall']:>10.3f} "
                f"{m['f1']:>10.3f} {m['iou']:>10.3f}"
            )
    print("-" * 55)
    print(
        f"{'MACRO':<12} {metrics['precision_macro']:>10.3f} {metrics['recall_macro']:>10.3f} "
        f"{metrics['f1_macro']:>10.3f} {metrics['iou_macro']:>10.3f}"
    )
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    print("=" * 55 + "\n")


# =============================================================================
# GeoTIFF helpers
# =============================================================================

def save_geotiff(
    array: np.ndarray,
    reference_path: str,
    output_path: str,
    dtype: str = "float32",
    nodata: float = -9999.0,
):
    """
    Save a (C, H, W) array as a GeoTIFF, inheriting CRS and transform from reference.
    """
    import rasterio

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(reference_path) as ref:
        profile = ref.profile.copy()

    if array.ndim == 2:
        array = array[np.newaxis]

    profile.update(
        count=array.shape[0],
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        blockxsize=256,
        blockysize=256,
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(array.astype(dtype))
    logger.info(f"GeoTIFF saved to {output_path}  shape={array.shape}")


def load_geotiff(path: str) -> Tuple[np.ndarray, Any]:
    """
    Load a GeoTIFF.

    Returns
    -------
    array   : np.ndarray  shape (C, H, W)
    profile : rasterio profile dict
    """
    import rasterio

    with rasterio.open(path) as src:
        array = src.read().astype(np.float32)
        profile = src.profile
    return array, profile


# =============================================================================
# Normalisation
# =============================================================================

class ChannelwiseNorm:
    """Per-channel mean/std normalisation (fitted on training data)."""

    def __init__(self):
        self.mean_ = None
        self.std_ = None

    def fit(self, patches: np.ndarray):
        """
        Parameters
        ----------
        patches : np.ndarray  shape (N, C, H, W)
        """
        # Compute per-channel statistics over all spatial / sample dims
        flat = patches.reshape(patches.shape[0], patches.shape[1], -1)  # (N, C, H*W)
        self.mean_ = flat.mean(axis=(0, 2))  # (C,)
        self.std_ = flat.std(axis=(0, 2)) + 1e-8
        return self

    def transform(self, patches: np.ndarray) -> np.ndarray:
        return (patches - self.mean_[np.newaxis, :, np.newaxis, np.newaxis]) / \
               self.std_[np.newaxis, :, np.newaxis, np.newaxis]

    def fit_transform(self, patches: np.ndarray) -> np.ndarray:
        return self.fit(patches).transform(patches)

    def save(self, path: str):
        np.savez(path, mean=self.mean_, std=self.std_)

    @classmethod
    def load(cls, path: str) -> "ChannelwiseNorm":
        obj = cls()
        data = np.load(path)
        obj.mean_ = data["mean"]
        obj.std_ = data["std"]
        return obj
