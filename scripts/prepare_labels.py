"""
prepare_labels.py — Ground Truth Label Preparation
====================================================
Author: Samuel Appiah Kubi
Copernicus Master's in Digital Earth, Paris Lodron University Salzburg

Supports three label sources:
  1. CropHarvest global dataset (auto-download)
  2. Manual KML/GeoJSON field polygons (from Google Earth)
  3. Synthetic demo labels (for testing the pipeline without field data)

Workflow
--------
1. Load labels from chosen source
2. Rasterise polygon labels to match Sentinel-2 grid
3. Spatial train/val/test split (NOT random)
4. Save label raster + split indices

Usage
-----
    python scripts/prepare_labels.py --config configs/config.yaml --source synthetic
    python scripts/prepare_labels.py --config configs/config.yaml --source kml --kml_path data/labels/fields.kml
    python scripts/prepare_labels.py --config configs/config.yaml --source cropharvest
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import (
    load_config, setup_logging, timer, save_checkpoint,
    spatial_train_val_test_split,
)

logger = logging.getLogger("crop_mapping.prepare_labels")


# =============================================================================
# CropHarvest loader
# =============================================================================

@timer
def load_cropharvest(study_bbox: list, label_map: Dict) -> "GeoDataFrame":
    """
    Download and filter CropHarvest labels for the study area.

    Parameters
    ----------
    study_bbox : [min_lat, min_lon, max_lat, max_lon]
    label_map  : dict mapping class name → integer id

    Returns
    -------
    GeoDataFrame with columns: geometry, crop_type, label_id
    """
    try:
        import geopandas as gpd
        from shapely.geometry import box
    except ImportError:
        raise ImportError("Run: pip install geopandas shapely")

    try:
        import cropharvest  # noqa
        logger.info("CropHarvest package found; loading dataset…")
        # CropHarvest provides labels for training crop classifiers
        # This is a simplified interface — see: github.com/nasaharvest/cropharvest
        from cropharvest.datasets import CropHarvest

        dataset = CropHarvest.create_benchmark_datasets("data/raw/cropharvest")[0]
        # Filter to study area bounding box
        min_lat, min_lon, max_lat, max_lon = study_bbox
        study_geom = box(min_lon, min_lat, max_lon, max_lat)

        features = []
        for idx in range(len(dataset)):
            item = dataset[idx]
            lat, lon = item.lat, item.lon
            if study_geom.contains(box(lon - 0.001, lat - 0.001, lon + 0.001, lat + 0.001)):
                crop_name = item.crop_type if hasattr(item, "crop_type") else "other"
                label_id = label_map.get(crop_name.lower(), label_map.get("other", 4))
                features.append({
                    "geometry": box(lon - 0.0005, lat - 0.0005, lon + 0.0005, lat + 0.0005),
                    "crop_type": crop_name,
                    "label_id": label_id,
                })

        gdf = gpd.GeoDataFrame(features, crs="EPSG:4326")
        logger.info(f"CropHarvest: {len(gdf)} labels in study area")
        return gdf

    except ImportError:
        logger.warning("CropHarvest package not installed. Falling back to synthetic labels.")
        return create_synthetic_labels(study_bbox, label_map)


# =============================================================================
# KML / GeoJSON loader
# =============================================================================

@timer
def load_from_file(file_path: str, label_column: str, label_map: Dict) -> "GeoDataFrame":
    """
    Load crop type labels from a KML, GeoJSON, or Shapefile.

    Parameters
    ----------
    file_path    : path to the file
    label_column : column name that contains the crop type string
    label_map    : dict mapping crop name → integer label id

    Returns
    -------
    GeoDataFrame with geometry, crop_type, label_id columns
    """
    import geopandas as gpd

    ext = Path(file_path).suffix.lower()
    if ext == ".kml":
        gpd.io.file.fiona_env()
        import fiona
        fiona.drvsupport.supported_drivers["KML"] = "rw"
        gdf = gpd.read_file(file_path, driver="KML")
    else:
        gdf = gpd.read_file(file_path)

    if label_column not in gdf.columns:
        logger.warning(
            f"Column '{label_column}' not found. Available: {gdf.columns.tolist()}. "
            f"Assigning 'other' to all features."
        )
        gdf["crop_type"] = "other"
    else:
        gdf["crop_type"] = gdf[label_column].str.lower().str.strip()

    gdf["label_id"] = gdf["crop_type"].map(label_map).fillna(label_map.get("other", 4)).astype(int)
    gdf = gdf[["geometry", "crop_type", "label_id"]].to_crs("EPSG:4326")

    logger.info(f"Loaded {len(gdf)} polygons from {file_path}")
    logger.info(f"Class distribution:\n{gdf['crop_type'].value_counts()}")
    return gdf


# =============================================================================
# Synthetic label generator (for pipeline testing)
# =============================================================================

def create_synthetic_labels(study_bbox: list, label_map: Dict, n_per_class: int = 60) -> "GeoDataFrame":
    """
    Generate synthetic rectangular field polygons for testing.
    Each field is ~200m × 200m (about 0.002° × 0.002°).
    """
    import geopandas as gpd
    from shapely.geometry import box

    min_lat, min_lon, max_lat, max_lon = study_bbox
    rng = np.random.default_rng(42)
    records = []

    for crop_name, label_id in label_map.items():
        for _ in range(n_per_class):
            lat = rng.uniform(min_lat + 0.01, max_lat - 0.01)
            lon = rng.uniform(min_lon + 0.01, max_lon - 0.01)
            size = rng.uniform(0.001, 0.003)  # ~100–300m
            geom = box(lon, lat, lon + size, lat + size)
            records.append({"geometry": geom, "crop_type": crop_name, "label_id": label_id})

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    logger.info(f"Synthetic labels created: {len(gdf)} polygons ({n_per_class} per class)")
    return gdf


# =============================================================================
# Rasterisation
# =============================================================================

@timer
def rasterise_labels(
    gdf: "GeoDataFrame",
    reference_raster: str,
    output_path: str,
    label_column: str = "label_id",
    background: int = 255,
) -> str:
    """
    Burn polygon labels into a raster matching the reference GeoTIFF.

    Parameters
    ----------
    gdf              : GeoDataFrame with geometry + label_id
    reference_raster : path to a Sentinel-2 GeoTIFF for CRS/transform reference
    output_path      : where to save the label raster
    background       : value for unlabelled pixels (255 = ignore index)

    Returns
    -------
    output_path
    """
    import rasterio
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    with rasterio.open(reference_raster) as ref:
        profile = ref.profile.copy()
        transform = ref.transform
        H, W = ref.height, ref.width
        dst_crs = ref.crs

    gdf_proj = gdf.to_crs(dst_crs)
    shapes = list(zip(gdf_proj.geometry, gdf_proj[label_column].astype(int)))

    label_raster = rasterize(
        shapes,
        out_shape=(H, W),
        transform=transform,
        fill=background,
        dtype=np.uint8,
        all_touched=False,
    )

    profile.update(count=1, dtype="uint8", nodata=background, compress="lzw")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(label_raster[np.newaxis])

    unique, counts = np.unique(label_raster[label_raster != background], return_counts=True)
    class_info = {int(u): int(c) for u, c in zip(unique, counts)}
    logger.info(f"Label raster saved: {output_path}  class pixel counts: {class_info}")
    return output_path


# =============================================================================
# Save spatial split indices
# =============================================================================

def save_split(
    gdf: "GeoDataFrame",
    train_gdf: "GeoDataFrame",
    val_gdf: "GeoDataFrame",
    test_gdf: "GeoDataFrame",
    output_dir: str,
):
    """Save split GeoJSONs and a JSON summary."""
    os.makedirs(output_dir, exist_ok=True)

    for name, split in [("train", train_gdf), ("val", val_gdf), ("test", test_gdf)]:
        split.to_file(os.path.join(output_dir, f"{name}_labels.geojson"), driver="GeoJSON")

    summary = {
        "total": len(gdf),
        "train": len(train_gdf),
        "val": len(val_gdf),
        "test": len(test_gdf),
        "train_classes": train_gdf["crop_type"].value_counts().to_dict(),
        "val_classes": val_gdf["crop_type"].value_counts().to_dict(),
        "test_classes": test_gdf["crop_type"].value_counts().to_dict(),
    }
    with open(os.path.join(output_dir, "split_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Split summary: {summary}")


# =============================================================================
# Patch-level label extraction
# =============================================================================

def extract_patch_labels(
    label_raster: np.ndarray,
    positions: list,
    patch_size: int = 32,
    majority_threshold: float = 0.5,
    background: int = 255,
) -> np.ndarray:
    """
    For each patch position, determine the majority class label.
    Returns -1 for patches with no valid labels or ambiguous majority.

    Parameters
    ----------
    label_raster       : np.ndarray  shape (H, W)
    positions          : list of (row, col) top-left corners
    patch_size         : int
    majority_threshold : fraction of patch that must be labelled to accept
    background         : unlabelled pixel value

    Returns
    -------
    labels : np.ndarray  shape (N,)  — integer class label, -1 = discard
    """
    labels = []
    for (r, c) in positions:
        patch = label_raster[r : r + patch_size, c : c + patch_size]
        valid = patch[patch != background]
        if len(valid) < majority_threshold * patch_size * patch_size:
            labels.append(-1)
            continue
        # Majority vote
        unique, counts = np.unique(valid, return_counts=True)
        majority_class = unique[np.argmax(counts)]
        majority_frac = counts.max() / len(valid)
        labels.append(int(majority_class) if majority_frac >= 0.6 else -1)

    return np.array(labels, dtype=np.int64)


# =============================================================================
# Entry point
# =============================================================================

def run(
    config_path: str = "configs/config.yaml",
    source: str = "synthetic",
    kml_path: str = None,
    label_column: str = "crop_type",
):
    """Prepare ground truth labels and rasterise to Sentinel-2 grid."""
    cfg = load_config(config_path)
    label_map = cfg["classes"]["label_map"]
    study_bbox = cfg["study_area"]["bbox"]
    labels_dir = cfg["paths"]["labels"]
    processed_dir = cfg["paths"]["processed"]
    os.makedirs(labels_dir, exist_ok=True)

    # 1. Load labels
    if source == "cropharvest":
        gdf = load_cropharvest(study_bbox, label_map)
    elif source in ("kml", "geojson", "shp"):
        if not kml_path:
            raise ValueError("--kml_path required for file-based labels")
        gdf = load_from_file(kml_path, label_column, label_map)
    else:  # synthetic
        logger.info("Using synthetic labels (for testing only)")
        gdf = create_synthetic_labels(study_bbox, label_map)

    # 2. Spatial split
    cv_cfg = cfg["cross_validation"]
    train_gdf, val_gdf, test_gdf = spatial_train_val_test_split(
        gdf,
        n_folds=cv_cfg["n_folds"],
        test_fold=cv_cfg["test_fold"],
        epsg=cfg["study_area"]["epsg"],
    )
    save_split(gdf, train_gdf, val_gdf, test_gdf, labels_dir)

    # 3. Rasterise (using composite as reference grid)
    composite_path = os.path.join(processed_dir, "s2_composite.tif")
    if not os.path.exists(composite_path):
        logger.warning(
            f"Reference raster not found at {composite_path}. "
            "Run download_s2.py first, or provide a reference GeoTIFF."
        )
        # Create a minimal synthetic reference for testing
        _create_synthetic_reference(composite_path, study_bbox)

    label_raster_path = os.path.join(labels_dir, "label_raster.tif")
    rasterise_labels(gdf, composite_path, label_raster_path)

    # Per-split rasters
    for name, split_gdf in [("train", train_gdf), ("val", val_gdf), ("test", test_gdf)]:
        rasterise_labels(split_gdf, composite_path,
                         os.path.join(labels_dir, f"{name}_label_raster.tif"))

    logger.info("✓ Label preparation complete.")
    return label_raster_path


def _create_synthetic_reference(output_path: str, bbox: list):
    """Create a minimal 256×256 reference GeoTIFF for testing."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    H, W = 256, 256
    transform = from_bounds(bbox[1], bbox[0], bbox[3], bbox[2], W, H)
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": 8,
        "height": H, "width": W, "crs": CRS.from_epsg(4326),
        "transform": transform, "nodata": -9999.0,
    }
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(np.zeros((8, H, W), dtype=np.float32))
    logger.info(f"Synthetic reference raster created: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare crop type labels")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument(
        "--source", choices=["synthetic", "cropharvest", "kml", "geojson", "shp"],
        default="synthetic", help="Label data source"
    )
    parser.add_argument("--kml_path", default=None, help="Path to KML/GeoJSON/SHP file")
    parser.add_argument("--label_column", default="crop_type",
                        help="Column name with crop type strings")
    args = parser.parse_args()
    run(args.config, args.source, args.kml_path, args.label_column)
