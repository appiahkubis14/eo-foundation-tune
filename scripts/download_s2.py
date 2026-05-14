"""
download_s2.py — Sentinel-2 L2A Download and Cloud Masking
===========================================================
Author: Samuel Appiah Kubi
Copernicus Master's in Digital Earth, Paris Lodron University Salzburg

Workflow
--------
1. Authenticate with Copernicus Data Space via eodag
2. Search for S2 L2A scenes over the study area and time window
3. Download up to max_scenes scenes below cloud cover threshold
4. Cloud-mask each scene using the SCL band
5. Compute spectral indices (NDVI, NDWI, EVI)
6. Create a temporal composite (nanmedian across scenes)
7. Save per-scene stacks + composite to data/processed/

Usage
-----
    python scripts/download_s2.py --config configs/config.yaml
"""

import os
import sys
import json
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import (
    load_config, setup_logging, timer, save_checkpoint, load_checkpoint,
    make_cloud_mask, apply_cloud_mask, add_spectral_indices,
    temporal_median, save_geotiff, load_geotiff,
)

logger = logging.getLogger("crop_mapping.download_s2")


# =============================================================================
# Authentication
# =============================================================================

def setup_eodag(username: str = None, password: str = None) -> "EODataAccessGateway":
    """
    Initialise eodag with Copernicus Data Space credentials.
    Credentials can be provided via:
      - Function arguments
      - Environment variables: EODAG_USERNAME, EODAG_PASSWORD
      - ~/.config/eodag/eodag.yaml
    """
    try:
        from eodag import EODataAccessGateway
    except ImportError:
        raise ImportError("Run: pip install eodag")

    username = username or os.environ.get("EODAG_USERNAME")
    password = password or os.environ.get("EODAG_PASSWORD")

    if not username or not password:
        logger.warning(
            "No Copernicus credentials found. Set EODAG_USERNAME and EODAG_PASSWORD "
            "environment variables or pass via --username / --password. "
            "You can register free at: https://dataspace.copernicus.eu/"
        )

    dag = EODataAccessGateway()
    if username and password:
        dag.update_providers_config(
            f"""
            cop_dataspace:
                credentials:
                    username: '{username}'
                    password: '{password}'
            """
        )
    return dag


# =============================================================================
# Scene Search
# =============================================================================

@timer
def search_scenes(dag, cfg: Dict) -> List:
    """Search for Sentinel-2 L2A scenes matching config parameters."""
    s2_cfg = cfg["sentinel2"]
    sa = cfg["study_area"]

    bbox = sa["bbox"]  # [min_lat, min_lon, max_lat, max_lon]

    logger.info(
        f"Searching {s2_cfg['product_type']} "
        f"{s2_cfg['start_date']} → {s2_cfg['end_date']} "
        f"cloud ≤ {s2_cfg['cloud_cover_max']}%"
    )

    products, total = dag.search(
        productType=s2_cfg["product_type"],
        start=s2_cfg["start_date"],
        end=s2_cfg["end_date"],
        geom={
            "type": "Polygon",
            "coordinates": [[
                [bbox[1], bbox[0]], [bbox[3], bbox[0]],
                [bbox[3], bbox[2]], [bbox[1], bbox[2]],
                [bbox[1], bbox[0]],
            ]],
        },
        cloudCover=s2_cfg["cloud_cover_max"],
        provider=s2_cfg.get("provider", "cop_dataspace"),
    )

    max_scenes = s2_cfg.get("max_scenes", 10)
    products = products[: max_scenes]
    logger.info(f"Found {total} scenes, selecting {len(products)}")
    return products


# =============================================================================
# Download
# =============================================================================

@timer
def download_scenes(dag, products: List, raw_dir: str) -> List[Path]:
    """Download scenes to raw_dir, skip already-downloaded ones."""
    os.makedirs(raw_dir, exist_ok=True)
    downloaded = []
    for i, product in enumerate(products):
        scene_dir = Path(raw_dir) / product.properties.get("id", f"scene_{i:03d}")
        if scene_dir.exists():
            logger.info(f"[{i+1}/{len(products)}] Already downloaded: {scene_dir.name}")
            downloaded.append(scene_dir)
            continue
        logger.info(f"[{i+1}/{len(products)}] Downloading {product.properties.get('id', '?')}…")
        try:
            paths = dag.download(product, output_dir=str(raw_dir))
            downloaded.append(Path(paths))
        except Exception as e:
            logger.warning(f"  Failed: {e}")
    return downloaded


# =============================================================================
# Band loading
# =============================================================================

def load_scene_bands(
    scene_path: Path,
    bands: List[str],
    scl_class: str = "SCL",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load spectral bands + SCL from a Sentinel-2 SAFE directory or flat folder.

    Returns
    -------
    stack   : np.ndarray  shape (C, H, W) – spectral bands (DN, not reflectance)
    scl     : np.ndarray  shape (H, W)    – scene classification layer
    meta    : dict                        – rasterio profile of first band
    """
    import rasterio
    from rasterio.enums import Resampling

    # Find all .jp2 / .tif files
    all_files = list(scene_path.rglob("*.jp2")) + list(scene_path.rglob("*.tif"))

    def find_band(band_name: str) -> Path:
        for f in all_files:
            if f"_{band_name}_" in f.name or f.stem.endswith(f"_{band_name}"):
                return f
        raise FileNotFoundError(f"Band {band_name} not found in {scene_path}")

    # Load first band to get reference shape
    ref_band_path = find_band(bands[0])
    with rasterio.open(ref_band_path) as ref:
        H, W = ref.height, ref.width
        profile = ref.profile.copy()

    stack = np.zeros((len(bands), H, W), dtype=np.uint16)
    for i, band in enumerate(bands):
        band_path = find_band(band)
        with rasterio.open(band_path) as src:
            data = src.read(
                1,
                out_shape=(H, W),
                resampling=Resampling.bilinear,
            )
            stack[i] = data

    # SCL band (20 m → resample to 10 m if needed)
    scl_path = find_band(scl_class)
    with rasterio.open(scl_path) as src:
        scl = src.read(1, out_shape=(H, W), resampling=Resampling.nearest)

    return stack, scl, profile


# =============================================================================
# Processing pipeline
# =============================================================================

@timer
def process_scenes(
    scene_paths: List[Path],
    cfg: Dict,
    processed_dir: str,
) -> List[str]:
    """
    For each downloaded scene:
      1. Load bands + SCL
      2. Apply cloud mask
      3. Add spectral indices
      4. Save as GeoTIFF
    Returns list of processed file paths.
    """
    os.makedirs(processed_dir, exist_ok=True)
    bands = cfg["sentinel2"]["bands"]
    mask_classes = cfg["sentinel2"]["scl_mask_classes"]
    checkpoint_path = os.path.join(processed_dir, "process_checkpoint.json")
    ckpt = load_checkpoint(checkpoint_path) or {"done": []}

    output_paths = []
    for i, scene_path in enumerate(scene_paths):
        scene_id = scene_path.name
        out_path = os.path.join(processed_dir, f"{scene_id}_masked.tif")

        if scene_id in ckpt["done"]:
            logger.info(f"[{i+1}/{len(scene_paths)}] Already processed: {scene_id}")
            output_paths.append(out_path)
            continue

        logger.info(f"[{i+1}/{len(scene_paths)}] Processing {scene_id}…")
        try:
            raw_stack, scl, profile = load_scene_bands(scene_path, bands)
            cloud_mask = make_cloud_mask(scl, mask_classes)
            masked = apply_cloud_mask(raw_stack.astype(np.float32), cloud_mask, fill_value=np.nan)
            augmented, new_bands = add_spectral_indices(masked, bands)

            # Update profile for augmented bands
            profile.update(count=augmented.shape[0], dtype="float32", nodata=np.nan)
            import rasterio
            with rasterio.open(out_path, "w", **profile) as dst:
                dst.write(augmented)

            output_paths.append(out_path)
            ckpt["done"].append(scene_id)
            save_checkpoint(ckpt, checkpoint_path)
            logger.info(f"  → {out_path}  bands={new_bands}")

        except Exception as e:
            logger.warning(f"  Failed to process {scene_id}: {e}")

    return output_paths


# =============================================================================
# Temporal composite
# =============================================================================

@timer
def create_composite(processed_paths: List[str], output_path: str) -> str:
    """
    Load all processed scenes and compute a pixel-wise nanmedian composite.
    Saves result as GeoTIFF.
    """
    import rasterio

    scenes = []
    profile = None
    for p in processed_paths:
        if not os.path.exists(p):
            continue
        with rasterio.open(p) as src:
            arr = src.read().astype(np.float32)
            if profile is None:
                profile = src.profile.copy()
        scenes.append(arr)

    if not scenes:
        raise ValueError("No valid processed scenes to composite.")

    composite = temporal_median(scenes)
    profile.update(count=composite.shape[0], dtype="float32", compress="deflate")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(composite)

    logger.info(f"Composite saved: {output_path}  shape={composite.shape}")
    return output_path


# =============================================================================
# Temporal stack (all scenes aligned)
# =============================================================================

@timer
def create_temporal_stack(
    processed_paths: List[str],
    output_path: str,
    max_timesteps: int = 6,
) -> str:
    """
    Create a temporal stack (T, C, H, W) → saved as a numpy .npz file.
    Selects up to max_timesteps scenes, evenly spaced.
    """
    import rasterio

    if len(processed_paths) > max_timesteps:
        indices = np.linspace(0, len(processed_paths) - 1, max_timesteps, dtype=int)
        selected = [processed_paths[i] for i in indices]
    else:
        selected = processed_paths

    scenes = []
    for p in selected:
        if not os.path.exists(p):
            continue
        with rasterio.open(p) as src:
            arr = src.read().astype(np.float32)
        scenes.append(arr)

    stack = np.stack(scenes, axis=0)  # (T, C, H, W)
    np.savez_compressed(output_path, stack=stack, n_timesteps=len(scenes))
    logger.info(f"Temporal stack saved: {output_path}  shape={stack.shape}")
    return output_path


# =============================================================================
# CLI entry point
# =============================================================================

def run(config_path: str = "configs/config.yaml", username: str = None, password: str = None):
    """Full download + processing pipeline."""
    cfg = load_config(config_path)

    raw_dir = cfg["paths"]["raw"]
    processed_dir = cfg["paths"]["processed"]

    composite_path = os.path.join(processed_dir, "s2_composite.tif")
    stack_path = os.path.join(processed_dir, "s2_temporal_stack")

    # Authenticate
    dag = setup_eodag(username, password)

    # Search + download
    products = search_scenes(dag, cfg)
    if not products:
        logger.error("No scenes found. Check study area, dates, and cloud cover settings.")
        return

    scene_paths = download_scenes(dag, products, raw_dir)

    # Process
    processed_paths = process_scenes(scene_paths, cfg, processed_dir)

    # Composite & stack
    create_composite(processed_paths, composite_path)
    create_temporal_stack(
        processed_paths,
        stack_path,
        max_timesteps=cfg["patches"]["timesteps"],
    )

    logger.info("✓ Sentinel-2 download and processing complete.")
    return composite_path, stack_path + ".npz"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process Sentinel-2 data")
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--username", default=None, help="Copernicus username")
    parser.add_argument("--password", default=None, help="Copernicus password")
    args = parser.parse_args()
    run(args.config, args.username, args.password)
