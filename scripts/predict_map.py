"""
predict_map.py — Generate Full Crop Type Classification Map
============================================================
Author: Samuel Appiah Kubi
Copernicus Master's in Digital Earth, Paris Lodron University Salzburg

Streams patches through the best model checkpoint and reconstructs
a full GeoTIFF crop classification map with:
  - Class label raster (uint8)
  - Confidence raster (float32, max softmax probability)
  - Colour PNG visualisation with legend
"""

import os
import sys
import logging
import argparse
from pathlib import Path
from typing import Dict, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.utils import (
    load_config, timer, extract_patches, reconstruct_from_patches,
    save_geotiff, load_geotiff, ChannelwiseNorm,
)
from scripts.model import build_model

logger = logging.getLogger("crop_mapping.predict_map")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


# =============================================================================
# Batch prediction on patches
# =============================================================================

@torch.no_grad()
def predict_patches_batch(
    model: "torch.nn.Module",
    patches: np.ndarray,
    batch_size: int = 32,
    device: str = "cpu",
) -> np.ndarray:
    """
    Run batch inference on patches.

    Parameters
    ----------
    patches : np.ndarray  shape (N, C, H, W)

    Returns
    -------
    probs : np.ndarray  shape (N, num_classes) — softmax probabilities
    """
    model.eval()
    all_probs = []
    n = len(patches)

    for start in range(0, n, batch_size):
        batch = patches[start : start + batch_size]
        x = torch.from_numpy(batch).float().to(device)
        logits = model(x)
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        all_probs.append(probs)

        if (start // batch_size) % 10 == 0:
            logger.info(f"  Processed {min(start + batch_size, n)}/{n} patches…")

    return np.concatenate(all_probs, axis=0)  # (N, num_classes)


# =============================================================================
# Full map prediction
# =============================================================================

@timer
def predict_full_map(cfg: Dict) -> Dict[str, str]:
    """
    Produce a full crop type classification map.

    Steps:
    1. Load composite GeoTIFF
    2. Extract all patches (sliding window)
    3. Run batch inference through best model
    4. Reconstruct probability map (averaging overlaps)
    5. Take argmax for class label map
    6. Save: class_map.tif, confidence_map.tif, visualisation.png

    Returns
    -------
    dict of output file paths
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    models_dir = cfg["paths"]["models"]
    maps_dir = cfg["paths"]["maps"]
    os.makedirs(maps_dir, exist_ok=True)

    # Load model
    best_ckpt = os.path.join(models_dir, "best_model.pth")
    if not os.path.exists(best_ckpt):
        logger.warning(f"No checkpoint found at {best_ckpt}. Using untrained model.")
        ckpt = None
    else:
        ckpt = torch.load(best_ckpt, map_location=device)

    model = build_model(cfg).to(device)
    if ckpt:
        model.load_state_dict(ckpt["model_state"])
        logger.info(f"Checkpoint loaded from epoch {ckpt.get('epoch', '?')}")

    # Load normaliser
    norm_path = os.path.join(models_dir, "normaliser.npz")
    normaliser = ChannelwiseNorm.load(norm_path) if os.path.exists(norm_path) else None

    # Load composite
    composite_path = os.path.join(cfg["paths"]["processed"], "s2_composite.tif")
    if not os.path.exists(composite_path):
        logger.warning(f"Composite not found at {composite_path}. Creating synthetic for demo.")
        _create_demo_composite(composite_path, cfg)

    image, ref_profile = load_geotiff(composite_path)
    image = np.nan_to_num(image, nan=0.0)
    _, H, W = image.shape

    logger.info(f"Image shape: {image.shape}  ({H}×{W} pixels)")

    # Extract patches
    patch_size = cfg["patches"]["size"]
    stride = cfg["patches"].get("stride", patch_size // 2)
    patches, positions = extract_patches(image, patch_size=patch_size, stride=stride)
    logger.info(f"Extracted {len(patches)} patches (size={patch_size}, stride={stride})")

    if len(patches) == 0:
        logger.error("No patches extracted. Check image and patch settings.")
        return {}

    # Normalise
    if normaliser:
        patches = normaliser.transform(patches)

    # Inference
    logger.info("Running inference…")
    probs = predict_patches_batch(
        model, patches,
        batch_size=cfg["training"]["batch_size"],
        device=device,
    )  # (N, num_classes)

    # Reconstruct probability map
    prob_map = reconstruct_from_patches(probs, positions, (H, W), patch_size)
    # (num_classes, H, W)

    # Class label map
    class_map = prob_map.argmax(axis=0).astype(np.uint8)  # (H, W)
    confidence_map = prob_map.max(axis=0).astype(np.float32)  # (H, W)

    # Save GeoTIFFs
    class_map_path = os.path.join(maps_dir, "crop_class_map.tif")
    confidence_map_path = os.path.join(maps_dir, "confidence_map.tif")

    save_geotiff(class_map[np.newaxis], composite_path, class_map_path, dtype="uint8")
    save_geotiff(confidence_map[np.newaxis], composite_path, confidence_map_path, dtype="float32")

    # Colour visualisation
    vis_path = os.path.join(maps_dir, "crop_map_visualisation.png")
    visualise_class_map(class_map, confidence_map, cfg, ref_profile, save_path=vis_path)

    outputs = {
        "class_map": class_map_path,
        "confidence_map": confidence_map_path,
        "visualisation": vis_path,
    }
    logger.info(f"✓ Map prediction complete. Outputs: {outputs}")
    return outputs


# =============================================================================
# Visualisation
# =============================================================================

def visualise_class_map(
    class_map: np.ndarray,
    confidence_map: Optional[np.ndarray],
    cfg: Dict,
    ref_profile: Dict,
    save_path: Optional[str] = None,
):
    """
    Create a colour-coded crop type map with legend.
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from matplotlib.colors import ListedColormap
    except ImportError:
        logger.warning("matplotlib not installed; skipping visualisation")
        return

    class_names = cfg["classes"]["names"]
    color_map = cfg["classes"]["colors"]
    colors = [color_map.get(n, "#AAAAAA") for n in class_names]
    cmap = ListedColormap(colors)

    ncols = 2 if confidence_map is not None else 1
    fig, axes = plt.subplots(1, ncols, figsize=(6 * ncols, 6))
    if ncols == 1:
        axes = [axes]

    # Class map
    im = axes[0].imshow(class_map, cmap=cmap, vmin=0, vmax=len(class_names) - 1,
                        interpolation="nearest")
    axes[0].set_title("Predicted Crop Types", fontsize=12, fontweight="bold")
    axes[0].axis("off")

    # Legend
    patches = [mpatches.Patch(facecolor=c, label=n) for c, n in zip(colors, class_names)]
    axes[0].legend(handles=patches, loc="lower left", fontsize=8,
                   framealpha=0.85, title="Crop Class")

    # Confidence map
    if confidence_map is not None and ncols > 1:
        im2 = axes[1].imshow(confidence_map, cmap="viridis", vmin=0, vmax=1)
        axes[1].set_title("Model Confidence (max softmax)", fontsize=12, fontweight="bold")
        axes[1].axis("off")
        plt.colorbar(im2, ax=axes[1], shrink=0.8, label="Confidence")

    plt.suptitle(
        f"Crop Type Map — {cfg['study_area']['name']}",
        fontsize=13, fontweight="bold", y=1.01,
    )
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=200, bbox_inches="tight")
        logger.info(f"Crop map visualisation saved: {save_path}")
    plt.show()


# =============================================================================
# Folium interactive map export
# =============================================================================

def export_folium_map(
    class_map_path: str,
    cfg: Dict,
    output_path: Optional[str] = None,
):
    """
    Export crop classification map as an interactive Folium HTML map.
    Overlays the classified raster on a satellite base map.
    """
    try:
        import folium
        from folium import raster_layers
        import rasterio
        import base64
        from io import BytesIO
        from PIL import Image
    except ImportError:
        logger.warning("folium/PIL not installed; skipping interactive map export")
        return

    with rasterio.open(class_map_path) as src:
        class_map = src.read(1)
        bounds = src.bounds
        crs = src.crs

    class_names = cfg["classes"]["names"]
    color_map = cfg["classes"]["colors"]

    # Convert class map to RGBA PNG
    H, W = class_map.shape
    rgba = np.zeros((H, W, 4), dtype=np.uint8)
    hex_to_rgb = lambda h: tuple(int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))

    for cls_id, name in enumerate(class_names):
        color_hex = color_map.get(name, "#AAAAAA")
        r, g, b = hex_to_rgb(color_hex)
        mask = class_map == cls_id
        rgba[mask] = [r, g, b, 180]   # semi-transparent

    img = Image.fromarray(rgba, mode="RGBA")
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")

    # Create Folium map
    centre_lat = (bounds.bottom + bounds.top) / 2
    centre_lon = (bounds.left + bounds.right) / 2
    m = folium.Map(location=[centre_lat, centre_lon], zoom_start=12,
                   tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                   attr="ESRI World Imagery")

    # Overlay raster
    folium.raster_layers.ImageOverlay(
        image=f"data:image/png;base64,{encoded}",
        bounds=[[bounds.bottom, bounds.left], [bounds.top, bounds.right]],
        opacity=0.7,
        name="Crop Type Map",
    ).add_to(m)

    # Legend
    legend_html = '<div style="position: fixed; bottom: 30px; left: 30px; z-index: 9999; background: white; padding: 10px; border-radius: 5px; font-family: Arial; font-size: 12px; box-shadow: 2px 2px 6px rgba(0,0,0,0.3);">'
    legend_html += "<b>Crop Classes</b><br>"
    hex_to_rgb = lambda h: tuple(int(h.lstrip("#")[i:i+2], 16) for i in (0, 2, 4))
    for name in class_names:
        color = color_map.get(name, "#AAAAAA")
        legend_html += f'<span style="background:{color};width:14px;height:14px;display:inline-block;margin-right:5px;border:1px solid #888;"></span>{name}<br>'
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    if output_path is None:
        output_path = os.path.join(cfg["paths"]["maps"], "crop_map_interactive.html")
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    m.save(output_path)
    logger.info(f"Interactive map saved: {output_path}")
    return output_path


# =============================================================================
# Demo composite helper
# =============================================================================

def _create_demo_composite(path: str, cfg: Dict):
    """Create a synthetic composite for demo / testing."""
    import rasterio
    from rasterio.transform import from_bounds
    from rasterio.crs import CRS

    bbox = cfg["study_area"]["bbox"]
    H, W = 256, 256
    n_bands = cfg["patches"]["bands"]
    transform = from_bounds(bbox[1], bbox[0], bbox[3], bbox[2], W, H)
    rng = np.random.default_rng(42)
    data = rng.random((n_bands, H, W)).astype(np.float32)

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    profile = {
        "driver": "GTiff", "dtype": "float32", "count": n_bands,
        "height": H, "width": W, "crs": CRS.from_epsg(4326),
        "transform": transform, "nodata": -9999.0,
    }
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(data)
    logger.info(f"Demo composite created: {path}")


# =============================================================================
# Entry point
# =============================================================================

def run(config_path: str = "configs/config.yaml"):
    cfg = load_config(config_path)
    outputs = predict_full_map(cfg)

    if outputs.get("class_map"):
        export_folium_map(outputs["class_map"], cfg)

    return outputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate full crop type map")
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    run(args.config)
