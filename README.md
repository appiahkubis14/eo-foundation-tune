# 🛰️ Foundation Model Fine-Tuning for Crop Mapping

**Author:** Samuel Appiah Kubi  
**Programme:** Copernicus Master's in Digital Earth (Erasmus Mundus)  
**University:** Paris Lodron University Salzburg

---

## Overview

This project fine-tunes an **Earth Observation Foundation Model** (Clay / ResNet-50) to classify crop types — maize, cocoa, oil palm, fallow, and other — from Sentinel-2 satellite imagery over the Ejura agricultural zone in Ghana.

**Key idea:** Instead of training a deep neural network from scratch (which requires thousands of labelled examples), we start from a model pre-trained on millions of EO images and only teach it to recognise local crop-type patterns. This works with as few as 50–100 labelled field polygons per class.

```
Sentinel-2 Imagery → Cloud Masking → Spectral Indices → Patch Extraction
    → Spatial Split → Fine-tune Clay/ResNet-50 → Evaluate vs Baselines
        → Full Crop Type Map → Interactive Folium Export
```

---

## Repository Structure

```
project1_foundation_model/
├── configs/
│   └── config.yaml              # All settings: study area, model, training
├── scripts/
│   ├── utils.py                 # NDVI/NDWI/EVI, cloud masking, patch extraction,
│   │                            #   metrics, normalisation, spatial split
│   ├── download_s2.py           # Sentinel-2 L2A download via eodag, SCL cloud
│   │                            #   masking, temporal compositing
│   ├── prepare_labels.py        # CropHarvest / KML / synthetic labels →
│   │                            #   rasterised to S2 grid, spatial train/val/test split
│   ├── dataset.py               # PyTorch CropPatchDataset, augmentations,
│   │                            #   WeightedRandomSampler, DataLoader factory
│   ├── model.py                 # Clay EO foundation model + ResNet-50 fallback,
│   │                            #   two-phase freeze strategy, param groups
│   ├── train.py                 # Two-phase training loop, AMP, early stopping,
│   │                            #   checkpoint saving, training curve plots
│   ├── evaluate.py              # Foundation model + RF + U-Net evaluation,
│   │                            #   confusion matrix, comparison table
│   └── predict_map.py           # Full-image sliding-window inference, GeoTIFF
│                                #   export, Folium interactive map
├── notebooks/
│   └── Project1_CropMapping_Colab.ipynb   # Complete self-contained Colab notebook
├── data/
│   ├── raw/                     # Downloaded Sentinel-2 SAFE directories
│   ├── processed/               # Cloud-masked composites, temporal stacks
│   ├── labels/                  # Label rasters + split GeoJSONs
│   └── outputs/
│       ├── models/              # best_model.pth, normaliser.npz
│       ├── maps/                # crop_class_map.tif, confidence_map.tif
│       └── reports/             # Metrics JSON, confusion matrix PNGs, comparison CSV
├── main.py                      # CLI orchestrator
└── requirements.txt
```

---

## Quickstart

### Option A — Google Colab (recommended for students)

1. Open `notebooks/Project1_CropMapping_Colab.ipynb` in Colab
2. Set runtime to **T4 GPU** (Runtime → Change runtime type)
3. Run all cells — synthetic data is generated automatically
4. To use real Sentinel-2: set `USE_REAL_DATA = True` and add your Copernicus credentials

### Option B — Local environment

```bash
# 1. Clone / download this folder
cd project1_foundation_model

# 2. Create virtual environment
python3.11 -m venv .venv && source .venv/bin/activate

# 3. Install PyTorch (GPU)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# 4. Install other dependencies
pip install -r requirements.txt

# 5. Run full pipeline (synthetic data, no credentials needed)
python main.py --step all

# 6. Run with real Sentinel-2 data
export EODAG_USERNAME="your.email@example.com"
export EODAG_PASSWORD="your_password"
python main.py --step all --label_source synthetic
```

---

## Pipeline Steps

Run any individual step with `--step <name>`:

| Step | Command | What it does |
|------|---------|-------------|
| `download` | `python main.py --step download` | Download Sentinel-2 L2A, cloud mask, composite |
| `labels` | `python main.py --step labels --label_source synthetic` | Prepare crop type labels (synthetic / CropHarvest / KML) |
| `train` | `python main.py --step train` | Two-phase fine-tuning |
| `evaluate` | `python main.py --step evaluate` | Metrics + baseline comparison |
| `map` | `python main.py --step map` | Full crop type classification map |
| `all` | `python main.py --step all` | Run everything in order |

---

## Configuration (`configs/config.yaml`)

Key settings to adjust for your project:

```yaml
study_area:
  bbox: [7.15, -1.55, 7.60, -1.10]   # Change to your study area [min_lat, min_lon, max_lat, max_lon]

sentinel2:
  start_date: "2023-03-01"            # Start of growing season
  end_date:   "2023-10-31"            # End of growing season
  cloud_cover_max: 20                 # Max cloud cover % for scene selection

classes:
  names: ["maize", "cocoa", "oil_palm", "fallow", "other"]  # Your crop classes

foundation_model:
  name: "resnet50"                    # Change to "clay" for Clay EO model
  freeze_ratio: 0.85                  # Fraction of backbone to freeze

training:
  epochs: 50
  batch_size: 16                      # Reduce to 8 if out of memory
  learning_rate_head: 1.0e-3
  learning_rate_backbone: 1.0e-5
```

---

## Label Sources

Three ways to get ground truth crop type labels:

### 1. Synthetic (default — for testing)
```bash
python main.py --step labels --label_source synthetic
```
Generates 60 random rectangular field polygons per class. Use this to test the pipeline before getting real labels.

### 2. CropHarvest (global open dataset)
```bash
pip install cropharvest
python main.py --step labels --label_source cropharvest
```
Downloads crop type labels from [NASA Harvest CropHarvest](https://github.com/nasaharvest/cropharvest). Coverage varies by country.

### 3. Your own KML / GeoJSON (recommended for real results)
1. Open **Google Earth Pro** → navigate to Ejura area
2. Draw polygons around known crop fields, name each with the crop type
3. File → Save Place As → KML
4. Run:
```bash
python main.py --step labels --label_source kml --kml_path path/to/fields.kml --label_column Name
```
Aim for **50–100 polygons per crop class**.

---

## Two-Phase Fine-Tuning

| Phase | Epochs | What's trained | Learning rate |
|-------|--------|----------------|---------------|
| **Phase 1** | 0 → 60% of total | Classification head only | 1×10⁻³ (head) |
| **Phase 2** | 60% → end | Head + top 2 backbone layers | 1×10⁻⁵ (backbone), 1×10⁻³ (head) |

**Why two phases?** Abruptly unfreezing the backbone at the start causes catastrophic forgetting. Phase 1 stabilises the head first, then Phase 2 gently adapts the backbone to your local crop patterns.

---

## Model Options

### Clay (EO Foundation Model — recommended)
```yaml
foundation_model:
  name: "clay"
  variant: "small"   # 10M parameters, fits on Colab T4
```
Pre-trained on Sentinel-2, Landsat, SAR, and DEM data globally. Falls back to ResNet-50 automatically if Clay fails to load.

### ResNet-50 (ImageNet — always available)
```yaml
foundation_model:
  name: "resnet50"
```
Not an EO-specific model, but ImageNet features (edges, textures, shapes) still transfer well. Always available without any additional dependencies.

---

## Outputs

```
data/outputs/
├── models/
│   ├── best_model.pth             # Best checkpoint (lowest val loss)
│   ├── last_model.pth             # Last epoch checkpoint
│   └── normaliser.npz             # Channel-wise mean/std for inference
├── maps/
│   ├── crop_class_map.tif         # Full-area classification raster (uint8)
│   ├── confidence_map.tif         # Max softmax probability per pixel (float32)
│   ├── crop_map_visualisation.png # Colour-coded PNG with legend
│   └── crop_map_interactive.html  # Folium interactive map (open in browser)
└── reports/
    ├── metrics_foundation_model.json
    ├── metrics_random_forest.json
    ├── model_comparison.csv        # Side-by-side metrics table
    ├── confusion_matrix_fm.png
    ├── training_curves.png
    └── sample_patches.png
```

---

## Spatial Cross-Validation

This pipeline uses **spatial** (not random) cross-validation to prevent data leakage:

```
Study Area (top view):
┌─────────────┬─────────────┐
│   Fold 0    │   Fold 1    │
│    (NW)     │    (NE)     │
│   Train     │    Val      │
├─────────────┴─────────────┤
│         Fold 2            │
│      (South half)         │
│          Test             │
└───────────────────────────┘
```

Adjacent pixels share similar spectral characteristics, so a random split would inflate accuracy. The spatial split ensures the test area is geographically separated from the training area.

---

## Evaluation Metrics

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | correct / total | Overall % correctly classified |
| **Precision** | TP / (TP + FP) | Of predicted crop X, how many are actually X? |
| **Recall** | TP / (TP + FN) | Of actual crop X pixels, how many detected? |
| **F1** | 2·P·R / (P+R) | Harmonic mean — main metric |
| **IoU** | TP / (TP+FP+FN) | Overlap between prediction and truth |

**Target:** F1 > 0.7 on test set, outperforming both Random Forest and U-Net baselines.

---

## Computing Requirements

| Component | Colab Free (T4) | Local GPU |
|-----------|----------------|-----------|
| Clay small (10M) | ✅ | ✅ |
| ResNet-50 | ✅ | ✅ |
| Patch size 32×32 | ✅ | ✅ |
| Patch size 64×64 | ⚠️ Reduce batch | ✅ |
| Training time | ~45–90 min | ~20–40 min |

If you run out of memory on Colab, reduce:
- `patches.size`: 32 → 16
- `training.batch_size`: 16 → 8

---

## Citation

```bibtex
@mastersthesis{appiah-kubi2024crop,
  author    = {Appiah Kubi, Samuel},
  title     = {Foundation Model Fine-Tuning for Crop Type Mapping
               in Ghana Using Sentinel-2 Imagery},
  school    = {Paris Lodron University Salzburg},
  year      = {2024},
  programme = {Copernicus Master's in Digital Earth (Erasmus Mundus)},
}
```

---

## Key References

- Singh et al. (2023). *Clay: A Foundation Model for Earth Observation.* [GitHub](https://github.com/Clay-foundation/model)
- Jakubik et al. (2023). *Prithvi: Foundation Models for Earth Observation.* NASA/IBM.
- Tseng et al. (2021). *CropHarvest: A Global Dataset for Crop-Type Classification.* [GitHub](https://github.com/nasaharvest/cropharvest)
- Sentinel-2: [Copernicus Data Space](https://dataspace.copernicus.eu/)

---

## Acknowledgements

This work is part of the Copernicus Master's in Digital Earth (Erasmus Mundus) programme. Sentinel-2 data provided by ESA/Copernicus. CropHarvest dataset by NASA Harvest.

**License:** MIT