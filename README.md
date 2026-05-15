# Satellite-Based Crop Type Mapping for Sub-Saharan Africa
## Foundation Model Fine-Tuning with Sentinel-2 Imagery

**Author:** Samuel Appiah Kubi  
**Programme:** Copernicus Master's in Digital Earth (Erasmus Mundus)  
**University:** Paris Lodron University Salzburg  
**Study Area:** Ejura Agricultural Zone, Ashanti Region, Ghana

---

## Why This Work Matters

Africa feeds itself — but increasingly, it cannot count what it grows.

Across Sub-Saharan Africa, **smallholder farmers cultivate over 80% of the food consumed on the continent**, yet national agricultural statistics are produced from ground surveys that are expensive, slow, and often inaccurate. When a drought strikes the Ashanti Region or a pest sweeps through the cocoa belt, governments and aid agencies learn about it weeks or months after the fact — when it is already too late to intervene.

**This project addresses that gap directly.** By combining ESA's free Sentinel-2 satellite data with state-of-the-art deep learning, it produces near-real-time, wall-to-wall crop type maps at 10-metre resolution — the kind of spatial detail that tells a district agricultural officer not just "there is farmland here" but "this is cocoa, this is maize, this field has been left fallow."

---

## Significance for ESA and the Copernicus Programme

The Copernicus Land Service is mandated to provide authoritative land cover and land use information globally, yet **crop-type mapping at field scale in tropical Africa remains a critical gap** in the programme's portfolio. This project:

**Demonstrates Sentinel-2 utility for smallholder agriculture** in one of the world's most challenging observation environments — high cloud frequency, fragmented field sizes (often less than 1 hectare), and mixed crop systems that confuse conventional classifiers.

**Validates foundation model fine-tuning** as a scalable approach for regions where labelled training data is scarce — a fundamental bottleneck across Africa where systematic crop surveys do not exist. The approach reduces the labelled data requirement from thousands of samples to fewer than 200 field polygons per class.

**Produces STAC-compliant, Cloud-Optimised GeoTIFF outputs** that plug directly into Copernicus Data Space and GEOSS discovery infrastructure, making results immediately usable by downstream ESA services, national mapping agencies, and international food security organisations.

**Supports ESA's Digital Twin Earth initiative** by providing the crop-type layer that anchors agricultural modelling, carbon stock accounting, and food security forecasting at continental scale. Without knowing what crop occupies each field, Digital Twin Earth cannot simulate agricultural systems — only land surfaces.

**Advances Copernicus EO data uptake in Africa.** ESA's strategy for Africa explicitly targets increasing the use of Copernicus data by African institutions. An open-source, documented pipeline that any government statistician can run with free satellite data is a direct implementation of that strategy.

---

## Significance for Africa

Ghana's three principal food and cash crops — **cocoa, oil palm, and maize** — together account for over 40% of agricultural GDP and employ more than half the rural workforce. Accurate, timely knowledge of what is grown, where, and how that is changing year-on-year is the foundation of decisions that affect tens of millions of lives:

**Food security early warning.** The FAO's Global Information and Early Warning System and WFP's HungerMap rely on crop area estimates to project production. Satellite-derived maps replace slow, costly field surveys with consistent, repeatable measurements available within days of a satellite overpass. Early warning of a 30% maize area reduction in the Brong-Ahafo Region means food aid can be pre-positioned before hunger becomes crisis.

**Climate adaptation policy.** Ghana's Nationally Determined Contributions under the Paris Agreement include a commitment to climate-smart agriculture covering 250,000 hectares by 2030. Governments cannot track whether such targets are being met without knowing what is being grown where. This pipeline provides the baseline measurement and the annual update in a single reproducible workflow.

**Smallholder finance and insurance.** Agricultural insurance is nearly absent across West Africa because insurers cannot verify claims at scale. A 10 m crop type map makes area-yield index insurance possible — premiums and payouts are calculated from satellite observations of actual planted area and crop condition, without a claims adjuster ever visiting the farm. The World Bank estimates that closing the agricultural insurance gap in Africa could unlock USD 65 billion in additional farm investment annually.

**Deforestation and zero-deforestation supply chains.** Cocoa expansion into forest reserves is one of the leading drivers of deforestation in Ghana and Côte d'Ivoire, threatening globally significant biodiversity and carbon stocks. International buyers including Nestlé, Barry Callebaut, and Mondelèz have made zero-deforestation commitments that require farm-level verification. A classification system that reliably distinguishes cocoa from forest edge vegetation enables satellite-based compliance monitoring across hundreds of thousands of smallholder farms that cannot be visited individually.

**National statistics modernisation.** Ghana Statistical Service, in partnership with FAO, is actively integrating EO data into its agricultural census methodology. A validated, open-source pipeline that demonstrates end-to-end reproducibility — from raw satellite download to labelled crop map — is a direct contribution to that process and a template for other national statistical offices across the continent.

---

## What This Project Builds

A complete, production-ready pipeline transforming raw Sentinel-2 imagery into a field-scale crop type map:

```
ESA Sentinel-2 L2A (free, open access)
    |
    |-- Cloud masking via Scene Classification Layer (SCL)
    |-- Spectral index computation (NDVI, NDWI, EVI)
    |-- Temporal median composite (reduces cloud contamination)
    |
    v
Crop Type Ground Truth Labels
    |-- Field survey KML polygons (Google Earth Pro)
    |-- CropHarvest global dataset (NASA Harvest)
    |-- Rasterised to 10 m Sentinel-2 grid
    |-- Spatial train / val / test split (no data leakage)
    |
    v
Foundation Model Fine-Tuning (Clay EO Model / ResNet-50 fallback)
    |-- Phase 1: Train classification head only (backbone frozen)
    |-- Phase 2: Fine-tune backbone + head end-to-end (low LR)
    |-- Class-weighted loss for imbalanced crop distributions
    |-- Mixed-precision training (fits on free Colab GPU)
    |
    v
Evaluation vs Baselines
    |-- Random Forest on spectral features
    |-- U-Net trained from scratch
    |-- Per-class precision, recall, F1, IoU
    |-- Spatial confusion matrix
    |
    v
Outputs
    |-- 10 m crop type GeoTIFF (COG, STAC-catalogued)
    |-- Confidence layer (max softmax probability per pixel)
    |-- Interactive Folium map (browser-viewable, no server)
    |-- PDF / HTML accuracy report
```

**Crop classes:** Cocoa · Oil Palm · Maize · Fallow · Other vegetation  
**Input:** Sentinel-2 L2A — free via Copernicus Data Space, no commercial licence needed  
**Output resolution:** 10 metres (Sentinel-2 native B04, B08 bands)

---

## Repository Structure

```
project1_foundation_model/
├── configs/
│   └── config.yaml              # Study area bbox, model choice, training params
├── scripts/
│   ├── utils.py                 # Spectral indices, SCL cloud masking, patch
│   │                            #   extraction, normalisation, spatial split, metrics
│   ├── download_s2.py           # Sentinel-2 L2A via eodag (Copernicus Data Space),
│   │                            #   SCL cloud masking, temporal median composite
│   ├── prepare_labels.py        # KML / CropHarvest / synthetic labels →
│   │                            #   rasterised to S2 grid, spatial train/val/test split
│   ├── dataset.py               # PyTorch Dataset with augmentation,
│   │                            #   WeightedRandomSampler for class imbalance
│   ├── model.py                 # Clay EO foundation model + ResNet-50 fallback,
│   │                            #   two-phase fine-tuning, differential learning rates
│   ├── train.py                 # Training loop, AMP, early stopping, checkpointing
│   ├── evaluate.py              # Foundation model + RF + U-Net baselines,
│   │                            #   confusion matrix, comparison table
│   └── predict_map.py           # Sliding-window inference, GeoTIFF + Folium export
├── notebooks/
│   └── Project1_CropMapping_Colab.ipynb   # Self-contained notebook, Colab free tier
├── data/
│   ├── raw/                     # Sentinel-2 SAFE directories
│   ├── processed/               # Cloud-masked composites, temporal stacks
│   ├── labels/                  # Label rasters, split GeoJSONs
│   └── outputs/
│       ├── models/              # best_model.pth, normaliser.npz
│       ├── maps/                # crop_class_map.tif, confidence_map.tif, HTML map
│       └── reports/             # Metrics JSON, confusion matrices, comparison CSV
├── main.py                      # CLI orchestrator
└── requirements.txt
```

---

## Quickstart

### Option A — Google Colab (recommended, no local setup)

1. Open `notebooks/Project1_CropMapping_Colab.ipynb` in Google Colab
2. Runtime → Change runtime type → **T4 GPU**
3. Run all cells — synthetic data is generated automatically; no credentials needed
4. For real Sentinel-2: set `USE_REAL_DATA = True` and enter Copernicus credentials

### Option B — Local machine

```bash
cd project1_foundation_model

# Create environment
python3.11 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

# Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt

# Run with synthetic data (instant, no credentials)
python main.py --step all

# Run with real Sentinel-2 (register free at dataspace.copernicus.eu)
set EODAG_USERNAME=your.email@example.com
set EODAG_PASSWORD=your_password
python main.py --step all
```

---

## Pipeline Steps

| Step | Command | Output |
|------|---------|--------|
| `download` | `python main.py --step download` | Cloud-masked S2 composite GeoTIFF |
| `labels` | `python main.py --step labels --label_source kml` | Label raster + spatial split GeoJSONs |
| `train` | `python main.py --step train` | `best_model.pth` + training curves |
| `evaluate` | `python main.py --step evaluate` | Per-class metrics, baseline comparison |
| `map` | `python main.py --step map` | 10 m crop type map + interactive HTML |
| `all` | `python main.py --step all` | Complete pipeline |

---

## Ground Truth Labels — Practical Guide

Collecting good labels is the most important field task in this project. Three sources are supported:

### 1. Your own field survey (best accuracy)
Visit representative fields, record GPS coordinates and crop type, then draw boundaries in Google Earth Pro. Export as KML and run:
```bash
python main.py --step labels --label_source kml \
    --kml_path data/labels/field_survey.kml --label_column CropType
```
Target: 50–100 polygons per class. Prioritise variety — different field sizes, soil types, and growing stages — over sheer quantity.

### 2. CropHarvest global dataset (no fieldwork)
```bash
pip install cropharvest
python main.py --step labels --label_source cropharvest
```
Open dataset from NASA Harvest covering crop types in multiple African countries. Supplement with local labels for best results in Ghana.

### 3. Synthetic labels (pipeline testing only)
```bash
python main.py --step labels --label_source synthetic
```
Randomly generated field polygons. Use this to verify the pipeline runs correctly before investing in fieldwork.

---

## Why Foundation Models Solve the Label Scarcity Problem

The core challenge across Africa is not satellite data — Sentinel-2 covers Ghana every five days for free. The challenge is **ground truth**. Training a conventional deep learning model from scratch requires thousands of labelled field samples. Collecting those samples across a country-scale study area costs more than most research budgets allow.

Foundation models reverse this equation. A model pre-trained on millions of global EO images already understands spectral patterns, vegetation textures, field boundaries, and seasonal rhythms. Fine-tuning teaches it only the local vocabulary — "in this part of Ghana, cocoa has this texture, maize has this NDVI trajectory" — which requires far fewer examples:

| Approach | Labels required | Ghana applicability |
|----------|----------------|---------------------|
| Random Forest | 500+ labelled pixels | Feasible but low accuracy |
| U-Net from scratch | 5,000+ labelled pixels | Difficult in smallholder systems |
| **Foundation model fine-tuned** | **50–200 field polygons/class** | **Practical with one field campaign** |

The **two-phase fine-tuning** strategy used here protects the model's general EO knowledge while adapting it to local conditions:

| Phase | Epochs | What trains | Learning rate |
|-------|--------|-------------|---------------|
| Phase 1 | 0 → 60% | New classification head only | 1e-3 (head) |
| Phase 2 | 60% → end | Head + top backbone layers | 1e-5 (backbone), 1e-3 (head) |

---

## Spatial Cross-Validation — Why It Matters

Standard random train/test splits are not valid for spatial data. Neighbouring pixels share nearly identical spectral signatures — a random split leaks spatial autocorrelation into the test set and produces falsely optimistic accuracy scores. The same problem undermines published accuracy numbers for many African crop mapping studies.

This pipeline enforces **geographic hold-out validation**, the same approach used by ESA's Copernicus Global Land Service:

```
+-------------------+-------------------+
|                   |                   |
|   TRAINING set    |  VALIDATION set   |
|   (NW quadrant)   |  (NE quadrant)    |
|                   |                   |
+-------------------+-------------------+
|                                       |
|         TEST set (South half)         |
|  (never seen during training or       |
|   hyperparameter selection)           |
+---------------------------------------+
```

An F1 score computed this way is a genuine estimate of how the model will perform when deployed over a new area — the operationally relevant metric.

---

## Outputs and Standards Compliance

| Output | Format | Applicable standard |
|--------|--------|---------------------|
| Crop type raster | Cloud-Optimised GeoTIFF | OGC, Copernicus Land Service |
| Confidence layer | Float32 GeoTIFF | OGC |
| STAC catalogue | STAC 1.0 Item + root Catalog | STAC spec, GEOSS |
| Interactive map | Self-contained HTML (Folium) | No server required |

---

## Performance Targets

| Metric | Target | Typical RF baseline |
|--------|--------|--------------------|
| Overall Accuracy | > 80% | 65–70% |
| F1 Score (macro) | > 0.70 | 0.55–0.65 |
| IoU (macro) | > 0.55 | 0.40–0.55 |

Achieving F1 > 0.70 with fewer than 200 labelled training fields per class validates the foundation model approach as a practical, deployable solution for national crop mapping programmes across Africa.

---

## Computing Requirements

| Scenario | Colab Free (T4) | Local GPU |
|----------|----------------|-----------|
| Clay small (10 M params) | Works (12 GB VRAM) | Works |
| ResNet-50 fallback | Works | Works |
| Training 50 epochs | 45–90 min | 20–40 min |
| Full-area map generation | ~10 min | ~5 min |

If out of memory on Colab: reduce `patches.size` 32 → 16 and `training.batch_size` 16 → 8.

---

## Citation

```bibtex
@mastersthesis{appiah-kubi2024crop,
  author    = {Appiah Kubi, Samuel},
  title     = {Foundation Model Fine-Tuning for Crop Type Mapping in Ghana:
               A Scalable Approach for Sub-Saharan Africa Using Sentinel-2},
  school    = {Paris Lodron University Salzburg},
  year      = {2024},
  programme = {Copernicus Master's in Digital Earth (Erasmus Mundus)},
  note      = {Study area: Ejura, Ashanti Region, Ghana},
}
```

---

## Key References

- Singh et al. (2023). *Clay: A Foundation Model for Earth Observation.* [github.com/Clay-foundation/model](https://github.com/Clay-foundation/model)
- Jakubik et al. (2023). *Prithvi: Foundation Models for Earth Observation.* NASA / IBM Research.
- Tseng et al. (2021). *CropHarvest: A Global Dataset for Crop-Type Classification.* NeurIPS Datasets Track.
- Kerner et al. (2020). *Rapid Response Crop Maps in Data Sparse Regions.* KDD Humanitarian Mapping Workshop.
- ESA Copernicus Land Service: [land.copernicus.eu](https://land.copernicus.eu)
- Sentinel-2 free data: [dataspace.copernicus.eu](https://dataspace.copernicus.eu)

---

## Acknowledgements

This work is conducted as part of the **Copernicus Master's in Digital Earth** (Erasmus Mundus Joint Master's Degree), co-funded by the European Union. Sentinel-2 data are provided free of charge by the European Space Agency under the Copernicus open data policy.

The methods and code are released under the MIT licence and are intended for free use by national statistical offices, agricultural ministries, research institutions, and development organisations across Africa.

---

*"The Copernicus satellites observe every field in Africa every five days. The limiting factor is no longer the data — it is the computational pipeline to turn those observations into decisions. This project builds that pipeline."*