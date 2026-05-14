"""
main.py — Project 1 Pipeline Orchestrator
==========================================
Author: Samuel Appiah Kubi
Copernicus Master's in Digital Earth, Paris Lodron University Salzburg

Usage
-----
    # Run full pipeline
    python main.py --step all

    # Individual steps
    python main.py --step download
    python main.py --step labels --label_source synthetic
    python main.py --step train
    python main.py --step train --resume
    python main.py --step evaluate
    python main.py --step map
"""

import os
import sys
import argparse
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from scripts.utils import load_config, set_seeds, setup_logging

logger = logging.getLogger("crop_mapping.main")

STEPS = [
    "download",
    "labels",
    "train",
    "evaluate",
    "map",
]


def run_download(cfg, args):
    from scripts.download_s2 import run
    run(args.config, args.username, args.password)


def run_labels(cfg, args):
    from scripts.prepare_labels import run
    run(args.config, args.label_source, args.kml_path, args.label_column)


def run_train(cfg, args):
    from scripts.train import run
    run(args.config, args.resume)


def run_evaluate(cfg, args):
    from scripts.evaluate import run
    run(args.config, args.skip_baselines)


def run_map(cfg, args):
    from scripts.predict_map import run
    run(args.config)


STEP_RUNNERS = {
    "download": run_download,
    "labels": run_labels,
    "train": run_train,
    "evaluate": run_evaluate,
    "map": run_map,
}


def main():
    parser = argparse.ArgumentParser(
        description="Project 1: Foundation Model Fine-Tuning for Crop Mapping",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Steps:
  download   Download and preprocess Sentinel-2 imagery
  labels     Prepare ground truth labels
  train      Fine-tune the foundation model
  evaluate   Evaluate and compare with baselines
  map        Generate full crop type classification map
  all        Run all steps in order

Examples:
  python main.py --step all
  python main.py --step labels --label_source synthetic
  python main.py --step train --resume
  python main.py --step evaluate --skip_baselines
        """,
    )
    parser.add_argument("--step", choices=STEPS + ["all"], default="all",
                        help="Pipeline step to run")
    parser.add_argument("--config", default="configs/config.yaml",
                        help="Path to config YAML")
    # Download args
    parser.add_argument("--username", default=None, help="Copernicus username")
    parser.add_argument("--password", default=None, help="Copernicus password")
    # Label args
    parser.add_argument("--label_source",
                        choices=["synthetic", "cropharvest", "kml", "geojson", "shp"],
                        default="synthetic")
    parser.add_argument("--kml_path", default=None, help="Path to KML/GeoJSON file")
    parser.add_argument("--label_column", default="crop_type")
    # Training args
    parser.add_argument("--resume", action="store_true", help="Resume training from checkpoint")
    # Evaluation args
    parser.add_argument("--skip_baselines", action="store_true")

    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        logger.error(f"Config file not found: {args.config}")
        sys.exit(1)
    cfg = load_config(args.config)
    set_seeds(cfg["training"].get("seed", 42))

    # Run steps
    steps = STEPS if args.step == "all" else [args.step]

    logger.info("=" * 60)
    logger.info("Project 1: Foundation Model Fine-Tuning for Crop Mapping")
    logger.info(f"Study area : {cfg['study_area']['name']}")
    logger.info(f"Model      : {cfg['foundation_model']['name']} ({cfg['foundation_model']['variant']})")
    logger.info(f"Classes    : {cfg['classes']['names']}")
    logger.info("=" * 60)

    for step in steps:
        logger.info(f"\n{'─'*50}")
        logger.info(f"  STEP: {step.upper()}")
        logger.info(f"{'─'*50}")
        STEP_RUNNERS[step](cfg, args)

    logger.info("\n✓ Pipeline complete.")


if __name__ == "__main__":
    main()
