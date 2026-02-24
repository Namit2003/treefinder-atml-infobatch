#!/usr/bin/env python3
"""
main.py: Training and evaluation pipeline for dead tree dataset.

This script supports flexible train-test splits (random, geographic, state-based) and multiple model backbones (CNNs, ViTs, SegFormer, etc.).
Usage:
    python main.py --config configs/debug.yaml --exp_name exp1

Recommendation: use a YAML config (e.g., configs.yaml) to centralize parameters and ensure reproducibility, following common ML conventions.
"""


import yaml
import logging
from pathlib import Path
import time

from utils.tools import (
    parse_args, load_config, overwrite_config,
    setup_logging, set_seed, save_results
)
from data_loader import get_dataloader
from models import get_model
from exps.train import train_model
from exps.evaluate import evaluate_model


def main():    
    # Parse command-line args and load config
    args = parse_args()
    cfg = load_config(args.config)
    if args.overwrite_cfg:
        cfg = overwrite_config(args, cfg)
    
    exp_name = cfg['experiment']['id'].zfill(3)
    exp_name = f"exp{exp_name}_{cfg['model']['name']}"

    # Log experiment configuration
    logger = setup_logging(cfg['logging']['log_dir'], exp_name)
    logger.info(f"Starting experiment: {exp_name}")
    logger.info("Configuration:\n" + yaml.dump(cfg, sort_keys=False))
    
    # Set random seed for reproducibility
    set_seed(cfg['experiment']['seed'])

    # Prepare data loaders — get_dataloader returns (train_loader, val_loader, test_loader, train_dataset)
    train_loader, val_loader, test_loader, train_dataset = get_dataloader(cfg)
    
    # Initialize model
    model = get_model(cfg['model'])

    # Train or skip training
    if args.eval_only:
        logger.info("Evaluation only mode: Skipping training.")
        train_metrics = {}
        train_time = 0.0
    else:
        t0 = time.time()
        train_metrics = train_model(
            model,
            train_loader,
            val_loader,
            train_dataset,
            cfg,
            exp_name
        )
        train_time = time.time() - t0
        logger.info(f"Training completed in {train_time/3600:.2f}h ({train_time/60:.1f}m).")

    # Evaluate model
    t0 = time.time()
    eval_metrics = evaluate_model(
        model,
        test_loader,
        cfg,
        exp_name
    )
    eval_time = time.time() - t0
    logger.info(f"Evaluation completed in {eval_time/3600:.2f}h ({eval_time/60:.1f}m).")

    # Combine and save metrics
    results_root = Path(cfg['output']['results_dir']) / exp_name
    all_metrics = {
        **train_metrics,
        **eval_metrics,
        'train_time_h': train_time,
        'eval_time_h': eval_time,
    }
    save_results(all_metrics, exp_name, str(results_root))
    logger.info(f"Experiment {exp_name} completed. Combined metrics: {all_metrics}")


if __name__ == "__main__":
    main()
