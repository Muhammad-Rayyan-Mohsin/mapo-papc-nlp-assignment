"""Training entrypoint — DPO + LoRA on MAPO/PAPC preference pairs.

Thin wrapper that loads config.yaml and delegates to the staged pipeline in
src/. Reproducing the full result requires the four data-prep stages first
(translate -> build preferences -> train -> eval); this script runs the
training stage, which assumes data/preferences.jsonl already exists.

Usage:
    python train.py --config config.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from src.model import train_dpo
from src.utils import set_seed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--preferences",
        default="data/preferences.jsonl",
        help="JSONL of preference pairs (prompt/chosen/rejected). See data/sample_data.csv.",
    )
    parser.add_argument("--output-dir", default="checkpoints")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg["seed"])
    train_dpo(
        cfg=cfg,
        preferences_path=Path(args.preferences),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
