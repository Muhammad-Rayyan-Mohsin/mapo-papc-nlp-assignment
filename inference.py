"""Inference entrypoint — generate a multilingual math answer with the trained adapter.

Usage:
    python inference.py --problem "What is 12 * 7?" --lang en
    python inference.py --problem "12 * 7 ni ngapi?" --lang sw --adapter checkpoints/mapo-adapter
"""
from __future__ import annotations

import argparse

import yaml

from src.model import load_policy, generate_answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--problem", required=True, help="Math word problem text.")
    parser.add_argument("--lang", default="en", help="ISO code (en/zh/sw/bn).")
    parser.add_argument(
        "--adapter",
        default="checkpoints/mapo-adapter",
        help="LoRA adapter directory. Pass '' to run the base model.",
    )
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    model, tokenizer = load_policy(
        base_model=cfg["models"]["base_model"],
        adapter_path=args.adapter or None,
    )
    answer = generate_answer(
        model=model,
        tokenizer=tokenizer,
        problem=args.problem,
        gen_cfg=cfg["generation"],
    )
    print(answer)


if __name__ == "__main__":
    main()
