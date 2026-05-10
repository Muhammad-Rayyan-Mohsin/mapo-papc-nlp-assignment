"""Preference-pair dataset loading + PAPC pair-construction helpers.

Two responsibilities:
  1. load_preferences(): read data/preferences.jsonl into the shape expected
     by trl.DPOTrainer (prompt / chosen / rejected).
  2. papc_augment(): given natural pairs + per-language counts, inject
     synthetic chosen for low-resource langs (the v3 contribution).
"""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Iterable

from datasets import Dataset

from .utils import iter_jsonl


def load_preferences(path: Path) -> Dataset:
    """Return a HF Dataset with columns: prompt, chosen, rejected, lang."""
    rows: list[dict] = []
    for r in iter_jsonl(path):
        rows.append({
            "prompt": r["prompt"],
            "chosen": r["chosen"],
            "rejected": r["rejected"],
            "lang": r.get("lang", "unk"),
        })
    if not rows:
        raise ValueError(f"No preference rows in {path}")
    return Dataset.from_list(rows)


def count_pairs_per_lang(records: Iterable[dict]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for r in records:
        if not r.get("synthetic", False):
            counts[r.get("lang", "unk")] += 1
    return dict(counts)


def papc_should_augment(natural_count: int, min_threshold: int) -> bool:
    """Decide whether a language needs synthetic chosen pairs."""
    return natural_count < min_threshold
