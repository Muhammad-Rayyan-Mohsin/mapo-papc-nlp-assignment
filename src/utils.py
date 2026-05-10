"""Shared utilities: seeding, answer extraction, light I/O helpers."""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch


ANSWER_RE = re.compile(r"####\s*(-?\d+(?:\.\d+)?)")
BOXED_RE = re.compile(r"\\boxed\{\s*(-?\d+(?:\.\d+)?)\s*\}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def extract_answer(text: str) -> str | None:
    """Pull the final numeric answer from a generated solution.

    Tries '#### N' first (GSM8K convention), then a boxed{N}. Returns None
    if neither pattern matches.
    """
    if not text:
        return None
    m = ANSWER_RE.search(text)
    if m:
        return m.group(1)
    m = BOXED_RE.search(text)
    if m:
        return m.group(1)
    return None


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
