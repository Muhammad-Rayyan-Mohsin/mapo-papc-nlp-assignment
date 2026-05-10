"""Shared utilities: seeding, math answer extraction, JSONL I/O.

Mirrors mapo-repro/scripts/_utils.py — unicode-digit handling matters here
because Bengali (and similar scripts) use native digits in MGSM gold answers.
"""
from __future__ import annotations

import json
import random
import re
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np
import torch


# Devanagari, Bengali, Thai digit blocks -> ASCII
_DIGIT_MAP = {
    ord(c): str(i)
    for base in (0x0966, 0x09E6, 0x0E50)
    for i, c in enumerate(chr(base + j) for j in range(10))
}


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _normalize(s: str) -> str | None:
    s = s.replace(",", "").rstrip(".")
    try:
        v = float(s)
        return str(int(v)) if v.is_integer() else str(v)
    except ValueError:
        return None


def extract_answer(text: str | None) -> str | None:
    """Pull the final numeric answer from a generated solution.

    Tries '#### N' (GSM8K convention), then 'answer is/=/: N', then \\boxed{N},
    then the last number in the text. Normalizes unicode digits to ASCII first.
    """
    if text is None:
        return None
    text = text.translate(_DIGIT_MAP)
    m = re.search(r"####\s*(-?\d[\d,]*\.?\d*)", text)
    if m:
        return _normalize(m.group(1))
    m = re.search(r"answer\s*(?:is|=|:)\s*\$?\s*(-?\d[\d,]*\.?\d*)", text, re.I)
    if m:
        return _normalize(m.group(1))
    m = re.search(r"\\boxed\{\s*(-?\d[\d,]*\.?\d*)\s*\}", text)
    if m:
        return _normalize(m.group(1))
    nums = re.findall(r"-?\d[\d,]*\.?\d*", text)
    return _normalize(nums[-1]) if nums else None


def answers_match(a: str | None, b: str | None) -> bool:
    if a is None or b is None:
        return False
    try:
        return abs(float(a) - float(b)) < 1e-4
    except ValueError:
        return a == b


def iter_jsonl(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def dump_jsonl(records: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
