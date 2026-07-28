from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def batches(length: int, batch_size: int) -> Iterator[tuple[int, int]]:
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)


def minmax_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    lo = values.min(axis=1, keepdims=True)
    hi = values.max(axis=1, keepdims=True)
    return (values - lo) / np.maximum(hi - lo, eps)


def timestamped_message(message: str) -> None:
    from datetime import datetime

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)

