from __future__ import annotations

# Gemeinsame Hilfsfunktionen für Reproduzierbarkeit, Dateien und Batch-Verarbeitung.
# 用于可复现性、文件和分批处理的公共工具函数。
# Shared helpers for reproducibility, files, and batch processing.
import json
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np


def set_seed(seed: int) -> None:
    # Setzt alle verwendeten Zufallszahlengeneratoren auf denselben Startwert.
    # 将所有随机数生成器设为同一种子。
    # Sets all random number generators to the same seed.
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except ImportError:
        pass


def save_json(path: Path, value: dict[str, Any]) -> None:
    # Speichert strukturierte Ergebnisse lesbar und mit UTF-8-Kodierung.
    # 使用 UTF-8 以可读格式保存结构化结果。
    # Saves structured results readably using UTF-8.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def load_json(path: Path) -> dict[str, Any]:
    # Lädt eine zuvor gespeicherte JSON-Datei als Python-Wörterbuch.
    # 将之前保存的 JSON 文件加载为 Python 字典。
    # Loads a previously saved JSON file as a Python dictionary.
    return json.loads(path.read_text(encoding="utf-8"))


def batches(length: int, batch_size: int) -> Iterator[tuple[int, int]]:
    # Liefert Start- und Endpositionen für speicherschonende Datenblöcke.
    # 返回节省内存的数据块起止位置。
    # Yields start and end positions for memory-efficient data blocks.
    for start in range(0, length, batch_size):
        yield start, min(start + batch_size, length)


def minmax_rows(values: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    # Skaliert jede Zeile unabhängig auf den Wertebereich von 0 bis 1.
    # 将每一行独立缩放到 0 至 1。
    # Scales each row independently to the range 0 to 1.
    lo = values.min(axis=1, keepdims=True)
    hi = values.max(axis=1, keepdims=True)
    return (values - lo) / np.maximum(hi - lo, eps)


def timestamped_message(message: str) -> None:
    # Gibt Fortschrittsmeldungen mit einer gut lesbaren Uhrzeit aus.
    # 输出带可读时间戳的进度消息。
    # Prints progress messages with a readable timestamp.
    from datetime import datetime

    print(f"[{datetime.now().strftime('%H:%M:%S')}] {message}", flush=True)
