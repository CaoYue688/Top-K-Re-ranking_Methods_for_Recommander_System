"""Print aligned German/Chinese manuscript blocks for manual translation review."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import translate_thesis_de_to_zh_full as translation


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=1)
    parser.add_argument("--end", type=int, default=10_000)
    parser.add_argument("--kinds", nargs="*", default=["translated"])
    args = parser.parse_args()

    source = translation.blocks((ROOT / "thesis" / "manuscript.md").read_text(encoding="utf-8"))
    target = translation.blocks((ROOT / "thesis" / "manuscript_zh_full.md").read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in (ROOT / "tmp" / "translation" / "full_translation_alignment.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    if not (len(source) == len(target) == len(records)):
        raise ValueError(f"alignment mismatch: {len(source)}, {len(target)}, {len(records)}")
    sys.stdout.reconfigure(encoding="utf-8")
    for index, (de, zh, record) in enumerate(zip(source, target, records), start=1):
        if not (args.start <= index <= args.end):
            continue
        if args.kinds and record.get("kind") not in args.kinds:
            continue
        print(f"===== BLOCK {index} | {record.get('kind')} =====")
        print("[DE]")
        print(de)
        print("[ZH]")
        print(zh)
        print()


if __name__ == "__main__":
    main()
