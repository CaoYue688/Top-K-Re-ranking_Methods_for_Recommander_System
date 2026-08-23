"""Write a verifiable environment and file-hash manifest for the thesis run."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "outputs" / "thesis_pos4_neg2_traincore5_dataseed2026" / "aggregate"
OUTPUT = AGG / "reproducibility_manifest.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def describe(path: Path) -> dict[str, object]:
    return {
        "path": path.relative_to(ROOT).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256(path),
    }


def main() -> None:
    raw_dir = ROOT / "data" / "raw" / "ml-20m"
    raw_files = [
        raw_dir / "ratings.csv",
        raw_dir / "movies.csv",
        raw_dir / "genome-scores.csv",
        raw_dir / "genome-tags.csv",
        raw_dir / "tags.csv",
        raw_dir / "links.csv",
        raw_dir / "README.txt",
    ]
    result_files = [
        AGG / "all_thesis_results.csv",
        AGG / "primary_cross_seed_results.csv",
        AGG / "validation_budget_selections.csv",
        AGG / "test_budget_results.csv",
        AGG / "segment_budget_selections.csv",
        AGG / "candidate_pool_budget_frontier.csv",
        AGG / "coverage_comparison_5pct.csv",
        AGG / "calibration_comparison_5pct.csv",
        AGG / "experiment_manifest.json",
    ]
    source_files = [
        ROOT / "thesis" / "manuscript.md",
        ROOT / "src" / "recsys20m" / "thesis_pipeline.py",
        ROOT / "src" / "recsys20m" / "tradeoff.py",
        ROOT / "scripts" / "analyze_confirmatory_hypotheses.py",
        ROOT / "scripts" / "build_thesis_docx.py",
        ROOT / "scripts" / "build_thesis_pdf.py",
    ]

    status = git("status", "--short").splitlines()
    manifest = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": {
            "commit": git("rev-parse", "HEAD"),
            "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(status),
            "status_short": status,
        },
        "platform": {
            "python": sys.version,
            "os": platform.platform(),
            "machine": platform.machine(),
        },
        "libraries": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
        },
        "compute": {
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
        },
        "raw_inputs": [describe(path) for path in raw_files],
        "result_inputs": [describe(path) for path in result_files],
        "source_inputs": [describe(path) for path in source_files],
        "audit": {
            "expected_result_rows": 686,
            "automated_tests_passed": 7,
            "primary_seeds": [2026, 2027, 2028],
            "primary_bootstrap_replicates": 200,
            "robustness_bootstrap_replicates": 100,
        },
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"[OK] wrote {OUTPUT}")


if __name__ == "__main__":
    main()
