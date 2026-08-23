from __future__ import annotations

# Dieses Modul orchestriert die vollständigen Accuracy-Diversity-Paper-Experimente.
# 本模块编排完整的准确率-多样性论文实验。
# This module orchestrates the complete accuracy-diversity paper experiments.
import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import evaluate_sampled
from .models import TrainingConfig, train
from .preprocess import PreprocessConfig, preprocess
from .reporting import write_experiment_summary, write_tradeoff_svgs
from .retrieval import retrieve_top_k
from .tradeoff import run_tradeoff
from .utils import load_json, save_json, timestamped_message


@dataclass(frozen=True)
class ResearchConfig:
    # Reproduzierbare Konfiguration für Daten, Modelle und statistische Auswertung.
    # 用于数据、模型和统计评估的可复现配置。
    # Reproducible configuration for data, models, and statistical evaluation.
    root: Path
    positive_threshold: float = 4.0
    negative_threshold: float = 2.0
    min_interactions: int = 5
    embedding_dim: int = 64
    epochs: int = 3
    steps_per_epoch: int | None = None
    batch_size: int = 4096
    explicit_negative_ratio: float = 0.5
    lambda_step: float = 0.05
    bootstrap_samples: int = 200
    max_accuracy_loss: float = 0.05
    max_rows: int | None = None
    methods: tuple[str, ...] = (
        "diversity",
        "mmr",
        "calibration",
        "combined",
    )
    data_seed: int = 2026
    seed: int = 2026
    device: str = "cpu"
    force: bool = False


def _select_operating_points(
    results: pd.DataFrame,
    max_accuracy_loss: float,
) -> dict[str, object]:
    # Wählt auf Validation den diversesten Punkt mit höchstens x Prozent NDCG-Verlust.
    # 在验证集上选择 NDCG 损失不超过 x% 的最多样点。
    # Selects the most diverse validation point with at most x percent NDCG loss.
    selections: dict[str, object] = {"by_method": {}}
    for model in ("mf",):
        validation = results[
            (results["model"] == model)
            & (results["split"] == "val")
        ].copy()
        if validation.empty:
            continue
        baseline = validation[validation["lambda"] == validation["lambda"].min()].sort_values(
            "ndcg@20", ascending=False
        ).iloc[0]
        accuracy_floor = float(baseline["ndcg@20"]) * (1.0 - max_accuracy_loss)
        feasible = validation[validation["ndcg@20"] >= accuracy_floor]
        selected = feasible.sort_values(
            ["ild@20", "ndcg@20"], ascending=False
        ).iloc[0]
        test_match = results[
            (results["model"] == model)
            & (results["split"] == "test")
            & (results["method"] == selected["method"])
            & np.isclose(results["lambda"], selected["lambda"])
        ]
        selections[model] = {
            "selection_rule": (
                f"maximize validation ILD@20 across methods with <= "
                f"{max_accuracy_loss:.1%} validation NDCG@20 loss"
            ),
            "selected_method": str(selected["method"]),
            "selected_lambda": float(selected["lambda"]),
            "validation": selected.to_dict(),
            "test": test_match.iloc[0].to_dict() if not test_match.empty else None,
        }
        method_selections: dict[str, object] = {}
        for method, method_validation in validation.groupby("method"):
            method_baseline = method_validation.loc[
                method_validation["lambda"].idxmin()
            ]
            method_floor = float(method_baseline["ndcg@20"]) * (
                1.0 - max_accuracy_loss
            )
            method_selected = method_validation[
                method_validation["ndcg@20"] >= method_floor
            ].sort_values(["ild@20", "ndcg@20"], ascending=False).iloc[0]
            method_test = results[
                (results["model"] == model)
                & (results["split"] == "test")
                & (results["method"] == method)
                & np.isclose(results["lambda"], method_selected["lambda"])
            ]
            method_selections[str(method)] = {
                "selected_lambda": float(method_selected["lambda"]),
                "validation": method_selected.to_dict(),
                "test": (
                    method_test.iloc[0].to_dict()
                    if not method_test.empty
                    else None
                ),
            }
        selections["by_method"][model] = method_selections
    return selections


def _mark_global_pareto(results: pd.DataFrame) -> pd.DataFrame:
    # Markiert pro Modell und Split die Pareto-Front über alle Re-Ranking-Methoden.
    # 对每个模型和分割，标记跨所有重排方法的 Pareto 前沿。
    # Marks the Pareto frontier across all re-ranking methods per model and split.
    results = results.copy()
    results["global_pareto"] = False
    for (_, _), indices in results.groupby(["model", "split"]).groups.items():
        group = results.loc[indices]
        for index, row in group.iterrows():
            dominated = (
                (group["ndcg@20"] >= row["ndcg@20"])
                & (group["ild@20"] >= row["ild@20"])
                & (
                    (group["ndcg@20"] > row["ndcg@20"])
                    | (group["ild@20"] > row["ild@20"])
                )
            ).any()
            results.loc[index, "global_pareto"] = not bool(dominated)
    return results


def _validate_research_results(
    results: pd.DataFrame,
    selections: dict[str, object],
    methods: tuple[str, ...],
    lambdas: np.ndarray,
    max_accuracy_loss: float,
    output_dir: Path,
) -> dict[str, object]:
    # Prüft Vollständigkeit, eindeutige Konfigurationen und die Validation-Auswahlregel.
    # 检查完整性、唯一配置和验证选择规则。
    # Checks completeness, unique configurations, and the validation selection rule.
    key_columns = ["model", "split", "method", "lambda"]
    expected_rows = 2 * len(methods) * len(lambdas)
    group_sizes = results.groupby(["model", "split", "method"]).size()
    lambda_zero = results[np.isclose(results["lambda"], 0.0)]
    baseline_consistent = True
    for _, group in lambda_zero.groupby(["model", "split"]):
        for metric in ("ndcg@20", "recall@20", "ild@20", "calibration@20"):
            if not np.allclose(group[metric], group.iloc[0][metric]):
                baseline_consistent = False

    selection_within_budget = True
    for model in ("mf",):
        selection = selections.get(model)
        if not isinstance(selection, dict):
            selection_within_budget = False
            continue
        validation = results[
            (results["model"] == model) & (results["split"] == "val")
        ]
        baseline = validation[np.isclose(validation["lambda"], 0.0)][
            "ndcg@20"
        ].max()
        selected_ndcg = float(selection["validation"]["ndcg@20"])
        selection_within_budget &= bool(
            selected_ndcg >= baseline * (1.0 - max_accuracy_loss) - 1e-12
        )

    checks = {
        "expected_rows": expected_rows,
        "actual_rows": len(results),
        "row_count_ok": len(results) == expected_rows,
        "duplicate_configurations": int(results.duplicated(key_columns).sum()),
        "all_groups_have_every_lambda": bool(
            (group_sizes == len(lambdas)).all()
        ),
        "core_metric_nan_count": int(
            results[
                ["ndcg@20", "recall@20", "ild@20", "calibration@20"]
            ].isna().sum().sum()
        ),
        "lambda_zero_consistent_across_methods": baseline_consistent,
        "selection_within_validation_budget": selection_within_budget,
        "svg_count": len(list(output_dir.glob("*_accuracy_diversity.svg"))),
    }
    boolean_checks = [
        value for value in checks.values() if isinstance(value, bool)
    ]
    report = {
        "ok": (
            all(boolean_checks)
            and checks["duplicate_configurations"] == 0
            and checks["core_metric_nan_count"] == 0
            and checks["svg_count"] == 1
        ),
        "checks": checks,
    }
    save_json(output_dir / "research_validation_report.json", report)
    return report


def run_research(config: ResearchConfig) -> Path:
    # Verwendet eigene Unterordner, damit bestehende Baseline-Ergebnisse erhalten bleiben.
    # 使用独立子目录，以保留现有基线结果。
    # Uses separate subdirectories to preserve existing baseline results.
    if not 0.0 < config.lambda_step <= 1.0:
        raise ValueError("lambda_step must be in the interval (0, 1].")
    if config.bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1.")
    if not 0.0 <= config.max_accuracy_loss <= 1.0:
        raise ValueError("max_accuracy_loss must be between 0 and 1.")
    if config.negative_threshold >= config.positive_threshold:
        raise ValueError("negative_threshold must be lower than positive_threshold.")
    if not 0.0 <= config.explicit_negative_ratio <= 1.0:
        raise ValueError("explicit_negative_ratio must be between 0 and 1.")
    positive_tag = f"{config.positive_threshold:g}".replace(".", "p")
    negative_tag = f"{config.negative_threshold:g}".replace(".", "p")
    ratio_tag = f"{config.explicit_negative_ratio:g}".replace(".", "p")
    row_tag = f"_rows{config.max_rows}" if config.max_rows is not None else ""
    dataset_tag = (
        f"research_pos{positive_tag}_neg{negative_tag}_core{config.min_interactions}"
        f"_dataseed{config.data_seed}{row_tag}"
    )
    step_tag = config.steps_per_epoch if config.steps_per_epoch is not None else "all"
    run_tag = (
        f"seed{config.seed}_e{config.epochs}_steps{step_tag}"
        f"_d{config.embedding_dim}_enr{ratio_tag}"
    )
    raw_dir = config.root / "data" / "raw"
    processed_dir = config.root / "data" / "processed" / dataset_tag
    artifact_dir = config.root / "artifacts" / dataset_tag / run_tag
    output_dir = config.root / "outputs" / dataset_tag / run_tag
    stats_path = processed_dir / "stats.json"
    scheme_b_files = (stats_path, processed_dir / "train_explicit_negatives.npz")
    if config.force or not all(path.exists() for path in scheme_b_files):
        preprocess(
            PreprocessConfig(
                raw_dir=raw_dir,
                output_dir=processed_dir,
                min_interactions=config.min_interactions,
                positive_threshold=config.positive_threshold,
                negative_threshold=config.negative_threshold,
                seed=config.data_seed,
                max_rows=config.max_rows,
            )
        )
    else:
        timestamped_message("Using existing rating-threshold research dataset")

    lambdas = np.arange(
        0.0, 1.0 + config.lambda_step / 2.0, config.lambda_step
    )
    result_paths: list[Path] = []
    for model, model_batch_size in (("mf", max(config.batch_size, 8192)),):
        embeddings = artifact_dir / f"{model}_embeddings.npz"
        if config.force or not embeddings.exists():
            train(
                TrainingConfig(
                    processed_dir=processed_dir,
                    artifact_dir=artifact_dir,
                    embedding_dim=config.embedding_dim,
                    batch_size=model_batch_size,
                    epochs=config.epochs,
                    steps_per_epoch=config.steps_per_epoch,
                    explicit_negative_ratio=config.explicit_negative_ratio,
                    seed=config.seed,
                    device=config.device,
                )
            )
        else:
            timestamped_message(f"Using existing {model} research embeddings")

        for split, seen_file in (
            ("val", "train_seen.npz"),
            ("test", "train_val_seen.npz"),
        ):
            # Die 1+100-Auswertung bleibt als ergänzende Vergleichszahl erhalten.
            # 保留 1+100 评估作为补充比较数字。
            # Keeps the 1+100 evaluation as a supplementary comparison.
            evaluate_sampled(processed_dir, artifact_dir, model, split)
            candidates_path = output_dir / f"{model}_{split}_top100.npz"
            if config.force or not candidates_path.exists():
                retrieve_top_k(
                    processed_dir,
                    embeddings,
                    candidates_path,
                    k=100,
                    seen_file=seen_file,
                )
            for method in config.methods:
                result_path = (
                    output_dir / f"{model}_{split}_{method}_tradeoff.csv"
                )
                if config.force or not result_path.exists():
                    result_path = run_tradeoff(
                        model,
                        split,  # type: ignore[arg-type]
                        method,  # type: ignore[arg-type]
                        processed_dir,
                        candidates_path,
                        output_dir,
                        lambdas,
                        top_k=20,
                        bootstrap_samples=config.bootstrap_samples,
                        seed=config.seed,
                    )
                result_paths.append(result_path)

    results = pd.concat(
        (pd.read_csv(path) for path in result_paths), ignore_index=True
    )
    results = _mark_global_pareto(results)
    combined_path = output_dir / "all_tradeoff_results.csv"
    results.to_csv(combined_path, index=False)
    selections = _select_operating_points(results, config.max_accuracy_loss)
    save_json(output_dir / "selected_operating_points.json", selections)
    stats = load_json(stats_path)
    write_tradeoff_svgs(results, output_dir, cutoff=20, split="test")
    write_experiment_summary(
        results,
        selections,
        stats,
        output_dir / "experiment_summary.md",
    )
    validation_report = _validate_research_results(
        results,
        selections,
        config.methods,
        lambdas,
        config.max_accuracy_loss,
        output_dir,
    )
    if not validation_report["ok"]:
        raise RuntimeError("Research output validation failed.")
    save_json(
        output_dir / "experiment_config.json",
        {
            "positive_threshold": config.positive_threshold,
            "negative_threshold": config.negative_threshold,
            "feedback_scheme": "three_level_explicit_negative",
            "min_interactions": config.min_interactions,
            "embedding_dim": config.embedding_dim,
            "epochs": config.epochs,
            "steps_per_epoch": config.steps_per_epoch,
            "batch_size": config.batch_size,
            "explicit_negative_ratio": config.explicit_negative_ratio,
            "lambda_step": config.lambda_step,
            "bootstrap_samples": config.bootstrap_samples,
            "max_accuracy_loss": config.max_accuracy_loss,
            "max_rows": config.max_rows,
            "methods": list(config.methods),
            "data_seed": config.data_seed,
            "seed": config.seed,
            "device": config.device,
        },
    )
    timestamped_message(f"Research experiment complete: {combined_path}")
    return combined_path


def main() -> None:
    # Kommandozeile für schnelle Pilotläufe und vollständige Paper-Experimente.
    # 用于快速试运行和完整论文实验的命令行入口。
    # Command line for quick pilot runs and complete paper experiments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--positive-threshold", type=float, default=4.0)
    parser.add_argument("--negative-threshold", type=float, default=2.0)
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--explicit-negative-ratio", type=float, default=0.5)
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--max-accuracy-loss", type=float, default=0.05)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--methods",
        default="diversity,mmr,calibration,combined",
        help="Comma-separated subset of diversity,mmr,calibration,combined.",
    )
    parser.add_argument("--data-seed", type=int, default=2026)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    methods = tuple(
        method.strip() for method in args.methods.split(",") if method.strip()
    )
    allowed = {"diversity", "mmr", "calibration", "combined"}
    if not methods or not set(methods).issubset(allowed):
        parser.error(f"--methods must be a subset of {sorted(allowed)}")
    run_research(
        ResearchConfig(
            root=args.root.resolve(),
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold,
            min_interactions=args.min_interactions,
            embedding_dim=args.embedding_dim,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            batch_size=args.batch_size,
            explicit_negative_ratio=args.explicit_negative_ratio,
            lambda_step=args.lambda_step,
            bootstrap_samples=args.bootstrap_samples,
            max_accuracy_loss=args.max_accuracy_loss,
            max_rows=args.max_rows,
            methods=methods,
            data_seed=args.data_seed,
            seed=args.seed,
            device=args.device,
            force=args.force,
        )
    )


if __name__ == "__main__":
    # Startet den Paper-Workflow bei direktem Modulaufruf.
    # 在模块被直接调用时启动论文实验流程。
    # Starts the paper workflow when the module is invoked directly.
    main()
