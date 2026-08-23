from __future__ import annotations

"""Run the experiment matrix promised in the FH Wedel thesis exposé."""

import argparse
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import evaluate_sampled
from .features import build_tag_genome_features, tag_complete_candidate_pool
from .models import TrainingConfig, train
from .preprocess import PreprocessConfig, preprocess
from .reporting import write_tradeoff_svgs
from .retrieval import retrieve_top_k
from .tradeoff import run_tradeoff
from .utils import load_json, save_json, timestamped_message


@dataclass(frozen=True)
class ThesisConfig:
    root: Path
    seeds: tuple[int, ...] = (2026, 2027, 2028)
    data_seed: int = 2026
    epochs: int = 10
    embedding_dim: int = 64
    batch_size: int = 8192
    explicit_negative_ratio: float = 0.5
    primary_lambda_step: float = 0.05
    robustness_lambda_step: float = 0.10
    primary_bootstrap_samples: int = 200
    robustness_bootstrap_samples: int = 100
    device: str = "cuda"
    force: bool = False


METHODS = ("mmr", "xquad", "calibration")
BUDGETS = (0.01, 0.03, 0.05, 0.10)
CONSTRUCT_TARGETS = {
    "mmr": "ild@10",
    "xquad": "subtopic_recall@10",
    "calibration": "calibration@10",
}


def _lambdas(step: float) -> np.ndarray:
    return np.arange(0.0, 1.0 + step / 2.0, step, dtype=np.float32)


def _holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=np.float64)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[int(index)])
        running = max(running, value)
        adjusted[int(index)] = running
    return adjusted.tolist()


def _select_feasible(
    frame: pd.DataFrame,
    *,
    baseline_ndcg: float,
    budget: float,
    target_metric: str,
) -> pd.Series:
    """Select a validation operating point without inspecting test outcomes."""
    feasible = frame[
        frame["ndcg@10"] >= baseline_ndcg * (1.0 - budget)
    ]
    if feasible.empty:
        raise ValueError(
            f"No feasible point for budget={budget:.3f}, target={target_metric}."
        )
    return feasible.sort_values(
        [target_metric, "ndcg@10", "lambda"],
        ascending=[False, False, True],
    ).iloc[0]


def _budget_selections(primary: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create confirmatory ILD and construct-aligned validation selections.

    Holm adjustment is applied only to the four confirmatory, across-method
    ILD operating points (one family across the pre-specified accuracy
    budgets).  Method-specific and construct-aligned rows are explicitly
    exploratory and retain their unadjusted descriptive p-values.
    """
    validation = primary[primary["split"] == "val"]
    test = primary[primary["split"] == "test"]
    group_keys = ["method", "lambda"]
    validation_mean = validation.groupby(group_keys, as_index=False).mean(
        numeric_only=True
    )
    baseline_ndcg = float(
        validation_mean[np.isclose(validation_mean["lambda"], 0.0)]["ndcg@10"].max()
    )
    selections: list[dict[str, object]] = []
    for budget in BUDGETS:
        selected = _select_feasible(
            validation_mean,
            baseline_ndcg=baseline_ndcg,
            budget=budget,
            target_metric="ild@10",
        )
        selections.append(
            {
                "budget": budget,
                "scope": "primary_ild_across_methods",
                "method": selected["method"],
                "lambda": float(selected["lambda"]),
                "target_metric": "ild@10",
                "validation_target_value": float(selected["ild@10"]),
                "validation_ndcg@10": float(selected["ndcg@10"]),
                "validation_ild@10": float(selected["ild@10"]),
            }
        )
        for method, method_frame in validation_mean.groupby("method"):
            method_baseline = float(
                method_frame[np.isclose(method_frame["lambda"], 0.0)]["ndcg@10"].max()
            )
            method_selected = _select_feasible(
                method_frame,
                baseline_ndcg=method_baseline,
                budget=budget,
                target_metric="ild@10",
            )
            selections.append(
                {
                    "budget": budget,
                    "scope": "within_method_ild",
                    "method": method,
                    "lambda": float(method_selected["lambda"]),
                    "target_metric": "ild@10",
                    "validation_target_value": float(method_selected["ild@10"]),
                    "validation_ndcg@10": float(method_selected["ndcg@10"]),
                    "validation_ild@10": float(method_selected["ild@10"]),
                }
            )
            target_metric = CONSTRUCT_TARGETS[str(method)]
            construct_selected = _select_feasible(
                method_frame,
                baseline_ndcg=method_baseline,
                budget=budget,
                target_metric=target_metric,
            )
            selections.append(
                {
                    "budget": budget,
                    "scope": "construct_aligned",
                    "method": method,
                    "lambda": float(construct_selected["lambda"]),
                    "target_metric": target_metric,
                    "validation_target_value": float(
                        construct_selected[target_metric]
                    ),
                    "validation_ndcg@10": float(construct_selected["ndcg@10"]),
                    "validation_ild@10": float(construct_selected["ild@10"]),
                }
            )

    tests: list[dict[str, object]] = []
    for choice in selections:
        method_test = test[test["method"] == choice["method"]]
        matched = method_test[
            np.isclose(method_test["lambda"], choice["lambda"])
        ]
        baseline = method_test[np.isclose(method_test["lambda"], 0.0)].groupby(
            "seed"
        ).first()
        selected_by_seed = matched.groupby("seed").first()
        common = baseline.index.intersection(selected_by_seed.index)
        if len(common) == 0:
            raise ValueError(f"No matched test seeds for selection {choice}.")
        ndcg_delta = (
            selected_by_seed.loc[common, "ndcg@10"]
            - baseline.loc[common, "ndcg@10"]
        )
        ild_delta = (
            selected_by_seed.loc[common, "ild@10"]
            - baseline.loc[common, "ild@10"]
        )
        subtopic_delta = (
            selected_by_seed.loc[common, "subtopic_recall@10"]
            - baseline.loc[common, "subtopic_recall@10"]
        )
        calibration_delta = (
            selected_by_seed.loc[common, "calibration@10"]
            - baseline.loc[common, "calibration@10"]
        )
        tests.append(
            {
                **choice,
                "seeds": len(common),
                "test_ndcg@10_mean": float(selected_by_seed.loc[common, "ndcg@10"].mean()),
                "test_ndcg@10_std_seed": float(selected_by_seed.loc[common, "ndcg@10"].std(ddof=1)),
                "test_ild@10_mean": float(selected_by_seed.loc[common, "ild@10"].mean()),
                "test_ild@10_std_seed": float(selected_by_seed.loc[common, "ild@10"].std(ddof=1)),
                "test_subtopic_recall@10_mean": float(selected_by_seed.loc[common, "subtopic_recall@10"].mean()),
                "test_calibration@10_mean": float(selected_by_seed.loc[common, "calibration@10"].mean()),
                "test_catalog_coverage@10_mean": float(selected_by_seed.loc[common, "catalog_coverage@10"].mean()),
                "test_exposure_gini@10_mean": float(selected_by_seed.loc[common, "exposure_gini@10"].mean()),
                "test_long_tail_share@10_mean": float(selected_by_seed.loc[common, "long_tail_share@10"].mean()),
                "baseline_ndcg@10_mean": float(baseline.loc[common, "ndcg@10"].mean()),
                "baseline_ild@10_mean": float(baseline.loc[common, "ild@10"].mean()),
                "baseline_subtopic_recall@10_mean": float(baseline.loc[common, "subtopic_recall@10"].mean()),
                "baseline_calibration@10_mean": float(baseline.loc[common, "calibration@10"].mean()),
                "delta_ndcg@10_mean": float(ndcg_delta.mean()),
                "delta_ild@10_mean": float(ild_delta.mean()),
                "delta_subtopic_recall@10_mean": float(subtopic_delta.mean()),
                "delta_calibration@10_mean": float(calibration_delta.mean()),
                "delta_ndcg_sign_p_conservative": float(matched["delta_ndcg_sign_p"].max()),
                "delta_ild_sign_p_conservative": float(matched["delta_ild_sign_p"].max()),
                "delta_ndcg_effect_dz_mean": float(matched["delta_ndcg_effect_dz"].mean()),
                "delta_ild_effect_dz_mean": float(matched["delta_ild_effect_dz"].mean()),
                "sign_test_variant": "normal approximation with continuity correction",
                "multiplicity_family": "exploratory_unadjusted",
                "holm_family_size": 0,
                "delta_ndcg_sign_p_holm": np.nan,
                "delta_ild_sign_p_holm": np.nan,
            }
        )

    primary_indices = [
        index
        for index, row in enumerate(tests)
        if row["scope"] == "primary_ild_across_methods"
    ]
    ndcg_adjusted = _holm_adjust(
        [float(tests[index]["delta_ndcg_sign_p_conservative"]) for index in primary_indices]
    )
    ild_adjusted = _holm_adjust(
        [float(tests[index]["delta_ild_sign_p_conservative"]) for index in primary_indices]
    )
    for index, ndcg_p, ild_p in zip(primary_indices, ndcg_adjusted, ild_adjusted):
        tests[index]["multiplicity_family"] = "primary_ild_across_four_budgets"
        tests[index]["holm_family_size"] = len(primary_indices)
        tests[index]["delta_ndcg_sign_p_holm"] = ndcg_p
        tests[index]["delta_ild_sign_p_holm"] = ild_p
    return pd.DataFrame(selections), pd.DataFrame(tests)


def _run_configuration(
    *,
    seed: int,
    split: str,
    experiment: str,
    methods: tuple[str, ...],
    processed_dir: Path,
    candidate_path: Path,
    output_root: Path,
    candidate_k: int,
    top_k: int,
    lambdas: np.ndarray,
    bootstrap_samples: int,
    feature_path: Path | None = None,
    feature_space: str = "genre",
    force: bool = False,
) -> list[Path]:
    paths: list[Path] = []
    output_dir = output_root / experiment / split
    for method in methods:
        path = output_dir / f"mf_{split}_{method}_tradeoff.csv"
        if force or not path.exists():
            path = run_tradeoff(
                "mf",
                split,  # type: ignore[arg-type]
                method,  # type: ignore[arg-type]
                processed_dir,
                candidate_path,
                output_dir,
                lambdas,
                top_k=top_k,
                candidate_k=candidate_k,
                feature_path=feature_path,
                feature_space=feature_space,
                batch_size=64,
                bootstrap_samples=bootstrap_samples,
                seed=seed,
            )
        paths.append(path)
    return paths


def run_thesis(config: ThesisConfig) -> Path:
    root = config.root.resolve()
    dataset_tag = "thesis_pos4_neg2_traincore5_dataseed2026"
    processed_dir = root / "data" / "processed" / dataset_tag
    stats_path = processed_dir / "stats.json"
    if config.force or not stats_path.exists():
        preprocess(
            PreprocessConfig(
                raw_dir=root / "data" / "raw",
                output_dir=processed_dir,
                min_interactions=5,
                positive_threshold=4.0,
                negative_threshold=2.0,
                seed=config.data_seed,
                negatives=100,
                core_on_train_only=True,
            )
        )
    tag_features = build_tag_genome_features(
        root / "data" / "raw",
        processed_dir,
        processed_dir / "item_tag_genome_svd64.npy",
        components=64,
        seed=config.data_seed,
        device=config.device,
    )
    tag_coverage = tag_features.with_name(tag_features.stem + "_coverage.npy")

    collected: list[pd.DataFrame] = []
    for seed in config.seeds:
        run_tag = f"seed{seed}_e{config.epochs}_d{config.embedding_dim}_enr0p5"
        artifact_dir = root / "artifacts" / dataset_tag / run_tag
        seed_output = root / "outputs" / dataset_tag / run_tag
        embedding_path = artifact_dir / "mf_embeddings.npz"
        if config.force or not embedding_path.exists():
            train(
                TrainingConfig(
                    processed_dir=processed_dir,
                    artifact_dir=artifact_dir,
                    embedding_dim=config.embedding_dim,
                    batch_size=config.batch_size,
                    epochs=config.epochs,
                    explicit_negative_ratio=config.explicit_negative_ratio,
                    seed=seed,
                    device=config.device,
                )
            )
        for split, seen_file in (("val", "train_seen.npz"), ("test", "train_val_seen.npz")):
            evaluate_sampled(processed_dir, artifact_dir, "mf", split)
            candidate_path = seed_output / f"mf_{split}_top200.npz"
            if config.force or not candidate_path.exists():
                retrieve_top_k(
                    processed_dir,
                    embedding_path,
                    candidate_path,
                    k=200,
                    user_batch_size=1024,
                    seen_file=seen_file,
                    device=config.device,
                )
            paths = _run_configuration(
                seed=seed,
                split=split,
                experiment="primary_n100_k10_genre",
                methods=METHODS,
                processed_dir=processed_dir,
                candidate_path=candidate_path,
                output_root=seed_output,
                candidate_k=100,
                top_k=10,
                lambdas=_lambdas(config.primary_lambda_step),
                bootstrap_samples=config.primary_bootstrap_samples,
                force=config.force,
            )
            for path in paths:
                frame = pd.read_csv(path)
                frame["seed"] = seed
                frame["experiment"] = "primary_n100_k10_genre"
                collected.append(frame)

            if seed == config.seeds[0]:
                robustness = (
                    ("robust_n50_k10_genre", 50, 10),
                    ("robust_n200_k10_genre", 200, 10),
                    ("robust_n100_k5_genre", 100, 5),
                    ("robust_n100_k20_genre", 100, 20),
                )
                for experiment, candidate_k, top_k in robustness:
                    paths = _run_configuration(
                        seed=seed,
                        split=split,
                        experiment=experiment,
                        methods=METHODS,
                        processed_dir=processed_dir,
                        candidate_path=candidate_path,
                        output_root=seed_output,
                        candidate_k=candidate_k,
                        top_k=top_k,
                        lambdas=_lambdas(config.robustness_lambda_step),
                        bootstrap_samples=config.robustness_bootstrap_samples,
                        force=config.force,
                    )
                    for path in paths:
                        frame = pd.read_csv(path)
                        frame["seed"] = seed
                        frame["experiment"] = experiment
                        collected.append(frame)

                tagged_candidates = tag_complete_candidate_pool(
                    candidate_path,
                    tag_coverage,
                    seed_output / f"mf_{split}_top100_tag_complete.npz",
                    k=100,
                )
                for feature_space, feature_path in (
                    ("genre", None),
                    ("tag_genome_svd64", tag_features),
                ):
                    experiment = f"tag_sensitivity_n100_k10_{feature_space}"
                    paths = _run_configuration(
                        seed=seed,
                        split=split,
                        experiment=experiment,
                        methods=("mmr",),
                        processed_dir=processed_dir,
                        candidate_path=tagged_candidates,
                        output_root=seed_output,
                        candidate_k=100,
                        top_k=10,
                        lambdas=_lambdas(config.robustness_lambda_step),
                        bootstrap_samples=config.robustness_bootstrap_samples,
                        feature_path=feature_path,
                        feature_space=feature_space,
                        force=config.force,
                    )
                    for path in paths:
                        frame = pd.read_csv(path)
                        frame["seed"] = seed
                        frame["experiment"] = experiment
                        collected.append(frame)

    results = pd.concat(collected, ignore_index=True)
    final_dir = root / "outputs" / dataset_tag / "aggregate"
    final_dir.mkdir(parents=True, exist_ok=True)
    all_path = final_dir / "all_thesis_results.csv"
    results.to_csv(all_path, index=False)
    primary = results[results["experiment"] == "primary_n100_k10_genre"]
    primary_mean = primary.groupby(
        ["model", "split", "method", "lambda"], as_index=False
    ).mean(numeric_only=True)
    primary_mean.to_csv(final_dir / "primary_cross_seed_results.csv", index=False)
    selections, selected_tests = _budget_selections(primary)
    selections.to_csv(final_dir / "validation_budget_selections.csv", index=False)
    selected_tests.to_csv(final_dir / "test_budget_results.csv", index=False)
    write_tradeoff_svgs(primary_mean, final_dir, cutoff=10, split="test")
    save_json(
        final_dir / "experiment_manifest.json",
        {
            **asdict(config),
            "root": str(root),
            "dataset_tag": dataset_tag,
            "methods": list(METHODS),
            "accuracy_budgets": list(BUDGETS),
            "dataset_stats": load_json(stats_path),
            "tag_feature_stats": load_json(tag_features.with_suffix(".json")),
            "result_rows": len(results),
        },
    )
    timestamped_message(f"Thesis experiment complete: {all_path}")
    return all_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--seeds", default="2026,2027,2028")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    seeds = tuple(int(value.strip()) for value in args.seeds.split(",") if value.strip())
    run_thesis(
        ThesisConfig(
            root=args.root,
            seeds=seeds,
            epochs=args.epochs,
            device=args.device,
            force=args.force,
        )
    )


if __name__ == "__main__":
    main()
