"""Derive the Expose-aligned hypothesis checks from validation-selected results."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from recsys20m.thesis_pipeline import BUDGETS  # noqa: E402


AGG = ROOT / "outputs" / "thesis_pos4_neg2_traincore5_dataseed2026" / "aggregate"
PRIMARY = "primary_n100_k10_genre"


def select_metric(
    data: pd.DataFrame,
    *,
    experiment: str,
    method: str,
    metric: str,
    budget: float,
    split: str = "val",
    seed: int | None = None,
) -> pd.Series:
    frame = data[
        data["experiment"].eq(experiment)
        & data["method"].eq(method)
        & data["split"].eq(split)
    ].copy()
    if seed is not None:
        frame = frame[frame["seed"].eq(seed)]
    else:
        frame = frame.groupby("lambda", as_index=False).mean(numeric_only=True)
    baseline = float(frame.loc[np.isclose(frame["lambda"], 0.0), "ndcg@10"].iloc[0])
    feasible = frame[frame["ndcg@10"] >= baseline * (1.0 - budget)]
    return feasible.sort_values(
        [metric, "ndcg@10", "lambda"], ascending=[False, False, True]
    ).iloc[0]


def segment_selections(data: pd.DataFrame) -> pd.DataFrame:
    frame = data[
        data["experiment"].eq(PRIMARY)
        & data["method"].eq("mmr")
        & data["split"].eq("val")
    ].groupby("lambda", as_index=False).mean(numeric_only=True)
    rows: list[dict[str, float | str]] = []
    for segment in ("focused_profile", "medium_profile", "broad_profile"):
        ndcg = f"{segment}_ndcg@10"
        ild = f"{segment}_ild@10"
        baseline_ndcg = float(frame.loc[np.isclose(frame["lambda"], 0.0), ndcg].iloc[0])
        baseline_ild = float(frame.loc[np.isclose(frame["lambda"], 0.0), ild].iloc[0])
        for budget in BUDGETS:
            feasible = frame[frame[ndcg] >= baseline_ndcg * (1.0 - budget)]
            selected = feasible.sort_values(
                [ild, ndcg, "lambda"], ascending=[False, False, True]
            ).iloc[0]
            rows.append(
                {
                    "segment": segment,
                    "budget": budget,
                    "lambda": float(selected["lambda"]),
                    "validation_ndcg": float(selected[ndcg]),
                    "validation_ndcg_delta_relative": float(selected[ndcg] / baseline_ndcg - 1.0),
                    "validation_ild": float(selected[ild]),
                    "validation_ild_delta_relative": float(selected[ild] / baseline_ild - 1.0),
                }
            )
    return pd.DataFrame(rows)


def candidate_pool_frontier(data: pd.DataFrame) -> pd.DataFrame:
    experiments = (
        (50, "robust_n50_k10_genre"),
        (100, PRIMARY),
        (200, "robust_n200_k10_genre"),
    )
    rows: list[dict[str, float | int]] = []
    for candidate_k, experiment in experiments:
        for budget in BUDGETS:
            selected = select_metric(
                data,
                experiment=experiment,
                method="mmr",
                metric="ild@10",
                budget=budget,
                seed=2026,
            )
            rows.append(
                {
                    "candidate_k": candidate_k,
                    "budget": budget,
                    "lambda": float(selected["lambda"]),
                    "validation_ndcg": float(selected["ndcg@10"]),
                    "validation_ild": float(selected["ild@10"]),
                }
            )
    result = pd.DataFrame(rows)
    result["outward_vs_previous_pool"] = np.nan
    for budget in BUDGETS:
        index = result.index[result["budget"].eq(budget)]
        values = result.loc[index].sort_values("candidate_k")
        outward = values["validation_ild"].diff() > 0
        result.loc[values.index, "outward_vs_previous_pool"] = outward.astype(float)
    return result


def metric_comparison(
    data: pd.DataFrame, *, metric: str, budget: float
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    for method in ("mmr", "xquad", "calibration"):
        selected = select_metric(
            data,
            experiment=PRIMARY,
            method=method,
            metric=metric,
            budget=budget,
        )
        test = data[
            data["experiment"].eq(PRIMARY)
            & data["method"].eq(method)
            & data["split"].eq("test")
            & np.isclose(data["lambda"], float(selected["lambda"]))
        ].mean(numeric_only=True)
        baseline = data[
            data["experiment"].eq(PRIMARY)
            & data["method"].eq(method)
            & data["split"].eq("test")
            & np.isclose(data["lambda"], 0.0)
        ].mean(numeric_only=True)
        rows.append(
            {
                "method": method,
                "target_metric": metric,
                "budget": budget,
                "lambda": float(selected["lambda"]),
                "test_metric": float(test[metric]),
                "test_metric_delta": float(test[metric] - baseline[metric]),
                "test_metric_delta_relative": float(test[metric] / baseline[metric] - 1.0),
                "test_ndcg": float(test["ndcg@10"]),
                "test_ndcg_delta_relative": float(test["ndcg@10"] / baseline["ndcg@10"] - 1.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    data = pd.read_csv(AGG / "all_thesis_results.csv")
    segments = segment_selections(data)
    pools = candidate_pool_frontier(data)
    coverage = metric_comparison(data, metric="subtopic_recall@10", budget=0.05)
    calibration = metric_comparison(data, metric="calibration@10", budget=0.05)
    segments.to_csv(AGG / "segment_budget_selections.csv", index=False)
    pools.to_csv(AGG / "candidate_pool_budget_frontier.csv", index=False)
    coverage.to_csv(AGG / "coverage_comparison_5pct.csv", index=False)
    calibration.to_csv(AGG / "calibration_comparison_5pct.csv", index=False)
    print(segments[segments["budget"].eq(0.05)].to_string(index=False))
    print(pools.to_string(index=False))
    print(coverage.to_string(index=False))
    print(calibration.to_string(index=False))


if __name__ == "__main__":
    main()
