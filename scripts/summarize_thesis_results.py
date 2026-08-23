"""Create compact, reproducible thesis-result tables from the aggregate sweep."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "outputs" / "thesis_pos4_neg2_traincore5_dataseed2026" / "aggregate"


def select_on_validation(
    data: pd.DataFrame,
    experiment: str,
    method: str,
    budget: float = 0.05,
    objective: str | None = None,
    seed: int | None = None,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    val = data[
        (data["experiment"] == experiment)
        & (data["split"] == "val")
        & (data["method"] == method)
    ].copy()
    if seed is not None:
        val = val[val["seed"] == seed]
    if val.empty:
        raise ValueError((experiment, method))
    k = int(val["top_k"].iloc[0])
    ndcg = f"ndcg@{k}"
    if objective is None:
        objective = f"feature_ild@{k}"
    base = val[val["lambda"] == 0].iloc[0]
    feasible = val[val[ndcg] >= (1 - budget) * base[ndcg]]
    selected = feasible.sort_values([objective, ndcg, "lambda"], ascending=[False, False, True]).iloc[0]
    test = data[
        (data["experiment"] == experiment)
        & (data["split"] == "test")
        & (data["method"] == method)
        & np.isclose(data["lambda"], float(selected["lambda"]), atol=1e-7)
    ].iloc[0]
    test_base = data[
        (data["experiment"] == experiment)
        & (data["split"] == "test")
        & (data["method"] == method)
        & (data["lambda"] == 0)
    ].iloc[0]
    if seed is not None:
        test_candidates = data[
            (data["experiment"] == experiment)
            & (data["split"] == "test")
            & (data["method"] == method)
            & (data["seed"] == seed)
        ]
        test = test_candidates[np.isclose(test_candidates["lambda"], float(selected["lambda"]), atol=1e-7)].iloc[0]
        test_base = test_candidates[test_candidates["lambda"] == 0].iloc[0]
    return selected, test, test_base


def main() -> None:
    data = pd.read_csv(AGG / "all_thesis_results.csv")

    print("ROBUSTNESS, validation-selected within method at 5% budget")
    rows: list[dict[str, float | int | str]] = []
    for experiment in [
        "robust_n50_k10_genre",
        "robust_n100_k10_genre",
        "robust_n200_k10_genre",
        "robust_n100_k5_genre",
        "robust_n100_k20_genre",
    ]:
        if experiment not in set(data["experiment"]):
            if experiment == "robust_n100_k10_genre":
                experiment = "primary_n100_k10_genre"
            else:
                continue
        for method in ["mmr", "xquad", "calibration"]:
            selected, test, base = select_on_validation(data, experiment, method, seed=2026)
            k = int(selected["top_k"])
            rows.append(
                {
                    "experiment": experiment,
                    "method": method,
                    "lambda": selected["lambda"],
                    "candidate_recall": base[f"candidate_recall@{int(base['candidate_k'])}"],
                    "test_ndcg": test[f"ndcg@{k}"],
                    "ndcg_rel_pct": 100 * (test[f"ndcg@{k}"] / base[f"ndcg@{k}"] - 1),
                    "test_ild": test[f"ild@{k}"],
                    "ild_rel_pct": 100 * (test[f"ild@{k}"] / base[f"ild@{k}"] - 1),
                    "calibration": test[f"calibration@{k}"],
                    "subtopic_recall": test[f"subtopic_recall@{k}"],
                }
            )
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nTAG GENOME SENSITIVITY, same tag-complete candidate pool")
    tag_rows = []
    for experiment in [
        "tag_sensitivity_n100_k10_genre",
        "tag_sensitivity_n100_k10_tag_genome_svd64",
    ]:
        selected, test, base = select_on_validation(data, experiment, "mmr", seed=2026)
        tag_rows.append(
            {
                "experiment": experiment,
                "lambda": selected["lambda"],
                "test_ndcg": test["ndcg@10"],
                "ndcg_rel_pct": 100 * (test["ndcg@10"] / base["ndcg@10"] - 1),
                "genre_ild": test["ild@10"],
                "feature_ild": test["feature_ild@10"],
                "feature_ild_rel_pct": 100 * (test["feature_ild@10"] / base["feature_ild@10"] - 1),
            }
        )
    print(pd.DataFrame(tag_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nPRIMARY 5%-BUDGET SUBGROUPS, cross-seed mean")
    primary = data[
        (data["experiment"] == "primary_n100_k10_genre")
        & (data["split"] == "test")
        & (data["method"] == "mmr")
        & np.isclose(data["lambda"], 0.4, atol=1e-7)
    ]
    base = data[
        (data["experiment"] == "primary_n100_k10_genre")
        & (data["split"] == "test")
        & (data["method"] == "mmr")
        & (data["lambda"] == 0)
    ]
    sg_rows = []
    for subgroup in [
        "low_activity",
        "medium_activity",
        "high_activity",
        "focused_profile",
        "medium_profile",
        "broad_profile",
    ]:
        sg_rows.append(
            {
                "subgroup": subgroup,
                "users": int(primary[f"{subgroup}_users"].mean()),
                "base_ndcg": base[f"{subgroup}_ndcg@10"].mean(),
                "selected_ndcg": primary[f"{subgroup}_ndcg@10"].mean(),
                "ndcg_rel_pct": 100
                * (primary[f"{subgroup}_ndcg@10"].mean() / base[f"{subgroup}_ndcg@10"].mean() - 1),
                "base_ild": base[f"{subgroup}_ild@10"].mean(),
                "selected_ild": primary[f"{subgroup}_ild@10"].mean(),
                "ild_rel_pct": 100
                * (primary[f"{subgroup}_ild@10"].mean() / base[f"{subgroup}_ild@10"].mean() - 1),
            }
        )
    print(pd.DataFrame(sg_rows).to_string(index=False, float_format=lambda x: f"{x:.6f}"))

    print("\nPRIMARY BASELINE AND SYSTEM COST, cross-seed mean")
    selected = primary
    cols = [
        "recall@10",
        "ndcg@10",
        "ild@10",
        "calibration@10",
        "subtopic_recall@10",
        "catalog_coverage@10",
        "exposure_gini@10",
        "long_tail_share@10",
        "candidate_recall@100",
        "rerank_seconds_all_lambdas",
        "amortized_rerank_ms_per_user_config",
        "peak_traced_memory_mb",
    ]
    print(pd.DataFrame({"baseline": base[cols].mean(), "mmr_lambda_0.4": selected[cols].mean()}))


if __name__ == "__main__":
    main()
