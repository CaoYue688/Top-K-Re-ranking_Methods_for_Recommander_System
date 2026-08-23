from __future__ import annotations

# Dieses Modul führt gewichtete Re-Ranking-Sweeps und Pareto-Analysen aus.
# 本模块执行加权重排扫描和 Pareto 分析。
# This module runs weighted re-ranking sweeps and Pareto analyses.
import argparse
import math
import time
import tracemalloc
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

from .evaluation import full_ranking_metrics, per_user_list_quality
from .utils import minmax_rows, save_json, timestamped_message


@dataclass(frozen=True)
class SweepObjective:
    # Beschreibt einen Punkt im Accuracy-Diversity-Experiment.
    # 描述准确率-多样性实验中的一个点。
    # Describes one point in the accuracy-diversity experiment.
    method: str
    tradeoff: float
    relevance_weight: float
    calibration_weight: float
    diversity_weight: float


def make_objectives(
    method: Literal["diversity", "mmr", "xquad", "calibration", "combined"],
    lambdas: np.ndarray,
) -> list[SweepObjective]:
    # Erzeugt vergleichbare Gewichtskombinationen mit einer Summe von genau 1.
    # 生成权重和精确为 1 的可比较权重组合。
    # Creates comparable weight combinations whose sum is exactly 1.
    objectives: list[SweepObjective] = []
    for value in lambdas:
        tradeoff = float(value)
        if not 0.0 <= tradeoff <= 1.0:
            raise ValueError("Every lambda must be between 0 and 1.")
        if method in {"diversity", "mmr", "xquad"}:
            calibration, diversity = 0.0, tradeoff
        elif method == "calibration":
            calibration, diversity = tradeoff, 0.0
        else:
            calibration = diversity = tradeoff / 2.0
        objectives.append(
            SweepObjective(
                method=method,
                tradeoff=tradeoff,
                relevance_weight=1.0 - tradeoff,
                calibration_weight=calibration,
                diversity_weight=diversity,
            )
        )
    return objectives


def _jsd_components(
    profile: np.ndarray,
    recommendation: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    # JSD ist additiv über Genres; diese Funktion liefert Beiträge statt ihrer Summe.
    # JSD 在类型维度上可加；此函数返回各项贡献而非总和。
    # JSD is additive over genres; this function returns components instead of their sum.
    midpoint = 0.5 * (profile + recommendation)
    profile_term = np.where(
        profile > 0,
        profile * np.log((profile + eps) / (midpoint + eps)),
        0.0,
    )
    recommendation_term = np.where(
        recommendation > 0,
        recommendation
        * np.log((recommendation + eps) / (midpoint + eps)),
        0.0,
    )
    return 0.5 * (profile_term + recommendation_term)


def _sparse_calibration_scores(
    profiles: np.ndarray,
    genre_sum: np.ndarray,
    candidate_genre_indices: np.ndarray,
    candidate_genre_counts: np.ndarray,
    rank: int,
) -> np.ndarray:
    # Berechnet exakt dieselbe JSD, aktualisiert aber nur tatsächlich vorhandene Genres.
    # 计算完全相同的 JSD，但只更新实际存在的类型。
    # Computes exactly the same JSD while updating only genres that are present.
    batch_users, n_objectives, _ = genre_sum.shape
    n_candidates = candidate_genre_indices.shape[1]
    profile = profiles[:, None, :]
    base_mix = genre_sum / (rank + 1)
    base_components = _jsd_components(profile, base_mix)
    candidate_jsd = np.broadcast_to(
        base_components.sum(axis=2)[:, :, None],
        (batch_users, n_objectives, n_candidates),
    ).copy()
    objective_index = np.arange(n_objectives)[None, :, None]

    # Items mit gleich vielen Genres erhalten denselben Massenzuwachs je aktivem Genre.
    # 类型数相同的物品在每个活跃类型上获得相同增量。
    # Items with the same genre count receive the same increment per active genre.
    for genre_count in np.unique(candidate_genre_counts):
        count = int(genre_count)
        if count <= 0:
            continue
        after_components = _jsd_components(
            profile,
            base_mix + 1.0 / (count * (rank + 1)),
        )
        component_delta = after_components - base_components
        matching_users, matching_candidates = np.nonzero(
            candidate_genre_counts == count
        )
        matching_genres = candidate_genre_indices[
            matching_users, matching_candidates, :count
        ]
        gathered_delta = component_delta[
            matching_users[:, None, None],
            objective_index,
            matching_genres[:, None, :],
        ]
        candidate_jsd[matching_users, :, matching_candidates] += (
            gathered_delta.sum(axis=2)
        )
    return 1.0 - candidate_jsd / np.log(2.0)


def sweep_rerank(
    candidates: np.ndarray,
    relevance_scores: np.ndarray,
    user_profiles: np.ndarray,
    item_genres: np.ndarray,
    objectives: list[SweepObjective],
    item_diversity_features: np.ndarray | None = None,
    top_k: int = 20,
    batch_size: int = 64,
    diversity_mode: Literal["mean", "max", "xquad"] = "mean",
) -> np.ndarray:
    # Berechnet viele Gewichtspunkte gemeinsam, damit Features wiederverwendet werden.
    # 同时计算多个权重点，以复用特征。
    # Computes many weight points jointly so features can be reused.
    if top_k > candidates.shape[1]:
        raise ValueError("top_k cannot exceed the candidate count.")
    if not objectives:
        raise ValueError("At least one sweep objective is required.")
    if item_diversity_features is None:
        item_diversity_features = item_genres
    weight_matrix = np.asarray(
        [
            (
                objective.relevance_weight,
                objective.calibration_weight,
                objective.diversity_weight,
            )
            for objective in objectives
        ],
        dtype=np.float32,
    )
    if np.any(weight_matrix < 0) or not np.allclose(
        weight_matrix.sum(axis=1), 1.0
    ):
        raise ValueError("Every objective must contain nonnegative weights summing to 1.")

    relevance = minmax_rows(relevance_scores).astype(np.float32, copy=False)
    genre_distribution = item_genres / np.maximum(
        item_genres.sum(axis=1, keepdims=True), 1.0
    )
    normalized_features = item_diversity_features / np.maximum(
        np.linalg.norm(item_diversity_features, axis=1, keepdims=True), 1e-12
    )
    genre_presence = (item_genres > 0).astype(np.float32)
    n_users, n_candidates = candidates.shape
    n_objectives = len(objectives)
    recommendations = np.empty(
        (n_objectives, n_users, top_k), dtype=np.int32
    )
    objective_columns = np.arange(n_objectives)[None, :]
    needs_calibration = bool(np.any(weight_matrix[:, 1] > 0))
    genre_counts = item_genres.sum(axis=1).astype(np.int32)
    max_item_genres = int(genre_counts.max())
    all_genre_indices = np.broadcast_to(
        np.arange(item_genres.shape[1], dtype=np.int32),
        item_genres.shape,
    )
    item_genre_indices = np.sort(
        np.where(item_genres > 0, all_genre_indices, item_genres.shape[1]),
        axis=1,
    )[:, :max_item_genres]
    item_genre_indices[item_genre_indices == item_genres.shape[1]] = 0

    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        batch_users = end - start
        rows = np.arange(batch_users)[:, None]
        batch_candidates = candidates[start:end]
        candidate_genres = genre_distribution[batch_candidates]
        candidate_vectors = normalized_features[batch_candidates]
        candidate_aspects = genre_presence[batch_candidates]
        profiles = user_profiles[start:end]
        candidate_genre_counts = genre_counts[batch_candidates]
        candidate_genre_indices_batch = item_genre_indices[batch_candidates]

        chosen = np.zeros(
            (batch_users, n_objectives, n_candidates), dtype=bool
        )
        genre_sum = np.zeros(
            (batch_users, n_objectives, item_genres.shape[1]),
            dtype=np.float32,
        )
        selected_vector_sum = np.zeros(
            (
                batch_users,
                n_objectives,
                item_diversity_features.shape[1],
            ),
            dtype=np.float32,
        )
        max_similarity = np.zeros(
            (batch_users, n_objectives, n_candidates), dtype=np.float32
        )
        aspect_not_covered = np.ones_like(genre_sum)

        for rank in range(top_k):
            if needs_calibration:
                # Die sparse Form vermeidet Logarithmen für Kandidat-Genre-Nullen.
                # 稀疏形式避免对候选项的零类型计算对数。
                # The sparse form avoids logarithms for zero candidate genres.
                calibration = _sparse_calibration_scores(
                    profiles,
                    genre_sum,
                    candidate_genre_indices_batch,
                    candidate_genre_counts,
                    rank,
                )
            else:
                calibration = 0.0

            if diversity_mode == "xquad":
                # Binary user-aspect xQuAD: an observed genre covers the aspect
                # completely after its first selected occurrence.  This is a
                # hard-coverage adaptation, not a probabilistic soft-coverage
                # model.  At rank zero the aspect term is already non-zero and
                # can therefore change the first item when lambda > 0.
                weighted_uncovered = (
                    profiles[:, None, :] * aspect_not_covered
                )
                diversity = np.einsum(
                    "bcg,bwg->bwc", candidate_aspects, weighted_uncovered
                )
            elif rank == 0:
                diversity = np.zeros_like(max_similarity)
            elif diversity_mode == "max":
                # MMR verwendet die Ähnlichkeit zum ähnlichsten bereits gewählten Item.
                # MMR 使用候选项与已选物品中最相似者的相似度。
                # MMR uses similarity to the most similar already selected item.
                diversity = 1.0 - np.clip(max_similarity, 0.0, 1.0)
            else:
                # Marginale ILD verwendet die mittlere Distanz zur bisherigen Liste.
                # 边际 ILD 使用候选项到当前列表的平均距离。
                # Marginal ILD uses the mean distance to the current list.
                mean_similarity = np.einsum(
                    "bcg,bwg->bwc",
                    candidate_vectors,
                    selected_vector_sum,
                ) / rank
                diversity = 1.0 - np.clip(mean_similarity, 0.0, 1.0)

            objective_score = (
                weight_matrix[None, :, 0, None]
                * relevance[start:end, None, :]
                + weight_matrix[None, :, 1, None] * calibration
                + weight_matrix[None, :, 2, None] * diversity
            )
            objective_score[chosen] = -np.inf
            selected_indices = np.argmax(objective_score, axis=2)
            expanded_candidates = np.broadcast_to(
                batch_candidates[:, None, :],
                (batch_users, n_objectives, n_candidates),
            )
            selected_items = np.take_along_axis(
                expanded_candidates, selected_indices[:, :, None], axis=2
            ).squeeze(2)
            recommendations[:, start:end, rank] = selected_items.T
            chosen[rows, objective_columns, selected_indices] = True

            selected_genres = candidate_genres[rows, selected_indices]
            selected_vectors = candidate_vectors[rows, selected_indices]
            selected_aspects = candidate_aspects[rows, selected_indices]
            genre_sum += selected_genres
            selected_vector_sum += selected_vectors
            aspect_not_covered *= 1.0 - selected_aspects
            if diversity_mode == "max":
                new_similarity = np.einsum(
                    "bcg,bwg->bwc", candidate_vectors, selected_vectors
                )
                max_similarity = np.maximum(max_similarity, new_similarity)

        if end % 10_000 < batch_size or end == n_users:
            timestamped_message(
                f"Trade-off sweep reranked: {end:,}/{n_users:,} users"
            )
    return recommendations


def _bootstrap_means(
    values: np.ndarray,
    samples: int,
    seed: int,
) -> np.ndarray:
    # Paired Bootstrap: dieselben Benutzerindizes gelten für alle Kurvenpunkte.
    # 成对 Bootstrap：所有曲线点使用相同的用户索引。
    # Paired bootstrap: the same user indices apply to all curve points.
    rng = np.random.default_rng(seed)
    n_users = len(values)
    means = np.empty((samples, values.shape[1]), dtype=np.float32)
    for sample in range(samples):
        indices = rng.integers(0, n_users, size=n_users, dtype=np.int32)
        means[sample] = values[indices].mean(axis=0)
    return means


def _paired_effect_and_sign_p(delta: np.ndarray) -> tuple[float, float]:
    """Return Cohen's dz and a two-sided normal sign-test approximation.

    The approximation uses the continuity correction and deliberately keeps
    zero differences out of the sign count.  Result tables name this variant
    explicitly; it must not be described as an exact binomial test.
    """
    standard_deviation = float(delta.std(ddof=1)) if len(delta) > 1 else 0.0
    effect = float(delta.mean() / standard_deviation) if standard_deviation > 0 else 0.0
    nonzero = delta[delta != 0]
    if len(nonzero) == 0:
        return effect, 1.0
    positives = int((nonzero > 0).sum())
    expected = len(nonzero) / 2.0
    z = (abs(positives - expected) - 0.5) / math.sqrt(len(nonzero) / 4.0)
    return effect, float(math.erfc(max(z, 0.0) / math.sqrt(2.0)))


def _user_subgroups(
    processed_dir: Path,
    user_profiles: np.ndarray,
) -> dict[str, np.ndarray]:
    # Aktivität und Profilentropie erlauben eine robuste Analyse nach Benutzertyp.
    # 活跃度和画像熵支持按用户类型进行稳健分析。
    # Activity and profile entropy enable robust analysis by user type.
    with np.load(processed_dir / "train.npz") as train:
        activity = np.bincount(
            train["user"], minlength=len(user_profiles)
        ).astype(np.float32)
    profile_entropy = -np.sum(
        np.where(
            user_profiles > 0,
            user_profiles * np.log(user_profiles + 1e-12),
            0.0,
        ),
        axis=1,
    )
    activity_low, activity_high = np.quantile(activity, (1 / 3, 2 / 3))
    entropy_low, entropy_high = np.quantile(profile_entropy, (1 / 3, 2 / 3))
    return {
        "low_activity": activity <= activity_low,
        "medium_activity": (activity > activity_low) & (activity <= activity_high),
        "high_activity": activity > activity_high,
        "focused_profile": profile_entropy <= entropy_low,
        "medium_profile": (
            (profile_entropy > entropy_low) & (profile_entropy <= entropy_high)
        ),
        "broad_profile": profile_entropy > entropy_high,
    }


def _gini_from_counts(counts: np.ndarray) -> float:
    # Gini 0 means equal exposure; values near 1 indicate concentration.
    values = np.sort(counts.astype(np.float64, copy=False))
    total = float(values.sum())
    if total == 0.0:
        return 0.0
    indices = np.arange(1, len(values) + 1, dtype=np.float64)
    return float(
        (2.0 * np.dot(indices, values) / (len(values) * total))
        - (len(values) + 1.0) / len(values)
    )


def _long_tail_mask(processed_dir: Path, n_items: int) -> np.ndarray:
    # The long tail is the 80% least-popular items by positive train frequency.
    with np.load(processed_dir / "train.npz") as train:
        popularity = np.bincount(train["item"], minlength=n_items)
    order = np.argsort(popularity, kind="stable")
    mask = np.ones(n_items, dtype=bool)
    head_size = max(1, int(np.ceil(0.20 * n_items)))
    mask[order[-head_size:]] = False
    return mask


def _mark_pareto(rows: list[dict[str, object]], cutoff: int) -> None:
    # Ein Punkt ist Pareto-optimal, wenn kein anderer in NDCG und ILD besser ist.
    # 如果没有其他点在 NDCG 和 ILD 上同时更好，该点即为 Pareto 最优。
    # A point is Pareto-optimal if no other point is better in both NDCG and ILD.
    for index, row in enumerate(rows):
        accuracy = float(row[f"ndcg@{cutoff}"])
        diversity = float(row[f"ild@{cutoff}"])
        dominated = False
        for other_index, other in enumerate(rows):
            if other_index == index:
                continue
            other_accuracy = float(other[f"ndcg@{cutoff}"])
            other_diversity = float(other[f"ild@{cutoff}"])
            if (
                other_accuracy >= accuracy
                and other_diversity >= diversity
                and (other_accuracy > accuracy or other_diversity > diversity)
            ):
                dominated = True
                break
        row["pareto"] = not dominated


def evaluate_sweep(
    recommendations: np.ndarray,
    objectives: list[SweepObjective],
    candidates: np.ndarray,
    processed_dir: Path,
    split: Literal["val", "test"],
    model: str,
    item_diversity_features: np.ndarray | None = None,
    feature_space: str = "genre",
    rerank_seconds: float | None = None,
    peak_memory_mb: float | None = None,
    bootstrap_samples: int = 200,
    seed: int = 2026,
) -> list[dict[str, object]]:
    # Bewertet jeden Gewichtspunkt mit allen Accuracy- und Beyond-Accuracy-Metriken.
    # 使用所有准确性和超越准确性指标评估每个权重点。
    # Evaluates each weight point with all accuracy and beyond-accuracy metrics.
    with np.load(processed_dir / f"{split}.npz") as split_data:
        positive_users = split_data["user"]
        positive_items = split_data["item"]
    item_genres = np.load(processed_dir / "item_genres.npy")
    if item_diversity_features is None:
        item_diversity_features = item_genres
    user_profiles = np.load(processed_dir / "user_genre_profiles.npy")
    subgroup_masks = _user_subgroups(processed_dir, user_profiles)
    n_items = len(item_genres)
    long_tail = _long_tail_mask(processed_dir, n_items)
    cutoff = recommendations.shape[2]
    candidate_cutoff = candidates.shape[1]
    candidate_metrics, _ = full_ranking_metrics(
        candidates, positive_users, positive_items, n_items
    )

    rows: list[dict[str, object]] = []
    per_user_blocks: list[np.ndarray] = []
    metric_names = (
        "recall",
        "ndcg",
        "hit_rate",
        "mrr",
        "ild",
        "feature_ild",
        "calibration",
        "genre_entropy",
        "genre_count",
        "subtopic_recall",
    )
    for objective, recs in zip(objectives, recommendations):
        accuracy, per_user_accuracy = full_ranking_metrics(
            recs, positive_users, positive_items, n_items
        )
        quality = per_user_list_quality(
            recs,
            user_profiles,
            item_genres,
            item_diversity_features=item_diversity_features,
        )
        per_user_blocks.append(
            np.column_stack(
                (
                    per_user_accuracy["recall"],
                    per_user_accuracy["ndcg"],
                    per_user_accuracy["hit_rate"],
                    per_user_accuracy["mrr"],
                    quality["ild"],
                    quality["feature_ild"],
                    quality["calibration"],
                    quality["genre_entropy"],
                    quality["genre_count"],
                    quality["subtopic_recall"],
                )
            ).astype(np.float32)
        )
        unique_items = np.unique(recs)
        exposure_counts = np.bincount(recs.ravel(), minlength=n_items)
        rows.append(
            {
                "model": model,
                "split": split,
                "method": objective.method,
                "feature_space": feature_space,
                "candidate_k": candidate_cutoff,
                "top_k": cutoff,
                "lambda": objective.tradeoff,
                "relevance_weight": objective.relevance_weight,
                "calibration_weight": objective.calibration_weight,
                "diversity_weight": objective.diversity_weight,
                **accuracy,
                f"ild@{cutoff}": float(quality["ild"].mean()),
                f"feature_ild@{cutoff}": float(quality["feature_ild"].mean()),
                f"calibration@{cutoff}": float(quality["calibration"].mean()),
                f"js_distance@{cutoff}": float(quality["js_distance"].mean()),
                f"genre_entropy@{cutoff}": float(quality["genre_entropy"].mean()),
                f"genre_count@{cutoff}": float(quality["genre_count"].mean()),
                f"subtopic_recall@{cutoff}": float(
                    quality["subtopic_recall"].mean()
                ),
                f"catalog_coverage@{cutoff}": float(len(unique_items) / n_items),
                f"genre_coverage@{cutoff}": float(
                    item_genres[unique_items].any(axis=0).mean()
                ),
                f"exposure_gini@{cutoff}": _gini_from_counts(exposure_counts),
                f"long_tail_share@{cutoff}": float(long_tail[recs].mean()),
                f"candidate_recall@{candidate_cutoff}": candidate_metrics[
                    f"recall@{candidate_cutoff}"
                ],
                "rerank_seconds_all_lambdas": rerank_seconds,
                "amortized_rerank_ms_per_user_config": (
                    1000.0 * rerank_seconds / (len(recs) * len(objectives))
                    if rerank_seconds is not None
                    else None
                ),
                "peak_traced_memory_mb": peak_memory_mb,
            }
        )
        # Subgruppen zeigen, ob der Trade-off für kurze oder enge Profile anders ausfällt.
        # 子组分析显示交易取舍是否对短历史或窄偏好画像有所不同。
        # Subgroups show whether the trade-off differs for short or narrow profiles.
        for subgroup, mask in subgroup_masks.items():
            subgroup_size = int(mask.sum())
            rows[-1][f"{subgroup}_users"] = subgroup_size
            for metric_name, values in (
                ("ndcg", per_user_accuracy["ndcg"]),
                ("ild", quality["ild"]),
                ("calibration", quality["calibration"]),
                ("subtopic_recall", quality["subtopic_recall"]),
            ):
                rows[-1][f"{subgroup}_{metric_name}@{cutoff}"] = (
                    float(values[mask].mean()) if subgroup_size else float("nan")
                )

    # Alle Kurvenpunkte werden mit denselben Bootstrap-Stichproben verglichen.
    # 使用相同的 Bootstrap 样本比较所有曲线点。
    # Compares all curve points using the same bootstrap samples.
    stacked = np.concatenate(per_user_blocks, axis=1)
    bootstrap = _bootstrap_means(stacked, bootstrap_samples, seed)
    n_metrics = len(metric_names)
    baseline_index = int(
        np.argmin([objective.tradeoff for objective in objectives])
    )
    baseline_start = baseline_index * n_metrics
    baseline_ndcg = bootstrap[:, baseline_start + metric_names.index("ndcg")]
    baseline_ild = bootstrap[:, baseline_start + metric_names.index("ild")]
    for objective_index, row in enumerate(rows):
        start = objective_index * n_metrics
        for metric_index, metric_name in enumerate(metric_names):
            samples = bootstrap[:, start + metric_index]
            row[f"{metric_name}_ci_low"] = float(np.quantile(samples, 0.025))
            row[f"{metric_name}_ci_high"] = float(np.quantile(samples, 0.975))
        ndcg_delta = (
            bootstrap[:, start + metric_names.index("ndcg")] - baseline_ndcg
        )
        ild_delta = (
            bootstrap[:, start + metric_names.index("ild")] - baseline_ild
        )
        row["delta_ndcg_ci_low"] = float(np.quantile(ndcg_delta, 0.025))
        row["delta_ndcg_ci_high"] = float(np.quantile(ndcg_delta, 0.975))
        row["delta_ild_ci_low"] = float(np.quantile(ild_delta, 0.025))
        row["delta_ild_ci_high"] = float(np.quantile(ild_delta, 0.975))
        baseline_values = per_user_blocks[baseline_index]
        current_values = per_user_blocks[objective_index]
        ndcg_user_delta = (
            current_values[:, metric_names.index("ndcg")]
            - baseline_values[:, metric_names.index("ndcg")]
        )
        ild_user_delta = (
            current_values[:, metric_names.index("ild")]
            - baseline_values[:, metric_names.index("ild")]
        )
        ndcg_effect, ndcg_p = _paired_effect_and_sign_p(ndcg_user_delta)
        ild_effect, ild_p = _paired_effect_and_sign_p(ild_user_delta)
        row["delta_ndcg_effect_dz"] = ndcg_effect
        row["delta_ndcg_sign_p"] = ndcg_p
        row["delta_ild_effect_dz"] = ild_effect
        row["delta_ild_sign_p"] = ild_p

    _mark_pareto(rows, cutoff)
    return rows


def run_tradeoff(
    model: str,
    split: Literal["val", "test"],
    method: Literal["diversity", "mmr", "xquad", "calibration", "combined"],
    processed_dir: Path,
    candidate_path: Path,
    output_dir: Path,
    lambdas: np.ndarray,
    top_k: int = 20,
    candidate_k: int | None = None,
    feature_path: Path | None = None,
    feature_space: str = "genre",
    batch_size: int = 64,
    bootstrap_samples: int = 200,
    seed: int = 2026,
    save_recommendations: bool = False,
) -> Path:
    # Lädt Kandidaten, führt den Sweep aus und schreibt eine tabellarische Auswertung.
    # 加载候选集、执行扫描并写出表格评估。
    # Loads candidates, runs the sweep, and writes a tabular evaluation.
    if bootstrap_samples < 1:
        raise ValueError("bootstrap_samples must be at least 1.")
    with np.load(candidate_path) as candidate_data:
        candidates = candidate_data["items"]
        relevance_scores = candidate_data["scores"]
    if candidate_k is not None:
        if candidate_k < top_k or candidate_k > candidates.shape[1]:
            raise ValueError("candidate_k must be between top_k and stored candidates.")
        candidates = candidates[:, :candidate_k]
        relevance_scores = relevance_scores[:, :candidate_k]
    profiles = np.load(processed_dir / "user_genre_profiles.npy")
    item_genres = np.load(processed_dir / "item_genres.npy")
    item_diversity_features = (
        np.load(feature_path) if feature_path is not None else item_genres
    )
    if len(item_diversity_features) != len(item_genres):
        raise ValueError("Feature rows must match item_genres rows.")
    objectives = make_objectives(method, lambdas)
    tracemalloc.start()
    started = time.perf_counter()
    recommendations = sweep_rerank(
        candidates,
        relevance_scores,
        profiles,
        item_genres,
        objectives,
        item_diversity_features=item_diversity_features,
        top_k=top_k,
        batch_size=batch_size,
        diversity_mode=(
            "max" if method == "mmr" else "xquad" if method == "xquad" else "mean"
        ),
    )
    rerank_seconds = time.perf_counter() - started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    rows = evaluate_sweep(
        recommendations,
        objectives,
        candidates,
        processed_dir,
        split,
        model,
        item_diversity_features=item_diversity_features,
        feature_space=feature_space,
        rerank_seconds=rerank_seconds,
        peak_memory_mb=peak_bytes / (1024.0 * 1024.0),
        bootstrap_samples=bootstrap_samples,
        seed=seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    result_path = output_dir / f"{model}_{split}_{method}_tradeoff.csv"
    pd.DataFrame(rows).to_csv(result_path, index=False)
    save_json(
        output_dir / f"{model}_{split}_{method}_tradeoff.json",
        {"rows": rows},
    )
    if save_recommendations:
        np.savez(
            output_dir / f"{model}_{split}_{method}_recommendations.npz",
            items=recommendations,
            lambdas=lambdas,
        )
    timestamped_message(f"Saved trade-off results to {result_path}")
    return result_path


def main() -> None:
    # Kommandozeile für einzelne reproduzierbare Trade-off-Experimente.
    # 用于单个可复现取舍实验的命令行入口。
    # Command line for individual reproducible trade-off experiments.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("mf",))
    parser.add_argument(
        "method", choices=("diversity", "mmr", "xquad", "calibration", "combined")
    )
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--processed-dir", type=Path, default=Path("data/research/processed")
    )
    parser.add_argument("--candidate-path", type=Path, required=True)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/research")
    )
    parser.add_argument("--lambda-step", type=float, default=0.05)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--candidate-k", type=int)
    parser.add_argument("--feature-path", type=Path)
    parser.add_argument("--feature-space", default="genre")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--save-recommendations", action="store_true")
    args = parser.parse_args()
    lambdas = np.arange(0.0, 1.0 + args.lambda_step / 2, args.lambda_step)
    run_tradeoff(
        args.model,
        args.split,
        args.method,
        args.processed_dir,
        args.candidate_path,
        args.output_dir,
        lambdas,
        args.top_k,
        args.candidate_k,
        args.feature_path,
        args.feature_space,
        args.batch_size,
        args.bootstrap_samples,
        args.seed,
        args.save_recommendations,
    )


if __name__ == "__main__":
    # Startet den Trade-off-Sweep bei direktem Modulaufruf.
    # 在模块被直接调用时启动取舍扫描。
    # Starts the trade-off sweep when the module is invoked directly.
    main()
