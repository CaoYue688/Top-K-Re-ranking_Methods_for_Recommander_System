from __future__ import annotations

# Dieses Modul berechnet Ranking-, Kalibrierungs- und Diversitätsmetriken.
# 本模块计算排序、校准和多样性指标。
# This module computes ranking, calibration, and diversity metrics.
import argparse
from pathlib import Path

import numpy as np

from .utils import minmax_rows, save_json, timestamped_message


def ranking_metrics_from_ranks(
    ranks: np.ndarray, cutoffs: tuple[int, ...] = (10, 20, 100)
) -> dict[str, float]:
    # Berechnet Hit Rate, NDCG und MRR für mehrere Ranggrenzen.
    # 计算多个截止位置下的命中率、NDCG 和 MRR。
    # Computes Hit Rate, NDCG, and MRR at multiple cutoffs.
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        # Nur Treffer innerhalb des jeweiligen Cutoffs gehen in die Metrik ein.
        # 只有位于当前截止位置内的命中才计入指标。
        # Only hits within the current cutoff contribute to the metric.
        hit = ranks <= cutoff
        metrics[f"hr@{cutoff}"] = float(hit.mean())
        metrics[f"ndcg@{cutoff}"] = float(
            np.where(hit, 1.0 / np.log2(ranks + 1), 0.0).mean()
        )
        metrics[f"mrr@{cutoff}"] = float(
            np.where(hit, 1.0 / ranks, 0.0).mean()
        )
    return metrics


def full_ranking_metrics(
    recommendations: np.ndarray,
    positive_users: np.ndarray,
    positive_items: np.ndarray,
    n_items: int,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    # Bewertet eine Top-K-Liste gegen alle positiven Ereignisse jedes Benutzers.
    # 使用每个用户的全部正样本评估 Top-K 列表。
    # Evaluates a Top-K list against all positive events for each user.
    n_users, cutoff = recommendations.shape
    positive_counts = np.bincount(positive_users, minlength=n_users)
    if np.any(positive_counts == 0):
        raise ValueError("Every evaluated user needs at least one positive item.")

    # Kodierte Paare ermöglichen eine vektorisierte exakte Mitgliedschaftsprüfung.
    # 将用户-物品对编码后可以向量化地精确检查成员关系。
    # Encoded user-item pairs enable vectorized exact membership checks.
    positive_keys = np.sort(
        positive_users.astype(np.int64) * n_items
        + positive_items.astype(np.int64)
    )
    user_grid = np.arange(n_users, dtype=np.int64)[:, None]
    recommendation_keys = user_grid * n_items + recommendations.astype(np.int64)
    flat_keys = recommendation_keys.reshape(-1)
    locations = np.searchsorted(positive_keys, flat_keys)
    flat_hits = np.zeros(len(flat_keys), dtype=bool)
    valid = locations < len(positive_keys)
    flat_hits[valid] = positive_keys[locations[valid]] == flat_keys[valid]
    hits = flat_hits.reshape(n_users, cutoff)

    hit_counts = hits.sum(axis=1)
    recall = hit_counts / positive_counts
    hit_rate = (hit_counts > 0).astype(np.float64)
    discounts = 1.0 / np.log2(np.arange(2, cutoff + 2))
    dcg = (hits * discounts[None, :]).sum(axis=1)
    ideal_prefix = np.concatenate(([0.0], np.cumsum(discounts)))
    ideal_lengths = np.minimum(positive_counts, cutoff)
    ndcg = dcg / ideal_prefix[ideal_lengths]

    has_hit = hits.any(axis=1)
    first_hit = np.argmax(hits, axis=1) + 1
    mrr = np.where(has_hit, 1.0 / first_hit, 0.0)
    per_user = {
        "recall": recall.astype(np.float32),
        "ndcg": ndcg.astype(np.float32),
        "hit_rate": hit_rate.astype(np.float32),
        "mrr": mrr.astype(np.float32),
    }
    summary = {
        f"recall@{cutoff}": float(recall.mean()),
        f"ndcg@{cutoff}": float(ndcg.mean()),
        f"hr@{cutoff}": float(hit_rate.mean()),
        f"mrr@{cutoff}": float(mrr.mean()),
    }
    return summary, per_user


def per_user_list_quality(
    recommendations: np.ndarray,
    user_profiles: np.ndarray,
    item_genres: np.ndarray,
    item_diversity_features: np.ndarray | None = None,
    batch_size: int = 1024,
) -> dict[str, np.ndarray]:
    # Liefert Calibration, ILD, Entropie und Aspektabdeckung pro Benutzer.
    # 返回每个用户的校准度、ILD、熵和主题覆盖度。
    # Returns per-user calibration, ILD, entropy, and aspect coverage.
    if item_diversity_features is None:
        item_diversity_features = item_genres
    genre_distribution = item_genres / np.maximum(
        item_genres.sum(axis=1, keepdims=True), 1.0
    )
    normalized_genres = item_genres / np.maximum(
        np.linalg.norm(item_genres, axis=1, keepdims=True), 1e-12
    )
    normalized_features = item_diversity_features / np.maximum(
        np.linalg.norm(item_diversity_features, axis=1, keepdims=True), 1e-12
    )
    n_rec = recommendations.shape[1]
    upper = np.triu_indices(n_rec, k=1)
    calibration = np.empty(len(recommendations), dtype=np.float32)
    ild = np.empty(len(recommendations), dtype=np.float32)
    feature_ild = np.empty(len(recommendations), dtype=np.float32)
    genre_entropy = np.empty(len(recommendations), dtype=np.float32)
    genre_count = np.empty(len(recommendations), dtype=np.float32)
    subtopic_recall = np.empty(len(recommendations), dtype=np.float32)
    for start in range(0, len(recommendations), batch_size):
        end = min(start + batch_size, len(recommendations))
        rec = recommendations[start:end]
        genre_mix = genre_distribution[rec].mean(axis=1)
        calibration[start:end] = jensen_shannon_calibration(
            user_profiles[start:end], genre_mix
        )
        genre_vectors = normalized_genres[rec]
        similarities = np.einsum(
            "bkd,bld->bkl", genre_vectors, genre_vectors
        )
        ild[start:end] = (
            1.0 - similarities[:, upper[0], upper[1]]
        ).mean(axis=1)
        feature_vectors = normalized_features[rec]
        feature_similarities = np.einsum(
            "bkd,bld->bkl", feature_vectors, feature_vectors
        )
        feature_ild[start:end] = (
            1.0 - feature_similarities[:, upper[0], upper[1]]
        ).mean(axis=1)
        genre_entropy[start:end] = -np.sum(
            np.where(
                genre_mix > 0,
                genre_mix * np.log(genre_mix + 1e-12),
                0.0,
            ),
            axis=1,
        ) / max(float(np.log(item_genres.shape[1])), 1e-12)
        covered = item_genres[rec].any(axis=1)
        genre_count[start:end] = covered.sum(axis=1)
        subtopic_recall[start:end] = (
            covered * user_profiles[start:end]
        ).sum(axis=1)
    return {
        "calibration": calibration,
        "js_distance": 1.0 - calibration,
        "ild": ild,
        "feature_ild": feature_ild,
        "genre_entropy": genre_entropy,
        "genre_count": genre_count,
        "subtopic_recall": subtopic_recall,
    }


def sampled_candidate_scores(
    embedding_path: Path,
    candidates: np.ndarray,
    model: str,
    batch_size: int = 4096,
) -> np.ndarray:
    # Bewertet die 101 gesampelten Kandidaten durch Skalarprodukte der Embeddings.
    # 通过 embedding 点积为 101 个采样候选项打分。
    # Scores the 101 sampled candidates using embedding dot products.
    with np.load(embedding_path) as embeddings:
        user_vectors = embeddings["user_embedding"]
        item_vectors = embeddings["item_embedding"]
        item_bias = embeddings["item_bias"] if "item_bias" in embeddings else None
    scores = np.empty(candidates.shape, dtype=np.float32)
    # Batch-Verarbeitung verhindert große temporäre Matrizen im Arbeitsspeicher.
    # 分批处理可避免在内存中创建巨大的临时矩阵。
    # Batch processing avoids large temporary matrices in memory.
    for start in range(0, len(candidates), batch_size):
        end = min(start + batch_size, len(candidates))
        user_batch = user_vectors[start:end]
        item_batch = item_vectors[candidates[start:end]]
        scores[start:end] = np.einsum("bd,bkd->bk", user_batch, item_batch)
        if model == "mf" and item_bias is not None:
            scores[start:end] += item_bias[candidates[start:end]]
    return scores


def evaluate_sampled(
    processed_dir: Path,
    artifact_dir: Path,
    model: str,
    split: str = "test",
) -> dict[str, float]:
    # Führt die vollständige Sampled-Evaluation für ein Modell und einen Split aus.
    # 对指定模型和数据分割执行完整的采样评估。
    # Runs the complete sampled evaluation for a model and split.
    with np.load(processed_dir / "eval_candidates.npz") as data:
        candidates = data[split]
        positive_column = int(data["positive_column"])
    scores = sampled_candidate_scores(
        artifact_dir / f"{model}_embeddings.npz", candidates, model
    )
    positive = scores[:, positive_column]
    # Bei Gleichstand wird ein negatives Beispiel konservativ vor dem Positiven gewertet.
    # 分数相同时，保守地将负样本排在正样本之前。
    # On ties, a negative is conservatively ranked ahead of the positive.
    ranks = 1 + (scores >= positive[:, None]).sum(axis=1) - 1
    metrics = ranking_metrics_from_ranks(ranks)
    save_json(artifact_dir / f"{model}_{split}_sampled_metrics.json", metrics)
    timestamped_message(f"{model} {split} sampled metrics: {metrics}")
    return metrics


def jensen_shannon_calibration(
    user_profiles: np.ndarray,
    recommendation_genre_distributions: np.ndarray,
    eps: float = 1e-12,
) -> np.ndarray:
    # Vergleicht Benutzerprofil und Empfehlungsverteilung mit Jensen-Shannon-Ähnlichkeit.
    # 使用 Jensen-Shannon 相似度比较用户画像和推荐分布。
    # Compares user profiles and recommendation distributions using Jensen-Shannon similarity.
    p = user_profiles / np.maximum(user_profiles.sum(axis=1, keepdims=True), eps)
    q = recommendation_genre_distributions / np.maximum(
        recommendation_genre_distributions.sum(axis=1, keepdims=True), eps
    )
    midpoint = 0.5 * (p + q)
    # Beide KL-Divergenzen werden symmetrisch zur Jensen-Shannon-Divergenz kombiniert.
    # 将两个 KL 散度对称组合为 Jensen-Shannon 散度。
    # Symmetrically combines both KL divergences into Jensen-Shannon divergence.
    kl_p = np.sum(np.where(p > 0, p * np.log((p + eps) / (midpoint + eps)), 0), axis=1)
    kl_q = np.sum(np.where(q > 0, q * np.log((q + eps) / (midpoint + eps)), 0), axis=1)
    jsd = 0.5 * (kl_p + kl_q)
    return 1.0 - jsd / np.log(2.0)


def recommendation_quality(
    recommendations: np.ndarray,
    user_profiles: np.ndarray,
    item_genres: np.ndarray,
    item_diversity_features: np.ndarray,
    batch_size: int = 1024,
) -> dict[str, float]:
    # Misst Kalibrierung und Intra-List-Diversity einer fertigen Empfehlungsliste.
    # 测量最终推荐列表的校准度和列表内多样性。
    # Measures calibration and intra-list diversity of a final recommendation list.
    genre_denominator = np.maximum(item_genres.sum(axis=1, keepdims=True), 1.0)
    item_genre_distribution = item_genres / genre_denominator
    normalized_items = item_diversity_features / np.maximum(
        np.linalg.norm(item_diversity_features, axis=1, keepdims=True), 1e-12
    )
    calibrations: list[np.ndarray] = []
    ild_values: list[np.ndarray] = []
    n_rec = recommendations.shape[1]
    # Nur das obere Dreieck enthält eindeutige Artikelpaare ohne Diagonale.
    # 只取上三角区域，以保留不重复且不含对角线的物品对。
    # Only the upper triangle contains unique item pairs without the diagonal.
    upper = np.triu_indices(n_rec, k=1)
    for start in range(0, len(recommendations), batch_size):
        end = min(start + batch_size, len(recommendations))
        rec = recommendations[start:end]
        genre_mix = item_genre_distribution[rec].mean(axis=1)
        calibrations.append(
            jensen_shannon_calibration(user_profiles[start:end], genre_mix)
        )
        vectors = normalized_items[rec]
        # Kosinusdistanz ist 1 minus Kosinusähnlichkeit der normalisierten Vektoren.
        # 余弦距离等于 1 减去归一化向量的余弦相似度。
        # Cosine distance is one minus cosine similarity of normalized vectors.
        similarities = np.einsum("bkd,bld->bkl", vectors, vectors)
        ild_values.append((1.0 - similarities[:, upper[0], upper[1]]).mean(axis=1))
    calibration = np.concatenate(calibrations)
    ild = np.concatenate(ild_values)
    return {
        "calibration_mean": float(calibration.mean()),
        "calibration_std": float(calibration.std()),
        "ild_mean": float(ild.mean()),
        "ild_std": float(ild.std()),
    }


def main() -> None:
    # Stellt die Sampled-Evaluation als Kommandozeilenwerkzeug bereit.
    # 将采样评估提供为命令行工具。
    # Exposes sampled evaluation as a command-line tool.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("mf",))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    evaluate_sampled(args.processed_dir, args.artifact_dir, args.model, args.split)


if __name__ == "__main__":
    # Führt das Kommandozeilenwerkzeug nur bei direktem Aufruf aus.
    # 仅在模块被直接调用时运行命令行工具。
    # Runs the command-line tool only when invoked directly.
    main()
