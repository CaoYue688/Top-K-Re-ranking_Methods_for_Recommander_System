from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .evaluation import jensen_shannon_calibration, recommendation_quality
from .utils import minmax_rows, save_json, timestamped_message


@dataclass(frozen=True)
class RerankWeights:
    relevance: float = 0.70
    calibration: float = 0.15
    diversity: float = 0.15

    def validate(self) -> None:
        total = self.relevance + self.calibration + self.diversity
        if min(self.relevance, self.calibration, self.diversity) < 0:
            raise ValueError("Reranking weights cannot be negative.")
        if not np.isclose(total, 1.0):
            raise ValueError(f"Reranking weights must sum to 1, got {total}.")


def greedy_rerank(
    candidates: np.ndarray,
    relevance_scores: np.ndarray,
    user_profiles: np.ndarray,
    item_genres: np.ndarray,
    item_diversity_features: np.ndarray,
    top_k: int,
    weights: RerankWeights,
    batch_size: int = 512,
) -> tuple[np.ndarray, np.ndarray]:
    weights.validate()
    if top_k > candidates.shape[1]:
        raise ValueError("top_k cannot exceed the candidate count.")
    normalized_relevance = minmax_rows(relevance_scores)
    genre_distribution = item_genres / np.maximum(
        item_genres.sum(axis=1, keepdims=True), 1.0
    )
    normalized_diversity_features = item_diversity_features / np.maximum(
        np.linalg.norm(item_diversity_features, axis=1, keepdims=True), 1e-12
    )
    n_users, n_candidates = candidates.shape
    result_items = np.empty((n_users, top_k), dtype=np.int32)
    result_objectives = np.empty((n_users, top_k), dtype=np.float32)

    eps = 1e-12
    for start in range(0, n_users, batch_size):
        end = min(start + batch_size, n_users)
        rows = np.arange(end - start)
        user_candidates = candidates[start:end]
        candidate_genres = genre_distribution[user_candidates]
        candidate_vectors = normalized_diversity_features[user_candidates]
        profiles = user_profiles[start:end, None, :]
        chosen = np.zeros((end - start, n_candidates), dtype=bool)
        genre_sum = np.zeros(
            (end - start, item_genres.shape[1]), dtype=np.float32
        )
        selected_vector_sum = np.zeros(
            (end - start, item_diversity_features.shape[1]), dtype=np.float32
        )

        for rank in range(top_k):
            after_mix = (genre_sum[:, None, :] + candidate_genres) / (rank + 1)
            midpoint = 0.5 * (profiles + after_mix)
            kl_profile = np.sum(
                np.where(
                    profiles > 0,
                    profiles * np.log((profiles + eps) / (midpoint + eps)),
                    0,
                ),
                axis=2,
            )
            kl_recommendation = np.sum(
                np.where(
                    after_mix > 0,
                    after_mix * np.log((after_mix + eps) / (midpoint + eps)),
                    0,
                ),
                axis=2,
            )
            calibration = 1.0 - (
                0.5 * (kl_profile + kl_recommendation) / np.log(2.0)
            )
            if rank:
                mean_similarity = np.einsum(
                    "bcd,bd->bc", candidate_vectors, selected_vector_sum
                ) / rank
                diversity = np.clip((1.0 - mean_similarity) / 2.0, 0.0, 1.0)
            else:
                diversity = np.zeros(
                    (end - start, n_candidates), dtype=np.float32
                )
            objective = (
                weights.relevance * normalized_relevance[start:end]
                + weights.calibration * calibration
                + weights.diversity * diversity
            )
            objective[chosen] = -np.inf
            selected_indices = np.argmax(objective, axis=1)
            selected_items = user_candidates[rows, selected_indices]
            result_items[start:end, rank] = selected_items
            result_objectives[start:end, rank] = objective[rows, selected_indices]
            chosen[rows, selected_indices] = True
            genre_sum += candidate_genres[rows, selected_indices]
            selected_vector_sum += candidate_vectors[rows, selected_indices]

        if end % 10_000 < batch_size or end == n_users:
            timestamped_message(f"Reranked users: {end:,}/{n_users:,}")
    return result_items, result_objectives


def write_sample_csv(
    path: Path,
    recommendations: np.ndarray,
    objectives: np.ndarray,
    processed_dir: Path,
    sample_users: int = 100,
) -> None:
    users = pd.read_csv(processed_dir / "users.csv")
    items = pd.read_csv(processed_dir / "items.csv")
    sample_users = min(sample_users, len(recommendations))
    rows: list[dict[str, object]] = []
    for user_idx in range(sample_users):
        for rank, item_idx in enumerate(recommendations[user_idx], start=1):
            item = items.iloc[int(item_idx)]
            rows.append(
                {
                    "user_idx": user_idx,
                    "userId": int(users.iloc[user_idx]["userId"]),
                    "rank": rank,
                    "item_idx": int(item_idx),
                    "movieId": int(item["movieId"]),
                    "title": item["title"],
                    "genres": item["genres"],
                    "rerank_score": float(objectives[user_idx, rank - 1]),
                }
            )
    pd.DataFrame(rows).to_csv(path, index=False)


def rerank_model(
    model: str,
    processed_dir: Path,
    artifact_dir: Path,
    output_dir: Path,
    candidate_k: int = 100,
    output_k: int = 20,
    weights: RerankWeights = RerankWeights(),
) -> dict[str, float]:
    with np.load(output_dir / f"{model}_top{candidate_k}.npz") as data:
        candidates, scores = data["items"], data["scores"]
    profiles = np.load(processed_dir / "user_genre_profiles.npy")
    item_genres = np.load(processed_dir / "item_genres.npy")
    recommendations, objective = greedy_rerank(
        candidates,
        scores,
        profiles,
        item_genres,
        item_genres,
        output_k,
        weights,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / f"{model}_top{output_k}_reranked.npz",
        items=recommendations,
        objective=objective,
    )
    metrics = recommendation_quality(
        recommendations, profiles, item_genres, item_genres
    )
    metrics.update(
        {
            "relevance_weight": weights.relevance,
            "calibration_weight": weights.calibration,
            "diversity_weight": weights.diversity,
        }
    )
    save_json(output_dir / f"{model}_top{output_k}_quality.json", metrics)
    write_sample_csv(
        output_dir / f"{model}_top{output_k}_sample.csv",
        recommendations,
        objective,
        processed_dir,
    )
    timestamped_message(f"{model} reranked Top-{output_k}: {metrics}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("mf", "two-tower"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--candidate-k", type=int, default=100)
    parser.add_argument("--output-k", type=int, default=20)
    parser.add_argument("--relevance-weight", type=float, default=0.70)
    parser.add_argument("--calibration-weight", type=float, default=0.15)
    parser.add_argument("--diversity-weight", type=float, default=0.15)
    args = parser.parse_args()
    rerank_model(
        args.model,
        args.processed_dir,
        args.artifact_dir,
        args.output_dir,
        args.candidate_k,
        args.output_k,
        RerankWeights(
            args.relevance_weight,
            args.calibration_weight,
            args.diversity_weight,
        ),
    )


if __name__ == "__main__":
    main()
