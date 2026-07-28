from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils import minmax_rows, save_json, timestamped_message


def ranking_metrics_from_ranks(
    ranks: np.ndarray, cutoffs: tuple[int, ...] = (10, 20, 100)
) -> dict[str, float]:
    metrics: dict[str, float] = {}
    for cutoff in cutoffs:
        hit = ranks <= cutoff
        metrics[f"hr@{cutoff}"] = float(hit.mean())
        metrics[f"ndcg@{cutoff}"] = float(
            np.where(hit, 1.0 / np.log2(ranks + 1), 0.0).mean()
        )
        metrics[f"mrr@{cutoff}"] = float(
            np.where(hit, 1.0 / ranks, 0.0).mean()
        )
    return metrics


def sampled_candidate_scores(
    embedding_path: Path,
    candidates: np.ndarray,
    model: str,
    batch_size: int = 4096,
) -> np.ndarray:
    with np.load(embedding_path) as embeddings:
        user_vectors = embeddings["user_embedding"]
        item_vectors = embeddings["item_embedding"]
        item_bias = embeddings["item_bias"] if "item_bias" in embeddings else None
    scores = np.empty(candidates.shape, dtype=np.float32)
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
    with np.load(processed_dir / "eval_candidates.npz") as data:
        candidates = data[split]
        positive_column = int(data["positive_column"])
    scores = sampled_candidate_scores(
        artifact_dir / f"{model}_embeddings.npz", candidates, model
    )
    positive = scores[:, positive_column]
    # A tied negative is conservatively ranked ahead of the positive.
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
    p = user_profiles / np.maximum(user_profiles.sum(axis=1, keepdims=True), eps)
    q = recommendation_genre_distributions / np.maximum(
        recommendation_genre_distributions.sum(axis=1, keepdims=True), eps
    )
    midpoint = 0.5 * (p + q)
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
    genre_denominator = np.maximum(item_genres.sum(axis=1, keepdims=True), 1.0)
    item_genre_distribution = item_genres / genre_denominator
    normalized_items = item_diversity_features / np.maximum(
        np.linalg.norm(item_diversity_features, axis=1, keepdims=True), 1e-12
    )
    calibrations: list[np.ndarray] = []
    ild_values: list[np.ndarray] = []
    n_rec = recommendations.shape[1]
    upper = np.triu_indices(n_rec, k=1)
    for start in range(0, len(recommendations), batch_size):
        end = min(start + batch_size, len(recommendations))
        rec = recommendations[start:end]
        genre_mix = item_genre_distribution[rec].mean(axis=1)
        calibrations.append(
            jensen_shannon_calibration(user_profiles[start:end], genre_mix)
        )
        vectors = normalized_items[rec]
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("mf", "two-tower"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--split", choices=("val", "test"), default="test")
    args = parser.parse_args()
    evaluate_sampled(args.processed_dir, args.artifact_dir, args.model, args.split)


if __name__ == "__main__":
    main()
