from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils import load_json, save_json, timestamped_message


def _row_unique(values: np.ndarray) -> bool:
    ordered = np.sort(values, axis=1)
    return bool(np.all(np.diff(ordered, axis=1) != 0))


def validate_outputs(
    root: Path,
    sample_users: int = 5_000,
    seed: int = 2026,
) -> dict[str, object]:
    processed = root / "data" / "processed"
    artifacts = root / "artifacts"
    outputs = root / "outputs"
    stats = load_json(processed / "stats.json")
    n_users, n_items = int(stats["n_users"]), int(stats["n_items"])
    rng = np.random.default_rng(seed)
    sampled_users = np.sort(
        rng.choice(n_users, min(sample_users, n_users), replace=False)
    )
    checks: dict[str, object] = {}

    split_counts: dict[str, int] = {}
    boundaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    item_counts = np.zeros(n_items, dtype=np.int64)
    for split in ("train", "val", "test"):
        with np.load(processed / f"{split}.npz") as data:
            users, items, times = data["user"], data["item"], data["timestamp"]
        counts = np.bincount(users, minlength=n_users)
        offsets = np.concatenate(([0], np.cumsum(counts)))
        first_times = times[offsets[:-1]]
        last_times = times[offsets[1:] - 1]
        last_items = items[offsets[1:] - 1]
        boundaries[split] = (first_times, last_times, last_items)
        item_counts += np.bincount(items, minlength=n_items)
        split_counts[split] = len(users)
        checks[f"{split}_grouped_by_user"] = bool(np.all(np.diff(users) >= 0))
        checks[f"{split}_all_users_present"] = bool(np.all(counts > 0))
    checks["chronological_train_before_val"] = bool(
        np.all(boundaries["train"][1] <= boundaries["val"][0])
    )
    checks["chronological_val_before_test"] = bool(
        np.all(boundaries["val"][1] <= boundaries["test"][0])
    )
    checks["minimum_item_interactions"] = int(item_counts.min())
    checks["split_count_matches_stats"] = bool(
        split_counts["train"] == stats["train_interactions"]
        and split_counts["val"] == stats["val_interactions"]
        and split_counts["test"] == stats["test_interactions"]
    )

    profiles = np.load(processed / "user_genre_profiles.npy", mmap_mode="r")
    item_genres = np.load(processed / "item_genres.npy", mmap_mode="r")
    checks["profile_shape"] = list(profiles.shape)
    checks["item_genre_shape"] = list(item_genres.shape)
    checks["profiles_normalized"] = bool(
        np.allclose(profiles.sum(axis=1), 1.0, atol=1e-5)
    )

    with np.load(processed / "all_seen.npz") as seen_data:
        seen_items, offsets = seen_data["items"], seen_data["offsets"]
    with np.load(processed / "eval_candidates.npz") as eval_data:
        val_candidates, test_candidates = eval_data["val"], eval_data["test"]
    checks["eval_candidate_shape"] = list(test_candidates.shape)
    checks["val_positive_is_latest"] = bool(
        np.array_equal(val_candidates[:, 0], boundaries["val"][2])
    )
    checks["test_positive_is_latest"] = bool(
        np.array_equal(test_candidates[:, 0], boundaries["test"][2])
    )
    checks["sampled_eval_negatives_valid"] = True
    for user in sampled_users:
        seen = set(seen_items[offsets[user] : offsets[user + 1]].tolist())
        for candidates in (val_candidates[user], test_candidates[user]):
            negatives = candidates[1:]
            if len(set(negatives.tolist())) != len(negatives) or any(
                int(item) in seen for item in negatives
            ):
                checks["sampled_eval_negatives_valid"] = False
                break

    for model in ("mf", "two-tower"):
        with np.load(artifacts / f"{model}_embeddings.npz") as data:
            user_embedding = data["user_embedding"]
            item_embedding = data["item_embedding"]
        checks[f"{model}_embedding_shapes"] = [
            list(user_embedding.shape),
            list(item_embedding.shape),
        ]
        checks[f"{model}_embeddings_finite"] = bool(
            np.isfinite(user_embedding).all() and np.isfinite(item_embedding).all()
        )
        if model == "two-tower":
            checks["two_tower_embeddings_unit_norm"] = bool(
                np.allclose(
                    np.linalg.norm(user_embedding, axis=1), 1.0, atol=1e-4
                )
                and np.allclose(
                    np.linalg.norm(item_embedding, axis=1), 1.0, atol=1e-4
                )
            )
        with np.load(outputs / f"{model}_top100.npz") as data:
            top100 = data["items"]
            top100_scores = data["scores"]
        with np.load(outputs / f"{model}_top20_reranked.npz") as data:
            top20 = data["items"]
        checks[f"{model}_top100_shape"] = list(top100.shape)
        checks[f"{model}_top20_shape"] = list(top20.shape)
        sampled_top100 = top100[sampled_users]
        sampled_top20 = top20[sampled_users]
        checks[f"{model}_sampled_top100_unique"] = _row_unique(sampled_top100)
        checks[f"{model}_sampled_top20_unique"] = _row_unique(sampled_top20)
        checks[f"{model}_sampled_top100_scores_sorted"] = bool(
            np.all(np.diff(top100_scores[sampled_users], axis=1) <= 0)
        )
        checks[f"{model}_sampled_top20_in_top100"] = bool(
            np.all(
                np.any(
                    sampled_top20[:, :, None] == sampled_top100[:, None, :], axis=2
                )
            )
        )
        checks[f"{model}_sampled_top100_unseen_in_train"] = True
        with np.load(processed / "train_seen.npz") as train_seen:
            train_items, train_offsets = train_seen["items"], train_seen["offsets"]
        for user, recommendations in zip(sampled_users, sampled_top100):
            history = set(
                train_items[train_offsets[user] : train_offsets[user + 1]].tolist()
            )
            if any(int(item) in history for item in recommendations):
                checks[f"{model}_sampled_top100_unseen_in_train"] = False
                break

    with np.load(outputs / "two-tower_item_neighbors_top200.npz") as data:
        neighbors = data["items"]
        similarities = data["inner_product"]
    sampled_neighbors = neighbors[sampled_users % n_items]
    checks["item_neighbor_shape"] = list(neighbors.shape)
    checks["sampled_item_neighbors_exclude_self"] = bool(
        np.all(
            sampled_neighbors
            != (sampled_users % n_items)[:, None]
        )
    )
    checks["sampled_item_inner_products_sorted"] = bool(
        np.all(np.diff(similarities[sampled_users % n_items], axis=1) <= 0)
    )

    boolean_checks = [value for value in checks.values() if isinstance(value, bool)]
    report: dict[str, object] = {
        "ok": all(boolean_checks)
        and int(checks["minimum_item_interactions"]) >= int(stats["min_interactions"]),
        "sample_users": len(sampled_users),
        "checks": checks,
    }
    save_json(outputs / "validation_report.json", report)
    timestamped_message(f"Validation status: {report['ok']}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--sample-users", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = validate_outputs(args.root.resolve(), args.sample_users, args.seed)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

