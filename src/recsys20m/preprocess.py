from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .utils import save_json, timestamped_message


MOVIELENS_20M_URL = "https://files.grouplens.org/datasets/movielens/ml-20m.zip"


@dataclass(frozen=True)
class PreprocessConfig:
    raw_dir: Path
    output_dir: Path
    min_interactions: int = 5
    seed: int = 2026
    negatives: int = 100
    max_rows: int | None = None


def download_movielens(raw_dir: Path, force: bool = False) -> Path:
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "ml-20m.zip"
    extracted = raw_dir / "ml-20m"
    if extracted.joinpath("ratings.csv").exists() and not force:
        return extracted
    if force or not archive.exists():
        timestamped_message(f"Downloading {MOVIELENS_20M_URL}")
        urllib.request.urlretrieve(MOVIELENS_20M_URL, archive)
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
        zf.extractall(raw_dir)
    return extracted


def sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def iterative_k_core(
    users: np.ndarray,
    items: np.ndarray,
    minimum: int,
) -> np.ndarray:
    keep = np.ones(len(users), dtype=bool)
    iteration = 0
    while True:
        iteration += 1
        user_counts = np.bincount(users[keep], minlength=int(users.max()) + 1)
        item_counts = np.bincount(items[keep], minlength=int(items.max()) + 1)
        new_keep = keep & (user_counts[users] >= minimum) & (
            item_counts[items] >= minimum
        )
        timestamped_message(
            f"{minimum}-core iteration {iteration}: {new_keep.sum():,} interactions"
        )
        if np.array_equal(new_keep, keep):
            return keep
        keep = new_keep


def _split_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Largest practical chronological 80/10/10 split with non-empty holdouts."""
    train = np.floor(counts * 0.8).astype(np.int64)
    val = np.floor(counts * 0.1).astype(np.int64)
    test = counts.astype(np.int64) - train - val

    missing_val = val == 0
    can_take_test = missing_val & (test > 1)
    val[can_take_test] += 1
    test[can_take_test] -= 1

    take_train = missing_val & ~can_take_test
    val[take_train] += 1
    train[take_train] -= 1

    if np.any(train < 1) or np.any(val < 1) or np.any(test < 1):
        raise ValueError("Every user needs enough events for non-empty train/val/test.")
    return train, val, test


def _save_split(
    path: Path,
    users: np.ndarray,
    items: np.ndarray,
    ratings: np.ndarray,
    timestamps: np.ndarray,
    mask: np.ndarray,
) -> None:
    np.savez(
        path,
        user=users[mask].astype(np.int32, copy=False),
        item=items[mask].astype(np.int32, copy=False),
        rating=ratings[mask].astype(np.float32, copy=False),
        timestamp=timestamps[mask].astype(np.int64, copy=False),
    )


def _latest_positive(
    split_users: np.ndarray,
    split_items: np.ndarray,
    n_users: int,
) -> np.ndarray:
    counts = np.bincount(split_users, minlength=n_users)
    if np.any(counts == 0):
        raise ValueError("All retained users must have a positive in each holdout.")
    ends = np.cumsum(counts) - 1
    return split_items[ends]


def _eval_candidates(
    positives: np.ndarray,
    all_items_by_user: np.ndarray,
    offsets: np.ndarray,
    n_items: int,
    n_negative: int,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n_users = len(positives)
    candidates = np.empty((n_users, n_negative + 1), dtype=np.int32)
    candidates[:, 0] = positives

    for user in range(n_users):
        seen = set(
            all_items_by_user[offsets[user] : offsets[user + 1]].astype(int).tolist()
        )
        available = n_items - len(seen)
        if available < n_negative:
            raise ValueError(
                f"User {user} has only {available} unseen items; "
                f"cannot draw {n_negative} unique negatives."
            )
        negatives: set[int] = set()
        while len(negatives) < n_negative:
            need = n_negative - len(negatives)
            draws = rng.integers(0, n_items, size=max(need * 2, 32))
            negatives.update(int(x) for x in draws if int(x) not in seen)
        candidates[user, 1:] = np.fromiter(
            list(negatives)[:n_negative], dtype=np.int32, count=n_negative
        )
        if user and user % 25_000 == 0:
            timestamped_message(f"Sampled evaluation negatives: {user:,}/{n_users:,}")
    return candidates


def _genre_features(
    item_genre_matrix: np.ndarray,
    train_users: np.ndarray,
    train_items: np.ndarray,
    n_users: int,
) -> np.ndarray:
    # A multi-genre item's mass is split evenly, so every interaction contributes 1.
    denom = item_genre_matrix.sum(axis=1, keepdims=True)
    item_distribution = item_genre_matrix / np.maximum(denom, 1.0)
    profiles = np.empty((n_users, item_genre_matrix.shape[1]), dtype=np.float32)
    for genre_idx in range(item_genre_matrix.shape[1]):
        weights = item_distribution[train_items, genre_idx]
        profiles[:, genre_idx] = np.bincount(
            train_users, weights=weights, minlength=n_users
        ).astype(np.float32)
    profiles /= np.maximum(profiles.sum(axis=1, keepdims=True), 1e-12)
    return profiles


def preprocess(config: PreprocessConfig) -> dict[str, object]:
    source = config.raw_dir / "ml-20m"
    ratings_path = source / "ratings.csv"
    movies_path = source / "movies.csv"
    if not ratings_path.exists() or not movies_path.exists():
        source = download_movielens(config.raw_dir)
        ratings_path = source / "ratings.csv"
        movies_path = source / "movies.csv"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamped_message("Reading movies and ratings")
    movies = pd.read_csv(movies_path)
    ratings = pd.read_csv(
        ratings_path,
        nrows=config.max_rows,
        dtype={
            "userId": np.int32,
            "movieId": np.int32,
            "rating": np.float32,
            "timestamp": np.int64,
        },
    )

    raw_user_ids, users = np.unique(
        ratings["userId"].to_numpy(), return_inverse=True
    )
    movie_index = pd.Index(movies["movieId"].to_numpy())
    items = movie_index.get_indexer(ratings["movieId"].to_numpy())
    if np.any(items < 0):
        raise ValueError("ratings.csv contains movie IDs missing from movies.csv")
    users = users.astype(np.int32, copy=False)
    items = items.astype(np.int32, copy=False)
    values = ratings["rating"].to_numpy(dtype=np.float32, copy=False)
    times = ratings["timestamp"].to_numpy(dtype=np.int64, copy=False)
    raw_interactions = len(users)
    del ratings

    keep = iterative_k_core(users, items, config.min_interactions)
    users, items, values, times = (
        users[keep],
        items[keep],
        values[keep],
        times[keep],
    )

    active_old_users, users = np.unique(users, return_inverse=True)
    active_old_items, items = np.unique(items, return_inverse=True)
    users = users.astype(np.int32, copy=False)
    items = items.astype(np.int32, copy=False)
    n_users = len(active_old_users)
    n_items = len(active_old_items)
    retained_user_ids = raw_user_ids[active_old_users]
    retained_movies = movies.iloc[active_old_items].reset_index(drop=True).copy()
    retained_movies.insert(0, "item_idx", np.arange(n_items, dtype=np.int32))

    timestamped_message("Sorting each user's history chronologically")
    order = np.lexsort((times, users))
    users, items, values, times = (
        users[order],
        items[order],
        values[order],
        times[order],
    )
    counts = np.bincount(users, minlength=n_users)
    train_counts, val_counts, test_counts = _split_counts(counts)
    starts = np.cumsum(counts) - counts
    positions = np.arange(len(users), dtype=np.int64) - np.repeat(starts, counts)
    train_end = train_counts[users]
    val_end = (train_counts + val_counts)[users]
    train_mask = positions < train_end
    val_mask = (positions >= train_end) & (positions < val_end)
    test_mask = positions >= val_end

    timestamped_message("Writing split arrays")
    _save_split(
        config.output_dir / "train.npz",
        users,
        items,
        values,
        times,
        train_mask,
    )
    _save_split(
        config.output_dir / "val.npz",
        users,
        items,
        values,
        times,
        val_mask,
    )
    _save_split(
        config.output_dir / "test.npz",
        users,
        items,
        values,
        times,
        test_mask,
    )

    pd.DataFrame(
        {
            "user_idx": np.arange(n_users, dtype=np.int32),
            "userId": retained_user_ids,
        }
    ).to_csv(config.output_dir / "users.csv", index=False)
    retained_movies.to_csv(config.output_dir / "items.csv", index=False)

    genres = sorted(
        {
            genre
            for joined in retained_movies["genres"].fillna("(no genres listed)")
            for genre in joined.split("|")
        }
    )
    genre_to_idx = {genre: idx for idx, genre in enumerate(genres)}
    item_genres = np.zeros((n_items, len(genres)), dtype=np.float32)
    for item_idx, joined in enumerate(
        retained_movies["genres"].fillna("(no genres listed)")
    ):
        for genre in joined.split("|"):
            item_genres[item_idx, genre_to_idx[genre]] = 1.0
    np.save(config.output_dir / "item_genres.npy", item_genres)
    save_json(config.output_dir / "genres.json", {"genres": genres})

    train_users, train_items = users[train_mask], items[train_mask]
    timestamped_message("Building train-only user genre profiles")
    profiles = _genre_features(item_genres, train_users, train_items, n_users)
    np.save(config.output_dir / "user_genre_profiles.npy", profiles)

    # Histories stay grouped by user. Keys are sorted for fast rejection sampling.
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    np.savez(
        config.output_dir / "all_seen.npz",
        items=items.astype(np.int32, copy=False),
        offsets=offsets,
    )
    train_offsets = np.concatenate(([0], np.cumsum(train_counts))).astype(np.int64)
    np.savez(
        config.output_dir / "train_seen.npz",
        items=train_items.astype(np.int32, copy=False),
        offsets=train_offsets,
    )
    train_keys = np.sort(
        train_users.astype(np.int64) * n_items + train_items.astype(np.int64)
    )
    np.save(config.output_dir / "train_seen_keys.npy", train_keys)

    val_positive = _latest_positive(users[val_mask], items[val_mask], n_users)
    test_positive = _latest_positive(users[test_mask], items[test_mask], n_users)
    timestamped_message("Sampling validation candidates (1 positive + negatives)")
    val_candidates = _eval_candidates(
        val_positive,
        items,
        offsets,
        n_items,
        config.negatives,
        config.seed + 1,
    )
    timestamped_message("Sampling test candidates (1 positive + negatives)")
    test_candidates = _eval_candidates(
        test_positive,
        items,
        offsets,
        n_items,
        config.negatives,
        config.seed + 2,
    )
    np.savez(
        config.output_dir / "eval_candidates.npz",
        val=val_candidates,
        test=test_candidates,
        positive_column=np.array(0, dtype=np.int32),
    )

    stats: dict[str, object] = {
        "seed": config.seed,
        "min_interactions": config.min_interactions,
        "negative_samples": config.negatives,
        "raw_interactions": raw_interactions,
        "core_interactions": len(users),
        "n_users": n_users,
        "n_items": n_items,
        "n_genres": len(genres),
        "train_interactions": int(train_mask.sum()),
        "val_interactions": int(val_mask.sum()),
        "test_interactions": int(test_mask.sum()),
        "split_ratio_actual": {
            "train": float(train_mask.mean()),
            "val": float(val_mask.mean()),
            "test": float(test_mask.mean()),
        },
        "timestamp_min": int(times.min()),
        "timestamp_max": int(times.max()),
        "source": str(source),
    }
    save_json(config.output_dir / "stats.json", stats)
    timestamped_message(json.dumps(stats, ensure_ascii=False))
    return stats


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument("--negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-rows", type=int)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    preprocess(
        PreprocessConfig(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            min_interactions=args.min_interactions,
            negatives=args.negatives,
            seed=args.seed,
            max_rows=args.max_rows,
        )
    )


if __name__ == "__main__":
    main()
