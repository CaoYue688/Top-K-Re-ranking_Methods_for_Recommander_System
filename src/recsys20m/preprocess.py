from __future__ import annotations

# Dieses Modul lädt MovieLens 20M und erzeugt reproduzierbare Trainingsdaten.
# 本模块下载 MovieLens 20M 并生成可复现的训练数据。
# This module downloads MovieLens 20M and creates reproducible training data.
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
    # Zentrale Einstellungen für Download, Filterung, Sampling und Ausgabe.
    # 下载、过滤、采样和输出的核心配置。
    # Central settings for download, filtering, sampling, and output.
    raw_dir: Path
    output_dir: Path
    min_interactions: int = 5
    positive_threshold: float = 4.0
    negative_threshold: float = 2.0
    seed: int = 2026
    negatives: int = 100
    max_rows: int | None = None
    core_on_train_only: bool = False


def download_movielens(raw_dir: Path, force: bool = False) -> Path:
    # Lädt das offizielle Archiv und entpackt es nach einer ZIP-Prüfung.
    # 下载官方压缩包，检查 ZIP 后再解压。
    # Downloads the official archive and extracts it after a ZIP check.
    raw_dir.mkdir(parents=True, exist_ok=True)
    archive = raw_dir / "ml-20m.zip"
    extracted = raw_dir / "ml-20m"
    if extracted.joinpath("ratings.csv").exists() and not force:
        return extracted
    if force or not archive.exists():
        timestamped_message(f"Downloading {MOVIELENS_20M_URL}")
        urllib.request.urlretrieve(MOVIELENS_20M_URL, archive)
    # testzip erkennt beschädigte Dateien, bevor Daten verarbeitet werden.
    # testzip 在数据处理前检测损坏文件。
    # testzip detects corrupted files before data processing.
    with zipfile.ZipFile(archive) as zf:
        bad = zf.testzip()
        if bad is not None:
            raise RuntimeError(f"Corrupt ZIP member: {bad}")
        zf.extractall(raw_dir)
    return extracted


def sha256(path: Path, chunk_bytes: int = 1 << 20) -> str:
    # Berechnet den SHA-256-Hash blockweise, ohne die ganze Datei zu laden.
    # 分块计算 SHA-256，无需加载整个文件。
    # Computes SHA-256 in chunks without loading the entire file.
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
    # Entfernt Benutzer und Artikel iterativ, bis beide mindestens k Interaktionen haben.
    # 迭代删除用户和物品，直到两者都至少有 k 次交互。
    # Iteratively removes users and items until both have at least k interactions.
    keep = np.ones(len(users), dtype=bool)
    iteration = 0
    while True:
        iteration += 1
        # Die Häufigkeiten werden nur auf den aktuell gültigen Zeilen berechnet.
        # 频数只在当前有效行上计算。
        # Counts are computed only on currently valid rows.
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


def iterative_train_k_core(
    users: np.ndarray,
    items: np.ndarray,
    timestamps: np.ndarray,
    minimum: int,
) -> np.ndarray:
    """Iteratively require k interactions in the chronological train partition."""
    keep = np.ones(len(users), dtype=bool)
    n_users = int(users.max()) + 1
    n_items = int(items.max()) + 1
    iteration = 0
    while True:
        iteration += 1
        retained_indices = np.flatnonzero(keep)
        order = np.lexsort(
            (timestamps[retained_indices], users[retained_indices])
        )
        sorted_indices = retained_indices[order]
        sorted_users = users[sorted_indices]
        counts = np.bincount(sorted_users, minlength=n_users)

        train_counts = np.zeros(n_users, dtype=np.int64)
        splittable = counts >= 3
        if splittable.any():
            train_counts[splittable], _, _ = _split_counts(counts[splittable])
        starts = np.cumsum(counts) - counts
        positions = np.arange(len(sorted_indices), dtype=np.int64) - np.repeat(
            starts, counts
        )
        train_sorted = positions < train_counts[sorted_users]
        train_indices = sorted_indices[train_sorted]
        train_user_counts = np.bincount(
            users[train_indices], minlength=n_users
        )
        train_item_counts = np.bincount(
            items[train_indices], minlength=n_items
        )
        active_users = train_user_counts >= minimum
        active_items = train_item_counts >= minimum
        new_keep = keep & active_users[users] & active_items[items]
        timestamped_message(
            f"train-only {minimum}-core iteration {iteration}: "
            f"{new_keep.sum():,} positive interactions"
        )
        if np.array_equal(new_keep, keep):
            return keep
        keep = new_keep


def _split_counts(counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Largest practical chronological 80/10/10 split with non-empty holdouts."""
    # Zuerst wird die gewünschte 80/10/10-Aufteilung pro Benutzer berechnet.
    # 先为每个用户计算目标 80/10/10 切分。
    # First computes the desired 80/10/10 split per user.
    train = np.floor(counts * 0.8).astype(np.int64)
    val = np.floor(counts * 0.1).astype(np.int64)
    test = counts.astype(np.int64) - train - val

    missing_val = val == 0
    # Bei kurzen Historien bleibt mindestens ein Ereignis für Validierung und Test.
    # 对于较短历史，验证集和测试集各保留至少一个事件。
    # Keeps at least one event for validation and test in short histories.
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
    # Schreibt einen Datensplit mit kompakten numerischen Datentypen auf die Festplatte.
    # 使用紧凑数值类型将数据分割写入磁盘。
    # Writes a data split to disk using compact numeric types.
    np.savez(
        path,
        user=users[mask].astype(np.int32, copy=False),
        item=items[mask].astype(np.int32, copy=False),
        rating=ratings[mask].astype(np.float32, copy=False),
        timestamp=timestamps[mask].astype(np.int64, copy=False),
    )


def _save_ragged_items(
    path: Path,
    users: np.ndarray,
    items: np.ndarray,
    n_users: int,
    ratings: np.ndarray | None = None,
) -> np.ndarray:
    # Speichert benutzerspezifische Listen kompakt als Werte plus Offset-Tabelle.
    # 将用户专属列表紧凑保存为数值数组和偏移表。
    # Stores per-user lists compactly as values plus an offset table.
    counts = np.bincount(users, minlength=n_users)
    offsets = np.concatenate(([0], np.cumsum(counts))).astype(np.int64)
    payload: dict[str, np.ndarray] = {
        "items": items.astype(np.int32, copy=False),
        "offsets": offsets,
    }
    if ratings is not None:
        payload["ratings"] = ratings.astype(np.float32, copy=False)
    np.savez(path, **payload)
    return offsets


def _latest_positive(
    split_users: np.ndarray,
    split_items: np.ndarray,
    n_users: int,
) -> np.ndarray:
    # Wählt je Benutzer die zeitlich letzte positive Interaktion des Splits.
    # 选择每个用户在该分割中时间最晚的正交互。
    # Selects each user's latest positive interaction in the split.
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
    # Erzeugt je Benutzer eine positive Position und eindeutig ungesehene Negative.
    # 为每个用户生成一个正样本位置和不重复的未见负样本。
    # Creates one positive position and unique unseen negatives per user.
    rng = np.random.default_rng(seed)
    n_users = len(positives)
    candidates = np.empty((n_users, n_negative + 1), dtype=np.int32)
    candidates[:, 0] = positives

    for user in range(n_users):
        # Die Menge beschleunigt die Prüfung, ob ein zufälliger Artikel schon gesehen wurde.
        # 集合可加速检查随机物品是否已被见过。
        # The set speeds up checks for whether a random item was already seen.
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
        # Es wird so lange nachgezogen, bis 100 verschiedene gültige Negative vorliegen.
        # 持续重采样，直到得到 100 个不同的有效负样本。
        # Resamples until 100 distinct valid negatives are available.
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
    # Bei mehreren Genres wird die Masse geteilt; jede Interaktion trägt insgesamt 1 bei.
    # 多类型物品平分权重；每次交互总贡献为 1。
    # Multi-genre items split their mass; each interaction contributes 1 in total.
    denom = item_genre_matrix.sum(axis=1, keepdims=True)
    item_distribution = item_genre_matrix / np.maximum(denom, 1.0)
    profiles = np.empty((n_users, item_genre_matrix.shape[1]), dtype=np.float32)
    # bincount aggregiert die Genre-Gewichte effizient für jeden Benutzer.
    # bincount 高效汇总每个用户的类型权重。
    # bincount efficiently aggregates genre weights for each user.
    for genre_idx in range(item_genre_matrix.shape[1]):
        weights = item_distribution[train_items, genre_idx]
        profiles[:, genre_idx] = np.bincount(
            train_users, weights=weights, minlength=n_users
        ).astype(np.float32)
    profiles /= np.maximum(profiles.sum(axis=1, keepdims=True), 1e-12)
    return profiles


def preprocess(config: PreprocessConfig) -> dict[str, object]:
    # Führt Download, 5-Core-Filterung, Zeitaufteilung und Feature Engineering aus.
    # 执行下载、5-core 过滤、时间切分和特征工程。
    # Runs download, 5-core filtering, temporal splitting, and feature engineering.
    if config.negative_threshold >= config.positive_threshold:
        raise ValueError("negative_threshold must be lower than positive_threshold.")
    source = config.raw_dir / "ml-20m"
    ratings_path = source / "ratings.csv"
    movies_path = source / "movies.csv"
    if not ratings_path.exists() or not movies_path.exists():
        source = download_movielens(config.raw_dir)
        ratings_path = source / "ratings.csv"
        movies_path = source / "movies.csv"

    config.output_dir.mkdir(parents=True, exist_ok=True)
    timestamped_message("Reading movies and ratings")
    # Explizite Datentypen halten den Speicherverbrauch bei 20 Millionen Zeilen klein.
    # 显式数据类型可降低 2000 万行数据的内存占用。
    # Explicit data types keep memory usage low for 20 million rows.
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
    raw_interactions = len(ratings)
    raw_user_ids, all_users = np.unique(
        ratings["userId"].to_numpy(), return_inverse=True
    )
    # Externe MovieLens-IDs werden auf kompakte, nullbasierte Indizes abgebildet.
    # 将外部 MovieLens ID 映射为紧凑的从零开始索引。
    # Maps external MovieLens IDs to compact zero-based indices.
    movie_index = pd.Index(movies["movieId"].to_numpy())
    all_items = movie_index.get_indexer(ratings["movieId"].to_numpy())
    if np.any(all_items < 0):
        raise ValueError("ratings.csv contains movie IDs missing from movies.csv")
    all_users = all_users.astype(np.int32, copy=False)
    all_items = all_items.astype(np.int32, copy=False)
    all_values = ratings["rating"].to_numpy(dtype=np.float32, copy=False)
    all_times = ratings["timestamp"].to_numpy(dtype=np.int64, copy=False)
    del ratings

    # Nur positive Ereignisse definieren den 5-Core und die überwachte Zielmenge.
    # 只有正向事件用于定义 5-core 和监督学习目标集合。
    # Only positive events define the 5-core and supervised target set.
    positive_rows = all_values >= config.positive_threshold
    threshold_interactions = int(positive_rows.sum())
    if threshold_interactions == 0:
        raise ValueError("No positive interactions remain after thresholding.")
    positive_users = all_users[positive_rows]
    positive_items = all_items[positive_rows]
    if config.core_on_train_only:
        positive_keep = iterative_train_k_core(
            positive_users,
            positive_items,
            all_times[positive_rows],
            config.min_interactions,
        )
    else:
        positive_keep = iterative_k_core(
            positive_users, positive_items, config.min_interactions
        )
    if not positive_keep.any():
        raise ValueError("No positive interactions remain after k-core filtering.")

    # Nach der Filterung werden Benutzer und Artikel ohne Lücken neu indexiert.
    # 过滤后对用户和物品进行无空洞重编号。
    # Reindexes users and items contiguously after filtering.
    active_old_users = np.unique(positive_users[positive_keep])
    active_old_items = np.unique(positive_items[positive_keep])
    n_users = len(active_old_users)
    n_items = len(active_old_items)
    retained_user_ids = raw_user_ids[active_old_users]
    retained_movies = movies.iloc[active_old_items].reset_index(drop=True).copy()
    retained_movies.insert(0, "item_idx", np.arange(n_items, dtype=np.int32))

    user_reindex = np.full(len(raw_user_ids), -1, dtype=np.int32)
    item_reindex = np.full(len(movies), -1, dtype=np.int32)
    user_reindex[active_old_users] = np.arange(n_users, dtype=np.int32)
    item_reindex[active_old_items] = np.arange(n_items, dtype=np.int32)
    mapped_users = user_reindex[all_users]
    mapped_items = item_reindex[all_items]
    retained_rows = (mapped_users >= 0) & (mapped_items >= 0)
    all_users = mapped_users[retained_rows]
    all_items = mapped_items[retained_rows]
    all_values = all_values[retained_rows]
    all_times = all_times[retained_rows]

    # Niedrige und neutrale Ratings bleiben erhalten, werden aber nie positive Ziele.
    # 保留低分和中性评分，但它们绝不会成为正向训练目标。
    # Low and neutral ratings are retained but never become positive targets.
    positive_rows = all_values >= config.positive_threshold
    users = all_users[positive_rows]
    items = all_items[positive_rows]
    values = all_values[positive_rows]
    times = all_times[positive_rows]

    timestamped_message("Sorting each user's history chronologically")
    # lexsort sortiert primär nach Benutzer und innerhalb des Benutzers nach Zeit.
    # lexsort 先按用户排序，再在用户内按时间排序。
    # lexsort sorts primarily by user and then by time within each user.
    order = np.lexsort((times, users))
    users, items, values, times = (
        users[order],
        items[order],
        values[order],
        times[order],
    )
    counts = np.bincount(users, minlength=n_users)
    train_counts, val_counts, test_counts = _split_counts(counts)
    # Die lokale Position innerhalb einer Benutzerhistorie bestimmt den Datensplit.
    # 交互在用户历史中的局部位置决定其数据分割。
    # The local position in a user's history determines the data split.
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
    # Jeder Artikel erhält einen Multi-Hot-Vektor über alle vorkommenden Genres.
    # 每个物品都获得一个覆盖所有出现类型的 multi-hot 向量。
    # Each item receives a multi-hot vector over all observed genres.
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
    # Nur Trainingsdaten werden verwendet, damit keine Information aus der Zukunft leakt.
    # 只使用训练数据，以避免未来信息泄漏。
    # Uses only training data to prevent future-information leakage.
    profiles = _genre_features(item_genres, train_users, train_items, n_users)
    np.save(config.output_dir / "user_genre_profiles.npy", profiles)

    # Die letzten positiven Zeitpunkte begrenzen, welche Ratings im Training sichtbar sind.
    # 最后一个正向时间点限定训练阶段可以看到哪些评分。
    # The last positive timestamps bound which ratings are visible during training.
    train_cutoffs = times[starts + train_counts - 1]
    train_val_cutoffs = times[starts + train_counts + val_counts - 1]

    # Alle erhaltenen Ratings werden separat chronologisch sortiert.
    # 将所有保留的评分单独按时间顺序排序。
    # All retained ratings are sorted chronologically in a separate history.
    all_order = np.lexsort((all_times, all_users))
    all_users, all_items, all_values, all_times = (
        all_users[all_order],
        all_items[all_order],
        all_values[all_order],
        all_times[all_order],
    )
    train_rated_mask = all_times <= train_cutoffs[all_users]
    train_val_rated_mask = all_times <= train_val_cutoffs[all_users]

    # Alle jemals bewerteten Artikel werden bei der Evaluation als gesehen behandelt.
    # 评估时将用户曾经评分过的所有物品视为已见。
    # Every item ever rated by a user is treated as seen during evaluation.
    offsets = _save_ragged_items(
        config.output_dir / "all_seen.npz",
        all_users,
        all_items,
        n_users,
    )
    _save_ragged_items(
        config.output_dir / "train_seen.npz",
        all_users[train_rated_mask],
        all_items[train_rated_mask],
        n_users,
    )
    # Für den finalen Test gelten alle bis zum Validierungsende bewerteten Artikel als gesehen.
    # 最终测试将验证期结束前评分过的所有物品视为已见。
    # Final testing treats every item rated through validation as seen.
    _save_ragged_items(
        config.output_dir / "train_val_seen.npz",
        all_users[train_val_rated_mask],
        all_items[train_val_rated_mask],
        n_users,
    )

    # Explizite Negative stammen ausschließlich aus dem zeitlich zulässigen Trainingsfenster.
    # 明确负样本仅来自时间上允许使用的训练窗口。
    # Explicit negatives come only from the temporally valid training window.
    explicit_negative_mask = (
        train_rated_mask & (all_values <= config.negative_threshold)
    )
    _save_ragged_items(
        config.output_dir / "train_explicit_negatives.npz",
        all_users[explicit_negative_mask],
        all_items[explicit_negative_mask],
        n_users,
        all_values[explicit_negative_mask],
    )
    train_keys = np.unique(
        all_users[train_rated_mask].astype(np.int64) * n_items
        + all_items[train_rated_mask].astype(np.int64)
    )
    np.save(config.output_dir / "train_seen_keys.npy", train_keys)

    # Validierungs- und Testkandidaten werden mit verschiedenen Seeds erzeugt.
    # 使用不同随机种子生成验证和测试候选集。
    # Generates validation and test candidates with different seeds.
    val_positive = _latest_positive(users[val_mask], items[val_mask], n_users)
    test_positive = _latest_positive(users[test_mask], items[test_mask], n_users)
    timestamped_message("Sampling validation candidates (1 positive + negatives)")
    val_candidates = _eval_candidates(
        val_positive,
        all_items,
        offsets,
        n_items,
        config.negatives,
        config.seed + 1,
    )
    timestamped_message("Sampling test candidates (1 positive + negatives)")
    test_candidates = _eval_candidates(
        test_positive,
        all_items,
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

    # Die Statistikdatei dokumentiert Umfang und tatsächlich erreichte Split-Verhältnisse.
    # 统计文件记录数据规模和实际达到的切分比例。
    # The statistics file records data size and achieved split ratios.
    stats: dict[str, object] = {
        "feedback_scheme": "three_level_explicit_negative",
        "seed": config.seed,
        "min_interactions": config.min_interactions,
        "core_basis": (
            "chronological_training_partition"
            if config.core_on_train_only
            else "full_positive_history"
        ),
        "positive_threshold": config.positive_threshold,
        "negative_threshold": config.negative_threshold,
        "negative_samples": config.negatives,
        "raw_interactions": raw_interactions,
        "threshold_interactions": threshold_interactions,
        "core_interactions": len(users),
        "retained_ratings": len(all_users),
        "explicit_negative_interactions": int(
            (all_values <= config.negative_threshold).sum()
        ),
        "neutral_interactions": int(
            (
                (all_values > config.negative_threshold)
                & (all_values < config.positive_threshold)
            ).sum()
        ),
        "train_explicit_negative_interactions": int(
            explicit_negative_mask.sum()
        ),
        "train_rated_interactions": int(train_rated_mask.sum()),
        "train_val_rated_interactions": int(train_val_rated_mask.sum()),
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
        "timestamp_min": int(all_times.min()),
        "timestamp_max": int(all_times.max()),
        "source": str(source),
    }
    save_json(config.output_dir / "stats.json", stats)
    timestamped_message(json.dumps(stats, ensure_ascii=False))
    return stats


def build_parser() -> argparse.ArgumentParser:
    # Definiert die Kommandozeilenoptionen für die Vorverarbeitung.
    # 定义预处理命令行选项。
    # Defines command-line options for preprocessing.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--min-interactions", type=int, default=5)
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=4.0,
        help="Ratings greater than or equal to this value are positive.",
    )
    parser.add_argument(
        "--negative-threshold",
        type=float,
        default=2.0,
        help="Ratings less than or equal to this value are explicit negatives.",
    )
    parser.add_argument("--negatives", type=int, default=100)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--max-rows", type=int)
    parser.add_argument(
        "--core-on-train-only",
        action="store_true",
        help="Apply the iterative user/item k-core only to chronological train edges.",
    )
    return parser


def main() -> None:
    # Übersetzt Kommandozeilenargumente in die unveränderliche Konfiguration.
    # 将命令行参数转换为不可变配置。
    # Translates command-line arguments into the immutable configuration.
    args = build_parser().parse_args()
    preprocess(
        PreprocessConfig(
            raw_dir=args.raw_dir,
            output_dir=args.output_dir,
            min_interactions=args.min_interactions,
            positive_threshold=args.positive_threshold,
            negative_threshold=args.negative_threshold,
            negatives=args.negatives,
            seed=args.seed,
            max_rows=args.max_rows,
            core_on_train_only=args.core_on_train_only,
        )
    )


if __name__ == "__main__":
    # Startet die Vorverarbeitung nur bei direktem Modulaufruf.
    # 仅在模块被直接调用时启动预处理。
    # Starts preprocessing only when the module is invoked directly.
    main()
