from __future__ import annotations

# Dieses Modul prüft Daten, Embeddings und Empfehlungsdateien auf Konsistenz.
# 本模块检查数据、embedding 和推荐文件的一致性。
# This module checks data, embeddings, and recommendation files for consistency.
import argparse
from pathlib import Path

import numpy as np

from .utils import load_json, save_json, timestamped_message


def _row_unique(values: np.ndarray) -> bool:
    # Prüft zeilenweise, ob keine Artikel-ID doppelt vorkommt.
    # 逐行检查物品 ID 是否没有重复。
    # Checks row by row that no item ID is duplicated.
    ordered = np.sort(values, axis=1)
    return bool(np.all(np.diff(ordered, axis=1) != 0))


def validate_outputs(
    root: Path,
    sample_users: int = 5_000,
    seed: int = 2026,
) -> dict[str, object]:
    # Führt schnelle Vollprüfungen und reproduzierbare Benutzerstichproben aus.
    # 执行快速全量检查和可复现的用户抽样检查。
    # Runs fast full checks and reproducible user sampling checks.
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
    checks["scheme_b_feedback"] = bool(
        stats.get("feedback_scheme") == "three_level_explicit_negative"
    )

    # Prüft Split-Größen, Benutzergruppierung und zeitliche Grenzen.
    # 检查分割大小、用户分组和时间边界。
    # Checks split sizes, user grouping, and temporal boundaries.
    split_counts: dict[str, int] = {}
    boundaries: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    item_counts = np.zeros(n_items, dtype=np.int64)
    for split in ("train", "val", "test"):
        with np.load(processed / f"{split}.npz") as data:
            users, items, times = data["user"], data["item"], data["timestamp"]
            ratings = data["rating"]
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
        checks[f"{split}_contains_only_positives"] = bool(
            np.all(ratings >= float(stats["positive_threshold"]))
        )
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

    # Explizite Negative müssen niedrig bewertet und zeitlich im Training sichtbar sein.
    # 明确负样本必须是低评分，并且在时间上属于训练可见范围。
    # Explicit negatives must be low-rated and temporally visible in training.
    with np.load(processed / "train_explicit_negatives.npz") as explicit:
        explicit_items = explicit["items"]
        explicit_offsets = explicit["offsets"]
        explicit_ratings = explicit["ratings"]
    explicit_users = np.repeat(
        np.arange(n_users, dtype=np.int64), np.diff(explicit_offsets)
    )
    explicit_keys = explicit_users * n_items + explicit_items.astype(np.int64)
    train_seen_keys = np.load(processed / "train_seen_keys.npy", mmap_mode="r")
    checks["explicit_negative_offsets_valid"] = bool(
        len(explicit_offsets) == n_users + 1
        and explicit_offsets[0] == 0
        and explicit_offsets[-1] == len(explicit_items)
        and np.all(np.diff(explicit_offsets) >= 0)
    )
    checks["explicit_negative_ratings_valid"] = bool(
        np.all(explicit_ratings <= float(stats["negative_threshold"]))
    )
    checks["explicit_negatives_are_train_seen"] = bool(
        np.isin(explicit_keys, train_seen_keys).all()
    )
    checks["explicit_negative_count_matches_stats"] = bool(
        len(explicit_items) == stats["train_explicit_negative_interactions"]
    )

    # Benutzerprofile müssen gültige Wahrscheinlichkeitsverteilungen sein.
    # 用户画像必须是有效概率分布。
    # User profiles must be valid probability distributions.
    profiles = np.load(processed / "user_genre_profiles.npy", mmap_mode="r")
    item_genres = np.load(processed / "item_genres.npy", mmap_mode="r")
    checks["profile_shape"] = list(profiles.shape)
    checks["item_genre_shape"] = list(item_genres.shape)
    checks["profiles_normalized"] = bool(
        np.allclose(profiles.sum(axis=1), 1.0, atol=1e-5)
    )

    # Positive Kandidaten müssen die letzten Ereignisse sein; Negative bleiben ungesehen.
    # 正候选项必须是最后事件；负候选项必须未见。
    # Positive candidates must be the latest events; negatives remain unseen.
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

    # Kontrolliert Form, Endlichkeit und Listenqualität des MF-Modells.
    # 检查 MF 模型的形状、有限性和列表质量。
    # Checks the shape, finiteness, and list quality of the MF model.
    for model in ("mf",):
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
        # Empfohlene Artikel dürfen in der Trainingshistorie nicht vorkommen.
        # 推荐物品不得出现在训练历史中。
        # Recommended items must not occur in the training history.
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

    # Die gespeicherten Artikel-Nachbarn müssen sortiert sein und den Artikel ausschließen.
    # 保存的物品邻居必须已排序并排除物品本身。
    # Saved item neighbors must be sorted and exclude the item itself.
    with np.load(outputs / "mf_item_neighbors_top200.npz") as data:
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
    # Der Gesamtstatus ist nur wahr, wenn alle booleschen Prüfungen und 5-Core bestehen.
    # 只有所有布尔检查和 5-core 检查均通过时，总状态才为真。
    # Overall status is true only when all boolean checks and 5-core pass.
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
    # Stellt die Validierung als wiederholbares Kommandozeilenwerkzeug bereit.
    # 将验证提供为可重复运行的命令行工具。
    # Exposes validation as a repeatable command-line tool.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--sample-users", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()
    report = validate_outputs(args.root.resolve(), args.sample_users, args.seed)
    if not report["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    # Führt die Validierung nur bei direktem Modulaufruf aus.
    # 仅在模块被直接调用时执行验证。
    # Runs validation only when the module is invoked directly.
    main()
