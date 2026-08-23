from __future__ import annotations

# Dieses Modul erzeugt Top-K-Empfehlungen und Artikel-Artikel-Nachbarschaften.
# 本模块生成 Top-K 推荐和物品-物品邻域。
# This module generates Top-K recommendations and item-item neighborhoods.
import argparse
from pathlib import Path

import numpy as np

from .utils import load_json, timestamped_message


def _top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    # argpartition findet die besten k Werte schneller als eine vollständige Sortierung.
    # argpartition 比完整排序更快地找到最佳 k 个值。
    # argpartition finds the best k values faster than a full sort.
    if k >= scores.shape[1]:
        indices = np.argsort(-scores, axis=1)[:, :k]
    else:
        partition = np.argpartition(scores, -k, axis=1)[:, -k:]
        partition_scores = np.take_along_axis(scores, partition, axis=1)
        # Nur die kleine k-Auswahl wird anschließend exakt absteigend sortiert.
        # 随后只对小型 k 候选集做精确降序排序。
        # Only the small k selection is then sorted exactly in descending order.
        order = np.argsort(-partition_scores, axis=1)
        indices = np.take_along_axis(partition, order, axis=1)
    values = np.take_along_axis(scores, indices, axis=1)
    return indices.astype(np.int32), values.astype(np.float32)


def retrieve_top_k(
    processed_dir: Path,
    embedding_path: Path,
    output_path: Path,
    k: int = 100,
    user_batch_size: int = 512,
    seen_file: str = "train_seen.npz",
    device: str = "cpu",
) -> Path:
    # Berechnet für jeden Benutzer die besten ungesehenen Artikel aus dem Gesamtkatalog.
    # 从完整目录中为每个用户计算最佳未见物品。
    # Computes the best unseen items from the full catalog for each user.
    with np.load(embedding_path) as data:
        user_embeddings = data["user_embedding"]
        item_embeddings = data["item_embedding"]
        item_bias = data["item_bias"] if "item_bias" in data else None
    # Je nach Evaluationsphase wird nur Train oder Train+Validation ausgeschlossen.
    # 根据评估阶段，屏蔽训练集或训练集+验证集。
    # Excludes either Train or Train+Validation depending on the evaluation stage.
    with np.load(processed_dir / seen_file) as seen:
        seen_items = seen["items"]
        offsets = seen["offsets"]
    n_users, n_items = len(user_embeddings), len(item_embeddings)
    if k >= n_items:
        raise ValueError(f"k={k} must be smaller than n_items={n_items}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_items = np.empty((n_users, k), dtype=np.int32)
    all_scores = np.empty((n_users, k), dtype=np.float32)
    item_transpose = item_embeddings.T

    if device != "cpu":
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError(f"Requested retrieval device {device!r} is unavailable.")
        torch_device = torch.device(device)
        item_tensor = torch.from_numpy(item_embeddings).to(torch_device)
        bias_tensor = (
            torch.from_numpy(item_bias).to(torch_device)
            if item_bias is not None
            else None
        )
        for start in range(0, n_users, user_batch_size):
            end = min(start + user_batch_size, n_users)
            user_tensor = torch.from_numpy(user_embeddings[start:end]).to(torch_device)
            scores_tensor = user_tensor @ item_tensor.T
            if bias_tensor is not None:
                scores_tensor += bias_tensor[None, :]
            counts = offsets[start + 1 : end + 1] - offsets[start:end]
            mask_rows = np.repeat(np.arange(end - start, dtype=np.int64), counts)
            mask_items = seen_items[offsets[start] : offsets[end]].astype(
                np.int64, copy=False
            )
            scores_tensor[
                torch.from_numpy(mask_rows).to(torch_device),
                torch.from_numpy(mask_items).to(torch_device),
            ] = -torch.inf
            values, indices = torch.topk(scores_tensor, k=k, dim=1, sorted=True)
            all_items[start:end] = indices.cpu().numpy().astype(np.int32)
            all_scores[start:end] = values.cpu().numpy().astype(np.float32)
            if end % 10_000 < user_batch_size or end == n_users:
                timestamped_message(
                    f"Retrieved Top-{k} on {device}: {end:,}/{n_users:,} users"
                )
        np.savez(output_path, items=all_items, scores=all_scores)
        return output_path

    # Die große Benutzer-Artikel-Matrix wird blockweise berechnet.
    # 分块计算大型用户-物品矩阵。
    # Computes the large user-item matrix in blocks.
    for start in range(0, n_users, user_batch_size):
        end = min(start + user_batch_size, n_users)
        scores = user_embeddings[start:end] @ item_transpose
        if item_bias is not None:
            scores += item_bias[None, :]
        # Bereits im Training gesehene Artikel dürfen nicht erneut empfohlen werden.
        # 训练中已见的物品不能被再次推荐。
        # Items seen during training must not be recommended again.
        for local_user, user in enumerate(range(start, end)):
            scores[
                local_user, seen_items[offsets[user] : offsets[user + 1]]
            ] = -np.inf
        batch_items, batch_scores = _top_k(scores, k)
        all_items[start:end] = batch_items
        all_scores[start:end] = batch_scores
        if end % 10_000 < user_batch_size or end == n_users:
            timestamped_message(f"Retrieved Top-{k}: {end:,}/{n_users:,} users")

    np.savez(output_path, items=all_items, scores=all_scores)
    return output_path


def item_inner_product_neighbors(
    embedding_path: Path,
    output_path: Path,
    top_k: int = 200,
    block_size: int = 512,
    normalize: bool = True,
) -> Path:
    # Sucht für jeden Artikel die ähnlichsten Artikel anhand des inneren Produkts.
    # 根据内积为每个物品查找最相似物品。
    # Finds the most similar items for each item using inner products.
    with np.load(embedding_path) as data:
        item_embeddings = data["item_embedding"].astype(np.float32)
    if normalize:
        # Nach L2-Normalisierung entspricht das innere Produkt der Kosinusähnlichkeit.
        # L2 归一化后，内积等于余弦相似度。
        # After L2 normalization, the inner product equals cosine similarity.
        item_embeddings /= np.maximum(
            np.linalg.norm(item_embeddings, axis=1, keepdims=True), 1e-12
        )
    n_items = len(item_embeddings)
    if top_k >= n_items:
        raise ValueError("top_k must be smaller than the number of items")
    neighbor_items = np.empty((n_items, top_k), dtype=np.int32)
    neighbor_scores = np.empty((n_items, top_k), dtype=np.float32)
    transpose = item_embeddings.T
    # Blockweise Multiplikation vermeidet eine dauerhafte dichte n_items²-Matrix.
    # 分块乘法避免持久保存稠密 n_items² 矩阵。
    # Block multiplication avoids a persistent dense n_items² matrix.
    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        scores = item_embeddings[start:end] @ transpose
        rows = np.arange(end - start)
        # Ein Artikel darf nicht sein eigener nächster Nachbar sein.
        # 物品不能成为自己的最近邻居。
        # An item cannot be its own nearest neighbor.
        scores[rows, np.arange(start, end)] = -np.inf
        items, values = _top_k(scores, top_k)
        neighbor_items[start:end] = items
        neighbor_scores[start:end] = values
        if end % 5_000 < block_size or end == n_items:
            timestamped_message(
                f"Computed item inner-product neighbors: {end:,}/{n_items:,}"
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        items=neighbor_items,
        inner_product=neighbor_scores,
        normalized=np.array(normalize),
    )
    return output_path


def main() -> None:
    # Bietet Empfehlungen und Artikelähnlichkeit als getrennte Unterbefehle an.
    # 将推荐和物品相似度作为独立子命令提供。
    # Provides recommendations and item similarity as separate subcommands.
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    recommend = subparsers.add_parser("recommend")
    recommend.add_argument("model", choices=("mf",))
    recommend.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    recommend.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    recommend.add_argument("--output-dir", type=Path, default=Path("outputs"))
    recommend.add_argument("--top-k", type=int, default=100)
    recommend.add_argument("--batch-size", type=int, default=512)
    recommend.add_argument("--device", default="cpu")
    recommend.add_argument(
        "--seen-file",
        default="train_seen.npz",
        choices=("train_seen.npz", "train_val_seen.npz", "all_seen.npz"),
    )
    similarity = subparsers.add_parser("item-similarity")
    similarity.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    similarity.add_argument("--output-dir", type=Path, default=Path("outputs"))
    similarity.add_argument("--top-k", type=int, default=200)
    similarity.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    if args.command == "recommend":
        # Erstellt modellabhängige Top-K-Listen für alle Benutzer.
        # 为所有用户生成依赖模型的 Top-K 列表。
        # Creates model-dependent Top-K lists for all users.
        retrieve_top_k(
            args.processed_dir,
            args.artifact_dir / f"{args.model}_embeddings.npz",
            args.output_dir / f"{args.model}_top{args.top_k}.npz",
            args.top_k,
            args.batch_size,
            args.seen_file,
            args.device,
        )
    else:
        # Erstellt die Nachbarn der MF-Artikel-Embeddings.
        # 生成 MF 物品 embedding 的邻居。
        # Creates neighbors of the MF item embeddings.
        item_inner_product_neighbors(
            args.artifact_dir / "mf_embeddings.npz",
            args.output_dir / f"mf_item_neighbors_top{args.top_k}.npz",
            args.top_k,
            args.batch_size,
        )


if __name__ == "__main__":
    # Führt den gewählten Retrieval-Befehl bei direktem Modulaufruf aus.
    # 模块被直接调用时执行所选检索命令。
    # Runs the selected retrieval command when the module is invoked directly.
    main()
