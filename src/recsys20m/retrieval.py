from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .utils import load_json, timestamped_message


def _top_k(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    if k >= scores.shape[1]:
        indices = np.argsort(-scores, axis=1)[:, :k]
    else:
        partition = np.argpartition(scores, -k, axis=1)[:, -k:]
        partition_scores = np.take_along_axis(scores, partition, axis=1)
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
) -> Path:
    with np.load(embedding_path) as data:
        user_embeddings = data["user_embedding"]
        item_embeddings = data["item_embedding"]
        item_bias = data["item_bias"] if "item_bias" in data else None
    with np.load(processed_dir / "train_seen.npz") as seen:
        seen_items = seen["items"]
        offsets = seen["offsets"]
    n_users, n_items = len(user_embeddings), len(item_embeddings)
    if k >= n_items:
        raise ValueError(f"k={k} must be smaller than n_items={n_items}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    all_items = np.empty((n_users, k), dtype=np.int32)
    all_scores = np.empty((n_users, k), dtype=np.float32)
    item_transpose = item_embeddings.T

    for start in range(0, n_users, user_batch_size):
        end = min(start + user_batch_size, n_users)
        scores = user_embeddings[start:end] @ item_transpose
        if item_bias is not None:
            scores += item_bias[None, :]
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
    with np.load(embedding_path) as data:
        item_embeddings = data["item_embedding"].astype(np.float32)
    if normalize:
        item_embeddings /= np.maximum(
            np.linalg.norm(item_embeddings, axis=1, keepdims=True), 1e-12
        )
    n_items = len(item_embeddings)
    if top_k >= n_items:
        raise ValueError("top_k must be smaller than the number of items")
    neighbor_items = np.empty((n_items, top_k), dtype=np.int32)
    neighbor_scores = np.empty((n_items, top_k), dtype=np.float32)
    transpose = item_embeddings.T
    for start in range(0, n_items, block_size):
        end = min(start + block_size, n_items)
        scores = item_embeddings[start:end] @ transpose
        rows = np.arange(end - start)
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
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    recommend = subparsers.add_parser("recommend")
    recommend.add_argument("model", choices=("mf", "two-tower"))
    recommend.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    recommend.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    recommend.add_argument("--output-dir", type=Path, default=Path("outputs"))
    recommend.add_argument("--top-k", type=int, default=100)
    recommend.add_argument("--batch-size", type=int, default=512)
    similarity = subparsers.add_parser("item-similarity")
    similarity.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    similarity.add_argument("--output-dir", type=Path, default=Path("outputs"))
    similarity.add_argument("--top-k", type=int, default=200)
    similarity.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    if args.command == "recommend":
        retrieve_top_k(
            args.processed_dir,
            args.artifact_dir / f"{args.model}_embeddings.npz",
            args.output_dir / f"{args.model}_top{args.top_k}.npz",
            args.top_k,
            args.batch_size,
        )
    else:
        item_inner_product_neighbors(
            args.artifact_dir / "two-tower_embeddings.npz",
            args.output_dir / f"two-tower_item_neighbors_top{args.top_k}.npz",
            args.top_k,
            args.batch_size,
        )


if __name__ == "__main__":
    main()

