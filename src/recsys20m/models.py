from __future__ import annotations

import argparse
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .utils import load_json, save_json, set_seed, timestamped_message


class NegativeSampler:
    """Uniform negatives with exact rejection against sorted observed keys."""

    def __init__(
        self,
        sorted_seen_keys: np.ndarray,
        n_items: int,
        seed: int,
    ) -> None:
        self.keys = sorted_seen_keys
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)

    def _is_seen(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        codes = users.astype(np.int64) * self.n_items + items.astype(np.int64)
        locations = np.searchsorted(self.keys, codes)
        result = np.zeros(len(codes), dtype=bool)
        valid = locations < len(self.keys)
        result[valid] = self.keys[locations[valid]] == codes[valid]
        return result

    def sample(self, users: np.ndarray) -> np.ndarray:
        negatives = self.rng.integers(
            0, self.n_items, size=len(users), dtype=np.int32
        )
        invalid = self._is_seen(users, negatives)
        while invalid.any():
            negatives[invalid] = self.rng.integers(
                0, self.n_items, size=int(invalid.sum()), dtype=np.int32
            )
            invalid = self._is_seen(users, negatives)
        return negatives


class BPRMatrixFactorization(nn.Module):
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embedding_dim, sparse=True)
        self.item_embedding = nn.Embedding(n_items, embedding_dim, sparse=True)
        self.item_bias = nn.Embedding(n_items, 1, sparse=True)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (
            (self.user_embedding(users) * self.item_embedding(items)).sum(dim=-1)
            + self.item_bias(items).squeeze(-1)
        )

    def pairwise_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
        regularization: float,
    ) -> torch.Tensor:
        user_vectors = self.user_embedding(users)
        positive_vectors = self.item_embedding(positives)
        negative_vectors = self.item_embedding(negatives)
        positive_score = (user_vectors * positive_vectors).sum(dim=-1)
        positive_score = positive_score + self.item_bias(positives).squeeze(-1)
        negative_score = (user_vectors * negative_vectors).sum(dim=-1)
        negative_score = negative_score + self.item_bias(negatives).squeeze(-1)
        penalty = (
            user_vectors.square().mean()
            + positive_vectors.square().mean()
            + negative_vectors.square().mean()
        )
        return -F.logsigmoid(positive_score - negative_score).mean() + (
            regularization * penalty
        )


class TwoTowerDNN(nn.Module):
    def __init__(
        self,
        n_users: int,
        n_items: int,
        user_profiles: np.ndarray,
        item_genres: np.ndarray,
        id_embedding_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        n_genres = item_genres.shape[1]
        self.user_id_embedding = nn.Embedding(
            n_users, id_embedding_dim, sparse=True
        )
        self.item_id_embedding = nn.Embedding(
            n_items, id_embedding_dim, sparse=True
        )
        self.user_tower = nn.Sequential(
            nn.Linear(id_embedding_dim + n_genres, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        self.item_tower = nn.Sequential(
            nn.Linear(id_embedding_dim + n_genres, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
        nn.init.normal_(self.user_id_embedding.weight, std=0.05)
        nn.init.normal_(self.item_id_embedding.weight, std=0.05)
        self.register_buffer(
            "user_profiles", torch.as_tensor(user_profiles, dtype=torch.float32)
        )
        self.register_buffer(
            "item_genres", torch.as_tensor(item_genres, dtype=torch.float32)
        )

    def encode_user(self, users: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (self.user_id_embedding(users), self.user_profiles[users]), dim=-1
        )
        return F.normalize(self.user_tower(features), dim=-1)

    def encode_item(self, items: torch.Tensor) -> torch.Tensor:
        features = torch.cat(
            (self.item_id_embedding(items), self.item_genres[items]), dim=-1
        )
        return F.normalize(self.item_tower(features), dim=-1)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        return (self.encode_user(users) * self.encode_item(items)).sum(dim=-1)

    def pairwise_loss(
        self,
        users: torch.Tensor,
        positives: torch.Tensor,
        negatives: torch.Tensor,
    ) -> torch.Tensor:
        user_vectors = self.encode_user(users)
        positive_vectors = self.encode_item(positives)
        negative_vectors = self.encode_item(negatives)
        positive_score = (user_vectors * positive_vectors).sum(dim=-1)
        negative_score = (user_vectors * negative_vectors).sum(dim=-1)
        return -F.logsigmoid(positive_score - negative_score).mean()


@dataclass(frozen=True)
class TrainingConfig:
    processed_dir: Path
    artifact_dir: Path
    model: Literal["mf", "two-tower"]
    embedding_dim: int = 64
    hidden_dim: int = 128
    batch_size: int = 4096
    epochs: int = 3
    steps_per_epoch: int | None = None
    learning_rate: float = 0.01
    regularization: float = 1e-5
    seed: int = 2026
    device: str = "cpu"


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _export_mf(model: BPRMatrixFactorization, path: Path) -> None:
    np.savez(
        path,
        user_embedding=model.user_embedding.weight.detach().cpu().numpy(),
        item_embedding=model.item_embedding.weight.detach().cpu().numpy(),
        item_bias=model.item_bias.weight.detach().cpu().numpy().reshape(-1),
    )


@torch.inference_mode()
def _export_two_tower(
    model: TwoTowerDNN,
    path: Path,
    device: torch.device,
    batch_size: int = 8192,
) -> None:
    users: list[np.ndarray] = []
    items: list[np.ndarray] = []
    for start in range(0, model.user_id_embedding.num_embeddings, batch_size):
        ids = torch.arange(
            start,
            min(start + batch_size, model.user_id_embedding.num_embeddings),
            device=device,
        )
        users.append(model.encode_user(ids).cpu().numpy())
    for start in range(0, model.item_id_embedding.num_embeddings, batch_size):
        ids = torch.arange(
            start,
            min(start + batch_size, model.item_id_embedding.num_embeddings),
            device=device,
        )
        items.append(model.encode_item(ids).cpu().numpy())
    np.savez(
        path,
        user_embedding=np.concatenate(users),
        item_embedding=np.concatenate(items),
    )


def train(config: TrainingConfig) -> Path:
    set_seed(config.seed)
    device = torch.device(config.device)
    stats = load_json(config.processed_dir / "stats.json")
    n_users, n_items = int(stats["n_users"]), int(stats["n_items"])
    with np.load(config.processed_dir / "train.npz") as data:
        train_users = data["user"]
        train_items = data["item"]
    seen_keys = np.load(config.processed_dir / "train_seen_keys.npy", mmap_mode="r")
    sampler = NegativeSampler(seen_keys, n_items, config.seed + 31)

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    if config.model == "mf":
        model: BPRMatrixFactorization | TwoTowerDNN = BPRMatrixFactorization(
            n_users, n_items, config.embedding_dim
        ).to(device)
        sparse_optimizer = torch.optim.SparseAdam(
            model.parameters(), lr=config.learning_rate
        )
        dense_optimizer = None
    else:
        profiles = np.load(config.processed_dir / "user_genre_profiles.npy")
        item_genres = np.load(config.processed_dir / "item_genres.npy")
        model = TwoTowerDNN(
            n_users,
            n_items,
            profiles,
            item_genres,
            config.embedding_dim,
            config.hidden_dim,
            config.embedding_dim,
        ).to(device)
        sparse_optimizer = torch.optim.SparseAdam(
            [model.user_id_embedding.weight, model.item_id_embedding.weight],
            lr=config.learning_rate,
        )
        dense_optimizer = torch.optim.AdamW(
            list(model.user_tower.parameters()) + list(model.item_tower.parameters()),
            lr=config.learning_rate,
            weight_decay=config.regularization,
        )

    rng = np.random.default_rng(config.seed)
    default_steps = math.ceil(len(train_users) / config.batch_size)
    steps = config.steps_per_epoch or default_steps
    history: list[float] = []
    timestamped_message(
        f"Training {config.model}: {config.epochs} epochs x {steps} steps "
        f"(batch={config.batch_size}, device={device})"
    )
    model.train()
    for epoch in range(config.epochs):
        running = 0.0
        for step in range(steps):
            indices = rng.integers(0, len(train_users), size=config.batch_size)
            users_np = train_users[indices]
            positives_np = train_items[indices]
            negatives_np = sampler.sample(users_np)
            users = _tensor(users_np, device)
            positives = _tensor(positives_np, device)
            negatives = _tensor(negatives_np, device)

            sparse_optimizer.zero_grad(set_to_none=True)
            if dense_optimizer is not None:
                dense_optimizer.zero_grad(set_to_none=True)
            if config.model == "mf":
                assert isinstance(model, BPRMatrixFactorization)
                loss = model.pairwise_loss(
                    users, positives, negatives, config.regularization
                )
            else:
                assert isinstance(model, TwoTowerDNN)
                loss = model.pairwise_loss(users, positives, negatives)
            loss.backward()
            sparse_optimizer.step()
            if dense_optimizer is not None:
                dense_optimizer.step()
            running += float(loss.detach().cpu())
            if (step + 1) % max(1, steps // 5) == 0:
                timestamped_message(
                    f"{config.model} epoch {epoch + 1}/{config.epochs}, "
                    f"step {step + 1}/{steps}, loss={running / (step + 1):.5f}"
                )
        history.append(running / steps)

    model.eval()
    checkpoint = config.artifact_dir / f"{config.model}.pt"
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": {
                **asdict(config),
                "processed_dir": str(config.processed_dir),
                "artifact_dir": str(config.artifact_dir),
            },
            "n_users": n_users,
            "n_items": n_items,
        },
        checkpoint,
    )
    embedding_path = config.artifact_dir / f"{config.model}_embeddings.npz"
    if isinstance(model, BPRMatrixFactorization):
        _export_mf(model, embedding_path)
    else:
        _export_two_tower(model, embedding_path, device)
    save_json(
        config.artifact_dir / f"{config.model}_training.json",
        {
            "model": config.model,
            "loss": history,
            "epochs": config.epochs,
            "steps_per_epoch": steps,
            "batch_size": config.batch_size,
            "seed": config.seed,
        },
    )
    timestamped_message(f"Saved {checkpoint} and {embedding_path}")
    return embedding_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", choices=("mf", "two-tower"))
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(
        TrainingConfig(
            processed_dir=args.processed_dir,
            artifact_dir=args.artifact_dir,
            model=args.model,
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            seed=args.seed,
            device=args.device,
        )
    )


if __name__ == "__main__":
    main()

