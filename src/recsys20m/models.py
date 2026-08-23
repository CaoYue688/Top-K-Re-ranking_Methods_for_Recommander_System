from __future__ import annotations

# Dieses Modul definiert Negative Sampling, BPR-MF und den Trainingsablauf.
# 本模块定义负采样、BPR-MF 和训练流程。
# This module defines negative sampling, BPR-MF, and the training workflow.
import argparse
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from .utils import load_json, save_json, set_seed, timestamped_message


class NegativeSampler:
    """Mixed explicit and uniform-unrated negatives for BPR training."""

    # Zieht gleichverteilte negative Artikel und verwirft bereits gesehene Paare exakt.
    # 均匀抽取负物品，并精确排除已见过的用户-物品对。
    # Draws uniform negative items and exactly rejects observed pairs.
    def __init__(
        self,
        sorted_seen_keys: np.ndarray,
        n_items: int,
        seed: int,
        explicit_items: np.ndarray | None = None,
        explicit_offsets: np.ndarray | None = None,
        explicit_ratio: float = 0.0,
    ) -> None:
        if not 0.0 <= explicit_ratio <= 1.0:
            raise ValueError("explicit_ratio must be between 0 and 1.")
        if (explicit_items is None) != (explicit_offsets is None):
            raise ValueError(
                "explicit_items and explicit_offsets must be provided together."
            )
        self.keys = sorted_seen_keys
        self.n_items = n_items
        self.rng = np.random.default_rng(seed)
        self.explicit_items = explicit_items
        self.explicit_offsets = explicit_offsets
        self.explicit_ratio = explicit_ratio

    def _is_seen(self, users: np.ndarray, items: np.ndarray) -> np.ndarray:
        # Kodiert ein Benutzer-Artikel-Paar als einzelne sortierbare Ganzzahl.
        # 将用户-物品对编码成一个可排序整数。
        # Encodes a user-item pair as one sortable integer.
        codes = users.astype(np.int64) * self.n_items + items.astype(np.int64)
        # Binäre Suche prüft viele Paare ohne große Python-Mengen im Speicher.
        # 二分查找无需在内存中建立巨大 Python 集合即可检查大量对。
        # Binary search checks many pairs without large Python sets in memory.
        locations = np.searchsorted(self.keys, codes)
        result = np.zeros(len(codes), dtype=bool)
        valid = locations < len(self.keys)
        result[valid] = self.keys[locations[valid]] == codes[valid]
        return result

    def _sample_unrated(self, users: np.ndarray) -> np.ndarray:
        # Zieht pro Benutzer einen Artikel und ersetzt ungültige Treffer wiederholt.
        # 为每个用户抽取一个物品，并反复替换无效结果。
        # Draws one item per user and repeatedly replaces invalid samples.
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

    def sample(self, users: np.ndarray) -> np.ndarray:
        # Mischt explizite Negative mit wirklich unbewerteten Artikeln im Zielverhältnis.
        # 按目标比例混合明确负样本与真正未评分物品。
        # Mixes explicit negatives with truly unrated items at the target ratio.
        negatives = np.empty(len(users), dtype=np.int32)
        use_explicit = np.zeros(len(users), dtype=bool)
        starts = np.zeros(len(users), dtype=np.int64)
        lengths = np.zeros(len(users), dtype=np.int64)
        if self.explicit_items is not None and self.explicit_offsets is not None:
            starts = self.explicit_offsets[users]
            lengths = self.explicit_offsets[users + 1] - starts
            use_explicit = (
                (lengths > 0)
                & (self.rng.random(len(users)) < self.explicit_ratio)
            )
        if use_explicit.any():
            draws = (
                self.rng.random(int(use_explicit.sum()))
                * lengths[use_explicit]
            ).astype(np.int64)
            explicit_indices = starts[use_explicit] + draws
            negatives[use_explicit] = self.explicit_items[explicit_indices]
        use_unrated = ~use_explicit
        if use_unrated.any():
            negatives[use_unrated] = self._sample_unrated(users[use_unrated])
        return negatives


class BPRMatrixFactorization(nn.Module):
    # Klassische Matrixfaktorisierung für implizites Feedback mit BPR-Loss.
    # 使用 BPR 损失的经典隐式反馈矩阵分解。
    # Classical matrix factorization for implicit feedback with BPR loss.
    def __init__(self, n_users: int, n_items: int, embedding_dim: int) -> None:
        super().__init__()
        # Sparse Embeddings aktualisieren nur die im Batch vorkommenden IDs.
        # 稀疏 embedding 只更新当前批次中出现的 ID。
        # Sparse embeddings update only IDs present in the batch.
        self.user_embedding = nn.Embedding(n_users, embedding_dim, sparse=True)
        self.item_embedding = nn.Embedding(n_items, embedding_dim, sparse=True)
        self.item_bias = nn.Embedding(n_items, 1, sparse=True)
        nn.init.normal_(self.user_embedding.weight, std=0.05)
        nn.init.normal_(self.item_embedding.weight, std=0.05)
        nn.init.zeros_(self.item_bias.weight)

    def score(self, users: torch.Tensor, items: torch.Tensor) -> torch.Tensor:
        # Relevanz ist das Skalarprodukt plus ein artikelspezifischer Bias.
        # 相关性得分是点积加上物品偏置。
        # Relevance is the dot product plus an item-specific bias.
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
        # BPR maximiert den Abstand zwischen positivem und negativem Artikel.
        # BPR 最大化正物品与负物品的得分差。
        # BPR maximizes the score gap between positive and negative items.
        user_vectors = self.user_embedding(users)
        positive_vectors = self.item_embedding(positives)
        negative_vectors = self.item_embedding(negatives)
        positive_score = (user_vectors * positive_vectors).sum(dim=-1)
        positive_score = positive_score + self.item_bias(positives).squeeze(-1)
        negative_score = (user_vectors * negative_vectors).sum(dim=-1)
        negative_score = negative_score + self.item_bias(negatives).squeeze(-1)
        # L2-Regularisierung begrenzt übermäßig große latente Vektoren.
        # L2 正则化限制过大的潜向量。
        # L2 regularization limits excessively large latent vectors.
        penalty = (
            user_vectors.square().mean()
            + positive_vectors.square().mean()
            + negative_vectors.square().mean()
        )
        return -F.logsigmoid(positive_score - negative_score).mean() + (
            regularization * penalty
        )


@dataclass(frozen=True)
class TrainingConfig:
    # Enthält alle reproduzierbaren Hyperparameter und Ein-/Ausgabepfade.
    # 包含所有可复现超参数及输入输出路径。
    # Contains all reproducible hyperparameters and input/output paths.
    processed_dir: Path
    artifact_dir: Path
    embedding_dim: int = 64
    batch_size: int = 4096
    epochs: int = 3
    steps_per_epoch: int | None = None
    learning_rate: float = 0.01
    regularization: float = 1e-5
    explicit_negative_ratio: float = 0.5
    seed: int = 2026
    device: str = "cpu"


def _tensor(values: np.ndarray, device: torch.device) -> torch.Tensor:
    # Konvertiert NumPy-ID-Arrays direkt in LongTensoren auf dem Zielgerät.
    # 将 NumPy ID 数组直接转换为目标设备上的 LongTensor。
    # Converts NumPy ID arrays directly to LongTensors on the target device.
    return torch.as_tensor(values, dtype=torch.long, device=device)


def _export_mf(model: BPRMatrixFactorization, path: Path) -> None:
    # Exportiert nur die für Retrieval nötigen MF-Parameter in ein kompaktes NPZ.
    # 仅将检索所需的 MF 参数导出为紧凑 NPZ。
    # Exports only the MF parameters needed for retrieval to a compact NPZ.
    np.savez(
        path,
        user_embedding=model.user_embedding.weight.detach().cpu().numpy(),
        item_embedding=model.item_embedding.weight.detach().cpu().numpy(),
        item_bias=model.item_bias.weight.detach().cpu().numpy().reshape(-1),
    )


def train(config: TrainingConfig) -> Path:
    # Trainiert BPR-MF auf zufällig gezogenen positiven Interaktionen.
    # 使用随机抽取的正交互训练 BPR-MF。
    # Trains BPR-MF on randomly sampled positive interactions.
    set_seed(config.seed)
    if not 0.0 <= config.explicit_negative_ratio <= 1.0:
        raise ValueError("explicit_negative_ratio must be between 0 and 1.")
    device = torch.device(config.device)
    stats = load_json(config.processed_dir / "stats.json")
    n_users, n_items = int(stats["n_users"]), int(stats["n_items"])
    with np.load(config.processed_dir / "train.npz") as data:
        train_users = data["user"]
        train_items = data["item"]
    # Sortierte Schlüssel schließen jedes im Trainingsfenster bewertete Paar aus.
    # 排序键排除训练窗口中所有已评分的用户-物品对。
    # Sorted keys exclude every pair rated in the training window.
    seen_keys = np.load(config.processed_dir / "train_seen_keys.npy", mmap_mode="r")
    explicit_path = config.processed_dir / "train_explicit_negatives.npz"
    if not explicit_path.exists():
        raise FileNotFoundError(
            "Scheme B requires train_explicit_negatives.npz; rerun preprocessing."
        )
    with np.load(explicit_path) as explicit:
        explicit_items = explicit["items"].copy()
        explicit_offsets = explicit["offsets"].copy()
    sampler = NegativeSampler(
        seen_keys,
        n_items,
        config.seed + 31,
        explicit_items=explicit_items,
        explicit_offsets=explicit_offsets,
        explicit_ratio=config.explicit_negative_ratio,
    )

    config.artifact_dir.mkdir(parents=True, exist_ok=True)
    # MF besitzt ausschließlich sparse Parameter und benötigt nur SparseAdam.
    # MF 只有稀疏参数，因此只需要 SparseAdam。
    # MF has only sparse parameters and therefore needs only SparseAdam.
    model = BPRMatrixFactorization(
        n_users, n_items, config.embedding_dim
    ).to(device)
    optimizer = torch.optim.SparseAdam(
        model.parameters(), lr=config.learning_rate
    )

    rng = np.random.default_rng(config.seed)
    # Ohne explizite Schrittzahl entspricht eine Epoche ungefähr der Trainingsmenge.
    # 未指定步数时，一个 epoch 大约覆盖一遍训练集。
    # Without an explicit step count, one epoch roughly covers the training set.
    default_steps = math.ceil(len(train_users) / config.batch_size)
    steps = config.steps_per_epoch or default_steps
    history: list[float] = []
    timestamped_message(
        f"Training mf: {config.epochs} epochs x {steps} steps "
        f"(batch={config.batch_size}, device={device})"
    )
    # Misst nur die eigentliche Optimierung, nicht Datenladen oder Embedding-Export.
    # 仅测量实际优化过程，不包括数据加载或 embedding 导出。
    # Measures only the optimization loop, excluding data loading and embedding export.
    training_started = time.perf_counter()
    model.train()
    for epoch in range(config.epochs):
        running = 0.0
        for step in range(steps):
            # Positive Beispiele werden mit Zurücklegen aus dem Trainingssplit gezogen.
            # 从训练分割中有放回地抽取正样本。
            # Positive examples are sampled with replacement from the training split.
            indices = rng.integers(0, len(train_users), size=config.batch_size)
            users_np = train_users[indices]
            positives_np = train_items[indices]
            negatives_np = sampler.sample(users_np)
            users = _tensor(users_np, device)
            positives = _tensor(positives_np, device)
            negatives = _tensor(negatives_np, device)

            # SparseAdam wird vor jedem Update zurückgesetzt.
            # 每次更新前清零 SparseAdam 梯度。
            # Resets SparseAdam gradients before every update.
            optimizer.zero_grad(set_to_none=True)
            loss = model.pairwise_loss(
                users, positives, negatives, config.regularization
            )
            loss.backward()
            optimizer.step()
            running += float(loss.detach().cpu())
            if (step + 1) % max(1, steps // 5) == 0:
                timestamped_message(
                    f"mf epoch {epoch + 1}/{config.epochs}, "
                    f"step {step + 1}/{steps}, loss={running / (step + 1):.5f}"
                )
        history.append(running / steps)

    training_seconds = time.perf_counter() - training_started

    model.eval()
    # Checkpoint speichert den Modellzustand; NPZ dient schnellem Retrieval ohne PyTorch.
    # Checkpoint 保存模型状态；NPZ 用于不依赖 PyTorch 的快速检索。
    # The checkpoint stores model state; NPZ enables fast retrieval without PyTorch.
    checkpoint = config.artifact_dir / "mf.pt"
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
    embedding_path = config.artifact_dir / "mf_embeddings.npz"
    _export_mf(model, embedding_path)
    save_json(
        config.artifact_dir / "mf_training.json",
        {
            "model": "mf",
            "loss": history,
            "epochs": config.epochs,
            "steps_per_epoch": steps,
            "batch_size": config.batch_size,
            "device": str(device),
            "learning_rate": config.learning_rate,
            "explicit_negative_ratio": config.explicit_negative_ratio,
            "seed": config.seed,
            "training_seconds": training_seconds,
        },
    )
    timestamped_message(f"Saved {checkpoint} and {embedding_path}")
    return embedding_path


def build_parser() -> argparse.ArgumentParser:
    # Definiert alle Modell- und Trainingsoptionen der Kommandozeile.
    # 定义命令行中的所有模型与训练选项。
    # Defines all command-line model and training options.
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--processed-dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--artifact-dir", type=Path, default=Path("artifacts"))
    parser.add_argument("--embedding-dim", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--regularization", type=float, default=1e-5)
    parser.add_argument("--explicit-negative-ratio", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    return parser


def main() -> None:
    # Baut aus den Argumenten eine TrainingConfig und startet das Training.
    # 从命令行参数构建 TrainingConfig 并启动训练。
    # Builds a TrainingConfig from arguments and starts training.
    args = build_parser().parse_args()
    train(
        TrainingConfig(
            processed_dir=args.processed_dir,
            artifact_dir=args.artifact_dir,
            embedding_dim=args.embedding_dim,
            batch_size=args.batch_size,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            learning_rate=args.learning_rate,
            regularization=args.regularization,
            explicit_negative_ratio=args.explicit_negative_ratio,
            seed=args.seed,
            device=args.device,
        )
    )


if __name__ == "__main__":
    # Startet Training nur, wenn das Modul direkt ausgeführt wird.
    # 仅在模块被直接执行时启动训练。
    # Starts training only when the module is executed directly.
    main()
