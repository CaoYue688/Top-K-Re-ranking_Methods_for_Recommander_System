from __future__ import annotations

# Dieses Modul orchestriert die komplette Pipeline von Daten bis Top-20.
# 本模块编排从数据到 Top-20 的完整流程。
# This module orchestrates the complete pipeline from data to Top-20.
import argparse
from pathlib import Path

from .evaluation import evaluate_sampled
from .models import TrainingConfig, train
from .preprocess import PreprocessConfig, preprocess
from .rerank import RerankWeights, rerank_model
from .retrieval import item_inner_product_neighbors, retrieve_top_k
from .utils import timestamped_message


def run_all(
    root: Path,
    epochs: int,
    steps_per_epoch: int | None,
    batch_size: int,
    seed: int,
    device: str,
    force_preprocess: bool,
    explicit_negative_ratio: float = 0.5,
) -> None:
    # Leitet alle Standardpfade relativ zum Projektstamm ab.
    # 相对于项目根目录推导所有默认路径。
    # Derives all default paths relative to the project root.
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    artifact_dir = root / "artifacts"
    output_dir = root / "outputs"
    scheme_b_files = (
        processed_dir / "stats.json",
        processed_dir / "train_explicit_negatives.npz",
    )
    if force_preprocess or not all(path.exists() for path in scheme_b_files):
        # Vorverarbeitung läuft nur, wenn Ergebnisse fehlen oder explizit erzwungen werden.
        # 仅在结果缺失或明确强制时执行预处理。
        # Runs preprocessing only when results are missing or explicitly forced.
        preprocess(
            PreprocessConfig(
                raw_dir=raw_dir,
                output_dir=processed_dir,
                seed=seed,
            )
        )
    else:
        timestamped_message("Using existing preprocessed dataset")

    # Trainiert und bewertet die einzige Baseline BPR-MF.
    # 训练并评估唯一的基线模型 BPR-MF。
    # Trains and evaluates the sole BPR-MF baseline.
    train(
        TrainingConfig(
            processed_dir=processed_dir,
            artifact_dir=artifact_dir,
            batch_size=max(batch_size, 8192),
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            explicit_negative_ratio=explicit_negative_ratio,
            seed=seed,
            device=device,
        )
    )
    # Validation und Test verwenden jeweils 101 fest gesampelte Kandidaten.
    # 验证和测试各使用 101 个固定采样候选项。
    # Validation and test each use 101 fixed sampled candidates.
    evaluate_sampled(processed_dir, artifact_dir, "mf", "val")
    evaluate_sampled(processed_dir, artifact_dir, "mf", "test")
    retrieve_top_k(
        processed_dir,
        artifact_dir / "mf_embeddings.npz",
        output_dir / "mf_top100.npz",
        k=100,
    )

    # MF-Artikel-Embeddings liefern eine sparse Top-200-Ähnlichkeitsstruktur.
    # MF 物品 embedding 生成稀疏 Top-200 相似度结构。
    # MF item embeddings produce a sparse Top-200 similarity structure.
    item_inner_product_neighbors(
        artifact_dir / "mf_embeddings.npz",
        output_dir / "mf_item_neighbors_top200.npz",
        top_k=200,
    )
    # Reduziert die MF-Top-100-Liste mit festen Gewichten auf Top-20.
    # 使用固定权重将 MF Top-100 列表缩减为 Top-20。
    # Reduces the MF Top-100 list to Top-20 with fixed weights.
    weights = RerankWeights()
    rerank_model(
        "mf",
        processed_dir,
        artifact_dir,
        output_dir,
        candidate_k=100,
        output_k=20,
        weights=weights,
    )


def main() -> None:
    # Definiert eine kompakte Kommandozeile für den vollständigen Workflow.
    # 为完整工作流定义简洁命令行入口。
    # Defines a compact command line for the complete workflow.
    parser = argparse.ArgumentParser(description="Run the complete MovieLens pipeline.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--explicit-negative-ratio", type=float, default=0.5)
    parser.add_argument("--force-preprocess", action="store_true")
    args = parser.parse_args()
    run_all(
        args.root.resolve(),
        args.epochs,
        args.steps_per_epoch,
        args.batch_size,
        args.seed,
        args.device,
        args.force_preprocess,
        args.explicit_negative_ratio,
    )


if __name__ == "__main__":
    # Startet die Gesamtpipeline bei direktem Aufruf des Moduls.
    # 在模块被直接调用时启动总流程。
    # Starts the full pipeline when the module is invoked directly.
    main()
