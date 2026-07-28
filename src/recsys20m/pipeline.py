from __future__ import annotations

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
) -> None:
    raw_dir = root / "data" / "raw"
    processed_dir = root / "data" / "processed"
    artifact_dir = root / "artifacts"
    output_dir = root / "outputs"
    if force_preprocess or not (processed_dir / "stats.json").exists():
        preprocess(
            PreprocessConfig(
                raw_dir=raw_dir,
                output_dir=processed_dir,
                seed=seed,
            )
        )
    else:
        timestamped_message("Using existing preprocessed dataset")

    for model, model_batch_size in (
        ("mf", max(batch_size, 8192)),
        ("two-tower", batch_size),
    ):
        train(
            TrainingConfig(
                processed_dir=processed_dir,
                artifact_dir=artifact_dir,
                model=model,  # type: ignore[arg-type]
                batch_size=model_batch_size,
                epochs=epochs,
                steps_per_epoch=steps_per_epoch,
                seed=seed,
                device=device,
            )
        )
        evaluate_sampled(processed_dir, artifact_dir, model, "val")
        evaluate_sampled(processed_dir, artifact_dir, model, "test")
        retrieve_top_k(
            processed_dir,
            artifact_dir / f"{model}_embeddings.npz",
            output_dir / f"{model}_top100.npz",
            k=100,
        )

    item_inner_product_neighbors(
        artifact_dir / "two-tower_embeddings.npz",
        output_dir / "two-tower_item_neighbors_top200.npz",
        top_k=200,
    )
    weights = RerankWeights()
    for model in ("mf", "two-tower"):
        rerank_model(
            model,
            processed_dir,
            artifact_dir,
            output_dir,
            candidate_k=100,
            output_k=20,
            weights=weights,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the complete MovieLens pipeline.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--steps-per-epoch", type=int)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--device", default="cpu")
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
    )


if __name__ == "__main__":
    main()

