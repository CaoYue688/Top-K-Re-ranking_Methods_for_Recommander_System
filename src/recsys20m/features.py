from __future__ import annotations

"""Build optional MovieLens Tag Genome representations for sensitivity tests."""

from pathlib import Path

import numpy as np
import pandas as pd

from .utils import save_json, timestamped_message


def build_tag_genome_features(
    raw_dir: Path,
    processed_dir: Path,
    output_path: Path,
    components: int = 64,
    seed: int = 2026,
    device: str = "cuda",
) -> Path:
    """Create a deterministic low-rank Tag Genome item representation.

    MovieLens 20M stores 1,128 relevance values for every covered movie.  The
    matrix is reduced with an uncentred randomized SVD so that cosine-based MMR
    remains computationally tractable for the full user population.
    """
    stats_path = output_path.with_suffix(".json")
    coverage_path = output_path.with_name(output_path.stem + "_coverage.npy")
    if output_path.exists() and stats_path.exists() and coverage_path.exists():
        return output_path

    items = pd.read_csv(processed_dir / "items.csv", usecols=["item_idx", "movieId"])
    movie_ids = items["movieId"].to_numpy(dtype=np.int64)
    movie_lookup = pd.Index(movie_ids)
    genome_path = raw_dir / "ml-20m" / "genome-scores.csv"
    if not genome_path.exists():
        raise FileNotFoundError(genome_path)

    n_items = len(items)
    n_tags = 1128
    matrix = np.zeros((n_items, n_tags), dtype=np.float32)
    timestamped_message("Reading Tag Genome scores in chunks")
    for chunk_index, chunk in enumerate(
        pd.read_csv(
            genome_path,
            usecols=["movieId", "tagId", "relevance"],
            dtype={"movieId": np.int32, "tagId": np.int16, "relevance": np.float32},
            chunksize=1_000_000,
        ),
        start=1,
    ):
        item_indices = movie_lookup.get_indexer(chunk["movieId"].to_numpy())
        valid = item_indices >= 0
        matrix[
            item_indices[valid],
            chunk["tagId"].to_numpy(dtype=np.int32, copy=False)[valid] - 1,
        ] = chunk["relevance"].to_numpy(dtype=np.float32, copy=False)[valid]
        timestamped_message(f"Tag Genome chunk {chunk_index} loaded")

    covered = np.linalg.norm(matrix, axis=1) > 0
    if int(covered.sum()) <= components:
        raise ValueError("Too few Tag Genome-covered items for the requested rank.")

    import torch

    actual_device = device if device == "cpu" or torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    source = torch.from_numpy(matrix[covered]).to(actual_device)
    timestamped_message(
        f"Computing uncentred rank-{components} Tag Genome SVD on {actual_device}"
    )
    _, singular_values, basis = torch.pca_lowrank(
        source,
        q=components,
        center=False,
        niter=4,
    )
    reduced_covered = (source @ basis).cpu().numpy().astype(np.float32)
    reduced_covered /= np.maximum(
        np.linalg.norm(reduced_covered, axis=1, keepdims=True), 1e-12
    )
    reduced = np.zeros((n_items, components), dtype=np.float32)
    reduced[covered] = reduced_covered
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, reduced)
    np.save(coverage_path, covered)

    total_energy = float(np.square(matrix[covered]).sum())
    retained_energy = float(torch.square(singular_values).sum().cpu())
    save_json(
        stats_path,
        {
            "source": str(genome_path),
            "representation": "uncentred_randomized_svd",
            "components": components,
            "seed": seed,
            "device": actual_device,
            "n_items": n_items,
            "covered_items": int(covered.sum()),
            "item_coverage": float(covered.mean()),
            "retained_frobenius_energy": (
                retained_energy / total_energy if total_energy else 0.0
            ),
        },
    )
    return output_path


def tag_complete_candidate_pool(
    candidate_path: Path,
    coverage_path: Path,
    output_path: Path,
    k: int,
) -> Path:
    """Keep the first k scored candidates with a Tag Genome representation."""
    if output_path.exists():
        return output_path
    coverage = np.load(coverage_path)
    with np.load(candidate_path) as data:
        items = data["items"]
        scores = data["scores"]
    valid = coverage[items]
    counts = valid.sum(axis=1)
    if np.any(counts < k):
        raise ValueError(
            f"{int((counts < k).sum())} users have fewer than {k} covered candidates; "
            f"minimum is {int(counts.min())}."
        )
    positions = np.broadcast_to(np.arange(items.shape[1]), items.shape)
    covered_positions = np.where(valid, positions, items.shape[1])
    selected = np.sort(covered_positions, axis=1)[:, :k]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        items=np.take_along_axis(items, selected, axis=1),
        scores=np.take_along_axis(scores, selected, axis=1),
    )
    return output_path
