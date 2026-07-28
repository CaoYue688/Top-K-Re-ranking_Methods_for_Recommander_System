from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from recsys20m.evaluation import recommendation_quality
from recsys20m.models import NegativeSampler
from recsys20m.preprocess import PreprocessConfig, preprocess
from recsys20m.rerank import RerankWeights, greedy_rerank


class PipelineTest(unittest.TestCase):
    def test_preprocess_sampling_and_reranking(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw" / "ml-20m"
            source.mkdir(parents=True)
            movies = pd.DataFrame(
                {
                    "movieId": np.arange(1, 9),
                    "title": [f"Movie {i}" for i in range(1, 9)],
                    "genres": [
                        "Action",
                        "Comedy",
                        "Action|Comedy",
                        "Drama",
                        "Drama|Comedy",
                        "Action",
                        "Comedy",
                        "Drama",
                    ],
                }
            )
            interactions = []
            histories = {
                1: [1, 2, 3, 4, 5, 6],
                2: [1, 2, 3, 4, 7, 8],
                3: [3, 4, 5, 6, 7, 8],
            }
            for user, items in histories.items():
                for timestamp, item in enumerate(items, start=1):
                    interactions.append((user, item, 4.0, timestamp))
            movies.to_csv(source / "movies.csv", index=False)
            pd.DataFrame(
                interactions,
                columns=["userId", "movieId", "rating", "timestamp"],
            ).to_csv(source / "ratings.csv", index=False)

            output = root / "processed"
            stats = preprocess(
                PreprocessConfig(
                    raw_dir=root / "raw",
                    output_dir=output,
                    min_interactions=2,
                    negatives=2,
                    seed=7,
                )
            )
            self.assertEqual(stats["n_users"], 3)
            with np.load(output / "eval_candidates.npz") as data:
                self.assertEqual(data["test"].shape, (3, 3))
                self.assertTrue(np.all(data["test"][:, 0] >= 0))

            keys = np.load(output / "train_seen_keys.npy")
            sampler = NegativeSampler(keys, int(stats["n_items"]), seed=8)
            with np.load(output / "train.npz") as train:
                train_users = train["user"]
                negatives = sampler.sample(train_users)
                sampled_keys = train_users.astype(np.int64) * int(
                    stats["n_items"]
                ) + negatives
            self.assertFalse(np.isin(sampled_keys, keys).any())

            candidates = np.array([[0, 1, 2], [3, 4, 5], [5, 6, 7]])
            scores = np.array(
                [[0.9, 0.7, 0.1], [0.8, 0.6, 0.2], [0.9, 0.8, 0.7]],
                dtype=np.float32,
            )
            profiles = np.load(output / "user_genre_profiles.npy")
            item_genres = np.load(output / "item_genres.npy")
            embeddings = np.random.default_rng(3).normal(size=(8, 4)).astype(
                np.float32
            )
            recommendations, _ = greedy_rerank(
                candidates,
                scores,
                profiles,
                item_genres,
                embeddings,
                top_k=2,
                weights=RerankWeights(),
            )
            self.assertEqual(recommendations.shape, (3, 2))
            metrics = recommendation_quality(
                recommendations, profiles, item_genres, embeddings
            )
            self.assertGreaterEqual(metrics["calibration_mean"], 0.0)
            self.assertGreaterEqual(metrics["ild_mean"], 0.0)


if __name__ == "__main__":
    unittest.main()
