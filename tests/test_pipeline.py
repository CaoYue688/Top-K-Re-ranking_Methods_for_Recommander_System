from __future__ import annotations

# Dieser Test prüft die wichtigsten Pipeline-Schritte auf einem kleinen Datensatz.
# 本测试在小型数据集上检查最重要的流程步骤。
# This test checks the most important pipeline steps on a small dataset.
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from recsys20m.evaluation import full_ranking_metrics, recommendation_quality
from recsys20m.models import NegativeSampler, TrainingConfig, train as train_model
from recsys20m.preprocess import (
    PreprocessConfig,
    iterative_train_k_core,
    preprocess,
)
from recsys20m.rerank import RerankWeights, greedy_rerank
from recsys20m.tradeoff import (
    _sparse_calibration_scores,
    make_objectives,
    sweep_rerank,
)
from recsys20m.thesis_pipeline import _budget_selections, _holm_adjust
from recsys20m.utils import load_json


class PipelineTest(unittest.TestCase):
    # Ein End-to-End-Test deckt Vorverarbeitung, Sampling und Re-Ranking gemeinsam ab.
    # 端到端测试共同覆盖预处理、采样和重排序。
    # An end-to-end test jointly covers preprocessing, sampling, and re-ranking.
    def test_preprocess_sampling_and_reranking(self) -> None:
        # Temporäre Dateien werden nach dem Test automatisch entfernt.
        # 测试结束后自动删除临时文件。
        # Temporary files are removed automatically after the test.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw" / "ml-20m"
            source.mkdir(parents=True)
            # Acht künstliche Filme besitzen einfache, kontrollierbare Genre-Merkmale.
            # 八部人工电影具有简单且可控的类型特征。
            # Eight synthetic movies have simple, controllable genre features.
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
            # Jeder der drei Benutzer erhält sechs chronologisch geordnete Interaktionen.
            # 三个用户各有六次按时间排序的交互。
            # Each of the three users receives six chronologically ordered interactions.
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
            # Für den kleinen Test genügen 2-Core und zwei negative Evaluationsbeispiele.
            # 小型测试使用 2-core 和两个评估负样本即可。
            # A 2-core and two negative evaluation samples are sufficient for the small test.
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
            self.assertTrue((output / "train_val_seen.npz").exists())
            with np.load(output / "train_seen.npz") as train_seen:
                train_seen_count = len(train_seen["items"])
            with np.load(output / "train_val_seen.npz") as train_val_seen:
                self.assertGreater(len(train_val_seen["items"]), train_seen_count)
            # Eine positive Position plus zwei Negative ergeben drei Kandidaten je Benutzer.
            # 一个正样本位置加两个负样本构成每个用户的三个候选项。
            # One positive position plus two negatives gives three candidates per user.
            with np.load(output / "eval_candidates.npz") as data:
                self.assertEqual(data["test"].shape, (3, 3))
                self.assertTrue(np.all(data["test"][:, 0] >= 0))

            keys = np.load(output / "train_seen_keys.npy")
            sampler = NegativeSampler(keys, int(stats["n_items"]), seed=8)
            # Ein unbewertetes Negativsample darf im Trainingsfenster nie bewertet worden sein.
            # 未评分负样本不能在训练窗口中出现过任何评分。
            # An unrated negative sample must never have been rated in the training window.
            with np.load(output / "train.npz") as train:
                train_users = train["user"]
                negatives = sampler.sample(train_users)
                sampled_keys = train_users.astype(np.int64) * int(
                    stats["n_items"]
                ) + negatives
            self.assertFalse(np.isin(sampled_keys, keys).any())

            # Ein kleiner Kandidatenpool prüft Form und Gültigkeit des Greedy-Re-Rankings.
            # 小型候选池用于检查贪心重排序的形状和有效性。
            # A small candidate pool checks the shape and validity of greedy re-ranking.
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
            # Kalibrierung und ILD müssen für gültige Listen nichtnegativ sein.
            # 有效列表的校准度和 ILD 必须非负。
            # Calibration and ILD must be nonnegative for valid lists.
            metrics = recommendation_quality(
                recommendations, profiles, item_genres, embeddings
            )
            self.assertGreaterEqual(metrics["calibration_mean"], 0.0)
            self.assertGreaterEqual(metrics["ild_mean"], 0.0)

    def test_full_accuracy_metrics_and_tradeoff_sweep(self) -> None:
        # Die vollständige Accuracy-Auswertung berücksichtigt alle positiven Testartikel pro Benutzer.
        # 完整准确率评估考虑每个用户的全部测试正物品。
        # Full accuracy evaluation considers all positive test items per user.
        recommendations = np.array([[1, 3], [2, 0]], dtype=np.int32)
        positive_users = np.array([0, 0, 1], dtype=np.int32)
        positive_items = np.array([1, 2, 2], dtype=np.int32)
        metrics, per_user = full_ranking_metrics(
            recommendations, positive_users, positive_items, n_items=4
        )
        self.assertAlmostEqual(metrics["recall@2"], 0.75)
        self.assertAlmostEqual(metrics["hr@2"], 1.0)
        self.assertEqual(per_user["ndcg"].shape, (2,))

        candidates = np.array([[0, 1, 2], [1, 2, 3]], dtype=np.int32)
        scores = np.array([[1.0, 0.5, 0.0], [1.0, 0.5, 0.0]], dtype=np.float32)
        item_genres = np.array(
            [[1, 0], [1, 0], [0, 1], [0, 1]], dtype=np.float32
        )
        profiles = np.array([[0.5, 0.5], [0.5, 0.5]], dtype=np.float32)
        objectives = make_objectives(
            "diversity", np.array([0.0, 1.0], dtype=np.float32)
        )
        sweep = sweep_rerank(
            candidates,
            scores,
            profiles,
            item_genres,
            objectives,
            top_k=2,
            batch_size=2,
        )
        self.assertEqual(sweep.shape, (2, 2, 2))
        np.testing.assert_array_equal(sweep[0], candidates[:, :2])
        self.assertTrue(
            all(len(set(row.tolist())) == 2 for row in sweep.reshape(-1, 2))
        )

        xquad = sweep_rerank(
            np.array([[0, 1, 2]], dtype=np.int32),
            np.ones((1, 3), dtype=np.float32),
            np.array([[0.6, 0.3, 0.1]], dtype=np.float32),
            np.eye(3, dtype=np.float32),
            make_objectives("xquad", np.array([1.0], dtype=np.float32)),
            top_k=2,
            batch_size=1,
            diversity_mode="xquad",
        )
        np.testing.assert_array_equal(xquad[0, 0], [0, 1])

        # MMR has no diversity contribution for an empty prefix, whereas the
        # binary xQuAD aspect prior can already change the first position.
        first_candidates = np.array([[0, 1]], dtype=np.int32)
        first_scores = np.array([[1.0, 0.0]], dtype=np.float32)
        first_genres = np.array([[0, 1], [1, 0]], dtype=np.float32)
        first_profile = np.array([[0.9, 0.1]], dtype=np.float32)
        mmr_first = sweep_rerank(
            first_candidates,
            first_scores,
            first_profile,
            first_genres,
            make_objectives("mmr", np.array([0.75], dtype=np.float32)),
            top_k=1,
            batch_size=1,
            diversity_mode="max",
        )
        xquad_first = sweep_rerank(
            first_candidates,
            first_scores,
            first_profile,
            first_genres,
            make_objectives("xquad", np.array([0.75], dtype=np.float32)),
            top_k=1,
            batch_size=1,
            diversity_mode="xquad",
        )
        self.assertEqual(int(mmr_first[0, 0, 0]), 0)
        self.assertEqual(int(xquad_first[0, 0, 0]), 1)

    def test_budget_selection_uses_one_confirmatory_holm_family(self) -> None:
        rows: list[dict[str, float | int | str]] = []
        method_values = {
            "mmr": (0.096, 0.31, 0.40, 0.45),
            "xquad": (0.097, 0.25, 0.55, 0.50),
            "calibration": (0.098, 0.24, 0.48, 0.65),
        }
        for split in ("val", "test"):
            for seed in (1, 2, 3):
                for method, values in method_values.items():
                    for tradeoff, metrics in (
                        (0.0, (0.100, 0.20, 0.30, 0.40)),
                        (0.5, values),
                    ):
                        rows.append(
                            {
                                "split": split,
                                "seed": seed,
                                "method": method,
                                "lambda": tradeoff,
                                "ndcg@10": metrics[0],
                                "ild@10": metrics[1],
                                "subtopic_recall@10": metrics[2],
                                "calibration@10": metrics[3],
                                "catalog_coverage@10": 0.5,
                                "exposure_gini@10": 0.9,
                                "long_tail_share@10": 0.1,
                                "delta_ndcg_sign_p": 0.01,
                                "delta_ild_sign_p": 0.001,
                                "delta_ndcg_effect_dz": -0.1,
                                "delta_ild_effect_dz": 0.8,
                            }
                        )
        selections, tests = _budget_selections(pd.DataFrame(rows))
        self.assertEqual(len(selections), 28)
        primary = tests[tests["scope"] == "primary_ild_across_methods"]
        self.assertEqual(len(primary), 4)
        self.assertTrue((primary["holm_family_size"] == 4).all())
        self.assertTrue(primary["delta_ild_sign_p_holm"].notna().all())
        exploratory = tests[tests["scope"] != "primary_ild_across_methods"]
        self.assertTrue(exploratory["delta_ild_sign_p_holm"].isna().all())
        xquad = selections[
            (selections["scope"] == "construct_aligned")
            & (selections["method"] == "xquad")
        ]
        self.assertTrue((xquad["target_metric"] == "subtopic_recall@10").all())
        self.assertEqual(_holm_adjust([0.01, 0.04, 0.03, 0.002]), [0.03, 0.06, 0.06, 0.008])

    def test_train_only_core_uses_chronological_train_edges(self) -> None:
        users = np.repeat(np.arange(4, dtype=np.int32), 6)
        items = np.array(
            [
                0, 1, 2, 3, 4, 5,
                2, 3, 4, 5, 0, 1,
                0, 1, 2, 3, 4, 5,
                2, 3, 4, 5, 0, 1,
            ],
            dtype=np.int32,
        )
        timestamps = np.tile(np.arange(6, dtype=np.int64), 4)
        keep = iterative_train_k_core(users, items, timestamps, minimum=2)
        self.assertTrue(keep.all())

    def test_sparse_calibration_matches_dense_jsd(self) -> None:
        # Die schnelle Sparse-Formel muss mathematisch der direkten JSD-Berechnung entsprechen.
        # 快速稀疏公式在数学上必须与直接 JSD 计算一致。
        # The fast sparse formula must mathematically match the direct JSD calculation.
        profiles = np.array(
            [[0.5, 0.3, 0.2], [0.2, 0.2, 0.6]], dtype=np.float32
        )
        genre_sum = np.array(
            [
                [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5]],
            ],
            dtype=np.float32,
        )
        candidate_genres = np.array(
            [
                [[1.0, 0.0, 0.0], [0.5, 0.5, 0.0], [0.0, 0.0, 1.0]],
                [[0.0, 1.0, 0.0], [0.0, 0.5, 0.5], [1.0, 0.0, 0.0]],
            ],
            dtype=np.float32,
        )
        indices = np.array(
            [
                [[0, 0], [0, 1], [2, 0]],
                [[1, 0], [1, 2], [0, 0]],
            ],
            dtype=np.int32,
        )
        counts = np.array([[1, 2, 1], [1, 2, 1]], dtype=np.int32)
        sparse = _sparse_calibration_scores(
            profiles, genre_sum, indices, counts, rank=1
        )
        after = (
            genre_sum[:, :, None, :] + candidate_genres[:, None, :, :]
        ) / 2
        repeated_profiles = np.broadcast_to(
            profiles[:, None, None, :], after.shape
        )
        midpoint = 0.5 * (repeated_profiles + after)
        eps = 1e-12
        jsd = 0.5 * (
            np.sum(
                np.where(
                    repeated_profiles > 0,
                    repeated_profiles
                    * np.log((repeated_profiles + eps) / (midpoint + eps)),
                    0.0,
                ),
                axis=3,
            )
            + np.sum(
                np.where(
                    after > 0,
                    after * np.log((after + eps) / (midpoint + eps)),
                    0.0,
                ),
                axis=3,
            )
        )
        dense = 1.0 - jsd / np.log(2.0)
        np.testing.assert_allclose(sparse, dense, atol=1e-6)

    def test_positive_rating_threshold(self) -> None:
        # Schema B trennt positive, explizit negative und neutrale Bewertungen.
        # 方案 B 将评分分为正向、明确负向和中性三类。
        # Scheme B separates positive, explicit-negative, and neutral ratings.
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / "raw" / "ml-20m"
            source.mkdir(parents=True)
            pd.DataFrame(
                {
                    "movieId": np.arange(1, 9),
                    "title": [f"Movie {index}" for index in range(1, 9)],
                    "genres": ["Drama"] * 8,
                }
            ).to_csv(source / "movies.csv", index=False)
            ratings = []
            user_histories = {
                1: ([1, 2, 3, 4, 5, 6], [5.0, 2.0, 4.0, 3.0, 4.5, 1.0]),
                2: ([3, 4, 5, 6, 7, 8], [2.0, 5.0, 3.0, 4.0, 1.0, 4.5]),
            }
            for user, (movies, values) in user_histories.items():
                for timestamp, (movie, rating) in enumerate(
                    zip(movies, values), start=1
                ):
                    ratings.append((user, movie, rating, timestamp))
            pd.DataFrame(
                ratings,
                columns=["userId", "movieId", "rating", "timestamp"],
            ).to_csv(source / "ratings.csv", index=False)

            output = root / "processed"
            stats = preprocess(
                PreprocessConfig(
                    raw_dir=root / "raw",
                    output_dir=output,
                    min_interactions=1,
                    positive_threshold=4.0,
                    negative_threshold=2.0,
                    negatives=1,
                    seed=9,
                )
            )
            self.assertEqual(stats["raw_interactions"], 12)
            self.assertEqual(stats["threshold_interactions"], 6)
            self.assertEqual(stats["feedback_scheme"], "three_level_explicit_negative")
            self.assertEqual(stats["explicit_negative_interactions"], 2)
            self.assertEqual(stats["neutral_interactions"], 2)
            self.assertEqual(stats["train_explicit_negative_interactions"], 1)
            with np.load(output / "train.npz") as train:
                self.assertTrue(np.all(train["rating"] >= 4.0))
            with np.load(output / "train_explicit_negatives.npz") as explicit:
                np.testing.assert_array_equal(explicit["ratings"], [2.0])
                np.testing.assert_array_equal(explicit["offsets"], [0, 0, 1])

            # Auch neutrale und niedrige Trainingsratings werden aus dem Unrated-Pool entfernt.
            # 训练期的中性和低评分也会从未评分候选池中排除。
            # Neutral and low train-window ratings are also removed from the unrated pool.
            keys = np.load(output / "train_seen_keys.npy")
            self.assertIn(7, keys.tolist())

            # Ein Optimierungsschritt prüft die Übergabe der gemischten Negative an BPR-MF.
            # 一个优化步骤用于检查混合负样本是否正确传入 BPR-MF。
            # One optimization step verifies mixed negatives are passed into BPR-MF.
            artifacts = root / "artifacts"
            embedding_path = train_model(
                TrainingConfig(
                    processed_dir=output,
                    artifact_dir=artifacts,
                    embedding_dim=4,
                    batch_size=4,
                    epochs=1,
                    steps_per_epoch=1,
                    explicit_negative_ratio=0.5,
                    seed=9,
                )
            )
            self.assertTrue(embedding_path.exists())
            training_stats = load_json(artifacts / "mf_training.json")
            self.assertEqual(training_stats["explicit_negative_ratio"], 0.5)

    def test_mixed_negative_sampler(self) -> None:
        # Bei Quote 1 nutzt jeder geeignete Benutzer sein explizites Negativ.
        # 当比例为 1 时，每个有低分记录的用户都使用明确负样本。
        # At ratio 1, every eligible user uses an explicit negative.
        seen_keys = np.array([0, 6], dtype=np.int64)
        explicit_items = np.array([0, 1], dtype=np.int32)
        explicit_offsets = np.array([0, 1, 2, 2], dtype=np.int64)
        sampler = NegativeSampler(
            seen_keys,
            n_items=5,
            seed=13,
            explicit_items=explicit_items,
            explicit_offsets=explicit_offsets,
            explicit_ratio=1.0,
        )
        users = np.array([0, 1, 2], dtype=np.int32)
        sampled = sampler.sample(users)
        np.testing.assert_array_equal(sampled[:2], [0, 1])
        self.assertGreaterEqual(int(sampled[2]), 0)
        self.assertLess(int(sampled[2]), 5)


if __name__ == "__main__":
    # Erlaubt das direkte Starten dieser Testdatei.
    # 允许直接运行此测试文件。
    # Allows this test file to be run directly.
    unittest.main()
