[中文](RESULTS.md) | [English](RESULTS.en.md) | [Deutsch](RESULTS.de.md)

# Historical Baseline Results (Not Used in the Final Thesis)

> This page records an early single-seed experiment under an obsolete evaluation definition. The final thesis uses [RESEARCH_RESULTS.en.md](RESEARCH_RESULTS.en.md) and
> `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/`; do not cite the values on this page as final results.

Run configuration: random seed `2026`, embedding dimension `64`, and `1 epoch` at full training scale. These are the actual persisted outputs of that historical run. For a formal experiment, use the production procedure in [README.md](README.md), train the documented configuration, and select checkpoints only from validation metrics.

## Data

| Item | Value |
|---|---:|
| Raw interactions | 20,000,263 |
| Interactions after iterative 5-core | 19,984,024 |
| Users | 138,493 |
| Items | 18,345 |
| Train | 15,932,772 (79.73%) |
| Validation | 1,940,306 (9.71%) |
| Test | 2,110,946 (10.56%) |
| Evaluation candidates | 1 positive + 100 negatives per user |

## Ranking Metrics on the Sampled Test Set

| Model | HR@10 | NDCG@10 | MRR@10 | HR@20 | NDCG@20 | MRR@20 |
|---|---:|---:|---:|---:|---:|---:|
| BPR-MF | 0.7965 | 0.5063 | 0.4159 | 0.9178 | 0.5372 | 0.4246 |

These metrics were computed on a fixed set of 101 candidates and must not be compared directly with full-catalog ranking metrics.

## List Quality after Re-ranking Top 100 to Top 20

The re-ranking weights are relevance `0.70`, calibration `0.15`, and diversity `0.15`. Calibration is Jensen-Shannon similarity; ILD is the mean pairwise cosine distance between genre multi-hot vectors.

| Model | Calibration | ILD |
|---|---:|---:|
| BPR-MF | 0.8738 | 0.7159 |
