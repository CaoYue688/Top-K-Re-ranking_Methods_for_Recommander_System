[中文](RESEARCH_RESULTS.md) | [English](RESEARCH_RESULTS.en.md) | [Deutsch](RESEARCH_RESULTS.de.md)

# Final Thesis Experiment Results (Audited)

Data label: `thesis_pos4_neg2_traincore5_dataseed2026`. The audit confirmed 686 complete configuration rows, no duplicate composite keys, no missing core metrics for the applicable K, and identical lambda=0 baselines across all three methods.

## Primary Baseline (Test, Mean over Three Seeds)

| Metric | Value |
|---|---:|
| Recall@10 | 0.060897 |
| NDCG@10 | 0.058480 |
| ILD@10 | 0.669365 |
| Calibration@10 | 0.819102 |
| Subtopic Recall@10 | 0.862119 |
| Catalog Coverage@10 | 0.482772 |
| Exposure Gini@10 | 0.962414 |
| Long-tail share@10 | 0.016793 |
| Candidate Recall@100 | 0.276126 |

## Cross-Method Validation-Budget Selection

MMR is selected for all four budgets:

| Validation budget | Lambda | Relative Test NDCG change | Relative Test ILD change |
|---:|---:|---:|---:|
| 1% | 0.25 | -0.84% | +8.85% |
| 3% | 0.35 | -2.49% | +13.67% |
| 5% | 0.40 | -3.80% | +16.30% |
| 10% | 0.55 | -8.92% | +22.84% |

At the central 5% operating point, the means are NDCG@10=0.056257, ILD@10=0.778464, Calibration@10=0.841053, Catalog Coverage@10=0.497370, and Long-tail share@10=0.020364. The conservative sign-test and Holm-adjusted directional conclusions are significant; the mean Cohen's dz values for NDCG and ILD are approximately -0.049 and 1.386, respectively.

## Method Differences within the 5% Budget

- **MMR lambda=0.40**: highest pairwise genre ILD, +16.30%; NDCG -3.80%.
- **xQuAD lambda=0.80**: smaller ILD gain, but profile-weighted Subtopic Recall of approximately 0.943; most suitable for coverage of interest aspects.
- **Calibration lambda=0.85**: Calibration of approximately 0.899 and the smallest NDCG loss; most suitable for matching historical interest proportions.

There is therefore no objective-independent "best algorithm": MMR, xQuAD, and calibration optimize different diversity constructs.

## Robustness

MMR under seed 2026 and the 5% validation budget:

| Setting | Relative Test NDCG change | Relative Test ILD change |
|---|---:|---:|
| N=50, K=10 | -2.84% | +13.44% |
| N=100, K=10 | -4.06% | +16.34% |
| N=200, K=10 | -2.45% | +13.02% |
| N=100, K=5 | -4.16% | +26.05% |
| N=100, K=20 | -3.43% | +10.20% |

On the tag-complete candidate pool, genre-MMR produces a 16.32% Feature-ILD gain, while Tag Genome SVD64-MMR produces 13.47%; both incur approximately 4.1% Test NDCG loss. The qualitative conclusion is stable, but the effect size depends on the feature space.

## User-Group and Catalog Effects

- ILD gains are approximately 16.1-16.5% in all three activity groups.
- Focused profiles have a larger ILD gain (18.27%) but also the largest NDCG loss (5.16%).
- Broad profiles have a 14.77% ILD gain and a 3.04% NDCG loss.
- Catalog Coverage rises from 48.28% to 49.74%, and Long-tail share rises from 1.68% to 2.04%.
- Exposure Gini changes only from 0.962414 to 0.962382, showing that local list diversity cannot replace a global exposure-fairness objective.

## Auditable Files

- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/all_thesis_results.csv`
- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/validation_budget_selections.csv`
- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/test_budget_results.csv`
- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/experiment_manifest.json`

The earlier `RESULTS.md` and older outputs are historical baselines and do not replace these final audited results.
