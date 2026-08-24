[中文](RESEARCH_PROTOCOL.md) | [English](RESEARCH_PROTOCOL.en.md) | [Deutsch](RESEARCH_PROTOCOL.de.md)

# Final Thesis Experiment Protocol

This protocol corresponds to `recsys20m.thesis_pipeline` and the data label
`thesis_pos4_neg2_traincore5_dataseed2026`. The earlier plan A and the single-seed K=20 results are not part of the final thesis evidence.

## Research Questions

- How much test-set diversity gain can be achieved under validation NDCG loss budgets of 1%, 3%, 5%, and 10%?
- How do MMR, xQuAD, and calibration re-ranking affect ILD, user-interest coverage, and agreement with historical interest proportions?
- Are the conclusions robust to the training seed, candidate-set size N, list length K, user group, and feature space?
- Does local list diversity improve catalog coverage, long-tail exposure, and overall exposure concentration?

## Data and Leakage Prevention

- MovieLens 20M; `rating >= 4` is positive feedback, `rating <= 2` is explicit negative feedback, and `2.5-3.5` is neutral.
- Each user's positive feedback is split chronologically into approximately 80/10/10 train/validation/test partitions.
- The iterative 5-core is computed **only from the predefined chronologically earlier training partition**, so future interactions cannot determine the training population.
- User genre profiles use positive training interactions only.
- Validation retrieval excludes every item rated in the training time window; test retrieval also excludes the validation time window.
- With probability 0.5, BPR negative sampling uses an explicit negative from the user's training period; otherwise it uses a genuinely unrated item.

Final scale: 134,703 users, 11,851 items, and 9,952,928 positive interactions; 7,908,519 train, 939,551 validation, and 1,104,858 test interactions.

## Model and Candidates

- BPR-MF with 64-dimensional embeddings, 10 epochs, batch size 8,192, and CUDA.
- Training seeds: 2026, 2027, and 2028; the data seed is fixed at 2026.
- Primary experiment: N=100 and K=10; robustness checks: N in {50, 100, 200} and K in {5, 10, 20}.
- Candidate Recall is reported separately to distinguish the retrieval ceiling from losses introduced by re-ranking.

## Re-ranking Methods

- MMR: relevance balanced against the genre cosine distance to the closest item already selected.
- xQuAD: rewards user-interest aspects that remain insufficiently covered, using the user's historical genre prior.
- Calibration: balances relevance with the Jensen-Shannon similarity between the user's historical genre distribution and the recommendation-list distribution.
- Primary lambda grid: 0.00, 0.05, ..., 1.00; robustness grid step: 0.10.

## Selection and Statistics

For budget b and method m, only validation configurations satisfying
`NDCG_m(lambda) >= (1-b) * NDCG_baseline` are retained. Among them, the configuration with the highest ILD is selected; ties are resolved by higher NDCG and then lower lambda. Test data is never used for selection.

- 200 user-level paired bootstrap samples per primary configuration; 100 for robustness configurations.
- Reporting includes 95% confidence intervals, the paired sign test, and Cohen's dz.
- Claims across multiple budgets use Holm correction; cross-seed reporting uses the mean, seed standard deviation, and the most conservative p-value.

## Metrics

- Accuracy: NDCG@K, Recall@K, HR@K, MRR@K, and Candidate Recall@N.
- List/profile: ILD, Feature-ILD, Calibration, JS distance, genre entropy, genre count, and profile-weighted Subtopic Recall.
- System: Catalog Coverage, Genre Coverage, Exposure Gini, Long-tail share, runtime, and Python traced-memory peak.
- Groups: low/medium/high activity and focused/medium/broad profile tertiles.

## Tag Genome Sensitivity

A GPU randomized uncentered SVD64 is applied to the MovieLens Tag Genome. It covers 9,864 of 11,851 items and retains 89.24% of the Frobenius energy. The genre and Tag Genome variants use the same tag-complete candidate pool so that missing features cannot become a confounder.

## Execution and Artifacts

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Aggregate artifacts are stored in:

`outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/`

`experiment_manifest.json` records all parameters, `all_thesis_results.csv` contains 686 complete configuration rows, and `validation_budget_selections.csv` is kept separate from `test_budget_results.csv` to separate model selection from final reporting.
