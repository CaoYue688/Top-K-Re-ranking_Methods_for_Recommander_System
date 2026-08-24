[中文](RESEARCH_RESULTS.md) | [English](RESEARCH_RESULTS.en.md) | [Deutsch](RESEARCH_RESULTS.de.md)

# 最终论文实验结果（审计版）

对应数据标签：`thesis_pos4_neg2_traincore5_dataseed2026`。审计确认 686 行配置完整、组合键无重复，针对各自 K 的核心指标无缺失，三种方法在 λ=0 时基线完全一致。

## 主实验基线（Test，三随机种子均值）

| 指标 | 值 |
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

## 跨方法 Validation 预算选择

四个预算都选择 MMR：

| Validation 预算 | λ | Test NDCG 相对变化 | Test ILD 相对变化 |
|---:|---:|---:|---:|
| 1% | 0.25 | −0.84% | +8.85% |
| 3% | 0.35 | −2.49% | +13.67% |
| 5% | 0.40 | −3.80% | +16.30% |
| 10% | 0.55 | −8.92% | +22.84% |

5% 核心点的均值为 NDCG@10=0.056257、ILD@10=0.778464、Calibration@10=0.841053、Catalog Coverage@10=0.497370、Long-tail share@10=0.020364。保守 sign-test 与 Holm 校正后的方向结论均显著；NDCG/ILD 的平均 Cohen dz 分别约为 −0.049 与 1.386。

## 5% 预算内的方法差异

- **MMR λ=0.40**：最高的成对 genre ILD，+16.30%；NDCG −3.80%。
- **xQuAD λ=0.80**：ILD 增益较小，但 profile-weighted Subtopic Recall 约 0.943，最适合兴趣方面覆盖。
- **Calibration λ=0.85**：Calibration 约 0.899，且 NDCG 损失最小，最适合历史兴趣比例匹配。

因此不存在与目标无关的“最佳算法”：MMR、xQuAD 与 Calibration 分别优化不同的多样性构念。

## 稳健性

Seed 2026、5% Validation 预算下的 MMR：

| 设置 | Test NDCG 相对变化 | Test ILD 相对变化 |
|---|---:|---:|
| N=50, K=10 | −2.84% | +13.44% |
| N=100, K=10 | −4.06% | +16.34% |
| N=200, K=10 | −2.45% | +13.02% |
| N=100, K=5 | −4.16% | +26.05% |
| N=100, K=20 | −3.43% | +10.20% |

Tag-complete 候选池上，Genre-MMR 的 feature-ILD 增益为 16.32%，Tag Genome SVD64-MMR 为 13.47%；两者 Test NDCG 损失约 4.1%。定性结论稳定，但效应大小依赖特征空间。

## 用户群组与目录效应

- 活跃度三组的 ILD 增益均约 16.1–16.5%。
- focused profile 的 ILD 增益更高（18.27%），但 NDCG 损失也最高（5.16%）。
- broad profile 的 ILD 增益为 14.77%，NDCG 损失为 3.04%。
- Catalog Coverage 从 48.28% 升至 49.74%，Long-tail share 从 1.68% 升至 2.04%。
- Exposure Gini 仅从 0.962414 变为 0.962382，说明局部列表多样性不能替代全局曝光公平目标。

## 可复核文件

- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/all_thesis_results.csv`
- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/validation_budget_selections.csv`
- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/test_budget_results.csv`
- `outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/experiment_manifest.json`

论文 DOCX 位于 `outputs/thesis/`。早期 `RESULTS.md` 与旧输出仅为历史基线，不能替代本页结果。
