[中文](RESEARCH_PROTOCOL.md) | [English](RESEARCH_PROTOCOL.en.md) | [Deutsch](RESEARCH_PROTOCOL.de.md)

# 最终论文实验协议

本协议对应 `recsys20m.thesis_pipeline` 和数据标签
`thesis_pos4_neg2_traincore5_dataseed2026`。旧的方案 A、K=20 单随机种子结果不属于最终论文证据。

## 研究问题

- 在 1%、3%、5%、10% 的 Validation NDCG 损失预算下，能够获得多少 Test 多样性增益？
- MMR、xQuAD、校准重排分别如何影响 ILD、用户兴趣覆盖和历史兴趣比例匹配？
- 结论是否对训练随机种子、候选数 N、列表长度 K、用户群组和特征空间稳健？
- 局部多样性是否改善目录覆盖、长尾曝光和总体曝光集中度？

## 数据与防泄漏

- MovieLens 20M；`rating >= 4` 为正反馈，`rating <= 2` 为明确负反馈，`2.5–3.5` 为中性。
- 每个用户按时间构造约 80/10/10 的正反馈 Train/Validation/Test。
- 迭代 5-core **只根据预定的 chronologically earlier Train 分区计算**，避免未来交互决定训练人口。
- 用户 genre 画像只使用正向 Train。
- Validation 检索屏蔽 Train 时间窗内所有已评分物品；Test 还屏蔽 Validation 时间窗。
- BPR 负采样以 0.5 概率使用用户训练期明确负反馈，否则使用真正未评分物品。

最终规模：134,703 用户、11,851 物品、9,952,928 个正反馈；Train 7,908,519、Validation 939,551、Test 1,104,858。

## 模型与候选

- BPR-MF，embedding 64，10 epochs，batch 8,192，CUDA。
- 训练随机种子：2026、2027、2028；数据随机种子固定 2026。
- 主实验 N=100、K=10；稳健性检查 N∈{50,100,200}、K∈{5,10,20}。
- Candidate Recall 单独报告，区分召回上限与重排损失。

## 重排方法

- MMR：相关性与距已选列表最近对象的 genre 余弦距离。
- xQuAD：按用户历史 genre 先验，奖励尚未充分覆盖的兴趣方面。
- Calibration：相关性与用户历史/推荐列表 genre 分布的 Jensen–Shannon 相似度。
- 主实验 λ=0.00,0.05,…,1.00；稳健性实验步长 0.10。

## 选参与统计

对预算 b 和方法 m，只在 Validation 上保留
`NDCG_m(λ) >= (1-b) * NDCG_baseline` 的配置，并在其中选择 ILD 最大者；平局依次选择更高 NDCG、更小 λ。Test 不参与选择。

- 主实验每配置 200 次用户级 paired bootstrap；稳健性 100 次。
- 报告 95% CI、paired sign test、Cohen dz。
- 多预算结论用 Holm 校正；跨随机种子使用均值、seed 标准差和最保守 p 值。

## 指标

- Accuracy：NDCG@K、Recall@K、HR@K、MRR@K、Candidate Recall@N。
- 列表/画像：ILD、Feature-ILD、Calibration、JS distance、genre entropy、genre count、profile-weighted Subtopic Recall。
- 系统：Catalog Coverage、Genre Coverage、Exposure Gini、Long-tail share、运行时间和 Python traced-memory peak。
- 群组：低/中/高活跃度与 focused/medium/broad profile tertiles。

## Tag Genome 敏感性

对 MovieLens Tag Genome 做 GPU randomised uncentred SVD64，覆盖 9,864/11,851 个物品，保留 89.24% Frobenius energy。Genre 与 Tag 版本使用同一 tag-complete 候选池，避免缺失特征成为混杂因素。

## 运行与产物

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root .
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

聚合产物位于：

`outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/`

其中 `experiment_manifest.json` 记录全部参数，`all_thesis_results.csv` 含 686 个完整配置行，`validation_budget_selections.csv` 与 `test_budget_results.csv` 分离选参与最终报告。
