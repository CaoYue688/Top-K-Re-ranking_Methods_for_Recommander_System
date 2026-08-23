# MovieLens 20M：BPR-MF 推荐与重排序流水线

这是一个可复现的隐式反馈推荐基线，包含：

- GroupLens 官方 MovieLens 20M 下载与 ZIP 校验；
- 在 `rating >= 4` 正反馈上执行迭代式 5-core；
- 每个用户内部按时间排序后约 80% / 10% / 10% 划分；
- BPR 训练混合 50% 明确负反馈（`rating <= 2`）与 50% 真正未评分物品；
- 验证、测试各使用“最新 1 个正样本 + 100 个全历史未评分负样本”；
- 仅用训练集构造用户 genre 分布画像；
- BPR Matrix Factorization；
- MF 用户/物品 embedding、分块物品内积近邻和 Top-100；
- 用 relevance + calibration + ILD 的贪心目标重排为 Top-20。

## 已采用的口径

项目采用三段式反馈口径：`rating >= 4` 是正反馈，`rating <= 2` 是明确负反馈，`2.5–3.5` 是中性反馈。Train / Validation / Test 只切分正反馈；低分和中性评分仍保留，用于确保“未评分负样本”确实从未被该用户评分。明确负反馈只从对应用户的训练时间窗口抽取，避免使用未来评分。

BPR 默认按 `0.5` 的概率为有低分记录的用户抽取明确负样本，其余情况抽取训练窗口内真正未评分的物品。没有明确负反馈的用户会自动退回未评分采样。对于只有 5–9 条正反馈历史的用户，会优先保证 train、validation、test 都非空，因此个人比例无法精确等于 80/10/10，但全量总体比例会接近目标。

Calibration 使用用户训练历史 genre 分布与推荐列表 genre 分布之间的 Jensen–Shannon 相似度：

`Calibration = 1 - JSD(P_user, Q_rec) / ln(2)`

ILD 使用推荐物品 genre 多热向量两两余弦距离的均值。物品相似度分析则对 MF 物品 embedding 归一化后计算内积，因此内积等于余弦相似度。

完整 `27k × 27k` float32 相似度矩阵约 3 GB。默认按块计算并保存每个物品最相似的 200 个邻居，信息足以用于检索/分析，也避免无谓的稠密矩阵落盘。

## 环境

本项目的 Windows 虚拟环境已位于 `.venv`。从头安装可运行：

```powershell
python -m venv --system-site-packages .venv
.\.venv\Scripts\python.exe -m pip install --index-url https://download.pytorch.org/whl/cpu torch
$env:PYTHONPATH = "$PWD\src"
```

## 一次运行全部步骤

生产基线（默认 3 个 epoch，每个 epoch 覆盖约一个训练集大小）：

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m recsys20m.pipeline --root .
```

快速验证整个流程：

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m recsys20m.pipeline --root . --epochs 1 --steps-per-epoch 100
```

固定随机种子默认为 `2026`。CPU 是默认设备；有 CUDA 时可以传 `--device cuda`。

## 分阶段运行

```powershell
$env:PYTHONPATH = "$PWD\src"

# 预处理、三段式反馈、5-core、时间切分、评估负采样与 genre 画像
.\.venv\Scripts\python.exe -m recsys20m.preprocess `
  --positive-threshold 4 --negative-threshold 2

# 训练 BPR-MF；明确负样本目标占比为 50%
.\.venv\Scripts\python.exe -m recsys20m.models `
  --epochs 3 --explicit-negative-ratio 0.5

# 101 个采样候选上的排序评估
.\.venv\Scripts\python.exe -m recsys20m.evaluation mf --split test

# 全物品精确 Top-100（屏蔽训练集已交互物品）
.\.venv\Scripts\python.exe -m recsys20m.retrieval recommend mf

# MF 物品 embedding 的分块内积 Top-200
.\.venv\Scripts\python.exe -m recsys20m.retrieval item-similarity

# 将 MF Top-100 重排为 Top-20
.\.venv\Scripts\python.exe -m recsys20m.rerank mf
```

## 主要产物

| 路径 | 内容 |
|---|---|
| `data/processed/{train,val,test}.npz` | 时间切分后的交互 |
| `data/processed/train_explicit_negatives.npz` | 训练时间窗口中的 `rating <= 2` 明确负反馈 |
| `data/processed/train_seen_keys.npy` | 训练窗口内所有已评分用户-物品对，用于排除伪负样本 |
| `data/processed/eval_candidates.npz` | 每用户 1 正 + 100 负候选 |
| `data/processed/user_genre_profiles.npy` | 用户 genre 分布 |
| `artifacts/mf_embeddings.npz` | MF 用户/物品 embedding 与物品偏置 |
| `outputs/mf_top100.npz` | MF Top-100 候选 |
| `outputs/mf_item_neighbors_top200.npz` | MF 物品内积近邻 |
| `outputs/mf_top20_reranked.npz` | MF 重排 Top-20 |
| `outputs/mf_top20_quality.json` | Calibration 与 ILD 汇总 |
| `outputs/mf_top20_sample.csv` | 前 100 位用户的可读推荐样例 |

## 测试

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

MovieLens 数据的使用遵循压缩包内 `README.txt` 所述条款。

当前下载文件 `data/raw/ml-20m.zip` 的 SHA-256 为：

`96F243C338A8665F6BCC89C53EDF6EE39162A846940DE6B7C8C48AEADA765FF3`

可复跑最终一致性检查：

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python.exe -m recsys20m.validate --root .
```

## Accuracy–Diversity 论文实验（最终口径）

最终论文流水线与早期基线目录隔离。它使用训练分区上的迭代 5-core、三段式评分口径、
50/50 混合 BPR 负采样、三个训练随机种子、MMR/xQuAD/Calibration、1/3/5/10% NDCG
预算、用户级 paired bootstrap、Holm 校正、N/K 稳健性和 Tag Genome SVD64 敏感性分析。

```powershell
$env:PYTHONPATH = "$PWD\src"

# 完整论文实验；默认 CUDA，约需较长时间
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root .

# 从最终 CSV 重新生成审计摘要和论文图
.\.venv\Scripts\python.exe scripts\summarize_thesis_results.py
& 'C:\Users\Aroeh\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe' `
  scripts\generate_thesis_figures.py
```

最终总表位于
`outputs/thesis_pos4_neg2_traincore5_dataseed2026/aggregate/all_thesis_results.csv`；
完整实验协议和审计后的结果分别见 [`RESEARCH_PROTOCOL.md`](RESEARCH_PROTOCOL.md) 与
[`RESEARCH_RESULTS.md`](RESEARCH_RESULTS.md)。早期方案 A/Top-20 数值只保留作历史记录，
不得用于最终论文结论。
