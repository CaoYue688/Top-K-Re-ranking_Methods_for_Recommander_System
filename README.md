# MovieLens 20M：MF + Two-Tower 推荐流水线

这是一个可复现的隐式反馈推荐基线，包含：

- GroupLens 官方 MovieLens 20M 下载与 ZIP 校验；
- 迭代式 5-core 过滤（每个保留用户、物品均至少 5 次交互）；
- 每个用户内部按时间排序后约 80% / 10% / 10% 划分；
- 验证、测试各使用“最新 1 个正样本 + 100 个全历史未交互负样本”；
- 仅用训练集构造用户 genre 分布画像；
- BPR Matrix Factorization 与带 genre 特征的 Two-Tower DNN；
- 用户/物品 embedding、分块物品内积近邻和两个模型各自的 Top-100；
- 用 relevance + calibration + ILD 的贪心目标重排为 Top-20。

## 已采用的口径

所有评分（包括低分）均代表一次发生过的交互，因此都作为隐式正反馈。时间切分在用户内部完成；对于只有 5–9 条历史的用户，会优先保证 train、validation、test 都非空，所以这些用户的个人比例无法精确等于 80/10/10，但全量总体比例会接近目标。

Calibration 使用用户训练历史 genre 分布与推荐列表 genre 分布之间的 Jensen–Shannon 相似度：

`Calibration = 1 - JSD(P_user, Q_rec) / ln(2)`

ILD 使用推荐物品 genre 多热向量两两余弦距离的均值，因此两个 baseline 可以在同一个内容语义空间中公平比较。Two-Tower 的输出已经 L2 归一化，其另行导出的物品内积近邻中，内积即余弦相似度。

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

# 预处理、5-core、时间切分、评估负采样与 genre 画像
.\.venv\Scripts\python.exe -m recsys20m.preprocess

# 两个召回模型
.\.venv\Scripts\python.exe -m recsys20m.models mf --epochs 3
.\.venv\Scripts\python.exe -m recsys20m.models two-tower --epochs 3

# 101 个采样候选上的排序评估
.\.venv\Scripts\python.exe -m recsys20m.evaluation mf --split test
.\.venv\Scripts\python.exe -m recsys20m.evaluation two-tower --split test

# 全物品精确 Top-100（屏蔽训练集已交互物品）
.\.venv\Scripts\python.exe -m recsys20m.retrieval recommend mf
.\.venv\Scripts\python.exe -m recsys20m.retrieval recommend two-tower

# Two-Tower 物品 embedding 的分块内积 Top-200
.\.venv\Scripts\python.exe -m recsys20m.retrieval item-similarity

# 两个 Top-100 分别重排为 Top-20
.\.venv\Scripts\python.exe -m recsys20m.rerank mf
.\.venv\Scripts\python.exe -m recsys20m.rerank two-tower
```

## 主要产物

| 路径 | 内容 |
|---|---|
| `data/processed/{train,val,test}.npz` | 时间切分后的交互 |
| `data/processed/eval_candidates.npz` | 每用户 1 正 + 100 负候选 |
| `data/processed/user_genre_profiles.npy` | 用户 genre 分布 |
| `artifacts/mf_embeddings.npz` | MF 用户/物品 embedding 与物品偏置 |
| `artifacts/two-tower_embeddings.npz` | 双塔用户/物品 embedding |
| `outputs/*_top100.npz` | 每模型 Top-100 候选 |
| `outputs/two-tower_item_neighbors_top200.npz` | 物品内积近邻 |
| `outputs/*_top20_reranked.npz` | 每模型重排 Top-20 |
| `outputs/*_top20_quality.json` | Calibration 与 ILD 汇总 |
| `outputs/*_top20_sample.csv` | 前 100 位用户的可读推荐样例 |

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
