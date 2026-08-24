[English](README.md) | [Deutsch](README.de.md) | [中文](README.zh-CN.md)

# 推荐系统中的多样性导向重排序

本仓库包含硕士论文中使用的可复现实验实现。研究基于 MovieLens 20M，在明确限制准确率损失的条件下比较 MMR、xQuAD 和校准式重排序。

> 论文最终结果由 `recsys20m.thesis_pipeline` 生成。较早的 `recsys20m.pipeline` 仅作为历史工程基线保留，不是论文结论的证据来源。

## 已实现的内容

- 按时间顺序划分训练集、验证集和测试集，并且只在训练数据上迭代执行 5-core 过滤；
- 使用三个固定随机种子训练 BPR-MF；
- 基于 MovieLens Tag Genome 和 64 维 SVD 构建特征空间；
- 确定性生成 Top-N 候选集；
- 使用 MMR、xQuAD 和校准方法进行重排序；
- 评估准确率、多样性、校准程度、流行度偏差和运行时间；
- 使用 1%、3%、5% 和 10% 的准确率损失预算；
- 针对不同候选集大小和推荐列表长度进行稳健性分析；
- 使用配对 Bootstrap 置信区间和 Holm 校正的验证性检验。

## 最终实验配置

| 项目 | 设置 |
|---|---|
| 数据集 | MovieLens 20M |
| 正反馈 | 评分 >= 4.0 |
| 明确负反馈 | 评分 <= 2.0 |
| 用户/物品过滤 | 仅在训练数据上拟合的迭代 5-core |
| 数据划分 | 按时间顺序划分训练/验证/测试 |
| 训练随机种子 | 2026、2027、2028 |
| BPR-MF | 10 个 epoch、64 个隐因子、batch size 8192 |
| 负样本采样 | 条件允许时，50% 使用明确负反馈 |
| 特征空间 | MovieLens Tag Genome，SVD 维数 64 |
| 召回候选池 | 内部取 Top 200；主实验使用 N = 100 |
| 推荐列表长度 | 主实验 K = 10 |
| 重排序方法 | MMR、xQuAD、校准 |
| Lambda 网格 | 主实验步长 0.05；稳健性分析步长 0.10 |
| 准确率损失预算 | 1%、3%、5%、10% |
| Bootstrap | 主实验 200 次，稳健性分析 100 次 |
| 稳健性网格 | N 属于 {50, 100, 200}；K 属于 {5, 10, 20} |
| 计算设备 | 默认 CUDA GPU；支持 CPU，但速度明显更慢 |

完整实验步骤和指标定义见 [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md)。

## 环境要求与安装

需要：

- Python 3.11 或更高版本；
- 官方 MovieLens 20M 数据集；
- 为获得预期运行时间，建议使用支持 CUDA 的 GPU；也可以仅使用 CPU。

Windows PowerShell：

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Linux/macOS：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

如果自动安装的 PyTorch 与本机 CUDA 环境不匹配，请先安装适合该系统的 PyTorch 版本，再重新执行可编辑安装命令。由于公开的图片生成脚本使用了 `Pillow`，该库现已被明确写入依赖。

## 数据准备

请从 GroupLens 官方来源下载并解压 MovieLens 20M。本仓库不重新分发该数据集。

文件应放在：

```text
data/raw/ml-20m/
├── ratings.csv
├── movies.csv
├── genome-scores.csv
└── genome-tags.csv
```

预期压缩包校验值：

```text
SHA256 ml-20m.zip
96F1322B342E074A2B251BB4C1E1990AB58082C228A430029A258A4E4393F51A
```

## 复现论文最终实验

在仓库根目录中执行以下 Windows 命令：

```powershell
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root .
```

仅使用 CPU：

```powershell
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root . --device cpu
```

如果需要忽略缓存并重新运行所有步骤，请加上 `--force`。

实验完成后，汇总结果并重新生成论文图片：

```powershell
.\.venv\Scripts\python.exe scriptssummarize_thesis_results.py
.\.venv\Scripts\python.exe scriptsgenerate_thesis_figures.py
```

在 Linux/macOS 下，将 `.\.venv\Scripts\python.exe` 替换为 `.venv/bin/python`。

## 主要输出

最终流程将聚合结果写入 `outputs/thesis/aggregate/`。

| 输出 | 用途 |
|---|---|
| `summary_seed_level.csv` | 每个随机种子的聚合指标 |
| `budget_selection_seed_level.csv` | 各准确率损失预算下选出的配置 |
| `accuracy_comparison_seed_level.csv` | 与 BPR-MF 基线的准确率比较 |
| `robustness_summary_seed_level.csv` | 不同 N 和 K 下的稳健性结果 |
| `robustness_budget_selection_seed_level.csv` | 各预算下的稳健性配置选择 |
| `runtime_seed_level.csv` | 实际测量的运行时间 |
| `figures/` | 生成的论文图片 |

生成的数据、中间模型和实验输出体积较大并且可以重现，因此不提交到 Git。

## 测试

Windows 下运行完整的公开测试：

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS：

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## 研究文档

- 实验协议：[中文](RESEARCH_PROTOCOL.md) | [English](RESEARCH_PROTOCOL.en.md) | [Deutsch](RESEARCH_PROTOCOL.de.md)
- 经审计的最终结果：[中文](RESEARCH_RESULTS.md) | [English](RESEARCH_RESULTS.en.md) | [Deutsch](RESEARCH_RESULTS.de.md)
- 历史基线：[中文](RESULTS.md) | [English](RESULTS.en.md) | [Deutsch](RESULTS.de.md)，不能与最终论文结果混淆。

## 仓库结构

```text
src/recsys20m/                 核心实现
scripts/                       可复现的结果汇总和图片生成脚本
tests/                         公开自动化测试
RESEARCH_PROTOCOL*.md          中、英、德三语实验协议
RESEARCH_RESULTS*.md           中、英、德三语最终结果报告
pyproject.toml                 包信息和唯一正式依赖定义
requirements.txt               转交给 pyproject.toml 的便捷安装入口
LICENSE                        MIT 许可证
```
## 历史基线

较早的工程基线仍可通过以下命令运行：

```powershell
.\.venv\Scripts\python.exe -m recsys20m.pipeline --root .
```

该流程采用较小的配置，包括三个训练 epoch、采样评估和 Top-20 输出。保留它是为了历史可追溯性和测试。论文关于最终实验的论述不能以 `RESULTS.md` 或该旧流程作为证据。

## 论文固定版本

与提交论文对应的不可变代码版本以带注释的 Tag 和 GitHub Release [`thesis-v1.0`](https://github.com/CaoYue688/Top-K-Re-ranking_Methods_for_Recommander_System/releases/tag/thesis-v1.0) 发布。论文附录 B 还会记录完整的 40 位 commit 标识。复现或引用时应使用该 commit 或 Release，而不是会继续变化的 `main` 分支。

## 许可证

源代码采用 [MIT License](LICENSE)。MovieLens 数据仍独立受 GroupLens 数据使用条款约束。
## 数据使用条款

MovieLens 数据受 GroupLens 数据集条款约束。请从官方来源获取数据，并遵守其使用和引用要求。
