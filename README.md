[English](README.md) | [Deutsch](README.de.md) | [中文](README.zh-CN.md)

# Diversity-Oriented Re-ranking for Recommender Systems

This repository contains the reproducible implementation used for the Master's thesis on diversity-oriented top-K re-ranking with MovieLens 20M. It compares MMR, xQuAD, and calibration under explicit accuracy-loss budgets.

> The thesis results are produced by `recsys20m.thesis_pipeline`. The older `recsys20m.pipeline` is retained only as a legacy engineering baseline and is not the evidential basis of the thesis.

## What is implemented

- chronological train/validation/test splitting with a train-only iterative 5-core,
- BPR-MF training with three fixed random seeds,
- Tag Genome feature construction with a 64-dimensional SVD,
- deterministic top-N candidate generation,
- re-ranking with MMR, xQuAD, and calibration,
- evaluation of accuracy, diversity, calibration, popularity bias, and runtime,
- accuracy-loss budgets of 1%, 3%, 5%, and 10%,
- robustness analyses for different candidate and recommendation-list sizes,
- paired bootstrap confidence intervals and Holm-adjusted confirmatory tests.

## Final experiment configuration

| Item | Setting |
|---|---|
| Dataset | MovieLens 20M |
| Positive feedback | rating >= 4.0 |
| Explicit negative feedback | rating <= 2.0 |
| User/item filtering | iterative 5-core, fitted only on training data |
| Split | chronological train/validation/test |
| Training seeds | 2026, 2027, 2028 |
| BPR-MF | 10 epochs, 64 latent factors, batch size 8192 |
| Negative sampling | 50% explicit negatives where available |
| Feature space | MovieLens Tag Genome, SVD dimension 64 |
| Retrieval pool | top 200 internally; primary evaluation uses N = 100 |
| Recommendation length | primary K = 10 |
| Re-rankers | MMR, xQuAD, calibration |
| Lambda grid | primary step 0.05; robustness step 0.10 |
| Accuracy-loss budgets | 1%, 3%, 5%, 10% |
| Bootstrap | 200 primary and 100 robustness samples |
| Robustness grid | N in {50, 100, 200}; K in {5, 10, 20} |
| Compute target | CUDA GPU by default; CPU is supported but much slower |

The complete preregistered-style procedure and metric definitions are documented in [RESEARCH_PROTOCOL.md](RESEARCH_PROTOCOL.md).

## Requirements and setup

Required:

- Python 3.11 or newer,
- the official MovieLens 20M dataset,
- a CUDA-capable GPU for the intended runtime profile; CPU execution is possible.

Windows PowerShell:

```powershell
py -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -e .
```

Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

If the automatically installed PyTorch build does not match the local CUDA setup, install the appropriate PyTorch build for that system and then repeat the editable installation. `Pillow` is declared because the public figure-generation script imports it.

## Data preparation

Download and extract MovieLens 20M from the official GroupLens source. The repository does not redistribute the dataset.

Place the extracted files here:

```text
data/raw/ml-20m/
├── ratings.csv
├── movies.csv
├── genome-scores.csv
└── genome-tags.csv
```

Expected archive checksum:

```text
SHA256 ml-20m.zip
96F1322B342E074A2B251BB4C1E1990AB58082C228A430029A258A4E4393F51A
```

## Reproduce the thesis pipeline

From the repository root on Windows:

```powershell
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root .
```

For a CPU-only run:

```powershell
.\.venv\Scripts\python.exe -m recsys20m.thesis_pipeline --root . --device cpu
```

For a clean rerun that replaces cached intermediate results, add `--force`.

After the experiment has finished, summarize the aggregate outputs and regenerate the figures:

```powershell
.\.venv\Scripts\python.exe scriptssummarize_thesis_results.py
.\.venv\Scripts\python.exe scriptsgenerate_thesis_figures.py
```

On Linux/macOS, replace `.\.venv\Scripts\python.exe` with `.venv/bin/python`.

## Main outputs

The final pipeline writes its aggregate results below `outputs/thesis/aggregate/`. Important files include:

| Output | Purpose |
|---|---|
| `summary_seed_level.csv` | seed-level aggregate metrics |
| `budget_selection_seed_level.csv` | selected configurations under each accuracy budget |
| `accuracy_comparison_seed_level.csv` | accuracy comparison with the BPR-MF baseline |
| `robustness_summary_seed_level.csv` | robustness results across N and K |
| `robustness_budget_selection_seed_level.csv` | robustness selections under the budgets |
| `runtime_seed_level.csv` | measured execution times |
| `figures/` | generated thesis figures |

Generated data, checkpoints, and experiment outputs are excluded from Git because they are reproducible and can be large.

## Tests

Run the complete public test suite on Windows:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tests -v
```

Linux/macOS:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

## Research documentation

- Experiment protocol: [English](RESEARCH_PROTOCOL.en.md) | [Deutsch](RESEARCH_PROTOCOL.de.md) | [中文](RESEARCH_PROTOCOL.md)
- Audited final results: [English](RESEARCH_RESULTS.en.md) | [Deutsch](RESEARCH_RESULTS.de.md) | [中文](RESEARCH_RESULTS.md)
- [RESULTS.md](RESULTS.md): legacy baseline output; it must not be confused with the final thesis results.

## Repository structure

```text
src/recsys20m/                 core implementation
scripts/                       reproducible result summarization and figure generation
tests/                         public automated tests
RESEARCH_PROTOCOL*.md          experiment protocol in Chinese, English, and German
RESEARCH_RESULTS*.md           final result report in Chinese, English, and German
pyproject.toml                 package metadata and canonical dependencies
requirements.txt               convenience dependency list
LICENSE                        MIT License
```
## Legacy baseline

The following command runs the earlier engineering baseline:

```powershell
.\.venv\Scripts\python.exe -m recsys20m.pipeline --root .
```

That path uses a smaller configuration (including three training epochs, sampled evaluation, and top-20 output) and is kept for historical traceability and tests. Do not use `RESULTS.md` or this legacy command as the source for claims about the final thesis experiment.

## Archived thesis version

The immutable software version associated with the submitted thesis is published as the annotated tag and GitHub Release [`thesis-v1.0`](https://github.com/CaoYue688/Top-K-Re-ranking_Methods_for_Recommander_System/releases/tag/thesis-v1.0). The thesis appendix records the exact 40-character commit identifier. Reproducibility references should cite that commit or release instead of the mutable `main` branch.

## License

The source code is released under the [MIT License](LICENSE). MovieLens data remains subject to the separate GroupLens terms.
## Data terms

MovieLens data is subject to the GroupLens dataset terms. Obtain the dataset from the official source and observe its usage and citation requirements.
