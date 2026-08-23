from __future__ import annotations

# Dieses Modul erzeugt leicht zitierbare Tabellen und SVG-Abbildungen ohne Zusatzpakete.
# 本模块无需额外依赖即可生成易于引用的表格和 SVG 图。
# This module creates easily citable tables and SVG figures without extra packages.
from html import escape
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_COLORS = {
    "diversity": "#2563eb",
    "mmr": "#dc2626",
    "calibration": "#059669",
    "combined": "#7c3aed",
}


def _axis_bounds(values: np.ndarray) -> tuple[float, float]:
    # Ein kleiner Rand verhindert, dass Punkte direkt auf dem Rahmen liegen.
    # 小边距可避免数据点直接落在边框上。
    # A small margin prevents points from lying directly on the frame.
    minimum = float(np.nanmin(values))
    maximum = float(np.nanmax(values))
    span = maximum - minimum
    padding = span * 0.08 if span > 0 else max(abs(minimum) * 0.05, 0.01)
    return minimum - padding, maximum + padding


def write_tradeoff_svgs(
    results: pd.DataFrame,
    output_dir: Path,
    cutoff: int = 20,
    split: str = "test",
) -> list[Path]:
    # Pro Kandidatenmodell entsteht eine Accuracy-Diversity-Kurve für den Testsplit.
    # 为每个候选模型生成测试分割的准确率-多样性曲线。
    # Creates a test-split accuracy-diversity curve for each candidate model.
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    width, height = 920, 600
    left, right, top, bottom = 100, 35, 55, 85
    plot_width = width - left - right
    plot_height = height - top - bottom
    x_column = f"ndcg@{cutoff}"
    y_column = f"ild@{cutoff}"

    for model in sorted(results["model"].unique()):
        frame = results[
            (results["model"] == model) & (results["split"] == split)
        ].copy()
        if frame.empty:
            continue
        x_min, x_max = _axis_bounds(frame[x_column].to_numpy())
        y_min, y_max = _axis_bounds(frame[y_column].to_numpy())

        def x_position(value: float) -> float:
            return left + (value - x_min) / (x_max - x_min) * plot_width

        def y_position(value: float) -> float:
            return top + (y_max - value) / (y_max - y_min) * plot_height

        svg: list[str] = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" '
                f'height="{height}" viewBox="0 0 {width} {height}">'
            ),
            '<rect width="100%" height="100%" fill="white"/>',
            (
                '<style>text{font-family:Arial,sans-serif;fill:#111827}'
                '.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#374151;'
                'stroke-width:1.5}.curve{fill:none;stroke-width:2.5}</style>'
            ),
            (
                f'<text x="{width / 2}" y="30" text-anchor="middle" '
                f'font-size="19" font-weight="700">{escape(model.upper())}: '
                f'Accuracy–Diversity Trade-off ({escape(split)})</text>'
            ),
        ]
        for tick in range(6):
            fraction = tick / 5
            x = left + fraction * plot_width
            y = top + (1.0 - fraction) * plot_height
            x_value = x_min + fraction * (x_max - x_min)
            y_value = y_min + fraction * (y_max - y_min)
            svg.extend(
                [
                    (
                        f'<line class="grid" x1="{x:.2f}" y1="{top}" '
                        f'x2="{x:.2f}" y2="{top + plot_height}"/>'
                    ),
                    (
                        f'<line class="grid" x1="{left}" y1="{y:.2f}" '
                        f'x2="{left + plot_width}" y2="{y:.2f}"/>'
                    ),
                    (
                        f'<text x="{x:.2f}" y="{top + plot_height + 25}" '
                        f'text-anchor="middle" font-size="12">{x_value:.4f}</text>'
                    ),
                    (
                        f'<text x="{left - 12}" y="{y + 4:.2f}" '
                        f'text-anchor="end" font-size="12">{y_value:.4f}</text>'
                    ),
                ]
            )
        svg.extend(
            [
                (
                    f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
                    f'x2="{left + plot_width}" y2="{top + plot_height}"/>'
                ),
                (
                    f'<line class="axis" x1="{left}" y1="{top}" '
                    f'x2="{left}" y2="{top + plot_height}"/>'
                ),
                (
                    f'<text x="{left + plot_width / 2}" y="{height - 28}" '
                    f'text-anchor="middle" font-size="15">NDCG@{cutoff} '
                    '(higher is better)</text>'
                ),
                (
                    f'<text x="25" y="{top + plot_height / 2}" '
                    f'text-anchor="middle" font-size="15" '
                    f'transform="rotate(-90 25 {top + plot_height / 2})">'
                    f'Genre ILD@{cutoff} (higher is better)</text>'
                ),
            ]
        )

        # Linien verbinden zunehmende Lambda-Werte; große Ringe markieren die Pareto-Front.
        # 线连接递增的 lambda 值；大圆环标记 Pareto 前沿。
        # Lines connect increasing lambda values; large rings mark the Pareto frontier.
        legend_x = left + 15
        legend_y = top + 18
        for method_index, (method, method_frame) in enumerate(
            frame.groupby("method", sort=True)
        ):
            method_frame = method_frame.sort_values("lambda")
            color = METHOD_COLORS.get(str(method), "#4b5563")
            points = " ".join(
                f"{x_position(float(row[x_column])):.2f},"
                f"{y_position(float(row[y_column])):.2f}"
                for _, row in method_frame.iterrows()
            )
            svg.append(
                f'<polyline class="curve" points="{points}" stroke="{color}"/>'
            )
            for _, row in method_frame.iterrows():
                x = x_position(float(row[x_column]))
                y = y_position(float(row[y_column]))
                is_pareto = bool(row.get("global_pareto", row.get("pareto", False)))
                if is_pareto:
                    svg.append(
                        f'<circle cx="{x:.2f}" cy="{y:.2f}" r="7" fill="white" '
                        f'stroke="{color}" stroke-width="2"/>'
                    )
                svg.append(
                    f'<circle cx="{x:.2f}" cy="{y:.2f}" r="3.5" fill="{color}">'
                    f'<title>{escape(str(method))}, λ={float(row["lambda"]):.2f}, '
                    f'NDCG={float(row[x_column]):.5f}, '
                    f'ILD={float(row[y_column]):.5f}</title></circle>'
                )
            current_y = legend_y + method_index * 23
            svg.extend(
                [
                    (
                        f'<line x1="{legend_x}" y1="{current_y}" '
                        f'x2="{legend_x + 25}" y2="{current_y}" '
                        f'stroke="{color}" stroke-width="3"/>'
                    ),
                    (
                        f'<text x="{legend_x + 34}" y="{current_y + 4}" '
                        f'font-size="13">{escape(str(method))}</text>'
                    ),
                ]
            )
        svg.append("</svg>")
        path = output_dir / f"{model}_{split}_accuracy_diversity.svg"
        path.write_text("\n".join(svg), encoding="utf-8")
        paths.append(path)
    return paths


def write_experiment_summary(
    results: pd.DataFrame,
    selections: dict[str, object],
    stats: dict[str, object],
    output_path: Path,
) -> Path:
    # Die Markdown-Datei fasst Datengröße und ausschließlich auf Validation gewählte Punkte zusammen.
    # Markdown 文件汇总数据规模和仅基于验证集选择的点。
    # The Markdown file summarizes data size and points selected only on validation.
    lines = [
        "# Accuracy–Diversity experiment summary",
        "",
        "## Dataset",
        "",
        f"- Positive rating threshold: `{stats.get('positive_threshold')}`",
        f"- Positive 5-core users: `{int(stats['n_users']):,}`",
        f"- Positive 5-core items: `{int(stats['n_items']):,}`",
        f"- Positive interactions: `{int(stats['core_interactions']):,}`",
        (
            "- Chronological split: "
            f"`{int(stats['train_interactions']):,}` train / "
            f"`{int(stats['val_interactions']):,}` validation / "
            f"`{int(stats['test_interactions']):,}` test"
        ),
        "",
        "## Validation-selected operating points",
        "",
        "| Model | Method | λ | NDCG@20 baseline → selected | ΔNDCG | ILD@20 baseline → selected | ΔILD | Calibration@20 |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("mf",):
        selection = selections.get(model)
        if not isinstance(selection, dict) or not isinstance(selection.get("test"), dict):
            continue
        test = selection["test"]
        baseline = results[
            (results["model"] == model)
            & (results["split"] == "test")
            & np.isclose(results["lambda"], 0.0)
        ].iloc[0]
        ndcg_change = (
            float(test["ndcg@20"]) / float(baseline["ndcg@20"]) - 1.0
        )
        ild_change = float(test["ild@20"]) / float(baseline["ild@20"]) - 1.0
        lines.append(
            f"| {model} | {selection['selected_method']} | "
            f"{float(selection['selected_lambda']):.2f} | "
            f"{float(baseline['ndcg@20']):.5f} → {float(test['ndcg@20']):.5f} | "
            f"{ndcg_change:+.2%} | "
            f"{float(baseline['ild@20']):.5f} → {float(test['ild@20']):.5f} | "
            f"{ild_change:+.2%} | "
            f"{float(test['calibration@20']):.5f} |"
        )
    test_rows = results[results["split"] == "test"]
    lines.extend(
        [
            "",
            "The operating point is selected on validation by maximizing ILD@20 "
            "under the configured NDCG-loss budget. Test labels are not used for tuning.",
            "",
            "Candidate Recall@100 on Test: "
            + ", ".join(
                (
                    f"`{model}={test_rows[(test_rows['model'] == model) & np.isclose(test_rows['lambda'], 0.0)].iloc[0]['candidate_recall@100']:.5f}`"
                )
                for model in ("mf",)
            )
            + ".",
            "",
            f"Total evaluated test configurations: `{len(test_rows):,}`.",
            "Pareto-optimal points are marked in `all_tradeoff_results.csv` and the SVG figures.",
            "",
        ]
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path
