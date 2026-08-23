"""Generate publication PNG figures from the audited thesis result CSVs.

The implementation deliberately uses Pillow, which is part of the bundled
document runtime, so figure generation does not depend on a GUI or TeX stack.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
AGG = ROOT / "outputs" / "thesis_pos4_neg2_traincore5_dataseed2026" / "aggregate"
OUT = ROOT / "outputs" / "thesis" / "figures"
COLORS = {"mmr": "#0072B2", "xquad": "#D55E00", "calibration": "#009E73"}
LABELS = {"mmr": "MMR", "xquad": "xQuAD", "calibration": "Kalibrierung"}
FONT = Path(r"C:\Windows\Fonts\arial.ttf")
BOLD = Path(r"C:\Windows\Fonts\arialbd.ttf")


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(str(BOLD if bold else FONT), size)


def canvas(title: str, width: int = 2400, height: int = 1450) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    im = Image.new("RGB", (width, height), "white")
    d = ImageDraw.Draw(im)
    d.text((width // 2, 65), title, font=font(48, True), fill="#1F2933", anchor="ma")
    return im, d


def save(im: Image.Image, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    im.save(OUT / name, dpi=(300, 300), optimize=True)


def text_center(d: ImageDraw.ImageDraw, xy: tuple[int, int], text: str, size: int = 34, fill: str = "#1F2933") -> None:
    d.multiline_text(xy, text, font=font(size), fill=fill, anchor="mm", align="center", spacing=int(size * 0.25))


def arrow(d: ImageDraw.ImageDraw, start: tuple[int, int], end: tuple[int, int]) -> None:
    d.line([start, end], fill="#43505A", width=7)
    x2, y2 = end
    x1, y1 = start
    angle = np.arctan2(y2 - y1, x2 - x1)
    length = 28
    for delta in (2.55, -2.55):
        p = (int(x2 + length * np.cos(angle + delta)), int(y2 + length * np.sin(angle + delta)))
        d.line([end, p], fill="#43505A", width=7)


def pipeline_figure() -> None:
    im, d = canvas("Experimentelle Verarbeitungskette", height=1050)
    boxes = [
        (80, 250, 410, 500, "MovieLens 20M\nRatings + Metadaten"),
        (480, 250, 810, 500, "Chronologischer Split\nTrain-Kern + Val/Test"),
        (880, 250, 1210, 500, "BPR-MF\n3 Seeds, GPU"),
        (1280, 250, 1610, 500, "Top-N-Kandidaten\nN=50/100/200"),
        (1680, 250, 2300, 500, "Re-Ranking\nMMR / xQuAD / Kalibrierung"),
        (460, 680, 1040, 910, "Diversitätsräume\nGenre / Tag Genome SVD"),
        (1250, 680, 2040, 910, "Offline-Evaluation\nAccuracy · Diversity · Bias · Laufzeit"),
    ]
    for x1, y1, x2, y2, txt in boxes:
        d.rounded_rectangle((x1, y1, x2, y2), radius=24, fill="#EAF2F8", outline="#2C3E50", width=5)
        text_center(d, ((x1 + x2) // 2, (y1 + y2) // 2), txt)
    for s, e in [
        ((410, 375), (480, 375)), ((810, 375), (880, 375)), ((1210, 375), (1280, 375)),
        ((1610, 375), (1680, 375)), ((750, 680), (750, 500)), ((1990, 500), (1680, 680)),
        ((1040, 795), (1250, 795)),
    ]:
        arrow(d, s, e)
    save(im, "01_experiment_pipeline.png")


def draw_axes(
    d: ImageDraw.ImageDraw,
    xlim: tuple[float, float],
    ylim: tuple[float, float],
    xlabel: str,
    ylabel: str,
    plot: tuple[int, int, int, int] = (300, 180, 2250, 1190),
    x_ticks: int = 6,
    y_ticks: int = 6,
) -> tuple:
    left, top, right, bottom = plot
    d.line((left, top, left, bottom), fill="#1F2933", width=5)
    d.line((left, bottom, right, bottom), fill="#1F2933", width=5)
    for value in np.linspace(ylim[0], ylim[1], y_ticks):
        y = int(bottom - (value - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top))
        d.line((left, y, right, y), fill="#D9DEE3", width=2)
        d.text((left - 24, y), f"{value:.1f}", font=font(29), fill="#333333", anchor="rm")
    for value in np.linspace(xlim[0], xlim[1], x_ticks):
        x = int(left + (value - xlim[0]) / (xlim[1] - xlim[0]) * (right - left))
        d.line((x, bottom, x, bottom + 12), fill="#1F2933", width=3)
        d.text((x, bottom + 28), f"{value:.1f}", font=font(29), fill="#333333", anchor="ma")
    d.text(((left + right) // 2, bottom + 105), xlabel, font=font(34), fill="#1F2933", anchor="ma")
    d.text((left, top - 18), ylabel, font=font(31), fill="#1F2933", anchor="lb")
    return (
        lambda x: int(left + (x - xlim[0]) / (xlim[1] - xlim[0]) * (right - left)),
        lambda y: int(bottom - (y - ylim[0]) / (ylim[1] - ylim[0]) * (bottom - top)),
    )


def tradeoff_curves(data: pd.DataFrame) -> None:
    d0 = data[(data.experiment == "primary_n100_k10_genre") & (data.split == "test")]
    agg = d0.groupby(["method", "lambda"], as_index=False)[["ndcg@10", "ild@10"]].mean()
    series = {}
    for method in ["mmr", "xquad", "calibration"]:
        g = agg[agg.method == method].sort_values("lambda")
        base = g[g["lambda"] == 0].iloc[0]
        series[method] = (
            100 * (1 - g["ndcg@10"].to_numpy() / base["ndcg@10"]),
            100 * (g["ild@10"].to_numpy() / base["ild@10"] - 1),
        )
    xmin = min(float(x.min()) for x, _ in series.values())
    xmax = max(float(x.max()) for x, _ in series.values())
    ymin = min(float(y.min()) for _, y in series.values())
    ymax = max(float(y.max()) for _, y in series.values())
    im, dr = canvas("Accuracy–Diversity-Trade-off im Hauptversuch (Mittel über drei Seeds)")
    sx, sy = draw_axes(
        dr,
        (np.floor(xmin / 5) * 5, np.ceil(xmax / 5) * 5),
        (np.floor(ymin / 5) * 5, np.ceil(ymax / 5) * 5),
        "relativer Verlust NDCG@10 [%]",
        "relative Steigerung ILD@10 [%]",
    )
    for budget in [1, 3, 5, 10]:
        x = sx(budget)
        dr.line((x, 180, x, 1190), fill="#8B949E", width=2)
        dr.text((x + 8, 205), f"{budget}%", font=font(26), fill="#555555")
    lx, ly = 1660, 240
    for idx, method in enumerate(["mmr", "xquad", "calibration"]):
        xs, ys = series[method]
        pts = [(sx(float(x)), sy(float(y))) for x, y in zip(xs, ys)]
        dr.line(pts, fill=COLORS[method], width=7, joint="curve")
        for p in pts:
            dr.ellipse((p[0] - 7, p[1] - 7, p[0] + 7, p[1] + 7), fill=COLORS[method])
        yy = ly + idx * 58
        dr.line((lx, yy, lx + 70, yy), fill=COLORS[method], width=8)
        dr.text((lx + 90, yy), LABELS[method], font=font(30), fill="#222222", anchor="lm")
    save(im, "02_primary_tradeoff.png")


def grouped_bar(
    title: str,
    categories: list[str],
    groups: list[tuple[str, list[float], str]],
    ylabel: str,
    name: str,
    zero_line: bool = False,
) -> None:
    im, d = canvas(title)
    all_values = [v for _, vals, _ in groups for v in vals]
    ymin = min(0.0, min(all_values))
    ymax = max(all_values)
    span = max(ymax - ymin, 1e-6)
    ymin -= 0.12 * span if ymin < 0 else 0
    ymax += 0.18 * span
    left, top, right, bottom = 250, 220, 2260, 1160
    d.line((left, top, left, bottom), fill="#1F2933", width=5)
    d.line((left, bottom, right, bottom), fill="#1F2933", width=5)
    for value in np.linspace(ymin, ymax, 6):
        y = int(bottom - (value - ymin) / (ymax - ymin) * (bottom - top))
        d.line((left, y, right, y), fill="#D9DEE3", width=2)
        d.text((left - 20, y), f"{value:.1f}", font=font(29), fill="#333333", anchor="rm")
    if zero_line:
        y0 = int(bottom - (0 - ymin) / (ymax - ymin) * (bottom - top))
        d.line((left, y0, right, y0), fill="#555555", width=4)
    ncat = len(categories)
    cat_w = (right - left) / ncat
    group_w = cat_w * 0.72 / len(groups)
    for gi, (label, vals, color) in enumerate(groups):
        for ci, value in enumerate(vals):
            center = left + cat_w * (ci + 0.5)
            x1 = int(center - cat_w * 0.36 + gi * group_w)
            x2 = int(x1 + group_w * 0.9)
            yv = int(bottom - (value - ymin) / (ymax - ymin) * (bottom - top))
            yz = int(bottom - (0 - ymin) / (ymax - ymin) * (bottom - top))
            d.rectangle((x1, min(yv, yz), x2, max(yv, yz)), fill=color)
        yleg = 205 + gi * 52
        d.rectangle((1750, yleg - 14, 1800, yleg + 14), fill=color)
        d.text((1820, yleg), label, font=font(28), fill="#222222", anchor="lm")
    for ci, category in enumerate(categories):
        center = int(left + cat_w * (ci + 0.5))
        text_center(d, (center, bottom + 90), category, size=27)
    d.text((left, top - 18), ylabel, font=font(31), fill="#1F2933", anchor="lb")
    save(im, name)


def budget_figure() -> None:
    d = pd.read_csv(AGG / "test_budget_results.csv")
    d = d[d.scope == "primary_ild_across_methods"].sort_values("budget").copy()
    if d.empty:
        raise ValueError("No primary_ild_across_methods rows found in test_budget_results.csv")
    baseline = d["test_ndcg@10_mean"] - d["delta_ndcg@10_mean"]
    loss = 100 * (-d["delta_ndcg@10_mean"] / baseline)
    ild_base = d["test_ild@10_mean"] - d["delta_ild@10_mean"]
    gain = 100 * d["delta_ild@10_mean"] / ild_base
    grouped_bar(
        "Validierungsselektierte MMR-Betriebspunkte auf dem Testset",
        [f"{int(100*b)}%\nλ={l:.2f}" for b, l in zip(d.budget, d["lambda"])],
        [("NDCG-Verlust", loss.tolist(), "#CC79A7"), ("ILD-Steigerung", gain.tolist(), "#0072B2")],
        "relative Änderung [%]", "03_budget_operating_points.png",
    )


def robustness_figure(data: pd.DataFrame) -> None:
    settings = [
        ("robust_n50_k10_genre", "N=50\nK=10"),
        ("primary_n100_k10_genre", "N=100\nK=10"),
        ("robust_n200_k10_genre", "N=200\nK=10"),
        ("robust_n100_k5_genre", "N=100\nK=5"),
        ("robust_n100_k20_genre", "N=100\nK=20"),
    ]
    values: dict[str, list[float]] = {m: [] for m in COLORS}
    for experiment, _ in settings:
        for method in values:
            val = data[(data.experiment == experiment) & (data.split == "val") & (data.method == method) & (data.seed == 2026)]
            k = int(val.top_k.iloc[0])
            base = val[val["lambda"] == 0].iloc[0]
            feasible = val[val[f"ndcg@{k}"] >= 0.95 * base[f"ndcg@{k}"]]
            sel = feasible.sort_values([f"ild@{k}", f"ndcg@{k}"], ascending=False).iloc[0]
            test = data[(data.experiment == experiment) & (data.split == "test") & (data.method == method) & (data.seed == 2026)]
            chosen = test[np.isclose(test["lambda"], sel["lambda"], atol=1e-7)].iloc[0]
            tb = test[test["lambda"] == 0].iloc[0]
            values[method].append(100 * (chosen[f"ild@{k}"] / tb[f"ild@{k}"] - 1))
    grouped_bar(
        "Robustheit bei 5% Validierungsbudget (Seed 2026)",
        [label for _, label in settings],
        [(LABELS[m], values[m], COLORS[m]) for m in ["mmr", "xquad", "calibration"]],
        "relative Steigerung ILD@K [%]", "04_robustness.png",
    )


def subgroup_figure(data: pd.DataFrame) -> None:
    q = data[(data.experiment == "primary_n100_k10_genre") & (data.split == "test") & (data.method == "mmr")]
    chosen = q[np.isclose(q["lambda"], 0.4, atol=1e-7)]
    base = q[q["lambda"] == 0]
    groups = ["low_activity", "medium_activity", "high_activity", "focused_profile", "medium_profile", "broad_profile"]
    labels = ["geringe\nAktivität", "mittlere\nAktivität", "hohe\nAktivität", "fokussiertes\nProfil", "mittleres\nProfil", "breites\nProfil"]
    ndcg = [100 * (chosen[f"{g}_ndcg@10"].mean() / base[f"{g}_ndcg@10"].mean() - 1) for g in groups]
    ild = [100 * (chosen[f"{g}_ild@10"].mean() / base[f"{g}_ild@10"].mean() - 1) for g in groups]
    grouped_bar(
        "MMR λ=0,40 nach Nutzergruppen (Mittel über drei Seeds)", labels,
        [("NDCG@10", ndcg, "#CC79A7"), ("ILD@10", ild, "#0072B2")],
        "relative Änderung [%]", "05_subgroups.png", zero_line=True,
    )


def tag_figure(data: pd.DataFrame) -> None:
    labels, losses, gains = [], [], []
    for experiment, label in [
        ("tag_sensitivity_n100_k10_genre", "Genre-Raum"),
        ("tag_sensitivity_n100_k10_tag_genome_svd64", "Tag Genome\nSVD64"),
    ]:
        val = data[(data.experiment == experiment) & (data.split == "val") & (data.method == "mmr") & (data.seed == 2026)]
        base = val[val["lambda"] == 0].iloc[0]
        feasible = val[val["ndcg@10"] >= 0.95 * base["ndcg@10"]]
        sel = feasible.sort_values(["feature_ild@10", "ndcg@10"], ascending=False).iloc[0]
        test = data[(data.experiment == experiment) & (data.split == "test") & (data.method == "mmr") & (data.seed == 2026)]
        chosen = test[np.isclose(test["lambda"], sel["lambda"], atol=1e-7)].iloc[0]
        tb = test[test["lambda"] == 0].iloc[0]
        labels.append(f"{label}\nλ={sel['lambda']:.2f}")
        losses.append(100 * (1 - chosen["ndcg@10"] / tb["ndcg@10"]))
        gains.append(100 * (chosen["feature_ild@10"] / tb["feature_ild@10"] - 1))
    grouped_bar(
        "Sensitivität gegenüber der Diversitätsrepräsentation", labels,
        [("NDCG-Verlust", losses, "#CC79A7"), ("Merkmals-ILD-Steigerung", gains, "#009E73")],
        "relative Änderung [%]", "06_tag_sensitivity.png",
    )


def exposure_figure(data: pd.DataFrame) -> None:
    q = data[(data.experiment == "primary_n100_k10_genre") & (data.split == "test") & (data.method == "mmr")]
    base = q[q["lambda"] == 0]
    chosen = q[np.isclose(q["lambda"], 0.4, atol=1e-7)]
    before = [base["catalog_coverage@10"].mean(), base["long_tail_share@10"].mean(), 1 - base["exposure_gini@10"].mean()]
    after = [chosen["catalog_coverage@10"].mean(), chosen["long_tail_share@10"].mean(), 1 - chosen["exposure_gini@10"].mean()]
    grouped_bar(
        "Aggregierte Verteilungseffekte", ["Katalog-\nabdeckung", "Long-Tail-\nAnteil", "1 − Exposure-\nGini"],
        [("Baseline", before, "#999999"), ("MMR λ=0,40", after, "#0072B2")],
        "Anteil / normierter Wert", "07_exposure.png",
    )


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data = pd.read_csv(AGG / "all_thesis_results.csv")
    pipeline_figure()
    tradeoff_curves(data)
    budget_figure()
    robustness_figure(data)
    subgroup_figure(data)
    tag_figure(data)
    exposure_figure(data)
    print(f"[OK] wrote {len(list(OUT.glob('*.png')))} figures to {OUT}")


if __name__ == "__main__":
    main()
