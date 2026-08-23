"""Regenerate the seven experiment figures with Chinese labels."""

from __future__ import annotations

from pathlib import Path

import generate_thesis_figures as figures


ROOT = Path(__file__).resolve().parents[1]
TRANSLATIONS = {
    "Experimentelle Verarbeitungskette": "实验处理流程",
    "MovieLens 20M\nRatings": "MovieLens 20M\n评分",
    "Chronologischer Split\nTrain-Kern\nValidierung | Test": "时间顺序划分\n训练 Core\n验证 | 测试",
    "BPR-MF\nTraining: Train-Kern\n3 Seeds, GPU": "BPR-MF\n仅训练 Core\n3 个随机种子，GPU",
    "Kandidatenpools\nN = 50 / 100 / 200\nfür Validierung und Test": "候选池\nN = 50 / 100 / 200\n用于验证与测试",
    "MovieLens 20M\nMetadaten": "MovieLens 20M\n元数据",
    "Diversitätsräume\nGenre | Tag Genome\n(SVD)": "多样性空间\nGenre | Tag Genome\n(SVD)",
    "Validierungsselektion\nλ, N und Methode\nnur auf Validierung": "验证集选择\nλ、N 与方法\n仅使用验证集",
    "Fixiertes Re-Ranking\nMMR | xQuAD |\nKalibrierung, K = 10": "固定重排序\nMMR | xQuAD |\n校准，K = 10",
    "Finale Offline-Evaluation auf Testdaten\nNDCG | ILD | Subtopic Recall | Kalibrierung\nPopularitätsbias | Laufzeit": "测试集最终离线评价\nNDCG | ILD | Subtopic Recall | 校准\n流行度偏差 | 运行时间",
    "Accuracy–Diversity-Trade-off im Hauptversuch (Mittel über drei Seeds)": "主实验准确性-多样性权衡（三个种子均值）",
    "relativer Verlust NDCG@10 [%]": "NDCG@10 相对损失 [%]",
    "relative Steigerung ILD@10 [%]": "ILD@10 相对增长 [%]",
    "Kalibrierung": "校准",
    "Validierungsselektierte MMR-Betriebspunkte auf dem Testset": "验证集选择的 MMR 运行点在测试集上的表现",
    "NDCG-Verlust": "NDCG 损失",
    "ILD-Steigerung": "ILD 增长",
    "relative Änderung [%]": "相对变化 [%]",
    "Robustheit bei 5% Validierungsbudget (Seed 2026)": "5% 验证预算下的稳健性（Seed 2026）",
    "relative Steigerung ILD@K [%]": "ILD@K 相对增长 [%]",
    "MMR λ=0,40 nach Nutzergruppen (Mittel über drei Seeds)": "MMR λ=0.40 的用户组结果（三个种子均值）",
    "geringe\nAktivität": "低\n活跃度",
    "mittlere\nAktivität": "中\n活跃度",
    "hohe\nAktivität": "高\n活跃度",
    "fokussiertes\nProfil": "集中\n画像",
    "mittleres\nProfil": "中等\n画像",
    "breites\nProfil": "宽\n画像",
    "Sensitivität gegenüber der Diversitätsrepräsentation": "对多样性特征表示的敏感性",
    "Genre-Raum": "Genre 空间",
    "Merkmals-ILD-Steigerung": "Feature-ILD 增长",
    "Aggregierte Verteilungseffekte": "聚合分布效应",
    "Katalog-\nabdeckung": "目录\n覆盖率",
    "Long-Tail-\nAnteil": "长尾\n占比",
    "1 − Exposure-\nGini": "1 - Exposure-\nGini",
    "Anteil / normierter Wert": "占比 / 归一化数值",
}


ORIGINAL_CANVAS = figures.canvas
ORIGINAL_DRAW_AXES = figures.draw_axes
ORIGINAL_GROUPED_BAR = figures.grouped_bar
ORIGINAL_TEXT_CENTER = figures.text_center


def tr(value: str) -> str:
    result = TRANSLATIONS.get(value, value)
    for source, target in (
        ("Genre-Raum", "Genre 空间"),
        ("Baseline", "基线"),
        ("MMR λ=0,40", "MMR λ=0.40"),
    ):
        result = result.replace(source, target)
    return result


def canvas_zh(title: str, width: int = 2400, height: int = 1450, title_size: int = 48):
    return ORIGINAL_CANVAS(tr(title), width=width, height=height, title_size=title_size)


def draw_axes_zh(d, xlim, ylim, xlabel, ylabel, **kwargs):
    return ORIGINAL_DRAW_AXES(d, xlim, ylim, tr(xlabel), tr(ylabel), **kwargs)


def grouped_bar_zh(title, categories, groups, ylabel, name, zero_line=False):
    return ORIGINAL_GROUPED_BAR(
        tr(title),
        [tr(value) for value in categories],
        [(tr(label), values, color) for label, values, color in groups],
        tr(ylabel),
        name,
        zero_line=zero_line,
    )


def text_center_zh(d, xy, text, size=34, fill="#1F2933"):
    return ORIGINAL_TEXT_CENTER(d, xy, tr(text), size=size, fill=fill)


def main() -> None:
    figures.OUT = ROOT / "outputs" / "thesis_zh" / "figures"
    figures.FONT = Path(r"C:\Windows\Fonts\simhei.ttf")
    figures.BOLD = Path(r"C:\Windows\Fonts\simhei.ttf")
    figures.LABELS["calibration"] = "校准"
    figures.canvas = canvas_zh
    figures.draw_axes = draw_axes_zh
    figures.grouped_bar = grouped_bar_zh
    figures.text_center = text_center_zh
    figures.main()


if __name__ == "__main__":
    main()
