"""Build the complete Chinese-language thesis PDF from the translated manuscript."""

from __future__ import annotations

import argparse
from pathlib import Path

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

import build_thesis_pdf as base


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "outputs" / "thesis_zh" / "Diversitaetsorientiertes_Re-Ranking_Masterarbeit_Yue_Cao_ZH.pdf"
ORIGINAL_MAKE_TABLES = base.make_tables
ORIGINAL_BUILD_STYLES = base.build_styles


FOOTNOTES_ZH = {
    "[[FN001]]": "参见 McNee/Riedl/Konstan (2006)，第 1097-1101 页。",
    "[[FN002]]": "参见 Rendle/Freudenthaler/Gantner/Schmidt-Thieme (2009)，第 452-461 页。",
    "[[FN003]]": "参见 Koren/Bell/Volinsky (2009)，第 30-37 页。",
    "[[FN004]]": "参见 Kaminskas/Bridge (2017)，第 1-42 页。",
    "[[FN005]]": "参见 Steck (2018)，第 154-162 页。",
    "[[FN006]]": "参见 Carbonell/Goldstein (1998)，第 335-336 页。",
    "[[FN007]]": "参见 Santos/Peng/Macdonald/Ounis (2010)，第 87-99 页。",
    "[[FN008]]": "参见 Krichene/Rendle (2020)，第 1748-1757 页。",
    "[[FN009]]": "参见 Abdollahpouri/Burke/Mobasher (2017)，第 42-46 页。",
    "[[FN010]]": "参见 Herlocker/Konstan/Terveen/Riedl (2004)，第 5-53 页。",
    "[[FN011]]": "参见 Ziegler/McNee/Konstan/Lausen (2005)，第 22-32 页。",
    "[[FN012]]": "参见 Vargas/Castells (2011)，第 109-116 页。",
    "[[FN013]]": "参见 Steck (2018)，第 154-162 页。",
    "[[FN014]]": "参见 Vig/Sen/Riedl (2012)，第 1-44 页。",
    "[[FN015]]": "参见 Zangerle/Bauer (2022)，第 1-38 页。",
    "[[FN016]]": "Bootstrap 方法参见 Efron/Tibshirani (1993)；多重检验校正参见 Holm (1979)，第 65-70 页；标准化效应量参见 Cohen (1988)。",
    "[[FN017]]": "参见 Harper/Konstan (2015)，第 1-19 页，以及 GroupLens 的 MovieLens 20M 官方数据说明。",
    "[[FN018]]": "参见 Jannach/Chen (2026)，第 1-15 页。",
    "[[FN019]]": "参见 Herlocker/Konstan/Terveen/Riedl (2004)，第 5-53 页，以及 Zangerle/Bauer (2022)，第 1-38 页。",
    "[[FN020]]": "参见 Carbonell/Goldstein (1998)，第 335-336 页。",
    "[[FN021]]": "参见 Kaya/Bridge (2019b)，第 1639-1646 页。",
    "[[FN022]]": "参见 Kaya/Bridge (2019a)，第 151-159 页。",
    "[[FN023]]": "参见 Wang 等 (2023)，第 223-233 页。",
    "[[FN024]]": "参见 Hidasi/Czapp (2023)。",
    "[[FN025]]": "参见 Kaya/Bridge (2019a)，第 151-159 页，以及 Carraro/Bridge (2026)，第 1-40 页。",
    "[[FN026]]": "参见 Kaya/Bridge (2019a)，第 151-159 页。",
    "[[FN027]]": "参见 Wang 等 (2023)，第 223-233 页。",
    "[[FN028]]": "参见 Hidasi/Czapp (2023)。",
    "[[FN029]]": "参见 Carraro/Bridge (2026)，第 1-40 页。",
}


TABLE_TITLES_ZH = {
    "dataset_stats": "按时间划分并在训练集构建 5-Core 后的数据统计",
    "baseline_metrics": "主实验中三个随机种子的平均基线指标",
    "budget_results": "跨方法预算选择在测试集上的结果",
    "method_comparison_5pct": "各方法在 5% 验证预算内的测试运行点",
    "construct_comparison_5pct": "各方法在 5% 验证预算内按相应构念选择的测试运行点",
    "coverage_comparison_5pct": "采用相同目标选择规则时的兴趣方面覆盖",
    "subgroup_results": "MMR λ=0.40 在不同活跃度与兴趣宽度组中的结果",
    "segment_lambdas": "5% 准确性预算下各用户画像分群的 MMR 运行点",
    "robustness_results": "5% 验证预算与 Seed 2026 下的稳健性矩阵",
    "candidate_frontier": "不同候选池大小下经验证集选择的 MMR 运行点",
    "tag_results": "MMR 权衡对特征空间的敏感性",
    "runtime_results": "主参数扫描的实测运行时间",
    "hypotheses": "研究假设检验汇总",
    "protocol_deviations": "相对于开题报告的方案偏差及其处理",
    "ai_use": "生成式人工智能支持记录",
    "parameters": "最终实验运行参数",
    "all_budget_methods": "各预算和方法的验证选择运行点",
}


HEADERS_ZH = {
    "dataset_stats": ["指标", "数值"],
    "baseline_metrics": ["指标", "平均值"],
    "budget_results": ["预算", "方法", "λ", "NDCG", "NDCG 相对变化", "ILD", "ILD 相对变化"],
    "method_comparison_5pct": ["方法", "λ", "NDCG", "ILD", "校准", "覆盖率", "长尾占比"],
    "construct_comparison_5pct": ["方法", "目标", "λ", "NDCG", "Δ NDCG", "目标值", "Δ 目标值"],
    "coverage_comparison_5pct": ["方法", "λ", "Subtopic Recall", "Δ Subtopic", "Δ NDCG"],
    "subgroup_results": ["用户组", "人数", "NDCG 相对变化", "ILD 相对变化"],
    "segment_lambdas": ["画像分群", "λ", "验证集 Δ NDCG", "验证集 Δ ILD"],
    "robustness_results": ["设置", "λ", "NDCG 相对变化", "ILD 相对变化", "Candidate Recall"],
    "candidate_frontier": ["N", "λ", "验证集 NDCG", "验证集 ILD", "ILD 是否高于前一个 N"],
    "tag_results": ["特征空间", "λ", "NDCG 相对变化", "Feature-ILD", "Feature-ILD 相对变化"],
    "runtime_results": ["方法", "扫描用时（秒）", "毫秒/用户/配置", "峰值内存（MB）"],
    "hypotheses": ["假设", "陈述", "结论", "证据"],
    "protocol_deviations": ["方面", "原方案", "最终处理", "影响"],
    "ai_use": ["工作环节", "AI 支持", "作者控制"],
    "parameters": ["参数", "数值"],
    "all_budget_methods": ["预算", "范围", "方法", "λ", "目标", "测试 NDCG", "测试 ILD"],
}


CELL_TRANSLATIONS = {
    "Rohbewertungen": "原始评分",
    "Positive Rohinteraktionen (≥4)": "原始正交互（≥4）",
    "Nutzende nach Core": "Core 后用户数",
    "Filme nach Core": "Core 后电影数",
    "Positive Interaktionen nach Core": "Core 后正交互",
    "Training": "训练",
    "Validierung": "验证",
    "Test": "测试",
    "Explizite Negativinteraktionen": "显式负交互",
    "Neutrale Interaktionen": "中性交互",
    "Genremerkmale": "电影类型特征",
    "Core-Basis": "Core 构建依据",
    "chronologische Trainingspartition": "按时间划分的训练部分",
    "Kalibrierungsähnlichkeit@10": "校准相似度@10",
    "Subtopic Recall@10": "Subtopic Recall@10",
    "Katalogcoverage@10": "目录覆盖率@10",
    "Exposure-Gini@10": "Exposure-Gini@10",
    "Long-Tail-Anteil@10": "长尾占比@10",
    "Kalibrierung": "校准",
    "KALIBRIERUNG": "校准",
    "XQUAD": "xQuAD",
    "Kal.": "校准",
    "unterstützt": "支持",
    "nicht unterstützt": "不支持",
    "ja": "是",
    "nein": "否",
    "primär über Methoden": "跨方法主分析",
    "ILD je Methode": "各方法 ILD",
    "Konstrukt je Methode": "各方法相应构念",
    "Geringe Aktivität": "低活跃度",
    "Mittlere Aktivität": "中活跃度",
    "Hohe Aktivität": "高活跃度",
    "Fokussiertes Profil": "集中兴趣画像",
    "Mittleres Profil": "中等兴趣画像",
    "Breites Profil": "宽兴趣画像",
    "Genre": "Genre",
    "angenommen": "支持",
    "teilweise": "部分支持",
    "ILD-Gewinn innerhalb 5 % Accuracy-Budget": "在 5% 准确性预算内获得 ILD 收益",
    "MMR erzielt höchste paarweise ILD": "MMR 获得最高成对 ILD",
    "xQuAD deckt Aspekte, Kalibrierung gleicht Profil an": "xQuAD 覆盖兴趣方面，校准匹配画像",
    "MMR-Gewinn robust über N und K": "MMR 收益在不同 N 和 K 下稳健",
    "Verbesserte Katalogverteilung": "目录分布改善",
    "+16,30 % ILD; −3,80 % NDCG": "+16.30% ILD；-3.80% NDCG",
    "MMR 0,7785; xQuAD 0,6934; Kal. 0,6883": "MMR 0.7785；xQuAD 0.6934；校准 0.6883",
    "zielmetrikspezifische Rangfolge": "排序取决于目标指标",
    "ILD-Zuwachs in allen fünf Einstellungen": "五种设置中的 ILD 均增长",
    "Coverage/Long Tail ↑; Gini nahezu konstant": "覆盖率/长尾上升；Gini 几乎不变",
    "Datensatz": "数据集",
    "Feedback": "反馈定义",
    "Core": "Core",
    "Seeds": "随机种子",
    "Basismodell": "基础模型",
    "Latente Dimension": "潜维度",
    "Epochen": "训练轮次",
    "Batchgröße": "批大小",
    "8.192": "8,192",
    "Explizite Negativquote": "显式负样本比例",
    "0,5": "0.5",
    "Gerät": "设备",
    "Hauptkonfiguration": "主配置",
    "Methoden": "方法",
    "λ-Hauptraster": "主 λ 网格",
    "Accuracy-Budgets": "准确性预算",
    "Bootstrap": "Bootstrap",
    "Primäre Endpunkte": "主要终点",
    "positiv ≥4; negativ ≤2; neutral 2,5–3": "正反馈 ≥4；负反馈 ≤2；中性 2.5-3",
    "iterativer 5-Core auf chronologischem Training": "只在时间训练集上构建迭代 5-Core",
    "CUDA / RTX 3070": "CUDA / RTX 3070",
    "MMR, xQuAD, Kalibrierung": "MMR、xQuAD、校准",
    "0,00 bis 1,00 in 0,05": "0.00 到 1.00，步长 0.05",
    "1 %, 3 %, 5 %, 10 %": "1%、3%、5%、10%",
    "200 Replikationen je Hauptkonfiguration": "主配置每次 200 个重复样本",
    "200 Replikationen Hauptkurven; 100 Robustheitskurven": "主曲线 200 次重复；稳健性曲线 100 次重复",
    "NDCG@10, ILD@10": "NDCG@10、ILD@10",
    "Vorzeichentest": "符号检验",
    "zweiseitige Normalapproximation mit Stetigkeitskorrektur": "带连续性校正的双侧正态近似",
    "Holm-Familie": "Holm 检验族",
    "vier primäre ILD-Budgetpunkte; sekundäre Tests explorativ": "四个主要 ILD 预算点；次要检验为探索性",
    "übergreifend": "跨方法",
    "je Methode": "各方法",
}


FULL_ROWS_ZH = {
    "hypotheses": [
        ["H1", "三种核心方法均提高至少一项多样性指标。", "支持", "三种方法按相应构念选择的 5% 预算点均高于基线；NDCG 下降。"],
        ["H2", "xQuAD 的类型覆盖收益强于 MMR。", "支持", "Subtopic Recall：xQuAD +9.45%，MMR +7.17%；校准方法在探索性分析中达到 +9.79%。"],
        ["H3", "校准重排序能更好地保持个体兴趣比例。", "支持", "校准相似度提高 +12.35%，而按全局 ILD 选择的 MMR 为 +2.68%。"],
        ["H4", "更大的候选池会使 Pareto 前沿向外移动。", "不支持", "N=100 在所有预算下优于 N=50；N=200 仅在 1% 预算点优于 N=100。"],
        ["H5", "全局 λ 并非对所有画像分群都最优。", "支持", "5% 预算选择：集中、中等和宽兴趣画像分别为 λ=0.35、0.40、0.45。"],
    ],
    "protocol_deviations": [
        ["假设", "后来形成的 H1-H5 版本在内容上偏离开题报告。", "恢复开题报告 H1-H5；后续陈述标记为探索性。", "防止 HARKing；H4 不获支持"],
        ["Core 过滤", "早期管线使用完整正向交互历史。", "最终 5-Core 仅依据按时间划分的训练部分构建。", "防止前视信息泄漏"],
        ["λ 约定", "开题报告：λ 加权相关性；实现：λ 加权多样性。", "完整记录实现约定；λ=0 为基线。", "参数网格覆盖相同混合比例，但数值名称不能直接对应"],
        ["兴趣方面模型", "xQuAD 被设定为面向兴趣方面的方法。", "采用二元类型硬覆盖，而非概率式软覆盖。", "解释范围限于首次类型覆盖"],
        ["BPR 调参", "开题报告计划进行敏感性检验。", "把 BPR 作为固定参考检索器；不宣称模型比较。", "绝对检索质量仍是一项局限"],
        ["随机基线", "随机重排序原计划作为可选基线。", "未纳入核心分析；λ=0 与纯多样性点作为技术边界。", "不能据此声称优于随机列表"],
    ],
    "ai_use": [
        ["翻译与语言", "翻译、语法检查、语言润色和术语一致性。", "作者检查含义、专业术语和最终版本。"],
        ["源代码", "生成和修改分析、测试及文档生成代码。", "作者负责本地执行、测试和结果核对。"],
        ["方法与统计检查", "发现不一致；对 Holm 检验族、指标定义和假设状态提出建议。", "决策和科学责任由作者承担。"],
        ["文献检索", "辅助查找相关来源和书目信息。", "作者核对原始来源并按规范引用。"],
        ["结构与文本补充", "草拟补充说明、方案偏差和讨论段落。", "作者检查、修改并对每段采用的文字负责。"],
    ],
}


NUMERIC_COLUMNS = {
    "dataset_stats": {1},
    "baseline_metrics": {1},
    "budget_results": {0, 2, 3, 4, 5, 6},
    "method_comparison_5pct": {1, 2, 3, 4, 5, 6},
    "construct_comparison_5pct": {2, 3, 4, 5, 6},
    "coverage_comparison_5pct": {1, 2, 3, 4},
    "subgroup_results": {1, 2, 3},
    "segment_lambdas": {1, 2, 3},
    "robustness_results": {1, 2, 3, 4},
    "candidate_frontier": {0, 1, 2, 3},
    "tag_results": {1, 2, 3, 4},
    "runtime_results": {1, 2, 3},
    "all_budget_methods": {0, 3, 5, 6},
}


def normalize_numeric_cell(value: object) -> str:
    """Convert German-formatted result cells to decimal-dot notation."""
    text = str(value).replace("−", "-")
    if "." in text and "," not in text:
        text = text.replace(".", ",")
    elif "," in text and "." not in text:
        text = text.replace(",", ".")
    return text.replace(" %", "%")


def register_fonts_zh() -> None:
    font_dir = Path(r"C:\Windows\Fonts")
    pdfmetrics.registerFont(TTFont(base.FONT_REGULAR, str(font_dir / "Deng.ttf")))
    pdfmetrics.registerFont(TTFont(base.FONT_BOLD, str(font_dir / "Dengb.ttf")))
    pdfmetrics.registerFont(TTFont(base.FONT_SYMBOL, str(font_dir / "seguisym.ttf")))
    pdfmetrics.registerFontFamily(
        base.FONT_REGULAR,
        normal=base.FONT_REGULAR,
        bold=base.FONT_BOLD,
        italic=base.FONT_REGULAR,
        boldItalic=base.FONT_BOLD,
    )


def make_tables_zh(data):
    tables = ORIGINAL_MAKE_TABLES(data)
    translated = {}
    for name, (_, rows) in tables.items():
        rows = FULL_ROWS_ZH.get(name, rows)
        new_rows = []
        for row in rows:
            new_row = []
            for column, value in enumerate(row):
                if str(value) in CELL_TRANSLATIONS:
                    new_row.append(CELL_TRANSLATIONS[str(value)])
                elif column in NUMERIC_COLUMNS.get(name, set()):
                    new_row.append(normalize_numeric_cell(value))
                else:
                    new_row.append(value)
            new_rows.append(new_row)
        translated[name] = (HEADERS_ZH[name], new_rows)
    return translated


def build_styles_zh():
    styles = ORIGINAL_BUILD_STYLES()
    styles["Body"].fontSize = 10.5
    styles["Body"].leading = 17
    styles["Bibliography"].fontSize = 9.2
    styles["Bibliography"].leading = 12.2
    styles["Footnote"].fontSize = 7
    styles["Footnote"].leading = 8.5
    return styles


def configure() -> None:
    base.MANUSCRIPT = ROOT / "thesis" / "manuscript_zh_full.md"
    base.FIGURES = ROOT / "outputs" / "thesis_zh" / "figures"
    base.HEADER_TEXT = "推荐系统中的多样性重排序"
    base.DOC_TITLE = "推荐系统中的多样性重排序"
    base.DOC_SUBJECT = "准确性-多样性权衡的比较研究（中文全译本）"
    base.DOC_AUTHOR = "Yue Cao"
    base.SCHOOL_NAME = "韦德尔应用科学大学"
    base.THESIS_TITLE = "推荐系统中的<br/>多样性重排序"
    base.THESIS_SUBTITLE = "准确性-多样性权衡的比较研究"
    base.DEGREE_TEXT = "数据科学与人工智能专业<br/>硕士论文（中文全译本）"
    base.TITLE_METADATA = [
        ("作者", "Yue Cao"),
        ("学号", "____________________________"),
        ("地址", "____________________________"),
        ("电子邮箱", "____________________________"),
        ("学期", "____________________________"),
        ("第一导师", "____________________________"),
        ("第二导师/企业导师", "____________________________"),
        ("提交日期", "____________________________"),
    ]
    base.BODY_START_HEADINGS = {"1 引言"}
    base.BIBLIOGRAPHY_HEADINGS = {"参考文献"}
    base.APPENDIX_PREFIXES = ("附录",)
    base.DECLARATION_HEADINGS = {"学术诚信声明", "生成式人工智能使用声明"}
    base.FIGURE_LABEL = "图"
    base.TABLE_LABEL = "表"
    base.WORD_WRAP_MODE = "CJK"
    base.FONT_REGULAR = "DengXian"
    base.FONT_BOLD = "DengXian-Bold"
    base.FONT_ITALIC = "DengXian"
    base.FONT_BOLD_ITALIC = "DengXian-Bold"
    base.FONT_SYMBOL = "SegoeUI-Symbol"
    base.FOOTNOTES = FOOTNOTES_ZH
    base.FOOTNOTE_NUMBER = {marker: index for index, marker in enumerate(base.FOOTNOTES, start=1)}
    base.TABLE_TITLES = TABLE_TITLES_ZH
    base.make_tables = make_tables_zh
    base.register_fonts = register_fonts_zh
    base.build_styles = build_styles_zh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure()
    base.build(args.output)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
