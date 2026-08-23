"""Build the paragraph-aligned Chinese thesis DOCX in the FH Wedel template."""

from __future__ import annotations

import argparse
from pathlib import Path

import build_thesis_docx as base
import build_thesis_pdf_zh as zh


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    ROOT
    / "outputs"
    / "thesis_zh"
    / ".Diversitaetsorientiertes_Re-Ranking_Masterarbeit_Yue_Cao_ZH.base.docx"
)


def configure() -> None:
    base.MANUSCRIPT = ROOT / "thesis" / "manuscript_zh_full.md"
    base.FIGURES = ROOT / "outputs" / "thesis_zh" / "figures"
    base.DEFAULT_OUTPUT = DEFAULT_OUTPUT
    base.HEADER_TEXT = "推荐系统中的多样性重排序"
    base.DOC_TITLE = "推荐系统中的多样性重排序"
    base.DOC_SUBJECT = "准确性-多样性权衡的比较研究（中文全译本）"
    base.SCHOOL_NAME = "韦德尔应用科学大学"
    base.THESIS_TITLE = "推荐系统中的多样性重排序"
    base.THESIS_SUBTITLE = "准确性-多样性权衡的比较研究"
    base.DEGREE_TEXT = "数据科学与人工智能专业\n硕士论文（中文全译本）"
    base.TITLE_METADATA = [
        ("作者", "Yue Cao"),
        ("学号", "[请补充学号]"),
        ("地址", "[请补充地址]"),
        ("电子邮箱", "[请补充电子邮箱]"),
        ("就读学期", "[请补充学期]"),
        ("第一导师", "[请补充第一导师]"),
        ("第二导师/企业导师", "[请补充第二导师]"),
        ("提交日期", "[请补充提交日期]"),
    ]
    base.BODY_START_HEADING = "1 引言"
    base.BIBLIOGRAPHY_HEADING = "参考文献"
    base.APPENDIX_PREFIX = "附录"
    base.DECLARATION_HEADINGS = {"学术诚信声明", "生成式人工智能使用声明"}
    base.FIGURE_LABEL = "图"
    base.TABLE_LABEL = "表"
    base.FONT_REGULAR = "DengXian"
    base.FONT_EAST_ASIA = "等线"
    base.LANGUAGE_TAG = "zh-CN"
    base.AUTO_HYPHENATION = False
    base.TOC_PLACEHOLDER = "请在 Word 中更新目录"
    base.FIGURE_TOC_PLACEHOLDER = "请在 Word 中更新插图目录"
    base.TABLE_TOC_PLACEHOLDER = "请在 Word 中更新表格目录"
    base.FOOTNOTES = zh.FOOTNOTES_ZH
    base.TABLE_TITLES = zh.TABLE_TITLES_ZH
    base.make_tables = zh.make_tables_zh


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", type=Path, default=base.DEFAULT_TEMPLATE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    configure()
    base.build(args.template, args.output)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
