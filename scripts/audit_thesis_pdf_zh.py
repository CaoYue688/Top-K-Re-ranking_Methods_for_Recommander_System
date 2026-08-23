"""Structural, content, and geometry audit for the Chinese thesis PDF."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


EXPECTED_TEXT = [
    "中文摘要",
    "1 引言",
    "2 理论基础",
    "3 相关研究",
    "4 研究方法",
    "5 实现与可复现性",
    "6 实验结果",
    "7 讨论",
    "8 结论与展望",
    "参考文献",
    "表 10:",
    "图 7:",
    "学术诚信声明",
    "生成式人工智能使用声明",
    "20,000,263",
    "15,101,200",
    "Candidate Recall@100 为 27,61 %",
    "全部 686 行结果",
    "Tag Genome 敏感性分析",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()

    reader = PdfReader(str(args.pdf))
    texts = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(texts)
    missing = [value for value in EXPECTED_TEXT if value not in full_text]
    unresolved = [
        marker
        for marker in (
            "@@", "[[FN", "PLACEHOLDER", "待补充", "Placeholder for",
            "ZXQNUM", "XQZ", "〔", "〕",
        )
        if marker in full_text
    ]
    blank_pages = [index for index, text in enumerate(texts, start=1) if len(text.strip()) < 20]

    out_of_bounds = []
    page_sizes = set()
    image_count = 0
    with pdfplumber.open(str(args.pdf)) as pdf:
        for index, page in enumerate(pdf.pages, start=1):
            page_sizes.add((round(page.width, 2), round(page.height, 2)))
            image_count += len(page.images)
            for char in page.chars:
                if (
                    char["x0"] < -0.5
                    or char["x1"] > page.width + 0.5
                    or char["top"] < -0.5
                    or char["bottom"] > page.height + 0.5
                ):
                    out_of_bounds.append((index, char.get("text")))

    a4_size = {(595.28, 841.89)}
    result = {
        "path": str(args.pdf),
        "bytes": args.pdf.stat().st_size,
        "pages": len(reader.pages),
        "page_sizes": sorted(page_sizes),
        "text_chars": len(full_text),
        "missing_expected": missing,
        "unresolved": unresolved,
        "blank_pages": blank_pages,
        "out_of_bounds_chars": len(out_of_bounds),
        "encrypted": reader.is_encrypted,
        "embedded_images_total": image_count,
        "key_results_present": all(
            value in full_text
            for value in ("16.30%", "3.80%", "48.28%", "49.74%", "1.68%", "2.04%")
        ),
    }
    print(json.dumps(result, ensure_ascii=True, indent=2))
    if (
        missing
        or unresolved
        or blank_pages
        or out_of_bounds
        or len(reader.pages) < 65
        or len(full_text) < 50000
        or page_sizes != a4_size
        or image_count < 7
        or not result["key_results_present"]
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
