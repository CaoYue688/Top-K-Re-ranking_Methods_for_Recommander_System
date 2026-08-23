"""Audit paragraph-aligned German/Chinese thesis text before PDF authoring."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import re

import translate_thesis_de_to_zh_full as translation


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "thesis" / "manuscript.md"
TARGET = ROOT / "thesis" / "manuscript_zh_full.md"
REPORT = ROOT / "tmp" / "translation" / "full_translation_content_audit.json"

SUPERSCRIPT_DIGITS = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹", "0123456789")
GERMAN_RESIDUAL_RE = re.compile(
    r"\b(?:der|die|das|den|dem|des|und|oder|wird|werden|wurde|wurden|"
    r"einer|eines|eine|einen|dass|nicht|Nutzende|Kapitel|Ergebnisse|"
    r"Verfahren|Datensatz|Forschungsfrage|Zusammenfassung)\b",
    re.IGNORECASE,
)
PLACEHOLDER_RE = re.compile(r"ZXQNUM|XQZ|<NUM\d+>", re.IGNORECASE)

# These paragraphs were manually checked after translation. Their apparent
# numeric differences are faithful rendering choices: German number words or
# compounds such as "Top-10" / "5-%" become Arabic numerals in Chinese, and
# "7,9 Millionen" becomes "790 万".
APPROVED_NUMERIC_RENDERING_BLOCKS = {
    9, 10, 45, 93, 117, 120, 130, 132, 133, 138, 202, 264, 272, 288, 336, 372,
    419,
}


def number_tokens(text: str) -> list[str]:
    return [match.group(0) for match in translation.NUMBER_RE.finditer(text)]


def canonical_number(value: str) -> str:
    result = value.translate(SUPERSCRIPT_DIGITS).replace("−", "-")
    return result.replace(" ", "").replace(".", "").replace(",", "").replace("%", "")


def number_counter(text: str) -> Counter[str]:
    return Counter(canonical_number(value) for value in number_tokens(text))


def main() -> None:
    source_text = SOURCE.read_text(encoding="utf-8")
    target_text = TARGET.read_text(encoding="utf-8")
    source_blocks = translation.blocks(source_text)
    target_blocks = translation.blocks(target_text)
    manifest_path = ROOT / "tmp" / "translation" / "full_translation_alignment.jsonl"
    records = [json.loads(line) for line in manifest_path.read_text(encoding="utf-8").splitlines()]

    if len(source_blocks) != len(target_blocks) or len(records) != len(source_blocks):
        raise ValueError(
            f"Alignment length mismatch: source={len(source_blocks)}, "
            f"target={len(target_blocks)}, records={len(records)}"
        )

    numeric_mismatches = []
    residuals = []
    placeholders = []
    meta_phrases = []
    for index, (source, target, record) in enumerate(
        zip(source_blocks, target_blocks, records), start=1
    ):
        source_numbers = number_counter(source)
        target_numbers = number_counter(target)
        if source_numbers != target_numbers:
            numeric_mismatches.append(
                {
                    "block": index,
                    "source_numbers": dict(source_numbers),
                    "target_numbers": dict(target_numbers),
                    "source_tokens": number_tokens(source),
                    "target_tokens": number_tokens(target),
                    "equal_token_count": len(number_tokens(source)) == len(number_tokens(target)),
                    "source": source,
                    "target": target,
                }
            )
        if record["kind"] == "translated":
            matches = sorted(set(match.group(0) for match in GERMAN_RESIDUAL_RE.finditer(target)))
            if matches:
                residuals.append({"block": index, "matches": matches, "target": target})
        if PLACEHOLDER_RE.search(target):
            placeholders.append({"block": index, "target": target})
        if re.search(r"(?:以下是|译文[:：]|中文翻译[:：]|抱歉|无法翻译)", target):
            meta_phrases.append({"block": index, "target": target})

    raw_numeric_mismatches = numeric_mismatches
    numeric_mismatches = [
        item for item in raw_numeric_mismatches
        if item["block"] not in APPROVED_NUMERIC_RENDERING_BLOCKS
    ]
    reviewed_numeric_variants = [
        item for item in raw_numeric_mismatches
        if item["block"] in APPROVED_NUMERIC_RENDERING_BLOCKS
    ]
    summary = {
        "source_blocks": len(source_blocks),
        "target_blocks": len(target_blocks),
        "source_headings": len(translation.heading_lines(source_text)),
        "target_headings": len(translation.heading_lines(target_text)),
        "source_figures": len(translation.figure_lines(source_text)),
        "target_figures": len(translation.figure_lines(target_text)),
        "source_tables": source_text.count("@@TABLE:"),
        "target_tables": target_text.count("@@TABLE:"),
        "source_footnotes": len(re.findall(r"\[\[FN\d+\]\]", source_text)),
        "target_footnotes": len(re.findall(r"\[\[FN\d+\]\]", target_text)),
        "numeric_mismatch_count": len(numeric_mismatches),
        "numeric_mismatch_blocks": [item["block"] for item in numeric_mismatches],
        "reviewed_numeric_rendering_count": len(reviewed_numeric_variants),
        "reviewed_numeric_rendering_blocks": [
            item["block"] for item in reviewed_numeric_variants
        ],
        "german_residual_count": len(residuals),
        "placeholder_count": len(placeholders),
        "meta_phrase_count": len(meta_phrases),
        "numeric_mismatches": numeric_mismatches,
        "reviewed_numeric_renderings": reviewed_numeric_variants,
        "german_residuals": residuals,
        "placeholders": placeholders,
        "meta_phrases": meta_phrases,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {key: value for key, value in summary.items() if not isinstance(value, list)},
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"numeric_mismatch_blocks={summary['numeric_mismatch_blocks']}")
    print(f"report={REPORT}")


if __name__ == "__main__":
    main()
