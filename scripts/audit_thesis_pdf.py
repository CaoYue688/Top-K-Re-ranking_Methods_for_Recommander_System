"""Structural and geometry audit for the generated thesis PDF."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pdfplumber
from pypdf import PdfReader


EXPECTED_TEXT = [
    "1 Einleitung",
    "2 Grundlagen",
    "3 Stand der Forschung",
    "4 Methodik",
    "5 Implementierung und Reproduzierbarkeit",
    "6 Ergebnisse",
    "7 Diskussion",
    "8 Fazit und Ausblick",
    "Formelverzeichnis",
    "Formel (12):",
    "Literaturverzeichnis",
    "Hilfsmittelverzeichnis",
    "Codex Desktop",
    "Da Deutsch nicht meine Muttersprache ist",
    "Anhang A: Vollständige Experimentparameter",
    "Anhang B: Reproduktions- und Auditprotokoll",
    "GroupLens Research (2016)",
    "Tabelle 10:",
    "Abbildung 7:",
    "Eidesstattliche Erklärung",
]

FORBIDDEN_TEXT = [
    "Reflexion zum Einsatz generativer KI",
    "Anhang A: Protokolländerungen gegenüber dem Exposé",
    "Anhang C: Ergänzende Budgettabellen",
    "Anhang D: Reproduktions- und Auditprotokoll",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    args = parser.parse_args()

    reader = PdfReader(str(args.pdf))
    texts = [page.extract_text() or "" for page in reader.pages]
    full_text = "\n".join(texts)
    missing = [value for value in EXPECTED_TEXT if value not in full_text]
    forbidden_present = [value for value in FORBIDDEN_TEXT if value in full_text]
    unresolved = [
        marker
        for marker in ("@@", "[[FN", "Placeholder for table of contents")
        if marker in full_text
    ]
    blank_pages = [index for index, text in enumerate(texts, start=1) if len(text.strip()) < 20]
    raw_inline_math = sorted(set(re.findall(r"\b(?:[LSpqbrfmx])_[A-Za-z]", full_text)))
    front_page_two_is_ii = bool(re.search(r"(?m)^ii$", texts[1].strip(), re.IGNORECASE))

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

    result = {
        "path": str(args.pdf),
        "bytes": args.pdf.stat().st_size,
        "pages": len(reader.pages),
        "page_sizes": sorted(page_sizes),
        "text_chars": len(full_text),
        "missing_expected": missing,
        "forbidden_present": forbidden_present,
        "unresolved": unresolved,
        "blank_pages": blank_pages,
        "out_of_bounds_chars": len(out_of_bounds),
        "encrypted": reader.is_encrypted,
        "embedded_images_total": image_count,
        "raw_inline_math": raw_inline_math,
        "front_page_two_is_ii": front_page_two_is_ii,
        "key_result_present": "16,30 %" in full_text and "-3,80 %" in full_text,
    }
    print(result)
    if (
        missing
        or forbidden_present
        or unresolved
        or blank_pages
        or out_of_bounds
        or raw_inline_math
        or not front_page_two_is_ii
        or image_count < 19
        or not 80 <= len(reader.pages) <= 130
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
