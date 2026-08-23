"""Build a submission-style PDF directly from the audited thesis manuscript.

This is intentionally independent of Word/LibreOffice so that the PDF can be
rendered and visually checked in the current environment.  Content, tables and
figures are sourced from the same files as the DOCX builder.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import sys
from pathlib import Path
from typing import Iterable
from xml.sax.saxutils import escape

import pandas as pd

LOCAL_DEPS = Path(__file__).resolve().parents[1] / ".codex_deps"
if LOCAL_DEPS.exists():
    sys.path.insert(0, str(LOCAL_DEPS))

from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    LongTable,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents

from build_thesis_docx import FOOTNOTES, TABLE_TITLES, make_tables


ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT = ROOT / "thesis" / "manuscript.md"
AGG = ROOT / "outputs" / "thesis_pos4_neg2_traincore5_dataseed2026" / "aggregate"
FIGURES = ROOT / "outputs" / "thesis" / "figures"
DEFAULT_OUTPUT = ROOT / "outputs" / "thesis" / "Diversitaetsorientiertes_Re-Ranking_Masterarbeit_Yue_Cao.pdf"

PAGE_W, PAGE_H = A4
LEFT = 3.5 * cm
RIGHT = 4.0 * cm
TOP = 3.0 * cm
BOTTOM = 3.0 * cm
BODY_BOTTOM = 3.85 * cm  # reserves a real footnote band above the footer
TEXT_W = PAGE_W - LEFT - RIGHT
HEADER_TEXT = "Diversitätsorientiertes Re-Ranking"
DOC_TITLE = "Diversitätsorientiertes Re-Ranking in Recommender-Systemen"
DOC_SUBJECT = "Masterarbeit - Accuracy-Diversity-Trade-off"
DOC_AUTHOR = "Yue Cao"
SCHOOL_NAME = "Fachhochschule Wedel"
THESIS_TITLE = "Diversitätsorientiertes Re-Ranking<br/>in Recommender-Systemen"
THESIS_SUBTITLE = "Eine vergleichende Analyse des Accuracy-Diversity-Trade-offs"
DEGREE_TEXT = "Masterarbeit<br/>in der Fachrichtung Data Science &amp; Artificial Intelligence"
TITLE_METADATA = [
    ("Vorgelegt von", os.environ.get("THESIS_AUTHOR", "Yue Cao")),
    ("Matrikelnummer", os.environ.get("THESIS_STUDENT_ID", "[Matrikelnummer ergänzen]")),
    ("Anschrift", os.environ.get("THESIS_ADDRESS", "[Anschrift ergänzen]")),
    ("E-Mail", os.environ.get("THESIS_EMAIL", "[E-Mail-Adresse ergänzen]")),
    ("Verwaltungssemester", "5. Semester"),
    ("Semester der Abgabe", "Sommersemester 2026"),
    ("Betreuer (FH Wedel)", "Prof. Dr. Gerd Beuster"),
    ("Abgabedatum", "25. August 2026"),
]
BODY_START_HEADINGS = {"1 Einleitung"}
BIBLIOGRAPHY_HEADINGS = {"Literaturverzeichnis"}
APPENDIX_PREFIXES = ("Anhang",)
DECLARATION_HEADINGS = {"Eidesstattliche Erklärung"}
AUXILIARY_HEADINGS = {"Hilfsmittelverzeichnis"}
FIGURE_LABEL = "Abbildung"
TABLE_LABEL = "Tabelle"
WORD_WRAP_MODE: str | None = None

FONT_REGULAR = "Arial"
FONT_BOLD = "Arial-Bold"
FONT_ITALIC = "Arial-Italic"
FONT_BOLD_ITALIC = "Arial-BoldItalic"
FONT_SYMBOL = "SegoeUI-Symbol"
EQUATION_DPI = 600
EQUATION_DIR = ROOT / "tmp" / "pdfs" / "thesis_equations"
INLINE_EQUATION_DIR = ROOT / "tmp" / "pdfs" / "thesis_inline_equations"
EQUATION_LATEX = {
    "1": r"s(u,i)=p_u^{\mathsf{T}}q_i+b_i",
    "2": r"x_{uij}=s(u,i)-s(u,j)",
    "3": r"\mathcal{L}=-\operatorname{mean}\!\left[\log \sigma(x_{uij})\right]+\lambda_{\mathrm{reg}}\operatorname{mean}\!\left(\Vert p_u\Vert_2^2+\Vert q_i\Vert_2^2+\Vert q_j\Vert_2^2\right)",
    "4": r"\operatorname{ILD}(L_u)=\frac{2}{K(K-1)}\sum_{1\leq a<b\leq K}d(i_a,i_b)",
    "5": r"H_{\mathrm{norm}}(q)=-\frac{\sum_{g\in G}q(g)\log q(g)}{\log\left|G\right|}",
    "6": r"\operatorname{score}_{\mathrm{MMR}}(i\mid u,S)=(1-\lambda)\,\tilde r_{ui}+\lambda\!\left[1-\max_{j\in S}\operatorname{sim}(i,j)\right]",
    "7": r"D_{\mathrm{xQuAD}}(i\mid u,S)=\sum_{g\in G}p_u(g)P(g\mid i)\prod_{j\in S}\!\left[1-P(g\mid j)\right]",
    "8": r"\operatorname{score}_{\mathrm{CAL}}(i\mid u,S)=(1-\lambda)\,\tilde r_{ui}+\lambda\!\left[1-\frac{\operatorname{JSD}\!\left(p_u,q_{S\cup\{i\}}\right)}{\ln 2}\right]",
    "9": r"\operatorname{DCG}_u@K=\sum_{r=1}^{K}\frac{\operatorname{rel}(u,r)}{\log_2(r+1)}",
    "10": r"\operatorname{IDCG}_u@K=\sum_{r=1}^{\min(m_u,K)}\frac{1}{\log_2(r+1)}",
    "11": r"\operatorname{NDCG}_u@K=\frac{\operatorname{DCG}_u@K}{\operatorname{IDCG}_u@K}",
    "12": r"F_{m,b}=\left\{\lambda\in[0,1]\ \mid\ \operatorname{NDCG}_m(\lambda)\geq(1-b)\operatorname{NDCG}_{\mathrm{baseline}}\right\}",
}
EQUATION_TITLES = {
    "1": "Präferenzscore der Matrixfaktorisierung",
    "2": "Paarweise Score-Differenz für BPR",
    "3": "Regularisierte BPR-Verlustfunktion",
    "4": "Intra-List-Diversity",
    "5": "Normalisierte Genreentropie",
    "6": "MMR-Auswahlfunktion",
    "7": "xQuAD-Diversitätsterm",
    "8": "Auswahlfunktion des kalibrierten Re-Rankings",
    "9": "Discounted Cumulative Gain",
    "10": "Ideal Discounted Cumulative Gain",
    "11": "Normalized Discounted Cumulative Gain",
    "12": "Zulässige Menge unter einem Accuracy-Budget",
}

# Inline mathematics is rendered by the same MathText engine as the numbered
# equations. Longest-first replacement prevents a short token such as p_u
# from consuming a more specific expression such as p_u(g).
INLINE_MATH_LATEX = {
    "s: U × I → R": r"s\colon U\times I\to\mathbb{R}",
    "L_u=(i_1,…,i_K)": r"L_u=(i_1,\ldots,i_K)",
    "s(u,i)": r"s(u,i)",
    "p_u∈R^d": r"p_u\in\mathbb{R}^d",
    "q_i∈R^d": r"q_i\in\mathbb{R}^d",
    "b_i": r"b_i",
    "O(|U||I|)": r"O(|U|\,|I|)",
    "[0,1]": r"[0,1]",
    "C_u": r"C_u",
    "S_u": r"S_u",
    "O(KN²d_f)": r"O(KN^2d_f)",
    "d_f": r"d_f",
    "d(i,j)=1-cos(f_i,f_j)": r"d(i,j)=1-\cos(f_i,f_j)",
    "cos(f_i,f_j)": r"\cos(f_i,f_j)",
    "f_i": r"f_i",
    "f_j": r"f_j",
    "p_u(g)": r"p_u(g)",
    "q_L(g)": r"q_L(g)",
    "1-JSD(p_u,q_L)/ln 2": r"1-\operatorname{JSD}(p_u,q_L)/\ln 2",
    "1-JS(p_u,q_L)": r"1-\operatorname{JSD}(p_u,q_L)/\ln 2",
    "p_u": r"p_u",
    "q_L": r"q_L",
    "r_ui": r"\tilde r_{ui}",
    "P(g|i)=1": r"P(g\mid i)=1",
    "(1-λ)r_ui+λD_xQuAD": r"(1-\lambda)\tilde r_{ui}+\lambda D_{\mathrm{xQuAD}}",
    "q_(S∪{i})": r"q_{S\cup\{i\}}",
    "rel_(u,r)∈{0,1}": r"\operatorname{rel}(u,r)\in\{0,1\}",
    "min(m_u,K)": r"\min(m_u,K)",
    "m_u": r"m_u",
    "b∈{0,01;0,03;0,05;0,10}": r"b\in\{0{,}01;0{,}03;0{,}05;0{,}10\}",
    "O(B|I|d)": r"O(B|I|d)",
    "O(|U|N)": r"O(|U|N)",
    "1,91×10^-110": r"1{,}91\times10^{-110}",
    "5,72×10^-110": r"5{,}72\times10^{-110}",
    "λ_u": r"\lambda_u",
    "r=1,…,K": r"r=1,\ldots,K",
}


def register_fonts() -> None:
    font_dir = Path(r"C:\Windows\Fonts")
    files = {
        FONT_REGULAR: font_dir / "arial.ttf",
        FONT_BOLD: font_dir / "arialbd.ttf",
        FONT_ITALIC: font_dir / "ariali.ttf",
        FONT_BOLD_ITALIC: font_dir / "arialbi.ttf",
        FONT_SYMBOL: font_dir / "seguisym.ttf",
    }
    missing = [str(path) for path in files.values() if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Required Arial font files are missing: {missing}")
    for name, path in files.items():
        pdfmetrics.registerFont(TTFont(name, str(path)))
    pdfmetrics.registerFontFamily(
        FONT_REGULAR,
        normal=FONT_REGULAR,
        bold=FONT_BOLD,
        italic=FONT_ITALIC,
        boldItalic=FONT_BOLD_ITALIC,
    )


def roman(number: int) -> str:
    values = [
        (1000, "m"), (900, "cm"), (500, "d"), (400, "cd"),
        (100, "c"), (90, "xc"), (50, "l"), (40, "xl"),
        (10, "x"), (9, "ix"), (5, "v"), (4, "iv"), (1, "i"),
    ]
    result = []
    for value, symbol in values:
        while number >= value:
            result.append(symbol)
            number -= value
    return "".join(result)


def normalize_punctuation(text: str) -> str:
    return (
        text.replace("\u2010", "-")
        .replace("\u2011", "-")
        .replace("\u2012", "-")
        .replace("\u2013", "-")
        .replace("\u2014", "-")
        .replace("\u2212", "-")
        .replace("\u00a0", " ")
    )


FOOTNOTE_NUMBER = {marker: index for index, marker in enumerate(FOOTNOTES, start=1)}


def inline_markup(text: str) -> tuple[str, list[tuple[int, str]]]:
    """Return ReportLab-safe inline markup plus footnotes occurring in text."""
    text = normalize_punctuation(text.strip())
    text = re.sub(r"\[[^\]]+ ergänzen\]", "____________________________", text)
    inline_math_tags: list[tuple[str, str]] = []
    for source, latex in sorted(INLINE_MATH_LATEX.items(), key=lambda item: -len(item[0])):
        if source not in text:
            continue
        token = f"__INLINE_MATH_{len(inline_math_tags)}__"
        inline_math_tags.append((token, inline_math_tag(latex)))
        text = text.replace(source, token)
    notes: list[tuple[int, str]] = []
    for marker, number in FOOTNOTE_NUMBER.items():
        if marker in text:
            notes.append((number, normalize_punctuation(FOOTNOTES[marker])))
            text = text.replace(marker, f"__FOOTNOTE_{number}__")

    text = escape(text)
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"(?<!\*)\*([^*]+?)\*(?!\*)", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", rf'<font name="{FONT_REGULAR}">\1</font>', text)
    # Arial's Windows font file lacks a few mathematical glyphs used in the
    # methodology.  Explicit fallback avoids empty squares in PDF renderers.
    for symbol in ("×", "→", "∈", "∪", "≤", "≥"):
        text = text.replace(symbol, f'<font name="{FONT_SYMBOL}">{symbol}</font>')
    for token, tag in inline_math_tags:
        text = text.replace(token, tag)
    for number, _ in notes:
        text = text.replace(
            f"__FOOTNOTE_{number}__",
            f'<super><font size="7">{number}</font></super>',
        )
    return text, notes


def inline_math_tag(latex: str) -> str:
    """Render a compact, high-resolution MathText expression for body prose."""
    INLINE_EQUATION_DIR.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(latex.encode("utf-8")).hexdigest()[:16]
    image_path = INLINE_EQUATION_DIR / f"inline_{digest}.png"
    if not image_path.exists():
        math_to_image(
            "$" + latex + "$",
            image_path,
            prop=FontProperties(size=10.5),
            dpi=EQUATION_DPI,
            format="png",
            color="black",
        )
    width_px, height_px = ImageReader(str(image_path)).getSize()
    width = width_px * 72.0 / EQUATION_DPI
    height = height_px * 72.0 / EQUATION_DPI
    if height > 12.8:
        scale = 12.8 / height
        width *= scale
        height *= scale
    path = image_path.as_posix()
    return (
        f'<img src="{path}" width="{width:.2f}" height="{height:.2f}" '
        f'valign="-2.0"/>'
    )


def make_paragraph(text: str, style: ParagraphStyle, **attrs) -> Paragraph:
    markup, notes = inline_markup(text)
    paragraph = Paragraph(markup, style, **attrs)
    paragraph._thesis_footnotes = notes
    return paragraph

def equation_flowable(number: str, styles: dict[str, ParagraphStyle]):
    """Render a numbered equation with MathText and a fixed right number column."""
    if number not in EQUATION_LATEX:
        raise KeyError(f"No mathematical rendering configured for equation {number}")
    EQUATION_DIR.mkdir(parents=True, exist_ok=True)
    image_path = EQUATION_DIR / f"equation_{int(number):02d}.png"
    math_to_image(
        "$" + EQUATION_LATEX[number] + "$",
        image_path,
        prop=FontProperties(size=11),
        dpi=EQUATION_DPI,
        format="png",
        color="black",
    )
    formula_image = Image(str(image_path))
    natural_width = formula_image.imageWidth * 72.0 / EQUATION_DPI
    natural_height = formula_image.imageHeight * 72.0 / EQUATION_DPI
    max_width = TEXT_W - 1.7 * cm
    scale = min(1.0, max_width / natural_width)
    formula_image.drawWidth = natural_width * scale
    formula_image.drawHeight = natural_height * scale
    formula_image.hAlign = "CENTER"

    equation = Table(
        [[formula_image, make_paragraph(f"({number})", styles["EquationNumber"])]],
        colWidths=[TEXT_W - 1.2 * cm, 1.2 * cm],
        hAlign="LEFT",
    )
    equation.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("ALIGN", (0, 0), (0, 0), "CENTER"),
        ("LEFTPADDING", (0, 0), (-1, -1), 0),
        ("RIGHTPADDING", (0, 0), (-1, -1), 0),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    equation._thesis_equation_number = number
    equation._thesis_equation_title = EQUATION_TITLES[number]
    return equation

class NamedTableOfContents(TableOfContents):
    def __init__(self, notification_kind: str, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.notification_kind = notification_kind

    def notify(self, kind, stuff):
        if kind == self.notification_kind:
            self.addEntry(*stuff)


class ThesisDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, styles: dict[str, ParagraphStyle]):
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=LEFT,
            rightMargin=RIGHT,
            topMargin=TOP,
            bottomMargin=BOTTOM,
            title=DOC_TITLE,
            author=DOC_AUTHOR,
            subject=DOC_SUBJECT,
            creator="Codex / ReportLab",
        )
        self.styles = styles
        self.body_start_page: int | None = None
        self._body_seen_this_pass = False
        self._heading_counter = 0
        self._caption_counter = 0
        self._page_footnotes: dict[int, list[tuple[int, str]]] = {}

        regular_frame = Frame(
            LEFT,
            BOTTOM,
            TEXT_W,
            PAGE_H - TOP - BOTTOM,
            id="regular",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        body_frame = Frame(
            LEFT,
            BODY_BOTTOM,
            TEXT_W,
            PAGE_H - TOP - BODY_BOTTOM,
            id="body",
            leftPadding=0,
            rightPadding=0,
            topPadding=0,
            bottomPadding=0,
        )
        self.addPageTemplates(
            [
                PageTemplate(id="title", frames=[regular_frame], onPage=self._title_page),
                PageTemplate(id="front", frames=[regular_frame], onPage=self._front_page),
                PageTemplate(id="body", frames=[body_frame], onPage=self._body_page),
            ]
        )

    def beforeDocument(self):
        # Keep the previous pass' body start so TOCs, which are rendered before
        # the body, can format page numbers correctly during multiBuild.
        self._body_seen_this_pass = False
        self._heading_counter = 0
        self._caption_counter = 0
        self._page_footnotes = {}

    @staticmethod
    def _title_page(canvas, doc):
        canvas.saveState()
        canvas.setTitle(DOC_TITLE)
        canvas.setAuthor(DOC_AUTHOR)
        canvas.restoreState()

    def _draw_header_footer(self, canvas, page_label: str) -> None:
        canvas.saveState()
        canvas.setFont(FONT_REGULAR, 8)
        canvas.setFillColor(colors.HexColor("#5A5A5A"))
        canvas.drawRightString(PAGE_W - RIGHT, PAGE_H - 1.55 * cm, HEADER_TEXT)
        canvas.setStrokeColor(colors.HexColor("#B8C2CC"))
        canvas.setLineWidth(0.35)
        canvas.line(LEFT, PAGE_H - 1.77 * cm, PAGE_W - RIGHT, PAGE_H - 1.77 * cm)
        canvas.setFillColor(colors.black)
        canvas.drawCentredString(PAGE_W / 2, 1.35 * cm, page_label)
        canvas.restoreState()

    def _front_page(self, canvas, doc):
        # The title page counts as Roman page I but does not print its label.
        self._draw_header_footer(canvas, roman(canvas.getPageNumber()))

    def _body_page(self, canvas, doc):
        if not self._body_seen_this_pass:
            self.body_start_page = canvas.getPageNumber()
            self._body_seen_this_pass = True
        label = str(canvas.getPageNumber() - self.body_start_page + 1)
        self._draw_header_footer(canvas, label)

    def _display_page(self, physical_page: int, section: str) -> str:
        if section == "front":
            return roman(physical_page)
        if self.body_start_page is None:
            return str(physical_page)
        return str(physical_page - self.body_start_page + 1)

    def afterFlowable(self, flowable):
        notes = getattr(flowable, "_thesis_footnotes", [])
        if notes:
            bucket = self._page_footnotes.setdefault(self.page, [])
            seen = {number for number, _ in bucket}
            bucket.extend((number, text) for number, text in notes if number not in seen)

        equation_number = getattr(flowable, "_thesis_equation_number", None)
        if equation_number is not None:
            title = getattr(flowable, "_thesis_equation_title")
            key = f"equation-{equation_number}"
            self.canv.bookmarkPage(key)
            self.notify(
                "EQEntry",
                (0, f"Formel ({equation_number}): {title}", self.page, key),
            )
            return

        if not isinstance(flowable, Paragraph):
            return
        style_name = flowable.style.name
        if style_name in {"Heading1", "Heading2", "Heading3"}:
            level = {"Heading1": 0, "Heading2": 1, "Heading3": 2}[style_name]
            self._heading_counter += 1
            key = f"heading-{self._heading_counter}"
            self.canv.bookmarkPage(key)
            if level == 0:
                try:
                    self.canv.addOutlineEntry(flowable.getPlainText(), key, level=0, closed=False)
                except ValueError:
                    pass
            self.notify("TOCEntry", (level, flowable.getPlainText(), self.page, key))
        elif style_name in {"FigureCaption", "TableCaption"}:
            self._caption_counter += 1
            key = f"caption-{self._caption_counter}"
            self.canv.bookmarkPage(key)
            kind = "FIGEntry" if style_name == "FigureCaption" else "TABLEEntry"
            self.notify(kind, (0, flowable.getPlainText(), self.page, key))

    def afterPage(self):
        notes = self._page_footnotes.get(self.page, [])
        if not notes:
            return
        canvas = self.canv
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#606060"))
        canvas.setLineWidth(0.4)
        canvas.line(LEFT, 3.36 * cm, LEFT + 4.0 * cm, 3.36 * cm)

        rendered: list[tuple[Paragraph, float]] = []
        total_height = 0.0
        for number, text in notes:
            markup = f'<super>{number}</super> {escape(text)}'
            paragraph = Paragraph(markup, self.styles["Footnote"])
            _, height = paragraph.wrap(TEXT_W, 3.0 * cm)
            rendered.append((paragraph, height))
            total_height += height + 1.2
        y = 3.28 * cm
        for paragraph, height in rendered:
            y -= height
            paragraph.drawOn(canvas, LEFT, y)
            y -= 1.2
        canvas.restoreState()


def build_styles() -> dict[str, ParagraphStyle]:
    sample = getSampleStyleSheet()
    styles = {
        "Body": ParagraphStyle(
            "Body",
            parent=sample["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=11,
            leading=16.5,
            alignment=TA_JUSTIFY,
            textColor=colors.black,
            spaceAfter=6,
            splitLongWords=1,
            allowWidows=0,
            allowOrphans=0,
        ),
        "Heading1": ParagraphStyle(
            "Heading1",
            parent=sample["Heading1"],
            fontName=FONT_BOLD,
            fontSize=16,
            leading=19.5,
            textColor=colors.black,
            spaceBefore=0,
            spaceAfter=10,
            keepWithNext=1,
        ),
        "Heading2": ParagraphStyle(
            "Heading2",
            parent=sample["Heading2"],
            fontName=FONT_BOLD,
            fontSize=13,
            leading=16,
            textColor=colors.black,
            spaceBefore=12,
            spaceAfter=7,
            keepWithNext=1,
        ),
        "Heading3": ParagraphStyle(
            "Heading3",
            parent=sample["Heading3"],
            fontName=FONT_BOLD,
            fontSize=11.5,
            leading=14,
            textColor=colors.black,
            spaceBefore=9,
            spaceAfter=5,
            keepWithNext=1,
        ),
        "Bibliography": ParagraphStyle(
            "Bibliography",
            parent=sample["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=11,
            leading=13.2,
            alignment=TA_LEFT,
            leftIndent=0.6 * cm,
            firstLineIndent=-0.6 * cm,
            spaceAfter=5,
            splitLongWords=1,
        ),
        "Bullet": ParagraphStyle(
            "Bullet",
            parent=sample["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=11,
            leading=16.5,
            alignment=TA_JUSTIFY,
            leftIndent=0.75 * cm,
            firstLineIndent=-0.4 * cm,
            bulletIndent=0.15 * cm,
            spaceAfter=4,
        ),
        "Number": ParagraphStyle(
            "Number",
            parent=sample["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=11,
            leading=16.5,
            alignment=TA_JUSTIFY,
            leftIndent=0.85 * cm,
            firstLineIndent=-0.55 * cm,
            bulletIndent=0.12 * cm,
            spaceAfter=4,
        ),
        "FigureCaption": ParagraphStyle(
            "FigureCaption",
            parent=sample["BodyText"],
            fontName=FONT_ITALIC,
            fontSize=9,
            leading=11.3,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=9,
        ),
        "TableCaption": ParagraphStyle(
            "TableCaption",
            parent=sample["BodyText"],
            fontName=FONT_ITALIC,
            fontSize=9,
            leading=11.3,
            alignment=TA_CENTER,
            spaceBefore=5,
            spaceAfter=4,
            keepWithNext=1,
        ),
        "Equation": ParagraphStyle(
            "Equation",
            parent=sample["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=10.5,
            leading=14,
            alignment=TA_CENTER,
            spaceBefore=4,
            spaceAfter=4,
        ),
        "EquationNumber": ParagraphStyle(
            "EquationNumber",
            parent=sample["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=10.5,
            leading=14,
            alignment=TA_RIGHT,
        ),
        "Footnote": ParagraphStyle(
            "Footnote",
            parent=sample["BodyText"],
            fontName=FONT_REGULAR,
            fontSize=7.3,
            leading=8.8,
            alignment=TA_LEFT,
            textColor=colors.HexColor("#303030"),
        ),
        "TOC0": ParagraphStyle(
            "TOC0",
            fontName=FONT_BOLD,
            fontSize=10.5,
            leading=13.5,
            leftIndent=0,
            firstLineIndent=0,
            spaceBefore=4,
            spaceAfter=2,
        ),
        "TOC1": ParagraphStyle(
            "TOC1",
            fontName=FONT_REGULAR,
            fontSize=9.5,
            leading=12,
            leftIndent=0.45 * cm,
            firstLineIndent=0,
            spaceBefore=1,
            spaceAfter=1,
        ),
        "TOC2": ParagraphStyle(
            "TOC2",
            fontName=FONT_REGULAR,
            fontSize=9,
            leading=11,
            leftIndent=0.9 * cm,
            firstLineIndent=0,
            spaceBefore=0,
            spaceAfter=1,
        ),
        "ListEntry": ParagraphStyle(
            "ListEntry",
            fontName=FONT_REGULAR,
            fontSize=9.5,
            leading=12,
            leftIndent=2.25 * cm,
            firstLineIndent=-2.25 * cm,
            rightIndent=1.15 * cm,
            spaceBefore=2,
            spaceAfter=4,
        ),
    }
    if WORD_WRAP_MODE:
        for style in styles.values():
            style.wordWrap = WORD_WRAP_MODE
    return styles


def title_page(styles: dict[str, ParagraphStyle]) -> list:
    school = ParagraphStyle(
        "School",
        fontName=FONT_BOLD,
        fontSize=16,
        leading=19,
        alignment=TA_CENTER,
        textColor=colors.black,
    )
    title = ParagraphStyle(
        "Title",
        fontName=FONT_BOLD,
        fontSize=22,
        leading=27,
        alignment=TA_CENTER,
        textColor=colors.black,
        wordWrap=WORD_WRAP_MODE,
    )
    subtitle = ParagraphStyle(
        "Subtitle",
        fontName=FONT_REGULAR,
        fontSize=14,
        leading=18,
        alignment=TA_CENTER,
    )
    degree = ParagraphStyle(
        "Degree",
        fontName=FONT_REGULAR,
        fontSize=12,
        leading=16,
        alignment=TA_CENTER,
    )
    meta_left = ParagraphStyle(
        "MetaLeft", fontName=FONT_BOLD, fontSize=9.2, leading=12, alignment=TA_LEFT
    )
    meta_right = ParagraphStyle(
        "MetaRight", fontName=FONT_REGULAR, fontSize=9.2, leading=12, alignment=TA_LEFT
    )
    meta_rows = [
        [Paragraph(escape(key), meta_left), Paragraph(escape(value), meta_right)]
        for key, value in TITLE_METADATA
    ]
    meta_table = Table(meta_rows, colWidths=[5.0 * cm, TEXT_W - 5.0 * cm], hAlign="CENTER")
    meta_table.setStyle(
        TableStyle(
            [
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 2),
                ("RIGHTPADDING", (0, 0), (-1, -1), 2),
                ("TOPPADDING", (0, 0), (-1, -1), 3),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
                ("LINEBELOW", (0, -1), (-1, -1), 0.5, colors.HexColor("#AAB7C4")),
            ]
        )
    )
    return [
        Spacer(1, 0.6 * cm),
        Paragraph(SCHOOL_NAME, school),
        Spacer(1, 2.3 * cm),
        Paragraph(THESIS_TITLE, title),
        Spacer(1, 0.55 * cm),
        Paragraph(THESIS_SUBTITLE, subtitle),
        Spacer(1, 1.35 * cm),
        Paragraph(DEGREE_TEXT, degree),
        Spacer(1, 1.25 * cm),
        meta_table,
    ]


TABLE_WIDTHS_CM = {
    "dataset_stats": [8.2, 5.3],
    "baseline_metrics": [8.0, 5.5],
    "budget_results": [1.2, 2.0, 0.9, 1.8, 2.4, 1.8, 3.4],
    "method_comparison_5pct": [2.2, 0.9, 1.7, 1.5, 2.4, 2.3, 2.5],
    "construct_comparison_5pct": [1.8, 2.8, 0.8, 1.6, 1.8, 1.7, 2.0],
    "coverage_comparison_5pct": [2.2, 1.0, 3.2, 3.0, 3.0],
    "subgroup_results": [5.4, 1.5, 3.3, 3.3],
    "segment_lambdas": [5.0, 1.2, 3.5, 3.5],
    "robustness_results": [3.5, 1.0, 2.6, 2.6, 3.8],
    "candidate_frontier": [1.1, 1.1, 3.0, 3.0, 5.3],
    "tag_results": [3.7, 1.0, 2.7, 3.0, 3.1],
    "runtime_results": [3.0, 3.4, 4.2, 2.9],
    "hypotheses": [0.9, 4.6, 2.2, 5.8],
    "protocol_deviations": [2.1, 4.0, 4.4, 3.0],
    "ai_use": [2.6, 2.0, 5.1, 3.8],
    "parameters": [5.0, 8.5],
    "all_budget_methods": [1.1, 2.4, 1.4, 0.8, 2.3, 2.6, 2.9],
    "markdown": [3.6, 9.9],
}


def scale_widths(widths_cm: Iterable[float]) -> list[float]:
    widths = list(widths_cm)
    scale = (TEXT_W / cm) / sum(widths)
    return [width * scale * cm for width in widths]


def table_flowable(name: str, headers: list[str], rows: list[list[object]], styles) -> LongTable:
    wide = len(headers) >= 5
    font_size = 7.1 if wide else 8.2
    leading = 8.6 if wide else 10.2
    header_style = ParagraphStyle(
        f"TableHeader-{name}",
        fontName=FONT_BOLD,
        fontSize=font_size,
        leading=leading,
        alignment=TA_CENTER,
        textColor=colors.black,
    )
    cell_left = ParagraphStyle(
        f"TableCellLeft-{name}",
        fontName=FONT_REGULAR,
        fontSize=font_size,
        leading=leading,
        alignment=TA_LEFT,
        splitLongWords=1,
        wordWrap=WORD_WRAP_MODE,
    )
    cell_center = ParagraphStyle(
        f"TableCellCenter-{name}",
        fontName=FONT_REGULAR,
        fontSize=font_size,
        leading=leading,
        alignment=TA_CENTER,
        splitLongWords=1,
        wordWrap=WORD_WRAP_MODE,
    )
    formatted: list[list[Paragraph]] = []
    formatted.append([Paragraph(escape(normalize_punctuation(str(value))), header_style) for value in headers])
    for row in rows:
        cells = []
        for index, value in enumerate(row):
            prose_columns = {
                "hypotheses": {1, 3},
                "protocol_deviations": {1, 2, 3},
                "ai_use": {2, 3},
            }
            style = cell_left if index == 0 or index in prose_columns.get(name, set()) else cell_center
            cells.append(Paragraph(escape(normalize_punctuation(str(value))), style))
        formatted.append(cells)

    widths = TABLE_WIDTHS_CM.get(name)
    if widths is None or len(widths) != len(headers):
        widths = [TEXT_W / cm / len(headers)] * len(headers)
    table = LongTable(
        formatted,
        colWidths=scale_widths(widths),
        repeatRows=1,
        hAlign="LEFT",
        splitByRow=1,
    )
    vertical_padding = 1.0 if name == "all_budget_methods" else 3.6
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#D9EAF7")),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#8193A5")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 3.2),
        ("RIGHTPADDING", (0, 0), (-1, -1), 3.2),
        ("TOPPADDING", (0, 0), (-1, -1), vertical_padding),
        ("BOTTOMPADDING", (0, 0), (-1, -1), vertical_padding),
    ]
    for row_index in range(2, len(formatted), 2):
        commands.append(("BACKGROUND", (0, row_index), (-1, row_index), colors.HexColor("#F3F6F8")))
    table.setStyle(TableStyle(commands))
    return table


def parse_markdown_table(lines: list[str], start: int) -> tuple[int, list[str], list[list[str]]]:
    block = []
    index = start
    while index < len(lines) and lines[index].strip().startswith("|"):
        block.append(lines[index].strip())
        index += 1
    parsed = [[cell.strip() for cell in row.strip("|").split("|")] for row in block]
    if len(parsed) >= 2 and all(set(cell) <= {"-", ":"} for cell in parsed[1]):
        return index, parsed[0], parsed[2:]
    return index, [], []


def build_story(styles: dict[str, ParagraphStyle], doc: ThesisDocTemplate):
    raw = MANUSCRIPT.read_text(encoding="utf-8")
    first_break = raw.index("@@PAGEBREAK@@")
    content = raw[first_break + len("@@PAGEBREAK@@"):]
    for heading in BODY_START_HEADINGS:
        content = content.replace(f"@@PAGEBREAK@@\n\n# {heading}", f"# {heading}")
    lines = content.splitlines()

    data = pd.read_csv(AGG / "all_thesis_results.csv")
    tables = make_tables(data)
    story: list = []
    story.extend(title_page(styles))
    story.extend([NextPageTemplate("front"), PageBreak()])

    body_started = False
    in_bibliography = False
    heading_seen = False
    last_was_break = True
    figure_number = 0
    table_number = 0
    in_appendix = False
    appendix_letter = ""
    appendix_table_numbers: dict[str, int] = {}

    def add_break() -> None:
        nonlocal last_was_break
        if not last_was_break:
            story.append(PageBreak())
            last_was_break = True

    def add_flowable(flowable) -> None:
        nonlocal last_was_break
        story.append(flowable)
        last_was_break = isinstance(flowable, PageBreak)

    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped == "@@PAGEBREAK@@":
            add_break()
            index += 1
            continue
        if stripped == "@@TOC@@":
            toc = NamedTableOfContents("TOCEntry")
            toc.levelStyles = [styles["TOC0"], styles["TOC1"], styles["TOC2"]]
            toc.dotsMinLevel = 0
            toc.formatter = lambda page, d=doc: (
                "" if not page else
                str(page - d.body_start_page + 1)
                if d.body_start_page is not None and page >= d.body_start_page
                else roman(page)
            )
            add_flowable(toc)
            index += 1
            continue
        if stripped == "@@TOC_FIGURES@@":
            toc = NamedTableOfContents("FIGEntry")
            toc.levelStyles = [styles["ListEntry"]]
            toc.dotsMinLevel = 0
            toc.formatter = lambda page, d=doc: (
                "" if not page else
                str(page - d.body_start_page + 1)
                if d.body_start_page is not None and page >= d.body_start_page
                else roman(page)
            )
            add_flowable(toc)
            index += 1
            continue
        if stripped == "@@TOC_TABLES@@":
            toc = NamedTableOfContents("TABLEEntry")
            toc.levelStyles = [styles["ListEntry"]]
            toc.dotsMinLevel = 0
            toc.formatter = lambda page, d=doc: (
                "" if not page else
                str(page - d.body_start_page + 1)
                if d.body_start_page is not None and page >= d.body_start_page
                else roman(page)
            )
            add_flowable(toc)
            index += 1
            continue
        if stripped == "@@TOC_EQUATIONS@@":
            toc = NamedTableOfContents("EQEntry")
            toc.levelStyles = [styles["ListEntry"]]
            toc.dotsMinLevel = 0
            toc.formatter = lambda page, d=doc: (
                "" if not page else
                str(page - d.body_start_page + 1)
                if d.body_start_page is not None and page >= d.body_start_page
                else roman(page)
            )
            add_flowable(toc)
            index += 1
            continue

        figure_match = re.fullmatch(r"@@FIG:([^|]+)\|(.+)@@", stripped)
        if figure_match:
            filename, caption_text = figure_match.groups()
            figure_number += 1
            clean_caption = re.sub(
                rf"^(?:{re.escape(FIGURE_LABEL)})\s*\d+[：:]\s*", "", caption_text
            )
            image_path = FIGURES / filename
            if not image_path.exists():
                raise FileNotFoundError(image_path)
            image = Image(str(image_path))
            image._restrictSize(13.1 * cm, 10.3 * cm)
            image.hAlign = "CENTER"
            caption = make_paragraph(
                f"{FIGURE_LABEL} {figure_number}: {clean_caption}", styles["FigureCaption"]
            )
            caption._thesis_section = "body"
            add_flowable(KeepTogether([Spacer(1, 4), image, caption]))
            index += 1
            continue

        table_match = re.fullmatch(r"@@TABLE:([^@]+)@@", stripped)
        if table_match:
            name = table_match.group(1)
            if name != "ai_use":
                if in_appendix:
                    appendix_table_numbers[appendix_letter] = appendix_table_numbers.get(appendix_letter, 0) + 1
                    visible_number = f"{appendix_letter}.{appendix_table_numbers[appendix_letter]}"
                else:
                    table_number += 1
                    visible_number = str(table_number)
                caption = make_paragraph(
                    f"{TABLE_LABEL} {visible_number}: {TABLE_TITLES[name]}", styles["TableCaption"]
                )
                caption._thesis_section = "body"
                add_flowable(caption)
            headers, rows = tables[name]
            add_flowable(table_flowable(name, headers, rows, styles))
            add_flowable(Spacer(1, 5))
            index += 1
            continue

        equation_match = re.fullmatch(r"@@EQ:(\d+)\|(.+)@@", stripped)
        if equation_match:
            number, _formula_source = equation_match.groups()
            add_flowable(equation_flowable(number, styles))
            index += 1
            continue

        if stripped.startswith("|"):
            index, headers, rows = parse_markdown_table(lines, index)
            if headers:
                add_flowable(table_flowable("markdown", headers, rows, styles))
                add_flowable(Spacer(1, 5))
            continue

        if stripped.startswith("# "):
            heading = stripped[2:]
            if heading in BODY_START_HEADINGS and not body_started:
                add_flowable(NextPageTemplate("body"))
                add_break()
                body_started = True
            elif heading_seen:
                add_break()
            paragraph = make_paragraph(heading, styles["Heading1"])
            paragraph._thesis_section = "body" if body_started else "front"
            add_flowable(paragraph)
            heading_seen = True
            if heading in BIBLIOGRAPHY_HEADINGS:
                in_bibliography = True
                in_appendix = False
            elif heading in AUXILIARY_HEADINGS:
                in_bibliography = False
                in_appendix = False
            elif heading.startswith(APPENDIX_PREFIXES) or heading in DECLARATION_HEADINGS:
                in_bibliography = False
                appendix_match = re.match(r"Anhang\s+([A-Z])", heading)
                if appendix_match:
                    in_appendix = True
                    appendix_letter = appendix_match.group(1)
                elif heading in DECLARATION_HEADINGS:
                    in_appendix = False
            index += 1
            continue

        if stripped.startswith("## "):
            paragraph = make_paragraph(stripped[3:], styles["Heading2"])
            paragraph._thesis_section = "body" if body_started else "front"
            add_flowable(paragraph)
            index += 1
            continue
        if stripped.startswith("### "):
            paragraph = make_paragraph(stripped[4:], styles["Heading3"])
            paragraph._thesis_section = "body" if body_started else "front"
            add_flowable(paragraph)
            index += 1
            continue
        number_match = re.match(r"^(\d+)\.\s+(.+)", stripped)
        if number_match:
            paragraph = make_paragraph(
                number_match.group(2), styles["Number"], bulletText=f"{number_match.group(1)}."
            )
            add_flowable(paragraph)
            index += 1
            continue
        if stripped.startswith("- "):
            paragraph = make_paragraph(stripped[2:], styles["Bullet"], bulletText="•")
            add_flowable(paragraph)
            index += 1
            continue

        style = styles["Bibliography"] if in_bibliography else styles["Body"]
        paragraph = make_paragraph(stripped, style)
        add_flowable(KeepTogether([paragraph]) if in_bibliography else paragraph)
        index += 1

    return story


def build(output: Path) -> None:
    register_fonts()
    styles = build_styles()
    output.parent.mkdir(parents=True, exist_ok=True)
    doc = ThesisDocTemplate(str(output), styles)
    story = build_story(styles, doc)
    doc.multiBuild(story, maxPasses=6)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    build(args.output)
    print(f"[OK] wrote {args.output}")


if __name__ == "__main__":
    main()
