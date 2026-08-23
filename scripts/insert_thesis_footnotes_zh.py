"""Insert the 29 translated true Word footnotes into the Chinese thesis."""

from __future__ import annotations

import insert_thesis_footnotes as base
from build_thesis_pdf_zh import FOOTNOTES_ZH


if __name__ == "__main__":
    base.FOOTNOTES = FOOTNOTES_ZH
    base.main()
