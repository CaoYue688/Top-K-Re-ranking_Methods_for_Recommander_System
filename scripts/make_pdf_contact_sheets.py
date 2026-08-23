"""Compose four rendered PDF pages per contact sheet for visual QA."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    pages = sorted(args.input_dir.glob("page-*.png"))
    if not pages:
        raise SystemExit("no page images found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample = Image.open(pages[0])
    page_w, page_h = sample.size
    label_h = 34
    gap = 10
    font = ImageFont.truetype("arial.ttf", 22)
    for start in range(0, len(pages), 4):
        group = pages[start : start + 4]
        sheet = Image.new(
            "RGB",
            (2 * page_w + 3 * gap, 2 * (page_h + label_h) + 3 * gap),
            "#808080",
        )
        draw = ImageDraw.Draw(sheet)
        for offset, path in enumerate(group):
            row, col = divmod(offset, 2)
            x = gap + col * (page_w + gap)
            y = gap + row * (page_h + label_h + gap)
            page = Image.open(path).convert("RGB")
            sheet.paste(page, (x, y + label_h))
            draw.rectangle((x, y, x + page_w, y + label_h), fill="white")
            draw.text((x + 8, y + 4), path.stem, fill="black", font=font)
        end = start + len(group)
        output = args.output_dir / f"contact-{start + 1:03d}-{end:03d}.jpg"
        sheet.save(output, quality=88, subsampling=0)
    print(f"[OK] wrote {len(list(args.output_dir.glob('contact-*.jpg')))} contact sheets")


if __name__ == "__main__":
    main()
