"""Insert the manuscript's true Word footnotes using the bundled OOXML helper."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import shutil
import zipfile

from build_thesis_docx import FOOTNOTES


HELPER = Path(
    r"C:\Users\Aroeh\.codex\plugins\cache\openai-primary-runtime\documents\26.819.11345"
    r"\skills\documents\scripts\insert_note.py"
)


def load_helper():
    spec = importlib.util.spec_from_file_location("codex_insert_note", HELPER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load {HELPER}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    helper = load_helper()
    reset = args.output.with_name(f".{args.output.stem}.fn-reset.tmp.docx")
    with zipfile.ZipFile(args.input, "r") as zin, zipfile.ZipFile(reset, "w", zipfile.ZIP_DEFLATED) as zout:
        for info in zin.infolist():
            if info.filename == "word/footnotes.xml":
                zout.writestr(info, helper._xml_bytes(helper._make_empty_notes_part("footnote")))
            else:
                zout.writestr(info, zin.read(info.filename))
    current = reset
    temps: list[Path] = [reset]
    for index, (marker, text) in enumerate(FOOTNOTES.items(), start=1):
        target = args.output.with_name(f".{args.output.stem}.fn{index:02d}.tmp.docx")
        helper.insert_note(str(current), str(target), "footnote", marker, text)
        temps.append(target)
        current = target
    shutil.copy2(current, args.output)
    for temp in temps:
        temp.unlink(missing_ok=True)
    print(f"[OK] inserted {len(FOOTNOTES)} true footnotes and wrote {args.output}")


if __name__ == "__main__":
    main()
