"""Извлечение текста PDF с маршрутизацией сканов в vision-ветку."""

from pathlib import Path
from typing import Literal

import fitz


ExtractionMethod = Literal["text", "vision"]


def vision_extract(pages: list[fitz.Pixmap]) -> str:
    """Временная заглушка для будущего vision-вызова стадии 2."""
    return ""


def extract_text(path: Path) -> tuple[str, ExtractionMethod]:
    """Возвращает извлечённый текст и способ его получения."""
    with fitz.open(path) as document:
        text = "\n".join(page.get_text() for page in document)
        if len(text.strip()) < 200:
            pages = [page.get_pixmap(matrix=fitz.Matrix(2, 2)) for page in document[:3]]
            return vision_extract(pages), "vision"
    return text, "text"
