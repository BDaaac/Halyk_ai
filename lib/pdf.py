"""Извлечение текста PDF с маршрутизацией сканов в vision-ветку."""

from pathlib import Path
from typing import Literal

import fitz


ExtractionMethod = Literal["text", "vision"]


def vision_extract(pages: list[fitz.Pixmap]) -> str:
    """Временная заглушка для будущего vision-вызова стадии 2."""
    return ""


def low_text_page_indices(path: Path, *, minimum_characters: int = 50) -> list[int]:
    """Return zero-based pages whose visible content needs vision/OCR.

    A document can have a healthy text layer overall while a material table is
    embedded as an image.  The document-level fallback alone cannot see that.
    """
    with fitz.open(path) as document:
        return [
            page_number
            for page_number, page in enumerate(document)
            if len(page.get_text().strip()) < minimum_characters
        ]


def extract_text(path: Path) -> tuple[str, ExtractionMethod]:
    """Возвращает извлечённый текст и способ его получения."""
    with fitz.open(path) as document:
        text = "\n".join(page.get_text() for page in document)
        if len(text.strip()) < 200:
            pages = [page.get_pixmap(matrix=fitz.Matrix(2, 2)) for page in document[:3]]
            return vision_extract(pages), "vision"
    return text, "text"
