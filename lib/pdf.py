"""Постраничное извлечение текста PDF с vision-фоллбэком.

`extract_text` возвращает склеенный текст всего документа и общий метод. Если
хоть одна страница ушла в vision — метод документа считается `"vision"`.
Vision-ответы кэшируются в `workspace/vision/`, чтобы повторный прогон и тесты
не расходовали API.
"""

from __future__ import annotations

import base64
import os
from pathlib import Path
from typing import Literal

import fitz


ExtractionMethod = Literal["text", "vision"]

PAGE_TEXT_THRESHOLD = 80
"""Ниже этого — маршрутизируем страницу в vision. См. handoff: пороговые
значения на 7 проблемных страницах — от 0 до 37 символов."""

VISION_PROMPT = (
    "Ты извлекаешь содержимое отсканированной страницы PDF. "
    "На этих страницах могут быть таблицы: доли владения в процентах или "
    "аудиторские корректировки. Верни всё в plain text, построчно; для таблиц "
    "используй разделитель ' | '. Числа переноси ДОСЛОВНО, не округляй и не "
    "форматируй. Проценты сохраняй с исходной точностью (например 38.1%). "
    "Идентификаторы вида ACC-XXXX, TXN-XXX-NNNN, AR-XXXX-XXXX переноси "
    "без пробелов и без изменений. Ничего не добавляй от себя."
)


def _page_needs_vision(page: fitz.Page) -> bool:
    return len(page.get_text().strip()) < PAGE_TEXT_THRESHOLD


def low_text_page_indices(path: Path, *, minimum_characters: int = PAGE_TEXT_THRESHOLD) -> list[int]:
    """Zero-based indices of pages whose text layer is too thin to trust."""
    with fitz.open(path) as document:
        return [
            index
            for index, page in enumerate(document)
            if len(page.get_text().strip()) < minimum_characters
        ]


def _vision_cache_path(pdf_path: Path, page_index: int) -> Path:
    workspace = Path(os.getenv("WORKSPACE_DIR", pdf_path.parent.parent.parent / "workspace"))
    directory = workspace / "vision"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{pdf_path.stem}_p{page_index}.txt"


def _render_page_png(page: fitz.Page) -> bytes:
    pixmap = page.get_pixmap(matrix=fitz.Matrix(2, 2))
    return pixmap.tobytes("png")


def _call_vision_api(png_bytes: bytes) -> str:
    """Вызов Anthropic API с картинкой. При отсутствии ключа — пустая строка."""
    from config import get_settings
    from lib.anthropic_client import AnthropicClient

    settings = get_settings()
    if not settings.anthropic_api_key or not settings.select_model:
        return ""

    client = AnthropicClient(
        api_key=settings.anthropic_api_key,
        model=settings.select_model,
        timeout_seconds=settings.llm_timeout_seconds,
    )
    encoded = base64.standard_b64encode(png_bytes).decode("ascii")
    response = client.create_message(
        max_tokens=4096,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": encoded},
                    },
                    {"type": "text", "text": VISION_PROMPT},
                ],
            }
        ],
    )
    blocks = [block for block in response.get("content", []) if block.get("type") == "text"]
    return "\n".join(block.get("text", "") for block in blocks).strip()


def vision_extract_page(pdf_path: Path, page: fitz.Page, page_index: int) -> str:
    """Кэшированный vision-разбор одной страницы."""
    cache = _vision_cache_path(pdf_path, page_index)
    if cache.exists():
        return cache.read_text(encoding="utf-8")
    text = _call_vision_api(_render_page_png(page))
    cache.write_text(text, encoding="utf-8")
    return text


def extract_text(path: Path) -> tuple[str, ExtractionMethod]:
    """Возвращает склеенный текст документа и его общий метод извлечения."""
    parts: list[str] = []
    used_vision = False
    with fitz.open(path) as document:
        for index, page in enumerate(document):
            if _page_needs_vision(page):
                parts.append(vision_extract_page(path, page, index))
                used_vision = True
            else:
                parts.append(page.get_text())
    return "\n".join(parts), ("vision" if used_vision else "text")


# Backwards-compatible name kept for existing tests that patch it.
def vision_extract(pages: list[fitz.Pixmap]) -> str:  # pragma: no cover - legacy shim
    return ""
