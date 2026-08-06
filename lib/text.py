"""Точечная нормализация идентификаторов из PDF."""

import re
import unicodedata


ACC_RE = re.compile(r"[AА]\s*[CС]\s*[CС]\s*[-–—]\s*(\d)\s*(\d)\s*(\d)\s*(\d)")


def normalize_identifiers(text: str) -> str:
    """Исправляет только свёрстанные ACC-идентификаторы, не меняя прозу."""
    normalized = unicodedata.normalize("NFKC", text)
    return ACC_RE.sub(lambda match: "ACC-" + "".join(match.groups()), normalized)
