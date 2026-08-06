"""Точечная нормализация идентификаторов из PDF."""

import re


ACC_RE = re.compile(r"[AА]\s*[CС]\s*[CС]\s*[-–—]\s*(\d)\s*(\d)\s*(\d)\s*(\d)")


def normalize_identifiers(text: str) -> str:
    """Исправляет только свёрстанные ACC-идентификаторы, не меняя прозу."""
    return ACC_RE.sub(lambda match: "ACC-" + "".join(match.groups()), text)
