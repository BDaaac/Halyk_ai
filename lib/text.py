"""Точечная нормализация идентификаторов из PDF."""

import re


ACC_RE = re.compile(r"[AА]\s*[CС]\s*[CС]\s*[-–—]\s*(\d)\s*(\d)\s*(\d)\s*(\d)")
REP_RE = re.compile(r"A\s*R\s*[-–—]\s*(\d)\s*(\d)\s*(\d)\s*(\d)\s*[-–—]\s*(\d)\s*(\d)\s*(\d)\s*(\d)")
TXN_RE = re.compile(r"T\s*X\s*N\s*[-–—]\s*([A-Z0-9]+)\s*[-–—]\s*(\d+)")


def normalize_identifiers(text: str) -> str:
    """Исправляет только свёрстанные ACC-идентификаторы, не меняя прозу."""
    text = ACC_RE.sub(lambda match: "ACC-" + "".join(match.groups()), text)
    text = REP_RE.sub(lambda match: "AR-" + "".join(match.groups()[:4]) + "-" + "".join(match.groups()[4:]), text)
    return TXN_RE.sub(lambda match: f"TXN-{match.group(1)}-{match.group(2)}", text)
