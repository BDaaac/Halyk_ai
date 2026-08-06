"""Детерминированный патч related_parties из vision-извлечения KYC.

Vision-модель возвращает страницу дословно. Числа и имена достаём здесь
регексом, а не отдельным LLM-вызовом — правило handoff: ни одно число в
submission не приходит от модели.
"""

from __future__ import annotations

import re
from decimal import Decimal
from pathlib import Path
from typing import Any


PERCENT_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%")
THRESHOLD_RE = re.compile(
    r"(?:владе(?:ет|ют)|принадлежит|доля[^%]{0,40}?)\s*(\d+(?:\.\d+)?)\s*%\s+и\s+более",
    re.IGNORECASE,
)
SVYSHE_THRESHOLD_RE = re.compile(r"свыше\s+(\d+(?:\.\d+)?)\s*%", re.IGNORECASE)
FALLBACK_THRESHOLD_RE = re.compile(r"(\d+(?:\.\d+)?)\s*%\s+и\s+более", re.IGNORECASE)
ENTITY_LINE_RE = re.compile(r"^\s*(.+?)\s*\|\s*(\d+(?:\.\d+)?)\s*%\s*$", re.MULTILINE)


def vision_pages(doc_id: str, workspace_dir: Path) -> list[Path]:
    directory = workspace_dir / "vision"
    if not directory.exists():
        return []
    return sorted(directory.glob(f"{doc_id}_p*.txt"))


def load_vision_text(doc_id: str, workspace_dir: Path) -> str:
    return "\n".join(page.read_text(encoding="utf-8") for page in vision_pages(doc_id, workspace_dir))


def extract_related_parties(text: str) -> dict[str, Any] | None:
    """Return `{threshold_percent, entities}` when the text contains both signals."""
    threshold_match = (
        THRESHOLD_RE.search(text)
        or SVYSHE_THRESHOLD_RE.search(text)
        or FALLBACK_THRESHOLD_RE.search(text)
    )
    if threshold_match is None:
        return None
    threshold = str(Decimal(threshold_match.group(1)))

    entities: list[dict[str, str]] = []
    seen: set[str] = set()
    for match in ENTITY_LINE_RE.finditer(text):
        name = match.group(1).strip().strip('"').strip("«»").strip('"')
        if not name or name.lower().startswith(("организац", "доля", "конечный", "дочерн")):
            continue
        if name in seen:
            continue
        seen.add(name)
        entities.append({"name": name, "share_percent": str(Decimal(match.group(2)))})
    if not entities:
        return None
    return {"threshold_percent": threshold, "entities": entities}


def patch_extraction(extraction: dict[str, Any], vision_text: str, source_doc_id: str) -> bool:
    """Overwrite `related_parties` when the current value has no threshold.

    Returns True when a patch was applied. Does not touch covenants,
    adjustments or facts.
    """
    parsed = extract_related_parties(vision_text)
    if parsed is None:
        return False
    current = extraction.get("related_parties", {})
    if current.get("threshold_percent"):
        return False
    extraction["related_parties"] = {
        "threshold_percent": parsed["threshold_percent"],
        "entities": parsed["entities"],
        "source": {"doc_id": source_doc_id, "quote": "vision-derived"},
    }
    return True
