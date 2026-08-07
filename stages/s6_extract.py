"""Stage 6: one structured extraction call per borrower."""

from __future__ import annotations

import json
import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from models import DocumentSet


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "stage6_extract.md"
DERIVED_ROLES = {"ebitda", "net_income", "working_capital", "operating_profit"}


class DerivedRoleError(ValueError):
    pass


@dataclass(frozen=True)
class ExtractionResult:
    output: dict[str, Any]
    usage: dict[str, int]
    soft_warnings: list[str]


EXTRACTION_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "covenants": {"type": "array", "items": {"type": "object"}},
        "adjustments": {"type": "array", "items": {"type": "object"}},
        "related_parties": {"type": "object"},
        "document_facts": {"type": "array", "items": {"type": "object"}},
    },
    "required": ["covenants", "adjustments", "related_parties", "document_facts"],
}


def _text(document_texts: dict[str, str], doc_id: str | None) -> str:
    return document_texts.get(doc_id, "") if doc_id else ""


def _article_six(text: str) -> str:
    starts = list(re.finditer(r"(?:article|статья)\s*6\b", text, re.IGNORECASE))
    if not starts:
        return text
    candidates: list[str] = []
    for start in starts:
        following = re.search(r"(?:article|статья)\s*7\b", text[start.end() :], re.IGNORECASE)
        end = start.end() + following.start() if following else len(text)
        candidates.append(text[start.start() : end])
    with_clauses = [segment for segment in candidates if re.search(r"(?:пункт|clause)\s*6\.\d+", segment, re.IGNORECASE)]
    # If neither an "article 6" nor any of its candidate windows contains a
    # "clause 6.X" marker, the agreement almost certainly numbers its
    # covenants under a different heading. Hand the model the whole
    # document rather than a section that turned out to be empty.
    if not with_clauses:
        return text
    return max(with_clauses, key=len)


def build_context(
    *,
    documents: DocumentSet,
    document_texts: dict[str, str],
    clause_ids: list[str],
    extra_documents: list[str] | None = None,
) -> str:
    extras = "\n\n".join(_text(document_texts, doc_id) for doc_id in extra_documents or [])
    return "\n".join(
        [
            "<covenant_article>",
            _article_six(_text(document_texts, documents.agreement)),
            "</covenant_article>",
            "<audit_notes>",
            _text(document_texts, documents.audit_notes),
            "</audit_notes>",
            "<aup_report>",
            _text(document_texts, documents.aup_report),
            "</aup_report>",
            "<kyc>",
            _text(document_texts, documents.kyc),
            "</kyc>",
            "<extra_documents>",
            extras,
            "</extra_documents>",
            f"<clause_ids>{', '.join(clause_ids)}</clause_ids>",
        ]
    )


def _contains_op(node: Any, expected: str) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("op") == expected:
        return True
    return any(_contains_op(child, expected) for child in node.get("args", []))


def _clause_text(article: str, clause_id: str) -> str:
    marker_pattern = rf"(?im)(?:^|\n)\s*(?:пункт|clause)?\s*{re.escape(clause_id)}\b"
    marker = re.search(marker_pattern, article)
    if marker is None:
        return article
    next_marker = re.search(r"(?im)(?:^|\n)\s*(?:пункт|clause)?\s*6\.\d+\b", article[marker.end() :])
    end = marker.end() + next_marker.start() if next_marker else len(article)
    return article[marker.start() : end]


# Regex families for regex_guards. Kept as module-level tuples so the
# bilingual coverage can be grep-ed and extended in one place, and so
# tests can assert against them without stringifying the guard function.

LOWER_BOUND_PATTERNS = (
    # Russian
    r"не\s+допуск\w+\s+снижен",
    r"не\s+менее",
    r"не\s+ниже",
    # English
    r"not\s+(?:be\s+)?less\s+than",
    r"\bat\s+least\b",
    r"shall\s+not\s+(?:permit\s+.{0,40}\s+to\s+)?fall\s+below",
    r"shall\s+maintain\s+.{0,40}of\s+(?:not\s+)?(?:less\s+than\s+)?",
    r"minimum\s+of",
)
UPPER_BOUND_PATTERNS = (
    # Russian
    r"не\s+должн\w+\s+превыш",
    r"не\s+более",
    r"не\s+выше",
    r"не\s+превышал",
    # English
    r"shall\s+not\s+exceed",
    r"\bnot\s+(?:exceed|more\s+than)\b",
    r"\bno\s+greater\s+than\b",
    r"\bno\s+more\s+than\b",
    r"maximum\s+of",
)
MAX_NODE_PATTERNS = (
    # Russian
    r"наибольш",
    # English
    r"\bgreater\s+of\b",
    r"\bhigher\s+of\b",
    r"not\s+a\s+metric",
    r"\bseparately\b",
)
APPLICABILITY_PATTERNS = (
    # Russian
    r"только\s+при\s+условии",
    r"применяется,?\s+если",
    # English
    r"only\s+(?:if|when)",
    r"provided\s+that",
    r"subject\s+to",
    r"applies\s+only\s+(?:if|when)",
)
EXCEPTION_PATTERNS = (
    # Russian
    r"за\s+исключением\s+случаев",
    # English
    r"\bexcept\s+(?:as\s+.{0,40}\s+)?agreed\s+in\s+writing",
    r"unless\s+otherwise",
    r"\bexcept\b",
)


def _any_match(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def regex_guards(output: dict[str, Any], article: str) -> list[str]:
    """Flag deterministic disagreements; never change the model output."""
    warnings: list[str] = []
    for covenant in output.get("covenants", []):
        clause_id = str(covenant.get("clause_id", "?"))
        for role in covenant.get("role_descriptions", {}):
            if str(role).casefold() in DERIVED_ROLES:
                raise DerivedRoleError(f"{role} is a derived metric and must be expressed as a tree")
        clause = _clause_text(article, clause_id).lower()
        operator = covenant.get("operator")
        if _any_match(LOWER_BOUND_PATTERNS, clause) and operator != ">=":
            warnings.append(f"{clause_id}: operator disagrees with lower-bound clause")
        if _any_match(UPPER_BOUND_PATTERNS, clause) and operator != "<=":
            warnings.append(f"{clause_id}: operator disagrees with upper-bound clause")
        if _any_match(MAX_NODE_PATTERNS, clause) and not _contains_op(covenant.get("value_expr"), "max"):
            warnings.append(f"{clause_id}: max is expected by the clause")
        if _any_match(APPLICABILITY_PATTERNS, clause) and covenant.get("applicability") is None:
            warnings.append(f"{clause_id}: applicability condition is missing")
        if _any_match(EXCEPTION_PATTERNS, clause) and covenant.get("exception") is None:
            warnings.append(f"{clause_id}: exception condition is missing")
        threshold = str(covenant.get("threshold", "")).replace(",", "")
        if threshold and threshold not in clause.replace(",", ""):
            warnings.append(f"{clause_id}: threshold is not present verbatim in the clause")
    return warnings


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = file.name
    os.replace(temporary, path)


def extract_scenario(
    *,
    scenario_id: str,
    documents: DocumentSet,
    document_texts: dict[str, str],
    clause_ids: list[str],
    client: Any,
    cache_dir: Path,
    extra_documents: list[str] | None = None,
) -> ExtractionResult:
    cache_path = cache_dir / f"{scenario_id}.json"
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        regex_guards(cached["output"], _article_six(_text(document_texts, documents.agreement)))
        return ExtractionResult(**cached)

    context = build_context(
        documents=documents,
        document_texts=document_texts,
        clause_ids=clause_ids,
        extra_documents=extra_documents,
    )
    response = client.create_structured_message(
        system=PROMPT_PATH.read_text(encoding="utf-8"),
        user=context,
        tool_name="emit_extraction",
        input_schema=EXTRACTION_SCHEMA,
    )
    result = ExtractionResult(
        output=response.output,
        usage=response.usage,
        soft_warnings=regex_guards(response.output, _article_six(_text(document_texts, documents.agreement))),
    )
    _write_cache(cache_path, {"output": result.output, "usage": result.usage, "soft_warnings": result.soft_warnings})
    return result
