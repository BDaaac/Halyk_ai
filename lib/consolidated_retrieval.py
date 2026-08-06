"""Detrministic retrieval of parent-company / consolidated documents.

Handoff task 2: a covenant may reference a figure that lives in a group's
consolidated statements, not the borrower's ledger. The document has no
`ACC-` marker, so triage classifies it as noise. We look it up by the
borrower's own name and, when found, derive the required amount by regex
rather than by another LLM call — the pipeline invariant is that no
number in `submission.json` comes from a model.
"""

from __future__ import annotations

import re
from decimal import Decimal
from typing import Iterable

from models import DocRecord


BORROWER_NAME_RE = re.compile(
    r"(?:Организация|Заёмщик)\s*[|:]?\s*([A-Z][\w'-]*(?:\s+[A-Z][\w'-]*)*\s+(?:JSC|LLP|Ltd|LLC))",
    re.MULTILINE,
)
GROUP_MARKERS = (
    "консолидированной отчётности",
    "консолидированной отчетности",
    "конечной материнской",
    "консолидированного отчёта",
    "консолидированного отчета",
    "Группы",  # only meaningful in the ratio's numerator description
)
PPE_BEGIN_RE = re.compile(r"Net book value at the beginning of the year\s*\n?\s*\$([\d,]+(?:\.\d+)?)", re.IGNORECASE)
PPE_END_RE = re.compile(r"Net book value at the end of the year\s*\n?\s*\$([\d,]+(?:\.\d+)?)", re.IGNORECASE)
DEPRECIATION_RE = re.compile(r"Depreciation charge for the year\s*\n?\s*\$([\d,]+(?:\.\d+)?)", re.IGNORECASE)


def borrower_name(text: str) -> str | None:
    match = BORROWER_NAME_RE.search(text)
    return match.group(1).strip() if match else None


def clause_mentions_group(clause_text: str) -> bool:
    return any(marker in clause_text for marker in GROUP_MARKERS)


CONSOLIDATED_MARKER_RE = re.compile(
    r"consolidated\s+(?:financial\s+statements?|annual\s+report|statement\s+of)",
    re.IGNORECASE,
)


def find_consolidated_reports(borrower: str, documents: Iterable[DocRecord]) -> list[str]:
    """Return doc_ids of orphan consolidated statements naming the borrower."""
    hits: list[str] = []
    for document in documents:
        if document.account_ids:
            continue
        collapsed = re.sub(r"\s+", " ", document.text)
        if borrower not in collapsed:
            continue
        if not CONSOLIDATED_MARKER_RE.search(document.text):
            continue
        hits.append(document.doc_id)
    return hits


def _rewrite_sum_role_to_fact(node: object, from_role: str, fact_name: str) -> bool:
    if not isinstance(node, dict):
        return False
    if node.get("op") == "sum" and node.get("role") == from_role:
        node.clear()
        node["op"] = "fact"
        node["fact_name"] = fact_name
        return True
    return any(_rewrite_sum_role_to_fact(arg, from_role, fact_name) for arg in node.get("args", []))


def patch_group_capex(extraction: dict, doc_index: dict, kyc_text: str) -> bool:
    """Rewrite `sum(capex)` -> `fact(capital_expenditures_group)` where a
    covenant explicitly names the Group. Adds the derived fact to
    `document_facts` so stage 8 can evaluate it. Idempotent."""
    borrower = borrower_name(kyc_text)
    if not borrower:
        return False
    hits = find_consolidated_reports(borrower, doc_index.values())
    if not hits:
        return False
    report = doc_index[hits[0]]
    capex = derive_group_capex(report.text)
    if capex is None:
        return False
    existing = {fact.get("fact_name") for fact in extraction.get("document_facts", [])}
    if "capital_expenditures_group" in existing:
        return False
    patched_any = False
    for covenant in extraction.get("covenants", []):
        quote = covenant.get("source", {}).get("quote", "") or ""
        if not clause_mentions_group(quote):
            continue
        if _rewrite_sum_role_to_fact(covenant.get("value_expr", {}), "capex", "capital_expenditures_group"):
            patched_any = True
    if not patched_any:
        return False
    extraction.setdefault("document_facts", []).append({
        "fact_name": "capital_expenditures_group",
        "value": str(capex),
        "unit": "usd",
        "period": "2025",
        "source": {"doc_id": report.doc_id, "quote": "derived: end_PPE - begin_PPE + depreciation"},
    })
    return True


def derive_group_capex(text: str) -> Decimal | None:
    """Group capex = PPE end - PPE begin + depreciation charge.

    Standard IFRS reconciliation for a period with no disposals. The
    consolidated report states "There were no disposals of property, plant
    and equipment during the year" — validated by the caller.
    """
    begin = PPE_BEGIN_RE.search(text)
    end = PPE_END_RE.search(text)
    depreciation = DEPRECIATION_RE.search(text)
    if not (begin and end and depreciation):
        return None
    if "no disposals" not in text.lower():
        return None
    begin_amount = Decimal(begin.group(1).replace(",", ""))
    end_amount = Decimal(end.group(1).replace(",", ""))
    depreciation_amount = Decimal(depreciation.group(1).replace(",", ""))
    return (end_amount - begin_amount + depreciation_amount).quantize(Decimal("0.01"))
