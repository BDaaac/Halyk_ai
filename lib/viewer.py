"""Read-only HTML viewer for the covenants pipeline.

Assembles a self-contained HTML report from ``workspace/`` artifacts (the
submission, cached extractions and selections), the ledger, and — if
present — the ground-truth file. Zero mutations: only reads on-disk state
and re-invokes pure compute functions from stages/s{8,9} to fill in the
per-cell drill-down (formula substitution, counterfactual sweep). The
pipeline itself is untouched.

Usage:

    from lib.viewer import build_view
    build_view(
        workspace_dir=Path("workspace"),
        data_dir=Path("agentic-bank-public"),
        template_path=Path("templates/viewer.html"),
        output_path=Path("reports/view.html"),
        ground_truth_path=Path("agentic-bank-public/ground_truth.json"),
    )
"""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd

from stages.s8_compute import build_clause_selection, evaluate_covenant, specifications_from_extraction
from stages.s9_evidence import (
    counterfactual_kind,
    evidence_candidates,
    reverted_adjustments,
)
from lib.currency import normalize_ledger_to_usd


TOLERANCE_PCT = 0.02  # scorer's numeric tolerance; used only for display


def _decimal_to_native(value: Any) -> Any:
    """JSON-serializable form for Decimal and pandas scalars."""
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


def _first_line(text: str, max_chars: int = 90) -> str:
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            return stripped[:max_chars]
    return ""


def _load_json_or_none(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _cold_run_metrics(workspace_dir: Path) -> str | None:
    metrics_path = workspace_dir / "cold_run_metrics.txt"
    if not metrics_path.exists():
        return None
    return metrics_path.read_text(encoding="utf-8")


def _document_index_and_resolvers():
    """Import lazily so the viewer never triggers stage-2 vision spend at import."""
    from pipeline import build_document_index, resolve_documents, get_account_to_scenario
    return build_document_index, resolve_documents, get_account_to_scenario


def _expression_source(node: Any) -> str:
    """Pretty-print a value_expr as a compact expression string."""
    if not isinstance(node, dict):
        return str(node)
    op = node.get("op")
    if op == "sum":
        return f"sum({node.get('role', '?')})"
    if op == "fact":
        return f"fact({node.get('fact_name', '?')})"
    if op == "const":
        return str(node.get("value", "?"))
    args = [_expression_source(child) for child in node.get("args", [])]
    if op in {"add", "subtract", "divide", "max", "min"}:
        symbol = {"add": " + ", "subtract": " - ", "divide": " / ", "max": ", ", "min": ", "}[op]
        if op in {"max", "min"}:
            return f"{op}({symbol.join(args)})"
        return f"({symbol.join(args)})" if len(args) > 1 else args[0]
    return f"{op}({', '.join(args)})"


def _expression_with_values(node: Any, values: dict[str, list[Decimal]], facts: dict[str, Decimal]) -> str:
    if not isinstance(node, dict):
        return str(node)
    op = node.get("op")
    if op == "sum":
        role = node.get("role", "?")
        summands = values.get(role, [])
        joined = " + ".join(f"{float(v):,.2f}" for v in summands) or "0"
        total = sum(summands, Decimal("0")) if summands else Decimal("0")
        return f"sum({role}=[{joined}]={float(total):,.2f})"
    if op == "fact":
        name = node.get("fact_name", "?")
        return f"fact({name}={float(facts.get(name, Decimal('0'))):,.2f})"
    if op == "const":
        return str(node.get("value", "?"))
    args = [_expression_with_values(child, values, facts) for child in node.get("args", [])]
    if op in {"add", "subtract", "divide"}:
        symbol = {"add": " + ", "subtract": " - ", "divide": " / "}[op]
        return f"({symbol.join(args)})"
    if op in {"max", "min"}:
        return f"{op}({', '.join(args)})"
    return f"{op}({', '.join(args)})"


def _document_title(document_index: dict, doc_id: str | None) -> dict[str, str] | None:
    if not doc_id or doc_id not in document_index:
        return None
    record = document_index[doc_id]
    return {
        "doc_id": doc_id,
        "title": _first_line(record.text),
        "extraction_method": record.extraction_method,
        "version_status": record.version_status,
        "doc_type": record.doc_type,
    }


def _rejected_documents(document_index: dict, ids: list[str]) -> list[dict[str, str]]:
    payload: list[dict[str, str]] = []
    for doc_id in ids:
        if doc_id not in document_index:
            continue
        record = document_index[doc_id]
        payload.append(
            {
                "doc_id": doc_id,
                "title": _first_line(record.text),
                "version_status": record.version_status,
                "doc_type": record.doc_type,
                "reason": record.version_status,
            }
        )
    return payload


def _document_chain(
    scenario_id: str,
    document_index: dict,
    resolve_documents_fn,
    account_to_scenario: dict[str, str],
    audit_notes_report_number: str | None = None,
) -> dict[str, Any]:
    try:
        documents = resolve_documents_fn(scenario_id)
    except StopIteration:
        return {
            "agreement": None, "kyc": None, "aup_report": None, "audit_notes": None,
            "rejected": [], "account_id": None, "report_number": None,
        }
    account_id = next(
        (account for account, sid in account_to_scenario.items() if sid == scenario_id),
        None,
    )
    audit = _document_title(document_index, documents.audit_notes)
    report_number = None
    if documents.aup_report and documents.aup_report in document_index:
        report_number = document_index[documents.aup_report].report_number
    return {
        "account_id": account_id,
        "agreement": _document_title(document_index, documents.agreement),
        "kyc": _document_title(document_index, documents.kyc),
        "aup_report": _document_title(document_index, documents.aup_report),
        "audit_notes": audit,
        "report_number": report_number,
        "rejected": _rejected_documents(document_index, documents.rejected),
    }


def _txn_snapshot(
    ledger: pd.DataFrame,
    ledger_usd: pd.DataFrame,
    txn_id: str,
    role: str,
) -> dict[str, Any]:
    raw = ledger[ledger["txn_id"].astype(str) == txn_id]
    usd = ledger_usd[ledger_usd["txn_id"].astype(str) == txn_id]
    if raw.empty:
        return {"txn_id": txn_id, "role": role, "missing": True}
    row = raw.iloc[0]
    usd_amount = None
    if not usd.empty:
        u = usd.iloc[0]
        if pd.notna(u["amount"]):
            usd_amount = float(u["amount"])
    return {
        "txn_id": txn_id,
        "role": role,
        "date": str(row.get("date", "")),
        "counterparty": str(row.get("counterparty", "")),
        "description": str(row.get("description", "")),
        "amount_raw": float(row["amount"]) if pd.notna(row["amount"]) else None,
        "currency": str(row.get("currency", "")),
        "amount_usd": usd_amount,
    }


def _counterfactuals(
    spec,
    selected_roles: dict[str, list[str]],
    adjustments: list[dict[str, Any]],
    ledger: pd.DataFrame,
) -> list[dict[str, Any]]:
    candidates = evidence_candidates(selected_roles, adjustments, ledger)
    results: list[dict[str, Any]] = []
    for txn_id in candidates:
        kind = counterfactual_kind(txn_id, adjustments, ledger)
        if kind == "revert_adjustment":
            candidate_roles = selected_roles
            candidate_adjustments = reverted_adjustments(txn_id, adjustments, ledger)
        else:
            candidate_roles = {
                role: [candidate for candidate in ids if candidate != txn_id]
                for role, ids in selected_roles.items()
            }
            candidate_adjustments = adjustments
        try:
            values = build_clause_selection(spec, candidate_roles, ledger, candidate_adjustments)
            result = evaluate_covenant(spec, values)
            results.append({
                "txn_id": txn_id,
                "kind": kind,
                "new_status": result.status,
                "new_actual": float(result.actual.quantize(Decimal("0.01"))),
            })
        except Exception as error:
            results.append({"txn_id": txn_id, "kind": kind, "error": f"{type(error).__name__}: {error}"})
    return results


def _score_cell(
    scenario_id: str,
    clause_id: str,
    submission_cell: dict[str, Any],
    truth_cell: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if truth_cell is None:
        return None
    from scorer import score_cell
    try:
        detail = score_cell(scenario_id, clause_id, submission_cell, truth_cell)
        score = detail.total
    except Exception:
        score = None
    return {
        "status": truth_cell.get("status"),
        "actual": truth_cell.get("actual"),
        "evidence_txn_id": truth_cell.get("evidence_txn_id"),
        "score": score,
    }


def _build_cell(
    scenario_id: str,
    clause_id: str,
    submission_cell: dict[str, Any],
    truth_cell: dict[str, Any] | None,
    extraction: dict[str, Any] | None,
    selection: dict[str, Any] | None,
    document_index: dict,
    resolve_documents_fn,
    account_to_scenario: dict[str, str],
    ledger: pd.DataFrame,
) -> dict[str, Any]:
    truth = _score_cell(scenario_id, clause_id, submission_cell, truth_cell)
    result: dict[str, Any] = {
        "status": submission_cell.get("status"),
        "actual": submission_cell.get("actual"),
        "evidence_txn_id": submission_cell.get("evidence_txn_id"),
        "low_confidence": bool(submission_cell.get("low_confidence")),
        "truth": truth,
    }
    if extraction is None or selection is None:
        result["chain_missing"] = True
        return result

    documents = _document_chain(
        scenario_id, document_index, resolve_documents_fn, account_to_scenario
    )
    result["documents"] = documents

    covenant = next(
        (c for c in extraction.get("covenants", []) if str(c.get("clause_id")) == clause_id),
        None,
    )
    if covenant is None:
        result["chain_missing"] = True
        return result

    result["clause"] = {
        "operator": covenant.get("operator"),
        "threshold": covenant.get("threshold"),
        "role_descriptions": covenant.get("role_descriptions", {}),
        "applicability": covenant.get("applicability"),
        "exception": covenant.get("exception"),
        "quote": (covenant.get("source") or {}).get("quote", ""),
        "expression": covenant.get("value_expr"),
        "expression_source": _expression_source(covenant.get("value_expr")),
    }

    adjustments = extraction.get("adjustments", [])
    result["adjustments"] = [
        {
            "type": adj.get("type"),
            "match": adj.get("match", {}),
            "from_role": adj.get("from_role"),
            "to_role": adj.get("to_role"),
            "accepted": adj.get("accepted"),
            "sign": adj.get("sign"),
            "source_doc_id": (adj.get("source") or {}).get("doc_id"),
            "source_quote": (adj.get("source") or {}).get("quote"),
        }
        for adj in adjustments
    ]

    selected_roles = selection.get(clause_id, {}) if isinstance(selection.get(clause_id, {}), dict) else {}
    result["selection"] = selected_roles

    try:
        specs = specifications_from_extraction(scenario_id, extraction)
    except Exception as error:
        result["compute_error"] = f"{type(error).__name__}: {error}"
        return result
    spec = specs.get(clause_id)
    if spec is None:
        result["chain_missing"] = True
        return result

    ledger_usd = normalize_ledger_to_usd(ledger, adjustments)
    result["transactions"] = [
        _txn_snapshot(ledger, ledger_usd, txn_id, role)
        for role, txn_ids in selected_roles.items()
        for txn_id in txn_ids
    ]

    try:
        values = build_clause_selection(spec, selected_roles, ledger, adjustments)
        outcome = evaluate_covenant(spec, values)
    except Exception as error:
        result["compute_error"] = f"{type(error).__name__}: {error}"
        return result

    result["computed"] = {
        "actual": float(outcome.actual.quantize(Decimal("0.01"))),
        "status": outcome.status,
        "role_values": {
            role: [float(v) for v in vs]
            for role, vs in values.items()
        },
        "formula_symbolic": _expression_source(covenant.get("value_expr")),
        "formula_substituted": _expression_with_values(
            covenant.get("value_expr"), values, {k: Decimal(str(v)) for k, v in spec.facts.items()}
        ),
        "threshold": str(spec.threshold),
        "operator": spec.operator,
    }

    if outcome.status == "BREACH":
        result["counterfactuals"] = _counterfactuals(spec, selected_roles, adjustments, ledger)
    else:
        result["counterfactuals"] = []
    return result


def _cold_run_summary(metrics_text: str | None) -> dict[str, Any] | None:
    """Parse a handful of fields from cold_run_metrics.txt for the header stat strip."""
    if not metrics_text:
        return None
    fields = {}
    for line in metrics_text.splitlines():
        if "total" in line and ("197" in line or "мин" in line):
            fields["total_time"] = line.strip()
        elif "TOTAL LLM COST" in line:
            fields["total_cost"] = line.split(":", 1)[-1].strip()
        elif "average per-cell" in line:
            fields["average_per_cell"] = line.split(":", 1)[-1].strip()
        elif "status accuracy" in line:
            fields["status_accuracy"] = line.split(":", 1)[-1].strip()
    return fields or None


def build_view(
    *,
    workspace_dir: Path,
    data_dir: Path,
    template_path: Path,
    output_path: Path,
    ground_truth_path: Path | None = None,
) -> Path:
    submission_path = workspace_dir / "submission.json"
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    ledger = pd.read_csv(data_dir / "master_ledger_2025.csv")

    truth_scenarios: dict[str, Any] = {}
    if ground_truth_path is not None and ground_truth_path.exists():
        gt_root = json.loads(ground_truth_path.read_text(encoding="utf-8"))
        truth_scenarios = gt_root.get("scenarios", {})

    build_document_index, resolve_documents_fn, get_account_to_scenario = _document_index_and_resolvers()
    document_index = build_document_index()
    account_to_scenario = get_account_to_scenario()

    scenarios: list[str] = list(submission["answers"].keys())
    clauses_by_scenario: dict[str, list[str]] = {
        sid: list(clauses.keys()) for sid, clauses in submission["answers"].items()
    }
    all_clauses = sorted({cid for clauses in clauses_by_scenario.values() for cid in clauses})

    cells: dict[str, dict[str, dict[str, Any]]] = {}
    for scenario_id, clauses in submission["answers"].items():
        extraction = _load_json_or_none(workspace_dir / "extractions" / f"{scenario_id}.json")
        extraction_output = extraction["output"] if extraction else None
        selection = _load_json_or_none(workspace_dir / "selections" / f"{scenario_id}.json")
        selection_output = selection["output"] if selection else None
        truth_clauses = (truth_scenarios.get(scenario_id) or {}).get("covenants", {})

        cells[scenario_id] = {}
        for clause_id, cell in clauses.items():
            truth_cell = truth_clauses.get(clause_id) if truth_clauses else None
            cells[scenario_id][clause_id] = _build_cell(
                scenario_id=scenario_id,
                clause_id=clause_id,
                submission_cell=cell,
                truth_cell=truth_cell,
                extraction=extraction_output,
                selection=selection_output,
                document_index=document_index,
                resolve_documents_fn=resolve_documents_fn,
                account_to_scenario=account_to_scenario,
                ledger=ledger,
            )

    has_truth = bool(truth_scenarios)
    summary: dict[str, Any] = {"total_cells": sum(len(v) for v in clauses_by_scenario.values())}
    if has_truth:
        scored_cells = [
            cells[sid][cid]
            for sid in scenarios
            for cid in clauses_by_scenario[sid]
            if cells[sid][cid].get("truth") and cells[sid][cid]["truth"].get("score") is not None
        ]
        if scored_cells:
            total_score = sum(c["truth"]["score"] for c in scored_cells)
            n = len(scored_cells)
            correct = sum(
                1 for c in scored_cells
                if c.get("status") == c["truth"].get("status")
            )
            summary["mean_score"] = round(total_score / n, 4)
            summary["sum_score"] = round(total_score, 2)
            summary["status_accuracy_pct"] = round(100 * correct / n, 1)
            summary["status_correct"] = correct
    metrics_text = _cold_run_metrics(workspace_dir)
    summary["cold_run"] = _cold_run_summary(metrics_text)

    payload = {
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "workspace_dir": str(workspace_dir),
        "data_dir": str(data_dir),
        "has_ground_truth": has_truth,
        "scenarios": scenarios,
        "clauses": all_clauses,
        "clauses_by_scenario": clauses_by_scenario,
        "cells": cells,
        "summary": summary,
        "cold_run_metrics_text": metrics_text,
    }

    encoded = json.dumps(payload, ensure_ascii=False, indent=2, default=_decimal_to_native)
    html = template_path.read_text(encoding="utf-8").replace("__VIEW_DATA_PLACEHOLDER__", encoded)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(html, encoding="utf-8")
    return output_path
