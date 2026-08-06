"""Публичные границы будущего последовательного пайплайна."""

import json
import re
import time
from copy import deepcopy
from decimal import Decimal
from functools import lru_cache

import pandas as pd

from config import get_settings
from lib.pdf import extract_text


def _read_ledger() -> pd.DataFrame:
    """Читает сумму числом: пустые значения остаются pandas NaN."""
    return pd.read_csv(get_settings().data_dir / "master_ledger_2025.csv")


def run_pipeline(*, stop_after_stage: int | None = None):
    from stages.s0_baseline import run

    settings = get_settings()
    baseline = run(settings.data_dir, settings.workspace_dir)
    # Stages 1–12 are attached incrementally.  Until their orchestration is
    # ready, returning this persisted baseline keeps every run submit-safe.
    if stop_after_stage is not None and stop_after_stage <= 0:
        return baseline

    from stages.s0_baseline import _write_json_atomically
    from stages.s8_compute import build_clause_selection, evaluate_covenant, specifications_from_extraction
    from stages.s9_evidence import (
        counterfactual_kind,
        evidence_candidates,
        find_counterfactual_evidence,
        reverted_adjustments,
    )

    started = time.perf_counter()
    ledger = _read_ledger()
    template = json.loads((settings.data_dir / "submission_template.json").read_text(encoding="utf-8"))
    submission = deepcopy(template)

    from lib.consolidated_retrieval import patch_group_capex

    document_index = build_document_index()

    for scenario_id, clauses in submission["answers"].items():
        extraction_path = settings.workspace_dir / "extractions" / f"{scenario_id}.json"
        selection_path = settings.workspace_dir / "selections" / f"{scenario_id}.json"
        if not extraction_path.exists() or not selection_path.exists():
            continue
        extraction = json.loads(extraction_path.read_text(encoding="utf-8"))["output"]
        selection = json.loads(selection_path.read_text(encoding="utf-8"))["output"]
        documents = resolve_documents(scenario_id)
        if documents.kyc:
            patch_group_capex(extraction, document_index, document_index[documents.kyc].text)
        specs = specifications_from_extraction(scenario_id, extraction)
        adjustments = extraction.get("adjustments", [])

        for clause_id, cell in clauses.items():
            spec = specs.get(clause_id)
            selected_roles = selection.get(clause_id, {})
            if spec is None or not isinstance(selected_roles, dict):
                continue
            values = build_clause_selection(spec, selected_roles, ledger, adjustments)
            result = evaluate_covenant(spec, values)

            def recompute(txn_id: str):
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
                candidate_values = build_clause_selection(spec, candidate_roles, ledger, candidate_adjustments)
                return evaluate_covenant(spec, candidate_values)

            evidence = find_counterfactual_evidence(
                base_status=result.status,
                candidates=evidence_candidates(selected_roles, adjustments, ledger),
                recompute=recompute,
            )
            cell.update(
                status=result.status,
                actual=float(result.actual.quantize(Decimal("0.01"))),
                evidence_txn_id=evidence,
            )

    # Keep submission schema identical to the organizer template.  Timing is
    # diagnostic metadata, not a submission field.
    run_pipeline.last_timings = {"stage_8_9": time.perf_counter() - started}
    _write_json_atomically(settings.workspace_dir / "submission.json", submission)
    return submission


def gap_scan():
    ledger = _read_ledger()
    known_currencies = {"USD", "KZT", "EUR", "RUB", "GBP", "CNY"}
    return {
        "missing_amount": ledger.loc[ledger["amount"].isna(), "txn_id"].tolist(),
        "zero_amount": ledger.loc[ledger["amount"] == 0, "txn_id"].tolist(),
        "unknown_currency": ledger.loc[~ledger["currency"].isin(known_currencies), "txn_id"].tolist(),
    }


@lru_cache(maxsize=1)
def build_document_index():
    from stages.s2_pdf import run
    return run()


def read_document_text(doc_id: str):
    from stages.s2_pdf import decode_legacy_pdf_text
    text, _ = extract_text(get_settings().data_dir / "documents" / f"{doc_id}.pdf")
    return decode_legacy_pdf_text(text)


def build_mapping(ledger: pd.DataFrame, *, template: dict) -> dict[str, str]:
    """Map accounts to the scenario IDs declared by the current template."""
    targets = set(template["answers"])
    scoped = ledger.copy()
    scoped["scenario_id"] = scoped["txn_id"].str.extract(r"^TXN-([^-]+)-", expand=False)
    scoped = scoped[scoped["scenario_id"].isin(targets)]
    if scoped.empty and targets:
        raise ValueError("no ledger transactions match scenario IDs in submission template")
    grouped = scoped.groupby("account_id")["scenario_id"].agg(lambda ids: sorted(set(ids)))
    conflicts = {account: ids for account, ids in grouped.items() if len(ids) != 1}
    if conflicts:
        raise ValueError(f"account_id maps to multiple scenario_id values: {conflicts}")
    return {account: scenarios[0] for account, scenarios in grouped.items()}


def get_account_to_scenario():
    template_path = get_settings().data_dir / "submission_template.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    return build_mapping(_read_ledger(), template=template)


def parse_transaction_correction(text: str, txn_id: str) -> tuple[Decimal, str] | None:
    """Return the closest signed dollar amount associated with a transaction."""
    position = text.find(txn_id)
    if position < 0:
        return None
    start = max(0, position - 500)
    window = text[start : position + 500]
    candidates: list[tuple[int, Decimal, str]] = []
    for money in re.finditer(r"\$([\d,]+\.\d{2})", window):
        sign_context = window[max(0, money.start() - 120) : money.end() + 120].lower()
        if "\u0440\u0430\u0441\u0445\u043e\u0434" in sign_context:
            sign = "expense"
        elif "\u043f\u043e\u0441\u0442\u0443\u043f\u043b\u0435\u043d" in sign_context:
            sign = "income"
        else:
            continue
        absolute_position = start + money.start()
        candidates.append(
            (abs(absolute_position - position), Decimal(money.group(1).replace(",", "")), sign)
        )
    if not candidates:
        return None
    _, amount, sign = min(candidates, key=lambda candidate: candidate[0])
    return amount, sign


def find_counterfactual_evidence(*, base_status, candidates, recompute):
    from stages.s9_evidence import find_counterfactual_evidence as resolve

    return resolve(base_status=base_status, candidates=candidates, recompute=recompute)


def evaluate_covenant(spec, selection):
    from stages.s8_compute import evaluate_covenant as evaluate

    return evaluate(spec, selection)


def get_corrections():
    from models import Correction, SourceRef

    result = {}
    for doc in build_document_index().values():
        for txn in doc.mentioned_txn_ids:
            parsed = parse_transaction_correction(doc.text, txn)
            if parsed is None:
                continue
            amount, sign = parsed
            result[txn] = Correction(
                correct_amount=amount,
                source=SourceRef(doc.doc_id),
                sign=sign,
            )
    return result


def resolve_documents(scenario_id: str):
    from models import DocumentSet
    account = next(account for account, scenario in get_account_to_scenario().items() if scenario == scenario_id)
    documents = list(build_document_index().values())
    own = [document for document in documents if account in document.account_ids]
    active = [document for document in own if document.version_status == "active"]
    agreements = [document for document in active if document.doc_type == "agreement"]
    kyc = [document for document in active if document.doc_type == "kyc"]
    report_numbers = {reference for document in own for reference in document.references}
    reports = [
        document
        for document in documents
        if document.doc_type == "aup"
        and document.version_status == "active"
        and document.report_number in report_numbers
    ]
    report_ids = {document.report_number for document in reports}
    audit_notes = [
        document
        for document in own
        if document.version_status == "active" and document.doc_type == "audit_notes"
    ] + [
        document
        for document in own
        if document.version_status == "active"
        and document.doc_type == "noise"
        and any(reference in report_ids for reference in document.references)
    ]
    return DocumentSet(
        agreement=agreements[0].doc_id if agreements else None,
        kyc=kyc[0].doc_id if kyc else None,
        aup_report=reports[0].doc_id if reports else None,
        audit_notes=audit_notes[0].doc_id if audit_notes else None,
        rejected=[document.doc_id for document in own if document.version_status != "active"],
    )


from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioClauseResult:
    actual: Decimal
    status: str
    evidence_txn_id: str | None


def _load_extraction(scenario_id: str) -> dict:
    path = get_settings().workspace_dir / "extractions" / f"{scenario_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))["output"]


def _load_selection(scenario_id: str) -> dict:
    path = get_settings().workspace_dir / "selections" / f"{scenario_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))["output"]


@lru_cache(maxsize=None)
def get_specifications():
    from stages.s8_compute import specifications_from_extraction

    result = {}
    template = json.loads((get_settings().data_dir / "submission_template.json").read_text(encoding="utf-8"))
    for scenario_id in template["answers"]:
        try:
            extraction = _load_extraction(scenario_id)
        except FileNotFoundError:
            continue
        result[scenario_id] = specifications_from_extraction(scenario_id, extraction)
    return result


def related_party_threshold(scenario_id: str) -> Decimal:
    extraction = _load_extraction(scenario_id)
    return Decimal(str(extraction["related_parties"]["threshold_percent"]))


def related_parties(scenario_id: str) -> list[str]:
    from stages.s7_select import filtered_related_parties

    extraction = _load_extraction(scenario_id)
    filtered = filtered_related_parties(extraction)
    return [entity["name"] for entity in filtered.get("entities", [])]


def _scenario_results(scenario_id: str) -> dict[str, ScenarioClauseResult]:
    from stages.s8_compute import build_clause_selection, evaluate_covenant, specifications_from_extraction
    from stages.s9_evidence import (
        counterfactual_kind,
        evidence_candidates,
        find_counterfactual_evidence as resolve_evidence,
        reverted_adjustments,
    )

    extraction = _load_extraction(scenario_id)
    selection = _load_selection(scenario_id)
    specs = specifications_from_extraction(scenario_id, extraction)
    adjustments = extraction.get("adjustments", [])
    ledger = _read_ledger()

    results: dict[str, ScenarioClauseResult] = {}
    for clause_id, spec in specs.items():
        selected_roles = selection.get(clause_id, {})
        if not isinstance(selected_roles, dict):
            continue
        values = build_clause_selection(spec, selected_roles, ledger, adjustments)
        result = evaluate_covenant(spec, values)

        def recompute(txn_id: str, _spec=spec, _roles=selected_roles):
            kind = counterfactual_kind(txn_id, adjustments, ledger)
            if kind == "revert_adjustment":
                candidate_roles = _roles
                candidate_adjustments = reverted_adjustments(txn_id, adjustments, ledger)
            else:
                candidate_roles = {
                    role: [candidate for candidate in ids if candidate != txn_id]
                    for role, ids in _roles.items()
                }
                candidate_adjustments = adjustments
            candidate_values = build_clause_selection(_spec, candidate_roles, ledger, candidate_adjustments)
            return evaluate_covenant(_spec, candidate_values)

        evidence = resolve_evidence(
            base_status=result.status,
            candidates=evidence_candidates(selected_roles, adjustments, ledger),
            recompute=recompute,
        )
        results[clause_id] = ScenarioClauseResult(
            actual=result.actual.quantize(Decimal("0.01")),
            status=result.status,
            evidence_txn_id=evidence,
        )
    return results


def run_scenario(scenario_id: str) -> dict[str, ScenarioClauseResult]:
    return _scenario_results(scenario_id)


def recompute_without(scenario_id: str, clause_id: str, txn_id: str):
    from stages.s8_compute import build_clause_selection, evaluate_covenant, specifications_from_extraction
    from stages.s9_evidence import counterfactual_kind, reverted_adjustments

    extraction = _load_extraction(scenario_id)
    selection = _load_selection(scenario_id)
    specs = specifications_from_extraction(scenario_id, extraction)
    spec = specs[clause_id]
    selected_roles = selection.get(clause_id, {})
    adjustments = extraction.get("adjustments", [])
    ledger = _read_ledger()

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
    values = build_clause_selection(spec, candidate_roles, ledger, candidate_adjustments)
    return evaluate_covenant(spec, values)
