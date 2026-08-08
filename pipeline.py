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


def scenario_membership_mask(ledger: pd.DataFrame, scenario_id: str) -> pd.Series:
    """Boolean mask selecting ledger rows that belong to a scenario.

    The scenario key from the submission template is treated as a
    delimited token — bounded by non-alphanumeric characters or the
    ends of the string — inside the txn_id. This tolerates transaction
    formats other than ``TXN-{scenario}-{n}`` (e.g. ``TX_P1_0001`` or
    ``TXN-2025-P1-0001``); the only convention we require is that the
    scenario key appears somewhere as its own token.
    """
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(scenario_id)}(?![A-Za-z0-9])")
    return ledger["txn_id"].astype(str).str.contains(pattern, regex=True)


def scenario_txn_ids(ledger: pd.DataFrame, scenario_id: str) -> set[str]:
    """Set of transaction IDs the ledger attributes to a scenario."""
    return set(ledger.loc[scenario_membership_mask(ledger, scenario_id), "txn_id"].astype(str))


def run_pipeline(*, stop_after_stage: int | None = None):
    from stages.s0_baseline import run

    settings = get_settings()
    baseline = run(settings.data_dir, settings.workspace_dir)
    # Stages 1–12 are attached incrementally.  Until their orchestration is
    # ready, returning this persisted baseline keeps every run submit-safe.
    if stop_after_stage is not None and stop_after_stage <= 0:
        return baseline

    from stages.s0_baseline import _write_json_atomically
    from stages.s6_extract import extract_scenario
    from stages.s7_select import select_scenario
    from stages.s8_compute import (
        UnsupportedSpecError,
        _safe_positive_decimal,
        build_clause_selection,
        evaluate_covenant,
        specifications_from_extraction,
    )
    from stages.s9_evidence import (
        counterfactual_kind,
        evidence_candidates,
        find_counterfactual_evidence,
        reverted_adjustments,
    )
    from lib.anthropic_client import AnthropicClient

    total_started = time.perf_counter()
    ledger = _read_ledger()
    # baseline was written to disk by stage 0 with COMPLIANT/0.01/null in every
    # cell; use it as the working submission so any unfilled cell keeps that
    # safety-net value instead of reverting to the template's nulls.
    submission = deepcopy(baseline)

    from lib.consolidated_retrieval import patch_group_capex

    stage_2_started = time.perf_counter()
    document_index = build_document_index()
    # Level-2 doc-type classification: fires only for docs the regex triage
    # returned as ``noise`` yet still carry an ``ACC-XXXX`` marker. Cached in
    # workspace/doctypes/; no-op when ANTHROPIC_API_KEY is empty. The
    # candidate count is capped by lib.doc_classify.MAX_LLM_CANDIDATES to
    # protect against a whole-corpus encoding failure on a private set.
    from lib.doc_classify import apply_llm_fallback
    apply_llm_fallback(document_index, settings)
    timings: dict[str, float] = {"stage_2_pdf": time.perf_counter() - stage_2_started}
    usage_totals: dict[str, dict[str, int]] = {
        "stage_6_extract": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
        "stage_7_select": {"input_tokens": 0, "output_tokens": 0, "calls": 0},
    }

    def _extra_documents(scenario_id: str, documents_) -> list[str]:
        """Noise docs mentioning this scenario's missing-amount transactions."""
        own_txn_ids = scenario_txn_ids(ledger, scenario_id)
        missing_series = ledger.loc[ledger["amount"].isna(), "txn_id"].astype(str)
        scenario_missing = [txn for txn in missing_series if txn in own_txn_ids]
        if not scenario_missing:
            return []
        own_ids = {documents_.agreement, documents_.kyc, documents_.aup_report, documents_.audit_notes} - {None}
        hits: list[str] = []
        for doc_id, record in document_index.items():
            if doc_id in own_ids:
                continue
            if any(txn in record.mentioned_txn_ids for txn in scenario_missing):
                hits.append(doc_id)
        return hits

    errors_path = settings.workspace_dir / "errors.log"
    errors_path.parent.mkdir(parents=True, exist_ok=True)

    def log_error(scope: str, error: Exception) -> None:
        with errors_path.open("a", encoding="utf-8") as log:
            log.write(f"{scope}: {type(error).__name__}: {error}\n")

    def cell_is_valid(status, actual_decimal, evidence, ledger_frame) -> bool:
        """Hard invariants; on failure the baseline cell is kept."""
        if status not in ("COMPLIANT", "BREACH"):
            return False
        try:
            actual_value = Decimal(str(actual_decimal))
        except Exception:
            return False
        if not actual_value.is_finite() or actual_value <= 0:
            return False
        if evidence is not None:
            if not isinstance(evidence, str) or "-" not in evidence:
                return False
            if evidence not in set(ledger_frame["txn_id"].astype(str)):
                return False
        return True

    for scenario_id, clauses in submission["answers"].items():
        extraction_path = settings.workspace_dir / "extractions" / f"{scenario_id}.json"
        selection_path = settings.workspace_dir / "selections" / f"{scenario_id}.json"
        if not extraction_path.exists() or not selection_path.exists():
            if not settings.anthropic_api_key:
                log_error(
                    f"{scenario_id} extract/select",
                    RuntimeError("ANTHROPIC_API_KEY missing; scenario stays at baseline"),
                )
                continue
            try:
                documents = resolve_documents(scenario_id)
                document_texts = {doc_id: record.text for doc_id, record in document_index.items()}
                if not extraction_path.exists():
                    extract_client = AnthropicClient(
                        api_key=settings.anthropic_api_key,
                        model=settings.extract_model,
                        timeout_seconds=settings.llm_timeout_seconds,
                    )
                    started_extract = time.perf_counter()
                    extract_result = extract_scenario(
                        scenario_id=scenario_id,
                        documents=documents,
                        document_texts=document_texts,
                        clause_ids=list(clauses.keys()),
                        client=extract_client,
                        cache_dir=settings.workspace_dir / "extractions",
                        extra_documents=_extra_documents(scenario_id, documents),
                    )
                    timings["stage_6_extract"] = timings.get("stage_6_extract", 0.0) + (time.perf_counter() - started_extract)
                    usage_totals["stage_6_extract"]["input_tokens"] += extract_result.usage.get("input_tokens", 0)
                    usage_totals["stage_6_extract"]["output_tokens"] += extract_result.usage.get("output_tokens", 0)
                    usage_totals["stage_6_extract"]["calls"] += 1
                if not selection_path.exists():
                    select_client = AnthropicClient(
                        api_key=settings.anthropic_api_key,
                        model=settings.select_model,
                        timeout_seconds=settings.llm_timeout_seconds,
                    )
                    extraction_for_select = json.loads(extraction_path.read_text(encoding="utf-8"))["output"]
                    started_select = time.perf_counter()
                    select_result = select_scenario(
                        scenario_id=scenario_id,
                        extraction=extraction_for_select,
                        ledger=ledger,
                        client=select_client,
                        cache_dir=settings.workspace_dir / "selections",
                        select_mode=settings.select_mode,
                    )
                    timings["stage_7_select"] = timings.get("stage_7_select", 0.0) + (time.perf_counter() - started_select)
                    usage_totals["stage_7_select"]["input_tokens"] += select_result.usage.get("input_tokens", 0)
                    usage_totals["stage_7_select"]["output_tokens"] += select_result.usage.get("output_tokens", 0)
                    usage_totals["stage_7_select"]["calls"] += 1
            except Exception as error:
                log_error(f"{scenario_id} extract/select", error)
                continue
        if not extraction_path.exists() or not selection_path.exists():
            continue
        started_compute = time.perf_counter()
        started_compute_setup = time.perf_counter()
        unsupported_clauses: dict[str, dict[str, Any]] = {}
        try:
            extraction = json.loads(extraction_path.read_text(encoding="utf-8"))["output"]
            selection = json.loads(selection_path.read_text(encoding="utf-8"))["output"]
            documents = resolve_documents(scenario_id)
            if documents.kyc:
                patch_group_capex(extraction, document_index, document_index[documents.kyc].text)
            specs = specifications_from_extraction(
                scenario_id, extraction, unsupported_out=unsupported_clauses
            )
            adjustments = extraction.get("adjustments", [])
        except Exception as error:
            log_error(f"{scenario_id} setup", error)
            continue
        timings["stage_8_9"] = timings.get("stage_8_9", 0.0) + (time.perf_counter() - started_compute_setup)

        for clause_id, cell in clauses.items():
            started_compute = time.perf_counter()
            if clause_id in unsupported_clauses:
                _apply_unsupported_fallback(
                    cell,
                    scenario_id=scenario_id,
                    clause_id=clause_id,
                    info=unsupported_clauses[clause_id],
                    log_error=log_error,
                )
                _write_json_atomically(settings.workspace_dir / "submission.json", submission)
                timings["stage_8_9"] = timings.get("stage_8_9", 0.0) + (time.perf_counter() - started_compute)
                continue
            try:
                spec = specs.get(clause_id)
                selected_roles = selection.get(clause_id, {})
                if spec is None or not isinstance(selected_roles, dict):
                    continue
                try:
                    values = build_clause_selection(spec, selected_roles, ledger, adjustments)
                    result = evaluate_covenant(spec, values)
                except UnsupportedSpecError as unsupported_error:
                    _apply_unsupported_fallback(
                        cell,
                        scenario_id=scenario_id,
                        clause_id=clause_id,
                        info={
                            "threshold": _safe_positive_decimal(spec.threshold),
                            "reason": str(unsupported_error),
                        },
                        log_error=log_error,
                    )
                    _write_json_atomically(settings.workspace_dir / "submission.json", submission)
                    continue

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

                evidence = find_counterfactual_evidence(
                    base_status=result.status,
                    candidates=evidence_candidates(selected_roles, adjustments, ledger),
                    recompute=recompute,
                    trace_scope=f"{scenario_id} {clause_id}",
                )
                actual_quantized = float(result.actual.quantize(Decimal("0.01")))
                if not cell_is_valid(result.status, actual_quantized, evidence, ledger):
                    log_error(f"{scenario_id} {clause_id} invariants", ValueError(f"status={result.status} actual={actual_quantized}"))
                    continue
                cell.update(
                    status=result.status,
                    actual=actual_quantized,
                    evidence_txn_id=evidence,
                )
                _write_json_atomically(settings.workspace_dir / "submission.json", submission)
            except Exception as error:
                log_error(f"{scenario_id} {clause_id}", error)
                continue
            finally:
                timings["stage_8_9"] = timings.get("stage_8_9", 0.0) + (time.perf_counter() - started_compute)

    def _stage_cost(stage: str, pricing) -> Decimal:
        if pricing is None:
            return Decimal("0")
        totals = usage_totals[stage]
        input_cost = pricing.input_usd_per_mtok * Decimal(totals["input_tokens"]) / Decimal(1_000_000)
        output_cost = pricing.output_usd_per_mtok * Decimal(totals["output_tokens"]) / Decimal(1_000_000)
        return (input_cost + output_cost).quantize(Decimal("0.000001"))

    timings["total"] = time.perf_counter() - total_started
    run_pipeline.last_health = _compute_health(submission, settings.workspace_dir)
    run_pipeline.last_timings = timings
    run_pipeline.last_usage = usage_totals
    run_pipeline.last_cost_usd = {
        "stage_6_extract": _stage_cost("stage_6_extract", settings.extract_pricing),
        "stage_7_select": _stage_cost("stage_7_select", settings.select_pricing),
    }
    _write_json_atomically(settings.workspace_dir / "submission.json", submission)
    return submission


def _apply_unsupported_fallback(
    cell: dict,
    *,
    scenario_id: str,
    clause_id: str,
    info: dict,
    log_error,
) -> None:
    """Localised fallback for a single unsupported/missing-input cell.

    The other cells of the scenario continue with the normal compute
    path. The submission schema is not extended (validator forbids
    extra keys); low_confidence stays in the trace only.
    """
    threshold = info.get("threshold")
    if isinstance(threshold, Decimal) and threshold.is_finite() and threshold > 0:
        actual_value = float(threshold.quantize(Decimal("0.01")))
    else:
        actual_value = 0.01
    cell.update(status="COMPLIANT", actual=actual_value, evidence_txn_id=None)
    log_error(
        f"{scenario_id} {clause_id} unsupported",
        RuntimeError(f"low_confidence fallback (actual={actual_value}): {info.get('reason', 'unspecified')}"),
    )


def _compute_health(submission: dict, workspace_dir) -> dict[str, int]:
    """Post-hoc health signals from the final submission + workspace state.

    Ground truth is not available on the private-set day, so we pick
    between runs by these numbers alone:
      baseline_cells       — how many cells fell through to stage-0 default
      computed_cells       — total - baseline
      valid_selections     — scenarios with a saved stage-7 output on disk
      rejected_scenarios   — scenarios whose stage-7 output landed only in
                             selections/rejected/ (both retry attempts failed)
      salvaged_clauses     — soft_warnings across selections that dropped a
                             specific clause via clause-level salvage
      retried_selections   — soft_warnings mentioning "accepted on retry"
      soft_warnings        — total soft_warnings across all saved selections
    """
    baseline_cell = {"status": "COMPLIANT", "actual": 0.01, "evidence_txn_id": None}
    baseline_cells = 0
    total_cells = 0
    for clauses in submission["answers"].values():
        for cell in clauses.values():
            total_cells += 1
            snapshot = {
                "status": cell.get("status"),
                "actual": cell.get("actual"),
                "evidence_txn_id": cell.get("evidence_txn_id"),
            }
            if snapshot == baseline_cell:
                baseline_cells += 1

    selections_dir = workspace_dir / "selections"
    rejected_dir = selections_dir / "rejected"
    valid_selections = 0
    salvaged_clauses = 0
    retried_selections = 0
    soft_warnings_total = 0
    if selections_dir.exists():
        for path in selections_dir.glob("*.json"):
            valid_selections += 1
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            warnings = payload.get("soft_warnings", []) or []
            soft_warnings_total += len(warnings)
            salvaged_clauses += sum(1 for w in warnings if "stage7 salvage" in w)
            if any("accepted on retry" in w for w in warnings):
                retried_selections += 1
    rejected_scenarios = len(list(rejected_dir.glob("*.json"))) if rejected_dir.exists() else 0

    return {
        "total_cells": total_cells,
        "baseline_cells": baseline_cells,
        "computed_cells": total_cells - baseline_cells,
        "valid_selections": valid_selections,
        "rejected_scenarios": rejected_scenarios,
        "salvaged_clauses": salvaged_clauses,
        "retried_selections": retried_selections,
        "soft_warnings": soft_warnings_total,
    }


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
    """Map account_id → scenario_id using the template's own keys as tokens.

    For each scenario key from ``submission_template.json`` we look for
    transactions whose ``txn_id`` contains that key as a delimited token
    (``scenario_membership_mask``). All such transactions must share a
    single ``account_id``; the pair goes into the returned mapping.

    This replaces the previous ``^TXN-([^-]+)-`` extraction, which broke
    on any transaction format other than ``TXN-{scenario}-{n}``. The
    scenario keys themselves are still required to appear as tokens in
    the txn_id — that is the sole convention needed for a private
    dataset to work without code changes.
    """
    targets = list(template["answers"])
    mapping: dict[str, str] = {}
    for scenario_id in targets:
        mask = scenario_membership_mask(ledger, scenario_id)
        accounts = ledger.loc[mask, "account_id"].astype(str).unique().tolist()
        if not accounts:
            continue
        if len(accounts) > 1:
            raise ValueError(
                f"scenario {scenario_id} matches multiple account_ids: {accounts}"
            )
        account = accounts[0]
        if account in mapping:
            raise ValueError(
                f"account {account} is claimed by scenarios {mapping[account]!r} and {scenario_id!r}"
            )
        mapping[account] = scenario_id
    if not mapping and targets:
        raise ValueError("no ledger transactions match any scenario key from submission template")
    return mapping


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
