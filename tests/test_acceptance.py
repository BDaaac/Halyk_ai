"""Приёмочные контракты публичного набора.

Литералы этого файла допустимы только здесь: production-код должен читать
структуру задачи из submission_template.json текущего DATA_DIR.
"""

import json
from decimal import Decimal
from pathlib import Path

import pandas as pd
import pytest
import models
import pipeline

from lib.expressions import ExpressionTooDeep, add, divide, evaluate, max_, subtract, sum_
from config import get_settings
from lib.text import normalize_identifiers
from pipeline import (
    build_document_index,
    gap_scan,
    get_account_to_scenario,
    get_corrections,
    get_specifications,
    read_document_text,
    recompute_without,
    related_parties,
    related_party_threshold,
    resolve_documents,
    run_pipeline,
    run_scenario,
)


FIXTURES = Path(__file__).parent / "fixtures"


def test_gap_scan_finds_missing_amounts():
    assert gap_scan()["missing_amount"] == ["TXN-P7-0033", "TXN-P8-0031"]


def test_scanned_pdf_detected():
    assert build_document_index()["f3fa6d20c8a1"].extraction_method == "vision"


def test_no_document_deleted():
    assert len(build_document_index()) == 200


def test_b1_document_set():
    documents = resolve_documents("B1")
    assert documents.agreement == "b38facf69a94"
    assert documents.kyc == "c66b5b638410"
    assert documents.audit_notes == "a16d7a619f87"
    assert documents.aup_report == "448b59e12768"
    assert "b084add0e80a" in documents.rejected
    assert "2762d1c605e6" in documents.rejected


def test_p10_audit_notes_are_available_without_an_aup_report_number():
    documents = resolve_documents("P10")
    assert documents.audit_notes == "4db1d630055f"


def test_letter_spaced_ids_normalized():
    assert "ACC-7201" in normalize_identifiers(read_document_text("a16d7a619f87"))


def test_normalization_preserves_prose():
    source = "Заёмщик не вправе отчуждать основные средства и иные активы"
    assert normalize_identifiers(source) == source


def test_account_scenario_mapping():
    mapping = get_account_to_scenario()
    assert mapping["ACC-7201"] == "B1"
    assert mapping["ACC-7801"] == "P1"


def test_scenario_filter_uses_template_not_prefix():
    """Сценарии приватного набора могут не начинаться с B или P."""
    ledger = pd.DataFrame(
        {
            "txn_id": ["TXN-T1-0001", "TXN-ZZ9-0001", "TXN-9001-0001"],
            "account_id": ["ACC-T1", "ACC-ZZ9", "ACC-9001"],
        }
    )
    template = {"answers": {"T1": {}, "ZZ9": {}}}
    build_mapping = getattr(pipeline, "build_mapping", None)
    assert build_mapping is not None, "mapping must filter by template scenario keys"
    mapping = build_mapping(ledger, template=template)
    assert set(mapping.values()) == {"T1", "ZZ9"}


def test_correction_parser_keeps_sign_without_word_order():
    """Сумма и идентификатор могут стоять в любом порядке в примечании."""
    parser = getattr(pipeline, "parse_transaction_correction", None)
    assert parser is not None, "stage 5 must parse an amount and its sign together"
    before_id = parser(
        "\u0424\u0430\u043a\u0442\u0438\u0447\u0435\u0441\u043a\u0430\u044f \u0441\u0443\u043c\u043c\u0430 $10.00 (\u0440\u0430\u0441\u0445\u043e\u0434) \u043f\u043e TXN-T1-0001",
        "TXN-T1-0001",
    )
    after_id = parser(
        "TXN-ZZ9-0001: $20.00 (\u043f\u043e\u0441\u0442\u0443\u043f\u043b\u0435\u043d\u0438\u0435)",
        "TXN-ZZ9-0001",
    )
    assert before_id == (Decimal("10.00"), "expense")
    assert after_id == (Decimal("20.00"), "income")


def test_targeted_retrieval_closes_gaps():
    corrections = get_corrections()
    assert corrections["TXN-P7-0033"].correct_amount == Decimal("486204.19")
    assert corrections["TXN-P7-0033"].source.doc_id == "26acfab1e58b"
    assert corrections["TXN-P8-0031"].correct_amount == Decimal("884204.16")
    assert corrections["TXN-P8-0031"].source.doc_id == "bf9b7bffc514"


def test_b1_icr():
    expression = divide(
        subtract(sum_([Decimal("9741934.78")]), sum_([Decimal("6166592.66")])),
        sum_([Decimal("1540833.29"), Decimal("592296.10")]),
    )
    assert round(evaluate(expression), 2) == Decimal("1.68")


def test_p10_nested_max():
    revenue = [Decimal("9000000.00")]
    payroll = [Decimal("2999236.37")]
    taxes = [Decimal("1200000.00")]
    expression = subtract(sum_(revenue), max_(sum_(payroll), sum_(taxes)))
    assert round(evaluate(expression), 2) == Decimal("6000763.63")


def _nested_expr(depth: int):
    expression = sum_([Decimal("1")])
    for _ in range(depth):
        expression = add(expression, sum_([Decimal("1")]))
    return expression


def test_expression_depth_limit():
    with pytest.raises(ExpressionTooDeep):
        evaluate(_nested_expr(depth=5))


def test_evidence_returns_flipper_and_falls_back_deterministically(tmp_path, monkeypatch):
    """Contract: single flipper wins; multiple flippers fall back to the
    lexicographically-smallest txn_id (scoring fallback, logged); zero
    flippers still yields None."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    finder = getattr(pipeline, "find_counterfactual_evidence", None)
    assert finder is not None, "stage 9 must resolve evidence deterministically"
    statuses = {"TXN-B1-0020": "COMPLIANT", "TXN-B1-0023": "BREACH"}
    assert finder(
        base_status="BREACH",
        candidates=list(statuses),
        recompute=lambda txn_id: statuses[txn_id],
    ) == "TXN-B1-0020"
    assert finder(
        base_status="COMPLIANT",
        candidates=list(statuses),
        recompute=lambda txn_id: statuses[txn_id],
    ) is None
    # Two flippers → deterministic first (lex-min), previously was None.
    assert finder(
        base_status="BREACH",
        candidates=["TXN-B1-0040", "TXN-B1-0020"],
        recompute=lambda _: "COMPLIANT",
    ) == "TXN-B1-0020"


def _phase3_contract_types():
    expr = getattr(models, "Expr", None)
    condition = getattr(models, "Condition", None)
    specification = getattr(models, "CovenantSpec", None)
    assert expr and condition and specification, "stage 8 needs structured covenant contracts"
    return expr, condition, specification


def test_p3_springing_condition_has_structured_threshold():
    Expr, Condition, CovenantSpec = _phase3_contract_types()
    specification = CovenantSpec(
        scenario_id="P3",
        clause_id="6.1",
        value_expr=Expr(op="sum", role="debt_service"),
        operator=">=",
        threshold=Decimal("1.70"),
        applicability=Condition(
            expr=Expr(op="sum", role="financing_proceeds"),
            operator=">",
            threshold=Decimal("4000000.00"),
            raw_text="applies only when financing proceeds exceed $4,000,000.00",
        ),
    )
    assert specification.applicability is not None
    assert specification.applicability.threshold == Decimal("4000000.00")


def test_p7_exception_has_structured_condition():
    Expr, Condition, CovenantSpec = _phase3_contract_types()
    specification = CovenantSpec(
        scenario_id="P7",
        clause_id="6.3",
        value_expr=Expr(op="sum", role="restricted_payment"),
        operator="<=",
        threshold=Decimal("500000.00"),
        exception=Condition(
            expr=Expr(op="sum", role="creditor_approved_payment"),
            operator=">",
            threshold=Decimal("0"),
            raw_text="except transactions approved by the creditor in writing",
        ),
    )
    assert specification.exception is not None
    assert specification.exception.expr.role == "creditor_approved_payment"


def test_non_applicable_covenant_is_compliant_with_real_actual():
    Expr, Condition, CovenantSpec = _phase3_contract_types()
    evaluator = getattr(pipeline, "evaluate_covenant", None)
    assert evaluator is not None, "stage 8 must evaluate applicability separately from actual"
    specification = CovenantSpec(
        scenario_id="T1",
        clause_id="6.1",
        value_expr=Expr(op="sum", role="restricted_payment"),
        operator="<=",
        threshold=Decimal("100.00"),
        applicability=Condition(
            expr=Expr(op="sum", role="financing_proceeds"),
            operator=">",
            threshold=Decimal("4000000.00"),
            raw_text="springing condition",
        ),
    )
    result = evaluator(
        specification,
        {"restricted_payment": [Decimal("150.00")], "financing_proceeds": [Decimal("0")]},
    )
    assert result.status == "COMPLIANT"
    assert result.actual == Decimal("150.00")
    assert result.actual > specification.threshold


def test_b1_all_three():
    result = run_scenario("B1")
    assert result["6.1"].status == "BREACH"
    assert result["6.1"].actual == Decimal("1.68")
    assert result["6.1"].evidence_txn_id == "TXN-B1-0020"
    assert result["6.2"].status == "COMPLIANT"
    assert result["6.2"].actual == Decimal("1284663.42")
    assert result["6.3"].status == "COMPLIANT"
    assert result["6.3"].actual == Decimal("307018.08")


def test_b1_counterfactual():
    assert recompute_without("B1", "6.1", "TXN-B1-0020").status == "COMPLIANT"


def test_p6_related_party_threshold_is_forty_percent():
    assert related_party_threshold("P6") == Decimal("40.0")
    assert "Taraz Kiln Services LLP" not in related_parties("P6")


def test_p3_springing_condition_parsed():
    spec = get_specifications()["P3"]["6.1"]
    assert spec.applicability is not None
    assert spec.applicability.threshold == Decimal("4000000.00")


def test_baseline_written_first():
    assert run_pipeline(stop_after_stage=0)["answers"]


def test_stage_zero_writes_atomic_valid_baseline(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    submission = run_pipeline(stop_after_stage=0)
    on_disk = json.loads((tmp_path / "submission.json").read_text(encoding="utf-8"))
    template = json.loads((get_settings().data_dir / "submission_template.json").read_text(encoding="utf-8"))
    assert on_disk == submission
    assert set(on_disk["answers"]) == set(template["answers"])
    assert all(
        cell == {"status": "COMPLIANT", "actual": 0.01, "evidence_txn_id": None}
        for clauses in on_disk["answers"].values()
        for cell in clauses.values()
    )


def test_all_cells_present():
    submission = run_pipeline()
    template = json.loads((FIXTURES / "submission_template.json").read_text(encoding="utf-8"))
    assert set(submission["answers"]) == set(template["answers"])


def test_evidence_belongs_to_scenario():
    for scenario_id, clauses in run_pipeline()["answers"].items():
        for cell in clauses.values():
            if cell["evidence_txn_id"]:
                assert cell["evidence_txn_id"].split("-")[1] == scenario_id


def test_pipeline_survives_missing_extraction(monkeypatch):
    """A single missing cache file must not break the whole submission."""
    from config import get_settings

    # Force the LLM path off so this test never hits the live API when the
    # orchestrator now auto-fills missing cache via extract_scenario.
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    settings = get_settings()
    extraction = settings.workspace_dir / "extractions" / "B1.json"
    backup = extraction.with_suffix(".json.backup")
    extraction.rename(backup)
    try:
        submission = run_pipeline()
    finally:
        if extraction.exists():
            extraction.unlink()
        backup.rename(extraction)
    template = json.loads((FIXTURES / "submission_template.json").read_text(encoding="utf-8"))
    assert set(submission["answers"]) == set(template["answers"])
    for clauses in submission["answers"].values():
        for cell in clauses.values():
            assert "status" in cell
            assert "actual" in cell


def test_failed_scenario_keeps_baseline_not_nulls(monkeypatch):
    """A scenario whose extraction fails must keep stage-0 baseline values."""
    from config import get_settings

    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    settings = get_settings()
    extraction = settings.workspace_dir / "extractions" / "B1.json"
    selection = settings.workspace_dir / "selections" / "B1.json"
    extraction_backup = extraction.with_suffix(".json.backup2")
    selection_backup = selection.with_suffix(".json.backup2")
    extraction.rename(extraction_backup)
    selection.rename(selection_backup)
    try:
        submission = run_pipeline()
    finally:
        if extraction.exists():
            extraction.unlink()
        if selection.exists():
            selection.unlink()
        extraction_backup.rename(extraction)
        selection_backup.rename(selection)
    for clause_id, cell in submission["answers"]["B1"].items():
        assert cell["status"] == "COMPLIANT", f"B1 {clause_id} status collapsed to {cell['status']}"
        assert cell["actual"] == 0.01, f"B1 {clause_id} actual collapsed to {cell['actual']}"
        assert cell["evidence_txn_id"] is None


def test_actual_always_positive_two_decimals():
    for clauses in run_pipeline()["answers"].values():
        for cell in clauses.values():
            assert cell["actual"] > 0
            assert Decimal(str(cell["actual"])).as_tuple().exponent >= -2


def test_compute_health_signals_from_workspace(tmp_path: Path):
    """Every run must emit the internal signals we use to pick between
    runs on 9 August — that is, without any ground truth."""
    from pipeline import _compute_health

    workspace = tmp_path
    (workspace / "selections").mkdir()
    (workspace / "selections" / "rejected").mkdir()
    (workspace / "selections" / "A.json").write_text(
        json.dumps({
            "output": {"6.1": {"revenue": ["TXN-A-1"]}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "soft_warnings": ["stage7 salvage: 6.2 dropped (…)", "some other warning"],
        }),
        encoding="utf-8",
    )
    (workspace / "selections" / "B.json").write_text(
        json.dumps({
            "output": {"6.1": {}},
            "usage": {"input_tokens": 1, "output_tokens": 1},
            "soft_warnings": ["stage7 accepted on retry after: something"],
        }),
        encoding="utf-8",
    )
    (workspace / "selections" / "rejected" / "C.json").write_text(
        json.dumps({"reason": "both attempts failed"}), encoding="utf-8"
    )
    submission = {
        "answers": {
            "A": {
                "6.1": {"status": "BREACH", "actual": 1.5, "evidence_txn_id": "TXN-A-2"},
                "6.2": {"status": "COMPLIANT", "actual": 0.01, "evidence_txn_id": None},
            },
            "B": {
                "6.1": {"status": "COMPLIANT", "actual": 0.01, "evidence_txn_id": None},
            },
            "C": {
                "6.1": {"status": "COMPLIANT", "actual": 0.01, "evidence_txn_id": None},
            },
        }
    }

    health = _compute_health(submission, workspace)

    assert health["total_cells"] == 4
    assert health["baseline_cells"] == 3
    assert health["computed_cells"] == 1
    assert health["valid_selections"] == 2
    assert health["rejected_scenarios"] == 1
    assert health["salvaged_clauses"] == 1
    assert health["retried_selections"] == 1
    assert health["soft_warnings"] == 3
