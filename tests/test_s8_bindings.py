from decimal import Decimal

import pandas as pd
import pytest

from lib.currency import normalize_ledger_to_usd
from stages.s8_compute import build_clause_selection, evaluate_covenant, specifications_from_extraction


def test_stage8_applies_reclassification_once_to_matching_target_role():
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "value_expr": {"op": "divide", "args": [
                {"op": "subtract", "args": [{"op": "sum", "role": "revenue"}, {"op": "sum", "role": "opex"}]},
                {"op": "sum", "role": "interest_expenses"},
            ]},
            "operator": ">=",
            "threshold": "2.00",
            "role_descriptions": {"revenue": "Revenue", "opex": "Operating expense", "interest_expenses": "Interest expense"},
        }],
        "adjustments": [{
            "type": "reclassification",
            "match": {"txn_id": "TXN-T1-0004"},
            "to_role": "interest",
            "accepted": True,
        }],
    }
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0001", "amount": Decimal("9741934.78"), "counterparty": "Grid"},
        {"txn_id": "TXN-T1-0002", "amount": Decimal("-6166592.66"), "counterparty": "Plant"},
        {"txn_id": "TXN-T1-0003", "amount": Decimal("-1540833.29"), "counterparty": "Bank"},
        {"txn_id": "TXN-T1-0004", "amount": Decimal("-592296.10"), "counterparty": "Advisor"},
    ])
    selections = {"6.1": {"revenue": ["TXN-T1-0001"], "opex": ["TXN-T1-0002"], "interest_expenses": ["TXN-T1-0003"]}}

    specification = specifications_from_extraction("T1", extraction)["6.1"]
    selection = build_clause_selection(specification, selections["6.1"], ledger, extraction["adjustments"])
    result = evaluate_covenant(specification, selection)

    assert selection["interest_expenses"] == [Decimal("1540833.29"), Decimal("592296.10")]
    assert result.actual.quantize(Decimal("0.01")) == Decimal("1.68")


def test_stage8_parses_and_enforces_a_springing_applicability_condition():
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "value_expr": {"op": "sum", "role": "ratio"},
            "operator": "<=",
            "threshold": "1.70",
            "applicability": {
                "expr": {"op": "sum", "role": "financing_receipts"},
                "operator": ">",
                "threshold": "4000000.00",
                "raw_text": "applies only when financing receipts exceed $4,000,000.00",
            },
        }],
    }
    specification = specifications_from_extraction("T1", extraction)["6.1"]

    active = evaluate_covenant(specification, {"ratio": [Decimal("1.71")], "financing_receipts": [Decimal("5000000")]})
    inactive = evaluate_covenant(specification, {"ratio": [Decimal("1.71")], "financing_receipts": [Decimal("0")]})

    assert specification.applicability is not None
    assert specification.applicability.threshold == Decimal("4000000.00")
    assert active.status == "BREACH"
    assert inactive.status == "COMPLIANT"
    assert inactive.actual == Decimal("1.71")


def test_stage8_ignores_audit_rejected_reclassification():
    extraction = {
        "covenants": [{
            "clause_id": "6.2",
            "value_expr": {"op": "sum", "role": "interest"},
            "operator": "<=",
            "threshold": "100",
        }],
        "adjustments": [{
            "type": "reclassification",
            "match": {"txn_id": "TXN-T1-0001"},
            "to_role": "interest",
            "accepted": False,
        }],
    }
    ledger = pd.DataFrame([{"txn_id": "TXN-T1-0001", "amount": Decimal("-90"), "counterparty": "Auditor"}])
    specification = specifications_from_extraction("T1", extraction)["6.2"]

    selection = build_clause_selection(specification, {"interest": []}, ledger, extraction["adjustments"])

    assert selection["interest"] == []


def test_stage8_converts_selected_foreign_currency_before_role_sum():
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "value_expr": {"op": "sum", "role": "opex"},
            "operator": "<=",
            "threshold": "6000000",
            "role_descriptions": {"opex": "Operating expenses"},
        }],
        "adjustments": [{
            "type": "fx_rate",
            "match": {"txn_id": None, "amount": None, "counterparty": "Rheinland Katalyse Service GmbH"},
            "derived_from": {"foreign_amount": "72146.75", "foreign_currency": "EUR", "usd_amount": "83690.23"},
        }],
    }
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0001", "amount": Decimal("-4218006.51"), "currency": "USD", "counterparty": "Plant"},
        {"txn_id": "TXN-T1-0002", "amount": Decimal("-612884.25"), "currency": "EUR", "counterparty": "Rheinland Katalyse Service GmbH"},
    ])
    specification = specifications_from_extraction("T1", extraction)["6.1"]

    selection = build_clause_selection(
        specification,
        {"opex": ["TXN-T1-0001", "TXN-T1-0002"]},
        ledger,
        extraction["adjustments"],
    )

    assert selection["opex"] == [Decimal("4218006.51"), Decimal("710945.73")]


def test_currency_normalization_converts_audited_fx_before_selection_context():
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0001", "amount": Decimal("-612884.25"), "currency": "EUR", "counterparty": "Rheinland Katalyse Service GmbH"},
    ])
    adjustments = [{
        "type": "fx_rate",
        "match": {"txn_id": None, "amount": None, "counterparty": "Rheinland Katalyse Service GmbH"},
        "derived_from": {"foreign_amount": "72146.75", "foreign_currency": "EUR", "usd_amount": "83690.23"},
    }]

    normalized = normalize_ledger_to_usd(ledger, adjustments)

    assert normalized.loc[0, "amount"] == Decimal("-710945.73")
    assert normalized.loc[0, "currency"] == "USD"


def test_stage8_amount_correction_without_accepted_field_is_applied():
    """Haiku sometimes omits 'accepted' on amount_correction — treating that
    as accepted is the difference between P7 6.1 landing on the truth vs
    a silent NaN drop to baseline."""
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "value_expr": {"op": "sum", "role": "taxes"},
            "operator": "<=",
            "threshold": "1000000",
            "role_descriptions": {"taxes": "Taxes"},
        }],
        "adjustments": [{
            "type": "amount_correction",
            "match": {"txn_id": "TXN-T1-0033", "amount": "486204.19", "counterparty": "State Revenue"},
            "sign": "expense",
            # NOTE: no 'accepted' field — must be treated as True.
        }],
    }
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0010", "amount": Decimal("402118.64"), "counterparty": "Grid", "currency": "USD"},
        {"txn_id": "TXN-T1-0033", "amount": float("nan"), "counterparty": "State Revenue", "currency": "USD"},
    ])
    spec = specifications_from_extraction("T1", extraction)["6.1"]

    selection = build_clause_selection(spec, {"taxes": ["TXN-T1-0010", "TXN-T1-0033"]}, ledger, extraction["adjustments"])

    assert selection["taxes"] == [Decimal("402118.64"), Decimal("486204.19")]


def test_stage8_ignores_hallucinated_condition_with_non_numeric_threshold():
    """Haiku sometimes emits an applicability whose threshold is free text
    ('2025-01-01 to 2025-12-31'). Treat the condition as absent instead
    of taking down the whole scenario setup."""
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "value_expr": {"op": "sum", "role": "revenue"},
            "operator": ">=",
            "threshold": "1000",
            "applicability": {
                "expr": {"op": "fact", "fact_name": "covenant_period"},
                "operator": "==",
                "threshold": "2025-01-01 to 2025-12-31",  # not a Decimal
            },
        }],
    }
    specs = specifications_from_extraction("T1", extraction)

    assert "6.1" in specs
    assert specs["6.1"].applicability is None


def test_stage8_nan_amount_raises_labelled_error_naming_txn_and_role():
    """A NaN that no correction covered must produce a message the
    operator can actually act on — not a bare Decimal InvalidOperation."""
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "value_expr": {"op": "sum", "role": "payroll"},
            "operator": "<=",
            "threshold": "1000",
            "role_descriptions": {"payroll": "Payroll"},
        }],
    }
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0007", "amount": float("nan"), "counterparty": "Anyone", "currency": "USD"},
    ])
    spec = specifications_from_extraction("T1", extraction)["6.1"]

    with pytest.raises(ValueError, match="TXN-T1-0007.*payroll"):
        build_clause_selection(spec, {"payroll": ["TXN-T1-0007"]}, ledger, [])


def test_specifications_localises_unsupported_op_via_unsupported_out():
    """Failure containment #1: an unknown value_expr.op does not sink the
    whole scenario. The offending clause is recorded in unsupported_out
    with its extracted threshold; other clauses build normally."""
    from stages.s8_compute import UnsupportedSpecError, specifications_from_extraction

    extraction = {
        "covenants": [
            {
                "clause_id": "6.1",
                "value_expr": {"op": "concat", "role": "x"},  # unsupported op
                "operator": ">=",
                "threshold": "3500000.00",
                "role_descriptions": {"x": "Whatever"},
            },
            {
                "clause_id": "6.2",
                "value_expr": {"op": "sum", "role": "revenue"},
                "operator": ">=",
                "threshold": "1000000.00",
                "role_descriptions": {"revenue": "Revenue"},
            },
        ],
    }
    unsupported: dict = {}

    specs = specifications_from_extraction("T1", extraction, unsupported_out=unsupported)

    assert "6.1" not in specs
    assert "6.2" in specs
    assert "6.1" in unsupported
    assert unsupported["6.1"]["threshold"] == Decimal("3500000.00")
    assert "concat" in unsupported["6.1"]["reason"]

    # Without unsupported_out, the exception must propagate — nobody should
    # accidentally swallow it as a plain Exception.
    with pytest.raises(UnsupportedSpecError):
        specifications_from_extraction("T1", extraction)


def test_evaluate_raises_unsupported_for_missing_role_and_missing_fact():
    """Failure containment #2 & #3: missing role in a sum node and missing
    required fact both surface as UnsupportedSpecError, not bare ValueError."""
    from stages.s8_compute import UnsupportedSpecError, evaluate_expr
    from models import Expr

    with pytest.raises(UnsupportedSpecError, match="missing a role"):
        evaluate_expr(Expr(op="sum", role=None), selection={})

    with pytest.raises(UnsupportedSpecError, match="document fact is missing"):
        evaluate_expr(Expr(op="fact", fact_name="ghost"), selection={}, facts={})


def test_evaluate_still_raises_bare_valueerror_for_programming_errors():
    """Non-containment ValueErrors (e.g. depth-limit, invalid const) must
    keep surfacing as ValueError so a broad `except UnsupportedSpecError`
    in the pipeline does not swallow real bugs."""
    from stages.s8_compute import UnsupportedSpecError, evaluate_expr
    from models import Expr

    # Depth cap trips well before it would need to touch role/fact logic.
    deep = Expr(op="add", args=(Expr(op="sum", role="r"), Expr(op="sum", role="r")))
    for _ in range(6):
        deep = Expr(op="add", args=(deep, Expr(op="sum", role="r")))
    with pytest.raises(ValueError) as exc_info:
        evaluate_expr(deep, selection={"r": [Decimal("1")]})
    assert not isinstance(exc_info.value, UnsupportedSpecError)

    # const without value.
    with pytest.raises(ValueError) as exc_info:
        evaluate_expr(Expr(op="const", value=None), selection={})
    assert not isinstance(exc_info.value, UnsupportedSpecError)


def test_stage8_evaluates_document_fact_expression():
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "value_expr": {"op": "add", "args": [{"op": "sum", "role": "payroll"}, {"op": "fact", "fact_name": "severance"}]},
            "operator": "<=",
            "threshold": "1000",
            "role_descriptions": {"payroll": "Payroll"},
        }],
        "document_facts": [{"fact_name": "severance", "value": "25.50", "unit": "USD"}],
    }
    spec = specifications_from_extraction("T1", extraction)["6.1"]

    assert evaluate_covenant(spec, {"payroll": [Decimal("100")]}) .actual == Decimal("125.50")
