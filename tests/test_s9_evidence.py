from decimal import Decimal

import pandas as pd

from stages.s9_evidence import (
    counterfactual_kind,
    evidence_candidates,
    find_counterfactual_evidence,
    reverted_adjustments,
)


def test_evidence_candidates_resolve_audited_match_without_txn_id():
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0020", "amount": Decimal("-592296.10"), "counterparty": "Irtysh Advisory Bureau"},
    ])
    adjustments = [{
        "type": "reclassification",
        "match": {"txn_id": None, "amount": "592296.10", "counterparty": "Irtysh Advisory Bureau"},
        "accepted": True,
    }]

    candidates = evidence_candidates({"interest": ["TXN-T1-0033"]}, adjustments, ledger)

    assert candidates == ["TXN-T1-0020"]


def test_counterfactual_skips_candidate_that_makes_denominator_zero():
    def recompute(txn_id):
        if txn_id == "TXN-T1-revenue":
            raise ArithmeticError("division by zero")
        return "BREACH"

    assert find_counterfactual_evidence(
        base_status="BREACH",
        candidates=["TXN-T1-revenue", "TXN-T1-opex"],
        recompute=recompute,
    ) is None


def test_revert_adjustment_marks_only_matching_audit_adjustment_unaccepted():
    ledger = pd.DataFrame([{"txn_id": "TXN-T1-0020", "amount": Decimal("-10"), "counterparty": "Advisor"}])
    adjustments = [{"type": "reclassification", "match": {"txn_id": "TXN-T1-0020"}, "accepted": True}]

    reverted = reverted_adjustments("TXN-T1-0020", adjustments, ledger)

    assert reverted[0]["accepted"] is False
    assert adjustments[0]["accepted"] is True


def test_audited_adjustment_uses_revert_not_drop_counterfactual():
    ledger = pd.DataFrame([{"txn_id": "TXN-T1-0020", "amount": Decimal("-10"), "counterparty": "Advisor"}])
    adjustments = [{"type": "reclassification", "match": {"txn_id": "TXN-T1-0020"}, "accepted": True}]

    assert counterfactual_kind("TXN-T1-0020", adjustments, ledger) == "revert_adjustment"
    assert counterfactual_kind("TXN-T1-0099", adjustments, ledger) == "drop_txn"
