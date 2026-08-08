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
        "from_role": "consulting",
        "to_role": "interest",
        "accepted": True,
    }]

    candidates = evidence_candidates(
        {"interest": ["TXN-T1-0033"]},
        adjustments,
        ledger,
        clause_roles={"interest"},
    )

    assert candidates == ["TXN-T1-0020"]


def test_evidence_candidates_ignore_adjustment_when_roles_do_not_intersect_clause():
    """P2/P7/P8-type failure mode: an accepted adjustment about an
    unrelated role must not replace the selected candidate pool for this
    clause. The evidence for a related_party covenant is a selected
    related_party txn, not an opex reclassification that happens to be
    on the same scenario."""
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0016", "amount": Decimal("-100"), "counterparty": "Related Co"},
        {"txn_id": "TXN-T1-0031", "amount": Decimal("-884204.16"), "counterparty": "Unrelated Co"},
    ])
    adjustments = [{
        "type": "amount_correction",
        "match": {"txn_id": "TXN-T1-0031"},
        "from_role": None,
        "to_role": "payroll",
        "accepted": True,
    }]

    candidates = evidence_candidates(
        {"related_party": ["TXN-T1-0016"]},
        adjustments,
        ledger,
        clause_roles={"related_party", "revenue"},
    )

    assert candidates == ["TXN-T1-0016"]


def test_evidence_candidates_keep_adjustment_when_from_role_matches_clause():
    """An adjustment whose from_role is a role this clause uses is
    directly relevant — its counterfactual (was it really accepted?)
    is a legitimate evidence question."""
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0040", "amount": Decimal("-500"), "counterparty": "Consultants LLP"},
    ])
    adjustments = [{
        "type": "reclassification",
        "match": {"txn_id": "TXN-T1-0040"},
        "from_role": "consulting",
        "to_role": "opex",
        "accepted": True,
    }]

    candidates = evidence_candidates(
        {"opex": ["TXN-T1-0050"]},
        adjustments,
        ledger,
        clause_roles={"consulting", "revenue"},
    )

    assert candidates == ["TXN-T1-0040"]


def test_evidence_candidates_fall_back_to_selection_when_no_adjustments():
    ledger = pd.DataFrame([{"txn_id": "TXN-T1-0010", "amount": Decimal("100"), "counterparty": "X"}])
    candidates = evidence_candidates(
        {"revenue": ["TXN-T1-0010"]},
        [],
        ledger,
        clause_roles={"revenue"},
    )
    assert candidates == ["TXN-T1-0010"]


def test_evidence_candidates_do_not_include_txn_from_role_irrelevant_adjustment_even_if_borrower_matches():
    """The adjustment lives on this scenario's ledger, but its role is
    disjoint from the clause. Borrower-scope membership alone must not
    admit it — only role relevance does. Falls back to selection."""
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0012", "amount": Decimal("100"), "counterparty": "Y"},
        {"txn_id": "TXN-T1-0040", "amount": Decimal("-500"), "counterparty": "Z"},
    ])
    adjustments = [{
        "type": "reclassification",
        "match": {"txn_id": "TXN-T1-0040"},
        "from_role": "consulting",
        "to_role": "opex",
        "accepted": True,
    }]

    candidates = evidence_candidates(
        {"related_party": ["TXN-T1-0012"]},
        adjustments,
        ledger,
        clause_roles={"related_party", "revenue"},
    )

    assert candidates == ["TXN-T1-0012"]
    assert "TXN-T1-0040" not in candidates


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


def test_ambiguous_evidence_returns_deterministic_first(monkeypatch, tmp_path):
    """When multiple candidates independently flip the verdict, return
    the lexicographically-smallest txn_id (deterministic scoring
    fallback), and log the ambiguity to workspace/errors.log."""
    monkeypatch.setenv("WORKSPACE_DIR", str(tmp_path))
    # Three candidates, first two flip.
    flip = {"TXN-T1-0055", "TXN-T1-0011"}
    def recompute(txn_id):
        return "COMPLIANT" if txn_id in flip else "BREACH"

    picked = find_counterfactual_evidence(
        base_status="BREACH",
        candidates=["TXN-T1-0055", "TXN-T1-0033", "TXN-T1-0011"],
        recompute=recompute,
        trace_scope="T1 6.1",
    )
    assert picked == "TXN-T1-0011"  # lexicographic min of {0055, 0011}

    log = (tmp_path / "errors.log").read_text(encoding="utf-8")
    assert "AMBIGUOUS_EVIDENCE" in log
    assert "TXN-T1-0011" in log
    assert "TXN-T1-0055" in log
    assert "fallback=TXN-T1-0011" in log
    assert "deterministic scoring fallback" in log


def test_no_flippers_still_returns_none():
    def recompute(txn_id):
        return "BREACH"

    assert find_counterfactual_evidence(
        base_status="BREACH",
        candidates=["A", "B"],
        recompute=recompute,
    ) is None
