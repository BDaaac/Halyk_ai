import json
from decimal import Decimal

import pandas as pd
import pytest

from stages.s7_select import apply_retry_conservatism, select_scenario


class FakeClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create_structured_message(self, **kwargs):
        from lib.anthropic_client import StructuredResult

        self.calls.append(kwargs)
        return StructuredResult(output=self.output, usage={"input_tokens": 13, "output_tokens": 5})


class QueuedFakeClient:
    """Return a pre-scripted sequence of outputs across successive calls."""

    def __init__(self, outputs):
        self.outputs = list(outputs)
        self.calls = []

    def create_structured_message(self, **kwargs):
        from lib.anthropic_client import StructuredResult

        self.calls.append(kwargs)
        output = self.outputs.pop(0)
        return StructuredResult(output=output, usage={"input_tokens": 13, "output_tokens": 5})


def _ledger():
    return pd.DataFrame(
        [
            {"txn_id": "TXN-T1-0001", "description": "Revenue from core operations", "amount": 10, "counterparty": "Eligible LLP"},
            {"txn_id": "TXN-T1-0002", "description": "Operating maintenance", "amount": -4, "counterparty": "Low Share LLP"},
            {"txn_id": "TXN-T2-0001", "description": "Other borrower revenue", "amount": 8, "counterparty": "Eligible LLP"},
        ]
    )


def _extraction():
    return {
        "covenants": [{"clause_id": "6.1", "role_descriptions": {"revenue": "Revenue", "opex": "Operating maintenance"}}],
        "adjustments": [],
        "related_parties": {
            "threshold_percent": "20.0",
            "entities": [
                {"name": "Eligible, LLP", "share_percent": "31.4"},
                {"name": "Low Share LLP", "share_percent": "12.5"},
            ],
        },
    }


def test_stage7_calls_once_for_whole_scenario_then_uses_cache(tmp_path):
    client = FakeClient({"6.1": {"revenue": ["TXN-T1-0001"], "opex": ["TXN-T1-0002"]}, "uncertain": []})

    first = select_scenario(
        scenario_id="T1",
        extraction=_extraction(),
        ledger=_ledger(),
        client=client,
        cache_dir=tmp_path / "selections",
    )
    second = select_scenario(
        scenario_id="T1",
        extraction=_extraction(),
        ledger=_ledger(),
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert first.output == second.output
    assert len(client.calls) == 1
    assert "<transactions>" in client.calls[0]["user"]
    schema = client.calls[0]["input_schema"]
    assert "6.1" in schema["required"]
    assert schema["properties"]["6.1"]["required"] == ["revenue", "opex"]


def test_stage7_rejects_transaction_that_does_not_belong_to_scenario(tmp_path):
    client = FakeClient({"6.1": {"revenue": ["TXN-T2-0001"]}, "uncertain": []})

    with pytest.raises(ValueError, match="does not belong"):
        select_scenario(
            scenario_id="T1",
            extraction=_extraction(),
            ledger=_ledger(),
            client=client,
            cache_dir=tmp_path / "selections",
        )


def test_stage7_filters_related_party_context_and_drops_ineligible_selection(tmp_path):
    client = FakeClient({"6.3": {"related_party": ["TXN-T1-0002"]}, "uncertain": []})

    result = select_scenario(
        scenario_id="T1",
        extraction=_extraction(),
        ledger=_ledger(),
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert result.output["6.3"]["related_party"] == []
    related_context = client.calls[0]["user"].split("<related_parties>", 1)[1].split("</related_parties>", 1)[0]
    assert "Low Share LLP" not in related_context
    assert not any("Low Share" in warning for warning in result.soft_warnings)


def test_stage7_removes_a_reclassification_from_its_target_role(tmp_path):
    extraction = _extraction()
    extraction["adjustments"] = [{
        "type": "reclassification",
        "match": {"txn_id": "TXN-T1-0001"},
        "to_role": "revenue",
    }]
    client = FakeClient({"6.1": {"revenue": ["TXN-T1-0001"]}, "uncertain": []})

    result = select_scenario(
        scenario_id="T1",
        extraction=extraction,
        ledger=_ledger(),
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert result.output["6.1"]["revenue"] == []
    assert "TXN-T1-0001: model applied reclassification, removed from target role revenue" in result.soft_warnings


def test_stage7_rejects_a_role_that_is_not_a_list_of_transaction_ids(tmp_path):
    client = FakeClient({"6.1": {"revenue": [{"txn_id": "TXN-T1-0001"}]}, "uncertain": []})

    with pytest.raises(ValueError, match="must contain only transaction ID strings"):
        select_scenario(
            scenario_id="T1",
            extraction=_extraction(),
            ledger=_ledger(),
            client=client,
            cache_dir=tmp_path / "selections",
        )


def test_stage7_salvages_a_clause_with_empty_required_role_and_matching_uncertain(tmp_path):
    """A semantic error inside one clause must not kill the whole scenario.
    The offending clause drops to empty; the salvage is recorded in
    soft_warnings and the pipeline sees {} for that clause downstream."""
    client = FakeClient({
        "6.1": {"revenue": [], "opex": ["TXN-T1-0002"]},
        "uncertain": [{"txn_id": "TXN-T1-0001", "role": "revenue", "note": "close match"}],
    })

    result = select_scenario(
        scenario_id="T1",
        extraction=_extraction(),
        ledger=_ledger(),
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert result.output["6.1"] == {}
    assert any("stage7 salvage" in w and "6.1" in w for w in result.soft_warnings)
    # A cache file lands for the surviving clauses so run_pipeline does
    # not re-issue the paid call on the next run.
    cached = json.loads((tmp_path / "selections" / "T1.json").read_text(encoding="utf-8"))
    assert cached["output"]["6.1"] == {}


def test_stage7_second_attempt_accepted_after_first_shape_failure(tmp_path):
    """A schema-level failure (list where a dict is required) triggers a
    real retry with the error text fed back to the model. If the second
    call returns a valid selection, use it."""
    first_bad = {"6.1": [], "uncertain": []}  # 6.1 must be an object, not a list
    second_good = {"6.1": {"revenue": ["TXN-T1-0001"], "opex": ["TXN-T1-0002"]}, "uncertain": []}
    client = QueuedFakeClient([first_bad, second_good])

    result = select_scenario(
        scenario_id="T1",
        extraction=_extraction(),
        ledger=_ledger(),
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert len(client.calls) == 2
    # The retry's user prompt carries the validation error verbatim.
    assert "<repair>" in client.calls[1]["user"]
    assert "must map roles" in client.calls[1]["user"]
    assert result.output == second_good
    assert any("accepted on retry" in w for w in result.soft_warnings)


def test_stage7_two_failed_shape_attempts_raise_and_record_both_in_rejected(tmp_path):
    """When both attempts return structurally broken output, both raw
    responses land in rejected/ so an operator can inspect them."""
    first_bad = {"6.1": ["not-a-dict-1"], "uncertain": []}
    second_bad = {"6.1": ["not-a-dict-2"], "uncertain": []}
    client = QueuedFakeClient([first_bad, second_bad])

    with pytest.raises(ValueError, match="must map roles"):
        select_scenario(
            scenario_id="T1",
            extraction=_extraction(),
            ledger=_ledger(),
            client=client,
            cache_dir=tmp_path / "selections",
        )

    rejected = json.loads((tmp_path / "selections" / "rejected" / "T1.json").read_text(encoding="utf-8"))
    assert len(rejected["attempts"]) == 2
    assert rejected["attempts"][0]["output"]["6.1"] == ["not-a-dict-1"]
    assert rejected["attempts"][1]["output"]["6.1"] == ["not-a-dict-2"]


def test_stage7_salvages_only_the_clause_the_uncertain_item_blames(tmp_path):
    """Blame the clause whose role the uncertain item names; other
    clauses' selections are kept as-is."""
    extraction = {
        "covenants": [
            {"clause_id": "6.1", "role_descriptions": {"revenue": "Revenue", "opex": "Operating maintenance"}},
            {"clause_id": "6.2", "role_descriptions": {"payroll": "Payroll"}},
        ],
        "adjustments": [],
        "related_parties": {"threshold_percent": "20.0", "entities": []},
    }
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0001", "description": "Revenue", "amount": 10, "counterparty": "X"},
        {"txn_id": "TXN-T1-0002", "description": "Payroll", "amount": -4, "counterparty": "Y"},
        {"txn_id": "TXN-T1-0003", "description": "Salary bonus", "amount": -3, "counterparty": "Z"},
    ])
    client = FakeClient({
        "6.1": {"revenue": ["TXN-T1-0001"], "opex": ["TXN-T1-0002"]},
        "6.2": {"payroll": ["TXN-T1-0002"]},
        "uncertain": [{"txn_id": "TXN-T1-0003", "role": "opex", "note": "maybe operational"}],
    })

    result = select_scenario(
        scenario_id="T1",
        extraction=extraction,
        ledger=ledger,
        client=client,
        cache_dir=tmp_path / "selections",
    )

    # 6.1 owns role "opex" — it is dropped; 6.2 survives untouched.
    assert result.output["6.1"] == {}
    assert result.output["6.2"] == {"payroll": ["TXN-T1-0002"]}


def test_stage7_semantic_failure_recovered_on_retry(tmp_path):
    """A semantic per-clause error triggers ONE neutral clause-level retry
    before salvage. If the retry returns a self-consistent selection, the
    clause is preserved and the recovery is noted in soft_warnings."""
    first_bad = {
        "6.1": {"revenue": [], "opex": ["TXN-T1-0002"]},
        "uncertain": [{"txn_id": "TXN-T1-0001", "role": "revenue", "note": "close match"}],
    }
    second_good = {
        "6.1": {"revenue": ["TXN-T1-0001"], "opex": ["TXN-T1-0002"]},
        "uncertain": [],
    }
    client = QueuedFakeClient([first_bad, second_good])

    result = select_scenario(
        scenario_id="T1",
        extraction=_extraction(),
        ledger=_ledger(),
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert len(client.calls) == 2
    # The retry prompt carries the neutral repair note, not an instruction
    # to include a specific txn.
    retry_prompt = client.calls[1]["user"]
    assert "<repair>" in retry_prompt
    assert "internally inconsistent" in retry_prompt
    assert "Re-evaluate this clause only" in retry_prompt
    assert result.output["6.1"] == {"revenue": ["TXN-T1-0001"], "opex": ["TXN-T1-0002"]}
    assert any("accepted on retry after semantic" in w for w in result.soft_warnings)


def test_stage7_direction_mismatch_triggers_semantic_retry(tmp_path):
    """Selecting a positive (incoming) txn into a role whose description
    names outgoing payments IS a semantic error. It must route through the
    same clause-level retry as uncertain-substitution — Python does not
    decide by itself which txn is correct."""
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "role_descriptions": {
                "restricted_payment": "Ограниченные платежи в пользу аффилированных лиц",
            },
        }],
        "adjustments": [],
        "related_parties": {"threshold_percent": "20.0", "entities": []},
    }
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0001", "description": "Refinery product sales", "amount": 2000000.0, "counterparty": "Some LLP", "currency": "USD"},
        {"txn_id": "TXN-T1-0002", "description": "Management retainer",     "amount": -100000.0, "counterparty": "Other LLP", "currency": "USD"},
    ])
    # Attempt 1: model picked the positive-amount txn → direction mismatch.
    # Attempt 2: model corrects to the outgoing txn only.
    first_bad = {"6.1": {"restricted_payment": ["TXN-T1-0001"]}, "uncertain": []}
    second_good = {"6.1": {"restricted_payment": ["TXN-T1-0002"]}, "uncertain": []}
    client = QueuedFakeClient([first_bad, second_good])

    result = select_scenario(
        scenario_id="T1",
        extraction=extraction,
        ledger=ledger,
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert len(client.calls) == 2
    assert "outgoing payments but selected" in client.calls[1]["user"]
    assert result.output["6.1"] == {"restricted_payment": ["TXN-T1-0002"]}


def test_stage7_direction_mismatch_salvages_if_retry_also_wrong(tmp_path):
    """When both attempts violate direction, existing salvage applies —
    Python still does not silently filter the model's selection."""
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "role_descriptions": {
                "restricted_payment": "Ограниченные платежи в пользу аффилированных лиц",
            },
        }],
        "adjustments": [],
        "related_parties": {"threshold_percent": "20.0", "entities": []},
    }
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0001", "description": "Sales", "amount": 2000000.0, "counterparty": "X", "currency": "USD"},
    ])
    bad = {"6.1": {"restricted_payment": ["TXN-T1-0001"]}, "uncertain": []}
    client = QueuedFakeClient([bad, bad])

    result = select_scenario(
        scenario_id="T1",
        extraction=extraction,
        ledger=ledger,
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert result.output["6.1"] == {}
    assert any("stage7 salvage" in w and "outgoing" in w for w in result.soft_warnings)


def test_stage7_uncertain_transaction_missing_from_its_role_salvages_the_clause(tmp_path):
    client = FakeClient({
        "6.1": {"revenue": ["TXN-T1-0001"], "opex": ["TXN-T1-0002"]},
        "uncertain": [{"txn_id": "TXN-T1-0002", "role": "revenue", "note": "close match"}],
    })

    result = select_scenario(
        scenario_id="T1",
        extraction=_extraction(),
        ledger=_ledger(),
        client=client,
        cache_dir=tmp_path / "selections",
    )

    # 6.1 owns 'revenue'; the uncertain item names revenue but the txn
    # was placed under opex instead — clause 6.1 drops.
    assert result.output["6.1"] == {}


def test_stage7_warns_when_foreign_currency_transaction_is_not_selected(tmp_path):
    ledger = _ledger().copy()
    ledger["currency"] = ["USD", "EUR", "USD"]
    client = FakeClient({"6.1": {"revenue": ["TXN-T1-0001"], "opex": []}, "uncertain": []})

    result = select_scenario(
        scenario_id="T1",
        extraction=_extraction(),
        ledger=ledger,
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert "TXN-T1-0002 in EUR was not selected for any role" in result.soft_warnings


def test_stage7_sends_audited_foreign_amount_to_model_in_usd(tmp_path):
    ledger = _ledger().iloc[:2].copy()
    ledger["currency"] = ["USD", "EUR"]
    ledger["amount"] = [Decimal("10"), Decimal("-612884.25")]
    ledger.loc[1, "counterparty"] = "Rheinland Katalyse Service GmbH"
    extraction = _extraction()
    extraction["adjustments"] = [{
        "type": "fx_rate",
        "match": {"txn_id": None, "amount": None, "counterparty": "Rheinland Katalyse Service GmbH"},
        "derived_from": {"foreign_amount": "72146.75", "foreign_currency": "EUR", "usd_amount": "83690.23"},
    }]
    client = FakeClient({"6.1": {"revenue": ["TXN-T1-0001"], "opex": ["TXN-T1-0002"]}, "uncertain": []})

    select_scenario(scenario_id="T1", extraction=extraction, ledger=ledger, client=client, cache_dir=tmp_path / "selections")

    transactions = client.calls[0]["user"].split("<transactions>", 1)[1].split("</transactions>", 1)[0]
    assert "710945.73" in transactions
    assert '"currency":"USD"' in transactions


def test_apply_retry_conservatism_strips_flipped_uncertain_inclusion():
    """Attempt-1 flagged TXN-Y in uncertain[opex] but did NOT put it into
    opex. The retry (accepted) then placed it. The rule removes it and
    records a note. Nothing else in the selection is touched."""
    output = {
        "6.1": {"revenue": ["TXN-X"], "opex": ["TXN-BASE", "TXN-Y"]},
        "6.2": {"payroll": ["TXN-P"]},
        "uncertain": [],
    }
    soft_warnings = [
        "stage7 accepted on retry after semantic: semantic: 6.1: "
        "uncertain transaction TXN-Y for role 'opex' was not placed here",
    ]
    modified, notes = apply_retry_conservatism(output, soft_warnings)
    assert modified["6.1"]["opex"] == ["TXN-BASE"]
    assert modified["6.1"]["revenue"] == ["TXN-X"]
    assert modified["6.2"] == {"payroll": ["TXN-P"]}
    assert any("removed TXN-Y from 6.1/opex" in note for note in notes)


def test_apply_retry_conservatism_is_noop_when_retry_kept_exclusion():
    """If the txn the model flagged in attempt-1 is NOT in the final
    role, the rule is silent — retry already agreed."""
    output = {
        "6.1": {"revenue": ["TXN-X"], "opex": ["TXN-BASE"]},
        "uncertain": [],
    }
    soft_warnings = [
        "stage7 accepted on retry after semantic: semantic: 6.1: "
        "uncertain transaction TXN-Y for role 'opex' was not placed here",
    ]
    modified, notes = apply_retry_conservatism(output, soft_warnings)
    assert modified == output
    assert notes == []


def test_apply_retry_conservatism_ignores_direction_and_shape_retries():
    """Only the uncertain-substitution wording is actioned. Direction
    mismatch and shape retries have different remediation paths."""
    output = {"6.1": {"restricted_payment": ["TXN-Y"]}, "uncertain": []}
    soft_warnings = [
        "stage7 accepted on retry after semantic: semantic: 6.1: "
        "role 'restricted_payment' describes outgoing payments but "
        "selected transaction TXN-Y has positive amount (+2000000.00)",
        "stage7 accepted on retry after shape: shape: ValueError: bad schema",
    ]
    modified, notes = apply_retry_conservatism(output, soft_warnings)
    assert modified == output
    assert notes == []


def test_stage7_uncertain_substitution_retry_gets_conservatism_stripped(tmp_path):
    """End-to-end: attempt-1 puts TXN-T1-0001 in uncertain[revenue] but
    also fills opex non-emptily (Check 2 wording, not Check 1). The retry
    places TXN-T1-0001 in revenue. The final output must have revenue
    empty — the rule undoes the flip — and the note must be recorded."""
    extraction = {
        "covenants": [{
            "clause_id": "6.1",
            "role_descriptions": {"revenue": "Revenue", "opex": "Operating maintenance"},
        }],
        "adjustments": [],
        "related_parties": {"threshold_percent": "20.0", "entities": []},
    }
    ledger = pd.DataFrame([
        {"txn_id": "TXN-T1-0001", "description": "Revenue from core operations", "amount": 10, "counterparty": "X"},
        {"txn_id": "TXN-T1-0002", "description": "Operating maintenance", "amount": -4, "counterparty": "Y"},
        {"txn_id": "TXN-T1-0003", "description": "Revenue special", "amount": 20, "counterparty": "Z"},
    ])
    first_bad = {
        "6.1": {"revenue": ["TXN-T1-0003"], "opex": ["TXN-T1-0002"]},
        "uncertain": [{"txn_id": "TXN-T1-0001", "role": "revenue", "note": "borderline"}],
    }
    second_retry = {
        "6.1": {"revenue": ["TXN-T1-0003", "TXN-T1-0001"], "opex": ["TXN-T1-0002"]},
        "uncertain": [],
    }
    client = QueuedFakeClient([first_bad, second_retry])

    result = select_scenario(
        scenario_id="T1",
        extraction=extraction,
        ledger=ledger,
        client=client,
        cache_dir=tmp_path / "selections",
    )

    assert result.output["6.1"]["revenue"] == ["TXN-T1-0003"]
    assert result.output["6.1"]["opex"] == ["TXN-T1-0002"]
    assert any("retry conservatism" in w and "TXN-T1-0001" in w for w in result.soft_warnings)
