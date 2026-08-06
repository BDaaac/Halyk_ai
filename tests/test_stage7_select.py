import json
from decimal import Decimal

import pandas as pd
import pytest

from stages.s7_select import select_scenario


class FakeClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create_structured_message(self, **kwargs):
        from lib.anthropic_client import StructuredResult

        self.calls.append(kwargs)
        return StructuredResult(output=self.output, usage={"input_tokens": 13, "output_tokens": 5})


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


def test_stage7_rejects_uncertain_as_a_substitute_for_a_required_role(tmp_path):
    client = FakeClient({
        "6.1": {"revenue": [], "opex": ["TXN-T1-0002"]},
        "uncertain": [{"txn_id": "TXN-T1-0001", "role": "revenue", "note": "close match"}],
    })

    with pytest.raises(ValueError, match="6.1/revenue is empty despite uncertain candidates"):
        select_scenario(
            scenario_id="T1",
            extraction=_extraction(),
            ledger=_ledger(),
            client=client,
            cache_dir=tmp_path / "selections",
        )

    rejected = json.loads((tmp_path / "selections" / "rejected" / "T1.json").read_text(encoding="utf-8"))
    assert rejected["usage"] == {"input_tokens": 13, "output_tokens": 5}
    assert "empty despite uncertain" in rejected["reason"]


def test_stage7_rejects_uncertain_transaction_missing_from_its_role(tmp_path):
    client = FakeClient({
        "6.1": {"revenue": ["TXN-T1-0001"], "opex": ["TXN-T1-0002"]},
        "uncertain": [{"txn_id": "TXN-T1-0002", "role": "revenue", "note": "close match"}],
    })

    with pytest.raises(ValueError, match="uncertain transaction TXN-T1-0002 is not selected for role revenue"):
        select_scenario(
            scenario_id="T1",
            extraction=_extraction(),
            ledger=_ledger(),
            client=client,
            cache_dir=tmp_path / "selections",
        )


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
