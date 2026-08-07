from models import DocumentSet
import pytest

from stages.s6_extract import DerivedRoleError, _article_six, extract_scenario, regex_guards


class FakeClient:
    def __init__(self, output):
        self.output = output
        self.calls = []

    def create_structured_message(self, **kwargs):
        from lib.anthropic_client import StructuredResult

        self.calls.append(kwargs)
        return StructuredResult(output=self.output, usage={"input_tokens": 11, "output_tokens": 7})


def _documents():
    return DocumentSet(agreement="agreement", kyc="kyc", aup_report="aup", audit_notes="notes")


def _texts():
    return {
        "agreement": "Article 6\n6.1 Borrower shall maintain the ratio not less than 2.00.\nArticle 7",
        "kyc": "beneficial ownership",
        "aup": "agreed-upon procedures",
        "notes": "audit covenant notes",
    }


def _output(operator=">="):
    return {
        "covenants": [{"clause_id": "6.1", "operator": operator, "threshold": "2.00", "value_expr": {"op": "sum", "role": "revenue"}}],
        "adjustments": [],
        "related_parties": {"entities": []},
        "document_facts": [],
    }


def test_article_six_skips_table_of_contents_before_real_clause():
    text = "Статья 6\nФинансовые ковенанты\nСтатья 7\nОграничения\n...\nСтатья 6 — Финансовые ковенанты\nПункт 6.1 не менее 2.00\nСтатья 7 — Ограничения"

    assert "Пункт 6.1" in _article_six(text)


def test_article_six_falls_back_to_full_text_when_no_clauses_inside():
    """A contract that numbers covenants under a different heading must not
    send an empty section to the model — the whole text is safer."""
    text = "Статья 6\nПрочие условия\nСтатья 7\nФинансовые ковенанты\nПункт 7.1 не менее 2.00"

    assert _article_six(text) == text


def test_guards_compare_each_covenant_to_its_own_clause():
    article = "Пункт 6.1 не менее 2.00\nПункт 6.2 не более 100.00"
    output = {
        "covenants": [
            {"clause_id": "6.1", "operator": ">=", "threshold": "2.00", "value_expr": {"op": "sum", "role": "a"}},
            {"clause_id": "6.2", "operator": "<=", "threshold": "100.00", "value_expr": {"op": "sum", "role": "b"}},
        ]
    }

    assert regex_guards(output, article) == []


def test_stage6_rejects_derived_metric_as_a_ledger_role():
    output = {
        "covenants": [{
            "clause_id": "6.1",
            "role_descriptions": {"EBITDA": "earnings before interest, taxes, depreciation and amortization"},
        }]
    }

    with pytest.raises(DerivedRoleError, match="EBITDA"):
        regex_guards(output, "Clause 6.1")


def test_stage6_calls_once_then_uses_scenario_cache(tmp_path):
    client = FakeClient(_output())

    first = extract_scenario(
        scenario_id="T1",
        documents=_documents(),
        document_texts=_texts(),
        clause_ids=["6.1"],
        client=client,
        cache_dir=tmp_path / "extractions",
    )
    second = extract_scenario(
        scenario_id="T1",
        documents=_documents(),
        document_texts=_texts(),
        clause_ids=["6.1"],
        client=client,
        cache_dir=tmp_path / "extractions",
    )

    assert len(client.calls) == 1
    assert first.output == second.output == _output()
    assert first.usage == {"input_tokens": 11, "output_tokens": 7}
    assert "<covenant_article>" in client.calls[0]["user"]
    assert "<clause_ids>6.1</clause_ids>" in client.calls[0]["user"]


def test_stage6_guard_warns_but_does_not_change_model_operator(tmp_path):
    client = FakeClient(_output(operator="<="))

    result = extract_scenario(
        scenario_id="T1",
        documents=_documents(),
        document_texts=_texts(),
        clause_ids=["6.1"],
        client=client,
        cache_dir=tmp_path / "extractions",
    )

    assert result.output["covenants"][0]["operator"] == "<="
    assert any("operator" in warning for warning in result.soft_warnings)
