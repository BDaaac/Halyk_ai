"""Tests for the level-2 LLM document-type fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from lib import doc_classify
from lib.anthropic_client import StructuredResult
from models import DocRecord


class FakeClient:
    def __init__(self, output: dict[str, str]):
        self.output = output
        self.calls: list[dict] = []

    def create_structured_message(self, **kwargs):
        self.calls.append(kwargs)
        return StructuredResult(output=self.output, usage={"input_tokens": 7, "output_tokens": 3})


def _settings(workspace: Path, key: str = "test-key") -> SimpleNamespace:
    return SimpleNamespace(
        workspace_dir=workspace,
        anthropic_api_key=key,
        extract_model="haiku",
        llm_timeout_seconds=10,
    )


def _record(doc_id: str, doc_type: str = "noise", account: str = "ACC-9999") -> DocRecord:
    return DocRecord(
        doc_id=doc_id,
        text=f"borrower document {doc_id} " * 200,
        extraction_method="text",
        account_ids=[account] if account else [],
        mentioned_txn_ids=[],
        version_status="active",
        doc_type=doc_type,
    )


def test_regex_blind_borrower_doc_falls_through_to_llm(tmp_path):
    """Doc that regex saw as 'noise' but has an ACC-XXXX must reach the LLM
    and adopt the type the model returns."""
    fake = FakeClient({"doc_type": "agreement", "version_status": "active"})
    record = _record("blind-agreement")
    records = {"blind-agreement": record}

    doc_classify.apply_llm_fallback(
        records,
        _settings(tmp_path),
        client_factory=lambda: fake,
    )

    assert record.doc_type == "agreement"
    assert record.version_status == "active"
    assert len(fake.calls) == 1
    # The excerpt sent to the LLM is capped at 1500 characters.
    assert len(fake.calls[0]["user"]) <= doc_classify.TEXT_EXCERPT_CHARS


def test_llm_result_is_cached_per_doc(tmp_path):
    fake = FakeClient({"doc_type": "kyc", "version_status": "active"})
    records = {"c1": _record("c1")}

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)
    # Second pass on fresh records must hit the workspace cache, not the client.
    fake_second = FakeClient({"doc_type": "noise", "version_status": "active"})
    records2 = {"c1": _record("c1")}
    doc_classify.apply_llm_fallback(records2, _settings(tmp_path), client_factory=lambda: fake_second)

    assert records2["c1"].doc_type == "kyc"
    assert fake_second.calls == []
    cached = json.loads((tmp_path / "doctypes" / "c1.json").read_text(encoding="utf-8"))
    assert cached == {"doc_type": "kyc", "version_status": "active"}


def test_hard_brake_when_candidates_exceed_threshold(tmp_path, monkeypatch):
    """41 candidates trip the safety brake — no model call, no cache write."""
    monkeypatch.setattr(doc_classify, "MAX_LLM_CANDIDATES", 40)
    records = {f"d{i}": _record(f"d{i}", account=f"ACC-{i:04d}") for i in range(41)}
    fake = FakeClient({"doc_type": "kyc", "version_status": "active"})

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    assert fake.calls == []
    assert all(record.doc_type == "noise" for record in records.values())
    assert (tmp_path / "errors.log").exists()
    log_text = (tmp_path / "errors.log").read_text(encoding="utf-8")
    assert "skipped LLM fallback" in log_text


def test_conflict_guard_keeps_noise_when_active_type_already_claimed(tmp_path):
    """If the account already has an active KYC, a second KYC claim is kept
    as noise so resolve_documents does not silently pick one at random."""
    active_kyc = _record("existing-kyc", doc_type="kyc")
    blind = _record("blind-kyc")
    records = {"existing-kyc": active_kyc, "blind-kyc": blind}
    fake = FakeClient({"doc_type": "kyc", "version_status": "active"})

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    assert active_kyc.doc_type == "kyc"
    assert blind.doc_type == "noise"


def test_no_api_key_makes_fallback_a_noop(tmp_path):
    fake = FakeClient({"doc_type": "kyc", "version_status": "active"})
    record = _record("blind")
    records = {"blind": record}

    doc_classify.apply_llm_fallback(
        records,
        _settings(tmp_path, key=""),
        client_factory=lambda: fake,
    )

    assert record.doc_type == "noise"
    assert fake.calls == []


def test_docs_without_account_are_not_candidates(tmp_path):
    fake = FakeClient({"doc_type": "kyc", "version_status": "active"})
    record = _record("no-account", account="")
    records = {"no-account": record}

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    assert record.doc_type == "noise"
    assert fake.calls == []


def test_consolidated_report_about_a_borrower_reaches_the_classifier(tmp_path):
    """Group-level audits carry no ACC but they do mention a borrower name
    together with the 'consolidated financial statements' marker. Bare
    name mention was too loose (internal borrower memos also start with
    the borrower name), so we require both signals together."""
    kyc = DocRecord(
        doc_id="kyc-1",
        text="Организация: Aktau Port Services JSC\nСчёт ACC-7801\nДосье «Знай своего клиента»",
        extraction_method="text",
        account_ids=["ACC-7801"],
        mentioned_txn_ids=[],
        version_status="active",
        doc_type="kyc",
    )
    group_report = DocRecord(
        doc_id="group-report",
        text=("CONSOLIDATED ANNUAL REPORT · SARYBEL ENERGY HOLDING JSC. Independent Auditor's Report. "
              "We have audited the consolidated financial statements of Sarybel Energy Holding JSC and its "
              "subsidiaries. The Group's operations include Aktau Port Services JSC as a subsidiary."),
        extraction_method="text",
        account_ids=[],
        mentioned_txn_ids=[],
        version_status="active",
        doc_type="noise",
    )
    records = {"kyc-1": kyc, "group-report": group_report}
    fake = FakeClient({"doc_type": "aup", "version_status": "active"})

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    assert group_report.doc_type == "aup"
    assert len(fake.calls) == 1


def test_borrower_memo_without_consolidated_marker_is_not_a_candidate(tmp_path):
    """The public set has 124 internal borrower memos that mention the
    borrower name in their header (project status reports, IT manuals,
    onboarding checklists). Without the consolidated marker they must
    stay 'noise' — flooding them into the classifier costs money and
    trips the safety brake."""
    kyc = DocRecord(
        doc_id="kyc-1",
        text="Организация: Aktau Port Services JSC\nСчёт ACC-7801",
        extraction_method="text",
        account_ids=["ACC-7801"],
        mentioned_txn_ids=[],
        version_status="active",
        doc_type="kyc",
    )
    memo = DocRecord(
        doc_id="internal-memo",
        text=("Aktau Port Services JSC · Q1 2025 · Внутренний · Отчёт о статусе проекта. "
              "Общий статус: жёлтый. Aktau Port Services JSC подчёркивает риски по срокам..."),
        extraction_method="text",
        account_ids=[],
        mentioned_txn_ids=[],
        version_status="active",
        doc_type="noise",
    )
    records = {"kyc-1": kyc, "memo": memo}
    fake = FakeClient({"doc_type": "kyc", "version_status": "active"})

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    assert memo.doc_type == "noise"
    assert fake.calls == []
