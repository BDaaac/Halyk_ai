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
