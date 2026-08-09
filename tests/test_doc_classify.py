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


def test_corpus_wide_safety_brake_fires_at_pathological_total(tmp_path, monkeypatch):
    """A pathological corpus (e.g. encoding failure mislabelling hundreds of
    docs as noise-with-ACC) trips the corpus-wide brake — no model call,
    no cache write, all records stay noise."""
    monkeypatch.setattr(doc_classify, "MAX_LLM_CANDIDATES_TOTAL", 40)
    # 41 distinct accounts each with a single candidate — no per-account cap
    # hit, but the corpus total exceeds the safety brake.
    records = {f"d{i}": _record(f"d{i}", account=f"ACC-{i:04d}") for i in range(41)}
    fake = FakeClient({"doc_type": "kyc", "version_status": "active"})

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    assert fake.calls == []
    assert all(record.doc_type == "noise" for record in records.values())
    log_text = (tmp_path / "errors.log").read_text(encoding="utf-8")
    assert "corpus-wide safety brake" in log_text


def test_distributed_forty_candidates_all_classified(tmp_path):
    """Forty candidates spread across many accounts (well under any cap)
    must all reach the classifier — the previous global-40 brake fired
    here even though nothing was pathological."""
    records = {
        f"d{i}": _record(f"d{i}", account=f"ACC-{i:04d}")
        for i in range(40)
    }
    fake = FakeClient({"doc_type": "agreement", "version_status": "active"})

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    assert len(fake.calls) == 40
    assert all(record.doc_type == "agreement" for record in records.values())


def test_per_account_cap_trims_only_the_pathological_account(tmp_path):
    """One account with 20 noise-with-ACC candidates + 5 other accounts with
    1 each. The per-account cap trims the pathological account to
    MAX_LLM_CANDIDATES_PER_ACCOUNT API calls; healthy accounts each get
    one call. Downstream conflict-guard is orthogonal to this test — we
    assert the CAP is applied, using API call counts as the direct proof."""
    records = {}
    for i in range(20):
        records[f"pathological-{i}"] = _record(f"pathological-{i}", account="ACC-BAD")
    for i in range(5):
        records[f"healthy-{i}"] = _record(f"healthy-{i}", account=f"ACC-OK-{i:02d}")
    # audit_notes + draft avoids the "active type already claimed" conflict
    # guard so we can measure the cap in isolation.
    fake = FakeClient({"doc_type": "audit_notes", "version_status": "draft"})

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    expected_calls = doc_classify.MAX_LLM_CANDIDATES_PER_ACCOUNT + 5
    assert len(fake.calls) == expected_calls, (
        f"expected {expected_calls} API calls (cap {doc_classify.MAX_LLM_CANDIDATES_PER_ACCOUNT} "
        f"for ACC-BAD + 5 healthy), got {len(fake.calls)}"
    )
    log_text = (tmp_path / "errors.log").read_text(encoding="utf-8")
    assert "per-account cap" in log_text
    # None of the healthy accounts should show up in the skip log.
    assert "healthy-" not in log_text


def test_consolidated_candidates_not_capped_by_per_account_pool(tmp_path):
    """Six consolidated (no-ACC) group audits must not be squeezed out by a
    5-per-account cap: they share their own separate consolidated pool."""
    kyc = DocRecord(
        doc_id="kyc-anchor",
        text="Организация: Aktau Port Services JSC\nСчёт ACC-7801\nДосье «Знай своего клиента»",
        extraction_method="text",
        account_ids=["ACC-7801"],
        mentioned_txn_ids=[],
        version_status="active",
        doc_type="kyc",
    )
    records = {"kyc-anchor": kyc}
    for i in range(8):
        records[f"group-{i}"] = DocRecord(
            doc_id=f"group-{i}",
            text=(
                f"CONSOLIDATED ANNUAL REPORT · SARYBEL ENERGY HOLDING JSC {i}. "
                "Independent Auditor's Report on the consolidated financial statements "
                "of Sarybel Energy Holding JSC and its subsidiaries. The Group's "
                "operations include Aktau Port Services JSC as a subsidiary."
            ),
            extraction_method="text",
            account_ids=[],
            mentioned_txn_ids=[],
            version_status="active",
            doc_type="noise",
        )
    fake = FakeClient({"doc_type": "aup", "version_status": "active"})

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    classified = sum(1 for r in records.values() if r.doc_id.startswith("group-") and r.doc_type == "aup")
    assert classified == 8, f"consolidated docs must not be capped at 5 per-account: got {classified}"


def test_consolidated_pool_has_its_own_cap(tmp_path, monkeypatch):
    """When consolidated candidates exceed their dedicated cap, the excess is
    logged and the account-linked pool is unaffected."""
    monkeypatch.setattr(doc_classify, "MAX_LLM_CANDIDATES_CONSOLIDATED", 3)
    kyc = DocRecord(
        doc_id="kyc-anchor",
        text="Организация: Aktau Port Services JSC\nСчёт ACC-7801\nДосье «Знай своего клиента»",
        extraction_method="text",
        account_ids=["ACC-7801"],
        mentioned_txn_ids=[],
        version_status="active",
        doc_type="kyc",
    )
    records = {"kyc-anchor": kyc}
    for i in range(5):
        records[f"group-{i}"] = DocRecord(
            doc_id=f"group-{i}",
            text=(
                f"CONSOLIDATED ANNUAL REPORT · SARYBEL ENERGY HOLDING JSC {i}. "
                "Consolidated financial statements of Sarybel Energy Holding JSC "
                "and its subsidiaries. Aktau Port Services JSC subsidiary."
            ),
            extraction_method="text",
            account_ids=[],
            mentioned_txn_ids=[],
            version_status="active",
            doc_type="noise",
        )
    records["healthy"] = _record("healthy", account="ACC-9999")
    fake = FakeClient({"doc_type": "aup", "version_status": "active"})

    doc_classify.apply_llm_fallback(records, _settings(tmp_path), client_factory=lambda: fake)

    consolidated_classified = sum(
        1 for r in records.values() if r.doc_id.startswith("group-") and r.doc_type == "aup"
    )
    assert consolidated_classified == 3
    # ACC-linked candidate must not be affected by the consolidated cap.
    assert records["healthy"].doc_type == "aup"
    log_text = (tmp_path / "errors.log").read_text(encoding="utf-8")
    assert "consolidated cap" in log_text


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
