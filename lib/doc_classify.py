"""Level-2 LLM classifier for documents the regex triage could not name.

Only invoked when a document already carries a borrower account
(``ACC-XXXX``) but the regexes returned ``doc_type == "noise"``. Feeds
the model just the first 1500 characters and expects one of a small
enumerated set of types plus a version status.

Hard safety brake: if more than ``MAX_LLM_CANDIDATES`` documents fall
into the fallback in a single run, we log and skip the model entirely.
This protects against a whole-corpus encoding failure on a private set
that would otherwise fire hundreds of unbudgeted API calls.

Cache layout: ``workspace/doctypes/{doc_id}.json``.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
import re
from typing import Any, Iterable

from models import DocRecord


# Caps on documents that reach the LLM classifier in a single run.
#
# The old single global cap (40) fired all-or-nothing: on a corpus where
# candidates were evenly distributed across borrowers, hitting the cap by
# borrower_count * 2 was enough to disable classification entirely, even
# though no single borrower was pathological. The current caps separate
# three concerns:
#
#   MAX_LLM_CANDIDATES_PER_ACCOUNT — one borrower shouldn't dominate the
#     candidate list. If a single account contributes more than this, only
#     the first N of its docs are classified; the remainder stay as noise.
#
#   MAX_LLM_CANDIDATES_CONSOLIDATED — consolidated group reports (no ACC,
#     matched by borrower name) share a separate pool so a small handful
#     of legitimate group audits is not squeezed out by a large per-account
#     backlog and vice versa.
#
#   MAX_LLM_CANDIDATES_TOTAL — absolute safety brake against a corpus-wide
#     encoding failure that would surface hundreds of ACC-carrying "noise"
#     documents. Set well above realistic candidate counts.
MAX_LLM_CANDIDATES_PER_ACCOUNT = 5
MAX_LLM_CANDIDATES_CONSOLIDATED = 20
MAX_LLM_CANDIDATES_TOTAL = 200
TEXT_EXCERPT_CHARS = 1500

ALLOWED_DOC_TYPES = ("agreement", "kyc", "aup", "audit_notes", "noise")
ALLOWED_VERSION_STATUSES = ("active", "draft", "superseded")


CLASSIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "doc_type": {"type": "string", "enum": list(ALLOWED_DOC_TYPES)},
        "version_status": {"type": "string", "enum": list(ALLOWED_VERSION_STATUSES)},
    },
    "required": ["doc_type", "version_status"],
    "additionalProperties": False,
}


CLASSIFY_PROMPT = (
    "Ты классифицируешь один документ из папки заёмщика. Возможные типы:\n"
    "  agreement    — договор банковского займа, кредитный договор, loan agreement.\n"
    "  kyc          — досье конкретного клиента (Знай своего клиента / customer due diligence "
    "                 dossier) с его владельцами, долями, связанными сторонами. НЕ методическая "
    "                 процедура о том, как проводить KYC.\n"
    "  aup          — отчёт о выполнении согласованных процедур (agreed-upon procedures) "
    "                 либо промежуточная/черновая версия такого отчёта.\n"
    "  audit_notes  — аудиторское дело, рабочие заметки аудитора, audit workpapers.\n"
    "  noise        — всё остальное (служебные записки, отчёты о статусе проектов, "
    "                 методические процедуры, внутренние регламенты).\n"
    "\n"
    "Возможные версионные статусы:\n"
    "  active     — окончательная действующая редакция.\n"
    "  draft      — черновик / проект / промежуточная ведомость / не окончательная.\n"
    "  superseded — заменён другой редакцией / утратил силу / аннулирован.\n"
    "\n"
    "Верни ровно один JSON-объект с полями doc_type и version_status."
)


def _cache_path(workspace_dir: Path, doc_id: str) -> Path:
    directory = workspace_dir / "doctypes"
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{doc_id}.json"


def _write_atomically(path: Path, payload: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", dir=path.parent, delete=False
    ) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        temp = file.name
    os.replace(temp, path)


def classify_document(
    *,
    doc_id: str,
    text: str,
    client: Any,
    workspace_dir: Path,
) -> dict[str, str]:
    """Return {doc_type, version_status}. Cached per doc_id in workspace."""
    cache = _cache_path(workspace_dir, doc_id)
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))
    excerpt = text[:TEXT_EXCERPT_CHARS]
    response = client.create_structured_message(
        system=CLASSIFY_PROMPT,
        user=excerpt,
        tool_name="emit_document_type",
        input_schema=CLASSIFY_SCHEMA,
    )
    result = {
        "doc_type": response.output["doc_type"],
        "version_status": response.output["version_status"],
    }
    _write_atomically(cache, result)
    return result


def _log(workspace_dir: Path, message: str) -> None:
    log_path = workspace_dir / "errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"doc_classify: {message}\n")


def _existing_types_by_account(records: Iterable[DocRecord]) -> dict[str, set[str]]:
    """Active doc_types already claimed per account (kyc, agreement, etc.)."""
    claims: dict[str, set[str]] = {}
    for record in records:
        if record.version_status != "active" or record.doc_type == "noise":
            continue
        for account in record.account_ids:
            claims.setdefault(account, set()).add(record.doc_type)
    return claims


def _harvest_borrower_names(records: Iterable[DocRecord]) -> set[str]:
    """Names harvested from active KYC dossiers via the borrower-name regex
    shared with consolidated retrieval. Used to bring group-level documents
    (which carry no ACC-XXXX) into the LLM-classification candidate pool."""
    from lib.consolidated_retrieval import borrower_name

    names: set[str] = set()
    for record in records:
        if record.doc_type != "kyc" or record.version_status != "active":
            continue
        name = borrower_name(record.text)
        if name:
            names.add(name)
    return names


def _is_consolidated_about_borrower(text: str, names: set[str]) -> bool:
    """Consolidated group audit that also mentions one of our borrowers.

    Bare name mention was too loose: on the public set 124 internal
    borrower memos (weekly status reports, IT manuals, employee onboarding
    checklists) also start with the borrower's own name and would flood
    the classifier without adding useful signal. Requiring the
    'consolidated financial statements / annual report / statement of'
    marker narrows the extra candidates to actual group-level audit docs.
    """
    from lib.consolidated_retrieval import CONSOLIDATED_MARKER_RE

    if not names or CONSOLIDATED_MARKER_RE.search(text) is None:
        return False
    collapsed = re.sub(r"\s+", " ", text)
    return any(name in collapsed for name in names)


def apply_llm_fallback(
    records: dict[str, DocRecord],
    settings: Any,
    *,
    client_factory: Any = None,
) -> None:
    """Route unnamed borrower documents through the LLM classifier.

    A candidate is any doc the regex triage returned as ``noise`` that
    either carries an ``ACC-XXXX`` (a borrower's own paperwork) OR is a
    consolidated group report that mentions one of our borrower names
    (the group's audit of the subsidiary is relevant even though the
    document has no ACC of its own — see ``a5cc1400b640``).
    """
    borrower_names = _harvest_borrower_names(records.values())
    raw_candidates = [
        record
        for record in records.values()
        if record.doc_type == "noise"
        and (record.account_ids or _is_consolidated_about_borrower(record.text, borrower_names))
    ]
    if not raw_candidates:
        return

    # Separate per-account caps from a single consolidated pool: an ACC-linked
    # document belongs to one specific borrower and is capped alongside its
    # siblings; a consolidated group report has no ACC of its own and shares
    # a distinct pool so a borrower with many candidates does not starve out
    # legitimate group-level audits (and vice versa).
    from collections import defaultdict
    per_account: dict[str, int] = defaultdict(int)
    consolidated_count = 0
    candidates: list[DocRecord] = []
    for record in raw_candidates:
        if record.account_ids:
            if any(
                per_account[acc] >= MAX_LLM_CANDIDATES_PER_ACCOUNT
                for acc in record.account_ids
            ):
                _log(
                    settings.workspace_dir,
                    f"{record.doc_id}: per-account cap "
                    f"{MAX_LLM_CANDIDATES_PER_ACCOUNT} reached for "
                    f"{list(record.account_ids)}; kept as noise",
                )
                continue
            for acc in record.account_ids:
                per_account[acc] += 1
        else:
            if consolidated_count >= MAX_LLM_CANDIDATES_CONSOLIDATED:
                _log(
                    settings.workspace_dir,
                    f"{record.doc_id}: consolidated cap "
                    f"{MAX_LLM_CANDIDATES_CONSOLIDATED} reached; kept as noise",
                )
                continue
            consolidated_count += 1
        candidates.append(record)

    if len(candidates) > MAX_LLM_CANDIDATES_TOTAL:
        _log(
            settings.workspace_dir,
            f"skipped LLM fallback — {len(candidates)} candidates exceed "
            f"corpus-wide safety brake {MAX_LLM_CANDIDATES_TOTAL}",
        )
        return
    if not settings.anthropic_api_key:
        return

    if client_factory is None:
        from lib.anthropic_client import AnthropicClient

        client_factory = lambda: AnthropicClient(
            api_key=settings.anthropic_api_key,
            model=settings.extract_model,
            timeout_seconds=settings.llm_timeout_seconds,
        )
    client = client_factory()

    claims = _existing_types_by_account(records.values())
    for record in candidates:
        try:
            result = classify_document(
                doc_id=record.doc_id,
                text=record.text,
                client=client,
                workspace_dir=settings.workspace_dir,
            )
        except Exception as error:
            _log(settings.workspace_dir, f"{record.doc_id}: {type(error).__name__}: {error}")
            continue

        new_type = result.get("doc_type", "noise")
        new_status = result.get("version_status", "active")
        # Do not clobber an already-classified active document of the same
        # account: two active KYCs (or two agreements) would confuse
        # resolve_documents, which just picks index 0. Safer to keep the
        # candidate as noise than to overwrite silently. Name-only
        # candidates have no ACC and cannot conflict at all.
        if new_type != "noise" and new_status == "active" and record.account_ids:
            account = record.account_ids[0]
            existing = claims.get(account, set())
            if new_type in existing:
                _log(
                    settings.workspace_dir,
                    f"{record.doc_id}: LLM said {new_type} but account "
                    f"{account} already has an active {new_type}; kept as noise",
                )
                continue
            claims.setdefault(account, set()).add(new_type)

        record.doc_type = new_type
        record.version_status = new_status
