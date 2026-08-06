"""Публичные границы будущего последовательного пайплайна."""

import json
import re
from decimal import Decimal
from functools import lru_cache

import pandas as pd

from config import get_settings
from lib.pdf import extract_text


def _read_ledger() -> pd.DataFrame:
    """Читает сумму числом: пустые значения остаются pandas NaN."""
    return pd.read_csv(get_settings().data_dir / "master_ledger_2025.csv")


def run_pipeline(*, stop_after_stage: int | None = None):
    raise NotImplementedError("stage 0")


def gap_scan():
    ledger = _read_ledger()
    known_currencies = {"USD", "KZT", "EUR", "RUB", "GBP", "CNY"}
    return {
        "missing_amount": ledger.loc[ledger["amount"].isna(), "txn_id"].tolist(),
        "zero_amount": ledger.loc[ledger["amount"] == 0, "txn_id"].tolist(),
        "unknown_currency": ledger.loc[~ledger["currency"].isin(known_currencies), "txn_id"].tolist(),
    }


@lru_cache(maxsize=1)
def build_document_index():
    from stages.s2_pdf import run
    return run()


def read_document_text(doc_id: str):
    from stages.s2_pdf import decode_legacy_pdf_text
    text, _ = extract_text(get_settings().data_dir / "documents" / f"{doc_id}.pdf")
    return decode_legacy_pdf_text(text)


def build_mapping(ledger: pd.DataFrame, *, template: dict) -> dict[str, str]:
    """Map accounts to the scenario IDs declared by the current template."""
    targets = set(template["answers"])
    scoped = ledger.copy()
    scoped["scenario_id"] = scoped["txn_id"].str.extract(r"^TXN-([^-]+)-", expand=False)
    scoped = scoped[scoped["scenario_id"].isin(targets)]
    if scoped.empty and targets:
        raise ValueError("no ledger transactions match scenario IDs in submission template")
    grouped = scoped.groupby("account_id")["scenario_id"].agg(lambda ids: sorted(set(ids)))
    conflicts = {account: ids for account, ids in grouped.items() if len(ids) != 1}
    if conflicts:
        raise ValueError(f"account_id maps to multiple scenario_id values: {conflicts}")
    return {account: scenarios[0] for account, scenarios in grouped.items()}


def get_account_to_scenario():
    template_path = get_settings().data_dir / "submission_template.json"
    template = json.loads(template_path.read_text(encoding="utf-8"))
    return build_mapping(_read_ledger(), template=template)


def parse_transaction_correction(text: str, txn_id: str) -> tuple[Decimal, str] | None:
    """Return the closest signed dollar amount associated with a transaction."""
    position = text.find(txn_id)
    if position < 0:
        return None
    start = max(0, position - 500)
    window = text[start : position + 500]
    candidates: list[tuple[int, Decimal, str]] = []
    for money in re.finditer(r"\$([\d,]+\.\d{2})", window):
        sign_context = window[max(0, money.start() - 120) : money.end() + 120].lower()
        if "\u0440\u0430\u0441\u0445\u043e\u0434" in sign_context:
            sign = "expense"
        elif "\u043f\u043e\u0441\u0442\u0443\u043f\u043b\u0435\u043d" in sign_context:
            sign = "income"
        else:
            continue
        absolute_position = start + money.start()
        candidates.append(
            (abs(absolute_position - position), Decimal(money.group(1).replace(",", "")), sign)
        )
    if not candidates:
        return None
    _, amount, sign = min(candidates, key=lambda candidate: candidate[0])
    return amount, sign


def get_corrections():
    from models import Correction, SourceRef

    result = {}
    for doc in build_document_index().values():
        for txn in doc.mentioned_txn_ids:
            parsed = parse_transaction_correction(doc.text, txn)
            if parsed is None:
                continue
            amount, sign = parsed
            result[txn] = Correction(
                correct_amount=amount,
                source=SourceRef(doc.doc_id),
                sign=sign,
            )
    return result


def resolve_documents(scenario_id: str):
    from models import DocumentSet
    account = next(account for account, scenario in get_account_to_scenario().items() if scenario == scenario_id)
    documents = list(build_document_index().values())
    own = [document for document in documents if account in document.account_ids]
    active = [document for document in own if document.version_status == "active"]
    agreements = [document for document in active if document.doc_type == "agreement"]
    kyc = [document for document in active if document.doc_type == "kyc"]
    report_numbers = {reference for document in own for reference in document.references}
    reports = [
        document
        for document in documents
        if document.doc_type == "aup"
        and document.version_status == "active"
        and document.report_number in report_numbers
    ]
    return DocumentSet(
        agreement=agreements[0].doc_id if agreements else None,
        kyc=kyc[0].doc_id if kyc else None,
        aup_report=reports[0].doc_id if reports else None,
        rejected=[document.doc_id for document in own if document.version_status != "active"],
    )


def get_specifications():
    raise NotImplementedError("stage 6")


def related_party_threshold(scenario_id: str):
    raise NotImplementedError("stage 6")


def related_parties(scenario_id: str):
    raise NotImplementedError("stage 8")


def run_scenario(scenario_id: str):
    raise NotImplementedError("stage 8")


def recompute_without(scenario_id: str, clause_id: str, txn_id: str):
    raise NotImplementedError("stage 9")
