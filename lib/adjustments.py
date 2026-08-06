"""Shared deterministic resolution of audit adjustment matches to ledger IDs."""

from decimal import Decimal
from typing import Any

import pandas as pd


def resolve_txn_ids(match: dict[str, Any], ledger: pd.DataFrame) -> list[str]:
    if match.get("txn_id"):
        return [str(match["txn_id"])]
    candidates = ledger.copy()
    if match.get("amount") is not None:
        amount = Decimal(str(match["amount"])).copy_abs()
        candidates = candidates[candidates["amount"].map(lambda value: Decimal(str(value)).copy_abs() == amount)]
    if match.get("counterparty"):
        candidates = candidates[
            candidates["counterparty"].astype(str).str.casefold() == str(match["counterparty"]).casefold()
        ]
    return candidates["txn_id"].astype(str).tolist()
