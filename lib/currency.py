"""Deterministic currency normalization from audited FX facts."""

from __future__ import annotations

from decimal import Decimal, ROUND_HALF_UP
from typing import Any

import pandas as pd


def _party_key(value: object) -> str:
    return "".join(character for character in str(value).casefold() if character.isalnum())


def _matches(adjustment: dict[str, Any], row: pd.Series) -> bool:
    match = adjustment.get("match", {})
    txn_id = match.get("txn_id")
    if txn_id:
        return str(txn_id) == str(row["txn_id"])
    counterparty = match.get("counterparty")
    return bool(counterparty) and _party_key(counterparty) == _party_key(row.get("counterparty", ""))


def normalize_ledger_to_usd(ledger: pd.DataFrame, adjustments: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a copy with every *audited* foreign-currency fact shown in USD.

    A currency is changed only when stage 6 extracted an explicit rate from the
    resolved audit documents.  Unknown rates remain untouched rather than being
    guessed or silently treated as USD.
    """
    normalized = ledger.copy()
    normalized["amount"] = normalized["amount"].astype(object)
    if "currency" not in normalized:
        normalized["currency"] = "USD"
    for index, row in normalized.iterrows():
        currency = str(row["currency"]).upper()
        if currency == "USD" or pd.isna(row["amount"]):
            continue
        rates: list[Decimal] = []
        for adjustment in adjustments:
            if adjustment.get("type") != "fx_rate" or not _matches(adjustment, row):
                continue
            derived = adjustment.get("derived_from", {})
            if str(derived.get("foreign_currency", "")).upper() != currency:
                continue
            foreign_amount = Decimal(str(derived["foreign_amount"]))
            usd_amount = Decimal(str(derived["usd_amount"]))
            if foreign_amount <= 0:
                raise ValueError(f"invalid audited foreign amount for {row['txn_id']}")
            rates.append(usd_amount / foreign_amount)
        if len(rates) > 1:
            raise ValueError(f"multiple audited FX rates match {row['txn_id']}")
        if not rates:
            continue
        normalized.at[index, "amount"] = (
            Decimal(str(row["amount"])) * rates[0]
        ).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP)
        normalized.at[index, "currency"] = "USD"
    return normalized
