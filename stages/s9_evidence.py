"""Стадия 9: evidence через ограниченный контрфактический пул."""

from collections.abc import Callable, Iterable
from copy import deepcopy
from typing import Any, Literal

import pandas as pd

from lib.adjustments import resolve_txn_ids


def evidence_candidates(selected_roles: dict[str, list[str]], adjustments: list[dict], ledger: pd.DataFrame) -> list[str]:
    """Return candidates for one coherent counterfactual question.

    A documented accepted adjustment takes precedence: its counterfactual is
    whether the audit treatment were rejected.  In its absence, a selected
    transaction is tested as ``drop_txn``.  Mixing these two questions makes
    unrelated ordinary ledger rows compete with the audited evidence.
    """
    selected = [txn_id for ids in selected_roles.values() for txn_id in ids]
    adjusted = [
        txn_id
        for adjustment in adjustments
        if adjustment.get("accepted", True) is not False
        for txn_id in resolve_txn_ids(adjustment.get("match", {}), ledger)
    ]
    return list(dict.fromkeys(adjusted or selected))


def reverted_adjustments(txn_id: str, adjustments: list[dict], ledger: pd.DataFrame) -> list[dict]:
    """Counterfactual: the audit adjustment matching ``txn_id`` was not accepted."""
    reverted = deepcopy(adjustments)
    for adjustment in reverted:
        if txn_id in resolve_txn_ids(adjustment.get("match", {}), ledger):
            adjustment["accepted"] = False
    return reverted


def counterfactual_kind(
    txn_id: str, adjustments: list[dict], ledger: pd.DataFrame
) -> Literal["drop_txn", "revert_adjustment"]:
    """Choose the fact being tested, without combining counterfactuals.

    An accepted audit adjustment is evidence about acceptance of that adjustment;
    dropping its ledger transaction answers a different question.  Combining both
    produces unrelated flippers and makes deterministic evidence ambiguous.
    """
    for adjustment in adjustments:
        if adjustment.get("accepted", True) is not False and txn_id in resolve_txn_ids(adjustment.get("match", {}), ledger):
            return "revert_adjustment"
    return "drop_txn"


def find_counterfactual_evidence(
    *,
    base_status: str,
    candidates: Iterable[str],
    recompute: Callable[[str], Any],
) -> str | None:
    """Возвращает единственный ID, удаление которого меняет вердикт.

    Доказательство не указывается для соблюдённого ковенанта или когда
    несколько кандидатов одинаково меняют результат.
    """
    if base_status == "COMPLIANT":
        return None

    flippers: list[str] = []
    for txn_id in dict.fromkeys(candidates):
        try:
            recomputed = recompute(txn_id)
        except (ArithmeticError, ValueError):
            # Reverting an amount_correction leaves the ledger row with the
            # original NaN, so the hard invariant in build_clause_selection
            # raises ValueError. Treat that as an undefined counterfactual —
            # the candidate simply doesn't flip the verdict — instead of
            # bubbling the failure up as if the primary compute broke.
            continue
        status = getattr(recomputed, "status", recomputed)
        if status != base_status:
            flippers.append(txn_id)
    return flippers[0] if len(flippers) == 1 else None
