"""Стадия 9: evidence через ограниченный контрфактический пул."""

import os
from collections.abc import Callable, Iterable
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from lib.adjustments import resolve_txn_ids


def evidence_candidates(
    selected_roles: dict[str, list[str]],
    adjustments: list[dict],
    ledger: pd.DataFrame,
    *,
    clause_roles: Iterable[str],
) -> list[str]:
    """Return candidates for one coherent counterfactual question.

    A documented accepted adjustment takes precedence, but only when it
    actually touches a role this clause uses (``from_role`` or ``to_role``
    intersects ``clause_roles``). An adjustment about ``opex`` says nothing
    about a ``related_party`` clause; letting it override the candidate
    pool would replace the real evidence with an unrelated ledger row.
    When no accepted adjustment is role-relevant, fall back to the
    selected transactions and test each as ``drop_txn``.
    """
    roles = set(clause_roles)
    selected = [txn_id for ids in selected_roles.values() for txn_id in ids]
    adjusted = [
        txn_id
        for adjustment in adjustments
        if adjustment.get("accepted", True) is not False
        and (
            adjustment.get("from_role") in roles
            or adjustment.get("to_role") in roles
        )
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
    trace_scope: str | None = None,
) -> str | None:
    """Return the txn_id whose counterfactual flips the verdict.

    Contract:
      * COMPLIANT base status → None (no evidence needed).
      * Exactly one flipper → that txn_id.
      * Multiple flippers → deterministic scoring fallback: the
        lexicographically-smallest flipper, so the same input always
        produces the same output. The fallback is logged to
        ``workspace/errors.log`` as ``AMBIGUOUS_EVIDENCE …``; this is a
        SCORING fallback, not a claim that the returned txn is
        semantically the sole evidence for the breach.
      * Zero flippers → None.
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

    if not flippers:
        return None
    if len(flippers) == 1:
        return flippers[0]
    ordered = sorted(flippers)
    fallback = ordered[0]
    _log_ambiguous_evidence(trace_scope, ordered, fallback)
    return fallback


def _log_ambiguous_evidence(scope: str | None, candidates: list[str], fallback: str) -> None:
    """Trace tuning: ambiguous evidence is a scoring fallback, log every
    occurrence so an operator can see the underlying non-uniqueness."""
    workspace = Path(os.getenv("WORKSPACE_DIR", "workspace"))
    log_path = workspace / "errors.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    prefix = f"{scope}: " if scope else ""
    line = (
        f"{prefix}AMBIGUOUS_EVIDENCE candidates={candidates} "
        f"fallback={fallback} reason=deterministic scoring fallback\n"
    )
    with log_path.open("a", encoding="utf-8") as log:
        log.write(line)
