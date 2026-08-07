"""Stage 7: select ledger transactions once for a borrower."""

from __future__ import annotations

import json
import os
import re
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd

from lib.currency import normalize_ledger_to_usd
from lib.adjustments import resolve_txn_ids


PROMPT_PATH = Path(__file__).resolve().parents[1] / "prompts" / "stage7_select.md"


def _scenario_mask(ledger: pd.DataFrame, scenario_id: str) -> pd.Series:
    """Rows whose txn_id contains scenario_id as a delimited token."""
    pattern = re.compile(rf"(?<![A-Za-z0-9]){re.escape(scenario_id)}(?![A-Za-z0-9])")
    return ledger["txn_id"].astype(str).str.contains(pattern, regex=True)


@dataclass(frozen=True)
class SelectionResult:
    output: dict[str, Any]
    usage: dict[str, int]
    soft_warnings: list[str]


def selection_schema(extraction: dict[str, Any]) -> dict[str, Any]:
    properties: dict[str, Any] = {
        "uncertain": {"type": "array", "items": {"type": "object"}},
    }
    required: list[str] = []
    for covenant in extraction.get("covenants", []):
        clause_id = str(covenant["clause_id"])
        roles = list(covenant.get("role_descriptions", {}))
        properties[clause_id] = {
            "type": "object",
            "properties": {role: {"type": "array", "items": {"type": "string"}} for role in roles},
            "required": roles,
            "additionalProperties": False,
        }
        required.append(clause_id)
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


def build_context(*, extraction: dict[str, Any], ledger: pd.DataFrame) -> str:
    covenants = extraction.get("covenants", [])
    normalized_ledger = normalize_ledger_to_usd(ledger, extraction.get("adjustments", []))
    return "\n".join(
        [
            "<covenants>", json.dumps(covenants, ensure_ascii=False), "</covenants>",
            "<related_parties>", json.dumps(filtered_related_parties(extraction), ensure_ascii=False), "</related_parties>",
            "<adjustments>", json.dumps(extraction.get("adjustments", []), ensure_ascii=False), "</adjustments>",
            "<transactions>", normalized_ledger.to_json(orient="records", force_ascii=False), "</transactions>",
        ]
    )


def _party_key(value: str) -> str:
    return re.sub(r"[^\w]", "", value, flags=re.UNICODE).casefold()


def filtered_related_parties(extraction: dict[str, Any]) -> dict[str, Any]:
    related = deepcopy(extraction.get("related_parties", {}))
    try:
        threshold = Decimal(str(related["threshold_percent"]))
    except (KeyError, InvalidOperation):
        related["entities"] = []
        return related
    related["entities"] = [
        entity
        for entity in related.get("entities", [])
        if Decimal(str(entity.get("share_percent", "-1"))) >= threshold
    ]
    return related


def _eligible_party_keys(extraction: dict[str, Any]) -> set[str]:
    return {_party_key(str(entity["name"])) for entity in filtered_related_parties(extraction).get("entities", [])}


def _role_descriptions(extraction: dict[str, Any]) -> dict[str, str]:
    return {
        role: description
        for covenant in extraction.get("covenants", [])
        for role, description in covenant.get("role_descriptions", {}).items()
    }


def _selected_ids(output: dict[str, Any]):
    for clause_id, roles in output.items():
        if clause_id == "uncertain" or not isinstance(roles, dict):
            continue
        for role, txn_ids in roles.items():
            if isinstance(txn_ids, list):
                yield clause_id, role, txn_ids


def validate_selection_shape(output: dict[str, Any]) -> None:
    for clause_id, roles in output.items():
        if clause_id == "uncertain":
            if not isinstance(roles, list):
                raise ValueError("uncertain must be a list")
            continue
        if not isinstance(roles, dict):
            raise ValueError(f"{clause_id} must map roles to transaction ID lists")
        for role, txn_ids in roles.items():
            if not isinstance(txn_ids, list) or not all(isinstance(txn_id, str) for txn_id in txn_ids):
                raise ValueError(f"{clause_id}/{role} must contain only transaction ID strings")


_DIRECTION_OUTGOING = (
    r"в\s+пользу",
    r"paid\s+to",
    r"payment\s+to",
    r"payments?\s+to",
    r"distributed\s+to",
    r"payable\s+to",
    r"\boutgoing\b",
    r"restricted\s+payment",
    r"дивиденд",
    r"исходящ",
)
_DIRECTION_INCOMING = (
    r"поступлен",
    r"receipts?",
    r"received\s+from",
    r"\bincoming\b",
    r"выручк",
    r"revenue\s+from",
    r"proceeds\s+from",
)


def _direction_hint(description: str) -> str | None:
    text = description.lower()
    if any(re.search(pattern, text) for pattern in _DIRECTION_OUTGOING):
        return "outgoing"
    if any(re.search(pattern, text) for pattern in _DIRECTION_INCOMING):
        return "incoming"
    return None


def clause_semantic_errors(
    output: dict[str, Any],
    extraction: dict[str, Any],
    ledger: pd.DataFrame | None = None,
) -> dict[str, str]:
    """Per-clause semantic problems that are safe to salvage around.

    Returns ``{clause_id: reason}`` for clauses whose selection is
    self-inconsistent:
      * uncertain-substitution: the model listed a transaction under
        ``uncertain`` for a role that the clause never actually filled,
        or a required role is empty despite the model itself flagging a
        candidate for it;
      * direction mismatch: the role description names outgoing payments
        (в пользу / paid to / …) yet a selected txn has amount > 0
        (income), or vice versa. Python does not decide which txn is
        right — the mismatch surfaces as a semantic error and travels
        the same clause-level retry path as uncertain-substitution.
    Clauses not in the returned map are semantically OK; shape errors
    are a separate concern (see ``validate_selection_shape``).
    """
    errors: dict[str, str] = {}
    uncertain_items = [
        (str(item.get("txn_id")), str(item.get("role")))
        for item in output.get("uncertain", [])
        if isinstance(item, dict) and isinstance(item.get("txn_id"), str) and isinstance(item.get("role"), str)
    ]
    uncertain_roles = {role for _, role in uncertain_items}

    ledger_amounts: dict[str, Any] = {}
    if ledger is not None and not ledger.empty:
        ledger_amounts = dict(zip(ledger["txn_id"].astype(str), ledger["amount"]))

    covenants_by_clause = {str(c.get("clause_id")): c for c in extraction.get("covenants", [])}
    for clause_id, covenant in covenants_by_clause.items():
        roles = output.get(clause_id)
        if not isinstance(roles, dict):
            continue
        role_descriptions = covenant.get("role_descriptions", {}) or {}
        # Check 1: empty required role despite uncertain candidate for it.
        for role in role_descriptions:
            if roles.get(role):
                continue
            if role in uncertain_roles:
                errors[clause_id] = f"role {role!r} is empty despite an uncertain candidate for it"
                break
        if clause_id in errors:
            continue
        # Check 2: uncertain[txn, role] where role belongs to this clause
        # but the txn was never placed into it.
        for txn_id, role in uncertain_items:
            if role not in role_descriptions:
                continue
            if txn_id not in roles.get(role, []):
                errors[clause_id] = f"uncertain transaction {txn_id} for role {role!r} was not placed here"
                break
        if clause_id in errors:
            continue
        # Check 3: direction mismatch (role description names outgoing /
        # incoming payments but a selected txn has the opposite sign).
        for role, txn_ids in roles.items():
            desc = role_descriptions.get(role, "")
            hint = _direction_hint(desc)
            if hint is None or not isinstance(txn_ids, list):
                continue
            for txn_id in txn_ids:
                amount = ledger_amounts.get(str(txn_id))
                if amount is None:
                    continue
                try:
                    amount_value = float(amount)
                except (TypeError, ValueError):
                    continue
                if amount_value != amount_value:  # NaN
                    continue
                if hint == "outgoing" and amount_value > 0:
                    errors[clause_id] = (
                        f"role {role!r} describes outgoing payments but selected "
                        f"transaction {txn_id} has positive amount ({amount_value:+.2f})"
                    )
                    break
                if hint == "incoming" and amount_value < 0:
                    errors[clause_id] = (
                        f"role {role!r} describes incoming receipts but selected "
                        f"transaction {txn_id} has negative amount ({amount_value:+.2f})"
                    )
                    break
            if clause_id in errors:
                break
    return errors


def reject_uncertain_substitution(output: dict[str, Any], extraction: dict[str, Any]) -> None:
    """Back-compat wrapper: raise on any per-clause semantic error.

    Kept for tests that assert the exception path; the orchestrator
    (``select_scenario``) has moved to calling ``clause_semantic_errors``
    directly so it can salvage the valid clauses.
    """
    errors = clause_semantic_errors(output, extraction)
    if not errors:
        return
    clause_id, reason = next(iter(errors.items()))
    if "empty despite an uncertain candidate" in reason:
        role = reason.split("'")[1]
        raise ValueError(f"{clause_id}/{role} is empty despite uncertain candidates")
    if "was not placed here" in reason:
        parts = reason.split()
        txn_id = parts[2]
        role = parts[5].strip("'")
        raise ValueError(f"uncertain transaction {txn_id} is not selected for role {role}")
    raise ValueError(f"{clause_id}: {reason}")


def _words(value: str) -> set[str]:
    return {word.lower() for word in re.findall(r"[\w-]{4,}", value, flags=re.UNICODE)}


def _resolve_adjustment_txns(adjustment: dict[str, Any], ledger: pd.DataFrame) -> list[str]:
    return resolve_txn_ids(adjustment.get("match", {}), ledger)


def _target_roles(to_role: str, roles: dict[str, Any]) -> list[str]:
    expected = re.sub(r"[^a-z0-9]", "", to_role.lower())
    return [
        role
        for role in roles
        if expected == re.sub(r"[^a-z0-9]", "", role.lower())
        or expected in re.sub(r"[^a-z0-9]", "", role.lower())
    ]


def apply_deterministic_guards(*, output: dict[str, Any], ledger: pd.DataFrame, extraction: dict[str, Any]) -> tuple[dict[str, Any], list[str]]:
    cleaned = deepcopy(output)
    allowed_parties = _eligible_party_keys(extraction)
    counterparties = dict(zip(ledger["txn_id"].astype(str), ledger.get("counterparty", pd.Series(dtype=str)).astype(str)))
    for _, role, txn_ids in _selected_ids(cleaned):
        if role == "related_party":
            txn_ids[:] = [txn_id for txn_id in txn_ids if _party_key(counterparties.get(txn_id, "")) in allowed_parties]

    warnings: list[str] = []
    for adjustment in extraction.get("adjustments", []):
        if adjustment.get("type") != "reclassification":
            continue
        target = str(adjustment.get("to_role", ""))
        for txn_id in _resolve_adjustment_txns(adjustment, ledger):
            for clause_id, roles in cleaned.items():
                if clause_id == "uncertain" or not isinstance(roles, dict):
                    continue
                for role in _target_roles(target, roles):
                    if txn_id in roles.get(role, []):
                        roles[role].remove(txn_id)
                        warnings.append(f"{txn_id}: model applied reclassification, removed from target role {role}")
    return cleaned, warnings


def validate_selection(*, scenario_id: str, output: dict[str, Any], ledger: pd.DataFrame, extraction: dict[str, Any]) -> list[str]:
    available = set(ledger["txn_id"].astype(str))
    scenario_ids = set(ledger.loc[_scenario_mask(ledger, scenario_id), "txn_id"].astype(str))
    selected = {txn_id for _, _, txn_ids in _selected_ids(output) for txn_id in txn_ids}
    descriptions = dict(zip(ledger["txn_id"].astype(str), ledger.get("description", pd.Series(dtype=str)).astype(str)))
    role_descriptions = _role_descriptions(extraction)
    warnings: list[str] = []
    for clause_id, role, txn_ids in _selected_ids(output):
        if len(txn_ids) > 8:
            warnings.append(f"{clause_id}/{role}: role has more than 8 transactions")
        if not txn_ids:
            warnings.append(f"{clause_id}/{role}: role is empty")
        for txn_id in txn_ids:
            if txn_id not in available:
                raise ValueError(f"selected transaction does not exist in ledger: {txn_id}")
            if txn_id not in scenario_ids:
                raise ValueError(f"selected transaction does not belong to scenario {scenario_id}: {txn_id}")
            expected_words = _words(role_descriptions.get(role, role))
            if expected_words and not (_words(descriptions[txn_id]) & expected_words):
                warnings.append(f"{clause_id}/{role}: weak description overlap for {txn_id}")
    if "currency" in ledger:
        for row in ledger[["txn_id", "currency"]].itertuples(index=False):
            if str(row.currency).upper() != "USD" and str(row.txn_id) not in selected:
                warnings.append(f"{row.txn_id} in {row.currency} was not selected for any role")
    return warnings


def _write_cache(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent, delete=False) as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.write("\n")
        temporary = file.name
    os.replace(temporary, path)


def select_scenario(
    *,
    scenario_id: str,
    extraction: dict[str, Any],
    ledger: pd.DataFrame,
    client: Any,
    cache_dir: Path,
    select_mode: str = "scenario",
) -> SelectionResult:
    if select_mode != "scenario":
        raise ValueError("only SELECT_MODE=scenario is supported")
    cache_path = cache_dir / f"{scenario_id}.json"
    scoped_ledger = ledger[_scenario_mask(ledger, scenario_id)].reset_index(drop=True)
    prompt_ledger = normalize_ledger_to_usd(scoped_ledger, extraction.get("adjustments", []))
    validation_ledger = normalize_ledger_to_usd(ledger, extraction.get("adjustments", []))
    if cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        validate_selection_shape(cached["output"])
        # Cached selections have already been salvaged (dropped clauses
        # were replaced with {}), so per-clause validation would only
        # produce warnings — skip it and reuse the cached output.
        output, warnings = apply_deterministic_guards(output=cached["output"], ledger=prompt_ledger, extraction=extraction)
        return SelectionResult(output=output, usage=cached["usage"], soft_warnings=cached["soft_warnings"] + warnings)

    prompt_user = build_context(extraction=extraction, ledger=prompt_ledger)

    # Two failure classes, both bounded to one retry (total ≤ 2 calls):
    #   * shape errors (validate_selection_shape) — the whole response is
    #     unusable; retry with the validation error text fed back;
    #   * per-clause semantic errors (uncertain-substitution, direction
    #     mismatch) — retry only the offending clauses with a NEUTRAL
    #     repair note that asks the model to resolve the inconsistency,
    #     not to make a specific decision about which txn is right.
    # If the second attempt is still broken, keep the existing salvage
    # (drop only the still-broken clauses to {}). Python does not decide
    # which transaction is correct — that stays with the model.
    attempts: list[dict[str, Any]] = []
    last_response = None
    semantic_retry_used = False
    for attempt in range(2):
        user_prompt = prompt_user
        if attempts:
            last = attempts[-1]
            error_text = last["error"]
            repair_note = last.get("repair_note", error_text)
            user_prompt = (
                prompt_user
                + "\n\n<repair>Previous output failed validation:\n"
                + repair_note
                + "\nReturn a corrected selection. Do not change unrelated clauses.\n</repair>"
            )
        response = client.create_structured_message(
            system=PROMPT_PATH.read_text(encoding="utf-8"),
            user=user_prompt,
            tool_name="emit_selection",
            input_schema=selection_schema(extraction),
        )
        last_response = response
        try:
            validate_selection_shape(response.output)
        except (ValueError, AttributeError, TypeError, KeyError) as error:
            attempts.append({
                "output": response.output,
                "error": f"shape: {type(error).__name__}: {error}",
                "kind": "shape",
            })
            continue

        semantic = clause_semantic_errors(response.output, extraction, prompt_ledger)
        if semantic and not semantic_retry_used:
            attempts.append({
                "output": response.output,
                "error": "semantic: " + "; ".join(f"{cid}: {reason}" for cid, reason in semantic.items()),
                "kind": "semantic",
                "repair_note": _semantic_repair_note(semantic),
            })
            semantic_retry_used = True
            continue

        salvaged_output, salvage_warnings = _salvage_clauses(response.output, extraction, prompt_ledger)
        output, guard_warnings = apply_deterministic_guards(
            output=salvaged_output, ledger=prompt_ledger, extraction=extraction
        )
        retry_notes: list[str] = []
        for prior in attempts:
            kind = prior.get("kind", "unknown")
            retry_notes.append(f"stage7 accepted on retry after {kind}: {prior['error']}")
        result = SelectionResult(
            output=output,
            usage=response.usage,
            soft_warnings=retry_notes
            + salvage_warnings
            + guard_warnings
            + validate_selection(
                scenario_id=scenario_id,
                output=output,
                ledger=validation_ledger,
                extraction=extraction,
            ),
        )
        _write_cache(cache_path, {"output": result.output, "usage": result.usage, "soft_warnings": result.soft_warnings})
        return result

    # Two attempts, both structurally broken. Persist for inspection.
    reason = attempts[-1]["error"]
    _write_cache(
        cache_dir / "rejected" / f"{scenario_id}.json",
        {
            "output": last_response.output if last_response is not None else None,
            "usage": last_response.usage if last_response is not None else None,
            "reason": reason,
            "attempts": attempts,
        },
    )
    raise ValueError(reason)


def _semantic_repair_note(errors: dict[str, str]) -> str:
    """Neutral per-clause repair message. Asks the model to re-evaluate;
    does not tell it which transaction to include or exclude."""
    lines: list[str] = []
    for clause_id, reason in errors.items():
        lines.append(
            f"Clause {clause_id} is internally inconsistent: {reason}. "
            f"Re-evaluate this clause only and return a self-consistent "
            f"selection. Include the transaction only if the covenant text "
            f"and transaction support it; otherwise exclude it and resolve "
            f"the uncertainty consistently."
        )
    return "\n".join(lines)


def _salvage_clauses(
    output: dict[str, Any],
    extraction: dict[str, Any],
    ledger: pd.DataFrame | None = None,
) -> tuple[dict[str, Any], list[str]]:
    """Return (output_with_broken_clauses_dropped, warnings).

    Clauses with per-clause semantic errors are replaced by an empty
    dict — downstream ``run_pipeline`` sees the empty selection and
    leaves the cell at the stage-0 baseline instead of crashing the
    whole scenario.
    """
    errors = clause_semantic_errors(output, extraction, ledger)
    if not errors:
        return output, []
    salvaged = deepcopy(output)
    warnings: list[str] = []
    for clause_id, reason in errors.items():
        salvaged[clause_id] = {}
        warnings.append(f"stage7 salvage: {clause_id} dropped ({reason})")
    return salvaged, warnings
