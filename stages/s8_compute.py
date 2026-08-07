"""Стадия 8: расчёт фактического значения и статуса ковенанта."""

from decimal import Decimal, InvalidOperation
from typing import Any, Iterable

import pandas as pd

from lib.currency import normalize_ledger_to_usd
from lib.adjustments import resolve_txn_ids
from models import Condition, CovenantResult, CovenantSpec, Expr


def _expr_from_output(raw: dict[str, Any]) -> Expr:
    operation = raw.get("op")
    if operation == "fact":
        fact_name = raw.get("fact_name")
        if not isinstance(fact_name, str):
            raise ValueError("fact expression requires a fact_name")
        return Expr(op="fact", fact_name=fact_name)
    if operation == "sum":
        role = raw.get("role")
        if not isinstance(role, str):
            raise ValueError("sum expression requires a string role")
        return Expr(op="sum", role=role)
    if operation == "const":
        return Expr(op="const", value=Decimal(str(raw["value"])))
    if operation not in {"add", "subtract", "divide", "max", "min"}:
        raise ValueError(f"unsupported expression operation: {operation}")
    arguments = raw.get("args")
    if not isinstance(arguments, list) or len(arguments) < 2:
        raise ValueError(f"{operation} expression requires at least two args")
    return Expr(op=operation, args=tuple(_expr_from_output(argument) for argument in arguments))


def _condition_from_output(raw: dict[str, Any] | None) -> Condition | None:
    if not raw or not isinstance(raw.get("expr"), dict):
        return None
    return Condition(
        expr=_expr_from_output(raw["expr"]),
        operator=raw["operator"],
        threshold=Decimal(str(raw["threshold"])),
        raw_text=str(raw.get("raw_text", "")),
    )


def specifications_from_extraction(scenario_id: str, extraction: dict[str, Any]) -> dict[str, CovenantSpec]:
    specifications: dict[str, CovenantSpec] = {}
    facts: dict[str, Decimal] = {}
    for fact in extraction.get("document_facts", []):
        if fact.get("value") is None:
            continue
        try:
            facts[str(fact["fact_name"])] = Decimal(str(fact["value"]))
        except InvalidOperation:
            # Haiku occasionally emits a placeholder string instead of a
            # number in a document_fact. Skip such facts so one bad entry
            # does not sink the whole scenario; a covenant that actually
            # needs that fact will fail cleanly later.
            continue
    for covenant in extraction.get("covenants", []):
        clause_id = str(covenant["clause_id"])
        specifications[clause_id] = CovenantSpec(
            scenario_id=scenario_id,
            clause_id=clause_id,
            value_expr=_expr_from_output(covenant["value_expr"]),
            operator=covenant["operator"],
            threshold=Decimal(str(covenant["threshold"])),
            applicability=_condition_from_output(covenant.get("applicability")),
            exception=_condition_from_output(covenant.get("exception")),
            role_descriptions=dict(covenant.get("role_descriptions", {})),
            facts=facts,
        )
    return specifications


def _roles(expr: Expr) -> set[str]:
    if expr.op == "sum" and expr.role:
        return {expr.role}
    return set().union(*(_roles(argument) for argument in expr.args)) if expr.args else set()


def _role_key(value: str) -> str:
    return "".join(character for character in value.lower() if character.isascii() and character.isalnum())


def _matching_target_roles(target: str, roles: set[str]) -> set[str]:
    target_key = _role_key(target)
    return {role for role in roles if target_key and target_key in _role_key(role)}


def _adjustment_txns(adjustment: dict[str, Any], ledger: pd.DataFrame) -> list[str]:
    return resolve_txn_ids(adjustment.get("match", {}), ledger)


def build_clause_selection(
    specification: CovenantSpec,
    selected_ids: dict[str, list[str]],
    ledger: pd.DataFrame,
    adjustments: list[dict[str, Any]],
) -> dict[str, list[Decimal]]:
    ledger = normalize_ledger_to_usd(ledger, adjustments)
    amounts = dict(zip(ledger["txn_id"].astype(str), ledger["amount"]))
    for adjustment in adjustments:
        if adjustment.get("type") != "amount_correction" or not adjustment.get("accepted", False):
            continue
        corrected = Decimal(str(adjustment["match"]["amount"])).copy_abs()
        for txn_id in _adjustment_txns(adjustment, ledger):
            amounts[txn_id] = corrected
    selection = {
        role: [Decimal(str(amounts[txn_id])).copy_abs() for txn_id in selected_ids.get(role, [])]
        for role in _roles(specification.value_expr)
    }
    target_roles = _roles(specification.value_expr)
    for adjustment in adjustments:
        if adjustment.get("type") != "reclassification" or not adjustment.get("accepted", False):
            continue
        roles = _matching_target_roles(str(adjustment.get("to_role", "")), target_roles)
        for txn_id in _adjustment_txns(adjustment, ledger):
            if txn_id not in amounts:
                continue
            value = Decimal(str(amounts[txn_id])).copy_abs()
            for role in roles:
                selection[role].append(value)
    return selection


def _values(selection: dict[str, Iterable[Decimal]], role: str | None) -> list[Decimal]:
    if role is None:
        raise ValueError("sum expression requires a role")
    return [Decimal(str(value)) for value in selection.get(role, [])]


def evaluate_expr(expr: Expr, selection: dict[str, Iterable[Decimal]], facts: dict[str, Decimal] | None = None, *, depth: int = 0) -> Decimal:
    if depth >= 5:
        raise ValueError("covenant expression is too deep")
    if expr.op == "sum":
        return sum(_values(selection, expr.role), Decimal("0"))
    if expr.op == "const":
        if expr.value is None:
            raise ValueError("const expression requires a value")
        return expr.value
    if expr.op == "fact":
        if not expr.fact_name or expr.fact_name not in (facts or {}):
            raise ValueError(f"document fact is missing: {expr.fact_name}")
        return facts[expr.fact_name]

    values = [evaluate_expr(argument, selection, facts, depth=depth + 1) for argument in expr.args]
    if expr.op == "add":
        return values[0] + values[1]
    if expr.op == "subtract":
        return values[0] - values[1]
    if expr.op == "divide":
        return values[0] / values[1]
    if expr.op == "max":
        return max(values)
    if expr.op == "min":
        return min(values)
    raise ValueError(f"unsupported expression operation: {expr.op}")


def _matches(actual: Decimal, operator: str, threshold: Decimal) -> bool:
    return {
        "<=": actual <= threshold,
        ">=": actual >= threshold,
        "<": actual < threshold,
        ">": actual > threshold,
        "==": actual == threshold,
    }[operator]


def _condition_matches(condition: Condition, selection: dict[str, Iterable[Decimal]]) -> bool:
    return _matches(evaluate_expr(condition.expr, selection), condition.operator, condition.threshold)


def evaluate_covenant(spec: CovenantSpec, selection: dict[str, Iterable[Decimal]]) -> CovenantResult:
    actual = evaluate_expr(spec.value_expr, selection, spec.facts)
    applicable = spec.applicability is None or _matches(evaluate_expr(spec.applicability.expr, selection, spec.facts), spec.applicability.operator, spec.applicability.threshold)
    excepted = spec.exception is not None and _matches(evaluate_expr(spec.exception.expr, selection, spec.facts), spec.exception.operator, spec.exception.threshold)
    status = "COMPLIANT" if not applicable or excepted or _matches(actual, spec.operator, spec.threshold) else "BREACH"
    return CovenantResult(actual=actual, status=status)
