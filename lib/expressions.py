"""Детерминированный исполнитель арифметических деревьев ковенантов."""

from dataclasses import dataclass
from decimal import Decimal
from typing import Iterable


MAX_EXPRESSION_DEPTH = 5


class ExpressionTooDeep(Exception):
    """Выражение превышает безопасную глубину рекурсии."""


@dataclass(frozen=True)
class Expression:
    operation: str
    operands: tuple["Expression | Decimal", ...]


def _decimal(value: Decimal | int | str) -> Decimal:
    return value if isinstance(value, Decimal) else Decimal(str(value))


def sum_(values: Iterable[Expression | Decimal | int | str]) -> Expression:
    return Expression("sum", tuple(_decimal(value) if not isinstance(value, Expression) else value for value in values))


def add(left: Expression | Decimal, right: Expression | Decimal) -> Expression:
    return Expression("add", (left, right))


def subtract(left: Expression | Decimal, right: Expression | Decimal) -> Expression:
    return Expression("subtract", (left, right))


def divide(left: Expression | Decimal, right: Expression | Decimal) -> Expression:
    return Expression("divide", (left, right))


def max_(left: Expression | Decimal, right: Expression | Decimal) -> Expression:
    return Expression("max", (left, right))


def evaluate(expression: Expression | Decimal | int | str, *, _depth: int = 0) -> Decimal:
    if _depth >= MAX_EXPRESSION_DEPTH:
        raise ExpressionTooDeep(f"expression depth exceeds {MAX_EXPRESSION_DEPTH - 1}")
    if not isinstance(expression, Expression):
        return _decimal(expression)

    values = [evaluate(operand, _depth=_depth + 1) for operand in expression.operands]
    if expression.operation == "sum":
        return sum(values, Decimal("0"))
    if expression.operation == "add":
        return values[0] + values[1]
    if expression.operation == "subtract":
        return values[0] - values[1]
    if expression.operation == "divide":
        return values[0] / values[1]
    if expression.operation == "max":
        return max(values)
    raise ValueError(f"unsupported expression operation: {expression.operation}")
