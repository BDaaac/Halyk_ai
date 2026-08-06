from dataclasses import dataclass, field
from decimal import Decimal
from typing import Literal


@dataclass
class SourceRef:
    doc_id: str


@dataclass
class DocRecord:
    doc_id: str
    text: str
    extraction_method: str
    account_ids: list[str]
    mentioned_txn_ids: list[str]
    report_number: str | None = None
    references: list[str] = field(default_factory=list)
    version_status: str = "unknown"
    doc_type: str = "noise"


@dataclass
class Correction:
    correct_amount: Decimal
    source: SourceRef
    sign: str


@dataclass
class DocumentSet:
    agreement: str | None = None
    kyc: str | None = None
    aup_report: str | None = None
    audit_notes: str | None = None
    rejected: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Expr:
    """Ограниченное дерево расчёта, получаемое из стадии extract."""

    op: Literal["sum", "add", "subtract", "divide", "max", "min", "const", "fact"]
    role: str | None = None
    fact_name: str | None = None
    value: Decimal | None = None
    args: tuple["Expr", ...] = ()


@dataclass(frozen=True)
class Condition:
    expr: Expr
    operator: Literal["<=", ">=", "<", ">", "=="]
    threshold: Decimal
    raw_text: str


@dataclass(frozen=True)
class CovenantSpec:
    scenario_id: str
    clause_id: str
    value_expr: Expr
    operator: Literal["<=", ">=", "<", ">"]
    threshold: Decimal
    applicability: Condition | None = None
    exception: Condition | None = None
    role_descriptions: dict[str, str] = field(default_factory=dict)
    facts: dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True)
class CovenantResult:
    actual: Decimal
    status: Literal["COMPLIANT", "BREACH"]
