from dataclasses import dataclass, field
from decimal import Decimal

@dataclass
class SourceRef: doc_id: str
@dataclass
class DocRecord:
    doc_id: str; text: str; extraction_method: str; account_ids: list[str]; mentioned_txn_ids: list[str]
    report_number: str | None = None; references: list[str] = field(default_factory=list); version_status: str = "unknown"; doc_type: str = "noise"
@dataclass
class Correction:
    correct_amount: Decimal; source: SourceRef; sign: str
@dataclass
class DocumentSet:
    agreement: str | None = None; kyc: str | None = None; aup_report: str | None = None; rejected: list[str] = field(default_factory=list)
