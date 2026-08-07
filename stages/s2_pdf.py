import re
from config import get_settings
from lib.pdf import extract_text
from lib.text import normalize_identifiers
from models import DocRecord

ACC = re.compile(r"ACC-\d{4}")
TXN = re.compile(r"TXN-[A-Z0-9]+-\d+")
AR = re.compile(r"AR-\d{4}-\d{4}")
OWN_REPORT = re.compile(
    r"номер\s+(?:заключения|отч[её]та|ведомости)\s*(AR-\d{4}-\d{4})",
    re.IGNORECASE,
)


# Type markers. Order matters: audit_notes is checked before agreement, because
# audit notes reference the loan they audit. KYC and AUP markers are tight so a
# procedure manual that merely mentions "KYC" or "AUP" does not steal the type.
AUDIT_NOTES_PATTERNS = (
    # A CP1251-mangled run can split words: "аудитор ское дело". Keep the
    # tolerance the original regex had.
    r"аудитор\s*ское\s+дело",
    r"\baudit\s+notes?\b",
    r"\baudit\s+work[- ]?papers?\b",
)
AGREEMENT_PATTERNS = (
    r"договор\s+банковского\s+займа",
    r"кредитн\w+\s+(?:договор|соглашени)",
    r"договор\s+займа",
    r"\bloan\s+agreement\b",
)
KYC_PATTERNS = (
    # Real KYC files quote «Знай своего клиента» in the header. Compliance
    # procedure manuals mention "KYC" without this exact phrase, so it filters
    # the two apart.
    r"знай\s+свое\w*\s+клиент",
    r"надлежащая\s+проверка\s+клиент",
    r"customer\s+due\s+diligence",
    r"kyc\s+(?:dossier|file|record)",
)
AUP_PATTERNS = (
    r"отч[её]т\s+о\s+выполнении\s+согласованных\s+процедур",
    r"agreed[-\s]upon\s+procedures",
)

# Version markers.
DRAFT_PATTERNS = (
    r"промежуточная\s+ведомость",
    r"заменена\s+окончательным\s+отч[её]том",
    r"\bdraft\b",
    r"\bnot\s+final\b",
    r"\bworking\s+(?:draft|copy)\b",
    r"\bpreliminary\s+version\b",
)
SUPERSEDED_PATTERNS = (
    r"недействующая\s+редакция",
    r"не\s+применяется",
    r"утратил\s+силу",
    r"\bsuperseded\b",
    r"\bsuperseded\s+by\b",
    r"\breplaced\s+by\b",
    r"\bno\s+longer\s+(?:in\s+)?(?:effect|effective)\b",
)


def _any(patterns: tuple[str, ...], text: str) -> bool:
    return any(re.search(pattern, text, re.IGNORECASE) for pattern in patterns)


def decode_legacy_pdf_text(text: str) -> str:
    """Repair the CP1251 text emitted as Latin-1 glyphs by these PDFs."""
    # A few documents contain genuine Kazakh Unicode characters among the
    # legacy glyphs.  Decode only Latin-1 runs, leaving those characters intact.
    parts = re.split(r"([^\x00-\xff]+)", text)
    return "".join(
        part if index % 2 else part.encode("latin-1").decode("cp1251")
        for index, part in enumerate(parts)
    )


def _version_status(text: str) -> str:
    if _any(DRAFT_PATTERNS, text):
        return "draft"
    if _any(SUPERSEDED_PATTERNS, text):
        return "superseded"
    return "active"


def _document_type(text: str, report_number: str | None) -> str:
    if _any(AUDIT_NOTES_PATTERNS, text):
        return "audit_notes"
    if _any(AGREEMENT_PATTERNS, text):
        return "agreement"
    if _any(KYC_PATTERNS, text):
        return "kyc"
    if report_number and _any(AUP_PATTERNS, text):
        return "aup"
    return "noise"


def run(state=None):
    """Regex-only triage. LLM fallback for unnamed borrower docs is applied
    separately from run_pipeline via lib.doc_classify.apply_llm_fallback so
    that unit tests calling build_document_index() do not fire the API."""
    records: dict[str, DocRecord] = {}
    for path in get_settings().data_dir.joinpath("documents").glob("*.pdf"):
        raw_text, method = extract_text(path)
        text = normalize_identifiers(decode_legacy_pdf_text(raw_text))
        own_match = OWN_REPORT.search(text)
        report_number = own_match.group(1) if own_match else None
        # A report's own number is not a reference to itself.
        references = [number for number in AR.findall(text) if number != report_number]
        records[path.stem] = DocRecord(
            doc_id=path.stem,
            text=text,
            extraction_method=method,
            account_ids=ACC.findall(text),
            mentioned_txn_ids=TXN.findall(text),
            report_number=report_number,
            references=references,
            version_status=_version_status(text),
            doc_type=_document_type(text, report_number),
        )
    return records
