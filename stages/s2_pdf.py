import re
from config import get_settings
from lib.pdf import extract_text
from lib.text import normalize_identifiers
from models import DocRecord
ACC = re.compile(r"ACC-\d{4}")
TXN = re.compile(r"TXN-[A-Z0-9]+-\d+")
AR = re.compile(r"AR-\d{4}-\d{4}")
OWN_REPORT = re.compile(
    r"\u043d\u043e\u043c\u0435\u0440\s+(?:\u0437\u0430\u043a\u043b\u044e\u0447\u0435\u043d\u0438\u044f|\u043e\u0442\u0447[\u0435\u0451]\u0442\u0430|\u0432\u0435\u0434\u043e\u043c\u043e\u0441\u0442\u0438)\s*(AR-\d{4}-\d{4})",
    re.IGNORECASE,
)


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
    lower = text.lower()
    if re.search(r"\u043f\u0440\u043e\u043c\u0435\u0436\u0443\u0442\u043e\u0447\u043d\u0430\u044f\s+\u0432\u0435\u0434\u043e\u043c\u043e\u0441\u0442\u044c", lower) or re.search(
        r"\u0437\u0430\u043c\u0435\u043d\u0435\u043d\u0430\s+\u043e\u043a\u043e\u043d\u0447\u0430\u0442\u0435\u043b\u044c\u043d\u044b\u043c\s+\u043e\u0442\u0447[\u0435\u0451]\u0442\u043e\u043c", lower
    ):
        return "draft"
    if re.search(r"\u043d\u0435\u0434\u0435\u0439\u0441\u0442\u0432\u0443\u044e\u0449\u0430\u044f\s+\u0440\u0435\u0434\u0430\u043a\u0446\u0438\u044f|\u043d\u0435\s+\u043f\u0440\u0438\u043c\u0435\u043d\u044f\u0435\u0442\u0441\u044f", lower):
        return "superseded"
    return "active"


def _document_type(text: str, report_number: str | None) -> str:
    lower = text.lower()
    if re.search(r"\u0430\u0443\u0434\u0438\u0442\u043e\u0440\s*\u0441\u043a\u043e\u0435\s+\u0434\u0435\u043b\u043e", lower):
        return "audit_notes"
    if re.search(r"\u0434\u043e\u0433\u043e\u0432\u043e\u0440\s+\u0431\u0430\u043d\u043a\u043e\u0432\u0441\u043a\u043e\u0433\u043e\s+\u0437\u0430\u0439\u043c\u0430", lower):
        return "agreement"
    if re.search(r"\u0437\u043d\u0430\u0439\s+\u0441\u0432\u043e\u0435\u0433\u043e\s+\u043a\u043b\u0438\u0435\u043d\u0442\s*\u0430", lower):
        return "kyc"
    if report_number and re.search(r"\u043e\u0442\u0447[\u0435\u0451]\u0442\s+\u043e\s+\u0432\u044b\u043f\u043e\u043b\u043d\u0435\u043d\u0438\u0438\s+\u0441\u043e\u0433\u043b\u0430\u0441\u043e\u0432\u0430\u043d\u043d\u044b\u0445\s+\u043f\u0440\u043e\u0446\u0435\u0434\u0443\u0440", lower):
        return "aup"
    return "noise"


def run(state=None):
    records={}
    for path in get_settings().data_dir.joinpath('documents').glob('*.pdf'):
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
