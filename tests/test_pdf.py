from config import get_settings
from lib.pdf import extract_text
from lib.text import normalize_identifiers


DATA = get_settings().data_dir


def test_pdf_extraction_real_file():
    text, method = extract_text(DATA / "documents" / "b38facf69a94.pdf")

    assert method == "text"
    assert len(text) > 1000
    assert "ACC-7201" in normalize_identifiers(text)


def test_scanned_pdf_routes_to_vision():
    _, method = extract_text(DATA / "documents" / "f3fa6d20c8a1.pdf")

    assert method == "vision"


def test_normalization_preserves_non_identifier_prose():
    source = "ﬁнансовые условия не меняются без согласия сторон"

    assert normalize_identifiers(source) == source
