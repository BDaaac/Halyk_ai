from config import get_settings
from lib.pdf import extract_text, low_text_page_indices
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


def test_embedded_audit_table_is_routed_to_page_vision():
    # The P4 audit note has normal selectable text overall, but its adjustment
    # table lives on a page without a usable text layer.
    assert 3 in low_text_page_indices(DATA / "documents" / "2ed0b2ee4b57.pdf")


def test_normalization_preserves_non_identifier_prose():
    source = "ﬁнансовые условия не меняются без согласия сторон"

    assert normalize_identifiers(source) == source
