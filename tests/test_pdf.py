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


def test_version_status_recognises_english_draft_and_superseded_markers():
    """Private-set documents may carry English draft/superseded notes;
    the triage must not silently classify them as active."""
    from stages.s2_pdf import _version_status

    assert _version_status("DRAFT — not for distribution") == "draft"
    assert _version_status("Working draft, pending sign-off") == "draft"
    assert _version_status("This report is NOT FINAL and may be revised") == "draft"
    assert _version_status("Preliminary version, subject to review") == "draft"
    assert _version_status("This edition is superseded by the 2025 revision") == "superseded"
    assert _version_status("Replaced by the final report dated 2025-12-31") == "superseded"
    assert _version_status("This procedure is no longer in effect") == "superseded"
    # Plain active document is unaffected.
    assert _version_status("Final report as at 31 December 2025") == "active"


def test_version_status_keeps_springing_covenant_active():
    """A springing covenant reads 'пока <condition>, ограничение не
    применяется' — this is applicability wording inside an ACTIVE
    agreement, not a supersession marker. The triage used to grab the
    literal 'не применяется' and route the agreement into 'superseded',
    which hid the covenant text from Stage 6."""
    from stages.s2_pdf import _version_status

    assert _version_status(
        "Пока Коэффициент долговой нагрузки не превышает 3.00x, указанное "
        "ограничение Капитальных затрат не применяется."
    ) == "active"
    assert _version_status(
        "Пока Debt/EBITDA не превышает 2.40x, ограничение Распределений не применяется."
    ) == "active"


def test_version_status_explicit_russian_supersession_still_flagged():
    """Real supersession phrases in Russian must still route to
    'superseded' — 'недействующая редакция' and 'утратил силу' remain in
    the pattern set."""
    from stages.s2_pdf import _version_status

    assert _version_status(
        "НЕДЕЙСТВУЮЩАЯ РЕДАКЦИЯ (2024 г.). Заменена и изложена в новой редакции."
    ) == "superseded"
    assert _version_status(
        "Настоящий договор утратил силу с 1 января 2025 года."
    ) == "superseded"
