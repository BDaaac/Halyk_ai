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


def test_accounts_in_text_finds_arbitrary_ledger_prefix():
    """A private ledger may use account tokens whose prefix is not ACC
    (e.g. TELE-XXXX for a telco). Document→account association must fall
    out of the ledger's own account_id column, not a hardcoded regex."""
    from stages.s2_pdf import _accounts_in_text

    tokens = ["ACC-7001", "TELE-1234", "MFG-9002"]
    text = (
        "Договор № 2025-142\nСчёт заёмщика: TELE-1234.\n"
        "Историческая ссылка на ACC-7001 приложена в приложении А."
    )
    result = _accounts_in_text(text, tokens)

    assert "TELE-1234" in result
    assert "ACC-7001" in result
    assert "MFG-9002" not in result


def test_accounts_in_text_longer_token_wins_over_prefix_collision():
    """Sorting tokens longest-first prevents a shorter token that is a
    prefix of a longer one from masking it in the accounts list."""
    from stages.s2_pdf import _accounts_in_text

    # Longest-first order is what _ledger_account_tokens returns.
    tokens = sorted(["ACC-7001", "ACC-70011"], key=lambda t: (-len(t), t))
    text = "The referenced account is ACC-70011 (not the older ACC-7001)."
    result = _accounts_in_text(text, tokens)

    # Both appear in the text (ACC-7001 is a substring of ACC-70011).
    # We accept both — the routing layer picks the account whose scenario
    # actually matches the ledger, so a spurious substring hit cannot
    # silently reroute a document.
    assert "ACC-70011" in result
    assert "ACC-7001" in result
    assert result.index("ACC-70011") < result.index("ACC-7001")


def test_ledger_account_tokens_reads_column_and_sorts_longest_first(tmp_path):
    """Ledger account tokens are harvested from the account_id column,
    duplicates dropped, and returned longest-first for stable matching."""
    from stages.s2_pdf import _ledger_account_tokens

    ledger = tmp_path / "master_ledger_2025.csv"
    ledger.write_text(
        "txn_id,date,account_id,counterparty,description,amount,currency\n"
        "TXN-A-01,2025-01-01,ACC-7001,X,y,100,USD\n"
        "TXN-A-02,2025-01-02,ACC-7001,X,y,200,USD\n"
        "TXN-B-01,2025-01-03,TELE-1234,Z,w,300,USD\n"
        "TXN-C-01,2025-01-04,MFG-9002,Q,v,400,USD\n",
        encoding="utf-8",
    )
    tokens = _ledger_account_tokens(ledger)

    assert set(tokens) == {"ACC-7001", "TELE-1234", "MFG-9002"}
    # longest-first (with alphabetic tie-break) so 'ACC-70011' would come
    # before 'ACC-7001' if both existed
    assert tokens == sorted(tokens, key=lambda t: (-len(t), t))


def test_ledger_account_tokens_missing_ledger_returns_empty(tmp_path):
    """No ledger file → no tokens (tests that construct DocRecord directly
    must keep working without a data_dir setup)."""
    from stages.s2_pdf import _ledger_account_tokens

    assert _ledger_account_tokens(tmp_path / "missing.csv") == []
