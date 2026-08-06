# Covenant Pipeline Phase 0 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Сделать измеримую фазу 0: тесты собираются, скорер и PDF-проверки работают, CLI-команды объявлены, а неготовые стадии явно завершаются `NotImplementedError`.

**Architecture:** Корень проекта читает входной набор из `DATA_DIR` (по умолчанию `agentic-bank-public/`), а `tests/fixtures/` содержит только два малых JSON-файла для независимого scorer-теста. PyMuPDF извлекает PDF в одном процессе; значение `< 200` символов направляет документ в временную vision-заглушку. Будущие стадии представлены только стабильными API-заглушками и не производят данных.

**Tech Stack:** Python 3.12.13, pytest, PyMuPDF (`fitz`), pandas, Pydantic v2, argparse.

## Global Constraints

- Не менять корневой `scorer.py`.
- Не передавать числовые поля `submission.json` из LLM и не добавлять LLM-as-judge, RAG, векторный поиск или агентные фреймворки.
- `DATA_DIR` — единственный источник рабочего `submission_template.json`; тестовые JSON не используются production-кодом.
- Все незавершённые API выбрасывают `NotImplementedError("stage N")`, не возвращают `None`.
- Документы не удаляются; маркер — кандидат, окончательное разрешение версий остаётся стадией 4.
- Комментарии и README — на русском; идентификаторы кода — на английском.
- PyMuPDF используется под AGPL-3.0 только для хакатона; README содержит соответствующую оговорку.

---

### Task 1: Bootstrap проекта и конфигурация путей

**Files:**
- Create: `.gitignore`, `requirements.txt`, `.python-version`, `.env.example`, `config.py`, `tests/test_config.py`
- Modify: none

**Interfaces:**
- Produces: `config.Settings`, `get_settings() -> Settings`; поля `data_dir: Path`, `workspace_dir: Path`, `extract_model: str`, `select_model: str`, `verify_model: str`, `llm_timeout_seconds: int`, `llm_max_concurrency: int`.
- Consumes: `DATA_DIR`, `WORKSPACE_DIR`, `EXTRACT_MODEL`, `SELECT_MODEL`, `VERIFY_MODEL`, `LLM_TIMEOUT_SECONDS`, `LLM_MAX_CONCURRENCY` из окружения.

- [ ] **Step 1: Write the failing configuration test**

The bug this catches is silently reading fixtures or a hard-coded public dataset rather than the configured dataset.

```python
# tests/test_config.py
from pathlib import Path

from config import get_settings


def test_data_dir_can_be_overridden(monkeypatch, tmp_path: Path):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "private-data"))
    assert get_settings().data_dir == tmp_path / "private-data"
```

- [ ] **Step 2: Run it to verify RED**

Run: `python -m pytest tests/test_config.py -v`

Expected: collection fails because module `config` does not exist.

- [ ] **Step 3: Add minimal project configuration**

```python
# config.py
from dataclasses import dataclass
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

@dataclass(frozen=True)
class Settings:
    data_dir: Path
    workspace_dir: Path
    extract_model: str
    select_model: str
    verify_model: str
    llm_timeout_seconds: int
    llm_max_concurrency: int

def get_settings() -> Settings:
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", ROOT / "agentic-bank-public")),
        workspace_dir=Path(os.getenv("WORKSPACE_DIR", ROOT / "workspace")),
        extract_model=os.getenv("EXTRACT_MODEL", ""),
        select_model=os.getenv("SELECT_MODEL", ""),
        verify_model=os.getenv("VERIFY_MODEL", ""),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        llm_max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "3")),
    )
```

Write `.python-version` as `3.12.13`, `requirements.txt` with `pandas==2.3.2`, `pydantic==2.11.7`, `PyMuPDF==1.26.4`, `pytest==8.4.1`, and `.env.example` documenting every environment variable above. Add `workspace/`, `.env`, `.venv/`, `__pycache__/`, `.pytest_cache/`, `agentic-bank-public/`, `reports/` to `.gitignore`.

- [ ] **Step 4: Run the test to verify GREEN**

Run: `python -m pytest tests/test_config.py -v`

Expected: `1 passed`.

- [ ] **Step 5: Commit**

```bash
git add .gitignore requirements.txt .python-version .env.example config.py tests/test_config.py
git commit -m "chore: bootstrap covenant pipeline configuration"
```

### Task 2: Зафиксировать fixtures и acceptance-границу

**Files:**
- Create: `tests/fixtures/submission_template.json`, `tests/fixtures/ground_truth.json`, `pipeline.py`, `tests/test_acceptance.py`
- Create: `lib/__init__.py`, `lib/text.py`, `lib/expressions.py`
- Modify: none

**Interfaces:**
- Produces: `pipeline.run_pipeline(*, stop_after_stage: int | None = None)`, `pipeline.resolve_documents(scenario_id: str)`, `pipeline.run_scenario(scenario_id: str)`; каждая функция пока выбрасывает `NotImplementedError`.
- Consumes: JSON из `tests/fixtures/` только в тестах.

- [ ] **Step 1: Write the failing acceptance tests**

The bug this catches is turning missing stage implementation into a silent empty result or a collection error.

```python
# tests/test_acceptance.py
from pipeline import resolve_documents, run_pipeline, run_scenario

def test_b1_document_set_requires_stage_4():
    documents = resolve_documents("B1")
    assert documents.agreement == "b38facf69a94"

def test_b1_all_three_requires_compute_stages():
    results = run_scenario("B1")
    assert results["6.1"].status == "BREACH"

def test_baseline_written_first_requires_stage_0():
    submission = run_pipeline(stop_after_stage=0)
    assert submission["answers"]
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_acceptance.py -v`

Expected: collection fails because module `pipeline` does not exist.

- [ ] **Step 3: Add explicit stage stubs and fixture copies**

```python
# pipeline.py
def run_pipeline(*, stop_after_stage: int | None = None): raise NotImplementedError("stage 0")
def gap_scan(): raise NotImplementedError("stage 1")
def build_document_index(): raise NotImplementedError("stage 2")
def read_document_text(doc_id: str): raise NotImplementedError("stage 2")
def get_account_to_scenario(): raise NotImplementedError("stage 1")
def get_corrections(): raise NotImplementedError("stage 5")
def resolve_documents(scenario_id: str): raise NotImplementedError("stage 4")
def get_specifications(): raise NotImplementedError("stage 6")
def related_party_threshold(scenario_id: str): raise NotImplementedError("stage 6")
def related_parties(scenario_id: str): raise NotImplementedError("stage 8")
def run_scenario(scenario_id: str): raise NotImplementedError("stage 8")
def recompute_without(scenario_id: str, clause_id: str, txn_id: str): raise NotImplementedError("stage 9")
```

```python
# lib/text.py
def normalize_identifiers(text: str) -> str:
    raise NotImplementedError("stage 3")

# lib/expressions.py
class ExpressionTooDeep(Exception): pass
def sum_(values): raise NotImplementedError("stage 8")
def add(left, right): raise NotImplementedError("stage 8")
def subtract(left, right): raise NotImplementedError("stage 8")
def divide(left, right): raise NotImplementedError("stage 8")
def max_(left, right): raise NotImplementedError("stage 8")
def evaluate(expression): raise NotImplementedError("stage 8")
```

Copy `agentic-bank-public/submission_template.json` and `agentic-bank-public/ground_truth.json` byte-for-byte into `tests/fixtures/`. Add the following direct calls to the same test module; do not wrap any of them in `pytest.raises`, `xfail` or a catch-all helper. Before their stage exists, each test is red because its first production call raises the named `NotImplementedError`.

```python
from decimal import Decimal

from pipeline import (
    build_document_index, gap_scan, get_account_to_scenario,
    get_corrections, get_specifications, read_document_text,
    related_parties, related_party_threshold,
)
from lib.text import normalize_identifiers

def test_gap_scan_finds_missing_amounts():
    assert gap_scan()["missing_amount"] == ["TXN-P7-0033", "TXN-P8-0031"]

def test_scanned_pdf_detected():
    assert build_document_index()["f3fa6d20c8a1"].extraction_method == "vision"

def test_no_document_deleted():
    assert len(build_document_index()) == 200

def test_letter_spaced_ids_normalized():
    assert "ACC-7201" in normalize_identifiers(read_document_text("a16d7a619f87"))

def test_normalization_preserves_prose():
    source = "Заёмщик не вправе отчуждать основные средства и иные активы"
    assert normalize_identifiers(source) == source

def test_account_scenario_mapping():
    assert get_account_to_scenario()["ACC-7201"] == "B1"
    assert get_account_to_scenario()["ACC-7801"] == "P1"

def test_targeted_retrieval_closes_gaps():
    corrections = get_corrections()
    assert corrections["TXN-P7-0033"].correct_amount == Decimal("486204.19")
    assert corrections["TXN-P8-0031"].correct_amount == Decimal("884204.16")

def test_p6_related_party_threshold_is_forty_percent():
    assert related_party_threshold("P6") == Decimal("40.0")
    assert "Taraz Kiln Services LLP" not in related_parties("P6")

def test_p3_springing_condition_parsed():
    assert get_specifications()["P3"]["6.1"].applicability.threshold == Decimal("4000000.00")
```

Add the remaining direct contracts exactly as follows; these import future production APIs and therefore remain red until stages 8–10 exist.

```python
from lib.expressions import (
    ExpressionTooDeep, add, divide, evaluate, max_, subtract, sum_,
)
from pipeline import recompute_without

def nested_expr(depth: int):
    expression = sum_([Decimal("1")])
    for _ in range(depth):
        expression = add(expression, sum_([Decimal("1")]))
    return expression

def test_b1_icr():
    expression = divide(
        subtract(sum_([Decimal("9741934.78")]), sum_([Decimal("6166592.66")])),
        sum_([Decimal("1540833.29"), Decimal("592296.10")]),
    )
    assert round(evaluate(expression), 2) == Decimal("1.68")

def test_p10_nested_max():
    revenue = [Decimal("9000000.00")]
    payroll = [Decimal("2999236.37")]
    taxes = [Decimal("1200000.00")]
    expression = subtract(sum_(revenue), max_(sum_(payroll), sum_(taxes)))
    assert round(evaluate(expression), 2) == Decimal("6000763.63")

def test_expression_depth_limit():
    with pytest.raises(ExpressionTooDeep):
        evaluate(nested_expr(depth=5))

def test_b1_all_three():
    result = run_scenario("B1")
    assert result["6.1"].status == "BREACH"
    assert result["6.1"].actual == Decimal("1.68")
    assert result["6.1"].evidence_txn_id == "TXN-B1-0020"
    assert result["6.2"].status == "COMPLIANT"
    assert result["6.2"].actual == Decimal("1284663.42")
    assert result["6.3"].status == "COMPLIANT"
    assert result["6.3"].actual == Decimal("307018.08")

def test_b1_counterfactual():
    assert recompute_without("B1", "6.1", "TXN-B1-0020").status == "COMPLIANT"

def test_all_cells_present():
    submission = run_pipeline()
    template = json.loads((FIXTURES / "submission_template.json").read_text())
    assert set(submission["answers"]) == set(template["answers"])

def test_evidence_belongs_to_scenario():
    for scenario_id, clauses in run_pipeline()["answers"].items():
        for cell in clauses.values():
            if cell["evidence_txn_id"]:
                assert cell["evidence_txn_id"].split("-")[1] == scenario_id

def test_actual_always_positive_two_decimals():
    for clauses in run_pipeline()["answers"].values():
        for cell in clauses.values():
            assert cell["actual"] > 0
            assert Decimal(str(cell["actual"])).as_tuple().exponent >= -2
```

- [ ] **Step 4: Run tests to verify expected phase-0 state**

Run: `python -m pytest tests/test_acceptance.py -v`

Expected: all tests collect; `test_scorer_baseline_floor`, `test_pdf_extraction_real_file`, `test_scanned_pdf_routes_to_vision` and the no-op prose normalizer are green after their tasks, while every remaining future-stage test is red with the matching `NotImplementedError`, without import errors or `NoneType` assertions.

- [ ] **Step 5: Commit**

```bash
git add tests/fixtures pipeline.py lib tests/test_acceptance.py
git commit -m "test: establish covenant acceptance boundary"
```

### Task 3: Подключить точный scorer и baseline-тест

**Files:**
- Create: `tests/test_scorer_integration.py`
- Modify: `tests/test_acceptance.py`

**Interfaces:**
- Consumes: `scorer.score_submission(submission: dict, ground_truth: dict) -> RunScore` без изменения `scorer.py`.
- Produces: test-only `baseline_submission(template: dict) -> dict` с валидными значениями в каждой ячейке.

- [ ] **Step 1: Write the failing baseline scorer test**

The bug this catches is a disconnected or altered scorer whose baseline no longer matches the official formula.

```python
# tests/test_scorer_integration.py
import json
from copy import deepcopy
from pathlib import Path

from scorer import score_submission

FIXTURES = Path(__file__).parent / "fixtures"

def baseline_submission(template: dict) -> dict:
    result = deepcopy(template)
    for scenario in result["answers"].values():
        for cell in scenario.values():
            cell.update(status="COMPLIANT", actual=0.01, evidence_txn_id=None)
    return result

def test_scorer_baseline_floor():
    template = json.loads((FIXTURES / "submission_template.json").read_text())
    ground_truth = json.loads((FIXTURES / "ground_truth.json").read_text())
    assert 0.26 <= score_submission(baseline_submission(template), ground_truth).mean <= 0.27
```

- [ ] **Step 2: Run it to verify RED**

Run: `python -m pytest tests/test_scorer_integration.py::test_scorer_baseline_floor -v`

Expected: FAIL before fixture copies exist.

- [ ] **Step 3: Do not change production scorer code**

Use the fixtures copied in Task 2. If the test does not evaluate to the expected `0.2639` neighborhood, stop and report the mismatch instead of adapting the baseline or scorer.

- [ ] **Step 4: Run it to verify GREEN**

Run: `python -m pytest tests/test_scorer_integration.py::test_scorer_baseline_floor -v`

Expected: `1 passed` and a score within `[0.26, 0.27]`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_scorer_integration.py tests/test_acceptance.py
git commit -m "test: verify official scorer baseline"
```

### Task 4: Проверка извлечения PDF через PyMuPDF

**Files:**
- Create: `lib/__init__.py`, `lib/pdf.py`, `lib/text.py`, `tests/test_pdf.py`
- Modify: `requirements.txt`, `tests/test_acceptance.py`

**Interfaces:**
- Produces: `lib.pdf.extract_text(path: Path) -> tuple[str, str]`, `lib.text.normalize_identifiers(text: str) -> str`.
- Consumes: PDF из `get_settings().data_dir / "documents"` и PyMuPDF.

- [ ] **Step 1: Write the failing integration tests**

The first test catches broken Unicode-path extraction or a wrong non-scan classification. The second catches a scan being treated as ordinary text.

```python
# tests/test_pdf.py
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
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_pdf.py -v`

Expected: collection fails because modules `lib.pdf` and `lib.text` do not exist.

- [ ] **Step 3: Implement only the extraction boundary and targeted normalizer**

```python
# lib/pdf.py
from pathlib import Path
import fitz

def vision_extract(pages: list[fitz.Pixmap]) -> str:
    return ""

def extract_text(path: Path) -> tuple[str, str]:
    with fitz.open(path) as document:
        text = "\n".join(page.get_text() for page in document)
        if len(text.strip()) < 200:
            pages = [page.get_pixmap(matrix=fitz.Matrix(2, 2)) for page in document[:3]]
            return vision_extract(pages), "vision"
    return text, "text"
```

```python
# lib/text.py
import re
import unicodedata

ACC_RE = re.compile(r"[AА]\s*[CС]\s*[CС]\s*[-–—]\s*(\d)\s*(\d)\s*(\d)\s*(\d)")

def normalize_identifiers(text: str) -> str:
    return ACC_RE.sub(lambda match: "ACC-" + "".join(match.groups()), unicodedata.normalize("NFKC", text))
```

- [ ] **Step 4: Run tests to verify GREEN**

Run: `python -m pytest tests/test_pdf.py -v`

Expected: `2 passed`. If either test fails, stop phase 0 as an environment/data blocker and report the observed text length and method.

- [ ] **Step 5: Commit**

```bash
git add requirements.txt lib tests/test_pdf.py tests/test_acceptance.py
git commit -m "feat: verify PyMuPDF extraction boundary"
```

### Task 5: CLI skeleton, full gate and phase report

**Files:**
- Create: `main.py`, `tests/test_cli.py`, `README.md`
- Modify: `tests/test_acceptance.py`

**Interfaces:**
- Produces: `python main.py score SUBMISSION GROUND_TRUTH`, `python main.py run`, `python main.py eval`, `python main.py diff`.
- Consumes: immutable `scorer.score_submission`; `pipeline.run_pipeline` only when future stages exist.

- [ ] **Step 1: Write failing CLI tests**

The first test catches `score` accidentally becoming a pipeline evaluator; the second catches `eval` being silently aliased to `score`.

```python
# tests/test_cli.py
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parents[1]

def test_score_reports_score_for_ready_files():
    result = subprocess.run(
        [sys.executable, "main.py", "score", "tests/fixtures/submission_template.json", "tests/fixtures/ground_truth.json"],
        cwd=ROOT, text=True, capture_output=True,
    )
    assert result.returncode == 0
    assert "Средний балл по ячейке" in result.stdout

def test_eval_reports_unimplemented_pipeline():
    result = subprocess.run([sys.executable, "main.py", "eval"], cwd=ROOT, text=True, capture_output=True)
    assert result.returncode != 0
    assert "stage 0" in result.stderr
```

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_cli.py -v`

Expected: FAIL because `main.py` is absent.

- [ ] **Step 3: Implement the CLI without hidden fallbacks**

```python
# main.py
import argparse
import json
from pathlib import Path

from pipeline import run_pipeline
from scorer import score_submission

def score_command(submission_path: Path, ground_truth_path: Path) -> None:
    submission = json.loads(submission_path.read_text(encoding="utf-8"))
    ground_truth = json.loads(ground_truth_path.read_text(encoding="utf-8"))
    print(score_submission(submission, ground_truth).report())

def main() -> None:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    score = commands.add_parser("score")
    score.add_argument("submission", type=Path)
    score.add_argument("ground_truth", type=Path)
    commands.add_parser("run")
    commands.add_parser("eval")
    commands.add_parser("diff")
    args = parser.parse_args()
    if args.command == "score":
        score_command(args.submission, args.ground_truth)
    elif args.command in {"run", "eval"}:
        run_pipeline()
    else:
        raise NotImplementedError("diff")

if __name__ == "__main__":
    main()
```

Write a Russian README with setup using Python 3.12.13, data-path override, each CLI command’s current semantics, PyMuPDF AGPL-3.0 note and the command to run pytest.

- [ ] **Step 4: Verify the full phase gate**

Run: `python -m pytest -v`

Expected: scorer, config, PDF and CLI tests pass; acceptance tests for stages 0–12 are collected and fail only with their named `NotImplementedError`. Record exact pass/fail counts in the phase report.

- [ ] **Step 5: Commit and push**

```bash
git add main.py README.md tests/test_cli.py tests/test_acceptance.py
git commit -m "feat: add phase zero measurement harness"
git push
```

## Plan self-review

- Spec coverage: configuration, Python version, fixtures, immutable scorer, PyMuPDF route, Unicode path, scanned PDF, CLI semantics, explicit incomplete-stage failures, ignore rules, and license note map to Tasks 1–5.
- No LLM, numeric model output, public-dataset hardcoding in production code, document deletion, vector search, or agent framework is introduced.
- Current environment finding: the supplied Python 3.12.13 is not on PATH; the bundled interpreter is available at `C:\Users\Димаш\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe`, but lacks both `fitz` and `pytest`. Task 1 must install the pinned requirements into a project virtual environment before any pytest command can pass.
