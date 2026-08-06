# Пайплайн проверки кредитных ковенантов

Решение читает структуру задания из `submission_template.json` набора данных и
считает все суммы и коэффициенты в Python. Числовые поля итогового сабмита не
должны поступать из LLM.

## Подготовка

Нужен Python 3.12.13. Создайте виртуальное окружение и установите зависимости:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

По умолчанию данные ожидаются в `agentic-bank-public/` рядом с кодом. Для другого
набора задайте `DATA_DIR`; пример переменных находится в `.env.example`.

```powershell
$env:DATA_DIR = 'C:\path\to\private-dataset'
.\.venv\Scripts\python.exe -m pytest -v
```

## Команды

```powershell
# Оценить уже подготовленный submission, не запуская пайплайн.
.\.venv\Scripts\python.exe main.py score submission.json ground_truth.json

# Будущие команды пайплайна. В фазе 0 они завершаются NotImplementedError("stage 0").
.\.venv\Scripts\python.exe main.py run
.\.venv\Scripts\python.exe main.py eval

# Сравнение прогонов будет добавлено вместе с harness.
.\.venv\Scripts\python.exe main.py diff
```

## Лицензия PyMuPDF

Для извлечения текста используется PyMuPDF (AGPL-3.0). Для хакатона это допустимо;
перед использованием проекта вне его необходимо отдельно проверить лицензионные
условия.
