"""Настройки запуска пайплайна из окружения."""

from dataclasses import dataclass
import os
from pathlib import Path


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
    """Возвращает настройки без чтения тестовых фикстур."""
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", ROOT / "agentic-bank-public")),
        workspace_dir=Path(os.getenv("WORKSPACE_DIR", ROOT / "workspace")),
        extract_model=os.getenv("EXTRACT_MODEL", ""),
        select_model=os.getenv("SELECT_MODEL", ""),
        verify_model=os.getenv("VERIFY_MODEL", ""),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        llm_max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "3")),
    )
