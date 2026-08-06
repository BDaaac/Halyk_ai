"""Настройки запуска пайплайна из окружения."""

from dataclasses import dataclass
from decimal import Decimal
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent


@dataclass(frozen=True)
class ModelPricing:
    input_usd_per_mtok: Decimal
    output_usd_per_mtok: Decimal


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    workspace_dir: Path
    extract_model: str
    select_model: str
    verify_model: str
    llm_timeout_seconds: int
    llm_max_concurrency: int
    anthropic_api_key: str
    select_mode: str
    extract_pricing: ModelPricing | None
    select_pricing: ModelPricing | None


def load_dotenv(path: Path, environment: dict[str, str] | None = None) -> None:
    """Load simple KEY=VALUE pairs, preserving explicit process variables."""
    target = os.environ if environment is None else environment
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in target:
            target[key] = value.strip().strip('"').strip("'")


def _stage_pricing(stage: str) -> ModelPricing | None:
    input_rate = os.getenv(f"{stage}_INPUT_USD_PER_MTOK", "")
    output_rate = os.getenv(f"{stage}_OUTPUT_USD_PER_MTOK", "")
    if not input_rate and not output_rate:
        return None
    if not input_rate or not output_rate:
        raise ValueError(f"{stage} pricing requires both input and output USD/MTok rates")
    return ModelPricing(Decimal(input_rate), Decimal(output_rate))


def get_settings() -> Settings:
    """Возвращает настройки без чтения тестовых фикстур."""
    load_dotenv(ROOT / ".env")
    return Settings(
        data_dir=Path(os.getenv("DATA_DIR", ROOT / "agentic-bank-public")),
        workspace_dir=Path(os.getenv("WORKSPACE_DIR", ROOT / "workspace")),
        extract_model=os.getenv("EXTRACT_MODEL", ""),
        select_model=os.getenv("SELECT_MODEL", ""),
        verify_model=os.getenv("VERIFY_MODEL", ""),
        llm_timeout_seconds=int(os.getenv("LLM_TIMEOUT_SECONDS", "60")),
        llm_max_concurrency=int(os.getenv("LLM_MAX_CONCURRENCY", "3")),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
        select_mode=os.getenv("SELECT_MODE", "scenario"),
        extract_pricing=_stage_pricing("EXTRACT"),
        select_pricing=_stage_pricing("SELECT"),
    )
