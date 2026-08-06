"""Configuration-driven cost estimates for API usage."""

from decimal import Decimal

from config import ModelPricing


def estimate_cost(usage: dict[str, int], pricing: ModelPricing | None) -> Decimal | None:
    if pricing is None:
        return None
    return (
        Decimal(usage.get("input_tokens", 0)) * pricing.input_usd_per_mtok
        + Decimal(usage.get("output_tokens", 0)) * pricing.output_usd_per_mtok
    ) / Decimal("1000000")
