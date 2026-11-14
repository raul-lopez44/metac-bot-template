"""
Model tiers for the ensemble bot.

Every model the bot uses is named here, never in the pipeline code, so moving
from the low-budget setup to frontier models is a one-word change:

    BOT_TIER=strong            (in .env or as a GitHub Actions variable)
    uv run python bot.py --tier strong

Tiers:
- free:   OpenRouter ":free" models. Only for checking that the pipeline runs.
          Weak forecasts and prompts are shared with the provider. No web search.
- cheap:  Small models from three different labs. A few cents per question.
- strong: Frontier models from three different labs. Use when credits allow.

Model IDs and prices were read from https://openrouter.ai/api/v1/models on
2026-11-11. Re-check them when changing a tier: IDs get retired.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Tier:
    name: str
    # One independent forecast per entry; the bot publishes the median.
    # Use different labs so the errors are less correlated.
    forecasters: list[str]
    # Models that search the web themselves. Each one is an independent
    # research source for the first research pass.
    searchers: list[str]
    # Model that answers the follow-up (gap-fill) queries. None disables gap fill.
    gap_searcher: str | None
    # Cheap model for planning queries, parsing forecasts and summarising.
    utility: str
    max_gap_queries: int = 3
    forecaster_timeout: int = 180
    extra: dict = field(default_factory=dict)


TIERS: dict[str, Tier] = {
    "free": Tier(
        name="free",
        forecasters=[
            "openrouter/nvidia/nemotron-3-ultra-550b-a55b:free",
            "openrouter/google/gemma-4-31b-it:free",
            "openrouter/thinkingmachines/inkling-2-small:free",
        ],
        searchers=[],
        gap_searcher=None,
        utility="openrouter/google/gemma-4-31b-it:free",
        max_gap_queries=0,
    ),
    "cheap": Tier(
        name="cheap",
        forecasters=[
            "openrouter/anthropic/claude-haiku-5.5",
            "openrouter/openai/gpt-6-luna",
            "openrouter/google/gemini-3.5-flash-lite",
        ],
        # ":online" (provider's native web search, e.g. "...gpt-6-luna:online") is
        # NOT used here: OpenRouter charges $0.01 per search and the model ran
        # many per question (~78% of the first dry run's cost). Use it in "strong".
        searchers=[
            "openrouter/perplexity/sonar",
        ],
        gap_searcher="openrouter/perplexity/sonar",
        utility="openrouter/anthropic/claude-haiku-5.5",
        max_gap_queries=3,
    ),
    "strong": Tier(
        name="strong",
        forecasters=[
            "openrouter/anthropic/claude-opus-5.5",
            "openrouter/openai/gpt-6.1-sol",
            "openrouter/google/gemini-4-carbon",
        ],
        searchers=[
            "openrouter/perplexity/sonar-pro",
            "openrouter/openai/gpt-6.1-terra:online",
            "openrouter/google/gemini-3.8-flash:online",
        ],
        gap_searcher="openrouter/perplexity/sonar-pro",
        utility="openrouter/anthropic/claude-haiku-5.5",
        max_gap_queries=4,
    ),
}

DEFAULT_TIER = "cheap"


def get_tier(name: str | None = None) -> Tier:
    chosen = (name or os.getenv("BOT_TIER") or DEFAULT_TIER).strip().lower()
    if chosen not in TIERS:
        raise ValueError(f"Unknown tier '{chosen}'. Choose one of: {sorted(TIERS)}")
    return TIERS[chosen]


# ---- Post-processing limits -------------------------------------------------
# Binary forecasts are clipped to this range. A confident mistake at 0.1% costs
# far more spot peer score than the clip gives up when the bot is right.
BINARY_MIN = 0.02
BINARY_MAX = 0.98
# Minimum probability per option on multiple-choice questions (then renormalised).
MULTIPLE_CHOICE_FLOOR = 0.01
