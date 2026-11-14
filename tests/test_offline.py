"""
Checks that need no API keys and spend nothing.

    uv run python tests/test_offline.py            # logic only
    uv run python tests/test_offline.py --network  # also hits the public market APIs and a web page
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
os.environ.setdefault("OPENROUTER_API_KEY", "offline-test")  # only so model objects can be built

import httpx  # noqa: E402

import bot  # noqa: E402
import bot_config  # noqa: E402
import research  # noqa: E402
from forecasting_tools import GeneralLlm, PredictedOptionList  # noqa: E402
from forecasting_tools.data_models.multiple_choice_report import PredictedOption  # noqa: E402


def test_tiers_are_complete() -> None:
    for name, tier in bot_config.TIERS.items():
        assert tier.name == name
        assert len(tier.forecasters) >= 3, name
        labs = {model.split("/")[1] for model in tier.forecasters}
        assert len(labs) == len(tier.forecasters), f"{name}: forecasters should come from different labs"
    assert bot_config.get_tier("STRONG").name == "strong"
    try:
        bot_config.get_tier("nope")
    except ValueError:
        pass
    else:
        raise AssertionError("unknown tier should raise")


def test_clip_binary() -> None:
    assert bot.clip_binary(0.0005) == bot_config.BINARY_MIN
    assert bot.clip_binary(0.9999) == bot_config.BINARY_MAX
    assert bot.clip_binary(0.37) == 0.37


def test_floor_options() -> None:
    raw = PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name="A", probability=0.999),
            PredictedOption(option_name="B", probability=0.001),
            PredictedOption(option_name="C", probability=0.0),
        ]
    )
    floored = bot.floor_options(raw)
    probabilities = {o.option_name: o.probability for o in floored.predicted_options}
    assert abs(sum(probabilities.values()) - 1) < 1e-9
    assert min(probabilities.values()) >= bot_config.MULTIPLE_CHOICE_FLOOR * 0.97
    assert probabilities["A"] > 0.97


def test_extract_json_object() -> None:
    assert research.extract_json_object('Sure!\n```json\n{"queries": ["a", "b"]}\n```') == {
        "queries": ["a", "b"]
    }
    assert research.extract_json_object("no json here") == {}
    assert research.extract_json_object("{broken") == {}


def test_find_resolution_urls() -> None:
    question = SimpleNamespace(
        resolution_criteria="Resolves per https://example.org/data.html. See also (https://example.org/b).",
        fine_print="Ignore https://www.metaculus.com/questions/1/",
        background_info="Context: https://example.org/data.html and https://news.example.com/x",
    )
    assert research.find_resolution_urls(question) == [  # type: ignore[arg-type]
        "https://example.org/data.html",
        "https://example.org/b",
        "https://news.example.com/x",
    ]
    wiki = SimpleNamespace(
        resolution_criteria=r"Per [the article](https://en.wikipedia.org/wiki/Electoral_College_\(United_States\)).",
        fine_print="",
        background_info="(see https://en.wikipedia.org/wiki/Mercury_(planet))",
    )
    assert research.find_resolution_urls(wiki) == [  # type: ignore[arg-type]
        "https://en.wikipedia.org/wiki/Electoral_College_(United_States)",
        "https://en.wikipedia.org/wiki/Mercury_(planet)",
    ]


def test_date_facts() -> None:
    from datetime import datetime, timezone

    question = SimpleNamespace(
        close_time=datetime(2026, 12, 20, 12, tzinfo=timezone.utc),
        scheduled_resolution_time=datetime(2027, 1, 2, tzinfo=timezone.utc),
    )
    now = datetime(2026, 11, 14, 9, tzinfo=timezone.utc)
    text = bot.date_facts(question, now)  # type: ignore[arg-type]
    assert "Today is 2026-11-14 (Saturday, UTC)" in text
    assert "2026-12-20 (36 days from today)" in text
    assert "2027-01-02 (49 days from today)" in text
    assert "1 January 2027: 48 days from today" in text
    # Missing dates must not break it.
    assert "Today is" in bot.date_facts(SimpleNamespace(), now)  # type: ignore[arg-type]


def test_background_urls_marked() -> None:
    question = SimpleNamespace(
        resolution_criteria="Per https://example.org/source",
        fine_print="",
        background_info="Old article: https://example.org/old",
    )
    assert research._question_urls(question) == [  # type: ignore[arg-type]
        ("https://example.org/source", False),
        ("https://example.org/old", True),
    ]


def test_html_to_text() -> None:
    page = "<html><head><title>t</title></head><body><script>var x=1;</script><p>Rate: 4.25&nbsp;%</p><div>Next</div></body></html>"
    text = research.html_to_text(page)
    assert "var x" not in text and "Rate: 4.25" in text and "Next" in text


def test_forecasters_rotate() -> None:
    ensemble = bot.EnsembleBot(tier=bot_config.get_tier("cheap"))
    assert ensemble.predictions_per_research_report == 3
    question = SimpleNamespace(id_of_question=42)
    picked = [ensemble._next_forecaster(question).model for _ in range(6)]  # type: ignore[arg-type]
    assert picked[:3] == bot_config.get_tier("cheap").forecasters
    assert picked[3:] == picked[:3]
    # Outside a prediction task the default model is returned; inside, the assigned one.
    assert ensemble.get_llm("default", "llm").model == picked[0]
    token = bot._current_forecaster.set(GeneralLlm(model=picked[2]))
    try:
        assert ensemble.get_llm("default", "llm").model == picked[2]
        assert ensemble.get_llm("parser", "llm").model == bot_config.get_tier("cheap").utility
    finally:
        bot._current_forecaster.reset(token)


async def network_checks() -> None:
    async with httpx.AsyncClient(timeout=20, headers=research._BROWSER_HEADERS) as client:
        for site in (research._manifold, research._polymarket):
            try:
                lines = await site(client, "federal reserve rate")
                print(f"{site.__name__}: {len(lines)} market(s)")
                for line in lines[:3]:
                    print("   ", line[:160])
            except Exception as error:  # noqa: BLE001
                print(f"{site.__name__}: FAILED {type(error).__name__}: {error}")
    question = SimpleNamespace(
        resolution_criteria="Resolves according to https://en.wikipedia.org/wiki/Federal_funds_rate",
        fine_print="",
        background_info="",
    )
    pages = await research._resolution_pages(question)  # type: ignore[arg-type]
    print(f"resolution page: {len(pages)} characters")
    print("   ", pages[:300].replace("\n", " "))


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items()) if name.startswith("test_")]
    for test in tests:
        test()
        print(f"ok  {test.__name__}")
    if "--network" in sys.argv:
        asyncio.run(network_checks())
