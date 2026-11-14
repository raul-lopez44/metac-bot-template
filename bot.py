"""
Ensemble bot: the Metaculus template plus
  1. multi-source research with a gap-fill pass (research.py),
  2. one forecast from each of several model families, median-aggregated,
  3. forecasting guardrails and clipping of extreme probabilities,
  4. a dry-run mode that records forecasts so variants can be compared on the
     same questions once they resolve.

The prompts and the parsing are inherited from main.py (the template), so
upstream fixes keep applying. Models live in bot_config.py.

Examples:
    uv run python bot.py --mode test_questions --no-publish --limit 3
    uv run python bot.py --mode metaculus_cup --variant both --limit 10   # paired comparison, never publishes
    (compare on the Metaculus Cup, not on open tournament/MiniBench questions: the
    rules forbid previewing forecasts on those and then changing the bot)
    uv run python bot.py --mode tournament --tier strong              # live run
"""
from __future__ import annotations

import argparse
import asyncio
import contextvars
import itertools
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from main import FallTemplateBot2026  # also loads .env and quiets noisy dependencies

from bot_helpers import check_environment, print_run_summary_banner, print_startup_banner
from forecasting_tools import (
    BinaryQuestion,
    ForecastReport,
    GeneralLlm,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    PredictedOptionList,
    ReasonedPrediction,
)
from forecasting_tools.ai_models.resource_managers.monetary_cost_manager import (
    MonetaryCostManager,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption

import bot_config
from bot_config import Tier, get_tier
from research import gather_research

logger = logging.getLogger(__name__)

# The forecaster model for the prediction task currently running. Each
# prediction runs in its own asyncio task, so each sees its own value.
_current_forecaster: contextvars.ContextVar[GeneralLlm | None] = contextvars.ContextVar(
    "current_forecaster", default=None
)

GUIDELINES_HEADING = "===== FORECASTING GUIDELINES ====="
GUIDELINES = f"""
{GUIDELINES_HEADING}
Apply these before giving your final answer:
1. This question is still open. Treat the outcome as undecided unless a dated,
   explicit source above shows the resolution criteria have already been met.
   A report of something similar, planned or rumoured is not enough.
2. Read the resolution criteria literally: exact threshold, exact dates and time
   zone, exact source. "Before <date>" means the window from today to that date.
   Do not add assumptions the criteria do not state.
3. Start from an explicit base rate: say how often this kind of outcome happened
   in comparable past cases and give the number, then adjust for current evidence.
4. Check the date of every key fact. If the research lacks the latest value of
   what is being asked, or sources disagree, say so and widen your uncertainty.
5. Use a prediction-market price only if that market's wording matches this
   question's criteria and deadline.
6. For numeric and date questions, check the units, and make the 10th-90th
   percentile range wide enough that the true value would fall outside it about
   one time in five. Stay within the stated bounds.
7. Do not go below 2% or above 98% unless the outcome is effectively settled.
"""


def date_facts(question: MetaculusQuestion, now: datetime | None = None) -> str:
    """
    Day counts worked out in code. Models miscount them: in the first live test
    one forecaster read 14 Nov 2026 -> 1 Jan 2027 as "13.5 months".
    """
    now = now or datetime.now(timezone.utc)
    today = now.date()
    lines = [f"Today is {today.isoformat()} ({now.strftime('%A')}, UTC)."]

    def add(label: str, moment: datetime | None) -> None:
        if moment is None:
            return
        days = (moment.date() - today).days
        lines.append(f"- {label}: {moment.date().isoformat()} ({days} days from today)")

    add("This question closes to new forecasts", getattr(question, "close_time", None))
    add("Scheduled resolution", getattr(question, "scheduled_resolution_time", None))
    for year in (today.year + 1, today.year + 2):
        start = datetime(year, 1, 1, tzinfo=timezone.utc)
        lines.append(f"- 1 January {year}: {(start.date() - today).days} days from today")
    lines.append(
        "Use these counts. For any other deadline in the resolution criteria, count the "
        "days from today's date explicitly before reasoning about the time left."
    )
    return "===== DATES (computed, exact) =====\n" + "\n".join(lines)


def clip_binary(probability: float) -> float:
    return max(bot_config.BINARY_MIN, min(bot_config.BINARY_MAX, probability))


def floor_options(prediction: PredictedOptionList) -> PredictedOptionList:
    """Give every option at least the floor probability, then renormalise."""
    floored = [
        max(bot_config.MULTIPLE_CHOICE_FLOOR, option.probability)
        for option in prediction.predicted_options
    ]
    total = sum(floored)
    return PredictedOptionList(
        predicted_options=[
            PredictedOption(option_name=option.option_name, probability=value / total)
            for option, value in zip(prediction.predicted_options, floored)
        ]
    )


class EnsembleBot(FallTemplateBot2026):
    """Template bot with multi-source research and a multi-model median ensemble."""

    _max_concurrent_questions = 2
    _research_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *, tier: Tier, **kwargs: Any) -> None:
        self.tier = tier
        self._forecasters = [
            GeneralLlm(
                model=model,
                temperature=0.3,
                timeout=tier.forecaster_timeout,
                allowed_tries=2,
            )
            for model in tier.forecasters
        ]
        self._prediction_counters: dict[int, itertools.count] = {}
        utility = GeneralLlm(model=tier.utility, temperature=0.1, timeout=90, allowed_tries=2)
        kwargs.setdefault("research_reports_per_question", 1)
        kwargs["predictions_per_research_report"] = len(self._forecasters)
        kwargs["llms"] = {
            "default": self._forecasters[0],
            "summarizer": utility,
            "parser": utility,
            # Unused: run_research is overridden. Set so the library does not warn.
            "researcher": tier.searchers[0] if tier.searchers else "no_research",
        }
        super().__init__(**kwargs)

    # ------------------------------ research ------------------------------

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._research_limiter:
            research = await gather_research(question, self.tier)
        logger.info(f"Research for {question.page_url}:\n{research}")
        return research

    # ------------------------------ ensemble ------------------------------

    def get_llm(self, purpose: str = "default", guarantee_type: Any = None) -> Any:
        forecaster = _current_forecaster.get()
        if purpose == "default" and forecaster is not None:
            return forecaster.model if guarantee_type == "string_name" else forecaster
        return super().get_llm(purpose, guarantee_type)

    def _next_forecaster(self, question: MetaculusQuestion) -> GeneralLlm:
        key = question.id_of_question or id(question)
        counter = self._prediction_counters.setdefault(key, itertools.count())
        return self._forecasters[next(counter) % len(self._forecasters)]

    async def _make_prediction(
        self, question: MetaculusQuestion, research: str
    ) -> ReasonedPrediction[Any]:
        forecaster = self._next_forecaster(question)
        if GUIDELINES_HEADING not in research:
            research = f"{research}\n\n{date_facts(question)}\n{GUIDELINES}"
        token = _current_forecaster.set(forecaster)
        try:
            try:
                prediction = await super()._make_prediction(question, research)
            except Exception as error:  # noqa: BLE001
                # Small models sometimes write the reasoning and leave out the final
                # answer line, which cannot be parsed. Ask the same model once more
                # rather than lose one of the three forecasts.
                logger.warning(
                    f"{forecaster.model} gave an unusable forecast for {question.page_url} "
                    f"({type(error).__name__}); retrying once"
                )
                prediction = await super()._make_prediction(question, research)
        finally:
            _current_forecaster.reset(token)

        value = prediction.prediction_value
        if isinstance(question, BinaryQuestion) and isinstance(value, float):
            value = clip_binary(value)
        elif isinstance(question, MultipleChoiceQuestion) and isinstance(
            value, PredictedOptionList
        ):
            value = floor_options(value)
        return ReasonedPrediction(
            prediction_value=value,
            reasoning=f"**Forecaster model: {forecaster.model}**\n\n{prediction.reasoning}",
        )


# ----------------------------------------------------------------------------
# Recording forecasts (for comparing variants once questions resolve)
# ----------------------------------------------------------------------------


def _serialize_prediction(prediction: Any) -> dict[str, Any]:
    if isinstance(prediction, (float, int)):
        return {"kind": "binary", "probability": float(prediction)}
    if isinstance(prediction, PredictedOptionList):
        return {
            "kind": "multiple_choice",
            "options": {o.option_name: o.probability for o in prediction.predicted_options},
        }
    if isinstance(prediction, NumericDistribution):
        record: dict[str, Any] = {
            "kind": "numeric",
            "declared_percentiles": [
                [p.percentile, p.value] for p in prediction.declared_percentiles
            ],
        }
        try:
            record["cdf"] = [[p.value, p.percentile] for p in prediction.cdf]
        except Exception as error:  # noqa: BLE001
            record["cdf_error"] = f"{type(error).__name__}: {error}"
        return record
    return {"kind": type(prediction).__name__, "text": str(prediction)}


def _records_path(save_dir: Path, variant: str, tier: Tier) -> Path:
    return save_dir / f"{variant}-{tier.name}.jsonl"


def _already_recorded(path: Path) -> set[int]:
    if not path.exists():
        return set()
    seen: set[int] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            seen.add(int(json.loads(line)["question_id"]))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError):
            continue
    return seen


def _dry_run_question_ids(save_dir: Path) -> set[int]:
    """Questions that any earlier run forecast without publishing."""
    seen: set[int] = set()
    for path in save_dir.glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                record = json.loads(line)
                if not record.get("published"):
                    seen.add(int(record["question_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError):
                continue
    return seen


def _append_records(
    path: Path, reports: list[Any], variant: str, tier: Tier, published: bool
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        for report in reports:
            if not isinstance(report, ForecastReport):
                continue
            question = report.question
            record = {
                "recorded_at": datetime.now(timezone.utc).isoformat(),
                "variant": variant,
                "tier": tier.name,
                "published": published,
                "question_id": question.id_of_question,
                "post_id": question.id_of_post,
                "url": question.page_url,
                "title": question.question_text,
                "question_type": type(question).__name__,
                "close_time": question.close_time.isoformat() if question.close_time else None,
                "scheduled_resolution_time": (
                    question.scheduled_resolution_time.isoformat()
                    if question.scheduled_resolution_time
                    else None
                ),
                "prediction": _serialize_prediction(report.prediction),
                "cost_usd": report.price_estimate,
                "minutes": report.minutes_taken,
                "errors": report.errors,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")


# ----------------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------------

Mode = Literal["tournament", "main", "minibench", "metaculus_cup", "test_questions"]


def _tournament_ids(mode: Mode, client: MetaculusClient) -> list[int | str]:
    if mode == "tournament":
        return [client.CURRENT_AI_COMPETITION_ID, client.CURRENT_MINIBENCH_ID]
    if mode == "main":
        return [client.CURRENT_AI_COMPETITION_ID]
    if mode == "minibench":
        return [client.CURRENT_MINIBENCH_ID]
    if mode == "metaculus_cup":
        return [client.CURRENT_METACULUS_CUP_ID]
    return ["bot-testing-area"]


def _build_bot(variant: str, tier: Tier, publish: bool, skip_forecasted: bool) -> Any:
    common = dict(
        research_reports_per_question=1,
        use_research_summary_to_forecast=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=skip_forecasted,
        extra_metadata_in_explanation=True,
    )
    if variant == "ensemble":
        return EnsembleBot(tier=tier, **common)
    # Baseline: the unchanged template, with this tier's first forecaster and
    # first search model, so the comparison isolates the pipeline changes.
    utility = GeneralLlm(model=tier.utility, temperature=0.1, timeout=90, allowed_tries=2)
    return FallTemplateBot2026(
        predictions_per_research_report=5,
        llms={
            "default": GeneralLlm(
                model=tier.forecasters[0],
                temperature=0.3,
                timeout=tier.forecaster_timeout,
                allowed_tries=2,
            ),
            "summarizer": utility,
            "parser": utility,
            "researcher": (
                GeneralLlm(model=tier.searchers[0], temperature=0.1)
                if tier.searchers
                else "no_research"
            ),
        },
        **common,
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    parser = argparse.ArgumentParser(description="Run the ensemble forecasting bot")
    parser.add_argument(
        "--mode",
        choices=["tournament", "main", "minibench", "metaculus_cup", "test_questions"],
        default="tournament",
        help="tournament = seasonal tournament + MiniBench (default)",
    )
    parser.add_argument("--tier", default=None, help="free, cheap or strong (default: $BOT_TIER or cheap)")
    parser.add_argument(
        "--variant",
        choices=["ensemble", "baseline", "both"],
        default="ensemble",
        help="'both' runs the template baseline and the ensemble on the same questions and never publishes",
    )
    parser.add_argument("--no-publish", action="store_true", help="Forecast but do not post to Metaculus")
    parser.add_argument("--limit", type=int, default=None, help="Forecast at most this many questions")
    parser.add_argument(
        "--max-cost",
        type=float,
        default=float(os.getenv("BOT_MAX_COST_PER_RUN", "2.0")),
        help="Stop calling models once the run has spent this many USD",
    )
    parser.add_argument("--save-dir", default="runs", help="Folder for the forecast records")
    args = parser.parse_args()

    tier = get_tier(args.tier)
    variants = ["baseline", "ensemble"] if args.variant == "both" else [args.variant]
    publish = not args.no_publish and args.variant != "both"
    save_dir = Path(args.save_dir)

    check_environment(strict=True)
    print_startup_banner(args.mode, will_publish=publish)
    print(f"    tier={tier.name}  variants={variants}  max cost per run=${args.max_cost:.2f}\n")

    client = MetaculusClient()
    questions: list[MetaculusQuestion] = []
    for tournament_id in _tournament_ids(args.mode, client):
        questions.extend(client.get_all_open_questions_from_tournament(tournament_id))

    # Live runs skip what the bot already forecast on Metaculus. Dry runs skip
    # what is already in the record file, so repeating a dry run costs nothing.
    # The testing area is the exception: it exists to be rerun.
    scored_mode = args.mode in ("tournament", "main", "minibench")
    skip_forecasted = publish and args.mode != "test_questions"
    if skip_forecasted:
        questions = [q for q in questions if not q.already_forecasted]
    if publish and scored_mode:
        # Tournament rule: do not preview a forecast on an open question and then
        # publish a different one. A question seen in a dry run is never published.
        # Only scored modes: the testing area and the Metaculus Cup may be rerun.
        previewed = _dry_run_question_ids(save_dir)
        skipped = [q for q in questions if q.id_of_question in previewed]
        if skipped:
            print(f"    Skipping {len(skipped)} question(s) already forecast in a dry run.\n")
        questions = [q for q in questions if q.id_of_question not in previewed]
    if not publish:
        done = set.intersection(
            *[_already_recorded(_records_path(save_dir, v, tier)) for v in variants]
        )
        questions = [q for q in questions if q.id_of_question not in done]
    if args.limit is not None:
        questions = questions[: args.limit]

    async def run_variants() -> list[Any]:
        # One event loop for the whole run: the bots' semaphores bind to a loop.
        collected: list[Any] = []
        for variant in variants:
            bot = _build_bot(variant, tier, publish, skip_forecasted)
            reports = await bot.forecast_questions(questions, return_exceptions=True)
            _append_records(_records_path(save_dir, variant, tier), reports, variant, tier, publish)
            bot.log_report_summary(reports)
            collected.extend(reports)
        return collected

    with MonetaryCostManager(args.max_cost) as cost_manager:
        all_reports = asyncio.run(run_variants())
        total_cost = cost_manager.current_usage

    print_run_summary_banner(all_reports, will_publish=publish)
    forecast_count = sum(isinstance(r, ForecastReport) for r in all_reports)
    if forecast_count:
        print(f"💵  Cost: ${total_cost:.3f} total, ${total_cost / forecast_count:.3f} per forecast\n")


if __name__ == "__main__":
    main()
