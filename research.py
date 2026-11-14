"""
Multi-source research for the ensemble bot.

Pass 1 (in parallel): AskNews, one or more web-searching models, the pages the
question names as its resolution source, and prediction-market prices.
Pass 2 (gap fill): a cheap model reads pass 1, lists the facts still missing
for a forecast, and each one is searched for.

Every source is optional and failure-tolerant: a source that errors or times
out is logged and left out, and the bot still forecasts with the rest.
The output is the raw text of each source under its own heading. It is not
rewritten by a model, because a summariser can misstate what a source said.
"""
from __future__ import annotations

import asyncio
import html
import json
import logging
import os
import re
from datetime import datetime, timezone
from typing import Awaitable

import httpx
from forecasting_tools import AskNewsSearcher, GeneralLlm, MetaculusQuestion, clean_indents

from bot_config import Tier

logger = logging.getLogger(__name__)

SOURCE_TIMEOUT_SECONDS = 150
PAGE_TIMEOUT_SECONDS = 20
MAX_RESOLUTION_PAGES = 3
MAX_PAGE_CHARS = 8000
MAX_BACKGROUND_PAGE_CHARS = 2500
MAX_MARKETS_PER_SITE = 5
_BROWSER_HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _question_block(question: MetaculusQuestion) -> str:
    return clean_indents(
        f"""
        Question: {question.question_text}

        Resolution criteria:
        {question.resolution_criteria}

        {question.fine_print or ""}

        Background:
        {question.background_info or ""}
        """
    )


ONLINE_SUFFIX = ":online"


def _llm(model: str, timeout: int = 120) -> GeneralLlm:
    """
    Build a model. A name ending in ":online" (see bot_config.py) means "attach
    web search". The bare OpenRouter ":online" suffix routes searches to Exa,
    so the suffix is removed and the provider's own search engine is requested.
    """
    if model.endswith(ONLINE_SUFFIX):
        return GeneralLlm(
            model=model.removesuffix(ONLINE_SUFFIX),
            temperature=0.1,
            timeout=timeout,
            allowed_tries=2,
            extra_body={"plugins": [{"id": "web", "engine": "native"}]},
        )
    return GeneralLlm(model=model, temperature=0.1, timeout=timeout, allowed_tries=2)


async def _safe(label: str, coroutine: Awaitable[str]) -> tuple[str, str] | None:
    """Run one source. Returns (label, text), or None if it failed or was empty."""
    try:
        text = await asyncio.wait_for(coroutine, timeout=SOURCE_TIMEOUT_SECONDS)
    except Exception as error:  # noqa: BLE001 - a failed source must not stop the bot
        logger.warning(f"Research source '{label}' failed: {type(error).__name__}: {error}")
        return None
    text = (text or "").strip()
    if not text:
        logger.info(f"Research source '{label}' returned nothing")
        return None
    return label, text


def extract_json_object(text: str) -> dict:
    """Pull the first JSON object out of a model reply. Returns {} if there is none."""
    match = re.search(r"\{.*\}", text or "", flags=re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _clean_string_list(value: object, limit: int) -> list[str]:
    if not isinstance(value, list):
        return []
    cleaned = [str(item).strip() for item in value if str(item).strip()]
    return cleaned[:limit]


# ----------------------------------------------------------------------------
# Source: AskNews
# ----------------------------------------------------------------------------


def asknews_is_configured() -> bool:
    has_pair = bool(os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"))
    return has_pair or bool(os.getenv("ASKNEWS_API_KEY"))


async def _asknews(question: MetaculusQuestion) -> str:
    return await AskNewsSearcher().call_preconfigured_version(
        "asknews/news-summaries", question.question_text
    )


# ----------------------------------------------------------------------------
# Source: web-searching models
# ----------------------------------------------------------------------------


def _search_prompt(question: MetaculusQuestion) -> str:
    return clean_indents(
        f"""
        You are a research assistant to a professional forecaster. Today is {_today()} (UTC).
        Search the web and report what is known that bears on the question below.

        {_question_block(question)}

        Report, in this order:
        1. Current status: the most recent value, count, standing or state of whatever
           the question measures, with the date it refers to.
        2. Scheduled events, deadlines or decisions between today and the question's deadline.
        3. History: how often this kind of outcome happened in comparable past cases
           (a base rate), with the cases or numbers behind it.
        4. Recent news from the last few weeks, most recent first.
        5. What experts, officials, analysts or polls currently expect.

        Search for the event itself. Do not report on forecasting sites, betting sites
        or prediction markets as a topic: those prices are collected separately.

        Rules:
        - Give the date and the source of every fact, naming the source in words
          (publisher and URL), not only with a citation number.
        - Quote exact numbers and units as the source gives them.
        - If you cannot find something, write "not found". Never guess or fill in from memory
          without saying so.
        - Say explicitly whether the resolution criteria appear to be already met, and cite
          the dated source if so.
        - Do not give a forecast or a probability of your own.
        """
    )


async def _search_model(model: str, prompt: str) -> str:
    return await _llm(model).invoke(prompt)


async def _gemini_grounded(prompt: str) -> str:
    """Gemini with Google Search grounding, through a Google AI Studio key."""
    model = os.getenv("GEMINI_SEARCH_MODEL", "gemini/gemini-2.5-flash")
    return await GeneralLlm.grounded_model(model).invoke(prompt)


# ----------------------------------------------------------------------------
# Source: pages named in the resolution criteria
# ----------------------------------------------------------------------------

_URL_PATTERN = re.compile(r"https?://[^\s<>\]\"']+")


def _clean_url(url: str) -> str:
    """Undo markdown escapes and drop punctuation that is not part of the URL.

    Parentheses are kept when balanced (Wikipedia titles use them) and dropped
    when they only close a markdown link or a sentence in brackets.
    """
    url = url.replace("\\(", "(").replace("\\)", ")").replace("\\_", "_")
    url = url.rstrip(".,;:!?")
    while url.endswith(")") and url.count(")") > url.count("("):
        url = url[:-1].rstrip(".,;:!?")
    return url


def _question_urls(question: MetaculusQuestion) -> list[tuple[str, bool]]:
    """(url, is_background_only), resolution criteria and fine print first."""
    texts = [
        (question.resolution_criteria, False),
        (question.fine_print, False),
        (question.background_info, True),
    ]
    found: list[tuple[str, bool]] = []
    for text, is_background in texts:
        for url in _URL_PATTERN.findall(text or ""):
            url = _clean_url(url)
            if "metaculus.com" in url or any(url == seen for seen, _ in found):
                continue
            found.append((url, is_background))
    return found[:MAX_RESOLUTION_PAGES]


def find_resolution_urls(question: MetaculusQuestion) -> list[str]:
    """URLs in the resolution criteria and fine print first, then the background."""
    return [url for url, _ in _question_urls(question)]


def html_to_text(raw_html: str) -> str:
    # Keep only the main content when the page marks it, so the character budget
    # is not spent on menus.
    main_content = re.search(r"(?is)<(main|article)\b.*?</\1>", raw_html)
    if main_content and len(main_content.group(0)) > 2000:
        raw_html = main_content.group(0)
    text = re.sub(
        r"(?is)<(script|style|noscript|svg|head|nav|header|footer|aside|form)\b.*?</\1>",
        " ",
        raw_html,
    )
    text = re.sub(r"(?s)<!--.*?-->", " ", text)
    text = re.sub(r"(?i)<(br|/p|/div|/tr|/li|/h[1-6])\s*/?>", "\n", text)
    text = re.sub(r"(?s)<[^>]+>", " ", text)
    text = html.unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n\s*\n+", "\n", text)
    return text.strip()


async def _fetch_page(
    client: httpx.AsyncClient, url: str, max_chars: int = MAX_PAGE_CHARS
) -> str | None:
    try:
        response = await client.get(url)
        response.raise_for_status()
    except Exception as error:  # noqa: BLE001
        logger.info(f"Could not fetch resolution page {url}: {type(error).__name__}: {error}")
        return None
    content_type = response.headers.get("content-type", "")
    if "html" in content_type:
        text = html_to_text(response.text)
    elif "json" in content_type or content_type.startswith("text/"):
        text = response.text.strip()
    else:
        logger.info(f"Skipping resolution page {url}: unsupported type '{content_type}'")
        return None
    if len(text) < 200:  # Usually a page that needs JavaScript or a login.
        return None
    if len(text) > max_chars:
        text = text[:max_chars] + "\n[...page truncated...]"
    return f"URL: {url}\n{text}"


async def _resolution_pages(question: MetaculusQuestion) -> str:
    urls = _question_urls(question)
    if not urls:
        return ""
    async with httpx.AsyncClient(
        headers=_BROWSER_HEADERS, timeout=PAGE_TIMEOUT_SECONDS, follow_redirects=True
    ) as client:
        # Pages linked only from the background are context (often old articles),
        # not the resolution source, so they get a smaller share of the prompt.
        pages = await asyncio.gather(
            *[
                _fetch_page(
                    client, url, MAX_BACKGROUND_PAGE_CHARS if is_background else MAX_PAGE_CHARS
                )
                for url, is_background in urls
            ]
        )
    found = [page for page in pages if page]
    if not found:
        return ""
    note = (
        f"Text of pages linked from the question, fetched {_today()}. "
        "Plain text only: tables, charts and scripted content may be missing."
    )
    return note + "\n\n" + "\n\n---\n\n".join(found)


# ----------------------------------------------------------------------------
# Source: prediction markets
# ----------------------------------------------------------------------------


async def _plan_market_queries(question: MetaculusQuestion, utility_model: str) -> list[str]:
    prompt = clean_indents(
        f"""
        Give 2 short keyword searches (2 to 5 words each, no dates, no question marks)
        that would find prediction markets about the same event as this question:

        {question.question_text}

        Answer with JSON only: {{"queries": ["...", "..."]}}
        """
    )
    reply = await _llm(utility_model, timeout=60).invoke(prompt)
    return _clean_string_list(extract_json_object(reply).get("queries"), 2)


async def _manifold(client: httpx.AsyncClient, term: str) -> list[str]:
    response = await client.get(
        "https://api.manifold.markets/v0/search-markets",
        params={"term": term, "limit": MAX_MARKETS_PER_SITE, "sort": "score"},
    )
    response.raise_for_status()
    lines = []
    for market in response.json():
        probability = market.get("probability")
        if probability is None or market.get("isResolved"):
            continue
        lines.append(
            f"- Manifold: \"{market.get('question')}\" -> {probability:.0%} "
            f"({market.get('uniqueBettorCount', 0)} traders) {market.get('url', '')}"
        )
    return lines


async def _polymarket(client: httpx.AsyncClient, term: str) -> list[str]:
    response = await client.get(
        "https://gamma-api.polymarket.com/public-search",
        params={"q": term, "limit_per_type": MAX_MARKETS_PER_SITE, "events_status": "active"},
    )
    response.raise_for_status()
    lines = []
    for event in (response.json().get("events") or [])[:MAX_MARKETS_PER_SITE]:
        for market in (event.get("markets") or [])[:6]:
            if market.get("closed"):
                continue
            try:
                outcomes = json.loads(market.get("outcomes") or "[]")
                prices = json.loads(market.get("outcomePrices") or "[]")
            except (TypeError, json.JSONDecodeError):
                continue
            if not outcomes or len(outcomes) != len(prices):
                continue
            priced = ", ".join(
                f"{outcome} {float(price):.0%}" for outcome, price in zip(outcomes, prices)
            )
            lines.append(
                f"- Polymarket: \"{market.get('question')}\" -> {priced} "
                f"(closes {str(market.get('endDate', ''))[:10]})"
            )
    return lines


async def _prediction_markets(question: MetaculusQuestion, utility_model: str) -> str:
    terms = await _plan_market_queries(question, utility_model)
    if not terms:
        return ""
    async with httpx.AsyncClient(timeout=PAGE_TIMEOUT_SECONDS, headers=_BROWSER_HEADERS) as client:
        searches = [site(client, term) for term in terms for site in (_manifold, _polymarket)]
        results = await asyncio.gather(*searches, return_exceptions=True)
    lines: list[str] = []
    for result in results:
        if isinstance(result, BaseException):
            logger.info(f"Prediction market search failed: {type(result).__name__}: {result}")
            continue
        for line in result:
            if line not in lines:
                lines.append(line)
    lines = await _relevant_markets(question, lines[:30], utility_model)
    if not lines:
        return ""
    note = (
        f"Open markets found by keyword search on {_today()} (search terms: {', '.join(terms)}). "
        "They can differ from the question in event, deadline or threshold: "
        "use a price only if that market's wording matches the resolution criteria."
    )
    return note + "\n" + "\n".join(lines)


async def _relevant_markets(
    question: MetaculusQuestion, lines: list[str], utility_model: str
) -> list[str]:
    """
    Keyword search returns mostly unrelated markets (the first live test listed
    eurozone-inflation markets for a question about leaving the EU). Keep only
    the ones a model judges to be about the same or a closely related event.
    If the filter itself fails, keep nothing: noise is worse than no prices.
    """
    if not lines:
        return []
    numbered = "\n".join(f"{index}. {line}" for index, line in enumerate(lines, start=1))
    prompt = clean_indents(
        f"""
        A forecaster is answering this question:

        {question.question_text}

        Below are prediction markets found by keyword search. Pick the ones about the
        same event, or about an event whose outcome directly informs it (same subject,
        possibly a different deadline or threshold). Leave out everything else.

        {numbered}

        Answer with JSON only: {{"relevant": [numbers]}}. Use an empty list if none qualify.
        """
    )
    reply = await _llm(utility_model, timeout=60).invoke(prompt)
    picked = extract_json_object(reply).get("relevant")
    if not isinstance(picked, list):
        return []
    kept: list[str] = []
    for item in picked:
        try:
            index = int(item)
        except (TypeError, ValueError):
            continue
        if 1 <= index <= len(lines) and lines[index - 1] not in kept:
            kept.append(lines[index - 1])
    return kept[:10]


# ----------------------------------------------------------------------------
# Pass 2: gap fill
# ----------------------------------------------------------------------------


async def _plan_gap_queries(
    question: MetaculusQuestion, research_so_far: str, tier: Tier
) -> list[str]:
    prompt = clean_indents(
        f"""
        Today is {_today()} (UTC). A forecaster must now answer this question using only
        the research below.

        {_question_block(question)}

        ===== RESEARCH SO FAR =====
        {research_so_far}
        ===== END OF RESEARCH =====

        Imagine writing the forecast right now. Which specific facts are still missing,
        out of date, or contradictory between sources, and would change the forecast most?
        Typical gaps: the latest value of the quantity being asked about, an exact date
        of a scheduled event, a historical base rate, or what the resolution source
        currently shows.

        Write up to {tier.max_gap_queries} web searches that would fill those gaps. Each
        must stand alone (name the entities and dates; do not say "the question").
        If nothing important is missing, return an empty list.

        Answer with JSON only: {{"queries": ["...", "..."]}}
        """
    )
    reply = await _llm(tier.utility, timeout=90).invoke(prompt)
    return _clean_string_list(extract_json_object(reply).get("queries"), tier.max_gap_queries)


async def _answer_gap_query(model: str, query: str) -> str:
    prompt = clean_indents(
        f"""
        Today is {_today()} (UTC). Search the web and answer this as precisely as you can:

        {query}

        Give exact numbers, dates and units, and the date and source of each fact, naming
        the source in words (publisher and URL), not only with a citation number.
        If you cannot find it, write "not found" once. Do not guess.
        """
    )
    answer = await _llm(model).invoke(prompt)
    return f"Query: {query}\n{answer.strip()}"


async def _gap_fill(question: MetaculusQuestion, research_so_far: str, tier: Tier) -> str:
    if not tier.gap_searcher or tier.max_gap_queries <= 0:
        return ""
    queries = await _plan_gap_queries(question, research_so_far, tier)
    if not queries:
        return ""
    answers = await asyncio.gather(
        *[_answer_gap_query(tier.gap_searcher, query) for query in queries],
        return_exceptions=True,
    )
    good = [answer for answer in answers if isinstance(answer, str)]
    return "\n\n".join(good)


# ----------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------


def _format(sections: list[tuple[str, str]]) -> str:
    header = (
        f"Research gathered on {_today()} (UTC) from {len(sections)} independent source(s). "
        "Sources can disagree or be out of date: prefer dated primary sources and the "
        "question's own resolution source."
    )
    body = "\n\n".join(f"### Source: {label}\n{text}" for label, text in sections)
    return f"{header}\n\n{body}"


async def gather_research(question: MetaculusQuestion, tier: Tier) -> str:
    search_prompt = _search_prompt(question)

    first_pass: list[Awaitable[tuple[str, str] | None]] = []
    if asknews_is_configured():
        first_pass.append(_safe("AskNews (news articles)", _asknews(question)))
    for model in tier.searchers:
        label = f"Web search by {model.removeprefix('openrouter/')}"
        first_pass.append(_safe(label, _search_model(model, search_prompt)))
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        first_pass.append(_safe("Google Search (Gemini grounding)", _gemini_grounded(search_prompt)))
    first_pass.append(_safe("Pages linked from the question", _resolution_pages(question)))
    first_pass.append(
        _safe("Prediction markets", _prediction_markets(question, tier.utility))
    )

    sections = [section for section in await asyncio.gather(*first_pass) if section]

    if sections:
        gap_section = await _safe(
            "Follow-up searches on missing facts",
            _gap_fill(question, _format(sections), tier),
        )
        if gap_section:
            sections.append(gap_section)

    if not sections:
        logger.warning(f"No research source returned anything for {question.page_url}")
        return (
            "No research could be gathered for this question. Forecast from the question "
            "text and general knowledge, and widen the uncertainty accordingly."
        )
    return _format(sections)
