"""Indian financial news via RSS feeds.

RSS-first strategy (stable, no scraping): MoneyControl, ET Markets,
Business Standard, LiveMint. Company-name keyword filtering via yfinance
`longName` so "RELIANCE.NS" matches "Reliance Industries" in headlines.

Primary signal for the News Analyst. Articles are deduplicated by title and
ordered newest-first. Falls back gracefully — missing feed → "data
unavailable" for that source, not a hard error.
"""

from __future__ import annotations

import html
import logging
import re
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta, timezone
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import yfinance as yf

from ..config import get_config
from ..symbol_utils import normalize_symbol

logger = logging.getLogger(__name__)

_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_TIMEOUT = 12  # seconds per RSS fetch

# RSS feeds: (source_name, url, category)
_FEEDS = [
    ("MoneyControl Markets", "https://www.moneycontrol.com/rss/business.xml", "markets"),
    ("ET Markets", "https://economictimes.indiatimes.com/markets/rss.cms", "markets"),
    ("Business Standard", "https://www.business-standard.com/rss/markets-106.rss", "markets"),
    ("LiveMint", "https://www.livemint.com/rss/markets", "markets"),
    ("MoneyControl News", "https://www.moneycontrol.com/rss/latestnews.xml", "news"),
    ("ET Economy", "https://economictimes.indiatimes.com/news/economy/rss.cms", "macro"),
]

# Global / macro RSS feeds for the News Analyst's macro context
_MACRO_FEEDS = [
    ("ET Economy", "https://economictimes.indiatimes.com/news/economy/rss.cms", "macro"),
    ("Business Standard Economy", "https://www.business-standard.com/rss/economy-policy-101.rss", "macro"),
    ("LiveMint Economy", "https://www.livemint.com/rss/economy", "macro"),
    ("MoneyControl Economy", "https://www.moneycontrol.com/rss/economy.xml", "macro"),
]

_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_rss(url: str) -> Optional[ET.Element]:
    """Fetch and parse an RSS/Atom feed; return root element or None."""
    req = Request(url, headers={"User-Agent": _UA, "Accept": "application/xml, */*"})
    try:
        with urlopen(req, timeout=_TIMEOUT) as resp:
            return ET.fromstring(resp.read())
    except (HTTPError, URLError, TimeoutError, ET.ParseError) as exc:
        logger.debug("RSS fetch failed for %s: %s", url, exc)
        return None


def _parse_items(root: ET.Element) -> list[dict]:
    """Parse RSS or Atom feed items into a list of dicts."""
    items: list[dict] = []

    # RSS 2.0
    for item in root.findall(".//item"):
        title_el = item.find("title")
        desc_el = item.find("description")
        link_el = item.find("link")
        pub_el = item.find("pubDate")
        title = _clean(title_el.text if title_el is not None else "")
        desc = _clean(desc_el.text if desc_el is not None else "")
        link = (link_el.text or "").strip() if link_el is not None else ""
        pub_date = _parse_rss_date(pub_el.text if pub_el is not None else None)
        if title:
            items.append({"title": title, "desc": desc, "link": link, "pub_date": pub_date})

    # Atom
    for entry in root.findall("atom:entry", _ATOM_NS):
        title_el = entry.find("atom:title", _ATOM_NS)
        summary_el = entry.find("atom:summary", _ATOM_NS)
        link_el = entry.find("atom:link[@rel='alternate']", _ATOM_NS) or entry.find("atom:link", _ATOM_NS)
        pub_el = entry.find("atom:published", _ATOM_NS) or entry.find("atom:updated", _ATOM_NS)
        title = _clean(title_el.text if title_el is not None else "")
        desc = _clean(summary_el.text if summary_el is not None else "")
        link = (link_el.get("href") or "") if link_el is not None else ""
        pub_date = _parse_atom_date(pub_el.text if pub_el is not None else None)
        if title:
            items.append({"title": title, "desc": desc, "link": link, "pub_date": pub_date})

    return items


def _clean(text: str) -> str:
    if not text:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    return " ".join(text.split())


def _parse_rss_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %z", "%a, %d %b %Y %H:%M:%S GMT"):
        try:
            return datetime.strptime(s.strip(), fmt)
        except ValueError:
            pass
    return None


def _parse_atom_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        norm = s.strip()
        if norm.endswith("Z"):
            norm = norm[:-1] + "+00:00"
        return datetime.fromisoformat(norm)
    except ValueError:
        return None


def _resolve_company_name(ticker: str) -> str:
    """Return longName for ticker via yfinance; fallback to bare symbol."""
    try:
        canonical = normalize_symbol(ticker)
        if not canonical.endswith(".NS") and not canonical.endswith(".BO"):
            canonical = canonical + ".NS"
        info = yf.Ticker(canonical).info or {}
        return info.get("longName") or info.get("shortName") or ""
    except Exception:
        return ""


def _keyword_match(title: str, desc: str, keywords: list[str]) -> bool:
    haystack = (title + " " + desc).lower()
    return any(kw.lower() in haystack for kw in keywords)


def _format_item(item: dict, source: str) -> str:
    pub = ""
    if item["pub_date"]:
        pub = item["pub_date"].strftime("%Y-%m-%d")
    lines = [f"### {item['title']} [{pub}] (source: {source})"]
    if item["desc"]:
        snippet = item["desc"][:300] + ("…" if len(item["desc"]) > 300 else "")
        lines.append(snippet)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_india_news(
    ticker: str,
    start_date: str,
    end_date: str,
    article_limit: Optional[int] = None,
) -> str:
    """Fetch Indian financial news articles mentioning *ticker*.

    Searches all configured Indian news RSS feeds and filters by company name
    and/or ticker symbol. Articles outside [start_date, end_date] are dropped.

    Returns a formatted string for agent injection. Empty result → explicit
    "no news found" message rather than a blank string.
    """
    cfg = get_config()
    limit = article_limit or cfg.get("news_article_limit", 20)

    start_dt = datetime.strptime(start_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end_dt = (
        datetime.strptime(end_date, "%Y-%m-%d") + timedelta(days=1)
    ).replace(tzinfo=timezone.utc)

    # Build keyword list: ticker (bare), ticker.NS, company name words
    bare = ticker.upper().rstrip(".NS").rstrip(".BO").replace(".NS", "").replace(".BO", "")
    company_name = _resolve_company_name(ticker)
    keywords = [bare]
    if company_name:
        # Add first two significant words of the company name
        words = [w for w in company_name.split() if len(w) > 3]
        keywords += words[:2]

    collected: list[tuple[Optional[datetime], str, str]] = []  # (date, formatted, title)
    seen_titles: set[str] = set()

    for source_name, url, _ in _FEEDS:
        if len(collected) >= limit:
            break
        root = _fetch_rss(url)
        if root is None:
            continue
        items = _parse_items(root)
        for item in items:
            if item["title"] in seen_titles:
                continue
            # Date filter (skip if pub_date present but out of range)
            pd_dt = item["pub_date"]
            if pd_dt:
                if pd_dt.tzinfo is None:
                    pd_dt = pd_dt.replace(tzinfo=timezone.utc)
                if pd_dt > end_dt or pd_dt < start_dt:
                    continue
            if not _keyword_match(item["title"], item["desc"], keywords):
                continue
            seen_titles.add(item["title"])
            collected.append((pd_dt, _format_item(item, source_name), item["title"]))
            if len(collected) >= limit:
                break

    if not collected:
        return (
            f"No Indian news found for {ticker} ({company_name or bare}) "
            f"between {start_date} and {end_date} in monitored RSS feeds."
        )

    # Sort newest-first
    collected.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    body = "\n\n".join(fmt for _, fmt, _ in collected)
    return f"## Indian Market News for {ticker} ({company_name or bare}), {start_date} to {end_date}:\n\n{body}"


def get_india_global_news(
    curr_date: str,
    look_back_days: Optional[int] = None,
    limit: Optional[int] = None,
) -> str:
    """Fetch Indian macro / global news from Indian financial news RSS feeds.

    Used by the News Analyst for macroeconomic context when market_profile is
    "india". Covers RBI, SEBI, Nifty outlook, FII flows, INR/USD, etc.
    """
    cfg = get_config()
    lb_days = look_back_days or cfg.get("global_news_lookback_days", 7)
    art_limit = limit or cfg.get("global_news_article_limit", 10)

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start_dt = (curr_dt - timedelta(days=lb_days))

    collected: list[tuple[Optional[datetime], str]] = []
    seen_titles: set[str] = set()

    # Use India-specific macro queries as keyword filters
    india_queries = cfg.get("global_news_queries_india") or [
        "RBI", "Nifty", "Sensex", "FII", "DII", "SEBI", "India GDP", "inflation",
        "INR", "rupee", "monsoon", "Budget",
    ]

    for source_name, url, _ in _MACRO_FEEDS:
        if len(collected) >= art_limit:
            break
        root = _fetch_rss(url)
        if root is None:
            continue
        items = _parse_items(root)
        for item in items:
            if item["title"] in seen_titles:
                continue
            pd_dt = item["pub_date"]
            if pd_dt:
                if pd_dt.tzinfo is None:
                    pd_dt = pd_dt.replace(tzinfo=timezone.utc)
                if pd_dt > curr_dt + timedelta(days=1) or pd_dt < start_dt:
                    continue
            if not _keyword_match(item["title"], item["desc"], india_queries):
                continue
            seen_titles.add(item["title"])
            collected.append((pd_dt, _format_item(item, source_name)))
            if len(collected) >= art_limit:
                break

    if not collected:
        return (
            f"No Indian macro news found for the {lb_days} days ending {curr_date} "
            "in monitored RSS feeds. "
            "Check MoneyControl/ET Markets/Business Standard manually for macro context."
        )

    collected.sort(key=lambda x: x[0] or datetime.min.replace(tzinfo=timezone.utc), reverse=True)
    start_str = start_dt.strftime("%Y-%m-%d")
    body = "\n\n".join(fmt for _, fmt in collected[:art_limit])
    return f"## Indian Macro/Market News, {start_str} to {curr_date}:\n\n{body}"
