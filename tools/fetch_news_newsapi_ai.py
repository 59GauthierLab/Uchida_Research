#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Fetch news articles via NewsAPI.ai (Event Registry API) and save to:
  data/news/raw/{ticker}.csv

Free plan note:
- Effectively limited to "last month" (about 30 days).
- 2000 searches/month (as shown on Event Registry plans page).

Install:
  pip install eventregistry pandas python-dateutil

API Key:
  export NEWSAPI_AI_KEY="YOUR_KEY"

Typical usage:
  python scripts/fetch_news_newsapi_ai.py \
    --tickers 7203.T 6758.T \
    --query-map data/news/query_map.json \
    --days 30 \
    --lang jpn eng \
    --max-items 500
  # Use --chunk-days + --resume for archive backfill safely.

query_map.json example:
{
  "7203.T": {"keywords": ["トヨタ", "Toyota", "Toyota Motor"], "use_or": true},
  "6758.T": {"keywords": ["ソニー", "Sony", "Sony Group"], "use_or": true}
}

Optional (more precise, recommended for paid plan):
- Use conceptUri (entity disambiguation) instead of raw keywords:
  "AAPL": {"concept": "Apple Inc."}
The script will resolve concept -> conceptUri via API.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from dotenv import load_dotenv

# --- Hard requirements for downstream feature pipeline ---
REQUIRED_COLS = ["published_at", "title"]  # keep minimal; add more if available


@dataclass
class QuerySpec:
    ticker: str
    keywords: List[str]
    use_or: bool = True
    concept: Optional[str] = None  # e.g. "Apple Inc." to resolve to conceptUri
    source_group: Optional[str] = None  # optional, e.g. "business top100"
    # --- post filters (noise reduction) ---
    exclude_keywords: Optional[List[str]] = None
    # concept解決がズレる/広すぎる時に固定したい場合用（ログのconceptUriをコピペして固定できる）
    concept_uri: Optional[str] = None


def _utc_today() -> datetime:
    return datetime.now(timezone.utc)


def _to_yyyy_mm_dd(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _parse_yyyy_mm_dd(s: str) -> datetime:
    # interpret as UTC date boundary
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)

def _iter_date_windows(date_start: str, date_end: str, chunk_days: int) -> list[tuple[str, str]]:
    """
    Split [date_start, date_end] into multiple windows of length chunk_days.
    Endpoints are inclusive in the user's mental model; we implement as closed ranges.
    """
    if chunk_days <= 0:
        return [(date_start, date_end)]
    start_dt = _parse_yyyy_mm_dd(date_start)
    end_dt = _parse_yyyy_mm_dd(date_end)
    if end_dt < start_dt:
        raise ValueError(f"date_end < date_start: {date_start} .. {date_end}")
    out: list[tuple[str, str]] = []
    cur = start_dt
    while cur <= end_dt:
        nxt = min(cur + timedelta(days=chunk_days - 1), end_dt)
        out.append((_to_yyyy_mm_dd(cur), _to_yyyy_mm_dd(nxt)))
        cur = nxt + timedelta(days=1)
    return out

def _safe_get(d: Any, path: Sequence[str], default=None):
    cur = d
    for k in path:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def _normalize_article(ticker: str, art: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize Event Registry article dict into a flat row for CSV.

    We try multiple candidate keys because API responses can vary by endpoint/flags.
    """
    # publication datetime candidates
    published = (
        art.get("dateTimePub")
        or art.get("dateTime")
        or art.get("publishedAt")
        or art.get("pubDate")
        or art.get("date")
    )

    # Some APIs return dateTime as "YYYY-MM-DD" or "YYYY-MM-DDTHH:MM:SSZ".
    # We keep it as string; downstream can parse to timezone-aware datetime.
    published_at = str(published) if published is not None else ""

    title = art.get("title") or ""
    body = art.get("body") or art.get("content") or ""
    url = art.get("url") or art.get("uri") or ""

    # source can be nested
    source_title = (
        _safe_get(art, ["source", "title"])
        or _safe_get(art, ["source", "name"])
        or art.get("sourceTitle")
        or ""
    )

    # sentiment may be a scalar in [-1, 1] or a nested structure
    sentiment = art.get("sentiment")
    if isinstance(sentiment, dict):
        # keep a best-effort scalar if present
        sentiment = sentiment.get("score") or sentiment.get("value")

    row = {
        "ticker": ticker,
        "published_at": published_at,
        "title": title,
        "body": body,
        "source": source_title,
        "url": url,
        "sentiment_api": sentiment,
    }
    return row


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    _ensure_dir(path.parent)

    # ensure required columns exist
    for r in rows:
        for c in REQUIRED_COLS:
            r.setdefault(c, "")

    # stable columns (put required first)
    columns = [
        "ticker",
        "published_at",
        "title",
        "body",
        "source",
        "url",
        "sentiment_api",
    ]

    # dedupe by (published_at, title, url)
    seen = set()
    deduped = []
    for r in rows:
        key = (r.get("published_at", ""), r.get("title", ""), r.get("url", ""))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)

    # sort by time ascending (best effort)
    def _sort_key(r):
        return (r.get("published_at") or "", r.get("title") or "")

    deduped.sort(key=_sort_key)

    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        w.writeheader()
        for r in deduped:
            w.writerow(r)

def _read_existing_rows(path: Path) -> list[dict[str, Any]]:
    """
    Best-effort read of existing CSV so we can append + dedupe across chunks.
    """
    if not path.exists():
        return []
    try:
        import pandas as pd
        df = pd.read_csv(path, encoding="utf-8")
        return df.to_dict(orient="records")
    except Exception:
        return []

def _load_query_map(path: Optional[Path], tickers: List[str]) -> Dict[str, QuerySpec]:
    """
    If query_map is missing, fallback to a naive default:
    - use ticker string as a keyword (often too noisy; recommended to provide query_map.json)
    """
    specs: Dict[str, QuerySpec] = {}

    if path is None:
        for t in tickers:
            specs[t] = QuerySpec(ticker=t, keywords=[t], use_or=True)
        return specs

    data = json.loads(path.read_text(encoding="utf-8"))
    for t in tickers:
        obj = data.get(t, {})
        keywords = obj.get("keywords") or [t]
        use_or = bool(obj.get("use_or", True))
        concept = obj.get("concept")
        source_group = obj.get("source_group")
        exclude_keywords = obj.get("exclude_keywords") or []
        concept_uri = obj.get("concept_uri")
        specs[t] = QuerySpec(
            ticker=t,
            keywords=list(keywords),
            use_or=use_or,
            concept=concept,
            source_group=source_group,
            exclude_keywords=list(exclude_keywords),
            concept_uri=concept_uri,
        )
    return specs

def _contains_any(text: str, keywords: List[str]) -> bool:
    if not text:
        return False
    t = text.lower()
    for k in keywords or []:
        if k and (k.lower() in t):
            return True
    return False

def _published_day(published_at: str) -> str:
    # '2026-01-09T14:30:07Z' -> '2026-01-09'
    if not published_at:
        return ""
    return published_at.split("T")[0][:10]

def _score_row(row: Dict[str, Any], keywords: List[str]) -> float:
    title = (row.get("title") or "")
    body = (row.get("body") or "")
    score = 0.0
    tl = title.lower()
    bl = body.lower()
    for k in keywords or []:
        if not k:
            continue
        kl = k.lower()
        if kl in tl:
            score += 3.0
        if kl in bl:
            score += 1.0
    # sentiment magnitude (optional)
    try:
        s = float(row.get("sentiment_api"))
        score += min(2.0, abs(s))
    except Exception:
        pass
    return score

def _post_filter_rows(
    rows: List[Dict[str, Any]],
    *,
    keywords: List[str],
    exclude_keywords: List[str],
    require_keyword_hit: bool,
    per_day_cap: int,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        text = f"{r.get('title','')}\n{r.get('body','')}"
        if exclude_keywords and _contains_any(text, exclude_keywords):
            continue
        if require_keyword_hit and keywords and (not _contains_any(text, keywords)):
            continue
        out.append(r)

    if per_day_cap and int(per_day_cap) > 0:
        by_day = defaultdict(list)
        for r in out:
            by_day[_published_day(r.get("published_at",""))].append(r)
        capped: List[Dict[str, Any]] = []
        for day, lst in by_day.items():
            lst.sort(key=lambda x: _score_row(x, keywords), reverse=True)
            capped.extend(lst[: int(per_day_cap)])
        return capped
    return out

def fetch_and_save_news(
    api_key: str,
    out_dir: Path,
    query_specs: Dict[str, QuerySpec],
    date_start: str,
    date_end: str,
    langs: List[str],
    max_items: int,               # total cap per ticker (0 disables)
    max_items_per_window: int,    # per chunk cap (0 disables)
    per_day_cap: int,             # after fetch, keep <=N per day (0 disables)
    require_keyword_hit: bool,    # when conceptUri is used, require keyword presence in title/body
    sort_by: str,
    debug: bool = False,
    chunk_days: int = 0,
    resume: bool = False,
) -> None:
    """
    Fetch articles using Event Registry Python SDK (NewsAPI.ai).
    """
    try:
        from eventregistry import (
            ArticleInfoFlags,
            EventRegistry,
            QueryArticlesIter,
            QueryItems,
            ReturnInfo,
        )
    except ImportError as e:
        raise SystemExit(
            "Missing dependency: eventregistry\n"
            "Install with:\n"
            "  pip install eventregistry pandas python-dateutil\n"
        ) from e

    er = EventRegistry(apiKey=api_key)

    for ticker, spec in query_specs.items():
        out_path = out_dir / f"{ticker}.csv"
        existing = _read_existing_rows(out_path) if resume else []
        if resume and existing:
            print(f"[INFO] {ticker}: resume enabled, loaded existing rows={len(existing)} from {out_path}")

        windows = _iter_date_windows(date_start, date_end, chunk_days)
        all_rows: List[Dict[str, Any]] = list(existing)

        new_total_added = 0
        for (w_start, w_end) in windows:
            print(f"[INFO] {ticker}: window {w_start} .. {w_end} (chunk_days={chunk_days or 'off'})")

            # ---- Build query (PER WINDOW) ----
            query_kwargs: Dict[str, Any] = {}

            # Language filter (can pass list via QueryItems.OR)
            if len(langs) == 1:
                query_kwargs["lang"] = langs[0]
            else:
                query_kwargs["lang"] = QueryItems.OR(langs)

            # Time window
            query_kwargs["dateStart"] = w_start
            query_kwargs["dateEnd"] = w_end

            # Filtering by concept is usually cleaner than keyword search
            if spec.concept:
                concept_uri = None
                if spec.concept_uri:
                    concept_uri = spec.concept_uri
                    if debug:
                        print(f"[DEBUG] {ticker}: using fixed concept_uri='{concept_uri}'")
                else:
                    try:
                        concept_uri = er.getConceptUri(spec.concept)
                    except Exception as e:
                        if debug:
                            print(f"[DEBUG] {ticker}: getConceptUri('{spec.concept}') raised: {e}")

                # concept_uri が「固定」でも「解決」でも、ここで共通的に反映する
                if concept_uri:
                    query_kwargs["conceptUri"] = concept_uri
                    if debug and (not spec.concept_uri):
                        print(f"[DEBUG] {ticker}: concept='{spec.concept}' -> conceptUri='{concept_uri}'")
                else:
                    query_kwargs["keywords"] = (
                        QueryItems.OR(spec.keywords) if spec.use_or else QueryItems.AND(spec.keywords)
                    )
                    print(f"[WARN] {ticker}: concept '{spec.concept}' could not be resolved. Fallback to keywords={spec.keywords}")
            else:
                query_kwargs["keywords"] = (
                    QueryItems.OR(spec.keywords) if spec.use_or else QueryItems.AND(spec.keywords)
                )

            # Optional: source group
            if spec.source_group:
                try:
                    query_kwargs["sourceGroupUri"] = er.getSourceGroupUri(spec.source_group)
                except Exception:
                    pass

            if debug:
                print(f"[DEBUG] {ticker}: query_kwargs={query_kwargs}")

            it = QueryArticlesIter(**query_kwargs)

            ret_info = ReturnInfo(
                articleInfo=ArticleInfoFlags(
                    title=True,
                    body=True,
                    url=True,
                    authors=False,
                    concepts=False,
                    categories=False,
                    socialScore=False,
                    sentiment=True,
                )
            )

            rows: List[Dict[str, Any]] = []
            try:
                exec_kwargs = dict(er=er, sortBy=sort_by, returnInfo=ret_info)
                # cap = min(remaining_total, per_window) (whichever is enabled)
                cap = None
                if max_items_per_window and int(max_items_per_window) > 0:
                    cap = int(max_items_per_window)
                if max_items and int(max_items) > 0:
                    remaining = int(max_items) - (len(all_rows) - len(existing))
                    if remaining <= 0:
                        print(f"[INFO] {ticker}: reached max_items={max_items}. stop.")
                        break
                    cap = remaining if cap is None else min(cap, remaining)
                if cap is not None:
                    exec_kwargs["maxItems"] = cap

                for art in it.execQuery(er, **exec_kwargs):
                    rows.append(_normalize_article(ticker, art))

            except Exception as e:
                print(f"[WARN] fetch failed for {ticker} ({w_start}..{w_end}): {e}")
                # rows に途中まで入っている可能性があるので “捨てない”
            # ---- post filter: noise reduction + per-day cap ----
            rows = _post_filter_rows(
                rows,
                keywords=spec.keywords,
                exclude_keywords=list(spec.exclude_keywords or []),
                require_keyword_hit=bool(require_keyword_hit),
                per_day_cap=int(per_day_cap),
            )

            # ★重要：成功時も失敗時も、取れた分は必ず cumulative に足す
            before = len(all_rows)
            all_rows.extend(rows)
            added = len(all_rows) - before
            new_total_added += added
            print(f"[INFO] {ticker}: fetched {len(rows)} rows in this window; added={added}; cumulative={len(all_rows)}")

            _write_csv(out_path, all_rows)
            print(f"[OK] {ticker}: saved cumulative {len(all_rows)} rows -> {out_path}")

        # ティッカー全体で0件だった場合のガイド
        if (len(all_rows) - len(existing)) == 0:
            print(
                f"[WARN] {ticker}: 0 articles added in total. "
                f"Try --debug, relax --lang, remove concept, or reduce --days (free plan boundary)."
            )


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True, help="Tickers (e.g., 7203.T 6758.T)")
    p.add_argument(
        "--out-dir",
        type=str,
        default="data/news/raw",
        help="Output directory for per-ticker CSVs (default: data/news/raw)",
    )
    p.add_argument(
        "--query-map",
        type=str,
        default=None,
        help="JSON file mapping ticker->keywords/concept (recommended).",
    )
    p.add_argument(
        "--days",
        type=int,
        default=30,
        help="Lookback days for demo (default: 30). Ignored if --date-start/--date-end provided.",
    )
    p.add_argument("--date-start", type=str, default=None, help="YYYY-MM-DD (override)")
    p.add_argument("--date-end", type=str, default=None, help="YYYY-MM-DD (override)")
    p.add_argument(
        "--lang",
        nargs="+",
        default=["jpn"],
        help="Language codes (e.g., jpn eng). Default: jpn",
    )
    p.add_argument("--max-items", type=int, default=500, help="Max articles per ticker TOTAL (0 disables). Default: 500")
    p.add_argument("--max-items-per-window", type=int, default=300, help="Max articles per date window/chunk (0 disables). Default: 300")
    p.add_argument("--per-day-cap", type=int, default=0, help="After fetch, keep at most N articles per calendar day (0 disables).")
    p.add_argument("--require-keyword-hit", action="store_true", help="When using conceptUri, require that title/body contains any keywords.")
    p.add_argument("--sort-by", type=str, default="date", help='Sort field (default: "date")')
    p.add_argument("--debug", action="store_true", help="Print debug logs (conceptUri, query kwargs, etc.)")
    p.add_argument(
        "--chunk-days",
        type=int,
        default=0,
        help="Split the requested date range into chunks (days). 0 disables chunking.",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="If output CSV exists, load it and append + dedupe (useful with --chunk-days).",
    )
    return p


def main() -> None:
    load_dotenv()

    args = build_arg_parser().parse_args()

    api_key = os.environ.get("NEWSAPI_AI_KEY") or os.environ.get("EVENTREGISTRY_API_KEY")
    if not api_key:
        raise SystemExit(
            "API key not found. Set env var:\n"
            "  export NEWSAPI_AI_KEY='YOUR_KEY'\n"
            "or:\n"
            "  export EVENTREGISTRY_API_KEY='YOUR_KEY'\n"
        )

    out_dir = Path(args.out_dir)
    query_map = Path(args.query_map) if args.query_map else None

    # date range
    if args.date_start and args.date_end:
        date_start = args.date_start
        date_end = args.date_end
    else:
        end = _utc_today()
        # free plan boundary can be strict; make it "N days window" rather than "N+1 days inclusive"
        # e.g. days=30 -> go back 29 days so the window spans 30 calendar days (best-effort)
        days = max(1, int(args.days))
        start = end - timedelta(days=days - 1)
        date_start = _to_yyyy_mm_dd(start)
        date_end = _to_yyyy_mm_dd(end)

    specs = _load_query_map(query_map, args.tickers)

    fetch_and_save_news(
        api_key=api_key,
        out_dir=out_dir,
        query_specs=specs,
        date_start=date_start,
        date_end=date_end,
        langs=list(args.lang),
        max_items=int(args.max_items),
        max_items_per_window=int(args.max_items_per_window),
        per_day_cap=int(args.per_day_cap),
        require_keyword_hit=bool(args.require_keyword_hit),
        sort_by=str(args.sort_by),
        debug=bool(args.debug),
        chunk_days=int(args.chunk_days),
        resume=bool(args.resume),
    )


if __name__ == "__main__":
    main()
