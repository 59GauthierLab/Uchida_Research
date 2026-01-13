from __future__ import annotations

"""
tools/fetch_news_gdelt.py

GDELT DOC API (mode=artlist) からニュースを取得し、銘柄ごとに
  data/news/raw/{ticker}.csv
を生成します。

本プロジェクトは「ニュース特徴量の生成」を news_pipeline.py に寄せているため、
ここでは “記事粒度の生データ” を作るだけに徹します。

GDELT完全移行版の生CSVスキーマ（1記事=1行）:
  - ticker: 銘柄
  - published_at: UTC ISO8601 (Z) 例: 2025-12-11T03:20:00Z
  - title: 記事タイトル
  - url: 記事URL（重複排除キーとして必須）
  - source: 配信元（ドメイン等）
  - lang: 言語（取れない場合は空でもOK）
  - tone: GDELTのトーン（取れない場合は空）

実現したい仕様（あなたの要件）:
- “取得可能な日付から” 過去へ遡りつつ
- 毎日 N 件ほど（爆発しない程度）取得する

それを支えるオプション:
- --daily-n N          : 日ごとに最大N件に揃える（重要）
- --daily-overfetch K  : まず N*K 件まで取りに行って、|tone|が大きい順にN件残す
- --auto-backfill      : 今日から過去へ1日ずつ遡って取得（空振りが続いたら停止）
- --stop-after-empty-days M : 空の連続がM日続いたら打ち切り（auto_backfill用）
- --resume             : 既存CSVがあればURL重複を避けて追記（長期バックフィルに必須）
"""

import argparse
import csv
import io
import json
import re
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import requests
from dotenv import load_dotenv

# Downstream (news_pipeline.py) hard requirements
REQUIRED_COLS = ["published_at", "title", "url"]

# DOC API は原則「直近3か月」検索の制約があるため、古い期間はGKGへフォールバックする（auto時）
DOC_LOOKBACK_DAYS_DEFAULT = 95

# GKG 日次zip（YYYYMMDD.gkg.csv.zip）
GKG_DAILY_URL_TEMPLATE = "https://data.gdeltproject.org/gkg/{yyyymmdd}.gkg.csv.zip"
# 一部の環境で HTTPS 証明書検証が失敗する（プロキシ/TLSインスペクション等）ことがあるため
# その場合は HTTP にフォールバックする（GDELT公式の配布リストでもHTTPリンクが使われている）。
GKG_DAILY_URL_TEMPLATE_HTTP = "http://data.gdeltproject.org/gkg/{yyyymmdd}.gkg.csv.zip"

# GKG 日次ファイル (YYYYMMDD.gkg.csv.zip) は 11カラムTSV:
# DATE,NUMARTS,COUNTS,THEMES,LOCATIONS,PERSONS,ORGANIZATIONS,TONE,CAMEOEVENTIDS,SOURCES,SOURCEURLS
# ※あなたのログでも header はこの11カラム（sample_tabs=10）になっている。
GKG_IDX_DATE = 0           # DATE (YYYYMMDDHHMMSS)
GKG_IDX_SOURCE = 9         # SOURCES（複数;区切り）
GKG_IDX_URL = 10           # SOURCEURLS（複数;区切り）
GKG_IDX_ORGS = 6           # ORGANIZATIONS
GKG_IDX_V2ORGS = 6         # 日次では独立列が無いので互換のため同列扱い
GKG_IDX_V2TONE = 7         # TONE (Tone,Pos,Neg,...)
# ---------------------------
# Data structures
# ---------------------------
@dataclass
class QuerySpec:
    ticker: str
    keywords: List[str]
    use_or: bool = True

    # Optional: provide a fully-formed GDELT query string instead of keywords.
    gdelt_query: Optional[str] = None

    # Kept for compatibility with older query_map.json (ignored here).
    concept: Optional[str] = None


@dataclass
class FetchConfig:
    endpoint: str
    date_start: str  # YYYY-MM-DD
    date_end: str    # YYYY-MM-DD
    chunk_days: int

    max_items: int          # max articles per window
    max_records: int        # page size
    sort: str               # e.g. datedesc
    langs: List[str]        # e.g. ["jpn","eng"] -> sourcelang:japanese OR sourcelang:english

    per_day_cap: int        # optional cap per UTC day (0 disables)
    sleep_s: float
    timeout_s: float
    debug: bool


# ---------------------------
# Small utilities
# ---------------------------
def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_today() -> datetime:
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


def _to_yyyy_mm_dd(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%d")

def _to_yyyymmdd(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d")

def _parse_yyyy_mm_dd(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=timezone.utc)


def _iter_date_windows(date_start: str, date_end: str, chunk_days: int) -> List[Tuple[str, str]]:
    if chunk_days <= 0:
        return [(date_start, date_end)]

    s = _parse_yyyy_mm_dd(date_start)
    e = _parse_yyyy_mm_dd(date_end)

    out: List[Tuple[str, str]] = []
    cur = s
    while cur <= e:
        nxt = min(cur + timedelta(days=chunk_days - 1), e)
        out.append((_to_yyyy_mm_dd(cur), _to_yyyy_mm_dd(nxt)))
        cur = nxt + timedelta(days=1)
    return out


def _dt_to_gdelt(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%d%H%M%S")


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None or (isinstance(x, str) and x.strip() == ""):
            return None
        return float(x)
    except Exception:
        return None
    
def _parse_gkg_v2tone(v2tone: Any) -> Optional[float]:
    # V2Tone = "Tone,Positive,Negative,Polarity,ActivityRefDensity,SelfGroupRefDensity,WordCount"
    if v2tone is None:
        return None
    s = str(v2tone).strip()
    if not s:
        return None
    head = s.split(",")[0].strip()
    return _safe_float(head)


def _parse_gdelt_seendate(val: Any) -> Optional[datetime]:
    # Most common: "YYYYMMDDHHMMSS"
    if val is None:
        return None
    if isinstance(val, (int, float)):
        val = str(int(val))
    if not isinstance(val, str):
        return None
    s = val.strip()
    if len(s) == 14 and s.isdigit():
        try:
            return datetime.strptime(s, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        except Exception:
            return None
    # best-effort ISO-ish
    try:
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        return (dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc))
    except Exception:
        return None


# ---------------------------
# Query building
# ---------------------------
_LANG_MAP = {
    "eng": "english",
    "jpn": "japanese",
    "spa": "spanish",
    "fra": "french",
    "deu": "german",
    "ger": "german",
    "zho": "chinese",
    "kor": "korean",
    "rus": "russian",
}


def _normalize_langs(langs: Sequence[str]) -> List[str]:
    out: List[str] = []
    for l in langs:
        ll = (l or "").strip().lower()
        if not ll:
            continue
        out.append(_LANG_MAP.get(ll, ll))
    return sorted(set(out))


def _quote_kw(s: str) -> str:
    s = s.strip()
    if not s:
        return s
    # quote phrases / potentially problematic tokens
    if any(ch.isspace() for ch in s) or any(ch in s for ch in ['"', ":", "(", ")", "[", "]"]):
        s = s.replace('"', '\\"')
        return f'"{s}"'
    return s


def _build_gdelt_query(spec: QuerySpec, langs: Sequence[str], extra_query: Optional[str]) -> str:
    """
    Build QUERY parameter.
    - Advanced operators like sourcelang: / domain: are included *inside* QUERY.
    - OR blocks use (a OR b).
    """
    if spec.gdelt_query:
        base = spec.gdelt_query.strip()
    else:
        kws = [_quote_kw(str(k)) for k in (spec.keywords or []) if str(k).strip()]
        if not kws:
            raise ValueError(f"No keywords for ticker={spec.ticker}. Provide query_map keywords or gdelt_query.")
        base = "(" + " OR ".join(kws) + ")" if spec.use_or else " ".join(kws)

    nl = _normalize_langs(langs)
    if nl:
        base = f"{base} AND (" + " OR ".join([f"sourcelang:{l}" for l in nl]) + ")"

    if extra_query and extra_query.strip():
        base = f"{base} AND ({extra_query.strip()})"

    return base


# ---------------------------
# HTTP + ArtList fetching
# ---------------------------
def _http_get_json(url: str, params: Dict[str, Any], *, timeout_s: float, debug: bool) -> Dict[str, Any]:
    headers = {"User-Agent": "stockpred-gdelt-fetch/1.0", "Accept": "application/json"}

    for attempt in range(5):
        try:
            r = requests.get(url, params=params, headers=headers, timeout=timeout_s)
            if r.status_code == 200:
                return r.json()
            if r.status_code in (429, 500, 502, 503, 504):
                wait = min(10.0, (2 ** attempt) * 0.5)
                if debug:
                    print(f"[WARN] HTTP {r.status_code} -> retry in {wait:.1f}s")
                time.sleep(wait)
                continue
            raise RuntimeError(f"GDELT HTTP {r.status_code}: {r.text[:200]}")
        except Exception as e:
            wait = min(10.0, (2 ** attempt) * 0.5)
            if attempt >= 4:
                raise
            if debug:
                print(f"[WARN] request failed: {e} -> retry in {wait:.1f}s")
            time.sleep(wait)

    raise RuntimeError("Unreachable")


def _extract_artlist_articles(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "articles" in payload and isinstance(payload["articles"], list):
        return payload["articles"]
    # best-effort fallback
    v = payload.get("data")
    if isinstance(v, dict) and "articles" in v and isinstance(v["articles"], list):
        return v["articles"]
    return []


def _fetch_artlist_window(
    *,
    endpoint: str,
    query: str,
    start_dt: datetime,
    end_dt: datetime,
    max_items: int,
    max_records: int,
    sort: str,
    sleep_s: float,
    timeout_s: float,
    debug: bool,
) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    startrecord = 1

    while len(out) < max_items:
        remaining = max_items - len(out)
        batch = min(max_records, remaining)

        params = {
            "query": query,
            "mode": "artlist",
            "format": "json",
            "startdatetime": _dt_to_gdelt(start_dt),
            "enddatetime": _dt_to_gdelt(end_dt),
            "maxrecords": batch,
            "startrecord": startrecord,
            "sort": sort,
        }

        if debug:
            print(f"[DEBUG] GDELT page: startrecord={startrecord} maxrecords={batch}")

        payload = _http_get_json(endpoint, params, timeout_s=timeout_s, debug=debug)
        articles = _extract_artlist_articles(payload)
        if not articles:
            break

        out.extend(articles)
        startrecord += len(articles)

        if len(articles) < batch:
            break
        if sleep_s > 0:
            time.sleep(sleep_s)

    return out[:max_items]


# ---------------------------
# GKG daily (historical) backend
# ---------------------------
def _download_to_cache(
    url: str,
    dst: Path,
    *,
    timeout_s: float,
    debug: bool,
    tls_verify: bool,
) -> bool:
    """
    Download url -> dst (if not exists). Return True if file exists/was downloaded, False if 404.
    """
    _ensure_dir(dst.parent)
    if dst.exists() and dst.stat().st_size > 0:
        return True

    headers = {"User-Agent": "stockpred-gdelt-fetch/1.0"}
    try:
        r = requests.get(url, headers=headers, stream=True, timeout=timeout_s, verify=tls_verify)
        if r.status_code == 404:
            if debug:
                print(f"[DEBUG] GKG not found (404): {url}")
            return False
        r.raise_for_status()
        with dst.open("wb") as f:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
        return True
    except requests.exceptions.SSLError as e:
        # HTTPSで証明書検証に失敗する環境向けのフォールバック
        # 1) まず verify=False で同URLを再試行（ユーザが --no-tls-verify を指定した場合）
        # 2) verify=True で失敗した場合は HTTP へフォールバック
        if debug:
            print(f"[WARN] GKG HTTPS SSL error: {e}")

        # HTTPS -> HTTP フォールバック（verify指定は不要）
        if url.startswith("https://"):
            url2 = "http://" + url[len("https://"):]
            if debug:
                print(f"[WARN] fallback to HTTP for GKG download: {url2}")
            r2 = requests.get(url2, headers=headers, stream=True, timeout=timeout_s)
            if r2.status_code == 404:
                if debug:
                    print(f"[DEBUG] GKG not found (404): {url2}")
                return False
            r2.raise_for_status()
            with dst.open("wb") as f:
                for chunk in r2.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        f.write(chunk)
            return True
        raise
    except Exception as e:
        if debug:
            print(f"[WARN] GKG download failed: {e}")
        raise


def _iter_gkg_tsv_rows(zip_path: Path) -> Sequence[List[str]]:
    """
    Yield TSV-split rows from a GKG daily zip.
    """
    out: List[List[str]] = []
    with zipfile.ZipFile(zip_path, "r") as z:
        inner = None
        for n in z.namelist():
            if n.endswith(".csv"):
                inner = n
                break
        if inner is None:
            return out

        with z.open(inner, "r") as fp:
            for line in io.TextIOWrapper(fp, encoding="utf-8", errors="replace"):
                line = line.rstrip("\n")
                if not line:
                    continue
                out.append(line.split("\t"))
    return out


def _keyword_hit_count(blob: str, keywords: Sequence[str]) -> int:
    """
    Count how many keywords appear (case-insensitive for latin).
    Japanese etc. are left as-is in practice.
    """
    if not blob:
        return 0
    b = blob.lower()
    hit = 0
    for kw in keywords:
        k = str(kw or "").strip()
        if not k:
            continue
        if k.lower() in b:
            hit += 1
    return hit

def _fallback_keywords_from_gdelt_query(gdelt_query: str) -> list[str]:
    """
    GKG backend は「RAWを走査→keywordsヒットで候補化」する実装になりがち。
    query_map ロード不具合等で keywords が空になったときに 0件固定になるのを防ぐため、
    gdelt_query から OR 句のトークン/引用符フレーズを最低限抽出してフォールバックする。
    """
    if not gdelt_query:
        return []
    # 1) "..." のフレーズを優先的に拾う
    phrases = re.findall(r'"([^"]+)"', gdelt_query)
    # 2) OR 句や単語を雑に拾う（フィルタ系トークンは除外）
    tokens = re.findall(r"[^\s()]+", gdelt_query)
    drop_prefixes = ("sourcelang:", "sourcecountry:", "domain:", "theme:", "near", "repeat")
    words = []
    for t in tokens:
        u = t.strip()
        if not u or u.upper() in ("AND", "OR", "NOT"):
            continue
        if u.startswith(drop_prefixes):
            continue
        # 記号だけの断片は捨てる
        if all(ch in "\"'()[]" for ch in u):
            continue
        words.append(u.strip('"'))
    # phrases を先、words を後、重複排除
    seen = set()
    out = []
    for k in phrases + words:
        k = k.strip()
        if not k or k in seen:
            continue
        seen.add(k)
        out.append(k)
    return out

def _parse_gkg_date_fallback(raw: str):
    """
    GKG の DATE は YYYYMMDDHHMMSS（14桁）であることが多い。
    パーサが厳しすぎて None になると全行スキップ→0件になるため、ここで救済する。
    """
    if raw is None:
        return None
    s = str(raw).strip()
    # 数字以外が混ざっても救う（念のため）
    digits = "".join(ch for ch in s if ch.isdigit())
    try:
        if len(digits) == 14:
            return datetime.strptime(digits, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)
        if len(digits) == 8:
            return datetime.strptime(digits, "%Y%m%d").replace(tzinfo=timezone.utc)
    except Exception:
        return None
    return None

def _fetch_gkg_day_rows(
    *,
    spec: QuerySpec,
    day_dt: datetime,
    cache_dir: Path,
    per_day_cap: int,
    overfetch: int,
    timeout_s: float,
    debug: bool,
    existing_urls: Optional[set[str]],
    tls_verify: bool,
) -> List[Dict[str, Any]]:
    """
    GKG日次zipから、keywords一致するレコードを抽出し、上位N件を返す。
    """
    if per_day_cap <= 0:
        raise ValueError("GKG backend requires per_day_cap > 0 (use --daily-n).")
    # keywords が無い場合でも gdelt_query からフォールバック生成して動けるようにする
    if not (spec.keywords or spec.gdelt_query):
        raise ValueError(f"GKG backend requires keywords or gdelt_query for ticker={spec.ticker}.")

    yyyymmdd = _to_yyyymmdd(day_dt)
    url = GKG_DAILY_URL_TEMPLATE.format(yyyymmdd=yyyymmdd)
    zip_path = cache_dir / f"{yyyymmdd}.gkg.csv.zip"

    ok = _download_to_cache(
        url,
        zip_path,
        timeout_s=timeout_s,
        debug=debug,
        tls_verify=tls_verify,
    )
    if not ok:
        return []

    rows = _iter_gkg_tsv_rows(zip_path)
    if debug:
        if not rows:
            print(f"[DEBUG] {spec.ticker}: GKG {yyyymmdd} zip_ok but inner_rows=0 (check zip content)")
        else:
            sample = "\t".join(rows[0])[:200].replace("\t", "\\t")
            tabc = max(0, len(rows[0]) - 1)
            print(f"[DEBUG] {spec.ticker}: GKG {yyyymmdd} sample_tabs={tabc} sample='{sample}'")
    # keywords が空だと GKG 側は永遠に 0件になり得るのでフォールバック
    keywords = spec.keywords or _fallback_keywords_from_gdelt_query(getattr(spec, "gdelt_query", ""))
    if debug:
        print(f"[DEBUG] {spec.ticker}: GKG {yyyymmdd} keywords_n={len(keywords)} "
              f"(orig={len(spec.keywords) if getattr(spec,'keywords',None) is not None else 'NA'})")

    candidates = []
    scanned = 0
    max_idx = max(GKG_IDX_DATE, GKG_IDX_SOURCE, GKG_IDX_URL, GKG_IDX_ORGS, GKG_IDX_V2TONE)
    skipped_short = 0
    skipped_bad_date = 0
    skipped_empty_docid = 0
    kw_hits = 0
    matched = 0

    for cols in rows:
        # ヘッダ行（DATE\tNUMARTS...）はスキップ
        if cols and str(cols[0]).strip().upper() == "DATE":
            continue

        if len(cols) <= max_idx:
            skipped_short += 1
            continue
        scanned += 1

        dt_raw = cols[GKG_IDX_DATE]
        dt_obj = _parse_gdelt_seendate(dt_raw)
        if dt_obj is None:
            dt_obj = _parse_gkg_date_fallback(dt_raw)
        if dt_obj is None:
            skipped_bad_date += 1
            continue

        # SOURCES / SOURCEURLS は「複数;区切り」なので、CSVのurl/sourceは先頭要素を採用
        srcs = str(cols[GKG_IDX_SOURCE] or "").strip()
        src = srcs.split(";")[0].strip() if srcs else ""

        docids = str(cols[GKG_IDX_URL] or "").strip()
        docid = docids.split(";")[0].strip() if docids else ""
        if not docid:
            skipped_empty_docid += 1
            continue

        # 既存URLは早めに除外（resumeの高速化）
        if existing_urls and docid in existing_urls:
            continue

        orgs = str(cols[GKG_IDX_ORGS] or "").strip() if len(cols) > GKG_IDX_ORGS else ""
        v2orgs = str(cols[GKG_IDX_V2ORGS] or "").strip() if len(cols) > GKG_IDX_V2ORGS else ""
        v2tone = cols[GKG_IDX_V2TONE]
        tone = _parse_gkg_v2tone(v2tone)

        # キーワード当たり判定の母集団を少しだけ増やす（最小差分）
        themes = str(cols[3] or "").strip() if len(cols) > 3 else ""
        persons = str(cols[5] or "").strip() if len(cols) > 5 else ""
        blob = " ".join([docids, srcs, themes, persons, orgs, v2orgs])
        hit = _keyword_hit_count(blob, keywords)
        if hit <= 0:
            continue
        kw_hits += 1

        matched += 1
        tone_val = tone if tone is not None else 0.0
        # スコア：keyword一致数を優先しつつ、|tone|が大きいものを上位に
        score = float(hit) * 10.0 + abs(float(tone_val))

        candidates.append(
            (score, {
                "ticker": spec.ticker,
                "published_at": dt_obj.isoformat(),
                "title": "",           # GKG日次にはタイトルが無い（空でOK）
                "url": docid,
                "source": src,
                "lang": "",
                "tone": tone if tone is not None else "",
            })
        )

    if debug:
        print(f"[DEBUG] {spec.ticker}: GKG {yyyymmdd} scanned={scanned} short={skipped_short} "
              f"bad_date={skipped_bad_date} empty_url={skipped_empty_docid} kw_hits={kw_hits} "
              f"candidates={len(candidates)}")

    # overfetchしてからURL重複排除→N件に整形
    candidates.sort(key=lambda x: x[0], reverse=True)
    pre = [r for _, r in candidates[: max(per_day_cap, per_day_cap * max(1, int(overfetch)))]]

    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in pre:
        u = str(r.get("url", "")).strip()
        if not u or u in seen:
            continue
        seen.add(u)
        out.append(r)
        if len(out) >= per_day_cap:
            break
    return out

# ---------------------------
# Row shaping / size control
# ---------------------------
def _rows_from_articles(ticker: str, articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    GDELT ArtList のレスポンス（articles）を本プロジェクトの生CSV行へ変換する。
    - url が空のものは、重複排除や downstream で困るのでスキップする。
    """
    rows: List[Dict[str, Any]] = []
    for a in articles:
        dt = _parse_gdelt_seendate(a.get("seendate") or a.get("seenDate") or a.get("date"))
        if dt is None:
            continue

        url = str(a.get("url") or "").strip()
        if not url:
            continue

        tone = _safe_float(a.get("tone"))
        lang = a.get("language") or a.get("lang") or ""

        rows.append(
            {
                "ticker": ticker,
                "published_at": _iso_z(dt),
                "title": str(a.get("title") or ""),
                "url": url,
                "source": str(a.get("domain") or a.get("source") or ""),
                "lang": str(lang),
                "tone": tone if tone is not None else "",
            }
        )
    return rows


def _dedupe_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    URLを主キーとして重複排除する（url必須のためシンプルにできる）。
    """
    seen: set[str] = set()
    out: List[Dict[str, Any]] = []
    for r in rows:
        key = str(r.get("url", "")).strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(r)
    return out


def _apply_per_day_cap(rows: List[Dict[str, Any]], per_day_cap: int) -> List[Dict[str, Any]]:
    """
    Optional cap per UTC day:
    - group by YYYY-MM-DD (UTC) of published_at
    - keep up to N, preferring larger |tone| when available
    """
    if per_day_cap <= 0:
        return rows

    # stable base sort
    rows = sorted(rows, key=lambda r: (str(r.get("published_at", "")), str(r.get("url", ""))))

    buckets: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        day = str(r.get("published_at", ""))[:10]
        buckets.setdefault(day, []).append(r)

    out: List[Dict[str, Any]] = []
    for day, items in buckets.items():
        def score(rr: Dict[str, Any]) -> float:
            s = _safe_float(rr.get("tone"))
            return abs(s) if s is not None else -1.0

        chosen = sorted(items, key=lambda rr: (score(rr), rr.get("published_at", ""), rr.get("url", "")), reverse=True)[:per_day_cap]
        out.extend(sorted(chosen, key=lambda rr: (str(rr.get("published_at", "")), str(rr.get("url", "")))))

    return sorted(out, key=lambda r: (str(r.get("published_at", "")), str(r.get("url", ""))))


def _read_existing_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return [dict(r) for r in csv.DictReader(f)]


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    cols = ["ticker", "published_at", "title", "url", "source", "lang", "tone"]
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            for c in REQUIRED_COLS:
                if c not in r:
                    raise ValueError(f"row missing required column '{c}'")
            w.writerow({c: r.get(c, "") for c in cols})


# ---------------------------
# query_map.json loader (compatible-ish)
# ---------------------------
def _load_query_map(query_map_path: Optional[Path], tickers: Sequence[str]) -> List[QuerySpec]:
    """
    Supported formats:

    1) Keywords-based (same as NewsAPI.ai script):
      { "7203.T": {"keywords": ["トヨタ","Toyota"], "use_or": true} }

    2) Raw GDELT query:
      { "7203.T": {"gdelt_query": "(Toyota OR トヨタ) domain:reuters.com"} }

    Also allow mapping to a string (treated as gdelt_query):
      { "7203.T": "(Toyota OR トヨタ) domain:reuters.com" }
    """
    if query_map_path is None:
        return [QuerySpec(ticker=t, keywords=[t], use_or=True) for t in tickers]

    data = json.loads(query_map_path.read_text(encoding="utf-8"))
    specs: List[QuerySpec] = []

    for t in tickers:
        entry = data.get(t)
        if entry is None:
            specs.append(QuerySpec(ticker=t, keywords=[t], use_or=True))
            continue

        if isinstance(entry, str):
            specs.append(QuerySpec(ticker=t, keywords=[], use_or=True, gdelt_query=entry))
            continue

        keywords = entry.get("keywords") or entry.get("keyword") or []
        specs.append(
            QuerySpec(
                ticker=t,
                keywords=list(keywords) if isinstance(keywords, list) else [str(keywords)],
                use_or=bool(entry.get("use_or", True)),
                gdelt_query=entry.get("gdelt_query") or entry.get("query"),
                concept=entry.get("concept"),
            )
        )

    return specs


# ---------------------------
# Top-level
# ---------------------------
def fetch_and_save_gdelt(
    *,
    out_dir: Path,
    query_specs: Sequence[QuerySpec],
    config: FetchConfig,
    resume: bool,
    extra_query: Optional[str],
    backend: str,
    cache_dir: Optional[Path],
    daily_overfetch: int,
    tls_verify: bool,
) -> None:
    _ensure_dir(out_dir)
    windows = _iter_date_windows(config.date_start, config.date_end, config.chunk_days)

    for spec in query_specs:
        out_path = out_dir / f"{spec.ticker}.csv"

        existing = _read_existing_csv(out_path) if (resume and out_path.exists()) else []
        existing_urls = {str(r.get("url", "")).strip() for r in existing if str(r.get("url", "")).strip()}

        all_new: List[Dict[str, Any]] = []

        for ws, we in windows:
            print(f"[INFO] {spec.ticker}: window {ws} .. {we} (chunk_days={config.chunk_days})")

            if backend == "doc":
                start_dt = _parse_yyyy_mm_dd(ws).replace(hour=0, minute=0, second=0)
                end_dt = _parse_yyyy_mm_dd(we).replace(hour=23, minute=59, second=59)

                gdelt_query = _build_gdelt_query(spec, config.langs, extra_query)
                if config.debug:
                    print(f"[DEBUG] {spec.ticker}: gdelt_query={gdelt_query}")

                articles = _fetch_artlist_window(
                    endpoint=config.endpoint,
                    query=gdelt_query,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    max_items=config.max_items,
                    max_records=config.max_records,
                    sort=config.sort,
                    sleep_s=config.sleep_s,
                    timeout_s=config.timeout_s,
                    debug=config.debug,
                )

                rows = _rows_from_articles(spec.ticker, articles)

                # resume: drop URLs already on disk
                if existing_urls:
                    rows = [r for r in rows if str(r.get("url", "")).strip() not in existing_urls]

                rows = _apply_per_day_cap(rows, config.per_day_cap)
                all_new.extend(rows)

            elif backend == "gkg":
                if cache_dir is None:
                    raise ValueError("backend=gkg requires cache_dir")
                if config.per_day_cap <= 0:
                    raise ValueError("backend=gkg requires --daily-n (per_day_cap>0)")

                day_dt = _parse_yyyy_mm_dd(ws).replace(hour=0, minute=0, second=0)
                end_day = _parse_yyyy_mm_dd(we).replace(hour=0, minute=0, second=0)

                while day_dt <= end_day:
                    day_rows = _fetch_gkg_day_rows(
                        spec=spec,
                        day_dt=day_dt,
                        cache_dir=cache_dir,
                        per_day_cap=config.per_day_cap,
                        overfetch=daily_overfetch,
                        timeout_s=config.timeout_s,
                        debug=config.debug,
                        existing_urls=existing_urls if existing_urls else None,
                        tls_verify=tls_verify,
                    )
                    if day_rows:
                        if config.debug:
                            print(f"[DEBUG] {spec.ticker}: {day_dt.strftime('%Y-%m-%d')} kept={len(day_rows)}")
                        all_new.extend(day_rows)
                    day_dt += timedelta(days=1)
            else:
                raise ValueError(f"Unknown backend: {backend}")

        merged = _dedupe_rows(existing + all_new)
        merged = sorted(merged, key=lambda r: (str(r.get("published_at", "")), str(r.get("url", ""))))

        _write_csv(out_path, merged)
        print(f"[OK] wrote {len(merged)} rows -> {out_path}")

def fetch_and_save_gdelt_auto_backfill(
    *,
    out_dir: Path,
    query_specs: Sequence[QuerySpec],
    config: FetchConfig,
    resume: bool,
    extra_query: Optional[str],
    backfill_max_days: int,
    stop_after_empty_days: int,
    backend: str,
    cache_dir: Optional[Path],
    daily_overfetch: int,
    tls_verify: bool,
) -> None:
    """
    “取得可能な日付から”を近似するためのバックフィル。

    今日（または date_end）から1日ずつ遡り、記事が取れない日が
    stop_after_empty_days 連続したら打ち切る。

    注意:
    - GDELT DOC API は古い期間が常に取得できる保証が弱いので、これは“現実的な近似”。
    """
    _ensure_dir(out_dir)
    end_day_dt = _parse_yyyy_mm_dd(config.date_end)

    for spec in query_specs:
        out_path = out_dir / f"{spec.ticker}.csv"

        existing = _read_existing_csv(out_path) if (resume and out_path.exists()) else []
        existing_urls = {str(r.get("url", "")).strip() for r in existing if str(r.get("url", "")).strip()}

        all_new: List[Dict[str, Any]] = []
        empty_streak = 0

        for offset in range(max(1, int(backfill_max_days))):
            day_dt = end_day_dt - timedelta(days=offset)
            day_s = _to_yyyy_mm_dd(day_dt)

            start_dt = day_dt.replace(hour=0, minute=0, second=0)
            end_dt = day_dt.replace(hour=23, minute=59, second=59)

            if config.debug:
                print(f"[DEBUG] {spec.ticker}: backfill day {day_s} backend={backend}")

            rows: List[Dict[str, Any]] = []
            if backend == "doc":
                gdelt_query = _build_gdelt_query(spec, config.langs, extra_query)
                if config.debug:
                    print(f"[DEBUG] {spec.ticker}: gdelt_query={gdelt_query}")

                articles = _fetch_artlist_window(
                    endpoint=config.endpoint,
                    query=gdelt_query,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    max_items=config.max_items,
                    max_records=config.max_records,
                    sort=config.sort,
                    sleep_s=config.sleep_s,
                    timeout_s=config.timeout_s,
                    debug=config.debug,
                )
                rows = _rows_from_articles(spec.ticker, articles)

                if existing_urls:
                    rows = [r for r in rows if str(r.get("url", "")).strip() not in existing_urls]

                rows = _apply_per_day_cap(rows, config.per_day_cap)

            elif backend == "gkg":
                if cache_dir is None:
                    raise ValueError("backend=gkg requires cache_dir")
                if config.per_day_cap <= 0:
                    raise ValueError("backend=gkg requires --daily-n (per_day_cap>0)")

                rows = _fetch_gkg_day_rows(
                    spec=spec,
                    day_dt=day_dt.replace(hour=0, minute=0, second=0),
                    cache_dir=cache_dir,
                    per_day_cap=config.per_day_cap,
                    overfetch=daily_overfetch,
                    timeout_s=config.timeout_s,
                    debug=config.debug,
                    existing_urls=existing_urls if existing_urls else None,
                    tls_verify=tls_verify,
                )
            else:
                raise ValueError(f"Unknown backend: {backend}")

            if rows:
                empty_streak = 0
                all_new.extend(rows)
                print(f"[INFO] {spec.ticker}: {day_s} rows={len(rows)} (keep<= {config.per_day_cap or '∞'})")
            else:
                empty_streak += 1
                if config.debug:
                    print(f"[DEBUG] {spec.ticker}: {day_s} rows=0 empty_streak={empty_streak}")

            if empty_streak >= int(stop_after_empty_days):
                print(f"[INFO] {spec.ticker}: stop (empty_streak={empty_streak}) at {day_s} (approx earliest available)")
                break

        merged = _dedupe_rows(existing + all_new)
        merged = sorted(merged, key=lambda r: (str(r.get("published_at", "")), str(r.get("url", ""))))

        _write_csv(out_path, merged)
        print(f"[OK] wrote {len(merged)} rows -> {out_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True, help="Tickers (e.g., 7203.T 6758.T)")
    p.add_argument("--out-dir", type=str, default="data/news/raw", help="Output dir (default: data/news/raw)")
    p.add_argument("--query-map", type=str, default=None, help="JSON mapping ticker->keywords or gdelt_query")

    # date range (same UX as NewsAPI.ai script)
    p.add_argument("--days", type=int, default=30, help="Lookback days (ignored if --date-start/--date-end)")
    p.add_argument("--date-start", type=str, default=None, help="YYYY-MM-DD (override)")
    p.add_argument("--date-end", type=str, default=None, help="YYYY-MM-DD (override)")

    # language filter (inside query via sourcelang:)
    p.add_argument("--lang", nargs="+", default=["jpn"], help="Language codes (e.g., jpn eng). Default: jpn")

    # size control
    p.add_argument("--max-items", type=int, default=500, help="Max articles per window/day (default: 500)")
    p.add_argument("--per-day-cap", type=int, default=0, help="Keep at most N articles per UTC day (0 disables)")

    # Daily-mode convenience: fetch ~N items per day
    p.add_argument("--daily-n", type=int, default=0, help="If >0, enable daily mode and keep ~N articles/day")
    p.add_argument("--daily-overfetch", type=int, default=5, help="Daily mode: fetch N*overfetch then select top-N by |tone| (default: 5)")

    # auto-backfill: approximate “earliest available date”
    p.add_argument("--auto-backfill", action="store_true", help="Scan backwards day-by-day until empty streak, to approximate earliest available date")
    p.add_argument("--backfill-max-days", type=int, default=3650, help="Max days to scan backwards in --auto-backfill (default: 3650)")
    p.add_argument("--stop-after-empty-days", type=int, default=120, help="Stop scanning after this many consecutive empty days (default: 120)")

    # GDELT knobs
    p.add_argument("--endpoint", type=str, default="https://api.gdeltproject.org/api/v2/doc/doc", help="DOC API endpoint")
    p.add_argument("--backend", type=str, choices=["auto", "doc", "gkg"], default="auto",
                   help="Fetch backend. auto=recent->DOC, old->GKG. doc=DOC API only. gkg=GKG daily files.")
    p.add_argument("--doc-lookback-days", type=int, default=DOC_LOOKBACK_DAYS_DEFAULT,
                   help="For backend=auto/doc: if date_end is older than this many days, DOC is considered unavailable. Default: 95")
    p.add_argument("--cache-dir", type=str, default="data/news/cache/gdelt",
                   help="Cache dir for GKG daily zip files (backend=gkg). Default: data/news/cache/gdelt")
    p.add_argument("--no-tls-verify", action="store_true",
                   help="Disable TLS certificate verification (mainly for environments with HTTPS interception). "
                        "If HTTPS fails, the code also falls back to HTTP for GKG downloads.")
    p.add_argument("--max-records", type=int, default=250, help="Page size per request (default: 250)")
    p.add_argument("--sort", type=str, default="datedesc", help='Sort (default: "datedesc")')
    p.add_argument("--query-extra", type=str, default=None, help='Extra query AND-joined, e.g. "domain:reuters.com"')
    p.add_argument("--sleep", type=float, default=0.2, help="Sleep between pages (default: 0.2)")
    p.add_argument("--timeout", type=float, default=30.0, help="HTTP timeout seconds (default: 30)")

    # chunking + resume (same flags)
    p.add_argument("--chunk-days", type=int, default=0, help="Split range into chunks (days). 0 disables")
    p.add_argument("--resume", action="store_true", help="Append + dedupe if output exists (use with --chunk-days)")

    p.add_argument("--debug", action="store_true", help="Verbose logs")
    return p


def main() -> None:
    load_dotenv()
    args = build_arg_parser().parse_args()

    out_dir = Path(args.out_dir)
    query_map = Path(args.query_map) if args.query_map else None

    if args.date_start and args.date_end:
        date_start, date_end = args.date_start, args.date_end
    else:
        end = _utc_today()
        days = max(1, int(args.days))
        start = end - timedelta(days=days - 1)
        date_start, date_end = _to_yyyy_mm_dd(start), _to_yyyy_mm_dd(end)

    specs = _load_query_map(query_map, args.tickers)

    config = FetchConfig(
        endpoint=str(args.endpoint),
        date_start=date_start,
        date_end=date_end,
        chunk_days=int(args.chunk_days),
        max_items=int(args.max_items),
        max_records=max(1, int(args.max_records)),
        sort=str(args.sort),
        langs=list(args.lang),
        per_day_cap=int(args.per_day_cap),
        sleep_s=float(args.sleep),
        timeout_s=float(args.timeout),
        debug=bool(args.debug),
    )

    # Daily mode: enforce “毎日N件ほど”
    if int(getattr(args, "daily_n", 0) or 0) > 0:
        n = int(args.daily_n)
        over = max(1, int(getattr(args, "daily_overfetch", 5)))
        # 1日ずつ取得（爆発日でも “その日” で完結する）
        config.chunk_days = 1
        # 最終的に N 件まで残す
        config.per_day_cap = n
        # “重要っぽい”候補を拾うために多めに取る
        config.max_items = n * over
        if config.debug:
            print(f"[DEBUG] daily_mode enabled: n={n} overfetch={over} -> chunk_days=1 per_day_cap={config.per_day_cap} max_items={config.max_items}")
    
    # backend selection
    backend = str(getattr(args, "backend", "auto")).lower()
    lookback_days = int(getattr(args, "doc_lookback_days", DOC_LOOKBACK_DAYS_DEFAULT))
    end_dt = _parse_yyyy_mm_dd(date_end)
    if backend == "auto":
        backend = "gkg" if (_utc_now() - end_dt) > timedelta(days=lookback_days) else "doc"

    if backend == "doc" and (_utc_now() - end_dt) > timedelta(days=lookback_days):
        raise SystemExit(
            f"[ERROR] DOC backend cannot reliably fetch old ranges (date_end={date_end}). "
            f"Use --backend gkg (GKG daily files) for historical periods."
        )

    cache_dir = Path(getattr(args, "cache_dir", "data/news/cache/gdelt"))
    if config.debug:
        print(f"[DEBUG] backend={backend} doc_lookback_days={lookback_days} cache_dir={cache_dir}")

    if backend == "gkg" and config.per_day_cap <= 0:
        raise SystemExit("[ERROR] backend=gkg requires --daily-n (to cap per-day output).")

    # TLS verify for GKG downloads
    tls_verify = not bool(getattr(args, "no_tls_verify", False))

    # Execute
    if bool(getattr(args, "auto_backfill", False)):
        fetch_and_save_gdelt_auto_backfill(
            out_dir=out_dir,
            query_specs=specs,
            config=config,
            resume=bool(args.resume),
            extra_query=args.query_extra,
            backfill_max_days=int(args.backfill_max_days),
            stop_after_empty_days=int(args.stop_after_empty_days),
            backend=backend,
            cache_dir=cache_dir if backend == "gkg" else None,
            daily_overfetch=int(getattr(args, "daily_overfetch", 5)),
            tls_verify=tls_verify,
        )
    else:
        fetch_and_save_gdelt(
            out_dir=out_dir,
            query_specs=specs,
            config=config,
            resume=bool(args.resume),
            extra_query=args.query_extra,
            backend=backend,
            cache_dir=cache_dir if backend == "gkg" else None,
            daily_overfetch=int(getattr(args, "daily_overfetch", 5)),
            tls_verify=tls_verify,
        )


if __name__ == "__main__":
    main()