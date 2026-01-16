# src/stock_pred/news_pipeline.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Sequence, Optional, Tuple, Dict, List

import numpy as np
import pandas as pd

# ---------------------------
# GKG-only pipeline note
# - 全期間を GDELT GKG で取得する前提
# - title/body を使った辞書センチメント特徴量は作らない（GKGの数値シグナルのみ使用）
# ---------------------------

# 既存の feature_cols に足すための “代表セット”（A/Bの切替で使う）
DEFAULT_NEWS_META_COLS = [
    "news_count_0d",
    "news_source_nunique_0d",
    "news_after_close_ratio_0d",
    "news_count_3bd_sum",
    "news_count_5bd_sum",
    "news_count_surprise_20bd",

    # GKG: 強度（記事数）
    "news_gkg_numarts_sum_0d",
    "news_gkg_numarts_mean_0d",
    "news_gkg_numarts_3bd_sum",
    "news_gkg_numarts_5bd_sum",
    "news_gkg_numarts_surprise_20bd",

    # GKG: 文字列系（最小構成: 件数/ユニーク数/平均長）
    "news_gkg_themes_items_sum_0d",
    "news_gkg_themes_items_nunique_0d",
    "news_gkg_themes_strlen_mean_0d",
    "news_gkg_organizations_items_sum_0d",
    "news_gkg_organizations_items_nunique_0d",
    "news_gkg_organizations_strlen_mean_0d",
    "news_no_news_flag",
]
DEFAULT_NEWS_SENT_COLS = []  # legacy: NewsAPI.ai 時代の辞書センチメント（title等）由来。GKG-onlyでは未使用。

# GDELT の tone（実数）を日次集計する列
DEFAULT_NEWS_TONE_COLS = [
    "news_tone_mean_0d",
    "news_tone_sum_0d",
    "news_tone_min_0d",
    "news_tone_max_0d",
    "news_tone_valid_count_0d",
    "news_tone_valid_ratio_0d",
    "news_tone_pos_ratio_0d",
    "news_tone_neg_ratio_0d",
    "news_tone_abs_mean_0d",
    "news_tone_3bd_mean",
    "news_tone_5bd_mean",

    # GKG V2Tone components（CSVにあれば）
    "news_v2tone_pos_mean_0d",
    "news_v2tone_pos_sum_0d",
    "news_v2tone_pos_3bd_mean",
    "news_v2tone_pos_5bd_mean",
    "news_v2tone_neg_mean_0d",
    "news_v2tone_neg_sum_0d",
    "news_v2tone_neg_3bd_mean",
    "news_v2tone_neg_5bd_mean",
    "news_v2tone_pol_mean_0d",
    "news_v2tone_pol_sum_0d",
    "news_v2tone_pol_3bd_mean",
    "news_v2tone_pol_5bd_mean",
    "news_v2tone_wc_mean_0d",
    "news_v2tone_wc_sum_0d",
    "news_v2tone_wc_3bd_mean",
    "news_v2tone_wc_5bd_mean",
]

# Backward-compat alias（旧: NewsAPI.ai の sentiment_api を想定した名前）
DEFAULT_NEWS_API_SENT_COLS = DEFAULT_NEWS_TONE_COLS

REQUIRED_COLS = ["published_at", "url"]  # GKG-only: title は特徴量に使わないため必須にしない

# featureset / cache key version（特徴量定義を変えたら更新する）
NEWS_FEAT_CACHE_VER = "v4"

def _parse_close_minutes(market_close_time: str) -> int:
    hh, mm = market_close_time.strip().split(":")
    return int(hh) * 60 + int(mm)


def _ensure_datetime_tz(s: pd.Series, tz: str) -> pd.Series:
    """
    published_at を datetime 化して tz をそろえる。
    - tz情報が無い場合: tz を localize
    - tz情報がある場合: tz へ convert
    """
    dt = pd.to_datetime(s, errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(dt):
        return dt.dt.tz_convert(tz)
    # tzなし（naive）なら指定tzとして解釈
    return dt.dt.tz_localize(tz)


def _normalize_trading_days(price_index: pd.Index) -> pd.DatetimeIndex:
    td = pd.to_datetime(price_index).tz_localize(None).normalize()
    td = pd.DatetimeIndex(td).unique().sort_values()
    return td


def load_news(path: str) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"[news] not found: {p}")

    if p.suffix.lower() in [".csv"]:
        df = pd.read_csv(p)
    elif p.suffix.lower() in [".jsonl", ".jl"]:
        df = pd.read_json(p, lines=True)
    elif p.suffix.lower() in [".json"]:
        df = pd.read_json(p)
    elif p.suffix.lower() in [".parquet"]:
        df = pd.read_parquet(p)
    else:
        raise ValueError(f"[news] unsupported format: {p.suffix}")

    # 最低限の列を用意
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"[news] missing required column: {c}")

    # 取得元（ドメインなど）
    if "source" not in df.columns:
        df["source"] = ""

    # 言語（空でもOK）
    if "lang" not in df.columns:
        df["lang"] = ""

    # ticker列が無い場合も動くように（ファイル名側で補う想定）
    if "ticker" not in df.columns:
        df["ticker"] = ""

    # GDELTの tone（実数）。旧列 sentiment_api があれば tone に移し替えて互換対応。
    if "tone" not in df.columns:
        if "sentiment_api" in df.columns:
            df["tone"] = pd.to_numeric(df.get("sentiment_api"), errors="coerce")
        else:
            df["tone"] = np.nan
    else:
        df["tone"] = pd.to_numeric(df.get("tone"), errors="coerce")
    # GKG由来の追加数値列（存在すれば numeric 化しておく）
    for c in ("tone_pos", "tone_neg", "tone_pol", "tone_wc", "gkg_numarts"):
        if c in df.columns:
            df[c] = pd.to_numeric(df.get(c), errors="coerce")

    return df


def align_news_to_trading_day(
    news_df: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    *,
    tz: str = "Asia/Tokyo",
    market_close_time: str = "15:30",
    published_col: str = "published_at",
) -> pd.DataFrame:
    """
    ニュースを「どの取引日に効かせるか」に変換する（リーク対策の核心）。
    - 引け後記事は翌取引日へ
    - 非取引日は次の取引日へ丸め
    """
    out = news_df.copy()

    pub = _ensure_datetime_tz(out[published_col], tz=tz)
    close_min = _parse_close_minutes(market_close_time)

    minutes = pub.dt.hour * 60 + pub.dt.minute

    # --- Leakage guard for GKG daily ---
    # GKG日次は published_at が「00:00:00 UTC」固定になりがちで、これは実際の発信時刻ではない。
    # そのままJSTに変換すると 09:00 扱いになり “引け前” 判定→当日特徴量として使われうるため、
    # 「UTCでほぼ全件が00:00:00」の場合は強制的に翌取引日に送る（= after close 扱い）.
    has_gkg_cols = any(str(c).startswith("gkg_") for c in out.columns) or ("gkg_numarts" in out.columns)
    pub_utc = pub.dt.tz_convert("UTC")
    utc_midnight = (pub_utc.dt.hour == 0) & (pub_utc.dt.minute == 0) & (pub_utc.dt.second == 0)
    utc_midnight_ratio = float(utc_midnight.mean()) if len(utc_midnight) else 0.0

    if has_gkg_cols and utc_midnight_ratio > 0.95:
        is_after_close = pd.Series(1, index=out.index, dtype=int)
    else:
        is_after_close = (minutes >= close_min).astype(int)

    base_day = pub.dt.normalize().dt.tz_localize(None)  # その日の00:00（naive）
    target_day = base_day + pd.to_timedelta(is_after_close, unit="D")

    # target_day を trading_days に丸め（target_day以上で最初に現れる取引日）
    td_np = trading_days.values.astype("datetime64[ns]")
    tgt_np = target_day.values.astype("datetime64[ns]")
    idx = np.searchsorted(td_np, tgt_np, side="left")

    valid = idx < len(td_np)
    out = out.loc[valid].copy()
    out["trade_date"] = pd.to_datetime(td_np[idx[valid]]).normalize()
    out["is_after_close"] = is_after_close.loc[valid].astype(int).values
    # pandasでは tz-aware -> tz-naive を .astype("datetime64[ns]") で変換できず TypeError になる。
    # 取引日(trading_days)は tz-naive で扱っているため、保存用は明示的に tz を落とす。
    # （_ensure_datetime_tzで市場タイムゾーンへ揃えた後なので、時刻はAsia/Tokyo基準の“壁時計時刻”として残る）
    out[published_col] = pub.loc[valid].dt.tz_localize(None)

    return out


def featurize_news_daily(
    aligned_news: pd.DataFrame,
    trading_days: pd.DatetimeIndex,
    *,
    use_meta: bool = True,
    use_sentiment: bool = True,
    tz: str = "Asia/Tokyo",
    market_close_time: str = "15:30",
    pos_words: Optional[Sequence[str]] = None,
    neg_words: Optional[Sequence[str]] = None,
    rolling_windows: Sequence[int] = (3, 5),
    long_window: int = 20,
    text_cols: Sequence[str] = ("title",),
) -> pd.DataFrame:
    """
    trade_date単位の日次特徴量を作る（GKG-only）。

    NOTE:
      - title/body を使った辞書センチメントは生成しません（pos_words/neg_words/text_cols は互換のため残っていますが未使用）。
    返り値は trading_days と同じ index を持つ DataFrame（0埋め済み）。
    """
    if aligned_news.empty:
        # 完全ゼロの特徴量（列は固定で返す）
        z = pd.DataFrame(index=trading_days)
        for c in (DEFAULT_NEWS_META_COLS if use_meta else []):
            z[c] = 0.0
        for c in (DEFAULT_NEWS_SENT_COLS if use_sentiment else []):
            z[c] = 0.0
        # GDELT tone 列もゼロで返す
        for c in (DEFAULT_NEWS_TONE_COLS if use_sentiment else []):
            z[c] = 0.0
        if use_meta and "news_no_news_flag" in z.columns:
            z["news_no_news_flag"] = 1.0
        return z

    daily_parts = []

    # --- A) meta ---
    if use_meta:
        g = aligned_news.groupby("trade_date")
        meta = pd.DataFrame(index=g.size().index)
        meta["news_count_0d"] = g.size().astype(float)
        meta["news_source_nunique_0d"] = g["source"].nunique(dropna=True).astype(float)
        after_close = g["is_after_close"].sum().astype(float)
        meta["news_after_close_ratio_0d"] = (after_close / (meta["news_count_0d"] + 1e-9)).fillna(0.0)

        # rolling sums（当日含む、取引日単位）
        for w in rolling_windows:
            meta[f"news_count_{w}bd_sum"] = meta["news_count_0d"].rolling(w, min_periods=1).sum()

        # “驚き” = 当日 - 過去平均（前日まで）
        prev_mean = meta["news_count_0d"].rolling(long_window, min_periods=1).mean().shift(1)
        meta["news_count_surprise_20bd"] = (meta["news_count_0d"] - prev_mean).fillna(0.0)

        # --- GKG: numarts（記事量の強度） ---
        if "gkg_numarts" in aligned_news.columns:
            tmp_gkg = aligned_news[["trade_date", "gkg_numarts"]].copy()
            tmp_gkg["gkg_numarts"] = pd.to_numeric(tmp_gkg["gkg_numarts"], errors="coerce")
            g3 = tmp_gkg.groupby("trade_date")
            meta["news_gkg_numarts_sum_0d"] = g3["gkg_numarts"].sum(min_count=1).fillna(0.0).astype(float)
            meta["news_gkg_numarts_mean_0d"] = g3["gkg_numarts"].mean().fillna(0.0).astype(float)
        else:
            meta["news_gkg_numarts_sum_0d"] = 0.0
            meta["news_gkg_numarts_mean_0d"] = 0.0

        for w in rolling_windows:
            meta[f"news_gkg_numarts_{w}bd_sum"] = meta["news_gkg_numarts_sum_0d"].rolling(w, min_periods=1).sum()

        prev_mean_g = meta["news_gkg_numarts_sum_0d"].rolling(long_window, min_periods=1).mean().shift(1)
        meta["news_gkg_numarts_surprise_20bd"] = (meta["news_gkg_numarts_sum_0d"] - prev_mean_g).fillna(0.0)

        # --- GKG: 文字列系（最小の統計量） ---
        def _add_gkg_text_stats(col: str, prefix: str) -> None:
            if col not in aligned_news.columns:
                meta[f"{prefix}_items_sum_0d"] = 0.0
                meta[f"{prefix}_items_nunique_0d"] = 0.0
                meta[f"{prefix}_strlen_mean_0d"] = 0.0
                return

            s = aligned_news[col].fillna("").astype(str)
            # 区切り: ';' と ',' を想定（GKGのフィールドは両方が混ざり得る）
            item_cnt = np.where(s.str.len() > 0, s.str.count(r"[;,]") + 1, 0).astype(float)

            tmp_txt = pd.DataFrame({"trade_date": aligned_news["trade_date"].values})
            tmp_txt["item_cnt"] = item_cnt
            tmp_txt["strlen"] = s.str.len().astype(float)

            gtxt = tmp_txt.groupby("trade_date")
            meta[f"{prefix}_items_sum_0d"] = gtxt["item_cnt"].sum().astype(float)
            meta[f"{prefix}_strlen_mean_0d"] = gtxt["strlen"].mean().fillna(0.0).astype(float)

            # ユニーク数（トークン単位）
            tmp_items = aligned_news[["trade_date", col]].copy()
            tmp_items[col] = s
            tmp_items = tmp_items.assign(_item=tmp_items[col].str.split(r"[;,]")).explode("_item")
            tmp_items["_item"] = tmp_items["_item"].fillna("").astype(str).str.strip()
            tmp_items = tmp_items[tmp_items["_item"] != ""]
            uniq = tmp_items.groupby("trade_date")["_item"].nunique(dropna=True).astype(float)
            meta[f"{prefix}_items_nunique_0d"] = uniq.reindex(meta.index).fillna(0.0)

        _add_gkg_text_stats("gkg_themes", "news_gkg_themes")
        _add_gkg_text_stats("gkg_organizations", "news_gkg_organizations")
        
        daily_parts.append(meta)

    # --- B) tone / V2Tone（数値シグナルのみ。title/bodyベースの辞書センチメントは作らない） ---
    if use_sentiment:
        tmp = aligned_news.copy()

        # --- GDELT "tone" を日次集計 ---
        # GDELT DOC API の ArtList は記事ごとに tone（実数）を返すことがある。
        # （取得できない記事もあるので、NaNは無視して集計する）
        tmp["tone"] = pd.to_numeric(tmp.get("tone"), errors="coerce")
        tmp["tone_abs"] = tmp["tone"].abs()
        tmp["tone_pos"] = (tmp["tone"] > 0).astype(float)
        tmp["tone_neg"] = (tmp["tone"] < 0).astype(float)

        g2 = tmp.groupby("trade_date")
        tone = pd.DataFrame(index=g2.size().index)

        valid_cnt = g2["tone"].count().astype(float)  # NaNを除いた件数
        tone["news_tone_valid_count_0d"] = valid_cnt
        tone["news_tone_valid_ratio_0d"] = (valid_cnt / (g2.size().astype(float) + 1e-9)).fillna(0.0)

        tone["news_tone_mean_0d"] = g2["tone"].mean().fillna(0.0).astype(float)
        tone["news_tone_sum_0d"] = g2["tone"].sum().fillna(0.0).astype(float)
        tone["news_tone_min_0d"] = g2["tone"].min().fillna(0.0).astype(float)
        tone["news_tone_max_0d"] = g2["tone"].max().fillna(0.0).astype(float)
        tone["news_tone_abs_mean_0d"] = g2["tone_abs"].mean().fillna(0.0).astype(float)

        tone_pos = g2["tone_pos"].sum().astype(float)
        tone_neg = g2["tone_neg"].sum().astype(float)
        tone["news_tone_pos_ratio_0d"] = (tone_pos / (valid_cnt + 1e-9)).fillna(0.0)
        tone["news_tone_neg_ratio_0d"] = (tone_neg / (valid_cnt + 1e-9)).fillna(0.0)

        # rolling mean（有効件数で重み付け：sum / valid_count）
        tone = tone.join(valid_cnt.rename("tone_cnt"))
        for w in rolling_windows:
            roll_sum = tone["news_tone_sum_0d"].rolling(w, min_periods=1).sum()
            roll_cnt = tone["tone_cnt"].rolling(w, min_periods=1).sum()
            tone[f"news_tone_{w}bd_mean"] = (roll_sum / (roll_cnt + 1e-9)).fillna(0.0)
        tone = tone.drop(columns=["tone_cnt"])

        # --- GKG V2Tone components（pos/neg/pol/wc） ---
        v2_specs = [
            ("tone_pos", "pos"),
            ("tone_neg", "neg"),
            ("tone_pol", "pol"),
            ("tone_wc", "wc"),
        ]
        v2 = pd.DataFrame(index=g2.size().index)

        for raw_col, tag in v2_specs:
            if raw_col in tmp.columns:
                tmp[f"v2_{tag}"] = pd.to_numeric(tmp[raw_col], errors="coerce")
            else:
                tmp[f"v2_{tag}"] = np.nan

            g3 = tmp.groupby("trade_date")
            v_sum = g3[f"v2_{tag}"].sum(min_count=1).fillna(0.0).astype(float)
            v_mean = g3[f"v2_{tag}"].mean().fillna(0.0).astype(float)
            v_cnt = g3[f"v2_{tag}"].count().astype(float)

            v2[f"news_v2tone_{tag}_sum_0d"] = v_sum
            v2[f"news_v2tone_{tag}_mean_0d"] = v_mean

            for w in rolling_windows:
                roll_sum = v_sum.rolling(w, min_periods=1).sum()
                roll_cnt = v_cnt.rolling(w, min_periods=1).sum()
                v2[f"news_v2tone_{tag}_{w}bd_mean"] = (roll_sum / (roll_cnt + 1e-9)).fillna(0.0)

        tone = tone.join(v2, how="left")

        daily_parts.append(tone)

    # --- merge parts on trade_date ---
    daily = daily_parts[0]
    for part in daily_parts[1:]:
        daily = daily.join(part, how="outer")

    # reindex to trading_days & fill
    daily = daily.reindex(trading_days).fillna(0.0)

    # no_news_flag（Aが無い場合でも作りたいならここで作れるが、MVPはA前提）
    if use_meta:
        daily["news_no_news_flag"] = (daily["news_count_0d"] <= 0.0).astype(float)

    return daily


def add_news_features(
    price_feature_df: pd.DataFrame,
    *,
    ticker: str,
    news_path_template: str,
    cache_dir: str = "data/news/features",
    tz: str = "Asia/Tokyo",
    market_close_time: str = "15:30",
    use_meta: bool = True,
    use_sentiment: bool = True,
    rolling_windows: Sequence[int] = (3, 5),
    long_window: int = 20,
    text_cols: Sequence[str] = ("title",),
    allow_cache: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    price_feature_df（index=取引日）へニュース特徴量を列追加して返す（0埋め）。
    - ニュース raw を読み込む
    - 取引日に整列
    - 日次集約（A/B）
    - join
    - parquetキャッシュ（任意）
    """
    out = price_feature_df.copy()
    orig_shape = out.shape
    orig_cols = list(out.columns)

    def _p(msg: str) -> None:
        if verbose:
            print(msg)
    trading_days = _normalize_trading_days(out.index)

    news_path = news_path_template.format(ticker=ticker)
    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)

    # キャッシュキー（ticker + 期間 + 設定）
    start = str(trading_days.min().date())
    end = str(trading_days.max().date())
    _p(
        "[NEWS] add_news_features: "
        f"ticker={ticker} span={start}..{end} trading_days={len(trading_days)} "
        f"use_meta={use_meta} use_sentiment={use_sentiment} "
        f"roll={list(rolling_windows)} long_window={long_window} close={market_close_time} tz={tz}"
    )
    _p(f"[NEWS] raw_path={news_path} exists={Path(news_path).exists()} cache_dir={str(cache_root)}")
    rw = "-".join(map(str, rolling_windows))
    cache_name = f"newsfeat_{NEWS_FEAT_CACHE_VER}_{ticker}_{start}_{end}_A{int(use_meta)}B{int(use_sentiment)}_w{rw}_lw{long_window}_close{market_close_time.replace(':','')}.parquet"
    cache_path = cache_root / cache_name
    meta_path = cache_root / (cache_name + ".meta.json")

    if allow_cache and cache_path.exists():
        _p(f"[NEWS] cache_hit: {cache_path.name}")
        daily = pd.read_parquet(cache_path)
        daily.index = pd.to_datetime(daily.index).normalize()
        daily = daily.reindex(trading_days).fillna(0.0)
    else:
        _p(f"[NEWS] cache_miss: reading raw csv -> {news_path}")
        raw = load_news(news_path)
        _p(f"[NEWS] raw_loaded: shape={raw.shape} cols={list(raw.columns)}")

        # ticker列が空なら “このtickerのニュースだけ” を想定して埋める
        if raw["ticker"].astype(str).str.len().max() == 0:
            raw["ticker"] = ticker

        raw = raw.loc[raw["ticker"].astype(str) == ticker].copy()
        _p(f"[NEWS] raw_after_ticker_filter: shape={raw.shape}")

        aligned = align_news_to_trading_day(
            raw,
            trading_days,
            tz=tz,
            market_close_time=market_close_time,
            published_col="published_at",
        )
        if len(aligned) == 0:
            _p("[NEWS] aligned: 0 rows (no news in this span after alignment)")
        else:
            n_trade_dates = int(aligned["trade_date"].nunique()) if "trade_date" in aligned.columns else 0
            ac_ratio = float(aligned["is_after_close"].mean()) if "is_after_close" in aligned.columns else 0.0
            _p(f"[NEWS] aligned: rows={len(aligned)} trade_dates={n_trade_dates} after_close_ratio≈{ac_ratio:.3f}")

        daily = featurize_news_daily(
            aligned,
            trading_days,
            use_meta=use_meta,
            use_sentiment=use_sentiment,
            tz=tz,
            market_close_time=market_close_time,
            rolling_windows=rolling_windows,
            long_window=long_window,
            text_cols=text_cols,
        )
        _p(f"[NEWS] daily_features: shape={daily.shape} cols={len(daily.columns)}")

        if allow_cache:
            daily.to_parquet(cache_path)
            meta = {
                "ticker": ticker,
                "news_path": str(Path(news_path).as_posix()),
                "tz": tz,
                "market_close_time": market_close_time,
                "use_meta": use_meta,
                "use_sentiment": use_sentiment,
                "rolling_windows": list(rolling_windows),
                "long_window": long_window,
                "date_span": [start, end],
            }
            meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
            _p(f"[NEWS] cache_written: {cache_path.name}")

    # join & fill
    out = out.join(daily, how="left")
    # 念のため NaN は0へ（ただし既存特徴量は build_tabular_dataset がdropnaする）
    for c in daily.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce").fillna(0.0)
    
    added_cols = [c for c in daily.columns if c not in orig_cols]
    _p(f"[NEWS] joined: shape {orig_shape} -> {out.shape} (added_cols={len(added_cols)})")
    if added_cols:
        probe = [c for c in [
            "news_count_0d",
            "news_after_close_ratio_0d",
            "news_source_nunique_0d",
            "news_sent_score_mean_0d",
            "news_tone_mean_0d",
            "news_no_news_flag",
        ] if c in out.columns]
        if probe:
            _p("[NEWS] sample(last 5 trading days):")
            _p(out[probe].tail(5).to_string())

    return out
