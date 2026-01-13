EPS = 1e-8

TASK = "trinary"
if TASK != "trinary":
    raise SystemExit("[ERROR] This script is trinary-only. Please use a binary script for binary classification.")

#TOPIXは欠損が多いため一時的に無効化にするフラグを設置
USE_TOPIX_FEATURES = False  #falseで無効化

import time
import math
import numbers
import numpy as np
import pandas as pd
import yfinance as yf
import os
import json
import platform
from dataclasses import dataclass, field
from typing import Tuple, List, Dict, Optional
from matplotlib import patheffects

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif

import lightgbm as lgb

import matplotlib.pyplot as plt
import pathlib
from datetime import datetime

import sys  # noqa: F401  (kept for legacy compatibility)
from pathlib import Path
from stock_pred.common_shared import (
    _safe_float,
    _class_counts,
    _check_finite,
    compute_mda_importance_tabular,
    plot_importance,
    set_seeds,
    _ensure_dir,
    _plot_lgb_history,
    _plot_cm,
    finalize_run_summary,
    PWFESplit,
    LogConfig,
)

from stock_pred.dataset_pipeline import (
    _log_span,
    fetch_ohlcv as fetch_ohlcv_base,
    make_features,
    add_log1p_features,
    add_market_factors,
    add_rel_strength10,
    make_binary_excess_labels,
    to_windows,
    make_trinary_labels,
    build_tabular_dataset,
)
from stock_pred.news_pipeline import (
    add_news_features,
    DEFAULT_NEWS_META_COLS,
    DEFAULT_NEWS_SENT_COLS,
    DEFAULT_NEWS_TONE_COLS
)
# =========================
# Logging (match transformer/LSTM)
# =========================
LOGCFG = LogConfig(
    save_json=True,
    json_path="LightGBM/run_summary.json",
    print_fold_line=True,
)

def _is_scalar(x) -> bool:
    return isinstance(x, numbers.Number) or (np.isscalar(x))

# ======= Feature Utilities (adapted from transformer version) =======
def fetch_ohlcv(ticker: str, start: str, end: str) -> pd.DataFrame:
    """LightGBM側のMIN_ROWS条件だけ維持しつつ、取得ロジックはdataset_pipelineへ寄せる。"""
    df = fetch_ohlcv_base(ticker, start, end)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    min_rows = max(int(DATA.win) + int(DATA.horizon) + 1, 60)
    if len(df) < min_rows:
        raise RuntimeError(f"too few rows after fetch: {len(df)} < {min_rows}")
    return df


# ======= Configs =======
@dataclass
class DataConfig:
    ticker: str = "7203.T"
    start: str = "2001-01-01"
    end: str   = "2024-12-31"
    horizon: int = 1
    win: int = 60
    top_p: float = 0.10
    k_tau: float = 0.3

    k_tau_sweep: bool = False
    k_tau_grid: list[float] = field(default_factory=lambda: [0.5, 0.4, 0.3])
    win_sweep: bool = False
    win_grid: list[int] = field(default_factory=lambda: [30, 35, 40, 45, 50, 55, 60])
    best_win: int | None = None

    pooling: str = "last"   # "avg" or "last"
    use_log1p: bool = True
    log1p_candidates: list[str] = field(default_factory=lambda: [
        "sigma5","sigma20","bb_width","atr","atr_ratio","range_ma_ratio",
        "topix_sigma20","nikkei_sigma20","sp500_sigma20","vol_wk_ratio","vix"
    ])

    output_root: str = "LightGBM/figs"

    # ---- News features (early fusion) ----
    use_news: bool = False
    news_path: str = "data/news/raw/{ticker}.csv"      # {ticker} が置換される
    news_cache_dir: str = "data/news/features"
    news_tz: str = "Asia/Tokyo"
    news_market_close_time: str = "15:30"
    news_use_meta: bool = True       # 案A
    news_use_sent: bool = True       # 案B
    news_rolling_windows: tuple[int, int] = (3, 5)
    news_long_window: int = 20

@dataclass
class SplitConfig:
    n_splits: int = 5
    embargo: int = 5
    min_train: int = 252
    min_val: int = 63

@dataclass
class TrainConfig:
    # Map epochs ~ n_estimators, lr ~ learning_rate
    batch: int = 32
    epochs: int = 400         # trees
    lr: float = 0.05          # learning_rate
    es_tune: bool = False
    es_patience_default: int = 50
    es_mindelta_default: float = 0.0  # not used for LGBM
    es_patience_grid: list[int] = field(default_factory=lambda: [30, 50, 100])
    es_mindelta_grid: list[float] = field(default_factory=lambda: [0.0])
    mda_repeats: int = 3

@dataclass
class ModelConfig:
    num_leaves: int = 31
    max_depth: int = -1
    min_child_samples: int = 20
    subsample: float = 0.9
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 0.0

DATA = DataConfig()
SPLIT = SplitConfig()
TRAIN = TrainConfig()
MODEL = ModelConfig()


# ======= MI feature filter =======
def mi_select(Xtr_df: pd.DataFrame, ytr: np.ndarray, thr: float = 1e-4) -> List[str]:
    mi = mutual_info_classif(Xtr_df, ytr, discrete_features=False, random_state=42)
    s = pd.Series(mi, index=Xtr_df.columns).sort_values(ascending=False)
    feats = s[s > thr].index.tolist()
    return feats if len(feats) > 0 else Xtr_df.columns.tolist()


# ======= Windowing / pooling =======

def pool_windows(Xwin: np.ndarray, how: str) -> np.ndarray:
    if how == "avg":
        return Xwin.mean(axis=1)
    elif how == "last":
        return Xwin[:, -1, :]
    else:
        raise ValueError(f"Unknown pooling: {how}")


# ======= Metrics =======
def eval_trinary(y_true, y_pred) -> Dict:
    return {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "accuracy": (y_true == y_pred).mean(),
        # 固定ラベルでサイズを揃え、警告を回避
        "conf_mat": confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    }

def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    classes = np.unique(y)
    cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, cw_vals)}


# ======= MDA for tabular (pooled) =======
    sub = df_imp.iloc[:k].iloc[::-1]
    plt.figure(figsize=(8, 0.35*k + 1.5))
    plt.barh(sub["feature"], sub["importance"], xerr=sub.get("std", None))
    plt.xlabel("Score drop vs baseline")
    plt.title(title)
    plt.tight_layout(); plt.savefig(save_path, dpi=160); plt.close()

# ======= Metrics log helpers =======
def accumulate_metrics(metrics_log: list, fold: int, metrics_dict: dict, chosen_th: Optional[float] = None):
    row = {"fold": int(fold)}
    if chosen_th is not None:
        row["threshold"] = float(chosen_th)
    for k, v in metrics_dict.items():
        if _is_scalar(v):
            row[k] = float(v)
    metrics_log.append(row)

def save_and_plot_metrics(metrics_log: list, figdir: str, task: str):
    if not metrics_log:
        return
    df = pd.DataFrame(metrics_log).sort_values("fold")
    csv_path = f"{figdir}/metrics_all.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")
    num_cols = [c for c in df.columns if c not in ("fold",) and np.issubdtype(df[c].dtype, np.number)]
    for col in num_cols:
        vals = df[col].values
        mean = np.nanmean(vals)
        std  = np.nanstd(vals, ddof=1) if len(vals) > 1 else 0.0
        plt.figure(figsize=(7,4))
        plt.bar(df["fold"].astype(int).astype(str), vals)
        plt.ylabel(col); plt.xlabel("Fold")
        plt.title(f"{col} per Fold  (mean={mean:.3f}, sd={std:.3f})")
        plt.tight_layout(); plt.savefig(f"{figdir}/metric_{col}.png", dpi=160); plt.close()
    preferred = (["balanced_acc","f1","roc_auc","pr_auc","accuracy"] if task=="binary"
                 else ["macro_f1","balanced_acc","accuracy"])
    metrics_used = [m for m in preferred if m in df.columns]
    if metrics_used:
        means = [df[m].mean() for m in metrics_used]
        sds   = [df[m].std(ddof=1) if len(df) > 1 else 0.0 for m in metrics_used]
        x = np.arange(len(metrics_used))
        plt.figure(figsize=(8,4.5))
        plt.bar(x, means, yerr=sds, capsize=4)
        plt.xticks(x, metrics_used, rotation=15)
        plt.ylabel("score"); plt.title("Final Run – Metrics Summary (mean ± sd across folds)")
        plt.tight_layout(); plt.savefig(f"{figdir}/metrics_summary.png", dpi=160); plt.close()


# ======= Main pipeline =======
def main():
    run_start = time.time()

    # --- run summary header (JSON) ---
    run_summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "model": "lightgbm",
        "task": TASK,
        "data": {
            "ticker": getattr(DATA, "ticker", None),
            "start": getattr(DATA, "start", None),
            "end": getattr(DATA, "end", None),
            "win": int(getattr(DATA, "win", -1)),
            "k_tau": _safe_float(getattr(DATA, "k_tau", None)),
            "top_p": _safe_float(getattr(DATA, "top_p", None)),
            "output_root": getattr(DATA, "output_root", None),
        },
        "split": {
            "n_splits": int(getattr(SPLIT, "n_splits", -1)),
            "embargo": int(getattr(SPLIT, "embargo", -1)),
            "min_train": int(getattr(SPLIT, "min_train", -1)),
            "min_val": int(getattr(SPLIT, "min_val", -1)),
            "inner_val_frac": _safe_float(globals().get("INNER_VAL_FRAC", None)),
            "min_inner_val": int(globals().get("MIN_INNER_VAL", -1)),
        },
        "train": {
            "num_boost_round": int(getattr(TRAIN, "num_boost_round", getattr(TRAIN, "n_estimators", -1))),
            "early_stopping_rounds": int(getattr(TRAIN, "early_stopping_rounds", -1)),
        },
        "folds": [],
    }


    price = fetch_ohlcv(DATA.ticker, DATA.start, DATA.end)
    if price is None or price.empty:
        raise RuntimeError(f"No price data for {DATA.ticker} in [{DATA.start},{DATA.end}]")
    df = make_features(price)
    df = add_market_factors(df)
    df = add_rel_strength10(df)
    if DATA.use_log1p:
        df = add_log1p_features(df, DATA.log1p_candidates)

    # ---- add news features (early fusion; BEFORE labeling) ----
    if DATA.use_news:
        df = add_news_features(
            df,
            ticker=DATA.ticker,
            news_path_template=DATA.news_path,
            cache_dir=DATA.news_cache_dir,
            tz=DATA.news_tz,
            market_close_time=DATA.news_market_close_time,
            use_meta=DATA.news_use_meta,
            use_sentiment=DATA.news_use_sent,
            rolling_windows=DATA.news_rolling_windows,
            long_window=DATA.news_long_window,
        )
    else:
        print("[NEWS] add_news_features: skipped (DATA.use_news=False)")

    y_all, abs_metric = make_trinary_labels(df, horizon=DATA.horizon, k_tau=DATA.k_tau)

    feature_cols = [
        # price/return
        "ret1","ret5","ret10","ret20","devMA5","devMA20","devMA60",
        # vola
        "sigma5","sigma20","bb_width","atr","atr_ratio","range_ma_ratio","atr_diff",
        # vola regime
        "sigma_ratio",
        # technical
        "rsi14","macd","macd_sig","stoch_k","stoch_d","mom10","mom20","ema_diff",
        # candle ratios
        "body_ratio","upper_shadow_ratio","lower_shadow_ratio",
        # trend structure
        "slope30","slope60","resid30_z","resid60_z",
        # volume
        "vol_chg","vol_ratio","obv_ratio","vol_wk_ratio",
        # gap
        "gap",
        # anomalies
        "price_anom","vol_anom","gap_anom",
        # markets
        "topix_ret1","topix_sigma20","nikkei_ret1","nikkei_sigma20","sp500_ret1","sp500_sigma20","vix_ret1",
        # market-level
        "rel_strength10","vix_log1p"
    ]

    if DATA.use_news:
        if DATA.news_use_meta:
            feature_cols += DEFAULT_NEWS_META_COLS
        if DATA.news_use_sent:
            feature_cols += DEFAULT_NEWS_SENT_COLS
            feature_cols += DEFAULT_NEWS_TONE_COLS

    X_all, y_all_np, abs_all, feature_cols_final, data = build_tabular_dataset(
        df,
        y=y_all,
        abs_metric=abs_metric,
        feature_cols=feature_cols,
        use_topix_features=USE_TOPIX_FEATURES,
        use_log1p=DATA.use_log1p,
        log1p_candidates=DATA.log1p_candidates,
        extra_log1p_cols=("vix_log1p",),   # Transformerでは手動拾いがあったので維持
        nan_tail_days=120,
    )
    y_all = y_all_np  # 以降の既存コードが y_all を使うので上書きして互換維持

    news_cols = [c for c in feature_cols_final if c.startswith("news_")]
    if news_cols:
        print(f"[NEWS] feature_cols: {len(news_cols)} cols (head/tail) = {news_cols[:5]} ... {news_cols[-5:]}")
        probe = [c for c in ["news_count_0d","news_sent_score_mean_0d","news_tone_mean_0d","news_no_news_flag"] if c in data.columns]
        if probe:
            print("[NEWS] sample(last 5 rows):")
            print(data[probe].tail(5).to_string())
    else:
        print("[NEWS] feature_cols: 0 (news disabled or not merged)")

    print(f"[INFO] Final features d={len(feature_cols_final)} (head): {feature_cols_final[:12]}")
    _log_span(data, "final usable rows before split")

    splitter = PWFESplit(n_splits=SPLIT.n_splits, embargo=SPLIT.embargo, min_train=SPLIT.min_train, min_val=SPLIT.min_val)
    splits = splitter.split(len(data))
    print(f"[SPLIT] Effective folds: {len(splits)}")
    if splits:
        va_spans = [(int(va[0]), int(va[-1])) for (_, va) in splits]
        print("[SPLIT] VAL spans (idx):", va_spans)
        try:
            first_dt = data.index[va_spans[0][0]]
            last_dt  = data.index[va_spans[-1][1]]
            print(f"[SPLIT] VAL date range: {first_dt.date()} .. {last_dt.date()}")
        except Exception:
            pass

    fold_metrics = []
    subset_metrics = []
    all_true_main, all_pred_main = [], []
    all_true_sub,  all_pred_sub  = [], []
    figdir = f"{DATA.output_root}/{TASK}"
    _ensure_dir(figdir)
    metrics_log = []

    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        fold_start = time.time()
        Xtr_df = pd.DataFrame(X_all[tr_idx], columns=feature_cols_final)
        ytr    = y_all[tr_idx]

        INNER_VAL_FRAC = 0.20
        MIN_INNER_VAL  = 64
        n_tr_total = len(Xtr_df)
        n_iv = max(int(n_tr_total * INNER_VAL_FRAC), MIN_INNER_VAL)
        n_iv = min(n_iv, n_tr_total // 2)
        cut  = n_tr_total - n_iv
        tr_in_idx = np.arange(0, cut)
        iv_in_idx = np.arange(cut, n_tr_total)

        feats  = mi_select(Xtr_df.iloc[tr_in_idx], ytr[tr_in_idx], thr=1e-4)
        scaler = StandardScaler().fit(Xtr_df.iloc[tr_in_idx][feats].values)
        Xtr_all = scaler.transform(Xtr_df[feats].values)
        Xva = scaler.transform(pd.DataFrame(X_all[va_idx], columns=feature_cols_final)[feats].values)

        Xtr_in = Xtr_all[tr_in_idx]; ytr_in = ytr[tr_in_idx]
        Xiv_in = Xtr_all[iv_in_idx]; yiv_in = ytr[iv_in_idx]

        Xtr_win, ytr_win = to_windows(Xtr_in, ytr_in, DATA.win)
        Xiv_win,    yiv_win    = to_windows(Xiv_in, yiv_in, DATA.win)
        Xva_win,    yva_win    = to_windows(Xva,     y_all[va_idx], DATA.win)
        abs_va_win = abs_all[va_idx][DATA.win:]

        Xtr_inn_win = Xtr_win
        ytr_inn_win = ytr_win

        # inner/outer logging
        abs_tr_in_idx = tr_idx[tr_in_idx]
        abs_iv_in_idx = tr_idx[iv_in_idx]
        dt = data.index
        print(f"[SPLIT] Fold {fold} [INNER-POLICY] frac={INNER_VAL_FRAC:.2f}, min={MIN_INNER_VAL}, win={DATA.win}")
        print(f"[SPLIT] Fold {fold} [INNER] TR_IN[{abs_tr_in_idx[0]}..{abs_tr_in_idx[-1]}] len={len(abs_tr_in_idx)} | "
              f"IV_IN[{abs_iv_in_idx[0]}..{abs_iv_in_idx[-1]}] len={len(abs_iv_in_idx)} | "
              f"OUTER[{va_idx[0]}..{va_idx[-1]}] len={len(va_idx)}")
        print(f"[SPLIT] Fold {fold} [DATES] inner={dt[abs_tr_in_idx[0]].date()}..{dt[abs_iv_in_idx[-1]].date()} | "
              f"outer={dt[va_idx[0]].date()}..{dt[va_idx[-1]].date()}")
        print(f"[SPLIT] Fold {fold} [COUNT][RAW] tr_in={len(tr_in_idx)} | iv_in={len(iv_in_idx)} | va_out={len(va_idx)}")
        print(f"[SPLIT] Fold {fold} [COUNT][WIN] tr_in={len(ytr_inn_win)} | iv_in={len(yiv_win)} | va_out={len(yva_win)}")

        Xtr_pool = pool_windows(Xtr_inn_win, DATA.pooling)
        Xiv_pool = pool_windows(Xiv_win,    DATA.pooling)
        Xva_pool = pool_windows(Xva_win,    DATA.pooling)

        n_classes = 3
        class_weights = _compute_class_weights(ytr_inn_win)
        sample_weight = np.array([class_weights[int(c)] for c in ytr_inn_win])

        params = dict(
            n_estimators=TRAIN.epochs,
            learning_rate=TRAIN.lr,
            objective='multiclass',
            num_class=n_classes,
            subsample=MODEL.subsample,
            colsample_bytree=MODEL.colsample_bytree,
            num_leaves=MODEL.num_leaves,
            max_depth=MODEL.max_depth,
            min_child_samples=MODEL.min_child_samples,
            force_col_wise=True,
            reg_alpha=MODEL.reg_alpha,
            reg_lambda=MODEL.reg_lambda,
            random_state=42
        )
        clf = lgb.LGBMClassifier(**params)
        eval_result = {}
        # Put validation first so early_stopping monitors it
        clf.fit(
            Xtr_pool, ytr_inn_win,
            sample_weight=sample_weight,
            eval_set=[(Xiv_pool, yiv_win), (Xtr_pool, ytr_inn_win)],
            eval_metric='multi_logloss',
            callbacks=[
                lgb.early_stopping(stopping_rounds=TRAIN.es_patience_default, verbose=False),
                lgb.record_evaluation(eval_result)
            ]
        )

        # Plot learning curve similar to Keras hist
        _plot_lgb_history(eval_result if eval_result else getattr(clf, 'evals_result_', {}), figdir, fold, TASK,
                          extra_title=f"(win={DATA.win}, k_tau={DATA.k_tau:.2f})")

        proba = clf.predict_proba(Xva_pool)
        yhat = np.argmax(proba, axis=1)
        m = eval_trinary(yva_win, yhat)

        accumulate_metrics(metrics_log, fold, m)

        try:
            imp = compute_mda_importance_tabular(
                clf, Xva_pool, yva_win,
                metric_name="macro_f1",
                feature_names=feats, n_repeats=TRAIN.mda_repeats, random_state=42
            )
            imp_csv = f"{figdir}/mda_fold{fold}.csv"
            imp.to_csv(imp_csv, index=False, encoding="utf-8")
            plot_importance(
                imp, save_path=f"{figdir}/mda_fold{fold}.png", topk=20,
                title=f"MDA (metric=Macro-F1) Fold{fold} [win={DATA.win}, k_tau={DATA.k_tau:.2f}]"
            )
            print(f"[MDA] saved: {imp_csv} & {figdir}/mda_fold{fold}.png (repeats={TRAIN.mda_repeats})")
        except Exception as e:
            print(f"[MDA][WARN] importance failed: {e}")

        all_true_main.extend(yva_win.tolist())
        all_pred_main.extend(yhat.tolist())

        k = max(1, int(len(abs_va_win) * DATA.top_p))
        idx_top = np.argsort(abs_va_win)[-k:]
        m_sub = eval_trinary(yva_win[idx_top], yhat[idx_top])
        all_true_sub.extend(yva_win[idx_top].tolist())
        all_pred_sub.extend(yhat[idx_top].tolist())

        # --- fold record (JSON) ---
        n_classes = 3 if TASK == "trinary" else 2
        fold_rec = {
            "fold": int(fold),
            "counts_raw": {
                "tr_in": int(len(abs_tr_in_idx)),
                "iv_in": int(len(abs_iv_in_idx)),
                "va_out": int(len(va_idx)),
            },
            "counts_win": {
                "tr_in": int(len(ytr_inn_win)),
                "iv_in": int(len(yiv_win)),
                "va_out": int(len(yva_win)),
            },
            "class_counts": {
                "train": _class_counts(ytr_inn_win, n_classes),
                "inner_valid": _class_counts(yiv_win, n_classes),
                "outer_valid": _class_counts(yva_win, n_classes),
            },
            "finite_checks": {
                "Xtr_inn_win": _check_finite("Xtr_inn_win", Xtr_inn_win),
                "Xiv_win": _check_finite("Xiv_win", Xiv_win),
                "Xva_win": _check_finite("Xva_win", Xva_win),
            },
            "metrics_main": {k: v for k, v in m.items() if k != "conf_mat"},
            "metrics_subset": {k: v for k, v in m_sub.items() if k != "conf_mat"},
            "conf_mat_main": m["conf_mat"].tolist(),
            "conf_mat_subset": m_sub["conf_mat"].tolist(),
            "elapsed_sec": float(time.time() - fold_start),
        }
        run_summary["folds"].append(fold_rec)

        if LOGCFG.print_fold_line:
            print(
                f"[FOLD {fold}] "
                f"main(macro_f1={m.get('macro_f1'):.4f}, bal_acc={m.get('balanced_acc'):.4f}) | "
                f"sub(macro_f1={m_sub.get('macro_f1'):.4f}, bal_acc={m_sub.get('balanced_acc'):.4f}) | "
                f"counts(win tr={len(ytr_inn_win)}, iv={len(yiv_win)}, va={len(yva_win)})"
            )

        fold_metrics.append(m)
        subset_metrics.append(m_sub)
        if LOGCFG.print_confusion:
            print(f"[Fold {fold}] main={ {k: v for k, v in m.items() if k!='conf_mat'} }")
            print(f"[Fold {fold}] main confusion=\n{m['conf_mat']}")
            print(f"[Fold {fold}] subset(top{int(DATA.top_p*100)}%)={ {k: v for k, v in m_sub.items() if k!='conf_mat'} }")
            print(f"[Fold {fold}] subset confusion=\n{m_sub['conf_mat']}")
            print(f"[Fold {fold}] elapsed: {time.time()-fold_start:.2f}s")

    save_and_plot_metrics(metrics_log, figdir, TASK)

    labels = [0, 1, 2]
    cm_main = confusion_matrix(all_true_main, all_pred_main, labels=labels)
    cm_sub  = confusion_matrix(all_true_sub,  all_pred_sub,  labels=labels)
    print("\n=== Overall confusion matrix (main) ===")
    print(cm_main)
    print("\n=== Overall confusion matrix (subset top{:.0f}%) ===".format(DATA.top_p*100))
    print(cm_sub)

    # Row-normalized confusion matrices (to match transformer logs)
    def _row_norm(cm):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            r[np.isnan(r)] = 0.0
            return r
    print("\n=== Overall confusion matrix (main, row-normalized) ===")
    print(_row_norm(cm_main))
    print("\n=== Overall confusion matrix (subset, row-normalized) ===")
    print(_row_norm(cm_sub))

    # Save CM figures (counts & row-normalized)
    try:
        cls_labels = [0, 1, 2]
        _plot_cm(cm_main, cls_labels, f"{figdir}/cm_overall.png", normalize=False, title_prefix="Overall")
        _plot_cm(cm_main, cls_labels, f"{figdir}/cm_norm_overall.png", normalize=True,  title_prefix="Overall")
        _plot_cm(cm_sub,  cls_labels, f"{figdir}/cm_overall_subset.png", normalize=False, title_prefix="Subset")
        _plot_cm(cm_sub,  cls_labels, f"{figdir}/cm_norm_overall_subset.png", normalize=True,  title_prefix="Subset")
        print(f"[PLOT] saved overall & subset CM figures to {figdir}")
    except Exception as e:
        print(f"[PLOT][WARN] overall CM plotting failed: {e}")

    # Pooled metrics across all outer validations
    pooled_macro_f1_main = f1_score(all_true_main, all_pred_main, average="macro") if len(all_true_main) else np.nan
    pooled_bal_acc_main  = balanced_accuracy_score(all_true_main, all_pred_main)   if len(all_true_main) else np.nan
    pooled_macro_f1_sub  = f1_score(all_true_sub,  all_pred_sub,  average="macro") if len(all_true_sub)  else np.nan
    pooled_bal_acc_sub   = balanced_accuracy_score(all_true_sub,  all_pred_sub)    if len(all_true_sub)  else np.nan
    print(f"\n[POOLED][MAIN] macro_f1={pooled_macro_f1_main:.6f}, balanced_acc={pooled_bal_acc_main:.6f}, N={len(all_true_main)}")
    print(f"[POOLED][SUB ] macro_f1={pooled_macro_f1_sub:.6f}, balanced_acc={pooled_bal_acc_sub:.6f}, N={len(all_true_sub)}")

    # Aggregate per-fold metrics into mean/sd (same style as transformer)
    def agg(key, arr):
        vals = [d[key] for d in arr if key in d and not (isinstance(d[key], float) and np.isnan(d[key]))]
        return (float(np.mean(vals)), float(np.std(vals))) if len(vals) else (np.nan, np.nan)

    mu_bal, sd_bal = agg("balanced_acc", fold_metrics)
    mu_f1 , sd_f1  = agg("macro_f1",    fold_metrics)
    mu_bal_s, sd_bal_s = agg("balanced_acc", subset_metrics)
    mu_f1_s , sd_f1_s  = agg("macro_f1",    subset_metrics)

    summary_main = {
        "balanced_acc_mean":   mu_bal,
        "balanced_acc_sd":     sd_bal,
        "macro_f1_mean":       mu_f1,
        "macro_f1_sd":         sd_f1,
        "pooled_macro_f1":     float(pooled_macro_f1_main),
        "pooled_balanced_acc": float(pooled_bal_acc_main),
        "pooled_N":            int(len(all_true_main)),
    }
    summary_subset = {
        "subset_top_p":        DATA.top_p,
        "balanced_acc_mean":   mu_bal_s,
        "balanced_acc_sd":     sd_bal_s,
        "macro_f1_mean":       mu_f1_s,
        "macro_f1_sd":         sd_f1_s,
        "pooled_macro_f1":     float(pooled_macro_f1_sub),
        "pooled_balanced_acc": float(pooled_bal_acc_sub),
        "pooled_N":            int(len(all_true_sub)),
    }

    print(summary_main)
    print(summary_subset)
    print(f"[RUN] total elapsed: {time.time()-run_start:.2f}s")

    # --- finalize JSON summary ---
    run_summary["summary_main"] = summary_main
    run_summary["summary_subset"] = summary_subset
    finalize_run_summary(
        run_summary,
        pooled_main={"macro_f1": float(pooled_macro_f1_main), "balanced_acc": float(pooled_bal_acc_main), "N": int(len(all_true_main))},
        pooled_subset={"macro_f1": float(pooled_macro_f1_sub), "balanced_acc": float(pooled_bal_acc_sub), "N": int(len(all_true_sub))},
        logcfg=LOGCFG,
    )

    # Used by sweep (if enabled), consistent with transformer
    return {
        "K_TAU": DATA.k_tau,
        # Align with transformer_old: prefer pooled metric for selection
        "metric_name": "pooled_macro_f1",
        "select_metric": float(pooled_macro_f1_main) if not np.isnan(pooled_macro_f1_main) else float(mu_f1),
        "summary_main": summary_main,
        "summary_subset": summary_subset,
    }


if __name__ == "__main__":
    set_seeds(42)
    main()
