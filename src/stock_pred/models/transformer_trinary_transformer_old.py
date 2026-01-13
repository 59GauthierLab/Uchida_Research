EPS = 1e-8     # 0除算を避けるための極小値

TASK  = "trinary"   # このコードは三値分類専用
if TASK != "trinary":
    raise SystemExit("[ERROR] This script is trinary-only. Please use binary_transformer.py for binary classification.")

#TOPIXは欠損が多いため一時的に無効化にするフラグを設置
USE_TOPIX_FEATURES = False  #falseで無効化

import time
import os
import math
import numbers
import numpy as np
import pandas as pd
import yfinance as yf
from dataclasses import dataclass, field
from typing import Tuple, List, Dict
from sklearn.utils.class_weight import compute_class_weight

# ML / Metrics
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, balanced_accuracy_score, confusion_matrix, roc_auc_score
from sklearn.feature_selection import mutual_info_classif

# DL
import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dense, Dropout, LayerNormalization, MultiHeadAttention, GlobalAveragePooling1D, Lambda

# plots
import matplotlib.pyplot as plt
import pathlib
from datetime import datetime
from typing import Optional

# logs
import json
import platform
from dataclasses import dataclass
from datetime import datetime

import sys  # noqa: F401  (kept for legacy compatibility)
from pathlib import Path
from stock_pred.common_shared import (
    _safe_float,
    _class_counts,
    _check_finite,
    _score_trinary,
    compute_mda_importance,
    plot_importance,
    set_seeds,
    _ensure_dir,
    _plot_history,
    _plot_cm,
    finalize_run_summary,
    PWFESplit,
    LogConfig,
)
from stock_pred.dataset_pipeline import (
    _log_span,
    fetch_ohlcv,
    make_features,
    add_log1p_features,
    add_market_factors,
    add_rel_strength10,
    make_trinary_labels,
    make_binary_excess_labels,
    to_windows,
    build_tabular_dataset,
)
from stock_pred.news_pipeline import (
    add_news_features,
    DEFAULT_NEWS_META_COLS,
    DEFAULT_NEWS_SENT_COLS,
    DEFAULT_NEWS_TONE_COLS
)

LOGCFG = LogConfig(
    quiet_fit=True,
    save_json=True,
    json_path="transformer/run_summary.json",
    print_fold_line=True,
)

# =========================
# User settings
# =========================
@dataclass
class DataConfig:
    """データ前処理と特徴量派生・出力先など、データ寄りの設定集。"""
    ticker: str = "7203.T"          # 対象ティッカー
    start: str = "2001-01-01"       # 取得開始日
    end: str = "2024-12-31"         # 取得終了日
    horizon: int = 1                # 何日先(H)を予測するか
    win: int = 60                   # 時系列窓幅(Transformerへの入力長)
    top_p: float = 0.10             # 異常サブセット評価(|r_H|上位p%)
    k_tau: float = 0.3              # 三値ラベリングの閾値スケール係数

    # グリッド探索の可否と候補
    k_tau_sweep: bool = False
    k_tau_grid: list[float] = field(default_factory=lambda: [0.5, 0.4, 0.3])
    win_sweep: bool = False
    win_grid: list[int] = field(default_factory=lambda: [30, 35, 40, 45, 50, 55, 60])
    best_win: int | None = None     # SWEEP結果のベスト記録用変数

    # 前処理
    pooling: str = "last"           # Transformer出力の集約方式("avg" of "last")
    use_log1p: bool = True          # 非負量にLog1p派生を追加するか
    log1p_candidates: list[str] = field(default_factory=lambda: [
        "sigma5","sigma20","bb_width","atr","atr_ratio","range_ma_ratio",
        "topix_sigma20","nikkei_sigma20","sp500_sigma20","vol_wk_ratio","vix"
    ])

    # 図・CSVの出力先
    output_root: str = "transformer/figs"

    use_news: bool = True
    news_path: str = "data/news/raw/{ticker}.csv"
    news_cache_dir: str = "data/news/features"
    news_tz: str = "Asia/Tokyo"
    news_market_close_time: str = "15:30"
    news_use_meta: bool = True
    news_use_sent: bool = True
    news_rolling_windows: tuple[int, int] = (3, 5)
    news_long_window: int = 20

@dataclass
class SplitConfig:
    """Purged Walk-Forward with Embargo（PWFE）用の分割設定。"""
    n_splits: int = 5           # 分割数
    embargo: int = 5            # 検証直前の学習区間末尾に設ける禁止期間
    min_train: int = 252        # 各フォールドでの最低学習サンプル数 (1年 ≒ 252営業日)
    min_val: int = 63           # 各フォールドの最低検証サンプル数 (四半期 ≒ 63営業日)

@dataclass
class TrainConfig:
    """学習ハイパラとコールバック（EarlyStopping/LRスケジュール等）。"""
    batch: int = 32
    epochs: int = 80
    lr: float = 1e-4

    # EarlyStopping
    es_tune: bool = False
    es_patience_default: int = 10
    es_mindelta_default: float = 1e-3
    es_patience_grid: list[int] = field(default_factory=lambda: [10, 15, 20])
    es_mindelta_grid: list[float] = field(default_factory=lambda: [1e-4, 5e-4, 1e-3])

    # 学習率スケジューラ
    use_lr_scheduler: bool = False      # ReduceLROnPlateauを使う場合True
    lr_reduce_factor: float = 0.5
    lr_reduce_patience: int = 5
    lr_min: float = 1e-6
    use_cosine_lr: bool = True          # こちらがTrueならCosineを優先して使う
    cosine_base_lr: float = 1e-4
    cosine_alpha: float = 0.07          # 最小学習率 = base_lr * alpha
    cosine_alpha_sweep: bool = False
    cosine_alpha_grid: list[float] = field(default_factory=lambda: [0.03, 0.05, 0.07, 0.10])

    #LabelSmoothing
    use_label_smoothing: bool = False
    label_smoothing_eps: float = 0.05

    #MDAの反復回数
    mda_repeats: int = 3

@dataclass
class ModelConfig:
    """Transformerブロックの主ハイパラ。"""
    dropout: float   = 0.2      # Attention/FFN後のドロップアウト率
    head_size: int   = 64       # MHAのkey_dim (埋め込みの次元分割の基準)
    num_heads: int   = 4        # マルチヘッド数
    ff_dim: int      = 128      # FFNの中間層ユニット数

# グローバル設定オブジェクト
DATA = DataConfig()
SPLIT = SplitConfig()
TRAIN = TrainConfig()
MODEL = ModelConfig()

def make_cosine_lr_callback(base_lr: float, alpha: float, total_epochs: int):
    """Cosine Annealing（エポック単位）。最小学習率 = base_lr * alpha。
    - epochに応じてcosカーブで滑らかに学習率を下げる（Warm Restartsなしの単発版）。
    """
    def _schedule(epoch, lr):
        t = min(max(epoch / max(total_epochs, 1), 0.0), 1.0)
        new_lr = base_lr * (alpha + (1.0 - alpha) * 0.5 * (1.0 + math.cos(math.pi * t)))
        return float(new_lr)
    return tf.keras.callbacks.LearningRateScheduler(_schedule, verbose=0)

def _log_btf(name: str, x) -> None:
    """(B, T, F) 形状を安全にログする簡易ヘルパ。"""
    try:
        shp = getattr(x, "shape", None)
        if shp is None or len(shp) != 3:
            print(f"[SHAPE][WARN] {name}: unexpected shape={shp}")
            return
        b, t, f = shp
        print(f"[SHAPE] {name}: B={b}, T={t}, F={f}")
    except Exception as e:
        print(f"[SHAPE][WARN] {name}: failed to read shape ({e})")


# =========================
# Utilities
# =========================

# ---------- MI 選択（学習データでのみ） ----------
def mi_select(Xtr_df: pd.DataFrame, ytr: np.ndarray, thr: float = 1e-4) -> List[str]:
    """相互情報量（Mutual Information）で単純に有効特徴をフィルタ。
    - 学習データのみで実施（検証情報を一切見ない＝リーク防止）。
    - MI>thr の列を返す。該当が0本なら安全に“全列”を返すフェイルセーフ。
    """
    mi = mutual_info_classif(Xtr_df, ytr, discrete_features=False, random_state=42)
    s = pd.Series(mi, index=Xtr_df.columns).sort_values(ascending=False)
    feats = s[s > thr].index.tolist()
    return feats if len(feats) > 0 else Xtr_df.columns.tolist()

# ---------- Transformer ----------
class PositionalEncoding(tf.keras.layers.Layer):
    """
    位置エンコーディング（Sin/Cosの固定埋め込み）。
    - 連続的な時系列順序情報を、特徴埋め込みへ足し込むレイヤ。
    - 事前計算した (1, T, D) のテンソルを forward で加算するだけなので軽量。
    """
    def __init__(self, seq_len, d_model):
        super().__init__()
        pos = np.arange(seq_len)[:, None]       # 位置 0..T-1 (T, 1)
        i = np.arange(d_model)[None, :]         # 次元 0..D-1 (1, D)
        angle_rates = 1 / np.power(10000, (2 * (i//2)) / np.float32(d_model))
        angle_rads = pos * angle_rates          # (T, D)

        # 偶数次元は sin、奇数次元は cos
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        # (1, T, D) にして保存（学習しない固定PE）
        self.pe = tf.constant(angle_rads[None, ...], dtype=tf.float32)

    def call(self, x):  # (B,T,D)
        # バッチ全体に同じPEを加算（Tは実際の系列長でトリム）
        return x + self.pe[:, :tf.shape(x)[1], :]

def transformer_model(win: int, fdim: int, n_classes: int) -> tf.keras.Model:
    """
    シンプルなEncoder風ブロック ×2 層のTransformer分類器。
    - 入力: (B, T=win, F=fdim)
    - 構造: [LN → MHA → Dropout → 残差] → [LN → FFN → Dropout → 残差] を2回
    - プーリング: avg または last（設定 DATA.pooling に従う）
    - 出力: 2値は sigmoid、3値は softmax（label smoothing はオプション）
    """
    inp = Input(shape=(win, fdim))
    x = PositionalEncoding(win, fdim)(inp)

    # --- Encoderブロック ×2 ---
    for _ in range(2):
        # ブロック1: Self-Attention
        res = x
        x = LayerNormalization(epsilon=1e-6)(x)
        x = MultiHeadAttention(
            num_heads=MODEL.num_heads, 
            key_dim=MODEL.head_size,
            output_shape=(fdim,),       # 出力次元を元のfdimに合わせて残差接続可能に
            dropout=MODEL.dropout
            )(x, x)
        x = Dropout(MODEL.dropout)(x)
        x = x + res                     # 残差

        # ブロック2: Position-wise FFN
        res2 = x
        x = LayerNormalization(epsilon=1e-6)(x)
        x = Dense(MODEL.ff_dim, activation="relu")(x)
        x = Dropout(MODEL.dropout)(x)
        x = Dense(fdim)(x)              # 元次元に戻す（残差接続のため）
        x = x + res2

    # --- 出力前の正規化 ---
    x = LayerNormalization(epsilon=1e-6)(x)
    
    # --- プーリング選択 ---
    if DATA.pooling == "avg":
        # 全タイムステップの平均（系列全体の情報を滑らかに集約）
        x = GlobalAveragePooling1D(name="pool_avg")(x)
    elif DATA.pooling == "last":
        # 最終タイムステップのみ（直近情報を重視）
        x = Lambda(lambda t: t[:, -1, :], name="pool_last")(x)
    else:
        raise ValueError(f"Unknown POOLING: {DATA.pooling}. Use 'avg' or 'last'.")
    
    x = Dropout(MODEL.dropout)(x)

    # --- 出力層と損失 ---
    if n_classes == 2:
        # 2値分類（出力1ユニットの確率）
        out = Dense(1, activation="sigmoid")(x)
        loss_fn = tf.keras.losses.BinaryCrossentropy()
        metrics = ["accuracy", tf.keras.metrics.AUC(name="roc_auc")]
    else:
        # 3値分類
        out = Dense(n_classes, activation="softmax")(x)
        loss_fn = (tf.keras.losses.CategoricalCrossentropy(label_smoothing=TRAIN.label_smoothing_eps)
                if TRAIN.use_label_smoothing 
                else tf.keras.losses.SparseCategoricalCrossentropy()
        )
        metrics = ["accuracy"]

    # モデルを組み立てる前に想定形状をログ
    print(f"[MODEL] build: expect input (None, {win}, {fdim}), classes={n_classes}")
    m = Model(inp, out)
    # Keras summary も一度だけ出力（行頭にタグを付けて見分けやすく）
    try:
        m.summary(print_fn=lambda s: print("[MODEL][SUMMARY]", s))
    except Exception:
        pass
    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=TRAIN.lr),
        loss=loss_fn,
        metrics=metrics
    )
    return m

# ---------- 評価 ----------
def eval_trinary(y_true, y_pred) -> Dict:
    """三値：macro-F1 と balanced accuracy、混同行列を返す。"""
    return {
        "balanced_acc": _score_trinary(y_true, y_pred, metric="balanced_acc"),
        "macro_f1": _score_trinary(y_true, y_pred, metric="macro_f1"),
        "conf_mat": confusion_matrix(y_true, y_pred)
    }

def eval_binary(y_true, y_prob) -> Dict:
    """
    二値：確率ベースの評価（ROC-AUC）としきい値0.5のF1/BA。
    - y_prob は陽性確率（sigmoidの出力 or softmaxの第1クラス）。
    """
    y_pred = (y_prob >= 0.5).astype(int)
    out = {
        "balanced_acc": balanced_accuracy_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred, average="binary"),
    }
    try:
        out["roc_auc"] = roc_auc_score(y_true, y_prob)
    except Exception:
        out["roc_auc"] = np.nan
    out["conf_mat"] = confusion_matrix(y_true, y_pred)
    return out

# ---------- 不均衡対策（class_weight） ----------
def _compute_class_weights(y: np.ndarray) -> Dict[int, float]:
    """
    学習データのクラス分布から重みを計算。
    - “稀なクラスほど重く”なることで損失寄与をバランスさせる。
    """
    classes = np.unique(y)
    cw_vals = compute_class_weight(class_weight="balanced", classes=classes, y=y)
    return {int(c): float(w) for c, w in zip(classes, cw_vals)}

# ---------- labelsmoothing -------------
def _prepare_targets(y: np.ndarray, n_classes: int, use_label_smoothing: bool) -> np.ndarray:
    """
    Keras fit() に渡す教師データの整形。
    - 3値かつ label_smoothing=True のときは one-hot へ変換。
    - それ以外は整数ラベルのまま。
    """
    if n_classes == 3 and use_label_smoothing:
        return tf.keras.utils.to_categorical(y, num_classes=n_classes)
    return y

# ---------- コールバック構成 ----------
def _build_callbacks(task: str, patience: int, min_delta: float) -> list:
    """
    EarlyStopping + 学習率スケジューラをまとめて生成。
    - 2値: val_roc_auc 最大化を監視
    - 3値: val_loss 最小化を監視
    - Cosine LR を優先的に利用（指定があれば ReduceLROnPlateau にフォールバック）
    """
    if task == "binary":
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_roc_auc", mode="max",
            patience=patience, min_delta=min_delta,
            restore_best_weights=True,
            verbose = 0 if LOGCFG.quiet_fit else 1
        )
    else:
        es = tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", mode="min",
            patience=patience, min_delta=min_delta,
            restore_best_weights=True,
            verbose = 0 if LOGCFG.quiet_fit else 1
        )

    cbs = [es]

    # 優先: Cosine Annealing（LRをエポックに応じてなめらかに減衰）
    if TRAIN.use_cosine_lr:
        cbs.append(make_cosine_lr_callback(
            TRAIN.cosine_base_lr, TRAIN.cosine_alpha, total_epochs=TRAIN.epochs
        ))
        print("[INFO] Using Cosine LR:",
            f"base_lr={TRAIN.cosine_base_lr}, alpha={TRAIN.cosine_alpha}, epochs={TRAIN.epochs}")
    
    # 代替: 検証損失が改善しないときに段階的にLRを下げる
    elif TRAIN.use_lr_scheduler:
        cbs.append(tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=TRAIN.lr_reduce_factor,
            patience=TRAIN.lr_reduce_patience,
            min_lr=TRAIN.lr_min,
            verbose=0
        ))
        print("[INFO] Using LR scheduler: ReduceLROnPlateau",
            f"(factor={TRAIN.lr_reduce_factor}, patience={TRAIN.lr_reduce_patience}, min_lr={TRAIN.lr_min})")
    else:
        print("[INFO] Using LR scheduler: False")

    return cbs

# =========================
# Main pipeline
# =========================
def main():
    # ログ用
    run_summary = {
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "task": TASK,
        "data": {
            "win": DATA.win,
            "k_tau": _safe_float(DATA.k_tau),
            "top_p": _safe_float(DATA.top_p),
            "output_root": getattr(DATA, "output_root", None),
        },
        "train": {
            "epochs": int(TRAIN.epochs),
            "batch": int(TRAIN.batch),
            "lr": _safe_float(TRAIN.lr),
            "label_smoothing": bool(getattr(TRAIN, "use_label_smoothing", False)),
        },
        "folds": [],
    }

    """1回の実験（現行設定）を実行して、foldごとの評価とMDA・可視化を保存し、
    主要統計のサマリを返す（スイープ時はこの返り値で比較）。
    """
    # 1) 取得・特徴量
    run_start = time.time()

    price = fetch_ohlcv(DATA.ticker, DATA.start, DATA.end)
    if price is None or price.empty:
        raise RuntimeError(f"No price data for {DATA.ticker} in [{DATA.start},{DATA.end}]")
    df = make_features(price)         # OHLCV→派生特徴
    df = add_market_factors(df)       # TOPIX/N225/S&P/VIXなど
    df = add_rel_strength10(df)       # TOPIX控除の相対強度

    if DATA.use_log1p:
        df = add_log1p_features(df, DATA.log1p_candidates)      # 非負量のlog1p派生を追加

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

    # 2) ラベル
    if TASK == "trinary":
        y_all, abs_metric = make_trinary_labels(df, horizon=DATA.horizon, k_tau=DATA.k_tau)
    else:
        y_all, abs_metric = make_binary_excess_labels(
            df,
            horizon=DATA.horizon,
            k_tau=DATA.k_tau,
            use_k_tau_margin=True,
        )

    # 3) 学習に使う特徴カラムの列挙（存在チェック＋log1pの取り込み）
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

    # 4) 分割（PWFE）
    splitter = PWFESplit(n_splits=SPLIT.n_splits, embargo=SPLIT.embargo, min_train=SPLIT.min_train, min_val=SPLIT.min_val)
    splits = splitter.split(len(data))      # (train_idx, val_idx) のリスト
    # --- 分割サマリ（VALスパンと日付を簡易表示） ---
    print(f"[SPLIT] Effective folds: {len(splits)}")
    if splits:
        # VALの開始・終了インデックスと（可能なら）日付を表示
        va_spans = [(int(va[0]), int(va[-1])) for (_, va) in splits]
        print("[SPLIT] VAL spans (idx):", va_spans)
        try:
            first_dt = data.index[va_spans[0][0]]
            last_dt  = data.index[va_spans[-1][1]]
            print(f"[SPLIT] VAL date range: {first_dt.date()} .. {last_dt.date()}")
        except Exception:
            pass

    # ログ・蓄積バッファ
    fold_metrics = []
    subset_metrics = []
    all_true_main, all_pred_main = [], []
    all_true_sub,  all_pred_sub  = [], []
    best_es_params = None   # fold1でES最適化したら以後に流用
    figdir = f"{DATA.output_root}/{TASK}"
    _ensure_dir(figdir)
    metrics_log = []

    # ====== 各フォールドで学習・評価 ======
    for fold, (tr_idx, va_idx) in enumerate(splits, 1):
        fold_start = time.time()

        chosen_th = None

        # ① 学習データのみで前処理（リーク防止）
        #    さらに学習(tr_idx)を [tr_in | iv_in] に“時間順”で分割し、
        #    MI選択とStandardScalerのfitは tr_in のみで実施する。
        Xtr_df = pd.DataFrame(X_all[tr_idx], columns=feature_cols_final)
        ytr    = y_all[tr_idx]

        # --- 内側分割パラメータ（時間順で末尾を iv_in に） ---
        INNER_VAL_FRAC = 0.20
        MIN_INNER_VAL  = 64
        n_tr_total = len(Xtr_df)
        n_iv = max(int(n_tr_total * INNER_VAL_FRAC), MIN_INNER_VAL)
        n_iv = min(n_iv, n_tr_total // 2)  # innerが大きすぎるのを防ぐ
        cut  = n_tr_total - n_iv
        tr_in_idx = np.arange(0, cut)              # 内側学習（過去側）
        iv_in_idx = np.arange(cut, n_tr_total)     # 内側検証（直近側）

        # --- MI選択 & スケーラfit は tr_in のみ ---
        feats  = mi_select(Xtr_df.iloc[tr_in_idx], ytr[tr_in_idx], thr=1e-4)
        scaler = StandardScaler().fit(Xtr_df.iloc[tr_in_idx][feats].values)

        # 変換（tr全体→後で tr_in / iv_in に切り分け、outer は別に変換）
        Xtr_all = scaler.transform(Xtr_df[feats].values)
        Xva = scaler.transform(pd.DataFrame(X_all[va_idx], columns=feature_cols_final)[feats].values)

        # 内側の学習/検証へ切り分け（※インデックスは tr 内の相対）
        Xtr_in  = Xtr_all[tr_in_idx]
        ytr_in  = ytr[tr_in_idx]
        Xiv_in  = Xtr_all[iv_in_idx]
        yiv_in  = ytr[iv_in_idx]

        # ② 時系列→窓（Transformer入力形状へ）
        Xtr_inn_win, ytr_inn_win = to_windows(Xtr_in, ytr_in, DATA.win)
        Xiv_win,     yiv_win     = to_windows(Xiv_in, yiv_in, DATA.win)
        Xva_win,     yva_win     = to_windows(Xva,     y_all[va_idx], DATA.win)
        # === 入力形状ログ ===
        _log_btf("Xtr_inn_win", Xtr_inn_win)
        _log_btf("Xiv_win",     Xiv_win)
        _log_btf("Xva_win",     Xva_win)
        # 想定とズレていないか軽くチェック（ズレたら分かりやすくFail）
        assert Xtr_inn_win.shape[1] == DATA.win, f"win(T) mismatch: {Xtr_inn_win.shape[1]} vs DATA.win={DATA.win}"
        assert Xiv_win.shape[-1] == Xtr_inn_win.shape[-1] == Xva_win.shape[-1], \
            f"fdim mismatch: tr={Xtr_inn_win.shape[-1]}, iv={Xiv_win.shape[-1]}, va={Xva_win.shape[-1]}"
        print(f"[SHAPE] fdim(F)={Xtr_inn_win.shape[-1]}, win(T)={DATA.win}")

        abs_va_win = abs_all[va_idx][DATA.win:]         # サブセット抽出用（|r_H|上位p%）

        # 進捗ログ（スライドにも載せやすい）
        # --- 既存の [SPLIT] ログに追記する形で inner/outer を出力 ---
        # RAWインデックス範囲（絶対index）と日付レンジ
        abs_tr_in_idx = tr_idx[tr_in_idx]
        abs_iv_in_idx = tr_idx[iv_in_idx]
        dt = data.index
        # ポリシー（foldごとに再掲：スライド転記用）
        print(f"[SPLIT] Fold {fold} [INNER-POLICY] frac={INNER_VAL_FRAC:.2f}, min={MIN_INNER_VAL}, win={DATA.win}")
        # 範囲（idx）
        print(f"[SPLIT] Fold {fold} [INNER] TR_IN[{abs_tr_in_idx[0]}..{abs_tr_in_idx[-1]}] len={len(abs_tr_in_idx)} | "
            f"IV_IN[{abs_iv_in_idx[0]}..{abs_iv_in_idx[-1]}] len={len(abs_iv_in_idx)} | "
            f"OUTER[{va_idx[0]}..{va_idx[-1]}] len={len(va_idx)}")
        # 範囲（date）
        print(f"[SPLIT] Fold {fold} [DATES] inner={dt[abs_tr_in_idx[0]].date()}..{dt[abs_iv_in_idx[-1]].date()} | "
            f"outer={dt[va_idx[0]].date()}..{dt[va_idx[-1]].date()}")
        # 件数（RAW/ウィンドウ後）
        print(f"[SPLIT] Fold {fold} [COUNT][RAW] tr_in={len(tr_in_idx)} | iv_in={len(iv_in_idx)} | va_out={len(va_idx)}")
        print(f"[SPLIT] Fold {fold} [COUNT][WIN] tr_in={len(ytr_inn_win)} | iv_in={len(yiv_win)} | va_out={len(yva_win)}")

        # ③ モデルと不均衡対策
        n_classes = 3 if TASK == "trinary" else 2
        model = transformer_model(DATA.win, Xtr_inn_win.shape[-1], n_classes)
        class_weights = _compute_class_weights(ytr_inn_win)

        # ④ EarlyStoppingのチューニング（必要時は fold=1 だけで実施）
        if TRAIN.es_tune and (fold == 1):
            cand_scores = []

            for p in TRAIN.es_patience_grid:
                for d in TRAIN.es_mindelta_grid:
                    tf.random.set_seed(42); np.random.seed(42)
                    tmp_model = transformer_model(DATA.win, Xtr_inn_win.shape[-1], n_classes)
                    tmp_es = tf.keras.callbacks.EarlyStopping(
                        monitor = "val_loss", mode = "min",
                        patience = p, min_delta = d,
                        restore_best_weights = True,
                        verbose = 0 if LOGCFG.quiet_fit else 1
                    )
                    ytr_fit_es = _prepare_targets(ytr_inn_win, n_classes, TRAIN.use_label_smoothing)
                    yva_fit_es = _prepare_targets(yiv_win,     n_classes, TRAIN.use_label_smoothing)
                    tmp_hist = tmp_model.fit(
                        Xtr_inn_win, ytr_fit_es,
                        epochs=min(TRAIN.epochs, 120), batch_size=TRAIN.batch,
                        validation_data=(Xiv_win, yva_fit_es),
                        callbacks=[tmp_es], shuffle=False,
                        class_weight=class_weights,
                        verbose = 0 if LOGCFG.quiet_fit else 1
                    )
                    best_val_loss = np.min(tmp_hist.history["val_loss"])
                    cand_scores.append(((p, d), float(best_val_loss)))
            best_es_params, best_loss = min(cand_scores, key = lambda x: x[1])
            print(f"[ES-TUNE] selected patience={best_es_params[0]} "
                f"min_delta={best_es_params[1]} (best val_loss={best_loss:.5f})")

        patience = best_es_params[0] if best_es_params else TRAIN.es_patience_default
        min_delta = best_es_params[1] if best_es_params else TRAIN.es_mindelta_default
        lr_callbacks = _build_callbacks(TASK, patience, min_delta)

        # ⑤ 学習（label smoothing時はone-hotへ）
        ytr_inn_fit = _prepare_targets(ytr_inn_win, n_classes, TRAIN.use_label_smoothing)
        yiv_fit     = _prepare_targets(yiv_win,     n_classes, TRAIN.use_label_smoothing)

        fit_verbose = 0 if LOGCFG.quiet_fit else 1

        hist = model.fit(
            Xtr_inn_win, ytr_inn_fit,
            epochs=TRAIN.epochs, batch_size=TRAIN.batch,
            validation_data=(Xiv_win, yiv_fit),
            callbacks=lr_callbacks,
            shuffle=False,
            class_weight=class_weights,
            verbose=fit_verbose
        )

        # ⑥ 予測・スコア（outer = 従来の va を評価専用に使用）
        proba = model.predict(Xva_win, verbose=0)

        if TASK == "trinary":
            yhat = np.argmax(proba, axis=1)
            m = eval_trinary(yva_win, yhat)
        else:
            # 二値タスクの互換対応（(N,1) or (N,2)）
            if proba.ndim == 2 and proba.shape[1] == 1:
                yhat_p = proba.ravel()
            elif proba.ndim == 2 and proba.shape[1] == 2:
                yhat_p = proba[:, 1]
            else:
                raise ValueError(f"Unexpected proba shape for binary task: {proba.shape}")
            yhat = (yhat_p >= 0.5).astype(int)
            m = eval_binary(yva_win, yhat_p)
        
        accumulate_metrics(metrics_log, fold, m, chosen_th if TASK=="binary" else None)
        
        # ⑦ MDA（Permutation Importance, optional）
        try:
            imp = compute_mda_importance(
                model, Xva_win, yva_win,
                metric_name="macro_f1",
                feature_names=feats, n_repeats=TRAIN.mda_repeats, random_state=42
            )
            imp_csv = f"{figdir}/mda_fold{fold}.csv"
            imp.to_csv(imp_csv, index=False, encoding="utf-8")
            plot_importance(
                imp, save_path=f"{figdir}/mda_fold{fold}.png", topk=10,
                title=f"MDA (metric=Macro-F1) Fold{fold} [win={DATA.win}, k_tau={DATA.k_tau:.2f}]"
            )
            print(f"[MDA] saved: {imp_csv} & {figdir}/mda_fold{fold}.png (repeats={TRAIN.mda_repeats})")
        except Exception as e:
            print(f"[MDA][WARN] importance failed: {e}")

        # ⑧ 集計（全体・上位|r_H|サブセット）
        all_true_main.extend(yva_win.tolist())
        all_pred_main.extend(yhat.tolist())

        # 異常日サブセット（上位p%）
        k = max(1, int(len(abs_va_win) * DATA.top_p))       # 上位p%の閾
        idx_top = np.argsort(abs_va_win)[-k:]
        if TASK == "trinary":
            m_sub = eval_trinary(yva_win[idx_top], yhat[idx_top])
        else:
            m_sub = eval_binary(yva_win[idx_top], yhat_p[idx_top])

        all_true_sub.extend(yva_win[idx_top].tolist())
        all_pred_sub.extend(yhat[idx_top].tolist())

        fold_metrics.append(m)
        subset_metrics.append(m_sub)

        # ⑨ 学習曲線の保存
        figdir = f"{DATA.output_root}/{TASK}"
        _ensure_dir(figdir)
        _plot_history(hist, figdir, fold, TASK, extra_title=f"(win={DATA.win}, k_tau={DATA.k_tau:.2f})")

        # ログ出力（混同行列は視覚チェックに便利）
        print(f"[Fold {fold}] main={ {k:v for k,v in m.items() if k!='conf_mat'} }")
        print(f"[Fold {fold}] main confusion=\n{m['conf_mat']}")
        print(f"[Fold {fold}] subset(top{int(DATA.top_p*100)}%)={ {k:v for k,v in m_sub.items() if k!='conf_mat'} }")
        print(f"[Fold {fold}] subset confusion=\n{m_sub['conf_mat']}")
        print(f"[Fold {fold}] elapsed: {time.time()-fold_start:.2f}s")

        fold_rec = {
            "fold": int(fold),
            "counts_raw": {
                "tr_in": int(len(tr_in_idx)),
                "iv_in": int(len(iv_in_idx)),
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
    
    # ====== foldごとのメトリクスCSV/図の保存 ======
    save_and_plot_metrics(metrics_log, figdir, TASK)

    # === 全体（全フォールド合算）の混同行列 ===
    labels = [0, 1, 2] if TASK == "trinary" else [0, 1]
    cm_main = confusion_matrix(all_true_main, all_pred_main, labels=labels)
    cm_sub  = confusion_matrix(all_true_sub,  all_pred_sub,  labels=labels)

    print("\n=== Overall confusion matrix (main) ===")
    print(cm_main)
    print("\n=== Overall confusion matrix (subset top{:.0f}%) ===".format(DATA.top_p*100))
    print(cm_sub)

    # 行正規化版
    def _row_norm(cm):
        with np.errstate(invalid="ignore", divide="ignore"):
            r = cm.astype(float) / cm.sum(axis=1, keepdims=True)
            r[np.isnan(r)] = 0.0
            return r
    print("\n=== Overall confusion matrix (main, row-normalized) ===")
    print(_row_norm(cm_main))
    print("\n=== Overall confusion matrix (subset, row-normalized) ===")
    print(_row_norm(cm_sub))

    # --- Pooled（全outerを連結）での正式指標を算出＆ログ ---
    pooled_macro_f1_main = _score_trinary(all_true_main, all_pred_main, metric="macro_f1") if len(all_true_main) else np.nan
    pooled_bal_acc_main  = _score_trinary(all_true_main, all_pred_main, metric="balanced_acc")   if len(all_true_main) else np.nan
    pooled_macro_f1_sub  = _score_trinary(all_true_sub,  all_pred_sub,  metric="macro_f1") if len(all_true_sub)  else np.nan
    pooled_bal_acc_sub   = _score_trinary(all_true_sub,  all_pred_sub,  metric="balanced_acc")    if len(all_true_sub)  else np.nan
    print(f"\n[POOLED][MAIN] macro_f1={pooled_macro_f1_main:.6f}, balanced_acc={pooled_bal_acc_main:.6f}, N={len(all_true_main)}")
    print(f"[POOLED][SUB ] macro_f1={pooled_macro_f1_sub:.6f}, balanced_acc={pooled_bal_acc_sub:.6f}, N={len(all_true_sub)}")

    # --- Save figures for overall CM (counts & row-normalized) ---
    try:
        cls_labels = [0, 1, 2] if TASK == "trinary" else [0, 1]
        # main（全検証データ合算）
        _plot_cm(cm_main, cls_labels, f"{figdir}/cm_overall.png", normalize=False, title_prefix="Overall")
        _plot_cm(cm_main, cls_labels, f"{figdir}/cm_norm_overall.png", normalize=True,  title_prefix="Overall")
        # subset（トップp%などの重要部分）
        _plot_cm(cm_sub,  cls_labels, f"{figdir}/cm_overall_subset.png", normalize=False, title_prefix="Subset")
        _plot_cm(cm_sub,  cls_labels, f"{figdir}/cm_norm_overall_subset.png", normalize=True,  title_prefix="Subset")
        print(f"[PLOT] saved overall & subset CM figures to {figdir}")
    except Exception as e:
        print(f"[PLOT][WARN] overall CM plotting failed: {e}")

    # 5) 指標の集計（mean±sd）を作って返す（スイープ比較で使用）
    def agg(key, arr):
        vals = [d[key] for d in arr if key in d and not (isinstance(d[key], float) and np.isnan(d[key]))]
        return (float(np.mean(vals)), float(np.std(vals))) if len(vals) else (np.nan, np.nan)

    if TASK == "trinary":
        mu_bal, sd_bal = agg("balanced_acc", fold_metrics)
        mu_f1 , sd_f1  = agg("macro_f1", fold_metrics)
        mu_bal_s, sd_bal_s = agg("balanced_acc", subset_metrics)
        mu_f1_s , sd_f1_s  = agg("macro_f1", subset_metrics)
        summary_main = {
            "balanced_acc_mean": mu_bal, "balanced_acc_sd": sd_bal,
            "macro_f1_mean": mu_f1, "macro_f1_sd": sd_f1,
            # ★ pooled を正式指標として併記
            "pooled_macro_f1": float(pooled_macro_f1_main),
            "pooled_balanced_acc": float(pooled_bal_acc_main),
            "pooled_N": int(len(all_true_main)),
        }
        summary_sub = {
            "subset_top_p": DATA.top_p,
            "balanced_acc_mean": mu_bal_s, "balanced_acc_sd": sd_bal_s,
            "macro_f1_mean": mu_f1_s, "macro_f1_sd": sd_f1_s,
            "pooled_macro_f1": float(pooled_macro_f1_sub),
            "pooled_balanced_acc": float(pooled_bal_acc_sub),
            "pooled_N": int(len(all_true_sub)),
        }
        print(summary_main); print(summary_sub)
        res =  {
            "K_TAU": DATA.k_tau,
            # ★ 比較軸を fold平均 → pooled に切替（fallbackで平均値を使用）
            "metric_name": "pooled_macro_f1",
            "select_metric": float(pooled_macro_f1_main) if not np.isnan(pooled_macro_f1_main) else float(mu_f1),
            "summary_main": summary_main,
            "summary_subset": summary_sub
        }
        print(f"[RUN] total elapsed: {time.time()-run_start:.2f}s")

        finalize_run_summary(
            run_summary,
            pooled_main={"macro_f1": float(pooled_macro_f1_main), "balanced_acc": float(pooled_bal_acc_main), "N": int(len(all_true_main))},
            pooled_subset={"macro_f1": float(pooled_macro_f1_sub), "balanced_acc": float(pooled_bal_acc_sub), "N": int(len(all_true_sub))},
            logcfg=LOGCFG,
        )

        return res
    else:
        mu_bal, sd_bal = agg("balanced_acc", fold_metrics)
        mu_auc, sd_auc = agg("roc_auc", fold_metrics)
        mu_f1 , sd_f1  = agg("f1", fold_metrics)
        mu_bal_s, sd_bal_s = agg("balanced_acc", subset_metrics)
        mu_auc_s, sd_auc_s = agg("roc_auc", subset_metrics)
        mu_f1_s , sd_f1_s  = agg("f1", subset_metrics)
        summary_main = {"balanced_acc_mean":mu_bal, "balanced_acc_sd":sd_bal,
                        "roc_auc_mean":mu_auc, "roc_auc_sd":sd_auc, "f1_mean":mu_f1, "f1_sd":sd_f1}
        summary_sub = {"subset_top_p":DATA.top_p, "balanced_acc_mean":mu_bal_s, "balanced_acc_sd":sd_bal_s,
                        "roc_auc_mean":mu_auc_s, "roc_auc_sd":sd_auc_s, "f1_mean":mu_f1_s, "f1_sd":sd_f1_s}
        print(summary_main); print(summary_sub)
        res =  {
            "K_TAU": DATA.k_tau,
            "metric_name": "roc_auc_mean",
            "select_metric": float(mu_auc),
            "summary_main": summary_main,
            "summary_subset": summary_sub
        }
        print(f"[RUN] total elapsed: {time.time()-run_start:.2f}s")

        finalize_run_summary(
            run_summary,
            pooled_main={"macro_f1": float(pooled_macro_f1_main), "balanced_acc": float(pooled_bal_acc_main), "N": int(len(all_true_main))},
            pooled_subset={"macro_f1": float(pooled_macro_f1_sub), "balanced_acc": float(pooled_bal_acc_sub), "N": int(len(all_true_sub))},
            logcfg=LOGCFG,
        )

        return res

# ------ メトリクス蓄積・保存・可視化ユーティリティ ------
def _is_scalar(x):
    """Python数値 or 0次元ndarrayを“スカラ”とみなす判定。CSV化のときに利用。"""
    return isinstance(x, numbers.Number) and np.ndim(x) == 0

def accumulate_metrics(metrics_log: list, fold: int, metrics_dict: dict, chosen_th: Optional[float] = None):
    """foldごとの主要スカラ値だけを抽出し、CSV化しやすい辞書にして追加。"""
    row = {"fold": fold}
    if chosen_th is not None:
        row["threshold"] = float(chosen_th)
    for k, v in metrics_dict.items():
        if _is_scalar(v):
            row[k] = float(v)
    metrics_log.append(row)

def save_and_plot_metrics(metrics_log: list, figdir: str, task: str):
    """foldごとのスコアをCSV保存し、項目別の棒グラフと総括図（平均±SD）を作成。"""
    if not metrics_log:
        return
    df = pd.DataFrame(metrics_log).sort_values("fold")
    csv_path = f"{figdir}/metrics_all.csv"
    df.to_csv(csv_path, index=False, encoding="utf-8")

    # 指標ごとの fold 棒グラフ
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

    # まとめ図（平均±SDを1枚に）
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
        plt.ylabel("score"); plt.title("Final Run — Metrics Summary (mean ± sd across folds)")
        plt.tight_layout(); plt.savefig(f"{figdir}/metrics_summary.png", dpi=160); plt.close()

# ===== 実行エントリポイント（スイープ → 最終実行） =====
if __name__ == "__main__":
    set_seeds(42)
    did_sweep = False   #いずれかの探索を実行したかどうか

    #WIN値の最適値探索
    if DATA.win_sweep:
        results = []
        for w in DATA.win_grid:
            set_seeds(42)
            DATA.win = int(w)
            print("\n" + "="*70)
            print(f"Running with WIN = {w}")
            print("="*70)
            try:
                res = main()
                if res is not None:
                    res["WIN"] = int(w)
                    results.append(res)
            except Exception as e:
                print(f"[ERROR] WIN={w} run failed: {e}")
        
        if results:
            def _key(d):
                s = d.get("select_metric", np.nan)
                return -1e9 if (s is None or np.isnan(s)) else s
            best = max(results, key = _key)
            DATA.win = int(best["WIN"])
            DATA.best_win = int(best["WIN"])
            DATA.win = DATA.best_win

            print("\n" + "#"*70)
            print("# BEST RESULT ACROSS WIN")
            print(f"[WIN-SWEEP] Set WIN={DATA.win} for subsequent K_TAU run(s).")
            print("# Metric: {} | Score: {:.4f} | WIN = {}".format(
                best["metric_name"], best["select_metric"], best["WIN"]
            ))
            print("# Summary (main):", best["summary_main"])
            print("# Summary (subset):", best["summary_subset"])
            print("#" * 70)
            print(f"[WIN-SWEEP] Set WIN={DATA.win} for subsequent run(s).")

            did_sweep = True

    #K_TAU値の最適値探索
    if DATA.k_tau_sweep:
        results = []
        for k in DATA.k_tau_grid:
            # 乱数を毎回固定（学習のブレを最小化）
            set_seeds(42)

            DATA.k_tau = float(k)
            print("\n" + "="*70)
            print(f"Running with K_TAU = {k:.2f}")
            print("="*70)

            try:
                res = main()     # ★ mainの返り値（サマリ）を受け取る
                if res is not None:
                    results.append(res)
            except Exception as e:
                print(f"[ERROR] K_TAU={k:.2f} run failed: {e}")

        # ループ後に“最も良い結果”を1回だけ表示
        if results:
            # NaN を極小値扱いして除外的に評価
            def _key(d):
                s = d.get("select_metric", np.nan)
                return -1e9 if (s is None or np.isnan(s)) else s

            best = max(results, key=_key)
            DATA.k_tau = float(best["K_TAU"])
            print("\n" + "#"*70)
            print("# BEST RESULT ACROSS K_TAU")
            print("# Metric: {} | Score: {:.4f} | K_TAU = {:.2f}".format(
                best["metric_name"], best["select_metric"], best["K_TAU"]
            ))
            print("# Summary (main):", best["summary_main"])
            print("# Summary (subset):", best["summary_subset"])
            print("#" * 70)
            print(f"[KTAU-SWEEP] Set K_TAU={DATA.k_tau:.2f} for subsequent run(s).")

            did_sweep = True

    # COSINE_ALPHA の最適値探索（Cosine 使用時のみ）
    if TRAIN.use_cosine_lr and TRAIN.cosine_alpha_sweep:
        results = []
        for a in TRAIN.cosine_alpha_grid:
            set_seeds(42)
            TRAIN.cosine_alpha = float(a)
            print("\n" + "="*70)
            print(f"Running with COSINE_ALPHA = {TRAIN.cosine_alpha:.4f} "
                  f"(base_lr={TRAIN.cosine_base_lr}, WIN={DATA.win}, K_TAU={DATA.k_tau:.2f})")
            print("="*70)
            try:
                res = main()
                if res is not None:
                    res["COSINE_ALPHA"] = TRAIN.cosine_alpha
                    results.append(res)
            except Exception as e:
                print(f"[ERROR] COSINE_ALPHA={a:.4f} run failed: {e}")

        if results:
            def _key(d):
                s = d.get("select_metric", np.nan)
                return -1e9 if (s is None or np.isnan(s)) else s
            best = max(results, key=_key)

            # ベストαを確定（以後の実行に使用）
            TRAIN.cosine_alpha = float(best["COSINE_ALPHA"])
            print("\n" + "#"*70)
            print("# BEST RESULT ACROSS COSINE_ALPHA")
            print("# Metric: {} | Score: {:.4f} | COSINE_ALPHA = {:.4f}".format(
                best["metric_name"], best["select_metric"], best["COSINE_ALPHA"]
            ))
            print("# Summary (main):", best["summary_main"])
            print("# Summary (subset):", best["summary_subset"])
            print("#" * 70)
            print(f"[ALPHA-SWEEP] Set COSINE_ALPHA={TRAIN.cosine_alpha:.4f} for subsequent run(s).")

            did_sweep = True

    # --- 最終実行（スイープのベスト設定でラスト1回） ---
    set_seeds(42)
    print("\n" + "="*70)
    if did_sweep:
        print(f"Final run with BEST settings: WIN={DATA.win}, K_TAU={DATA.k_tau:.2f}, COSINE_ALPHA={TRAIN.cosine_alpha:.4f}")
    else:
        print(f"Final run (no sweeps): WIN={DATA.win}, K_TAU={DATA.k_tau:.2f}, COSINE_ALPHA={TRAIN.cosine_alpha:.4f}")
    print("="*70)
    _ = main()
