# Stock Price Prediction (Graduation Research) — 4 Models (Trinary Classification)

このリポジトリは、**株価の方向性を三値分類（trinary classification）**する卒業研究用の実験コードです。  
次の4つのモデルを同一のデータパイプラインで比較します。

- **Transformer**（時系列ウィンドウ入力）
- **LSTM**（時系列ウィンドウ入力）
- **Logistic Regression**（ウィンドウを集約して2D特徴にする）
- **LightGBM**（ウィンドウを集約して2D特徴にする）

> ✅ 重要：このリポジトリは「動作・結果を変えずに」構造だけ整理したものです。  
> そのため **実行入口（旧パス互換）** と、整理後の **本体（src/）** の2層構造になっています。

---

## 目次

- [1. まず何をすればいい？（最短で動かす）](#1-まず何をすればいい最短で動かす)
- [2. 4つのモデルを実行する](#2-4つのモデルを実行する)
- [3. 実行すると何が出力される？（結果の見方）](#3-実行すると何が出力される結果の見方)
- [4. フォルダ構成とファイルの役割](#4-フォルダ構成とファイルの役割)
- [5. 銘柄・期間・窓幅などを変えたい（設定変更の場所）](#5-銘柄期間窓幅などを変えたい設定変更の場所)
- [6. ニュース特徴量の追加（GDELT）](#6-ニュース特徴量の追加gdelt)
- [7. テスト（回帰テスト）を走らせる](#7-テスト回帰テストを走らせる)
- [8. 旧コードと結果が同じか比較したい](#8-旧コードと結果が同じか比較したい)
- [9. 再現性（seed / TFの注意）](#9-再現性seed--tfの注意)
- [10. よくあるつまずき（FAQ）](#10-よくあるつまずきfaq)

---

## 1. まず何をすればいい？（最短で動かす）

### 1) リポジトリ直下へ移動

> **必ず**「このREADMEがある階層（リポジトリ直下）」で実行してください。  
> 相対パスで出力する設計のため、別ディレクトリから実行すると出力先がズレます。

```bash
cd stockpred_refactored
```

### 2) 仮想環境を作って依存関係を入れる

#### macOS / Linux
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

#### Windows（PowerShell）
```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

> TensorFlow のインストールが環境依存で失敗する場合があります。  
> その場合は [FAQ](#10-よくあるつまずきfaq) を見てください。

---

## 2. 4つのモデルを実行する

このリポジトリでは、**実行用の入口（旧パス互換）**が用意されています。  
初心者はまずここを実行してください（これが一番安全で迷いません）。

### Transformer
```bash
python transformer/trinary_transformer_old.py
```

### LSTM
```bash
python LSTM/trinary_LSTM.py
```

### Logistic Regression
```bash
python Logistic_Regression/trinary_logistic_regression.py
```

### LightGBM
```bash
python LightGBM/trinary_LightGBM.py
```

---

## 3. 実行すると何が出力される？（結果の見方）

各モデルを実行すると、基本的に「そのモデルのフォルダ配下」に成果物が出ます。

例（Transformerの場合）：

- `transformer/run_summary.json`  
  実行結果の要約（fold平均のスコア、設定値など）が入ります。  
  **まずはこれを見るのが最短**です。

- `transformer/figs/...`  
  学習曲線、混同行列、指標の図、Permutation Importance（MDA）の出力などが入ります。

他のモデルも同様に、例えば：

- `LSTM/run_summary.json`, `LSTM/figs/...`
- `Logistic_Regression/run_summary.json`, `Logistic_Regression/figs/...`
- `LightGBM/run_summary.json`, `LightGBM/figs/...`

> ✅ “出力先パス”は、元コードと同じ相対パスになるように設計されています。

---

## 4. フォルダ構成とファイルの役割

このリポジトリは **2層構造**です。

### A) 実行入口（旧パス互換）

以下のフォルダは「入口」です。  
中身は薄いラッパーで、**本体（src/）の実験モジュールをそのまま起動**します。

- `transformer/trinary_transformer_old.py`
- `LSTM/trinary_LSTM.py`
- `Logistic_Regression/trinary_logistic_regression.py`
- `LightGBM/trinary_LightGBM.py`

> ✅ 共同開発や比較実験で「昔の実行手順がそのまま動く」ことを優先しています。

---

### B) 本体コード（整理後の正本）

`src/stock_pred/` が「本体」です。今後のレビューや拡張はここを見ます。

- `src/stock_pred/common_shared.py`  
  4モデル共通の処理（例：seed固定、評価、分割、図保存、run_summary保存など）

- `src/stock_pred/dataset_pipeline.py`  
  4モデル共通のデータ処理（例：株価取得、特徴量、ラベル作成、窓化/2D化など）

- `src/stock_pred/news_pipeline.py`  
  ニュース特徴量の生成（取引日整列、記事数/センチメント集約、キャッシュ）

- `src/stock_pred/models/`  
  実験スクリプト本体（4モデル分）
  - `transformer_trinary_transformer_old.py`
  - `lstm_trinary_LSTM.py`
  - `logistic_regression_trinary_logistic_regression.py`
  - `lightgbm_trinary_LightGBM.py`

---

### C) 互換 shim（importを壊さないための橋渡し）

ルート直下にある以下は「互換用」です。旧コードの import を壊しません。

- `common_shared.py` → `src/stock_pred/common_shared.py` を再export
- `dataset_pipeline.py` → `src/stock_pred/dataset_pipeline.py` を再export

---

### D) テスト / ツール

- `tests/`  
  回帰テスト（前処理や分割などの “壊れていないか” を確認する軽量テスト）

- `tools/compare_run_summaries.py`  
  `run_summary.json` 同士を比較して、差分を見やすくする補助スクリプト

- `tools/fetch_news_newsapi_ai.py`  
  GDELT DOC API からニュースを収集して `data/news/raw/{ticker}.csv` に保存するツール

- `data/news/`  
  ニュースデータの保存先（raw / features）

---

## 5. 銘柄・期間・窓幅などを変えたい（設定変更の場所）

このプロジェクトは、基本的に **コマンド引数ではなく「スクリプト内の設定（dataclass）」を編集**して実験します。

編集場所（例：Transformer）  
- `src/stock_pred/models/transformer_trinary_transformer_old.py`

各モデルのファイル内にはだいたい以下の設定クラスがあります：

- `DataConfig`：データ関連（ticker、期間、horizon、win、k_tau、pooling、出力先など）
- `SplitConfig`：分割（PWFE、embargoなど）
- `TrainConfig`：学習関連（epoch、batch、early stopping等がある場合）
- `ModelConfig`：モデル構造/ハイパーパラメータ（各モデル固有）

例（よく触る項目のイメージ）：
- `ticker`：銘柄（例 `"7203.T"`）
- `start`, `end`：期間（例 `"2001-01-01"`, `"2024-12-31"`）
- `horizon`：何日先を予測するか
- `win`：ウィンドウ長（時系列入力の長さ）
- `k_tau`：ラベル境界の係数（分類の閾値に関係）
- `output_root`：図の出力先（モデルごとに既存パス維持のため固定されがち）
- `use_news`：ニュース特徴量を使うか（追加した機能）

> ✅ 初心者向けおすすめ手順  
> 1) まずデフォルトのまま実行して動作確認  
> 2) 次に `ticker` と `start/end` だけ変えて再実行  
> 3) 最後に `win` や `k_tau` を触る

---

## 6. ニュース特徴量の追加（GDELT）

ニュース記事を特徴量として **early fusion** で株価特徴に結合できます。  
主な追加コードは `src/stock_pred/news_pipeline.py` です。

### 6.1 ニュースデータの形式

`data/news/raw/{ticker}.csv` に保存されたニュースを読み込みます。  
最低限必要な列は以下です（これだけあれば動きます）：

- `published_at`：公開日時（文字列でOK。後で日時に変換）
- `title`：記事タイトル

あれば使われる列：
- `body`：本文（タイトルと結合して簡易センチメント計算に使用）
- `source`：ソース名（ユニーク数を特徴量に使用）
- `ticker`：銘柄（無い場合はファイル名の銘柄として扱う）
- `url`：記事URL（重複排除の一意キーとして使用）
- `tone`：GDELTのtone（実数）。日次集約して特徴量に使用
- `lang`：言語（空でもOK）

### 6.2 取引日への整列（リーク対策）

`published_at` を **取引日単位**に揃えます。  
具体的には：

- 引け後の記事は **翌取引日** に割り当て
- 非取引日の記事は **次の取引日** に丸め

これにより、未来情報の混入（リーク）を抑えています。  
`news_market_close_time` と `news_tz` で市場時間を指定できます。

### 6.3 生成される特徴量（A/B + APIセンチメント）

**A: メタ系（記事数・量）**
- `news_count_0d`, `news_source_nunique_0d`, `news_after_close_ratio_0d`
- `news_count_3bd_sum`, `news_count_5bd_sum`
- `news_count_surprise_20bd`, `news_no_news_flag`

**B: 辞書ベースの簡易センチメント**
- `news_sent_score_mean_0d`, `news_sent_score_sum_0d`
- `news_pos_hits_0d`, `news_neg_hits_0d`, `news_pos_ratio_0d`
- `news_sent_score_3bd_mean`, `news_sent_score_5bd_mean`

**GDELT tone（tone）**
- `news_tone_mean_0d`, `news_tone_sum_0d`
- `news_tone_min_0d`, `news_tone_max_0d`
- `news_tone_valid_count_0d`, `news_tone_valid_ratio_0d`
- `news_tone_pos_ratio_0d`, `news_tone_neg_ratio_0d`, `news_tone_abs_mean_0d`
- `news_tone_3bd_mean`, `news_tone_5bd_mean`

> 辞書ベースの単語リストは `src/stock_pred/news_pipeline.py` の  
> `DEFAULT_POS_WORDS / DEFAULT_NEG_WORDS` を編集して変更できます。

### 6.4 使い方（設定）

各モデルの `DataConfig` に以下の設定を追加しています：

- `use_news`：ニュース特徴量を追加するか（デフォルトは False）
- `news_path`：`data/news/raw/{ticker}.csv` のテンプレート
- `news_cache_dir`：`data/news/features`（日次特徴量のparquetキャッシュ）
- `news_tz` / `news_market_close_time`：タイムゾーンと引け時刻
- `news_use_meta` / `news_use_sent`：A/B の切替
- `news_rolling_windows`, `news_long_window`：rolling と驚き度の窓幅

ニュース特徴量は `data/news/features/` に parquet でキャッシュされます  
（同名 `.meta.json` に生成条件も保存）。

### 6.5 ニュース収集ツール（GDELT）
`tools/fetch_news_gdelt.py` で GDELT DOC API から取得できます。  
事前に API キーを環境変数へ設定してください。

```bash
pip install eventregistry pandas python-dateutil python-dotenv
```

```bash
$env:NEWSAPI_AI_KEY="YOUR_KEY"   # PowerShell
# export NEWSAPI_AI_KEY="YOUR_KEY"  # macOS / Linux
```

`.env` に `NEWSAPI_AI_KEY=...` を書く方法でもOKです（`python-dotenv` で読み込み）。

```bash
python tools/fetch_news_newsapi_ai.py `
  --tickers 7203.T `
  --query-map data/news/query_map.json `
  --days 30 `
  --lang jpn eng `
  --max-items 500
```

- 取得結果は `data/news/raw/{ticker}.csv` に保存されます。  
- `data/news/query_map.json` で銘柄ごとのキーワード/概念を指定できます。  
- Free plan は過去約1か月の制限があるため、長期実験は別手段が必要です。

---

## 7. テスト（回帰テスト）を走らせる

テストは「学習をフルで回す」ものではなく、主に

- 前処理の出力形状が変わっていないか
- 分割が壊れていないか
- seed固定関数が呼べるか

などを軽く検証します。

```bash
pip install -r requirements-dev.txt
pytest -q
```

---

## 8. 旧コードと結果が同じか比較したい

旧コードで出した `run_summary.json` と、新コードで出した `run_summary.json` を比較できます。

```bash
python tools/compare_run_summaries.py path/to/old/run_summary.json transformer/run_summary.json
```

> TensorFlow/GPU を使う場合、完全一致が難しい環境もあります。  
> 詳細は [再現性の注意](#9-再現性seed--tfの注意) を参照してください。

---

## 9. 再現性（seed / TFの注意）

- 共通ユーティリティで seed 固定を行います（元コードに合わせるため）
- ただし **深層学習（TensorFlow）＋GPU** の場合、CUDA/cuDNNやカーネル実装により  
  **完全一致が崩れることがあります**（一般に起こり得ます）

何が保証できて、何が環境依存かは：
- `tests/test_reproducibility_notes.py`

にまとめています。

---

## 10. よくあるつまずき（FAQ）

### Q1. `pip install -r requirements.txt` で TensorFlow が入らない
A. OS / Python バージョン / CPU/GPU に依存します。  
まずは以下を確認してください：

- 使っている Python のバージョン（例：3.10 など）
- Apple Silicon / NVIDIA GPU など、環境差

対処の方向性：
- CPUのみで動かす（TensorFlow CPU版）
- PythonバージョンをTensorFlowが対応しているものに合わせる

---

### Q2. 実行したのに `run_summary.json` が見つからない
A. 「リポジトリ直下」で実行しているか確認してください。

✅ 正しい例：
```bash
cd stockpred_refactored
python transformer/trinary_transformer_old.py
```

---

### Q3. どのファイルを読めばロジックが分かる？
A. 基本はここです：

- 前処理：`src/stock_pred/dataset_pipeline.py`
- 共通処理：`src/stock_pred/common_shared.py`
- 各モデル本体：`src/stock_pred/models/*.py`

入口の `transformer/` や `LSTM/` の `.py` は「起動するだけ」なので、ロジックはほぼありません。

---

## ライセンス / 免責
- データ取得は外部サービス（例：株価データ）に依存します。取得結果は時点により変動する可能性があります。
- 研究目的のコードであり、投資助言を目的としません。
