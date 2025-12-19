# Inhouse Megatron-DeepSpeed

本リポジトリは DeepSpeed と Megatron-LM を統合した大規模言語モデルの事前学習・微調整のための実装です。ベースは公式の Megatron-DeepSpeed をフォーク/再構成したもので、一部に社内向けの拡張（ModernBERT 等）を含みます。

- 参照元: https://github.com/deepspeedai/Megatron-DeepSpeed
- ModernBERT の補足: `README_for_modernbert.md`
- 公式 DeepSpeed 例: `examples_deepspeed/README.md`

## 対応範囲（概要）
- GPT/BERT/T5/LLAMA 系の事前学習スクリプト（`pretrain_*.py`）
- DeepSpeed ZeRO、混合精度 (FP16/BF16)、テンソル並列/パイプライン並列
- 例とレシピ: `examples/`, `examples_deepspeed/`
- Optimizer/分散オプティマイザ: `docs/distrib_optimizer.md`
- ModernBERT の各種実装と切替（初期化、GeGLU、bias 無効化、ローカル/グローバル Attention 等）

## 前提環境
- Python と PyTorch（CUDA 対応の GPU マシン推奨）
- CUDA/NCCL を利用する分散学習は Linux もしくは WSL2 を推奨（Windows ネイティブ環境では分散通信に制約があります）
- DeepSpeed（`pip install deepspeed`）

目安（例）：
- Python: 3.8 以上
- CUDA: 11.x 以上（GPU/ドライバに依存）
- PyTorch: CUDA 対応版（バージョンは CUDA と合わせる）

> 注意: 本 README は最小限のセットアップ手順を示します。クラスタ/大規模分散の実運用は各環境ポリシーに合わせて調整してください。

## セットアップ（Windows PowerShell 例）
開発/検証用にローカル環境へインストールします。分散/大規模実行は Linux/WSL2 での実行を推奨します。

```powershell
# 仮想環境の作成（任意）
python -m venv .venv
. .\.venv\Scripts\Activate.ps1

# 必要パッケージのインストール（PyTorch は環境に合うホイールを選択）
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install deepspeed

# 本リポジトリを開発モードでインストール
pip install -e .
```

WSL2 を使用する場合は、WSL 側で上記コマンドをそのまま実行してください。

## クイックスタート

### 1) 単一 GPU でヘルプを確認（フラグ一覧の把握）
```powershell
python pretrain_gpt.py --help
python pretrain_bert.py --help
```

### 2) GPT/BERΤ の小規模テスト実行（例）
実データ準備がまだの場合は、まず実行経路や依存解決の確認を目的に、ごく小さな学習設定で動作チェックします（実データの与え方は次章）。

```powershell
# 例: GPT を極小ステップで動作確認（パラメータは環境に合わせて調整）
python pretrain_gpt.py `
	--num-layers 2 `
	--hidden-size 512 `
	--num-attention-heads 8 `
	--seq-length 512 `
	--max-position-embeddings 512 `
	--micro-batch-size 2 `
	--global-batch-size 8 `
	--train-iters 10 `
	--tokenizer-type GPT2BPETokenizer `
	--vocab-file path/to/gpt2-vocab.json `
	--merge-file path/to/gpt2-merges.txt `
	--data-path path/to/bin_dataset_prefix `
	--fp16
```

```powershell
# 例: BERT/ModernBERT 系の最小実行（MLM のみ）
python pretrain_bert.py `
	--num-layers 2 `
	--hidden-size 512 `
	--num-attention-heads 8 `
	--seq-length 512 `
	--max-position-embeddings 512 `
	--micro-batch-size 2 `
	--global-batch-size 8 `
	--train-iters 10 `
	--tokenizer-type BertWordPieceLowerCase `
	--vocab-file path/to/bert-vocab.txt `
	--data-path path/to/bin_dataset_prefix `
	--bert-no-binary-head `
	--fp16
```

> フラグはスクリプトにより異なります。`--help` で確認し、環境やデータに合わせて変更してください。

### 3) DeepSpeed での実行（Linux/WSL2 推奨）
`examples_deepspeed/` に DeepSpeed 統合済みの動作例があります。まずはレシピを確認してください。

- 参照: `examples_deepspeed/README.md`
- 例: `examples_deepspeed/pretrain_llama_distributed.sh`

Linux/WSL2 のシェルで（要: 正しい `deepspeed` インストールと NCCL 設定）:
```bash
deepspeed pretrain_gpt.py \
	--deepspeed \
	--deepspeed_config ds_config.json \
	--tensor-model-parallel-size 2 \
	--pipeline-model-parallel-size 2 \
	...
```

> Windows ネイティブ PowerShell での分散実行は非推奨です。WSL2 もしくは Linux サーバ上で実行してください。

## データ準備
Megatron 系スクリプトは一般に「事前にトークナイズ/バイナリ化したデータ」を `--data-path` で受け取ります。

- 典型的には（例）`--tokenizer-type GPT2BPETokenizer` と `--vocab-file`, `--merge-file`（GPT2/BPE）
- BERT 系は WordPiece 語彙（`--tokenizer-type BertWordPieceLowerCase` など）
- 入力フォーマットや事前処理の詳細は各スクリプトの `--help` と公式 Megatron-DeepSpeed のドキュメントを参照

内部で一般的に重要となるパラメータ:
- `--seq-length`, `--max-position-embeddings`: シーケンス長
- `--micro-batch-size`, `--global-batch-size`: バッチ関連
- 並列化: `--tensor-model-parallel-size`, `--pipeline-model-parallel-size`
- AMP: `--fp16` または `--bf16`

## 主要スクリプトとディレクトリ
- トップレベル学習スクリプト: `pretrain_gpt.py`, `pretrain_bert.py`, `pretrain_t5.py`, `continued_pretrain_llama.py`, ほか
- 追加資料: `docs/distrib_optimizer.md`
- 例（NVIDIA/Megatron-LM 由来）: `examples/`（DeepSpeed 未統合のものあり）
- DeepSpeed 統合例: `examples_deepspeed/`（推奨）
- ModernBERT 詳細: `README_for_modernbert.md`

## ModernBERT 拡張
ModernBERT 固有の初期化、GEGLU、bias 無効化、ローカル/グローバル Attention の切替などの説明は `README_for_modernbert.md` を参照してください。

## トラブルシュート
- CUDA/NCCL の初期化で停止/エラー: 分散実行は Linux/WSL2 を使用し、環境変数（`NCCL_DEBUG=INFO` 等）で原因を確認
- OOM（メモリ不足）: `--micro-batch-size` を縮小、`--tensor-model-parallel-size` を増やす、`--fp16`/`--bf16` を利用
- 依存関係エラー: `pip install -e .` を再実行、`pip list` で競合を確認
- DeepSpeed 実行時の設定: `--deepspeed_config` のパラメータ（ゼロステージ、オフロード、勾配チェックポイント等）を見直し

## FAQ
- Windows だけで分散学習できますか？
	- 推奨しません。WSL2 もしくは Linux 環境をご利用ください。
- どの例から始めればよいですか？
	- `examples_deepspeed/README.md` 掲載のレシピが最新かつ実践的です。
- ModernBERT のオプションはどこで確認？
	- `README_for_modernbert.md` と各スクリプトの `--help` を参照してください。

## ライセンス / 出典
- ベース実装は DeepSpeed/Megatron-LM（出典リンクは冒頭参照）
- LICENSE はこのリポジトリ同梱の `LICENSE` を参照