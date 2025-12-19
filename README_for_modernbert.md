# ModernBERT (DeepSpeed + Megatron)

この文書は、DeepSpeed / Megatron 環境で定義された ModernBERT の設計方針とトレーニング設定（拡張前フェーズ）をまとめたものです。

## 1. アーキテクチャ改善（概要）

- バイアスの取り扱い
  - 全ての Linear 層: bias を無効化
  - LayerNorm: bias を無効化
  - Decoder（最終線形層）: bias は残す

- 位置表現
  - RoPE（Rotary Positional Embedding）を使用
  - 現段階では標準的な θ を利用（後続の更新とは区別）

- 正規化構造
  - Pre-Norm Transformer（Attention / MLP の前に LayerNorm）
  - Embedding の直後にも LayerNorm を追加（初期 LayerNorm との重複を排除）

- 活性化関数
  - GeGLU（Gated Linear Units）を採用

### 実装チェックリスト (アーキテクチャ)
- 全 Linear バイアスの取り扱い — 実装箇所: 
    ``` shell
        group.add_argument('--disable-bias-linear', action='store_false',
                       help='Disable bias in the linear layers',
                       dest='add_bias_linear')
    ```
- [ ] 全 LayerNorm バイアスの取り扱い — 実装箇所: _______________________
- [ ] 全 Decoder バイアスの取り扱い — 実装箇所: _______________________
- [ ] RoPE の導入 (位置表現) — 実装箇所: _______________________
- [ ] Pre-Norm Transformer 構造 — 実装箇所:--apply-residual-connection-post-layernorm を 指定しない（False）
- [ ] Embedding 後の LayerNorm 追加 — 実装箇所:
    ``` shell
        group.add_argument('--layernorm-embedding', action='store_true',
                       help='If set, use layernorm on the input embeddings. '
                       'This is useful for training BERT-like models.')
    ```
- [ ] GeGLU の採用 (MLP) — 実装箇所: _______________________

## 2. 注意機構と効率化

- Alternating Attention（交互注意）
  - 層ごとに Global Attention と Local Attention を交互に切替
  - Local Attention: スライディングウィンドウ（window = 128）
  - Global Attention: 全トークン間の自己注意

- Unpadding（パディング除去）
  - Embedding 前に入力のパディングを除去して計算を効率化
  - Flash Attention と組み合わせて jagged attention を実現
  - 出力側で必要に応じて再パディング（re-padding）を行う

- Flash Attention の使い分け
  - Global attention → Flash Attention v3
  - Local attention → Flash Attention v2

- PyTorch コンパイル
  - `torch.compile` を使うことで約 10% のスループット改善を確認

### 実装チェックリスト (注意機構 / 効率化)
- [ ] Alternating Attention (Global / Local の切替) — 実装箇所: _______________________
- [ ] Local Attention (window=128, スライディングウィンドウ) — 実装箇所: _______________________
- [ ] Unpadding (入力のパディング削除) — 実装箇所: _______________________
- [ ] jagged attention / Flash Attention 統合 — 実装箇所: _______________________
- [ ] Flash Attention v2 / v3 の割当 (Local/Global) — 実装箇所: _______________________
- `torch.compile` 最適化の適用場所 — 実装箇所: 実装していない

## 3. モデル設計（base モデル）

- アーキテクチャ: Deep & Narrow
  - 層数（num_layers）: 22
  - 隠れ次元（hidden_size）: 768
  - MLP（GeGLU）拡張（intermediate_size）: 2304

- ハードウェア最適化
  - Tensor Core のタイル効率を意識した hidden / expansion の設計

### 実装チェックリスト (モデル設計)
- 層数・隠れ次元・MLP 拡張の設定 (config / CLI / code) — 実装箇所: 実行commandで指定
- Tensor Core 最適化パラメータの反映 (ブロックサイズ / データ配置) — 実装箇所: 実装していない
## 4. トークナイザ

- BPE ベースのトークナイザ（OLMo 派生）を使用（WordPiece ではない）
- 語彙サイズ: 50,368（GPU でのバッチ処理最適化のため 64 の倍数に調整）
- BERT 互換の特殊トークン（CLS/SEP/PAD 等）を利用

### 実装チェックリスト (トークナイザ)
- BPE トークナイザ定義 (学習 / ロード) — 実装箇所: 実装していない 各実行commandで指定 vocab_pathに使用するvocabを作成する段階でBPEにすればよい  
- 語彙サイズ（50,368）の設定箇所 — 実装箇所: 各実行commandで指定 vocab_pathに使用するvocabのファイルを設定すればよい     
- 特殊トークンのマッピング (CLS/SEP/PAD 等) — 実装箇所: 勝手に実装される    

## 5. トレーニング設定（拡張前フェーズ）

- Pretraining データ
  - 規模: 約 2 兆トークン（Web、コード、学術文献などの混合コーパス）

- Sequence Packing
  - Greedy packing を用い高効率にシーケンスを詰める（≈99% のパディング効率を目標）

- マスクド言語モデリング（MLM）
  - MLM のみを使用（NSP は行わない）
  - マスク率: 約 30%（BERT の 15% と比べ高め）

- 最適化
  - Optimizer: StableAdamW（AdamW と Adafactor スタイルの安定化/クリップを組み合わせた手法）
  - 学習率スケジュール: WSD（Warmup → Stable → Decay）
    - Warmup → 一定区間（Stable）→ 線形減衰（Decay）

- バッチサイズスケジューリング
  - トレーニング初期は小さいバッチサイズから始め、段階的に増加

### 実装チェックリスト (トレーニング設定)
- [ ] データセット読み込み / Pretraining コーパス設定 — 実装箇所: _______________________
- [ ] Sequence Packing 実装 (Greedy packing) — 実装箇所: _______________________
- NSP を行わない設定 — 実装箇所: 各実行commandで指定  
    ``` shell
    group.add_argument('--bert-no-binary-head', action='store_false',
                       help='Disable BERT binary head.',
                       dest='bert_binary_head')
    ```
- MLM マスク率 30% の実装箇所 — 実装箇所: 各実行commandで指定  
    ``` shell
        group.add_argument('--mask-prob', type=float, default=0.15,
                        help='Probability of replacing a token with mask.')
    ```
- Optimizer: StableAdamW の実装/設定箇所 — 実装箇所: megatron/optimizer/__init__.py  
--optimizer stable_adamw を指定すれば使える  
  
- 学習率スケジュール WSD の実装箇所 — 実装箇所: 実装していない      
    実行commandでconstatを指定することで、decay前までを実現。その後、そのcheckpointを指定して、commandで --finetuneとdecayを指定することで再現できる。
- バッチサイズスケジューリングのロジック — 実装箇所: 各実行commandで指定  
    ``` shell
        group.add_argument('--rampup-batch-size', nargs='*', default=None,
                       help='Batch size ramp up with the following values:'
                       '  --rampup-batch-size <start batch size> '
                       '                      <batch size incerement> '
                       '                      <ramp-up samples> '
                       'For example:'
                       '   --rampup-batch-size 16 8 300000 \ '
                       '   --global-batch-size 1024'
                       'will start with global batch size 16 and over '
                       ' (1024 - 16) / 8 = 126 intervals will increase'
                       'the batch size linearly to 1024. In each interval'
                       'we will use approximately 300000 / 126 = 2380 samples.')
    ```

## 6. 初期化

- モデル初期化: Megatron の初期化処理を利用
- base モデルはランダム初期化で開始

### 実装チェックリスト (初期化)
- [ ] Megatron 初期化の呼び出し箇所 (例: `initialize_megatron`) — 実装箇所: _______________________
- [ ] ランダム初期化が適用されるコード箇所 — 実装箇所: _______________________

## 7. 注釈 / 注意点

- 上記は拡張前（base）段階の設計と実装方針です。拡張や最適化を進めるにあたり、RoPE の細かなパラメータや Flash Attention のバージョン選定、トークナイザの語彙最適化などは再評価され得ます。
- 実装・実験時はハードウェア（GPU 世代、Tensor Core の仕様）に合わせてパラメータ調整を行ってください。

### 実装チェックリスト (その他 / 運用)
- [ ] RoPE のパラメータ（θ など）の設定箇所 — 実装箇所: _______________________
- [ ] Flash Attention のバージョン選定／依存関係管理 — 実装箇所: _______________________
- [ ] 実験ログと学習曲線の出力場所 (ログ形式 / 保存先) — 実装箇所: _______________________
- [ ] 設定ファイル（YAML / JSON / CLI 引数）の場所 — 実装箇所: _______________________
