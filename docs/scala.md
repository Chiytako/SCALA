# SCALA — 設計仕様と測定

自己相似の階層型言語モデル。参考文献は arXiv:2512.20687(PHOTON)の階層分解のみで、
本書に述べる機構はいずれも同論文には存在しない。

## 1. 構成要素

トークンを `C=4` 個ずつ level-1 ユニットへ、level-l ユニットを `C` 個ずつ
level-(l+1) ユニットへ畳み込む L 段の階層。レベル l は次の4モジュールを持つ:

- **chunker**: `C` 個の下位ユニット → 1 個の level-l ユニット(線形)
- **encoder**: level-l ユニット列の文脈化。非トップはスライディング窓
  `encoder_window`(既定 16 ユニット)、トップ(CAP)のみ大域
- **converter**: 上位からの条件ベクトル 1 本 → `R` 本の条件位置(恒等初期化)
- **decoder**: 窓付きストリーム `decoder_stream`(既定 64 位置)。グループあたり
  `R + C` 位置で、読み出しは `[R−1 : R−1+C]`

level-l エンコーダは `C^l` トークンに 1 回しか走らないため、上位の容量は償却される。
条件付けカスケードは 1 ユニットのパイプライン(最上位のみ右シフト、下位は非シフト
消費)。損失にシフトはない: `logits[:, i]` がトークン `i` を予測する。

## 2. 3つの設計決定

### 2.1 自己相似 MID(重み共有)

モジュール実体は 3 組: `level_token`(レベル 1)、`level_mid`(レベル 2..L−1 の
すべてで共有)、`level_cap`(常に最上位)。`ScalaConfig.tie_mid_levels` の
検証(`config.py::_validate_tie`)が共有可能条件を強制する: 全 MID エントリ同一・
幅が level-1 と同一・スパン有界(level 1 自身の span-boundedness も検証対象)・
レベル別条件付け経路なし。

共有が安全である根拠(すべてテストで固定):

- `max_seq_len` 由来の `max_units` は RoPE テーブル初期サイズにしか使われず、
  テーブルはオンデマンド成長する
- 推論キャッシュはすべて生成器側の per-level 状態が所有し、モジュールに載らない
- 非トップの `start_latent` は読まれない

登録は `level_token.* / level_mid.* / level_cap.*` の3名前空間(`levels` は平
リストのビュー)。**state_dict のキーとパラメータ数が深さ L に依存しない。**

### 2.2 深さ=推論時の自由パラメータ

`scala_config_at_depth(cfg, k)` が保存済み設定を任意の MID 適用回数 k に再表現し、
同一チェックポイントを `strict=True` でロードできる。契約: どの k でも生成はその
k 自身の学習時 forward と厳密一致(2e-4、`tests/test_scala.py`)。学習した k と
別の k の forward が等しいことは要求できない(別の関数)。

### 2.3 有界トップ方針 = O(log T) 状態

`k(T) = max(k_cfg, 最小の k s.t. T ≤ U_max · C₁ · C_mid^k · C_cap)`

**これはセッション開始前に一度だけ深さを選ぶ静的な方針であり、生成中に動的へ
深くなる機構ではない**(`scala_depth_for_context` は `scala/infer/generate.py`
のどこからも呼ばれず、深さは `ScalaGenerator` の構築時に固定される)。この方針
に従って `T` に対する深さを事前に選べば、`T` とともに伸びるキャッシュは存在
しない。CAP のエンコーダキャッシュだけは設計上つねに無制限(`_window_units` が
トップに対して常に `None` を返す)で、想定より長く生成すると警告なく線形に
伸びる ── under-provisioning を検知するランタイムチェックは現状ない。解析値は
`accounting.scala_depth_for_context / scala_state_bytes`、実測は
`ScalaGenerator.cache_bytes()`(両者は `tests/test_scala.py::
test_scala_state_bytes_matches_a_real_generator_for_bounded_caches` で
窓/ストリーム部分を突き合わせて固定)。probe 幾何(C=4, U_max=32, bf16):

| 文脈 | 深さ k | L | 常駐状態/系列 |
|---:|---:|---:|---:|
| 8,192 | 2 | 4 | 425.5 KiB |
| 32,768 | 3 | 5 | 515.5 KiB |
| 131,072 | 4 | 6 | 605.5 KiB |
| 1,048,576 | 6 | 8 | 769.5 KiB |

位置づけ: Transformer は状態 O(T)・厳密検索、Mamba は O(1)・有損失。SCALA は
**O(log T)・厳密な多重解像度**(距離 `d` の情報は粒度 ~`C^⌈log_C d⌉` の実
エンコーダ状態として保持され、厳密な注意で読める)。

## 3. 厳密性と位置

- 生成(`hiergen`/`recgen`)は学習時 forward と厳密一致。ローリング窓キャッシュの
  規律は「書き込みインデックス ≠ RoPE オフセット」(roll は位相を保つ)
- 位置範囲を出るスタックが存在しない: 非トップはスパン有界 RoPE、CAP は NoPE
  (+学習 sink、重み吸収 latent キャッシュ、decoupled RoPE key なし)→
  最大文脈長という概念がない
- RoPE テーブルは 131,072 位置で打ち切り(幾何成長)、超過分は fp64 位相の
  per-call 計算(fp32 位相は pos ~1e6 で ~0.1 rad 量子化するため)。この
  fp32/fp64 の切り替えは各呼び出しの絶対位置のみで決まり、呼び出しをどう
  チャンク分割するかには依存しない(境界を跨ぐ呼び出しは低位置側をキャッシュ
  済みテーブルから、高位置側を fp64 から取って結合する、`layers.py::
  RotaryEmbedding.forward`)。以前はこの決定が「その呼び出し自身の
  `offset+seq_len`」で行われており、`TiledScorer` の小さいブロック呼び出しと
  1回勝負の参照 forward とで同じ絶対位置が異なる経路を通り得た
  ── `tests/test_scoring.py::
  test_rope_forward_is_granularity_invariant_across_the_table_cap` で固定

## 4. タイル化厳密スコアリング(131K〜1M+)

全長 forward は S×S マスクにより 131K トークンで破綻する(実測 25 GiB 要求)。
`scala/infer/scoring.py::TiledScorer` は同じ関数を 2 相で計算する:

1. **Phase A**: 各窓付きエンコーダを生成器と同じローリングキャッシュ規律で
   大ブロックストリーミング。CAP は latent へタイル追記(1M で数 MB)。各レベルの
   末尾リングのみ保持
2. **Phase B**: デコーダは受容野 `n(w−1)` 位置ぶんの warm-up を先頭に付けた
   末尾セグメントのみ再生(T 非依存)。上位ストリームが warm-up より短い深い
   実体化では全ストリーム密復号にフォールバック(その場合 CAP はシフト経路)

学習時 forward との一致(2e-4)を全深さ・全タイル割りで `tests/test_scoring.py`
が固定。1M トークンの参照 forward は物理的に存在しないため、1M での厳密性は
テスト固定カーネルからの継承である(公表時にその旨を明記する)。

## 5. 測定

### 65M probe(30M トークン、held-out ja/en、Δ は同一データ 2 シード幅比)

| 検証軸 | 結果 |
|---|---|
| 厳密性 | recgen KL 1e-4、cos(X_top) 1.0000(k=2 および未学習の k=3) |
| 深さ転移(L=5/L=6) | CE 劣化 ≤ +0.0023 nats(シード幅 0.063 の 1/27) |
| 重み共有の代償 | 同シード tied vs untied: ±0.0002(untied は +4.8M params) |
| 長さ 131K→1M | CE 4桁不動(深さ 2/3/4/6、日英)。採点 4.5 s/系列・0.6 GiB |
| 検索 @1M | d=131K/262K/524K で +0.44/+0.94/+0.82 nats(粒度 256〜65,536 tok/unit で不変) |
| 固定グリッド代償 | ±0.003(ゼロ) |

### 0.4B(422.3M、330M トークン、8 ソース混合、GB10)

| 検証軸 | 結果 |
|---|---|
| 厳密性 | KL 0.000・greedy 一致 100%(k=2/k=3) |
| 深さ転移 | ≤ +0.0071 nats(全 8 ソース) |
| 長さ | 512→16K 最大 +0.0031、131K→1M 4桁不動(11 s/系列・2.6 GiB) |
| 検索 @1M | d=524K で **+2.28 nats**(65M の +0.82 から強化) |

### 未検証

- 検索は合成針(能力)であり、遠方文脈に需要のあるコーパスでの効果は未測定
- 学習文脈は最大 4096(1M の性質はすべてゼロショットの構造汎化)
- tie × FSDP / tie × ブロック compile(trainer にガードあり)。`parallel: fsdp`
  が既定でこのコードベースは単一GPUでも bf16 のために FSDP2 を使うため、
  デフォルト設定の単一GPU学習でも即座にこのガードに当たる ── tied
  チェックポイントを FSDP 学習経路で大規模(8B級)に学習した実績はまだない
- 深さ+1 で生成レイテンシ約 2 倍(launch-bound、未最適化。方針上 16K 超文脈のみ)
- 学習は k=2 のみ(k>2 でのネイティブ学習は未実施。深さ>2 の結果はすべて
  `scala_config_at_depth` によるゼロショット再表現)
- vs Celeritas L3(同一学習形状での比較): 未解決。ペア平均 CE 差がスライス間で
  符号反転する(ja +0.056 / en −0.013、2026-07-31 時点)。効果とは呼んでいない。
  Celeritas 自体は tying なしの旧世代であり SCALA の核心機構(自己相似・
  推論時深さ・O(log T))には無関係だが、記録として残す

## 6. 主要ファイル

| | |
|---|---|
| `scala/model/scala.py` | `scala_config` / `scala_config_at_depth` |
| `scala/model/config.py` | `tie_mid_levels` と共有可能条件の検証 |
| `scala/model/hierarchy.py` | 本体(構築ループでの MID 共有登録) |
| `scala/infer/scoring.py` | タイル化厳密スコアラ |
| `scripts/longctx_probe.py` | 1M 長さ/検索プローブ |
| `configs/scala_probe_k2.yaml` | 65M probe(プリセットとテストで同一性固定) |
| `configs/scala_04b.yaml` | 0.4B(GB10)── **未マージ**: このリポジトリにも git 履歴にも存在しない。GB10 側で実行・トンネル切断時に記録した設定で、§5 の 0.4B 数値はこのファイルではなく `docs/history/findings.md` §22 の記述に基づく。再現するにはまず GB10 側の設定一式をこのリポジトリへ移す必要がある |
