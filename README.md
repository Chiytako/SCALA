# SCALA

自己相似の階層型言語モデル。**深さは推論時の自由パラメータ、常駐状態は O(log T)、
生成は学習時 forward と厳密一致。** 1つのチェックポイントが任意の深さで実体化でき、
学習長 4096 のまま 1M トークン文脈で動作する。

> **SCALA** — a self-similar hierarchical LM: depth is an inference-time free
> parameter, resident state is O(log T), generation is exact against the
> training forward. Measured at 1M-token context.

## 参考文献

SCALA は論文 [arXiv:2512.20687](https://arxiv.org/abs/2512.20687)(PHOTON)の
階層分解 ── トークン列をチャンク単位で畳み込み、上位レベルほど償却して走らせる ──
を出発点として**参考にした、独立した新しいアーキテクチャ**です。論文著者・組織とは
無関係で、公式の重み・データ・コードは使用していません。SCALA の中核機構
(自己相似の重み共有、推論時の深さ、O(log T) 状態、タイル化厳密スコアリング)は
同論文には存在しません。研究用の実験成果物であり、製品ではありません。

## 設計

**階層。** トークンを `C=4` 個ずつ level-1 ユニットに、level-l ユニットを 4 個ずつ
level-(l+1) ユニットに畳み込む。各レベルは chunker(圧縮)/ encoder(文脈化)/
converter(条件展開)/ decoder(復元)を持ち、level-l のエンコーダは `C^l`
トークンに1回しか走らないため、上位の容量は償却される。

**モジュールは3組だけ。**

| 組 | 役割 | 適用 |
|---|---|---|
| `level_token` | トークン統計 | レベル 1 |
| `level_mid` | 1組の共有重み | レベル 2..L−1 のすべて |
| `level_cap` | 大域トップ(MLA + NoPE + 学習 sink) | 常に最上位 |

中間レベルが1組の重みを共有するため、**パラメータ数と state_dict のキーが深さ L に
依存しない**。1つのチェックポイントを `scala_config_at_depth(cfg, k)` で任意の深さに
再表現し、`load_state_dict(strict=True)` でそのままロードできる。

**深さ=推論時の自由パラメータ。** k=2 で学習したチェックポイントを L=5, L=6
(学習していない関数)として実体化しても、同一 held-out トークンの CE 劣化は
シード間ばらつきの 1/10 以下。全診断ツールに `--depth` が通る。

**有界トップ方針 → O(log T) 状態。** 「最上位が `U_max` ユニットを超えたら1段
深くする」のは**セッション開始前に一度だけ深さを選ぶ静的な方針**であり、生成中に
自動で深くなる仕組みではない(`scala_depth_for_context` は `generate.py` のどこ
からも呼ばれない)。この方針の下で深さを選べば、`T` とともに伸びるキャッシュは
存在しない: 各レベルはスライディング窓のエンコーダ(`encoder_window`)と窓付き
ストリームのデコーダ(`decoder_stream`)を1本ずつ持ち、実体化レベル数は
`log_C T` で増える。文脈 8K で常駐 425 KiB、1M でも 770 KiB
(この数値は解析式 `scala_state_bytes` によるもの。CAP のキャッシュだけは
意図的に無制限で、想定より長く生成すると警告なく線形に伸びる ── 詳細は
[`docs/scala.md`](docs/scala.md) の「未検証」節)。

**最大文脈長という概念がない。** 位置範囲を出るスタックが存在しない: 非トップは
すべてスパン有界(RoPE は学習済み相対範囲のみを評価)、トップは NoPE。

**厳密性。** 生成(`recgen` / `hiergen`)は学習時 forward とどの深さでも厳密一致
(KL ~1e-4 未満、テストで 2e-4 固定)。近似・代入・蒸留的補正は存在しない。

**タイル化厳密スコアリング。** 全長 forward は 131K トークンで S×S マスクにより
破綻する。`scala/infer/scoring.py` は同じ関数を「窓ストリーミング+受容野ぶんの
warm-up を持つ末尾セグメント復号」の2相で計算し、1M トークンを数 GiB・数秒で
厳密採点する。RoPE テーブルは 131,072 位置で打ち切り、超過分は fp64 位相で
per-call 計算する ── この fp32/fp64 切り替えは各呼び出し自身の絶対位置のみで
決まり(呼び出しの粒度に依存しない、`layers.py` の `RotaryEmbedding.forward`)、
タイル化スコアラの小さいブロック呼び出しと1回勝負の参照 forward とで同じ絶対
位置が異なる経路を通ることはない。

**位置づけ。**

| | 生成時の状態 | 遠方の取り出し |
|---|---|---|
| Transformer | O(T) | 厳密・全位置 |
| Mamba / SSM | O(1) | 有損失(固定状態に圧縮) |
| **SCALA** | **O(log T)** | **厳密・多重解像度(近くは細かく、遠くは指数的に粗く)** |

Mamba は時間を固定状態に畳み込む(忘れる)。SCALA は厳密な多重解像度ピラミッドに
ぼかす ── 距離 `d` の情報は粒度 ~`C^⌈log_C d⌉` の実エンコーダ状態として保持され、
厳密な注意で読める。ここで言う「厳密」は学習時 forward 自身の計算に対しての厳密
一致であり、任意の生の詳細を無損失に復元できるという意味ではない ── 距離 `d` の
情報は上記の粒度に不可逆に粗視化されている。

**新規性についての注記。** O(log T) 状態そのものは、効率的注意機構の文献
(Log-Linear Attention, HOMER など)に別名で既に存在する ── 空いている設計点では
ない。SCALA 固有と言えるのは、トークンの階層構造をまたいで重みを実際に共有し
(自己相似)、かつ深さを推論時の自由パラメータにできる、という組み合わせであり、
この2つを同時に満たす先行研究は調査時点で見つかっていない。

## 実測(設計性質の裏づけ)

65M probe(30M トークン)および 0.4B(330M トークン、8ソース混合)で:

- ゼロショット深さ転移: CE 劣化 ≤ +0.0023(65M)/ **+0.0071**(0.4B、全8ソース)nats
  (未学習の L=5/L=6)
- 重み共有の代償: 同シード比較で ±0.0002(untied 統制は +4.8M パラメータ)
- 長さ: 131K → 1M トークンで CE が4桁不動(学習長の 256 倍、日英)
- 検索: 1M 文脈・52万トークン先の 16 トークンの針に +0.82(65M)/ **+2.28**(0.4B)nats
- 固定グリッド(チャンク境界)の代償: 測定限界以下(1M でも ±0.003)

詳細・測定条件・未検証事項は [`docs/scala.md`](docs/scala.md)。

## 使い方

```bash
pip install torch numpy pyyaml   # torch >= 2.8
python -m pytest tests -q        # 90 tests

# モデルは1行で
python - <<'EOF'
from scala.model.scala import scala_config
from scala.model.hierarchy import ScalaForCausalLM
model = ScalaForCausalLM(scala_config(depth=2))   # 65M probe 幾何
EOF

# 学習(probe: RTX 5060 Ti 1枚で ~30分)
python scripts/train.py --config configs/train_probe_scala_k2.yaml

# 深さを変えて評価 / 1M トークン採点
python scripts/protocol_diag.py --ckpt runs/probe-scala-k2/final --depth 3
python scripts/longctx_probe.py --ckpt runs/probe-scala-k2/final --depth 2 \
    --mode length --lengths 131072,1048576 --win 256
```

## リポジトリ構成

```
scala/
  model/config.py     設定 dataclass と検証(tie_mid_levels ほか)
  model/hierarchy.py  本体(chunker / encoder / converter / decoder / 損失)
  model/scala.py      プリセット scala_config / scala_config_at_depth
  model/layers.py     RMSNorm, RoPE/NoPE, GQA/MLA(+sink), KVCache
  model/accounting.py パラメータ・FLOPs・状態の解析会計(深さ方針込み)
  infer/generate.py   厳密生成(hiergen/recgen)、ローリング窓キャッシュ
  infer/scoring.py    タイル化厳密スコアラ(131K〜1M+)
  train/              Muon+AdamW / WSD / FSDP2 学習ループ
  data/               memmap シャードローダ
configs/              モデル・学習・データ設定
scripts/              学習・生成・診断(--depth 対応)・1M プローブ
tests/                90 件(厳密性・因果性・有界性・深さ不変性)
docs/scala.md         設計仕様と全測定
```
