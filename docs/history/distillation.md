# Distillation in SCALA v3

Why there are two teachers, why neither of them is the strongest model
available, and what the measurements were.

---

## 1. The constraint that decides everything

Logit-level (white-box) distillation matches the student's output distribution
to the teacher's, token by token. That requires the two models to *agree on what
a token is*. Different tokenizer, no correspondence between the two logit
vectors, no loss to compute.

So "pick the best teacher" is not a free choice. It is:

> pick the best teacher **among models whose tokenizer we are willing to adopt**,
> because adopting it is what the student's vocabulary has to be.

And the student's vocabulary is not a free parameter either — it sets the size
of the embedding and the LM head, which at 196,608 entries is 402.7M parameters,
a third of this model's active budget, and it sets how many tokens a given
Japanese document costs to train on.

## 2. Tokenizer measurements

Characters per token, higher is better, measured 2026-07-28 on the same three
samples (`scratchpad/teacher_probe.py`):

| tokenizer | vocab | JA prose | JA technical | EN | code |
|---|---:|---:|---:|---:|---:|
| **llm-jp v4** | 196,608 | **2.150** | **2.302** | 6.148 | 2.463 |
| llm-jp v3 | 99,574 | 2.098 | 2.033 | 5.724 | 2.418 |
| Qwen 3.5 (Ornith-1.0) | 248,077 | 1.870 | 1.937 | 6.148 | 2.830 |
| Qwen 3 | 151,669 | 1.458 | 1.402 | 6.148 | 3.023 |
| poolside Laguna S 2.1 | 100,352 | 1.089 | 1.008 | 6.148 | 2.830 |

Laguna's tokenizer needs **2.3x more tokens for the same Japanese text** than
llm-jp v4. It is an excellent coding model and its tokenizer is built for that.
Adopting it to unlock logit KD would have more than doubled the compute cost of
every Japanese token in the run — on a budget that was already the binding
constraint. Ornith's Qwen-3.5 vocabulary is 15% worse than llm-jp v4 on Japanese
*and* 26% larger, so its LM head would cost more for less.

**Correction (measured after the 8B run).** The table above uses three
hand-written sentences and does not generalise. On real Japanese Wikipedia
llm-jp v4 measures 1.7915 chars/token against v3's 1.8320 -- 2.2% *worse*,
not better. What survives is the ordering against the non-Japanese
tokenizers, which is large enough to hold: Laguna still needs roughly twice
the tokens for Japanese. The reason to adopt v4 is that logit KD needs the
teacher's exact vocabulary and that v4 carries the harmony control tokens --
not compression. See findings.md 11b.

So the conclusion is not "llm-jp-4 is the best model", and it is not "v4 is the
best tokenizer" either. It is that the white-box teacher has to be llm-jp-4
because it is the only strong model whose vocabulary is worth adopting, and the
frontier models earn their place through **sequence-level** distillation
instead, where the tokenizer is irrelevant because you are training on text.

## 3. The two teachers

### White-box: `llm-jp/llm-jp-4-32b-a3b-base`

- Same tokenizer as the student, so top-K logit KD is well-defined.
- 32B total / ~3B active (Qwen3-MoE, 128 experts, 8 active). Cheaper to run
  *forward* than the dense 8B sibling — ~6 GFLOP/token against ~16 — and
  stronger.
- 64 GB in bf16, which fits GB10's 121 GiB unified memory, so teacher logits can
  be precomputed at zero marginal cost instead of on rented GPUs.

### Black-box (sequence-level): the frontier models

Training on text a model *emitted* has no tokenizer constraint at all, because
it is just text. That is what the mid-training mixture is: transcripts from

| corpus | teacher |
|---|---|
| `nvidia/Nemotron-SFT-Agentic-v2` | DeepSeek-V3.2 |
| `zake7749/deepseek-v4-pro-agent-tool-calling-trajectory` | DeepSeek-V4-pro |
| `greghavens/kimi-k3-coding-and-debugging-traces` | Kimi K3 (reasoning_effort max) |
| `mgoin/Laguna-S-2.1-trajectories` | poolside Laguna S 2.1 |
| `llm-jp/llm-jp-4-thinking-sft-data` | llm-jp-4's own SFT corpus |

Kimi K3 is 2.8T parameters with 16 of 896 experts active. There is no version of
this project that self-hosts it. Its *output* is free.

## 4. The Japanese-degradation problem, and what was done about it

Measured Japanese character fraction over the first 30 records of each agentic
corpus:

```
nvidia/Nemotron-SFT-Agentic-v2             0.0%
greghavens/kimi-k3-coding-...              0.0%
zake7749/deepseek-v4-pro-agent-...         0.0%
mgoin/Laguna-S-2.1-trajectories            0.0%
tokyotech-llm/Swallow-Nemotron-PT-v1       0.0%
```

Every frontier agentic dump is English — including the one from a Japanese lab.
Mid-training on "the best available agentic data" would have traded away the
thing this model is for, and that is not hypothetical: the 250M vast.ai run used
a 32% math cooldown and produced Japanese samples visibly contaminated with
formulae and English.

Four countermeasures, in `configs/data_ja_v3_mid.yaml`:

1. **Japanese reasoning data that already exists in the right format.**
   `llm-jp/llm-jp-4-thinking-sft-data` is the corpus llm-jp-4 was tuned on: 17
   subsets x {reasoning_low, medium, high}, ~2.9M conversations, already
   harmony-formatted. The Japanese subsets carry the reasoning and
   instruction-following behaviour *without* switching language. 0.35 of the
   mixture.
2. **Japanese pretraining replay at 0.30.** Standard anti-forgetting practice
   and the cheapest insurance available.
3. **English agentic capped at 0.18.** It is in the mixture for the shape of an
   agent turn — tool schemas, calls, results, recovery — which transfers across
   languages far better than register does.
4. **The white-box teacher pulls toward Japanese on every token**, including the
   English agentic ones. A mixture ratio is a static budget; KD is a force
   applied continuously. This is the reason the teacher was chosen on Japanese
   tokenizer fit rather than on benchmark scores.

The number to watch is `ppl/wiki_ja` across the mid-training phase. If it rises
while `ppl/wiki_en` falls, countermeasures 1–3 are underweighted and the English
agentic block comes down before anything else is touched.

## 5. Status

Sequence-level distillation is in the run: it is the mid-training mixture, and
it needs no extra machinery beyond `scala/data/chat.py`, which normalises
five different publisher formats into the tokenizer's harmony template.

Top-K logit KD from llm-jp-4-32b-a3b is the enhancement, not the critical path.
It requires precomputing teacher logits on GB10 and moving them to the training
host, and the run is not blocked on it — if the precompute does not land in
time, the mid-training phase still trains on the same data with plain
cross-entropy.
