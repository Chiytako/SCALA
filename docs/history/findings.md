# Measured findings

> A standalone Japanese summary of the whole arc -- lineage, every
> measurement, the retraction list, and the Transformer/Mamba
> positioning -- is in [`celeritas_report.ja.md`](celeritas_report.ja.md).

Everything here was measured on this code base, not taken from a paper. Where a
published claim and a measurement disagreed, the measurement is recorded along
with why the claim did not transfer.

Hardware referenced: **GB10** (DGX Spark class, sm_121, 122 GiB unified),
**H100 SXM** (sm_90, 80 GiB), **RTX 5060 Ti** (sm_120, 16 GiB).

---

## 1. The MoE dispatch is the throughput bottleneck, not attention

Predicted 89K tok/s for the 8B config on an H100 from the FLOP count. Measured
**17.1K tok/s** — about 5% MFU.

The cause is expert GEMM shape. PHOTON's level-2 encoder sees `T/16` units, so
at micro-batch 4 × 4096 tokens it routes only ~1024 units. With 256 experts and
top-4 that is **16 tokens per expert**: every expert GEMM has M≈16, which is
far too small to use a tensor core well, whichever dispatch you pick.

**This is a PHOTON-specific interaction.** In a flat transformer every layer
sees all `T` tokens, so 256 experts is fine. In PHOTON the upper levels are
starved by construction, and expert count should *shrink* going up the
hierarchy — the opposite of where you would put capacity if you only looked at
parameter counts. That trade is not yet reflected in the configs here.

### Corollary: the MoE is on the wrong levels

The same starvation shows up as a *balance* failure, and it is the clearest
architectural finding of the whole exercise. Matched runs, same code, same
controller:

| model | experts in the top stack | tokens/expert/step | MaxVio at step 400 |
|---|---|---|---|
| small (247M) | 64 | ~180 | **0.88**, flat |
| 8B | 256 | ~16 | **30.3**, still climbing |

With 16 tokens per expert the per-step load estimate is almost pure noise, so no
bias controller — sign or PID — can steer it. Both were tried: the sign rule
reached 32.8 by step 320, PID 23.4 at the same point. PID is ~30% better and
neither solves it, because the problem is not the controller.

**The design implication is the opposite of the obvious one.** Parameter
accounting says to put capacity in the top stack, because it is amortised 16x.
Routing says to put *experts* where the tokens are, which is the bottom. A
PHOTON MoE should therefore be fine-grained at level 1 (which sees every token)
and coarse — few large experts, or dense — at the top, rather than the
256-expert top stack used here.

That change did not make it into the 8B run: it needs a re-tune of the whole
parameter budget, and the run was already 15% in when the trend became clear.
It was tested afterwards at 250M scale — see §1b, which is where the prediction
above turns out to be only half right.

### 1b. The re-placement, tested: the prediction was backwards

Four matched 250M runs, same data, same seed, same schedule, same step count
(1,525 steps / 0.2B tokens), differing only in the model config. Perplexity is
`scripts/eval_ja.py` on Wikipedia articles **past the ones training consumed**
(see §1c — the first version of this table was measured on training data):

| run | total | active | ja-wiki PPL | en-wiki PPL | tok/s |
|---|---|---|---|---|---|
| v1 | 247.23M | 100.62M | 60.32 | 65.91 | 11,430 |
| v3 | 250.29M | 100.54M | 57.12 (−5.3%) | 62.17 (−5.7%) | 11,011 |
| **v4** | **251.88M** | **99.77M** | **52.11 (−13.6%)** | **59.52 (−9.7%)** | 11,029 |

Training loss agrees: over the last 100 steps v3 is −0.060 nats against v1 and
v4 is −0.085, on identical batches.

v3 changed three things at once (a budget decision, not a clean ablation):
expert count reordered to follow token flow, full attention restored on the
level-1 encoder, and the attention output gate re-parameterised to `2*sigmoid`
so it starts at 1.0 instead of 0.5.

**The MaxVio trace says the first of those three did not work the way §1
predicted.** v3's level-2 encoder was cut from 64 experts to 16, each 4x wider,
on the theory that fewer experts means a steadier per-step load estimate:

| step | v3 MaxVio (mean / max) | v1 MaxVio (mean / max) |
|---|---|---|
| 100 | 1.35 / 2.34 | 5.16 / 9.34 |
| 460 | 1.67 / 5.01 | 1.84 / 7.17 |
| 550 | 3.87 / **14.06** | 1.73 / 4.89 |
| 820 | 6.80 / **15.00** | 2.41 / 11.76 |
| 1520 | 6.64 / **14.98** | 0.76 / 1.76 |

**15.00 is the theoretical maximum for E=16** — every token in the layer routed
to one expert. All nine routed layers of the level-2 encoder collapsed onto a
single expert by step ~550 and never recovered; the layer-mean of ~6.6 is what
you get when half the model's MoE layers are pinned at maximum violation and
the other half sit near 0.5. Fewer experts did not make the top stack
steerable, it made it collapse *faster* (v1 at E=64 stayed under 12).

So the honest reading of v3's win: the level-2 encoder's 117M MoE parameters —
**47% of the whole model** — were doing the work of ~13M, and the model was
still better, because the gain came from the level-1 stacks and the two
attention fixes. The top-level router is not underperforming, it is inert.

**v4 tests that sharper hypothesis and confirms it.** Delete the top-level
router outright (dense FFN at width 768, matched to v3's *active* top-1 +
shared), and spend the freed ~106M on more experts in the level-1 encoder and
decoder (24→96 and 32→96), where MaxVio stayed between 0.5 and 2 all run. Same
total, same active, same FLOPs — the capacity just moves to the stacks that can
route it. The result is the best model of the four on both languages, and:

| | v1 (E=64 top) | v3 (E=16 top) | v4 (dense top) |
|---|---|---|---|
| final MaxVio, layer mean | 0.76 | 6.64 | **0.50** |
| final MaxVio, worst layer | 1.76 | **14.98** (= max) | **1.28** |

Note what this does *not* say. v4 does not have more capacity than v3, or fewer
starved layers by accident — it has **zero** routed layers above the 4-token
level, and 96-way routing below it, and it is both better and better-balanced.
The rule that falls out is simple: **route where the tokens are, and nowhere
else.** A PHOTON level that sees `T/16` units per batch should be dense.

The corollary for the 8B config (`base_8b_a1b_v2.yaml`, 256 experts on the
level-2 encoder, MaxVio 30+) is that its largest MoE stack is almost certainly
inert in the same way. That config has not been retrained — the vast.ai budget
was spent — but it should not be reused as written.

### 1c. The perplexity numbers in this file were wrong once

`scripts/eval_ja.py` streamed `wikimedia/wikipedia` **from record 0**, and
`scripts/prepare_data.py` streams the same dataset from record 0 to build the
training shards. So the "held-out Wikipedia perplexity" reported for v1, v2, v3
and the 8B model was measured on **training data**.

It is worth recording how the error surfaced, because it did not look like a
bug. It looked like a result:

| | train CE (last 100) | ja-wiki, from record 0 | ja-wiki, past training |
|---|---|---|---|
| v3 | 4.2454 | **45.71** | 57.12 |
| v4 | 4.2204 | 55.09 | **52.11** |

v4 had the better training loss on identical batches, and the better in-training
eval (43.81 vs 49.23 at step 1400), but the "independent" eval put it 17% worse
on Japanese. Two measurements of the same quantity disagreeing in *sign* is not
noise — increasing the sample eightfold (60 → 500 sequences) reproduced it
exactly. The only way both can be true is if the "independent" set is not
independent, and v3 memorised more of it.

The fix is `--ppl-skip-ja` / `--ppl-skip-en`, defaulting past what these runs
consumed (700K / 400K articles against 190.5M / 64.4M tokens tokenised). With
that, the eval ranks the models the same way the training loss does.

**The lesson is not "check your eval set".** It is that a disagreement between
two metrics that should agree is information, and the cheapest thing to do with
it is to distrust the metric you have not audited — not the one that is easy to
explain away.

### 1d. …and Wikipedia perplexity still cannot compare runs with different data

Even fixed, the Wikipedia number only ranks models that trained on the same
mixture. Two 250M models, same architecture family, evaluated identically on
307K tokens each:

| | 青空文庫 (neither trained on) | ja-wiki | en-wiki |
|---|---|---|---|
| v4 — Wikipedia only, 200M tokens | 365.49 | **51.61** | **59.29** |
| v3-900m — full mixture, 900M tokens | **237.10** | 67.60 | 69.37 |

Each model wins on its own domain, by a lot, in opposite directions. On text
neither has seen, the broad-mixture model trained on 4.5x more tokens is **35%
better** — which is the answer you would expect and the opposite of what the
Wikipedia column says.

So the architecture comparisons in §1b (v1/v3/v4, all Wikipedia-only, identical
batches) are valid, and any cross-mixture claim needs a third corpus. `eval_ja.py
--ppl-extra <hf-repo>[:text_key]` exists for that.

### Dispatch paths, measured

| path | works on | notes |
|---|---|---|
| `torch._grouped_mm`, ragged offsets | **nowhere** | device-side assert |
| `torch._grouped_mm`, 16-aligned offsets | sm_90 only | bit-exact, +43% padding at E=256 |
| padded batched GEMM (`bmm`) | everywhere | 17.1K tok/s on H100 |
| per-expert Python loop | everywhere | ~30x slower; a host sync per projection |

Three separate traps here:

1. **`torch._grouped_mm` requires offsets that are multiples of 16.** Real MoE
   routing produces arbitrary counts. Passing ragged offsets triggers a
   *device-side assert*, not a Python exception — which poisons the CUDA
   context, so `try/except` cannot recover. It has to be gated before the call.
2. **The capability gate is `== (9, 0)`, not `>= (9, 0)`.** Blackwell (sm_100,
   sm_120) and GB10 (sm_121) all raise. A `>=` check lets newer hardware
   through and the failure appears mid-forward, hours in.
3. **A dtype mismatch silently disables it.** Autocast rewrites `nn.Linear` but
   not raw-parameter matmuls, and the residual stream stays fp32 (the embedding
   output is fp32; bf16 + fp32 promotes). Without an explicit cast the kernel
   refused every call and fell through to the loop — **~25x slower end to end**,
   with no error.

---

## 2. Cross-entropy over a 100K vocabulary dominates everything else

Component timings, 8B-class model, 16,384 tokens, GB10:

| component | ms | share |
|---|---|---|
| encode (both levels) | 41.8 | 1.4% |
| decode (both levels) | 156.8 | 5.3% |
| **LM loss** | **349.9** | **11.7%** |
| **MTP loss** | **360.4** | **12.1%** |
| forward total | 893.3 | 30% |
| forward + backward | 2982.3 | 100% |

The LM head is ~78% of the model's FLOPs at this width. Three fixes, each
measured:

1. **Chunked cross-entropy.** Materialising `(B, T, V)` fp32 logits costs
   3.3 GiB at B=8, T=1024, V=99584 — more than the rest of the model. Slicing
   the sequence and recomputing each slice's logits in backward bounds it at
   `chunk × V`. Peak fell 13.6 → 7.0 GiB.
2. **Derive log Z from the cross-entropy instead of a second reduction.**
   `log Z = CE_i + logit_at_target`, so the z-loss costs one gather rather than
   a second full pass over a `(chunk, 100K)` tensor. **350 ms → 175 ms.**
3. **`torch.compile` the loss slice.** Inductor fuses the LM-head GEMM with
   log-softmax and never writes the fp32 logits. **175 ms → 53 ms**, loss
   bit-identical (11.6144 both ways). Total: **6.6x on the dominant component.**

Chunk size barely matters (9.1–9.6 TFLOPS across 673–16384 rows) — this loss is
memory-bandwidth-bound, not GEMM-shape-bound.

---

## 3. Weight tying is actively harmful in PHOTON

At d_token=2048 with a tied head, initial CE was **29.06** against a uniform
11.51, with logit std 0.91 but **max|logit| 37.25**.

The mechanism is PHOTON-specific. The level-1 decoder's residual stream is
seeded directly with `X^(0)_{i-1}` — the *previous token's* embedding. A tied
head therefore gives that exact token an enormous logit, and the model starts as
a confident copier. The effect scales with `sqrt(d_token)`, which is why a
d=512 model hid it (CE started correctly at 11.54).

Untying: CE **11.94**, max|logit| **5.81**.

---

## 4. bf16 master weights need stochastic rounding

An 8B model only fits on one 80 GB card with bf16 masters:

| configuration | memory | fits? |
|---|---|---|
| fp32 params + fp32 optimiser state | 93.2 GiB | no |
| FSDP2 MixedPrecisionPolicy + fp32 state | 62.9 GiB | no |
| bf16 masters + bf16 state | **46.6 GiB** | yes |

Two things to know:

* **FSDP2's `MixedPrecisionPolicy` does not reduce master-weight memory.** It
  keeps fp32 masters and *adds* a bf16 working copy, so on a single rank it
  makes memory worse. Only `model.to(bfloat16)` actually halves it.
* **bf16 masters silently stall without stochastic rounding.** 200 updates of
  1e-4 applied to a weight of 1.0: round-to-nearest gives **1.000000** (every
  update discarded); stochastic rounding gives **1.019947** against an exact
  1.020000.

bf16 Muon momentum costs +0.014 nats over 300 steps versus fp32 — cheap for
what it buys.

---

## 4b. RecGen does not reproduce HierGen — the recursive-consistency loss does not converge

> **Superseded in part by 4c-4f.** The diagnosis below ("the objective, not the
> architecture") holds, but its reading of the plateau was wrong: the levels are
> not both at cosine 0.5, one is at 0.66 and the other at 0.22 while scoring a
> substitution that never happens. And its list of untested ideas missed the
> structural one — RecGen's substitution carries no information about what was
> actually generated. The measured fix is 4f.

This is the most important result here, because RecGen is the paper's headline
efficiency claim.

Measured on the trained 247M model (200M tokens, `alpha = 0.3`, scheduled
sampling ramped to 0.25):

| | |
|---|---|
| RecGen vs HierGen greedy agreement | **15.4%** |
| KL(RecGen ‖ HierGen) | **3616** |
| throughput gain | 1.29x |
| KV cache reduction | 4.76x |

The efficiency is real. The output is not preserved at all — RecGen writes
different text.

The v3 architecture does not change this. Same measurement on the v3
checkpoint: **12.1%** agreement, KL **4608**, 1.27x throughput, 4.76x less KV.
v3's `loss_rec` ends at 0.907 against v1's 0.945 — better, and nowhere near
enough. So the failure is a property of the objective, not of the particular
expert layout or attention choices, which is what the next paragraph argues.

The cause is visible in the training log: `loss_rec` plateaus at **0.90–1.02**
and never falls further, in the v1, v2 and v3 runs and in the 8B run. That
term is `sum over levels of (1 - cos)`, so across two levels it means an average
cosine similarity of about **0.5** between `X_hat` and the encoder state `X` it
is supposed to reproduce. RecGen substitutes one for the other; at cos 0.5 that
substitution cannot hold.

Theorem A.6 states RecGen is exact *when* `X_hat^(L-1) = X^(L-1)`. That
premise is simply not reached here. It also may not be reachable by this
objective: `X_hat^(l-1)_m` is produced by the level-l decoder from
`X^(l-1)_{<m}`, i.e. **before** unit `m`'s own contents are seen, so `L_rec` is a
next-unit *prediction* loss, not a reconstruction loss. A predictor of the next
chunk summary cannot match it exactly, and 0.5 cosine may be close to what this
formulation can do.

Untested ideas, in rough order of how much they change the design:

1. Raise `alpha` well above 0.3 and `self_cond_prob` above 0.25 — cheapest, but
   it fights the token loss for capacity and may just trade perplexity for
   agreement.
2. Train much longer; the plateau looks like a floor rather than slow progress,
   but 200M tokens is not enough to be sure.
3. Change the upper decoders from predictors to reconstructors — let slot `j`
   see unit `j` itself and move the "prediction" burden entirely to the token
   level. That makes `L_rec` reachable in principle, at the cost of a
   redesigned indexing scheme.

Until one of those is demonstrated, **HierGen is the only protocol here whose
output can be trusted**, and it is the one verified bit-exact against the
training forward.

### 4c. Two of the three reasons RecGen fails, measured

§4b read the `loss_rec` plateau of 0.90-1.02 as "an average cosine of about 0.5
at both levels". A second reading fits the same number -- one level near 0.9 and
the other near 0.1 -- and those imply completely different fixes, so they were
separated by measurement on the trained 8B
(`scripts/recgen_diag.py`):

| level | A: cos(X_hat_j, X_j) | B: cos(post_j, X_j) | does RecGen substitute here? |
|---|---:|---:|---|
| L1 | **0.2226** | 0.3955 | **no** -- it emits a token and uses `embed(tok)` |
| L2 | 0.6567 | **0.8436** | yes |

`sum over levels of (1 - A) = 1.1207`, which is the plateau. Three things follow.

**1. 69% of the loss scored a substitution that never happens.** `_rec_loss`
looped `l` from 1, but `_emit_group` only substitutes for `l >= 2`; at `l == 1`
it calls `_emit_token`, which returns a real embedding. So the l=1 term was
asking the LM head's input to be cosine-similar to the *next token's embedding*
-- a next-token-prediction task in embedding space, unreachable for any
non-deterministic model -- and it dominated the gradient. Fixed by
`rec_loss_min_level`, default 2.

**2. The level that matters was never at 0.5.** L2 sits at 0.657, and the
reconstruction ceiling (B, the same decoder position *after* it has attended to
slot j) is 0.844. So §4b's third suggestion -- move the upper decoders from
predictors to reconstructors -- has a measured 0.19 of headroom behind it.

**3. RecGen's substitution carries no information about what was generated.**
This is the structural one and it was not in §4b at all. `X_hat_j` is the
decoder state that *predicted* slot j, computed before the slot was emitted. At
inference RecGen feeds it as the level-l decoder's content token for slot j, so
the upper decoder is conditioned on a guess made before generation rather than
on the text that actually came out. Nothing downstream can recover from that.

### 4d. ChunkGen: keep the saving, restore the dependence

What makes RecGen cheap is skipping the lower *encoders* and their KV caches.
The chunker is one linear map with no cache, so applying it to the units that
were actually emitted costs nothing and does depend on them:

    hiergen   unit_j = encoder(chunker(emitted))    exact, needs the cache
    recgen    unit_j = X_hat_j                      independent of `emitted`
    chunkgen  unit_j = chunker(emitted)             cache-free and dependent

Measured on the published 8B **with no retraining at all** (the decoder was
trained to consume encoder states, so this is a lower bound):

| protocol | throughput | KV B/token | agreement vs HierGen | KL |
|---|---:|---:|---:|---:|
| HierGen | 24.5 tok/s | 7512 | reference | -- |
| RecGen | 37.9 tok/s | 600 | 26.1% | 480.8 |
| **ChunkGen** | 37.8 tok/s | 600 | **30.1%** | **305.3** |

Identical cost, 15% better agreement, 36% lower KL. The remaining gap is the
train/inference mismatch that `chunk_cond_prob` exists to close --
cos(chunker(X^(l-2)), X^(l-1)) measured 0.4736, so the decoder is being handed
something well outside what it was trained on.

### 4e. The A/B: agreement doubled

Matched arms on GB10 -- v4 architecture, same data, same seed, same 610-step
schedule, 80M tokens each.  Two fields differ.

| arm | rec_loss_min_level | chunk_cond_prob | loss_rec (last 10) | token CE |
|---|---:|---:|---:|---:|
| ctl | 1 | 0.00 | 0.8549 | 4.4910 |
| fix | 2 | 0.25 | 0.2027 | 4.5100 |

`loss_rec` is not comparable between the arms -- ctl sums two levels and fix
sums one -- but its *trajectory* is.  ctl went 0.8774 -> 0.8510 over 400 steps,
a 3% improvement; fix went 0.2520 -> 0.1985, 21%.  The control plateaus exactly
as every previous run did.  Removing the unreachable term let the reachable one
move, which is the hypothesis behaving as predicted.  fix ends at
cos(X_hat, X) = 0.8015 at level 2, against 0.6567 measured on the 8B.

Agreement with HierGen, which is the number that matters:

| arm | RecGen | KL | ChunkGen | KL |
|---|---:|---:|---:|---:|
| ctl | 25.0% | 389.6 | 29.5% | 247.9 |
| **fix** | **43.8%** | 250.3 | **49.4%** | **137.4** |

Baseline (ctl + RecGen) to best (fix + ChunkGen): agreement 25.0% -> 49.4%,
KL 389.6 -> 137.4.  Agreement roughly doubled and KL fell to 35% of its former
value.  After four model sizes stuck between 12% and 30%, this is the first
movement.

**It is progress, not a solution.**  49.4% agreement still means half the tokens
differ, so ChunkGen output is not interchangeable with HierGen's.  HierGen
remains the only protocol whose output can be trusted.

There is also a cost: fix is 0.019 nats worse on token loss (~1.9% perplexity).
That is very likely `chunk_cond_prob` acting as input noise rather than the loss
restriction, since the latter only removes a term that could never be satisfied
-- but "very likely" is not measured, so a third arm with
`rec_loss_min_level: 2, chunk_cond_prob: 0.0` is running to attribute it.

### 4f. Five arms: the answer is to replace RecGen, not repair it

All five matched on GB10 -- v4 architecture, same data, same seed, same 610-step
80M-token schedule.  Only the objective differs.

| arm | rec_loss_min_level | alpha | chunk_cond_prob | token CE | RecGen | **ChunkGen** | KL(chunk) |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctl | 1 | 0.30 | 0.00 | 4.4649 | 25.0% | 29.5% | 248.0 |
| lossonly | 2 | 0.30 | 0.00 | 4.4954 | 31.8% | 42.6% | 265.3 |
| fix | 2 | 0.30 | 0.25 | 4.4641 | 43.8% | 49.4% | 137.4 |
| chunk10 | 2 | 0.30 | 1.00 | 4.5486 | 46.0% | 39.2% | 172.0 |
| **nolrec** | 2 | **0.00** | 1.00 | 4.4825 | 26.7% | **52.3%** | **73.3** |

**Best is `nolrec`: ChunkGen at 52.3% agreement and KL 73.3**, against the
baseline's 25.0% / 389.6 (RecGen) -- agreement 2.1x, KL down 5.3x -- for 0.018
nats of token loss.

Note what that arm is: `rec_loss_alpha = 0`.  **The best configuration removes
the recursive-consistency loss entirely.**  That is consistent with what the
loss is for -- it exists to make `X_hat ~= X` so RecGen's substitution is valid,
and ChunkGen does not perform that substitution.  Carrying it is paying for a
guarantee the protocol no longer needs.

**The two levers interact, and not additively.**  `chunk_cond_prob = 1.0` is
*worse* than 0.25 while L_rec is on (39.2% vs 49.4%) and *better* once it is off
(52.3%).  The reason is direct opposition: L_rec pulls `X_hat` toward the
encoder state `X`, while `chunk_cond_prob` trains the decoder to consume chunker
summaries instead.  At probability 1.0 the decoder never sees `X` as content,
yet L_rec still insists `X_hat` match it.  Anyone tuning one of these without
the other will read the result backwards.

`lossonly` settles the attribution question: the loss restriction alone is worth
+6.8pp RecGen and +13.1pp ChunkGen, and the chunker substitution supplies the
rest.  It also refutes an earlier guess of mine -- `chunk_cond_prob` was
supposed to be the term costing perplexity, but `fix` (with it) has the *lowest*
token CE of all five arms.

**Two caveats, both real.**  52.3% still means half the tokens differ, so
ChunkGen output is not interchangeable with HierGen's; HierGen remains the only
protocol whose output can be trusted.  And the CE spread across four of the five
arms is 0.031 nats on one seed each, which is not enough to rank them --
only `chunk10` sits clearly outside the pack.

### 4g. Every agreement number in 4b–4f was measured against a broken reference

The sentence that closes 4f -- "HierGen remains the only protocol whose output
can be trusted" -- was false when it was written, and it was the load-bearing
assumption under every agreement figure above it.

`scripts/protocol_diag.py` scores each protocol against **the teacher-forced
training forward** rather than against HierGen.  HierGen is supposed to *be*
that forward; `test_hiergen_matches_the_training_forward` asserts it.  On the
trained `nolrec` checkpoint it scored **61.0% agreement, KL 0.247**, with a
maximum logit difference of 7.65.  float32 returned the same numbers, so it was
not rounding, and an untrained model of the same config returned 100% / 7e-6,
so it was not the orchestration.

The cause: `Attention.forward_mla_absorbed` -- the weight-absorbed MLA path,
reachable **only** through a `LatentKVCache`, i.e. only during generation --
never applied the attention output gate.  The uncached path that training uses
does.  So every affected checkpoint generated with a different function from
the one it was trained as: both 8B releases and all six 250M v4 arms, i.e.
every config with `qk_norm: false` and `output_gate: true` on an MLA stack.

Two things hid it independently, and both are instances of §6:

* **`w_gate` is initialised to zero** and the gate is `2*sigmoid`, so it is
  exactly 1.0 on a fresh model.  A missing multiply by 1.0 is unobservable, and
  every equivalence test ran on a fresh model.
* **`configs/base_tiny.yaml` is the only config the tests load, and it
  sets `qk_norm: true` on its top encoder and omits `output_gate` entirely.**
  QK-Norm disables absorption, so the tested path was not the shipped path --
  and the module being dropped was not even instantiated.

The two new tests force `qk_norm=False, output_gate=True` and run
`_shake_zero_init_parameters`, which gives every parameter that `__init__`
leaves at zero a real value.  Both fail without the fix and pass with it,
verified by stashing the fix rather than by assuming.

**The rule this yields:** a test that two code paths compute the same function
is only valid on parameters no initialiser would produce.  Zero-init is not a
neutral choice of input; it is the one setting that makes whole modules
unobservable.

With the gate applied, HierGen scores **97.6% / KL 0.0017** on the same
checkpoint -- the bf16 noise floor, and what the metric was always meant to
read.

The bug is inference-only, so every training CE in 4e and 4f stands.  What
moves is every agreement and KL column, because they were compared against a
HierGen that was itself 39% wrong.

### 4h. The error is in the half that nothing was training

With a correct reference the two substitutions separate.  A finished group of
level-(l-1) units is consumed in **two** independent places:

    content   the level-l decoder's content token for slot j.
              This is what `chunk_cond_prob` trains for.
    up        the input to the level-l chunker, and so to the level-l encoder.
              Substituting here corrupts X^(L) itself, and no training term
              covers it -- `encode_all` runs the real encoders bottom-up at
              every value of `chunk_cond_prob`.

`recgen` and `chunkgen` substitute in both at once, which is why four rounds of
tuning could not tell the two errors apart.  The two crosses -- which keep the
lower caches, so they are diagnostics and not usable protocols -- attribute it.
`nolrec` checkpoint, 4x512 real tokens:

| protocol | content | up | agreement vs training forward | KL | cos(X_top) |
|---|---|---|---:|---:|---:|
| hiergen | encoder | encoder | 97.6% | 0.002 | 1.0000 |
| content_only | chunker | encoder | 79.1% | 0.192 | 1.0000 |
| up_only | encoder | chunker | 63.9% | 0.600 | 0.7296 |
| chunkgen | chunker | chunker | 56.0% | 0.877 | 0.7295 |
| recgen | xhat | xhat | 36.5% | 1.703 | 0.3558 |

**The `up` path costs 36.1pp of agreement, the `content` path 20.9pp.**  Four
rounds tuned the smaller half, and the larger half is the one no objective can
reach without changing what `encode_all` computes.

`cos(X_top)` says it more directly: substituting on the way up drives the
top-level conditioning state to 0.73 cosine against its true value -- 0.69 by
the last group, so still drifting -- and RecGen to 0.36.  That state is the
only thing carrying context across the whole sequence.

#### Re-scoring all five arms reverses 4f's conclusion

Same measurement on every arm, no retraining -- only the reference changed:

| arm | α | chunk_p | token CE | hiergen | content_only | up_only | **ChunkGen** | RecGen |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ctl | 0.30 | 0.00 | 4.4649 | 98.4% | 72.4% | 66.9% | 43.4% | 32.0% |
| lossonly | 0.30 | 0.00 | 4.4954 | 98.5% | 74.0% | 76.1% | 60.9% | 29.1% |
| **fix** | 0.30 | 0.25 | **4.4641** | 98.7% | 88.3% | 81.3% | **71.1%** | 31.0% |
| chunk10 | 0.30 | 1.00 | 4.5486 | 98.4% | 66.2% | 77.6% | 53.4% | **53.9%** |
| nolrec | 0.00 | 1.00 | 4.4825 | 97.6% | 79.2% | 64.6% | 56.2% | 36.3% |

**4f's headline is wrong.**  It ranked `nolrec` first at 52.3% and `fix` second
at 49.4%, and concluded that the best configuration deletes the
recursive-consistency loss.  Against a correct reference `fix` leads by 15
points -- 71.1% against 56.2% -- and `fix` is the arm that *keeps* `L_rec` at
α = 0.30.  It also has the lowest token CE of the five.  The configuration 4f
recommended (`rec_loss_alpha: 0.0`, `chunk_cond_prob: 1.0`) is the worse of the
two on the corrected metric, and the recommendation in the report's §6 has been
withdrawn.

The interaction 4f described is real but its sign was misread.  `L_rec` and
`chunk_cond_prob` do pull in different directions, and the resolution is a
*moderate* amount of both (α = 0.30, chunk_p = 0.25) rather than dropping
either: chunk_p = 1.0 costs agreement whether `L_rec` is on (53.4%) or off
(56.2%), and both trail 71.1%.

That a broken reference inverted a five-arm ranking is the cost of scoring a
protocol against another protocol.  Score against the training forward: it is
the only quantity in the system that is defined without reference to the code
being tested.

### 4i. So stop approximating the exact protocol; make it cheap instead

RecGen exists because an intermediate encoder's KV cache grows with the
sequence.  Its price is that `X_hat_j` is computed before slot `j` is emitted,
so it cannot depend on what was generated -- and that price cannot be paid by
training, because matching it to `X_j` would mean predicting the summary of
tokens that have not been sampled yet.  For a model with non-zero entropy that
is impossible.  That is the honest reading of Theorem A.6: its premise
(recursive consistency) is unreachable, not merely unreached.  The paper is
explicit that it never checked -- "we leave a more rigorous likelihood-based
evaluation of RecGen for future work" -- and reports no agreement numbers at
all.

The alternative is to attack the cache growth directly.  Confine a level-l
encoder (l < L) to the `C_{l+1}` units that become one level-(l+1) unit:
block-causal attention that never crosses a meta-group boundary.  Then

* nothing outside the current group is ever readable, so the cache holds one
  group and is rewritten at every boundary -- **O(1) in sequence length**,
  which is exactly the saving RecGen was buying;
* the summary handed up and across is still the real encoder state, so
  **HierGen stays bit-equal to the training forward**;
* `L_rec` and `chunk_cond_prob` both become unnecessary, because there is no
  substitution left to correct for.

Global context is not lost -- it never lived there.  It flows through the top
encoder, which keeps full attention and is already amortised over `C_<=L`
tokens, and reaches the lower decoders through the converter.

`configs/base_small_blocal.yaml` is identical to `nolrec` except for
`encoder_block_local: true` and both objective knobs set to zero:

| | nolrec | blocal |
|---|---:|---:|
| parameters | 0.252B | 0.252B (identical) |
| KV/token, HierGen | 0.945 KiB | **0.207 KiB** |
| KV/token, RecGen | 0.195 KiB | 0.195 KiB |

Exact generation now costs 6% more than RecGen's approximate generation,
against 4.8x more before.

Trained on GB10, matched to the other five arms (v4, same data, same seed, same
610-step 80M-token schedule).  All six re-scored identically, CE as the mean
over the last 100 steps:

| arm | α | chunk_p | token CE | **HierGen** | ChunkGen | RecGen | KV/token |
|---|---:|---:|---:|---:|---:|---:|---:|
| ctl | 0.30 | 0.00 | **4.4910** | 98.4% | 43.4% | 32.0% | 0.945 KiB |
| fix | 0.30 | 0.25 | 4.5100 | 98.7% | 71.1% | 31.0% | 0.945 KiB |
| **blocal** | 0.00 | 0.00 | 4.5164 | **98.3%** | 40.8% | 42.7% | **0.207 KiB** |
| lossonly | 0.30 | 0.00 | 4.5182 | 98.5% | 60.9% | 29.1% | 0.945 KiB |
| nolrec | 0.00 | 1.00 | 4.5278 | 97.6% | 56.2% | 36.3% | 0.945 KiB |
| chunk10 | 0.30 | 1.00 | 4.5726 | 98.4% | 53.4% | 53.9% | 0.945 KiB |

**The comparison that matters is 98.3% at 0.207 KiB against 71.1% at 0.195
KiB.**  Six percent more memory buys the difference between "three tokens in
ten differ" and "identical up to bf16 noise".  Four rounds of tuning the
objective moved ChunkGen from 43.4% to 71.1%; changing what the encoder is
allowed to attend to closed the rest of the gap and made the question moot.

`blocal`'s ChunkGen and RecGen columns are poor, and that is expected -- it was
trained with both objective knobs at zero, so nothing prepared it for a
substitution it never has to make.  Those columns are not the ones to read.

**The honest cost.** `blocal` is third of six on token CE, 0.025 nats behind
`ctl`.  The six arms span 0.082 nats on one seed each, so this does not
establish that block-local attention is worse -- but it does not establish that
it is free either, and it trailed at every intermediate step before closing
during cooldown.  Confining the level-1 encoder to 16 tokens is a real
restriction; the claim here is that it is a cheap one, not that it is free.

### 4j. The 8B has a *second*, unrelated reason HierGen is not exact

Re-running the diagnostic on the published 8B after fixing the output gate:

| protocol | agreement vs training forward | KL |
|---|---:|---:|
| hiergen | **42.6%** | 0.473 |
| chunkgen | 19.9% | 1.470 |
| recgen | 13.4% | 2.382 |

float32 returned 42.4%, so it is not precision.  Localising it the same way:
every one of the four stacks diverges between its batched and its incremental
path, *including the decoders*, which see 8 positions and cannot have a
length-dependent bug.  Replacing the routers with dense FFNs makes all four
agree to 8e-4, and the top-k selections themselves are identical (0.0% flipped,
no ties at the boundary).  So it is the MoE, but not the routing decision.

It is **`capacity_factor: 1.25`**, which the 8B sets on every MoE stack and the
250M configs leave at `None`:

```python
capacity = int(counts.max())
if self.cfg.capacity_factor:
    capacity = min(capacity, max(1, math.ceil(N * K / E * factor)))
```

`N` is the number of rows *in this call*. Training passes thousands and drops
the overflow; incremental generation passes one group and never drops anything.
Forcing `capacity_factor = None` makes both decoders exact again
(cos 1.000000, from 0.869 and 0.997).

**There is no shape-independent version of this rule.** "Was this expert
oversubscribed" is a question about the batch, not about the token, so a model
trained with capacity dropping is a function of its batch composition and no
cache discipline can make generation reproduce it.  `ScalaForCausalLM` now
warns when it is set.

What survives on the encoders after that is small (cos 0.9998, 0.99999 over 12
layers and 600 units) and is consistent with ordinary floating-point noise
amplified through depth.

**Consequence for the released models.** `unofficial-photon-repro-ja-8b-a1b-v2`
and `-v3` generate from a different function than they were trained as, for two
independent reasons, and only one of them is fixable after the fact. Their
perplexity numbers are unaffected -- those come from the training forward. The
250M v1–v4 line is unaffected by the capacity issue and reads 98% after the
gate fix.

### 4j-2. RecGen transmits exactly zero bits from what it generates

Everything in 4b–4i is about *training*: which objective, which content input,
how to close a train/inference gap. This measurement has nothing to do with
training, and so cannot be repaired by any of it.

Perturb one token, then ask whether a later token's logits move. Untrained
weights, tiny config, `forced_logits` so the comparison is exact:

| protocol | same unit (t34) | later unit, **same** meta-group (t45) | next meta-group (t49) |
|---|---:|---:|---:|
| HierGen | 5.747e-01 | 2.396e-02 | 6.845e-01 |
| ChunkGen | 5.255e-01 | 1.537e-02 | 2.316e-01 |
| **RecGen** | 2.545e-01 | **0.000e+00** | **0.000e+00** |

Not small — **exactly zero**, and zero for every later group as well.

The derivation is two lines. RecGen uses `X̂` in both roles: as the level-L
decoder's content token, and as what is handed to the top chunker
(`Â^(L)_{g+1} := C_θ^(L)(X̂^(L−1)_{I_{g+1}})`, the paper's own step (ii)).
`X̂_0` is a function of the conditioning alone, `X̂_1` of the conditioning and
`X̂_0`, so every `X̂_j` in group `g` is a function of `X^(L)_{g-1}` only. Then

```
X^(L)_g = F( C( X̂_{I_g} ) ) = F'( X^(L)_{g-1} )
```

and by induction the entire top-level trajectory is a deterministic unrolling
of the **prompt's** final state. Nothing RecGen emits influences anything
beyond its own `C_1`-token unit. The `C_L` units of a meta-group are decoded
from identical context — conditionally independent given the previous group.

**This is why no objective ever moved the number.** `L_rec`, its level
restriction, scheduled sampling, `chunk_cond_prob`, and reading Equation (6)
literally all adjust *what the channel carries*. The channel has no capacity.
The 13–54% agreement measured across seven arms and two model sizes is simply
how often a context-free continuation coincides with a context-aware one.

It also says exactly what a repair must do: make the upward feed depend on the
emitted tokens. There are only two ways — `chunker(emitted)` (one linear, no
cache) or `encoder(chunker(emitted))` (exact, needs a cache, made cheap by
§4i's block-local encoders).

**So `mode="recgen"` now uses the first one.** RecGen's contribution is its
*cache discipline* — only the top level is kept, which is where the 14x saving
comes from — and that part was always sound. What was broken was the one line
choosing the summary. Swapping it keeps the cache discipline, the amortisation,
and the KV figure exactly, and it restores the channel:

> **Naming, since §4l.** The chunker-summary protocol below is now called
> `chunkgen`; `recgen` is the bounded-encoder protocol that replaced it. The
> numbers here are unchanged, only the label.

| on `recgen-fix`, scored against the training forward | agreement | KL |
|---|---:|---:|
| `chunkgen` (repaired: chunker summary) | **71.2%** | 0.649 |
| `recgen_paper` (Equation (6) / step (ii) verbatim) | 31.3% | 1.622 |

Same checkpoint, same cache, same 0.195 KiB/token. 2.3x the agreement for one
line.

The paper's rule stays available as `recgen_paper` — the reproduction claim in
the README should stay honest, and the zero-propagation property has to remain
measurable. `test_recgen_generation_cannot_see_what_it_generated` asserts both
halves: `recgen_paper` moves a later unit by exactly 0.0, and `hiergen`,
`recgen` and `chunkgen` all move it by more than 1e-4, so neither assertion is
vacuous.

**This does not make RecGen exact**, and nothing will: 71.2% still means three
tokens in ten differ, and the residual is the `up` path (`cos(X_top)` 0.684).
For output that matches training, use HierGen with `encoder_block_local` —
98.3% at 0.207 KiB against this 71.2% at 0.195 KiB.

### 4k. Was the implementation wrong? Equation (6) is hatted and this repo was not

The obvious objection to 4i is that RecGen is the paper's method, so the likely
explanation is an implementation error rather than a defect in the method.
Checked, and there *is* a discrepancy.

The paper's Equation (6) writes the Context Decoder's content input **hatted**:

```
X̂^(l−1)_{I_g,j} = G_θ^(l)( U^(l)_{g−1} , X̂^(l−1)_{I_g,<j} ; M_{R_l,j}^(l−1) )
```

`hierarchy.py`'s docstring and `architecture.md` §2 both transcribed it *unhatted*,
and `decode()` feeds `content = enc[l-1]` — the true encoder states. Read
literally, no encoder state ever enters an upper decoder, and RecGen's content
substitution would not be a substitution at all: it would be what training did.
Every patch this project built — `L_rec` weighting, `self_cond_prob`,
`chunk_cond_prob` — would be closing a gap only this implementation had.

The paper does not adjudicate. Equation (6) sits in the *generation* section,
and the objective section says nothing about what the decoders consume during
training. So it was implemented (`recursive_decoder_input`) and measured.

**The literal reading is exact where it was supposed to be.**
`test_recursive_input_makes_recgens_content_substitution_exact` shows the
`xhat_content` protocol — RecGen's content choice with the bottom-up path left
intact — reproducing the training forward to 2e-4, which it provably does not
do under teacher forcing.

**And it blinds the model.** `X̂_0` is a function of `U^(l)_{g-1}` alone, `X̂_1`
of `U` and `X̂_0`, so by induction every `X̂_j` in group `g` depends only on
`X^(L)_{g-1}`. Nothing emitted inside group `g` reaches the conditioning, and a
token cannot see the earlier *units* of its own meta-group — 12 tokens here.
Measured directly: perturbing a token in an earlier unit of the same meta-group
moves a later token's logits by 2.4e-2 under teacher forcing and by 6.0e-7
under the recurrence, while the next meta-group moves identically under both
(`test_recursive_input_costs_within_group_context`).

That is the failure §3 of `architecture.md` calls "a blind spot exactly one
chunk wide", and it costs what a blind spot costs. Matched arm on GB10 — same
architecture, data, seed and 610-step schedule as the other six:

| arm | token CE | **HierGen** | `xhat_content` | ChunkGen | **RecGen** | cos(X_top) at RecGen |
|---|---:|---:|---:|---:|---:|---:|
| ctl | 4.4910 | 98.4% | — | 43.4% | 32.0% | 0.643 |
| fix | 4.5100 | 98.7% | — | 71.1% | 31.0% | 0.679 |
| blocal | 4.5164 | 98.3% | — | 40.8% | 42.7% | 0.453 |
| nolrec | 4.5278 | 97.6% | — | 56.2% | 36.3% | 0.356 |
| **recur** | **5.0477** | **85.5%** | **97.7%** | 54.6% | **50.0%** | 0.689 |

Three things happen at once, and all three were predicted.

1. **`xhat_content` is 97.7% / KL 0.001.** RecGen's content substitution is
   exactly the training forward, as intended. That half of the problem is gone.
2. **RecGen still only reaches 50.0%**, behind `chunk10`'s 53.9%. The binding
   constraint was never the content path — it is `up`, and `cos(X_top)` is
   0.689, better than any other arm but nowhere near 1.
3. **HierGen falls to 85.5%.** Training never shows an upper decoder an encoder
   state, so HierGen — which does — is now the off-distribution protocol. The
   trustworthy and the cheap protocol swap places, and neither is exact.

And it costs **0.53 nats** of token CE against `ctl` — six arms span 0.082
nats, so this is not seed noise, it is the blind spot.

**So the hats describe the generation recursion, not the training rule.**
Teacher forcing is what makes the receptive-field union in §3 come out to
exactly `t_{<i}`, and delivering full context cheaply is PHOTON's entire claim.
The implementation was right on this point; the docstring's transcription was
wrong, and that is now fixed in both places.

The objection that started this section was the right one to raise — "the
method is published, so suspect the implementation first" is the correct prior,
and testing it found a real discrepancy in the transcription. It just did not
find the cause: reading Equation (6) literally makes RecGen's content path
exact and RecGen as a whole is still unusable, because the error was always
somewhere else.

**What this buys is a sharper statement of the original problem, not a
retraction of it.** Definition A.4 requires `X̂^(L-1)_{I_g} = X^(L-1)_{I_g}`.
Under the literal reading the left side is a function of group `g-1` while the
right side summarises tokens inside group `g`, so equality demands that every
meta-group be determined by its predecessor — zero conditional entropy over
16-token blocks. Under teacher forcing the left side is a function of group
`g-1` plus the true earlier units of group `g`, which is strictly more
information but still excludes slot `j` itself. Either way `X̂_j` is formed
before slot `j` exists.

Theorem A.6 is true and vacuous: the premise is not unmet, it is unmeetable by
any model with something left to predict. That is a statement about RecGen, not
about PHOTON — the architecture, HierGen, and the KV savings all stand, and
§4i's block-local encoders deliver those savings exactly.

### 4l. The fix: RecGen's claim is about memory, so pay it with memory

Five rounds of §4b–§4k asked *which summary should be substituted*. All five
lost, and 4j-2 explains why in one line: the summary has to be chosen before
the meta-context is sampled, so no choice of it can carry what was sampled.
The sixth round asks a different question — **why is anything being substituted
at all?**

RecGen substitutes because the intermediate encoders' KV caches grow with `T`.
That is the entire motivation; the paper's own summary table is a memory table:

| | HierGen | RecGen |
|---|---|---|
| Size | `Σ_l O(T/C_≤l)` | `O(T/C_≤L)` |
| Access | `Σ_l O((T/C_≤l)²)` | `O((T/C_≤L)²)` |

Deleting a cache is one way to stop it growing. **Bounding it is another, and it
costs O(1) instead of costing the substitution.** A level-`l` encoder allowed to
read `w` blocks of `C_{l+1}` units back holds a fixed number of units forever,
so only the top level's cache still grows with `T` — the claim is met — and
what it hands upward is the real encoder state, so nothing is substituted and
`L_rec`, `self_cond_prob` and `chunk_cond_prob` have nothing left to correct.

`recgen` is now this protocol; `enc_window_groups` is `w`; the old
chunker-substituting protocol is `chunkgen`; the paper's rule is
`recgen_paper`.

#### Measured, no retraining

Probe model (65M dense, 2 levels, 4×4, 30M tokens on one 5060 Ti, `α = 0.3`,
`chunk_cond_prob = 0`), 512-token sequences, everything scored against the
teacher-forced training forward:

| protocol | content / up | agree | **KL** | cos(X_top) | KiB/token |
|---|---|---:|---:|---:|---:|
| `hiergen` | encoder / encoder | 99.2% | 0.0002 | 1.0000 | 0.652 |
| `recgen` w=8 | encoder / encoder | 92.0% | **0.021** | 0.9908 | 0.253 |
| `recgen` w=4 | encoder / encoder | 87.3% | 0.042 | 0.9810 | 0.191 |
| `recgen` w=2 | encoder / encoder | 82.2% | 0.073 | 0.9671 | 0.159 |
| `recgen` w=1 | encoder / encoder | 80.1% | 0.100 | 0.9544 | 0.144 |
| `chunkgen` | chunker / chunker | 70.4% | 0.265 | 0.8378 | 0.128 |
| `recgen_paper` | xhat / xhat | 49.5% | **1.302** | 0.7687 | 0.128 |

**At `w=1` — 12.5% more memory than the paper's rule — KL falls 13x, from
1.302 to 0.100. At `w=8`, still 2.6x cheaper than HierGen, it falls 62x.** The
window is a dial with no discontinuity in it: every step up costs O(1) memory
and buys agreement, and the limit is HierGen.

**`cos(X_top)` jumps from 0.77 to 0.95 at `w=1`.** The top-level conditioning
state is what §4h identified as the untrained half. It is not untrained, it is
*unsubstitutable* — and once you stop substituting, it stops drifting.

> **Correction, same day.** The first version of this table was measured with
> `protocol_diag.load_tokens` reading a `uint32` corpus as `uint16`. That does
> not raise: a 99,584-entry vocabulary means every id needs more than 16 bits,
> so the reader returned each id's low half interleaved with its high half —
> ids still inside the vocabulary, a loss curve that still looked like a loss
> curve, and a CE of 12.5 that nobody was printing. It was caught by
> `position_diag.py`, which is the first script here to compare an absolute CE
> against the training log instead of comparing two protocols to each other.
> Every number moved in the same direction and the ranking did not change, but
> one claim did: on the corrupted stream `chunkgen` had the better agreement
> and the worse KL than `recgen_paper`, which read as the two metrics
> disagreeing. On real tokens they agree (70.4% / 0.265 against 49.5% / 1.302).
> `load_tokens` now reads the dtype from `manifest.json` and refuses ids past
> the vocabulary.

#### And Definition A.4's residual is not a training gap

`scripts/bottleneck_entropy.py` samples 24 different meta-contexts from the
*same* `X^(L)_g` and measures the real `A^(L)_{g+1}` each produces. `Â^(L)_{g+1}`
is one point committed to before the draw, so the best it can possibly do is
the centroid of that cloud:

```
within-condition spread   cos = 0.4018   (1.0 would mean the summary ignores the sample)
ceiling for any A_hat     cos = 0.6528   (the centroid)
measured A_hat            cos = 0.6130

reachable by training     +0.0399
structurally out of reach +0.3472
```

`L_rec` has already closed **94% of everything it can close**. The remaining
0.35 is the conditional entropy of a 16-token block, and Lemma A.5 asks for it
to be zero. That is why every objective in §4c–§4f moved the number by so
little: they were optimising against a ceiling of 0.65, not 1.0.

So the honest summary of six rounds: the paper's Theorem A.6 is correct, its
premise is self-defeating rather than merely unmet, and RecGen's *actual*
contribution — `O(T/C_≤L)` global KV — survives intact once you buy it with a
bounded cache instead of a substitution. Trained with `encoder_block_local`
(§4i), the window *is* the receptive field and `recgen` reproduces the training
forward bit-for-bit
(`test_recgen_is_exact_when_the_window_is_the_trained_receptive_field`).

### What the bound costs

Three things, all measured; none of them is hidden in the KL table above.

**1. The forward pass.** Deleting the intermediate encoders also deleted their
compute; bounding them keeps it. 250M config, batch 32, 256-token prompt + 768
new, one 3070 Ti:

| protocol | throughput | KV cache | TPM |
|---|---:|---:|---:|
| `hiergen` | 1008.7 tok/s | 31.1 MiB | 33.2K |
| `recgen` (w=4) | 929.5 tok/s | **8.0 MiB** | **118.3K** |
| `chunkgen` | 1316.4 tok/s | 6.5 MiB | 206.0K |
| `recgen_paper` | 1401.8 tok/s | 6.5 MiB | 219.4K |

`recgen` is **3.9x less KV and 3.6x TPM against HierGen, at 0.92x its
throughput** — the level-1 encoder still runs, it just reads a bounded window.
It is the *slowest* of the four protocols. The Size row of the paper's table is
delivered in full; part of the Access row is traded back for the output being
right. `recgen_paper`'s 1.39x is the throughput number that reaching for the
substitution actually buys, and 49.5% agreement is its price.

**2. The window is tiled, not sliding — and that costs a 4x sawtooth.**
`_enc_step` recycles the buffer from position 0 once it fills, so a level-l
unit sees between 0 and `enc_capacity` units of history depending on where it
lands in the tile. Per-token KL against the training forward, 65M probe,
held-out Japanese, by position inside the tile:

| `enc_window_groups` | tile | KL at tile start | at tile end | overall |
|---|---:|---:|---:|---:|
| 1 | 16 tok | 0.173 | 0.184 | 0.220 |
| 2 | 32 tok | 0.074 | 0.099 | 0.160 |
| **4** | **64 tok** | **0.192** | **0.035** | **0.098** |

Widening the window improves the average and *concentrates* what is left at
the boundaries: at w=4 a token just after a reset is **4.1x** worse than one
just before the next. A true sliding window would hold every position near
0.035, i.e. roughly another 3x on the average — this is the largest known
improvement still on the table for the protocol. It is not free to implement:
`KVCache.update` derives both the write index and the RoPE offset from
`cache_start`, so sliding needs those decoupled (keys keep their absolute
phases while the buffer rolls), and the `block`/`window` attention masks index
the buffer rather than absolute position.

Chosen anyway, because recycling *is* the `encoder_block_local` function, which
is what makes `recgen` bit-exact on a model trained that way — a sliding window
would break that identity for the case that matters most.

**3. Prefill and decode use different spans.** The prompt is encoded with full
attention (exact, and what the paper's "the prefill remains unchanged" allows),
then only the current partial tile is replayed into the bounded cache. So the
cached keys were computed under a window while the states the prompt handed
upward were computed under full attention. Uniform windowing would remove the
seam at the cost of the prompt's own fidelity; that trade has not been
measured.

## 4m. Auditing the architecture itself: three instruments, and what they refuse to say

§4b–§4l were about one protocol. This section is about PHOTON, and it starts
with the measurement problem, because two of the three obvious instruments
gave a confidently wrong answer on the first try.

**The probe.** 65M dense, 2 levels, 4x4, 30M tokens on one 5060 Ti, ~25 min per
arm. Small enough to run several arms, and the point of §4l's `uint16` bug is
that it is worth having a model you can retrain rather than a checkpoint you
must trust.

**Two runs of the identical config, differing only in seed, land 0.065 apart
on held-out Japanese CE** (0.0645 and 0.0689 on two disjoint slices). Every
delta below is read against that, and most of them are smaller than it.

That number is *not* a floor and this section originally treated it as one.
It is a single realisation of `|X1 - X2|` with no confidence attached: it can
license "unresolved", never "equal". The first published version also measured
it at 0.053 on a training-loss tail and 0.048 on 3,072 English tokens the model
had trained on; both are retracted below.

### The instruments

`position_diag.py` — CE by position *inside* the meta-context. This is the
direct readout of what chunk-local decoding costs: slot 0 of a level-1 chunk
has no token-level context at all, slot 3 has three tokens of it.

```
mean by slot-in-chunk   0=5.0152  1=4.9409  2=4.9116  3=4.8960
mean by chunk-in-group  0=4.9178  1=4.9452  2=4.9319  3=4.9688
```

**+0.119 nats from slot 0 to slot 3, and nothing at all across chunks.** The
level-1 funnel is a real cost and the level-2 funnel is free. That asymmetry
is the useful part: the sawtooth is *within* a chunk, so it is about token
detail, not about global context.

This script also found the `uint16` bug (§4l), and the reason is worth keeping:
**it is the first diagnostic here that compares an absolute number against an
independent reference** — CE against the training log — instead of comparing
two of the model's own outputs to each other. Protocol agreement, KL between
protocols, cosine between states: all of those are self-referential and all of
them looked fine on a corrupted token stream.

`context_diag.py` — three views of long-range context, of which **only one is
interpretable**:

| | probe reading | what it looks like | what it is |
|---|---|---|---|
| CE by absolute position | rises past ~64 | dead long-range channel | shards concatenate documents, so a later position is a context more likely to span a boundary, not a longer one |
| max abs logit shift at distance `d` | 0.7% of `d=1`, flat | dead long-range channel | the local path dominates a max-norm, and there is no reference saying what 0.7% should be |
| **CE with everything past `d` replaced by noise** | **+0.34 nats at `d=16`** | — | the channel is alive and worth a third of a nat |

```
    d       CE   nats lost
  512   4.9214     +0.0000
  256   4.9570     +0.0356
  128   5.0046     +0.0832
   64   5.0723     +0.1509
   32   5.1735     +0.2521
   16   5.2590     +0.3376
```

The ablation needs no reference model and no interpretation: it is how many
nats the model's own predictions lose when the context is taken away, so it
measures **use**, not potential. The other two measure something that
correlates with use only when nothing else is going on. Both said "the
hierarchy is decorative"; it is not.

### Hypothesis 1: the cascade is too narrow — refuted

Everything a token knows about text outside its meta-context arrives through
`X^(L)_{g-1}` -> converter -> level-L decoder -> **one `D_l` vector** ->
converter -> level-1 decoder: a bottleneck applied once per level, in series,
with a shallow decoder between each squeeze. `ScalaConfig.global_skip` hands
every local decoder `X^(L)_{g-1}` directly as well, by widening the converter's
input matrix — no extra sequence positions, so the per-token decoder attention
is unchanged, and provably no wider a receptive field, since that is the state
the level-L decoder was already conditioned on
(`test_global_skip_is_causal_and_still_exact`).

Held-out CE as a delta against the two-seed mean, on three disjoint slices
(the numbers first published here were the contaminated ones; see below):

| arm | ja@48M | ja@60M | en@40M | ctx nats |
|---|---:|---:|---:|---:|
| probe | +0.032 | +0.035 | +0.002 | 0.323 / 0.327 / 0.325 |
| probe (seed 777) | -0.032 | -0.035 | -0.002 | 0.372 / 0.386 / 0.306 |
| **probe + `global_skip`** | -0.019 | +0.020 | -0.013 | 0.286 / 0.302 / 0.244 |

**No effect.** The delta is smaller than the two-seed spread on every slice and
changes sign between them, and 0.6M extra parameters bought it.

The reading that survives: everything the skip delivers was already reachable
through the cascade, so adding a second route buys nothing. The funnel is not
the binding constraint at this scale, which kills the whole "add another path"
family — including the obvious variants (feed the previous chunk's chunker
summary, widen `R_l`, stack more conditioning vectors). The flag stays,
defaulting off, because a refuted hypothesis with a test attached is cheaper to
keep than to re-derive.

### Where the compute actually goes, and why `converter_width` is suspicious

If adding capacity to the conditioning path does nothing, the next question is
what the existing capacity is spent on. Counting layer-positions per generated
token — a level-`l` encoder layer runs once per `C_<=l` tokens, a level-`l`
decoder layer runs over `R_l + C_l` positions per `C_l` units:

| stack | layer-positions / token | share |
|---|---:|---:|
| L1 encoder | 1.000 | 11.9% |
| **L1 decoder** | **6.000** | **71.6%** |
| L2 encoder | 0.375 | 4.5% |
| L2 decoder | 1.000 | 11.9% |

**Seventy-two percent of the layer-position budget is in the one stack that can
see eight positions.** And half of those eight are the converter's `R_1`
outputs, which every layer processes in full and which `decode()` then
discards: `out[:, R-1 : R-1+C]` keeps `C` of `R + C`.

> **Read that table as positions, not compute.** It excludes the LM head,
> which at `d_token=384` over a 99,584-entry vocabulary is 74% of the forward
> FLOPs and which no hierarchy knob touches. Using it as a compute number is
> exactly the error the next subsection retracts. `flops_per_token` now breaks
> out `head` and `stacks`; `layer_positions_per_token`'s docstring says so.

They are not free to drop — layer `n`'s attention reads layer `n-1`'s values at
those positions, so they are a prefix that evolves, not a static memory. But
`R_l` is a hyper-parameter nobody has swept, and it is a pure multiplier on the
stack with the largest position budget: `R_1 = 4 -> 2` takes the model from
8.375 to 6.625 layer-positions per token, i.e. **-21% of the stacks and -5.4%
of forward FLOPs**, for -0.9% parameters.

The same arithmetic says depth might be misallocated in the other direction. A
level-1 decoder layer costs `(R+C)/C = 2.0` layer-positions per token; a
level-1 encoder layer costs `1/C_1 = 0.25`. Moving one layer from the decoder
to the encoder is parameter-matched and 8x cheaper per layer, and would put the
depth where the context is — the depth analogue of §1b's "route where the
tokens are". (Measured below: it does not hold.)

Three arms follow: `r2` (`converter_width` 4->2), `deepenc` (one layer
decoder->encoder, parameter-matched), and `c8x2` (the same 16-token
meta-context split 8x2 instead of 4x4, which also takes the level-1 encoder
from `T/4` units to `T/8` and so **-40% of HierGen KV**).

### Hypotheses 2-5, measured — and then retracted

**The first version of this subsection was wrong.** It reported "-39% compute
and -40% KV at no measurable quality cost" from seven single-seed arms against
a "0.053-nat seed floor". Three independent defects, each of which alone sinks
the claim:

**(a) The train-CE column was a function of an unswept flag.** `compare_arms.py`
averaged the last `--tail` logged rows. Over `k in {3,5,8,10,15}`:

| arm | k=3 | k=5 | k=8 | k=10 | k=15 |
|---|---:|---:|---:|---:|---:|
| probe | 4.8071 | 4.8252 | 4.8793 | 4.9234 | 5.0859 |
| probe (seed 777) | 4.8533 | 4.8787 | 4.8922 | 4.9187 | 5.0797 |
| `r2` | 4.8363 | 4.8646 | 4.9253 | 4.9676 | 5.1069 |
| `norec` | 4.8506 | 4.8720 | 4.8829 | 4.9033 | 5.0678 |
| `c8x2` | 4.8910 | 4.9098 | 4.9162 | 4.9351 | 5.0867 |
| `deepenc` | 4.9025 | 4.9237 | 4.9353 | 4.9572 | 5.1283 |
| `skip` | 4.8603 | 4.8823 | 4.8954 | 4.9173 | 5.0872 |
| `all` | 4.8775 | 4.8926 | 4.8963 | 4.9183 | 5.0762 |
| **two-seed spread** | **0.0462** | **0.0535** | **0.0129** | **0.0046** | **0.0062** |

The "floor" moves by an order of magnitude and `r2` goes from best (+0.006 at
k=3) to worst (+0.047 at k=10). I published k=5. `compare_arms.py` now prints
every window and refuses to call the two-seed gap a floor.

**(b) `layer_positions_per_token` is not a compute metric.** It counts
positions and prices the LM head at zero. `flops_per_token` was no better — it
omitted the `(R_l + C_l)/C_l` decoder factor, i.e. it priced `converter_width`
at zero too. Both are fixed; with the head broken out:

| config | layer-pos/tok | stacks MFLOP | LM head MFLOP | head share |
|---|---:|---:|---:|---:|
| probe (4x4, R=4) | 8.375 | 26.5 | 76.5 | 74% |
| probe + all three | 5.125 | 16.2 | 76.5 | 82% |
| 250M v4 | 11.125 | 44.8 | 102.0 | 69% |
| 250M v5 | 6.875 | 27.6 | 102.0 | 78% |
| 8B v2 | — | 669.3 | 407.9 | 36% |

So "-39%" was the *stacks* column. Total forward FLOPs/token move **-10.1%** on
the probe and **-12.0%** at 250M. The head share falls with width, so the same
geometry change projects to **about -23%** at the 8B config (`d_token` 2048) —
the probe is the worst place in the family to measure it.

**And measured throughput sees none of it.** `scripts/probe_throughput.py`,
idle GPU, batch 16, seq 512, back-to-back on one 5060 Ti:

| config | tok/s | peak alloc |
|---|---:|---:|
| probe | 11,947 | 5.36 GiB |
| `r2` | 11,983 | 5.04 GiB |
| `c8x2` | 11,943 | 4.98 GiB |
| **`all`** | **12,073 (+1.1%)** | **4.80 GiB (-10.4%)** |

At 65M and batch 16 the step is launch-bound, so this measurement has no power
against a 10% FLOP change either — but it does refute a claim of 39%.

**(c) Both CE columns were English, and memorised.** `protocol_diag.load_tokens`
took `shards[0]` — `en_wikipedia-00000.bin` — for a 75%-Japanese model, at an
offset inside the prefix every dataloader stream reads first
(`dataset.py:_SourceCursor` reads a prefix and never wraps; `data/tokens_ja`
was built without `skip_files`, so it has no holdout). And the statistic was
`batches x batch x score_window` = **3,072 tokens**, whose own standard error
is the size of the effect.

### Re-measured on data no arm read

A run consumes at most 22.6M ja tokens and a ja shard holds 67.1M, so
everything past ~23M is held out in the model's own tokenisation, free, already
on disk. Scored at 262,144 tokens per arm, `--data-source ja_wikipedia`, two
disjoint slices:

Three disjoint slices — two Japanese, one English (`en_wikipedia` @40M of a
64.3M shard, where at most 7.6M were read) — as deltas against the two-seed
mean of each slice:

| arm | ja@48M | ja@60M | en@40M | **ctx nats, three slices** |
|---|---:|---:|---:|---:|
| probe | +0.032 | +0.035 | +0.002 | 0.323 / 0.327 / 0.325 |
| probe (seed 777) | -0.032 | -0.035 | -0.002 | 0.372 / 0.386 / 0.306 |
| `r2` | -0.009 | -0.044 | -0.001 | 0.311 / 0.337 / 0.289 |
| `norec` | -0.029 | +0.009 | -0.029 | 0.354 / 0.366 / 0.260 |
| `c8x2` | -0.006 | +0.001 | +0.010 | **0.210 / 0.211 / 0.199** |
| `deepenc` | **+0.025** | **+0.062** | **+0.017** | 0.279 / 0.276 / 0.293 |
| `skip` | -0.019 | +0.020 | -0.013 | 0.286 / 0.302 / 0.244 |
| `all` | -0.002 | +0.032 | -0.013 | 0.276 / 0.279 / 0.212 |

Two-seed spread: **0.0645**, **0.0689**, **0.0042**. The Japanese figure is
stable across two slices, so it is at least a reproducible quantity; the
English one is an order of magnitude tighter, which is not a better floor but
a luckier draw — English is 25% of the mixture and every run saw a similar
sliver of it, so two seeds landing 0.004 apart says nothing about the third.
Treating it as a floor would make four arms "significant" at a stroke, which is
precisely the trap the first version of this table fell into.

**Every arm except `deepenc` changes sign across the three slices.** `r2` is
-0.009, -0.044, -0.001; `all` is -0.002, +0.032, -0.013. Nothing about `r2`,
`norec`, `c8x2` or the combination is resolved by this experiment, in either
direction.

**The contamination, measured directly.** The same table at
`--data-offset 1024` — inside the trained prefix — gives a two-seed spread of
**0.2034**, three times the held-out value, with arm deltas up to 0.21. The
cause is compound: the prefix is memorised, *and* which shard got memorised is
a lottery. `dataset.py:_SourceCursor` seeds its shard-order RNG with
`hash(src.name)`, which Python salts per process, so the same `seed` draws a
different ja shard on different runs (verified: `[1,2,0]`, `[2,1,0]`, `[2,0,1]`
across three interpreters). Scoring shard 0's prefix therefore measures *which
shard a run happened to draw*. That is what the original table was ranking.

### What survives

* **`deepenc` is refuted** — worse than the two-seed mean on all three slices
  (+0.025, +0.062, +0.017), the only arm that never changes sign. Depth is not
  better spent in the level-1 encoder, and the "route depth where the context
  is" analogy to §1b does not hold.
* **`c8x2` measurably weakens long-range context use.** `ctx nats`
  **0.210 / 0.211 / 0.199** against baselines of 0.32-0.39 on the same slices —
  a ~0.11-nat drop reproduced across three independent held-out slices in two
  languages, far outside the ~0.05 two-seed spread on that statistic. `all`,
  which contains `c8x2`, is depressed the same way (0.276 / 0.279 / 0.212)
  while `r2` alone is not (0.311 / 0.337 / 0.289). The mechanism is direct:
  8x2 halves the number of level-1 units feeding the top stream (`T/8` instead
  of `T/4`). CE at 30M tokens and 512-token contexts cannot see the cost of
  that; a longer context would. **This is an argument against 8x2 that the CE
  table could not produce**, and it is the only architectural signal in the
  whole exercise that reproduces.
* **The KV number is untouched**: 0.617 -> 0.367 KiB/token is an accounting
  identity over cache tensors, confirmed by `gen.cache_bytes()`. Activation
  memory -10.4% is measured. Those are memory claims and they stand.

### The standing claim, and what would settle it

> Splitting the 16-token meta-context 8x2 instead of 4x4, with
> `converter_width` 2, removes **40% of HierGen KV bytes/token** and **10% of
> peak activation memory** at the same parameter count, and cuts **10-12% of
> forward FLOPs** at probe/250M widths (about 23% projected at 8B, where the
> LM head stops dominating) for **no measured throughput change at 65M**.
> **Its effect on quality is unmeasured**: every CE delta is inside the
> two-seed spread and changes sign across three held-out slices. The one
> reproducible effect is a **~0.11-nat reduction in measured long-range
> context use**, seen on all three slices, which argues against it.

`configs/base_small_v5.yaml` and `ScalaConfig.global_skip` are kept and
labelled unvalidated. Settling this needs what was deliberately not run here:
an 8-run variance decomposition of the unchanged config (init seed vs data
seed, ~3.7 h on two GPUs) to replace the two-seed gap with a real sigma, and a
250M geometry-only iso-compute A/B on GB10 (~10 GPU-h) — geometry-*only*
because `base_small_v5.yaml` also moves `chunk_cond_prob` 0.25 -> 0, a
fourth change worth 0.019 nats by §4e that the probe never tested.

`scripts/eval_ja.py` on a fresh HF Wikipedia slice was also *not* run: reaching
records >=700,000 costs ~90 min of streaming per invocation, and three disjoint
on-disk holdouts in two languages already agree that the answer is "unresolved"
— a fourth slice of the same corpus would not change it. The command is in the
plan if a BPC number comparable to the published 250M/8B rows is wanted:
`eval_ja.py --ckpt runs/<arm>/final --ppl-only --skip-hf-wiki --local-jsonl ...
--seq-len 512` after dumping the slice once (`--seq-len` must be 512; every
`runs/*/final/model_config.json` has `max_seq_len: 512` and `RotaryEmbedding`
would silently extrapolate rather than fail).

**The transferable lesson**, and it is the same one as §4l: every metric here
failed by being used one level past what it could support. `layer-positions` is
a good design intuition and a bad accounting unit. A `loss_token` tail is a
good divergence monitor and a bad endpoint estimator. A fixed 3,072-token
window of shard 0 is a good smoke test and a bad quality metric. Each was
believed because it was cheap, and all three happened to point the same way.


## 5. The PID routing controller did not survive contact with training

Isolated benchmark, 8 experts, same batch every step:

| controller | γ=0.002 | γ=0.01 | γ=0.05 |
|---|---|---|---|
| sign (DeepSeek-V3) | 0.0369 | 0.0781 | 0.2656 |
| PID (ZAYA1) | 0.0031 | 0.0013 | **0.0000** |

10–100x better, and *improving* with gain rather than degrading. It looked
decisive.

In real training it was worse at every checkpoint of a matched-pair A/B, and on
an 8B run it drove MaxVio from 0.45 to **32.75 in 80 steps** — routing collapse.

Two separate causes:

1. **The benchmark was stationary.** Feeding the same batch every step makes the
   load signal noiseless, which is the one regime where a derivative term is
   free. Real data makes the load signal noisy and the D term amplifies it.
2. **A scaling bug.** The error was multiplied by `n_routed_experts`, so at
   E=256 it reached 255 and a gain of 1e-2 moved the bias by 2.5 in a single
   step — saturating the clamp. Dividing by the mean instead keeps `|err| ≈ 1`
   for a 2x imbalance at any expert count.

The 8B run uses the sign rule at γ=1e-3 — the configuration that demonstrably
trained a model to in-mixture held-out ppl 49.84 at step 1400. The corrected
PID was then A/B'd at 250M rather than trusted.

**The corrected PID does hold up, but only where the load signal is dense.**
The v3 run (§1b) uses PID everywhere at γ=2e-3. Its level-1 stacks — which see
256 to 2048 rows per micro-batch — ended at MaxVio 0.5–2 across 1,525 steps.
Its level-2 encoder, at 64 rows, sat pinned at the theoretical maximum from
step 550 on. Same controller, same gain, same run: the difference is entirely
how many tokens the layer gets. That is consistent with cause (1) above, and it
means "which controller" is the wrong question for a starved layer — the fix is
to stop routing there at all.

**The general lesson:** a benchmark that holds the input fixed is not a
benchmark of a controller. It measures the controller's behaviour on a problem
it will never see.

---

## 6. Initialisation bugs are silent, so make them loud

Meta-device construction skips every initialiser, and the trainer replayed them
with a name-matching heuristic. Two v2 parameters fell through it:

* `res_attn` / `out_attn` / `res_ffn` / `out_ffn` — 1-D, no "norm" in the name,
  so they were **zeroed**. That multiplies the entire residual stream by zero.
  The symptom is `grad_norm = inf` and a loss that does not move — not an error.
* `start_latent` — same, zeroed.

Fixed by having modules expose `reset_parameters()` and adding a startup check
that **raises if any parameter is all zeros** (with an allow-list for the ones
that legitimately start at zero: the converter conv, the attention output gate,
attention sinks). That check caught the second bug automatically.

---

## 7. `torch.compile` and activation checkpointing do not compose here

Compiling a submodule *inside* a checkpointed block makes recompute take a
different path through the data-dependent MoE dispatch, and autograd rejects it:

```
recomputed metadata differs from saved metadata
saved:      torch.Size([384, 2048])
recomputed: torch.Size([16166, 192])
```

Compiling the *whole* checkpointed unit is fine — the loss slice is compiled and
checkpointed together with no trouble, and that is where nearly all the win is.
Block compilation is now disabled automatically when checkpointing is on.

---

## 8. Smaller operational traps that cost real time

* **PyYAML is YAML 1.1**: `4.0e6` parses as a *string*, not a float. `TrainConfig`
  coerces every field back to its declared type.
* **`--exclude=data` in tar excludes `scala/data/` too.** Anchor it: `./data`.
* **numpy ABI**: a `pip install` that upgrades numpy turns every
  `torch.from_numpy` into "Numpy is not available" — at the first training
  batch, after all the expensive setup. Pin numpy to whatever the image ships.
* **`tmux kill-session` does not kill the training process.** It kept the GPU
  and the next launch OOM'd. `pkill -f scripts/train.py` explicitly.
* **`--resume none` on the CLI arrives as Python `None`**, not the string, and
  `Path(None)` raises.
* **Checkpoints outlive the architecture.** Adding the PID buffers made every
  earlier checkpoint fail a strict load on 36 missing keys. Rule adopted:
  parameters must match exactly, buffers may be re-derived.
* **Streaming datasets have drifting schemas.** `swallow-code-v2` types
  `lint_report` as `struct<type,message>` in some shards and `null` in others;
  the Arrow reader aborts the stream partway through, and `select_columns()`
  does not help because the cast happens before projection.
* **A launcher that `docker rm -f`s a fixed container name kills the run
  already using it.** `gb10_run.sh` defaulted `NAME=photon` and removed that
  container on every start, so queueing a second job took out the first
  mid-training. Worse, the second job's "wait until the GPU is free" loop polled
  for `train.py` and saw none -- because it had just killed it -- so it started
  immediately and the collision was invisible. The launcher now refuses to
  remove a *running* container and says how to override; wait loops should poll
  for the container, not for a process name the launcher itself controls.
* **A harmony parser must accept a completion, not just a full turn.** The
  chat template's generation prompt ends with `<|start|>assistant`, so what the
  model returns begins *mid-segment*, at `<|channel|>`. The first version of
  `parse_harmony` anchored on `<|start|>` and returned zero segments for
  exactly the input a real harness passes it — while passing every unit test,
  because the tests were written from full transcripts. Tests built from
  training data do not exercise the inference path.
* **Tokenisation overshoot is bounded by batch size, not by the target.** The
  per-source target is only re-checked after a drain, so `batch_texts`
  documents is the granularity. Fine for web text at ~1 KB a document;
  catastrophic for agent transcripts — Laguna's trajectories average 425 KB
  (446 turns each), so one batch of 1,000 produced 270M tokens against a 5.4M
  target. Cap the batch by characters too (`batch_chars`, default 8M).
* **Verify the text column, not just reachability.** A dry-run that only checks
  the dataset loads will happily pass a source whose `text` field does not exist
  — and then tokenise nothing.

---

## 9. Choosing a distillation teacher is choosing a vocabulary

Logit-level KD needs the teacher and student to agree on what a token *is*.
That makes "pick the strongest teacher" a much narrower question than it looks,
because the student's vocabulary is not a free parameter either: it fixes the
size of the embedding and LM head (402.7M at 196,608 entries, a third of the
active budget) and how many tokens a Japanese document costs to train on.

Chars per token, measured 2026-07-28 on identical samples:

| tokenizer | vocab | JA prose | JA technical | EN | code |
|---|---:|---:|---:|---:|---:|
| **llm-jp v4** | 196,608 | **2.150** | **2.302** | 6.148 | 2.463 |
| llm-jp v3 | 99,574 | 2.098 | 2.033 | 5.724 | 2.418 |
| Qwen 3.5 (Ornith-1.0) | 248,077 | 1.870 | 1.937 | 6.148 | 2.830 |
| Qwen 3 | 151,669 | 1.458 | 1.402 | 6.148 | 3.023 |
| poolside Laguna S 2.1 | 100,352 | 1.089 | 1.008 | 6.148 | 2.830 |

poolside Laguna S 2.1 is a frontier agentic coder (118B/8B-active, 78.5%
SWE-Bench Multilingual) and adopting its tokenizer would have unlocked logit KD
from it — at **2.3x the tokens for the same Japanese text**. On a budget where
tokens were the binding constraint, that is disqualifying. Ornith-1.0's Qwen-3.5
vocabulary is 15% worse than llm-jp v4 on Japanese *and* 26% larger, so its LM
head would cost more for less.

The conclusion is not "llm-jp-4 is the best model". It is that the white-box
teacher has to be llm-jp-4 because everything else costs more than it returns,
and the frontier models earn their place through **sequence-level** distillation
instead, where the tokenizer is irrelevant because you are training on text.

### 9b. Every frontier agentic corpus is 0% Japanese

Japanese character fraction over the first 30 records:

```
nvidia/Nemotron-SFT-Agentic-v2             0.0%   (DeepSeek-V3.2)
greghavens/kimi-k3-coding-...              0.0%   (Kimi K3)
zake7749/deepseek-v4-pro-agent-...         0.0%   (DeepSeek-V4-pro)
mgoin/Laguna-S-2.1-trajectories            0.0%   (Laguna S 2.1)
tokyotech-llm/Swallow-Nemotron-PT-v1       0.0%   <- even the Japanese lab's
hotchpotch/japanese-qa-reasoning-100k     71.5%
```

Mid-training on "the best available agentic data" trades away the only thing a
Japanese model is for. This is not hypothetical — the 250M vast.ai run used a
32% math cooldown and its Japanese samples came out visibly contaminated with
formulae and English. `configs/data_ja_v3_mid.yaml` caps English agentic at 0.18
and carries 0.74 Japanese; the reasoning is in `docs/distillation.md` §4.

## 10. Quantisation has to know which axis the GEMM reduces over

Block-scaled 4-bit formats give sixteen (NVFP4) or thirty-two (MXFP4) adjacent
values one shared scale. "Adjacent" has to mean *along the reduction axis*, and
in this model that axis is not the same for the two kinds of weight:

```
nn.Linear.weight   [out, in]                 reduction = dim -1
MoE w_gate_up      [n_experts, d, 2*inter]   reduction = dim -2
MoE w_down         [n_experts, inter, d]     reduction = dim -2
```

(`scala/model/moe.py`: `din, dout = w.shape[-2], w.shape[-1]` feeding
`torch.bmm`.) Blocking along the last dim for the 3-D expert stacks groups
sixteen values that share nothing, and the block scale stops meaning anything —
silently, since the weights still load and the model still runs. The expert
stacks are most of the model, so this is not a corner case.

Measured weight RMS relative error on the real export:

| format | bits/weight | error | size |
|---|---:|---:|---:|
| FP8 E4M3, per-channel | 8 | **0.026** | 1.9x |
| NVFP4 (E2M1, block 16, E4M3 scale) | 4.5 | 0.096 | 3.3x |
| NVFP4 + level-2 encoder at FP8 | ~4.6 | 0.093 | 3.2x |
| MXFP4 (E2M1, block 32, E8M0 scale) | 4.25 | 0.117 | 3.4x |

MXFP4's coarser block and power-of-two scale cost ~20% more error than NVFP4 for
~5% less space; the reason to want it is that the toolchain understands it
outside Blackwell.

Three things are never quantised, and the reasons are structural rather than
conventional:

* **Routers.** Top-k over 192 experts is decided by margins routinely smaller
  than 4-bit resolution, and being wrong does not degrade the layer gracefully —
  it runs a different expert.
* **Embedding and LM head.** 402.7M parameters feeding a 196,608-way softmax,
  where the error lands directly on the logits.
* **Everything 1-D.** Rounding a per-channel gain multiplies a whole channel.

PHOTON adds a fourth consideration that generic pipelines have no way to know
about: the level-2 encoder runs once per sixteen tokens, so keeping it at 8 bits
costs almost nothing in FLOPs, but its output conditions the decoding of all
sixteen, and weight absorption for the latent KV cache *multiplies* its MLA
matrices together — turning additive quantisation error into multiplicative
error. `--policy hierarchy` protects exactly that stack.

### 10b. MaxVio is not comparable across stacks without normalising by E−1

During the 8B v3 run `moe/maxvio_max` climbed steadily — 6.97 at step 180, then
10.75, 15.06, 16.38, 17.88, 18.88 by step 380 — and against the ~20 threshold
written into the config that looked like the level-2 encoder's router going the
way v3-small's did. It was not, and the mistake was in the comparison.

MaxVio is `(max_load − mean_load) / mean_load`, so its ceiling is **E−1**: the
value when every token in the layer routes to one expert. The 250M runs used
E=16 there, and the collapse showed up as MaxVio pinned at exactly **15.00** —
the mathematical maximum, not merely a large number. The 8B config spreads
48/96/160/192 experts across four stacks, and `moe/maxvio_max` is the max over
all of them, so 18.88 on a 192-expert stack is 9.9% of the way to collapse while
15.00 on a 16-expert stack is all of the way there. Same units, different
meaning.

The diagnostic that actually settles it does not depend on expert count at all.
`scripts/router_health.py` reads the auxiliary-loss-free controller's
`expert_bias` out of a checkpoint and reports how many experts sit at the clamp.
A collapsing router saturates: the controller wants to push further than
`bias_clip` allows and cannot. At step 300 of the 8B run:

```
stack          layers  experts   bias sd  at -clip  at +clip     range
L1.decoder          5      192     0.006      0.0%      0.0%   -0.06..0.01
L1.encoder         11      160     0.015      0.0%      0.0%   -0.18..0.05
L2.decoder          3       96     0.043      0.0%      0.0%   -0.20..0.08
L2.encoder         10       48     0.025      0.0%      0.0%   -0.04..0.09
```

Clamp is ±2.0 and the largest bias anywhere is 0.20 — the controller is using a
tenth of its authority and no expert is pinned. Nothing is collapsing; the
routers are simply specialising, which is what they are for.

Two rules follow. Report MaxVio as a fraction of E−1 when comparing anything,
and prefer clamp saturation as the collapse test, because it needs no
normalisation and answers the question directly.

### 1e. The in-training `[eval]` is not held out either

Separate from §1c, and found the same way — by distrusting a number that moved
more than it should have. During the 8B v3 run:

```
[eval] step  400: ce=4.4505 ppl=85.67
[eval] step  800: ce=3.9310 ppl=50.96
[eval] step 1200: ce=3.1252 ppl=22.76   <- mid-training began at step 1168
```

A 55% perplexity drop in 400 steps, straddling exactly the point where the data
mixture changed. `Trainer.evaluate()` calls `self.next_batch()`, which pulls
from `self.data_iter` — the *training* iterator:

```python
def evaluate(self):
    for _ in range(self.cfg.eval_batches):
        b = self.next_batch()          # <- the training stream
```

The batches are at least fresh (not yet trained on), so this is not measuring
memorisation. But it is drawn from the training mixture, so when the mixture
switches to conversational SFT data — which is far more predictable than web
text — the number falls for reasons that have nothing to do with the model
getting better.

So `[eval]` is a running loss on the training distribution. It is useful for
spotting divergence within a phase and **worthless across a phase boundary**.
Every number that goes in a model card comes from `scripts/eval_ja.py` with
`--ppl-skip-*`, run after training against corpora the run did not consume.

The general rule this and §1c share: when a metric improves by more than the
training curve can justify, suspect the metric before believing it.

## 11. Perplexity cannot cross a tokenizer boundary — and the 8B lost

The 8B v3 run is the first SCALA model on llm-jp-tokenizer v4 (196,608);
everything before it used v3 (99,574). Held-out Japanese Wikipedia perplexity
came out at 69.00 against the 250M v3-900m's 67.60, which invites the reading
"the 8B is no better than a model 32x smaller". That reading is right, but not
for the reason the raw numbers suggest, and getting there needs the right unit.

Perplexity is per *token*. A tokenizer that packs more characters into each
token makes every token harder to predict, so per-token perplexity rises for
identical underlying quality. The invariant is bits per character:

    BPC = nll / (ln2 * chars_per_token)

Measured on the *exact* held-out slice the evaluation used (400 Japanese
Wikipedia articles beyond record 700,000, 516,716 characters):

| run | tokens | ja-wiki ppl | chars/token | **BPC** |
|---|---:|---:|---:|---:|
| 250M v4 | 200M | 51.61 | 1.8320 | **3.1056** |
| 250M v3-900m | 900M | 67.60 | 1.8320 | **3.3181** |
| 8B v3 | 344M | 69.00 | 1.7915 | **3.4098** |

**The 8B is the worst of the three.** Scaling parameters 32x while scaling
tokens 1.7x does not buy anything — 0.04 tokens per parameter against the 250M
v4's 0.8 — and no amount of architectural care compensates for that. This is
the clearest measurement in the project of where the binding constraint is, and
it is not the architecture.

### 11b. A correction: v4 does not compress Japanese better

The v3 configs and README justify the tokenizer switch partly on compression,
citing 2.302 chars/token for llm-jp v4 against 2.033 for v3 on "technical
Japanese prose". That figure came from three hand-written sentences, and it does
not replicate. On real Japanese Wikipedia:

    llm-jp v4   1.7915 chars/token
    llm-jp v3   1.8320 chars/token

v4 is 2.2% *worse* on this corpus. The measurement was real but the sample was
too small and too unrepresentative to generalise from, and it was then repeated
as established fact in three places.

The tokenizer switch is still correct, for the reasons that do not depend on
compression: logit-level distillation requires the teacher's exact vocabulary,
and v4 carries the openai-harmony control tokens (`<|channel|>`, `<|call|>`,
`<|constrain|>`) without which there is no agent format. Those are the
justification. The compression claim should be dropped.

Rule: a tokenizer comparison needs a corpus, not a sample, and it belongs in the
same units as whatever decision it is supporting.

---

## 12. ROCm: the CUDA "compute capability" of an MI210 is `(9, 0)`

Measured on ACRi as006 — 1x AMD Instinct MI210 (gfx90a), ROCm 6.1.3, torch
2.8.0+rocm6.4.

The MoE died on the first forward pass:

    RuntimeError: grouped gemm is not supported on ROCM
      scala/model/moe.py:344 in forward_aligned

which is surprising, because §1 already established that `torch._grouped_mm` is
gated to Hopper and the code gates on that:

```python
return torch.cuda.get_device_capability(0) == (9, 0)     # the old test
```

**On ROCm that call does not return a CUDA compute capability.** Torch derives
it from the gfx architecture string, so gfx90a reports exactly `(9, 0)` — the
same tuple an H100 reports, and the one the Hopper gate is looking for. The
gate passed, the aligned path ran, and the aligned path is the one place with
no `try/except`, because on CUDA the 16-row alignment is what makes the call
safe. So a HIP build fails in the single spot engineered on the assumption
that the arch check already succeeded.

Two things follow, and the second is the general one:

1. Reject HIP builds explicitly (`torch.version.hip is not None`).
2. **Probe the kernel instead of inferring it from a version number.**
   `_kernel_supports_grouped_mm()` now runs a real 32-row, 2-expert call with
   16-aligned offsets and believes the result. Aligned offsets cannot trip the
   device-side assert that ragged ones cause, so the probe is safe on every
   backend — which is the property that lets it replace the inference.

This is the third distinct way this one kernel has been mis-gated (§1 lists a
`>=` that should have been `==`, and a dtype mismatch that silently disabled
it). The pattern across all three: capability metadata was treated as a proxy
for "will this call work", and it is not one.

With the padded batched-GEMM path selected correctly, the 250M v4 trains at
**24.7-25.5K tok/s** on the MI210 at micro-batch 32, seq 1024 — against 66K on
a rented RTX 5090. The card measures 120.8 bf16 TFLOPS on an 8192³ matmul
(67% of its 181 catalogue figure, normal for rocBLAS), so the throughput ratio
tracks the FLOPS ratio; there is no ROCm-specific cliff beyond losing the fused
grouped GEMM, which was never available on this class of card anyway.

### 12b. An offline machine changes what "held out" can mean

The ACRi servers have **no route to the internet at all** — not to the Hub, and
`pip download` fails with ENETUNREACH too. Every data config in this repo
streams from the Hub, and `eval_ja.py` read `wikimedia/wikipedia` directly, so
neither worked. What the machine does have is the whole llm-jp corpus v3 on its
NFS home (2.0 TB), which is why `prepare_data.py` grew `reader: local_jsonl`
and `eval_ja.py` grew `--local-jsonl`.

The interesting part is the held-out slice. §1c is about an "independent"
evaluation that was reading the same dump from the same end as the tokenizer —
i.e. measuring training data — and the fix there was `--ppl-skip-ja`, a record
offset into a Hub stream. Offline there is no stream to offset into, so the
split moves up a level: `skip_files: 2` on the `ja_wiki` source means training
never opens `train_0.jsonl.gz` / `train_1.jsonl.gz`, and the evaluation opens
exactly those two. A file-level split is coarser than a record offset but it is
enforced by the reader rather than by two independently-set integers agreeing,
which is what failed the first time.

`evaluate_perplexity` now returns **bits per character** alongside perplexity,
and `eval_ja.py` prints BPC first. §11 had to recover chars/token by hand after
the fact to discover the 8B had lost; measuring it inside the harness, on the
corpus actually being evaluated, is the version of that lesson that cannot be
skipped.

### 12c. The ACRi *gateway* has internet and mounts the same home

as006 itself is genuinely sealed — DNS resolves (`huggingface.co` →
3.164.110.114) but nothing routes: IPv4 HTTPS times out, there is no IPv6
path, and there is no proxy. That much §12b had right.

What it missed is that **`gw.acri.c.titech.ac.jp` (fserv2) is a normal
internet-connected host, and `/home/$USER` on it is the same NFS export
(`172.16.2.13:/ehome/$USER`) that as006 mounts.** So the way to get a file
onto as006 is not to push it through the SSH tunnel — it is to `ssh acri-gw`,
download into `~/`, and read it from as006 as a local file.

The difference is not marginal:

    Windows -> as006 through the ProxyJump tunnel      ~50 KB/s
    gateway -> NFS home, visible to as006 immediately  125 MB/s

A 480 MB checkpoint took 20 seconds instead of the ~3 hours the tunnel was on
course for. Anything from the Hub or PyPI should be fetched this way.

### 12d. Re-measure the baselines; do not trust a published number across slices

The point of carrying held-out corpora onto the machine was to compare this
run against the published 250M results. Running those published checkpoints on
the carried slices shows why the comparison needed the checkpoints and not just
the numbers:

| | published | re-measured here |
|---|---|---|
| v4 aozora ppl | 365.49 | **363.01** |
| v3-900m aozora ppl | 237.10 | **240.97** |
| v4 ja-wiki ppl | 51.61 | **61.44** |
| v3-900m ja-wiki ppl | 67.60 | **82.23** |

Aozora reproduces to ~1% — that slice is effectively the same text. Japanese
Wikipedia is uniformly ~19% higher, at essentially identical compression
(1.8384 chars/token here against the published 1.8320), which says the slice is
the same *kind* of text taken from a different place in the dump rather than a
different measurement. The published protocol is `skip=700000` plus
`2 x --ppl-sequences` articles, and reconstructing it from the outside does not
land on the same articles.

The model-to-model gap survives intact: v4 beats v3-900m on Japanese Wikipedia
by 6.6% BPC here against 6.4% published, and loses to it on Aozora both times.

So: **an absolute BPC is only comparable within one slice.** The reference row
for the ACRi run is the re-measured one — v4 at 3.2317 and v3-900m at 3.4604 —
and quoting 3.1056 against it would have understated the run by 4%.

### 12e. Re-measuring the 8B partially reverses §11

With the gateway making checkpoints cheap to fetch, all three published models
were run on the same three carried slices. §11's headline — "the 8B lost to
both 250M models on BPC" — does not survive intact.

| model | tokens | ja_wiki BPC | en_wiki BPC | aozora BPC |
|---|---:|---:|---:|---:|
| 250M v4 | 200M | **3.2317** | **1.5078** | 5.1759 |
| 250M v3-900m | 900M | 3.4604 | 1.5764 | 4.8161 |
| 8B v3 | 344M | 3.4082 | 1.7118 | *3.9301* |

Two things changed.

**On Japanese Wikipedia the 8B now beats v3-900m** (3.4082 against 3.4604)
rather than losing to it. Published, the same pair was 3.4098 against 3.3181 —
the 8B behind by 0.9%. The gap was always small, and it changes sign when the
held-out slice changes. What survives is the comparison that was never close:
**v4 at 200M tokens beats the 8B at 344M**, by 5.5% here and 9.8% published.
The claim worth keeping from §11 is that one, not the three-way ranking.

**The Aozora column is not usable for the 8B.** `configs/data_ja_v3.yaml`
carries `ja_aozora` at weight 0.02 with the filter bug fixed, so the 8B is the
only one of the three that trained on this text (~6M tokens). v3-900m ran
`data_ja_v2.yaml`, where that source tokenised to nothing for the whole run,
and v4 trained on Wikipedia alone. So the 8B's 3.9301 is measured on its own
training distribution and the *3.93* above is italicised for that reason, not
because it is uncertain. Aozora remains a clean neutral corpus for v4,
v3-900m and the ACRi run — all three of which have never seen it.

Note also that the 8B's ja_wiki BPC reproduces its published value almost
exactly (3.4082 against 3.4098) while v4's does not (3.2317 against 3.1056).
That rules out a uniform slice-difficulty offset: the slice is harder for v4
specifically. Different models genuinely rank differently on different held-out
draws from the same dump, which is the same lesson as §12d with a sharper edge
— **a 1-3% BPC gap is not a result unless both models were scored on the same
text.**

## 13. What actually costs the throughput on MI210 — and why none of it is worth taking mid-run

Profiled by SIGSTOPping the live training job, measuring on the idle card, and
SIGCONTing it. Kernel-only self-CUDA time, 250M v4, batch 32 x seq 1024,
`compile: true`, 3 steps:

| | share |
|---|---:|
| rocBLAS GEMM (Tensile) | 30.9% |
| elementwise (Inductor triton) | 27.4% |
| CE / log_softmax (Inductor triton) | 21.1% |
| MoE dispatch (scatter/index) | 8.0% |
| reduce / norm | 6.6% |
| convolution (the converter's depthwise conv) | 2.1% |
| optimizer | 1.7% |
| attention | 0.7% |

Only 31% is GEMM. This is a memory-bound model, and **attention is 0.7%** —
another reminder that §1's conclusion (dispatch, not attention) understates it
at this scale.

A first pass at this table was wrong in a way worth recording: summing
`key_averages()` double-counts, because it returns both the `aten::` op and the
rocBLAS kernel it dispatched, each carrying the same device time. That put
`aten::mm` at 17% and `Cijk_*` at 8% for one piece of work and inflated the
total by 67% (5581 ms against the true 3333 ms). Filter to device kernels.

### The levers, measured

| change | tok/s | vs baseline |
|---|---:|---:|
| baseline (`compile: true`, chunk 2048, mbs 32) | 35.8K | — |
| loss chunk 1024 | 33.9K | −5% |
| loss chunk 4096 | 34.4K | −4% |
| loss chunk 8192 | 36.0K | +0.6% |
| micro-batch 64 | ~36K | 0% |
| `compile_mode: max-autotune-no-cudagraphs` | **37.0K** | **+3.4%** |
| **MTP disabled** | **49.0K** | **+37%** |

`loss_chunk_tokens = 2048` was tuned on GB10 and is still right here — the
sweep is flat between 2048 and 8192 and falls off below. Block compilation
(22 blocks, mode default) buys nothing: the production job compiles them and
matches a profile run that compiled only the loss.

**Multi-token prediction costs 27% of wall clock.** That is by far the largest
number in this table, and it is worth stating plainly: MTP is an auxiliary loss
whose weight anneals 0.3 -> 0.1, and this project has **never ablated its
effect on main-model quality**. Spending 27% of a fixed compute budget on an
unmeasured auxiliary is a bad trade if the auxiliary is worth less than the 37%
more tokens the same compute would buy. That is the next A/B to run, not a
change to make on a hunch.

### None of it is worth taking on the run in progress

At 25% complete with 11.5 h left, switching to MTP-off would take the remaining
1.5B tokens from 11.5 h to 8.5 h. But it would make the run half one objective
and half another, and it would stop being comparable to v4 and v3-900m, which
both train with MTP. A clean MTP-off run from scratch is 2B tokens at 49K =
11.3 h — the *same* wall clock as simply letting this one finish. There is no
version of the switch that both saves time and leaves a usable experiment.

max-autotune is +3.4%, worth ~23 min over the remaining run, against ~25 min
lost restarting from the last checkpoint. It nets out negative. It should be
the default for the *next* run, not a change to this one.

The general rule this is an instance of: **a mid-run optimisation has to beat
the restart cost plus the loss of comparability, and a 3% one never does.**

### 13b. Train CE rising while held-out BPC falls is the scheduled sampling, not divergence

Around 40% into the ACRi run the training cross-entropy turned around and
climbed for 800 steps while every held-out number kept improving:

| step | tokens | train CE | self_cond p | loss_rec | ja-wiki BPC |
|---:|---:|---:|---:|---:|---:|
| 1620 | 417M | 3.6093 | 0.106 | 0.225 | 3.4218 |
| 2020 | 522M | **3.5543** | 0.132 | 0.222 | 3.3543 |
| 2420 | 626M | 3.6643 | 0.159 | 0.219 | 3.3216 |
| 2820 | 731M | 3.7486 | 0.185 | 0.213 | 3.2904 |

The cause is `self_cond_prob`, which ramps linearly to 0.25 over the first
`self_cond_ramp_frac` (0.5) of the token budget. With probability p the upper
decoders are fed their own reconstruction `X̂` instead of the true encoder
state `X`, so **the training task itself gets harder as the run proceeds**.
The held-out evaluation is teacher-forced HierGen and sees none of that, which
is why the two curves separate.

`loss_rec` falling throughout is the confirmation: that is the quantity
scheduled sampling exists to improve, and it improves monotonically while the
token CE it is paid for with degrades.

Two consequences worth carrying:

* **A rising train loss is only evidence of a problem if nothing in the recipe
  is scheduled to make the task harder.** Any log table used to judge a run
  needs the scheduled quantities in it — `self_cond`, `mtp_w`, `lr`, the
  mid-training switch — or the reader is guessing.
* It gives a **falsifiable prediction**: `self_cond` saturates at 0.25 at 1B
  tokens, so train CE should resume falling after that point. If it does not,
  the benign explanation was wrong and something else is going on.

### 13c. A reservation boundary can hand you a CPU run that looks like a GPU run

The as006 reservation ended at 05:45 mid-run and a new one started at 07:23.
Both halves of that are worth recording.

**What worked.** `/scratch` survived the boundary completely — the 13 GB token
set, every checkpoint including one *newer* than the last NFS mirror
(step-00004000 against the mirror's step-00003800), and the 20-row BPC curve.
Nothing was lost. The 30-minute mirror to the NFS home turned out to be
insurance that was not needed, which is the correct outcome for insurance, and
it is still right to keep: `/scratch` is documented as resettable and this run
is the only evidence either way.

**What failed.** ACRi powers the server on at the reservation start time, and
ROCm is not up for the first minutes. A resume issued ~60 s after boot found
`torch.cuda.is_available() == False`, built the model on CPU, and **started
training**. The only symptom was a warning nine lines up in the log:

    UserWarning: 'pin_memory' argument is set as true but no accelerator is
    found, then device pinned memory won't be used.

Everything else looked healthy: the resume banner printed the right step and
token count, the process ran, the log file had recent timestamps. `rocm-smi`
showing **GPU use 0% / VRAM 0%** was the only unambiguous signal. Left alone it
would have produced a run roughly a hundred times slower that still emitted
plausible-looking step lines.

`acri_supervise.sh` now gates every launch on a real
`torch.cuda.is_available()` probe and waits up to 30 minutes rather than
starting a CPU run. The general shape: **a fallback that silently degrades by
100x is worse than a crash**, and the place to catch it is before launch, not
in the logs afterwards.

(The §13b prediction also came true across the restart: `self_cond` saturated
at 0.25 around 1B tokens and train CE resumed falling — 3.6893 at step 3920,
3.4816 at step 4060.)

---

## 14. The token axis, measured: 10x the tokens buys what 32x the parameters did not

The ACRi run finished 2.0B tokens on 2026-07-29 — the same 250M v4 architecture
that §11 measured at 200M tokens, given ten times the budget on a reserved
MI210 rather than a rented hour. All four models scored on the same carried
slices (§12d), so these numbers are comparable to each other and to nothing
published earlier.

| model | total / active | tokens | ja-wiki | en-wiki | aozora |
|---|---|---:|---:|---:|---:|
| 250M v4 | 0.252B / 0.100B | 200M | 3.2317 | **1.5078** | 5.1759 |
| 250M v3-900m | 0.252B / 0.100B | 900M | 3.4604 | 1.5764 | 4.8161 |
| 8B v3 | 8.06B / 1.21B | 344M | 3.4082 | 1.7118 | *3.9301* |
| **250M v4 @ 2B** | 0.252B / 0.100B | **2.0B** | **3.0308** | 1.5428 | **4.6585** |

(*Aozora is in the 8B's training mixture and in no one else's — §12e.*)

**§11 said scaling parameters 32x while scaling tokens 1.7x buys nothing. The
converse holds: scaling tokens 10x at fixed parameters buys 6.2% BPC.** The
250M at 2B tokens beats the 8B at 344M by 11% on Japanese Wikipedia while
using 32x fewer parameters and about a fifth of the training compute.

The curve, on the held-out slice, is close to log-linear in tokens right up to
the cooldown:

| tokens | ja-wiki BPC |
|---:|---:|
| 44M | 4.7857 |
| 254M | 3.5651 |
| 464M | 3.3826 |
| 883M | 3.2490 |
| 1.30B | 3.1865 |
| 1.51B | 3.1429 ← cooldown starts |
| 1.72B | 3.0781 ← mid-training blend starts |
| **2.0B** | **3.0308** |

It had not flattened. Whatever the next binding constraint is, 2B tokens at
250M parameters is not it.

**The one loss is English**, 1.5428 against v4's 1.5078. That is the mixture,
not the run: the ACRi machine's copy of llm-jp corpus v3 has an empty
`en/en_wiki` and only the `c4` subset of `en_dolma`, so the English share is
raw C4 where v3 used the Nemotron-CC high-quality tier. There is no math
corpus on the machine at all. Both are documented in
`configs/data_ja_acri.yaml` rather than being quietly absorbed.

### 14b. The token axis does not rescue RecGen — but this run tested only one of the two knobs

The standing negative result was measured at four model *sizes*. This run adds
the first point on the *token* axis: 10x the budget, with scheduled sampling at
its full 0.25 for the whole second half. Scored against the teacher-forced
training forward with the §4g fix in place, on Japanese data:

| protocol | content | up | agree (ja_wiki / ja_cc) | KL | cos(X_top) |
|---|---|---|---:|---:|---:|
| hiergen | encoder | encoder | 93.8 / 94.0% | 0.006 | 1.0000 |
| xhat_content | xhat | encoder | 44.5 / 48.7% | 1.17 | 0.9999 |
| content_only | chunker | encoder | 31.8 / 35.1% | 3.26 | 0.9999 |
| up_only | encoder | chunker | 50.2 / 49.8% | 1.28 | **0.482** |
| chunkgen | chunker | chunker | 8.4 / 10.5% | 4.82 | 0.482 |
| **recgen** | xhat | xhat | **6.5 / 8.8%** | 5.76 | 0.454 |

**RecGen is 6.5–8.8%. Ten times the tokens moved it nowhere.** For the recipe
this run used, the token axis is as dead as the size axis.

**What this does not show.** These numbers must not be read against the 31–54%
RecGen or 71.1% ChunkGen band in `docs/recgen_report.ja.md`. Those arms train
`chunk_cond_prob` — the *chunker* substitution. This run trained
`self_cond_prob_end: 0.25`, the *X̂* substitution. Different knobs, correcting
different errors, and the diagnostic shows the model behaving exactly as that
implies: `xhat_content` (44.5–48.7%) beats `content_only` (31.8–35.1%), i.e.
this model is better at the substitution it was trained on and worse at the one
it was not. Its ChunkGen of 8.4–10.5% against the `fix` arm's 71.1% is not a
contradiction; it is an untrained path.

So the honest scope: **a 10x token budget does not make X̂-substitution work.**
Whether it would help the chunker line — the one that reached 71.1% — is
untested, and that is the experiment worth running next, not another RecGen
tuning round.

§4h's attribution reproduces cleanly at this scale. `up_only` drops
`cos(X_top)` to **0.48** — substituting into the upward path destroys the
top-level state itself, and no training term covers that half. `content_only`
leaves `cos(X_top)` at 0.9999 and still loses two thirds of the agreement. Two
different failures; only one of them is trained against.

### 14c. Greedy agreement is a bf16 tie-breaking metric; KL is the real one

HierGen scored 93.8–94.0% here against the 97.6–98.7% §4g calls the bf16 noise
floor, and it is the one number theory says must be ~100%, so it was worth
chasing. Sequence length was not it (95.6 / 93.7 / 94.2% at 256 / 512 / 1024).
**Precision was:**

    bf16, seq 512    agree  93.7%   KL 0.005
    fp32, seq 512    agree 100.0%   KL 0.000

HierGen *is* the training forward for this checkpoint, exactly. Every point of
the missing 6% is bf16 rounding flipping an argmax between two near-tied
logits — which is also why the KL barely moves (0.005) while the agreement
moves six points.

The implication is not about this model. **Greedy agreement is a discontinuous
metric and a converged model makes it worse**: the better the model fits real
text, the more positions sit near a top-2 tie, and the more of them bf16 can
flip. §4g's reference checkpoint was at token CE 4.46 and scored 97.6%; this
one is at 3.39 and scores 93.7% on identical, correct code. Reading that as
"HierGen got worse" would be exactly backwards.

So: quote KL for exactness claims, and treat agreement as the human-readable
companion, not the test. Where agreement is the quantity of interest — the
protocol comparisons above, where the gaps are 40–90 points — the bf16 noise is
irrelevant. Where it is a few points, it is measuring the dtype.

---

## 15. Every stack but one has a bounded span, and that one wanted NoPE

This is the Celeritas section; the design is `docs/celeritas.md`, the numbers
below are the ones that survived measurement.

### 15a. Nobody had measured what happens past the training length

Every CE figure in this file is taken at the training shape. That is the one
place where a positional scheme cannot be distinguished from another one: RoPE
is exact inside the range its table was trained on, so §4m's arms were graded
on the only axis where the question does not arise.

`scripts/length_diag.py` takes the missing measurement. Design detail that
makes it a measurement rather than a slice comparison: **the scored tokens are
identical at every context length**. A sequence is loaded as
`tokens[end - T : end]` and only the last `--score-window` positions are
scored, so `end` is fixed and only the history in front of it grows. Loading
is one sequence at a time on purpose — `load_tokens` returns
`batch * seq_len` *contiguous* tokens, so a batched read would move every
sequence but the last when `T` changes, and the scored text moving with the
context length is exactly the confound this is built to avoid.

All arms trained at 512 tokens, ja@48M held out, 16 sequences, last 256
positions scored. **CE minus CE at the training length; positive is
degradation:**

| context | ×train | PHOTON | FENESTRA | `nope` only | `stream` only | **Celeritas** |
|---|---:|---:|---:|---:|---:|---:|
| 1024 | 2× | +0.0952 | +0.0049 | +0.0052 | 0.0000 | **−0.0001** |
| 2048 | 4× | +0.1739 | +0.0102 | +0.0105 | 0.0000 | **−0.0001** |
| 4096 | 8× | +0.2997 | +0.0194 | +0.0166 | 0.0000 | **−0.0001** |
| 8192 | 16× | **+0.4019** | **+0.0262** | **+0.0254** | **−0.0000** | **−0.0001** |

**PHOTON loses 0.40 nats at 16× its training length**, and FENESTRA loses
0.026. The 0.376-nat difference between them is `encoder_window`, which was
introduced for *cache* reasons: bounding the level-1 encoder's span so `recgen`
could be exact also bounded its RoPE offsets, and RoPE is exactly relative, so
that stack's function stopped depending on `T` at all. PHOTON's level-1
encoder is global — RoPE over `T/4` = 2048 units against 128 trained — and it
is the larger of its two extrapolating stacks. **A bounded span is a
length-generalisation mechanism, not only a memory one**, and FENESTRA got that
for free without claiming it.

### 15b. The remaining 0.026 is not the top encoder's RoPE — NoPE does not fix it

This is the part I expected to go the other way. `celeritas_probe_nope.yaml` is
FENESTRA's geometry with **only** the positional change: NoPE on the top
encoder, no decoupled RoPE key, a learned sink. It is the arm designed to
isolate exactly the residual +0.0262.

**It reads +0.0254.** Within noise of the RoPE model it replaced, at every
context length. And the complementary arm — `decoder_stream` with the top
encoder's RoPE left alone — is flat. The 2x2 is perfectly factorial with no
interaction, so the attribution is unambiguous: **the decoder change does all of
it and the positional change does none of it.**

Train CE and held-out CE agree that the positional change did nothing at the
training shape either: 4.7040 against FENESTRA's 4.7028 at step 440, held-out
-0.1114 / -0.1590 / -0.1083 against -0.1115 / -0.1579 / -0.1112, and the two
trajectories sit on top of each other from step 100 onward.

So the honest conclusion is the one the experiment gives, not the one it was
built to confirm: **at this scale, moving the global stack from RoPE to NoPE
buys no measurable length robustness.** The claim it *does* support is
structural and narrower — no stack evaluates a position outside its trained
range — and the things that follow from it are real and independently
verifiable: the decoupled RoPE key is deleted (−20% on the only cache that
grows with `T`), the learned sink becomes available on the absorbed path, and
`rope_theta` / YaRN / a context-extension phase stop being decisions for that
stack. None of those are the length curve.

Why doesn't NoPE help? Because "no position table to extrapolate" is not the
same as "length-invariant". A NoPE causal stack still learns its implicit
notion of position from the length distribution it saw, and attention over 512
units is out of distribution when it was trained on 32 — the dispersion
problem the sink was supposed to cover. Measured: the sink does not cover it
here.

**What does remove the residual is the streaming decoder** — Celeritas is flat
to four decimal places, and the only difference from the `nope` arm is
`decoder_stream`. That was not predicted; the mechanism is settled in 15d.

### 15b-2. …and the confound control survives, in a weaker form

These shards concatenate documents, so a longer context is also one more likely
to span an article boundary. That inflates every row of every column equally,
since the scored tokens are identical — which means **any arm that stays flat
puts an upper bound on it**. Celeritas reads −0.0001, so the document-boundary
effect is at most 0.0001 nats and every number in the table above is real model
degradation rather than a harder slice.

What the control can **no longer** be used for is attributing that degradation
to positional extrapolation specifically. The original reading here was "a
model with nothing to extrapolate measures the confound, so the rest is
extrapolation" — and `nope` breaks it, because that arm also has nothing to
extrapolate and still degrades. For PHOTON's +0.40 the positional reading is
still overwhelmingly likely (a global RoPE stack at 16× its trained range is a
textbook case, and bounding that stack is what removes 0.376 of it). For the
+0.026 residual it is now known to be false.

One thing the table does not say at all: that longer context *helps*. No arm
improves. These are 65M models trained on 30M tokens at 512-token contexts, and
asking them to profit from 8192 is not a fair question. The claim is that
Celeritas is *defined* out there and PHOTON is not.

### 15b-3. Celeritas uses more *near* context — and that is the whole of it

The obvious explanation for a flat length curve is the unflattering one — the
model stopped depending on the far stream, so lengthening it stopped mattering.
`scripts/context_diag.py` on the same held-out slice says the opposite. Nats
lost when everything further back than `d` is replaced with noise:

| d | 512 | 256 | 128 | 64 | 32 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| FENESTRA | 0.0000 | +0.0019 | +0.0099 | +0.0277 | +0.0741 | +0.1224 |
| `nope` only | 0.0000 | +0.0019 | +0.0112 | +0.0292 | +0.0773 | +0.1364 |
| **Celeritas** | 0.0000 | **−0.0001** | +0.0100 | +0.0316 | +0.0816 | **+0.1791** |

Celeritas leans on context **more** at every cut (0.179 nats beyond 16 tokens
against FENESTRA's 0.122) and is the only arm that loses nothing at all going
from 512 to 256. So the flat length curve is not indifference. This also
partially answers the standing FENESTRA caveat — `ctx nats` fell from 0.32–0.39
to 0.10–0.12 when `decoder_lookback` arrived, and `decoder_stream` moves it
back up while being cheaper than the thing that pushed it down.

The mechanism behind the flat curve is established in 15d, and it is simpler
than the account that stood here (a second route to recent global state through
the decoder's window). Varying `decoder_stream` alone moves the length
degradation monotonically to zero and the KL column tracks it step for step:
the window sets how far a token sees locally, and everything past that it stops
consulting. 15d also refutes the corollary this paragraph implied -- narrowing
the window does **not** push work back onto the hierarchy.

### 15b-4. The flat curve is indifference, not robustness

`context_diag` said Celeritas uses *more* context, so the obvious deflationary
reading looked wrong. It is not wrong, it was measured in the wrong band. The
direct test: take one held-out sequence, score the last 256 positions with 512
tokens of history and again with 8192, and compare the **distributions** rather
than the loss.

| arm | max abs delta logit | mean | KL(512 vs 8192) | argmax agreement |
|---|---:|---:|---:|---:|
| FENESTRA | 3.833 | 0.124 | 0.02105 | 91.4% |
| `nope` only | 3.354 | 0.170 | 0.03706 | 87.5% |
| **Celeritas** | **0.151** | **0.0032** | **0.00002** | **99.6%** |

**Celeritas' predictions barely move.** KL 2e-5 and 99.6% argmax agreement
between a 512-token context and a 16×-longer one: essentially nothing from
beyond ~512 tokens reaches the token prediction, so there is nothing there to
degrade. The flat length row is that, not superior generalisation.

Both facts hold at once, and the ablation table above shows how: Celeritas is
stronger than FENESTRA in every context band **up to 128 tokens** (0.0975
against 0.0483 in the 32→16 band) and uses **nothing at all** in the 256→512
band, where FENESTRA uses 0.0019. So `decoder_stream` moved the model's context
consumption *inward*, into the window it can serve exactly and cheaply.

Which reframes what the length curve measures at this scale. FENESTRA's top
stream is highly sensitive to sequence length — KL 0.021, 8.6% of argmaxes flip
— and that sensitivity is worth **+0.0019 nats of use and +0.0262 nats of harm**
at 16×. It is not a long-range channel doing work; it is an out-of-distribution
stack injecting noise. **The length column is a noise-injection measurement**,
and the reason PHOTON scores +0.40 on it is that a global RoPE encoder run 16×
past its trained range injects a great deal of noise. That is still exactly what
"extrapolation failure" means operationally, and bounding the stack is still
what fixes it.

Nothing here consumes meaningful context past a few hundred tokens *on
ordinary text*, and no measurement in this section should be read as evidence
that one of them would at scale.

> **Superseded in its stronger form by 16.** This paragraph originally read
> "none of these arms has a *useful* long-range channel", which is a claim
> about the model, and the evidence only supports a claim about the corpus.
> Plant something retrievable 192-768 tokens back and every arm delivers
> **+0.85 nats** through the hierarchy, at distances beyond the training
> context. The channel is unexercised here, not absent.

### 15b-5. The quality result: the streaming decoder clears the noise by 2.3x

Three disjoint held-out slices, two languages, 262,144 tokens each, deltas
against the two-seed mean of each slice (`scripts/arm_table.py`, which
mechanises the convention this file settled on in 4m so it stops being
re-derived by hand):

| arm | ja@48M | ja@60M | en@40M | ctx nats |
|---|---:|---:|---:|---:|
| PHOTON (seed A) | +0.0323 | +0.0345 | +0.0021 | 0.323 / 0.327 / 0.325 |
| PHOTON (seed B) | -0.0323 | -0.0345 | -0.0021 | 0.372 / 0.386 / 0.306 |
| FENESTRA full | -0.1115 | -0.1579 | -0.1112 | 0.124 / 0.124 / 0.098 |
| `nope` only | -0.1114 | -0.1590 | -0.1083 | 0.128 / 0.123 / 0.100 |
| `stream` only | -0.2618 | -0.3121 | -0.2022 | 0.169 / 0.166 / 0.146 |
| **Celeritas** | **-0.2615** | **-0.3146** | **-0.2065** | 0.172 / 0.167 / 0.152 |
| two-seed spread | 0.0645 | 0.0689 | 0.0042 | |

The FENESTRA row reproduces the published -0.112 / -0.158 / -0.111 to the fourth
decimal on a rebuilt measurement chain, which is the cross-check that makes the
rest of the table worth reading.

**`decoder_stream` is worth another -0.150 / -0.156 / -0.094 on top of
FENESTRA** -- 2.3x the ja two-seed spread, same sign and similar magnitude on
three slices in two languages, and *cheaper* than what it replaces (3.750
layer-positions per token against 6.750). It is the second change in this
project to clear the noise, and it clears it by more than `decoder_lookback`
did.

It also partly repairs the one reproducible regression FENESTRA introduced:
`ctx nats` 0.124 -> 0.172 on ja and 0.098 -> 0.152 on en, against PHOTON's
0.32-0.39. Moving back the right way, and for less compute than the change that
pushed it down. 15b-4 says where that context now lives, and it is not where the
label suggests.

### 15c. What the streaming decoder costs at decode time

`decoder_stream` replaces `M` blind blocks of `R + C` positions with one
windowed stream of the same `R + C` positions per group. Training-side that is
strictly cheaper — 3.750 layer-positions per token at level 1 against FENESTRA's
6.750 — but **generation is not free**, and the benchmark says so.
`scripts/benchmark_generation.py`, batch 8, 256-token prompt + 512 new, one
3070 Ti:

| | HierGen | RecGen | KV measured |
|---|---:|---:|---:|
| PHOTON | 546 | 553 tok/s | 3.84 / 1.00 MiB |
| FENESTRA | 517 | 538 tok/s | 1.06 / 1.06 MiB |
| **Celeritas** | **499** | **500 tok/s** | 3.41 / 3.41 MiB |

Two costs, both structural and both bounded:

* the per-token decoder now scores a 64-position window instead of a
  9-position block, so decode is **~5-7% slower** than FENESTRA at this batch
  and width. It does not grow with `T`.
* the rolling decoder caches hold a fixed **173.5 KiB per sequence** that
  PHOTON's per-group scratch buffer did not. At a 768-token context that is
  most of the measured 3.41 MiB; at 8K it is 0.005 KiB/token and the −20% on
  the top-level latent cache has overtaken it. `decoder_cache_bytes_per_token`
  is reported separately from `kv_cache_bytes_per_token` so this stays visible
  and so every KV number published before `decoder_stream` existed remains
  comparable.

**One of the two was a bug and is fixed.** The first version rolled the cache
after *every* write — a full copy of the buffer, per layer, per generated
token, in the one stack that runs once per token. Rolling lazily instead
(let the buffer run to `window + block` and drop `block` entries at a time)
took 476 → 499 tok/s, and it is still exact: the window mask is `ki > qi - w`
on buffer indices, so entries older than the window are masked out whether or
not they are still resident.


### 15d. The `decoder_stream` sweep: the window *is* the length curve

15b-4 left the mechanism behind the flat length row unestablished and named the
experiment that would test it — "sweep `decoder_stream` and watch
the length curve". Here it is: seven arms, `w ∈ {8, 16, 32, 64, 128, 256}` plus
the block layout, identical data / seed / steps / LR.

**The sweep is exactly compute-matched.** The windowed mask is built dense,
`S × S`, whatever the width, so every arm runs identical FLOPs, identical
activation memory and identical parameter counts. The window changes only what a
position may *read*, and the size of the rolling cache at generation. Every arm
was also checked to generate what it trained: `recgen` against the training
forward on the real checkpoints gives KL ~1e-9 and 100% argmax agreement at
`w = 8, 32, 64`.

| `w` | tokens | ja@48M | en@40M | len@16× | KL(512 vs 8192) | argmax | ctx>16 | ctx 256–512 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| block | 4 | −0.1115 | −0.1112 | +0.0262 | 0.01736 | 91.2% | +0.1224 | +0.0019 |
| 8 | 6 | −0.0792 | −0.0700 | +0.0223 | 0.01809 | 91.8% | +0.1431 | +0.0032 |
| 16 | 12 | −0.1076 | −0.0860 | +0.0110 | 0.01085 | 91.7% | +0.1712 | +0.0043 |
| 32 | 25 | −0.1659 | −0.1262 | +0.0016 | 0.00294 | 95.9% | +0.1719 | +0.0017 |
| **64** | 51 | **−0.2615** | **−0.2065** | **0.0000** | 0.00004 | 99.7% | +0.1791 | −0.0001 |
| 128 | 102 | −0.3017 | −0.2422 | 0.0000 | 0.00006 | 99.6% | +0.2990 | +0.0013 |
| 256 | 204 | −0.3382 | −0.2903 | 0.0000 | 0.00006 | 99.6% | +0.4822 | +0.0021 |
| two-seed spread | | 0.0645 | 0.0042 | | | | | |

Full curves, CE minus CE at the 512-token training length:

```
w=  8   1024:+0.0032  2048:+0.0086  4096:+0.0136  8192:+0.0223
w= 16   1024:+0.0031  2048:+0.0034  4096:+0.0061  8192:+0.0110
w= 32   1024:-0.0002  2048:+0.0003  4096:+0.0014  8192:+0.0016
w= 64   1024:-0.0001  2048:-0.0001  4096:-0.0001  8192:-0.0001
w=128   1024:-0.0000  2048:-0.0000  4096:-0.0000  8192:-0.0000
w=256   1024:-0.0001  2048:-0.0001  4096:-0.0001  8192:-0.0001
```

**1. The length curve is a monotone function of the window, and it saturates.**
+0.0262 at the block layout, falling through +0.0223 / +0.0110 / +0.0016 to
**zero at `w = 64`** (51 tokens of history), and flat thereafter. The window
*is* the length curve; nothing else in the sweep varies.

**2. The KL column proves what that flatness is.** It tracks the length column
step for step — 0.017 → 0.018 → 0.011 → 0.0029 → 0.00004 — and lands at the
noise floor exactly where the degradation does. **The model does not become
robust to more context; it stops consulting it.** A flat length row is bought,
not earned, and the price is paid in the far channel.

**3. Quality is monotone in the window and does *not* saturate.** −0.079 →
−0.108 → −0.166 → −0.262 → −0.302 → **−0.338** at `w = 256`, still improving
where the length column has been flat for three doublings. Same ordering on both
held-out slices and independently on train CE (4.7451 → 4.7206 → 4.6725 →
4.6016 → 4.5712 → 4.5444). The individual steps are mostly inside the ja
two-seed spread of 0.065; **the monotone ordering of seven arms, reproduced on
two slices in two languages and by a third independent statistic, is the
evidence** — noise does not order seven points the same way three times.

**4. And this refutes the hypothesis 15b-4 left standing.** It called the window
"the obvious first knob for anyone trying to get the long-range channel back".
It is not. The 256–512 band — the only context the
decoder window cannot reach at any width in this sweep, so the only context the
*hierarchy* must carry — is **≤ 0.0043 nats at every setting**, an order of
magnitude below the two-seed spread on that statistic, with no trend. Narrowing
the window does not push work back onto the hierarchy. It just makes the model
worse *and* more length-fragile at the same time: `w = 8` is the worst arm on
both axes.

There is no interior optimum and no trade-off to tune. At this scale the
hierarchy's far channel is simply a worse source of information than a wider
local window, and the model prefers the window whenever it is offered one.

> **`ctx>16` rises from 0.12 to 0.48 across the sweep and means nothing.** It
> measures how much is lost when everything past 16 tokens is corrupted, so as
> the window grows it increasingly measures the *window*, not the hierarchy.
> The 256–512 band is the column to read, and it is flat at zero. This is the
> same class of error as 4m's layer-positions: a statistic used one level past
> what it can support.

**Cost of a wide window: memory, not speed.** Generation throughput over the
sweep is 623–682 tok/s with no trend (batch 8, 256-token prompt, 512 new) — at
this width the decoder is launch-bound, and scoring 8 keys or 256 costs the
same. The rolling decoder cache is linear: 2.32 → 7.16 MiB at batch 8 over a
768-token context, i.e. 0.0041 → 0.0798 KiB/token at ctx 8192. So `w` is chosen
against a KV budget, not against a quality/latency curve.

**Practical answer, with its caveat.** At 65M / 30M tokens / 512-token contexts,
take the widest window the KV budget allows: quality improves monotonically to
at least 204 tokens of history and nothing measured gets worse. **But the whole
curve is measured where the hierarchy demonstrably has nothing to do** — the
256–512 band is worth ≤0.004 nats for every arm including the block layout, so
this sweep cannot see a cost to abandoning the far channel even if one exists.
`configs/celeritas_probe_ctx2k_w{8,64}.yaml` ask the same question at a
2048-token training context, where the top stream has 128 units instead of 32.
They are prepared and not run.


---

## 16. The long-range channel works. Ordinary text just never asks it anything.

§15 said, in three documents, that **"none of these arms has a *useful*
long-range channel"**. That was wrong, and it was wrong in a specific and
instructive way: it read a statement about *the average text* as a statement
about *the model*.

### 16a. The instrument that had been missing

`scripts/copy_diag.py`. A span of 16 real tokens is placed once, then repeated
`d` tokens later, and **only the second occurrence is scored**. The control is
the same sequence with an *unrelated* real span in that slot, so

    copy gain(d) = CE(unrelated span) − CE(repeated span)

is how many nats retrieval is worth at distance `d`, with the filler, the span
position and the "there is a span here" signal all held identical. Nothing but
retrievability differs.

**The control boundary is not the level-1 decoder window**, and getting that
wrong would have invalidated the whole thing. A token's *non-global* reach is
the maximum over every span-bounded stack, and the deepest one dominates:
Celeritas' level-2 decoder windows 64 positions over *level-1 units*, i.e. ~43
units, i.e. **~171 tokens** — against the 51 tokens the level-1 decoder alone
suggests. `bounded_reach_tokens()` computes it per architecture:

| arm | `C_<=L` | non-global reach |
|---|---:|---:|
| PHOTON | 16 | 16 tokens |
| FENESTRA | 16 | 64 |
| Celeritas | 16 | **171** |
| Celeritas `C=4x2` | 8 | 128 |
| Celeritas L=3 | 64 | **682** |

Only distances past that number test the hierarchy at all. (This is also a
finding in its own right: **adding a level multiplies the local reach by `C`**,
free, because the deeper decoder's window covers coarser units. Nobody
designed that.)

### 16b. It retrieves, at ~0.85 nats, past the training context

Held-out ja, `seq_len` 1024, span 16, mean over `d ∈ {192, 256, 384, 512, 768}`
— every one of them beyond the arm's non-global reach:

| arm | mean copy gain | at `d`=768 | fixed-grid penalty |
|---|---:|---:|---:|
| PHOTON | **+0.897** | +0.793 | +0.0004 |
| FENESTRA | **+0.827** | +0.720 | +0.0000 |
| Celeritas | **+0.868** | +0.790 | +0.0004 |

**The hierarchy retrieves.** Nearly a nat, in every variant, and it barely
decays: +0.95 at 256 tokens against +0.79 at 768. It is still doing it at
`d = 768`, which is **1.5x the 512-token context these models were trained on**.

The first version of this measurement showed decay to +0.59 at `d = 448` and I
nearly reported it. It was an artefact of `seq_len 512`: at `d = 448` the
*first* occurrence sits at position 48, in the first three meta-groups, with
almost no context of its own. Re-run at `seq_len 1024` — which Celeritas
permits because it has no maximum context length — the decay is gone. The
edge, not the distance.

### 16c. How that squares with §15, which was also right

Both instruments are correct and they measure different things:

* `context_diag` replaces far context with **noise** and scores **ordinary
  text**: the 256–512 band is worth ≤0.004 nats. Ordinary Japanese Wikipedia
  at this scale genuinely does not depend on what happened 300 tokens ago.
* `copy_diag` plants something **retrievable** at distance `d`: worth 0.85
  nats.

So the corrected claim is: **the channel has capacity and uses it when there is
something to carry; the corpus almost never gives it anything.** §15's
`ctx nats` numbers stand, §15b-4's KL ≈ 0 stands — on ordinary text — and the
sentence "no arm has a useful long-range channel" does not. A channel that
delivers 0.85 nats on demand is not useless; it is unexercised.

**The transferable error** is the same one this file keeps recording: a metric
used one level past what it supports. An average over ordinary text cannot see
a capability that ordinary text rarely invokes, and reading its silence as
absence is exactly the mistake §4m and §15b-3 already made in other clothes.

### 16d. Dynamic chunking would buy ~0 nats here — priced before building it

The user's call was to price H-Net-style dynamic chunking before attempting it,
and the copy probe prices it directly. Each distance is run twice:

* **aligned** (`d % C_<=L == 0`) — the chunker decomposes the span into the
  same groups both times, so the meta-unit vectors can match;
* **offset** (`d % C_<=L = C/2`) — the same text on a different chunk grid, so
  the meta-unit vectors differ although the tokens do not.

The gap between them, past the non-global reach, is the fixed-grid penalty with
everything else held constant. Measured: **+0.0004, +0.0000, +0.0004 nats**,
worst single pair 0.0017, against a per-arm effect size of ~0.85.

**The fixed chunk grid costs nothing measurable at this scale.** The chunker is
apparently robust to where the boundary falls — plausibly because the encoder
above it sees several chunks and can recompose across them. Dynamic chunking is
the largest outstanding idea for this architecture and it would break the
exactness tests, chunk-parallel decoding and the KV accounting all at once;
this says do not pay that, yet.

What it does **not** say: that dynamic chunking is worthless in general. This is
one tokenizer, one language pair, 65M parameters, spans of 16 tokens on a
16-token grid. A byte-level model, where one chunk is a handful of characters
and the grid cuts inside morphemes, is a different question and this experiment
does not touch it.

### 16e. An 8x range in retrieval granularity moves nothing

`C_<=L` sets how many tokens one vector must stand for before that content can
travel any distance, so it is the granularity wall's own dial. Two arms trained
identically at ctx 2048, parameter-matched to 0.44% and with **identical
layer-positions per token (5.875)**, differing only in whether the top of the
hierarchy is one level or two:

| | `C_<=L` | tokens per meta-unit | copy gain, d = 768 / 1024 / 1536 | mean |
|---|---:|---:|---|---:|
| Celeritas L=2 | 16 | 16 | +0.789 / +1.294 / +1.814 | **+1.2990** |
| Celeritas L=3 | 64 | 64 | +0.784 / +1.288 / +1.800 | **+1.2906** |

**A 4x coarser summary costs 0.0084 nats of retrieval** — 0.6% of the effect,
an order of magnitude below the two-seed spread. The granularity wall is real
as an argument and not measurable as a quantity here, at least between 16 and
64 tokens per vector, at this scale, on this corpus.

---

## 17. The three-level hierarchy: 24x the crossover, 4x the local reach, free

`C_<=L` 16 -> 64 was the proposed answer to "Celeritas is still `O(T^2)`, just
with a 1/256 constant". `configs/celeritas_probe_L3.yaml` is that, built to make
the comparison unambiguous rather than favourable: **level 1 is byte-identical**
to `celeritas_probe.yaml`, the middle level is carved out of the old top level
with `ffn_inter_size: 512` to pay for itself, and **layer-positions per token
land on exactly 5.875 in both**. Parameters +0.44%.

No code changed. `_emit_group` recurses, `_alloc_caches` and `_prefill` loop
`1..L`, and the accounting is written per level, so a third level was a config
file: `recgen` reproduced the training forward at 4.8e-06 the first time it ran.

### 17a. What it costs, and what it buys

| | Celeritas L=2 | Celeritas L=3 |
|---|---:|---:|
| eval CE, held-out ja @2048 | 4.8128 | 4.8165 (**+0.0037**) |
| train CE (k=5 tail) | 4.6853 | 4.6881 |
| context used beyond 64 tokens | +0.0640 | +0.0632 |
| copy gain, d = 768..1536 | +1.2990 | +1.2906 |
| **attention overtakes everything else at** | 1.75M tokens | **41.5M tokens** |
| **KV/token @ ctx 32768 (exact `recgen`)** | 0.1003 KiB | **0.0249 KiB** |
| non-global reach | 170 tokens | **682 tokens** |
| layer-positions / token | 5.875 | 5.875 |

+0.0037 nats against a two-seed spread of 0.065: **unresolved, and bounded well
below anything the KV number would need to justify.** The `c8x2` precedent was
the worry — fewer top units measurably weakened far-context use — and it does
not repeat: the far bands and the retrieval probe agree to three decimals.

### 17b. The part nobody designed: depth multiplies *local* reach

A token's non-global reach is the maximum over every span-bounded stack, and the
deepest decoder dominates because its window is measured in *coarser units*.
Celeritas' level-2 decoder windows 64 stream positions over level-1 units of 4
tokens: ~171 tokens. A level-3 decoder windows the same 64 positions over
level-2 units of **16** tokens: ~683.

**Adding a level multiplies the local, exactly-cached, O(1)-per-token reach by
`C`,** for no extra layer-positions and no parameters worth mentioning. Nobody
put that in; it falls out of the stream windows being per-level. It also makes
"how far can a token see without the global stream" a knob with a `C^L` range,
where 15b-3 had it as a fixed 51 tokens.

It cuts the other way too, and the numbers say so: a deeper hierarchy makes the
*global* stream less necessary. KL(2048 vs 8192) is 0.00052 at L=2 and
**0.00000** at L=3; argmax agreement 99.3% and **100.0%**. The three-level model
is even more indifferent to context it was not trained on — on ordinary text —
for the same reason it retrieves just as well when asked: more room locally.

---

## 18. The chunk is its own draft: exact speculative decoding, 1.15x

`MTPModule` already learns the chain `h_i, e(t_i) -> t_{i+d}`, and the
conditioning vector already determines its chunk. So the model ships with a
drafter of exactly the right length, with no attention in it, and verification
is **one** decoder pass: the decoder is causal, so writing slots `0 .. C-2` in a
single call yields the true slots `1 .. C-1` simultaneously.

Per chunk: 2 level-1 decoder calls when nothing is rejected, against `C_1 = 4`
sequential. Measured on `runs/probe-cel-mtp` (Celeritas + `mtp_depth: 3`),
greedy, 256-token held-out prompt + 512 new, batch 1, **eight independent
prompts**:

| prompt offset | seq tok/s | spec tok/s | speedup | accepted | L1 calls/chunk |
|---|---:|---:|---:|---:|---:|
| 48.0M | 63.2 | 87.8 | 1.39x | 92.2% | 2.20 |
| 48.1M | 69.2 | 73.5 | 1.06x | 45.3% | 3.00 |
| 48.2M | 69.4 | 96.4 | 1.39x | 96.4% | 2.09 |
| 48.3M | 64.1 | 63.2 | 0.99x | 0.0% | 4.08 |
| 48.4M | 66.7 | 69.3 | 1.04x | 16.7% | 3.58 |
| 48.5M | 70.3 | 88.1 | 1.25x | 98.4% | 2.11 |
| 48.6M | 65.8 | 72.0 | 1.09x | 22.9% | 3.44 |
| 48.7M | 69.2 | 68.9 | 1.00x | 15.1% | 3.66 |
| **mean** | | | **1.15x** | **48.4%** | |

**Every row is token-for-token identical to sequential decoding.** That is the
whole licence for it: greedy speculation accepts only exact matches, so it is
not an approximation of sequential decoding, it *is* sequential decoding.
`test_speculative_decoding_is_the_same_function_as_sequential` asserts equality
over 64 generated tokens rather than closeness — which is what catches the
subtle failure, since rejected drafts are written into the rolling cache and
must be un-written, and a cache carrying drafted-but-wrong embeddings would
diverge several chunks later, invisibly to any per-step tolerance.

**Report the spread, not the best row.** One prompt gave 1.50x on a
mid-training checkpoint and 1.44x on the final one, and quoting either would
have been 4m's `--tail 5` mistake in new clothes: acceptance ranges 0% to 98%
across eight prompts of the same corpus. The mean is 1.15x and the floor is
0.99x — speculation is never meaningfully worse, because a rejected slot is
corrected from the verify pass rather than re-stepped.

Two structural limits:

* **Acceptance is all-rows.** One shared rolling cache cannot advance different
  sequences by different amounts, so a batch accepts only what every row
  accepts: 48% at batch 1, ~32% at batch 4-8 on random prompts, near 0% at
  batch >= 4 on real ones, i.e. **1.00-1.09x past batch 1**. This is a latency
  optimisation, not a throughput one.
* **It is inert wherever exact-match acceptance would not be the identity** —
  temperature sampling (which needs the `min(1, p/q)` rule, not implemented), a
  repetition penalty, or `forced_logits`, which must record every position's
  logits one at a time because it is the equivalence harness.
  `test_speculation_refuses_what_it_cannot_reproduce` pins each refusal.

---

## 19. `T % C_<=L == 0` was never necessary

`forward` right-pads to a multiple of `C_<=L`, runs, and drops the extra
outputs. Exact rather than approximate, by the same causality argument the rest
of the architecture rests on: the pad is *appended*, the chunker folds it into
units that come after every real one, and no position `< T` can attend forward.
`test_any_sequence_length_is_accepted_exactly` asserts the ragged forward
reproduces the aligned one to 2e-5, and at lengths where the tensor shapes
coincide, to 0.000e+00. Cost: at most `C_<=L - 1` wasted positions, once.

`PackedTokenDataset` keeps its own alignment check, and should. That is a
different requirement — packing efficiency, paid on every window forever rather
than once — and conflating the two is how the constraint came to be enforced in
three places.

---

## 20. SCALA (2026-07-31): the hierarchy made self-similar; depth becomes an inference-time parameter, and it transfers

Design and measurements in `docs/scala.md`; tests in `tests/test_scala.py`
(14, all on shaken weights); preset `scala/model/scala.py`. One config
flag (`tie_mid_levels`), one construction-loop change, zero changes to the
forward, the losses, or the generator.

**The mechanism.** Three module sets -- level 1, one MID applied at levels
2..L-1, the Celeritas CAP on top -- registered once under depth-free names
(`level_token.* / level_mid.* / level_cap.*`; a ModuleList would serialise the
shared MID once per index and make the state_dict a function of depth).
Parameter count becomes independent of L: 64.48M at k=1 and at k=4 alike,
where the untied L=4 control carries 69.32M. Nothing a `ScalaLevel` stores
turned out to be level-specific: `max_units` only seeds a RoPE table that
grows on demand, every inference cache lives in the generator's per-level
state, and the non-top `start_latent` is never read.

**Measured, k=2-trained probe (65M, 30M tokens, seq 4096, n=1):**

* **Zero-shot depth transfer works.** The same checkpoint re-expressed at
  L=5 and L=6 -- functions it was never trained as -- costs **<= 0.0023 nats**
  on identical held-out tokens (ja@48M and en@40M, contexts 4096..32768),
  against a two-seed spread of 0.063/0.104. `protocol_diag --depth 3`: loads
  with 0 missing keys, recgen KL 0.0001 vs that depth''s own training forward.
* **Tying is free at this scale.** Same seed, same data: tied 5.0075 vs
  untied 5.0077 eval CE (ja); train CE 4.8010 vs 4.8011. The k-fold gradient
  confluence into the shared MID (risk R1) did not surface.
* **Retrieval survives re-expression.** At depth 3, copy gain +0.78/+0.27/
  +1.48 nats at d = 4096/8192/16384 -- distances and a depth the model never
  trained at. Fixed-grid penalty still +/-0.003.
* **Bounded-top policy: no cache grows with T.** Grow k when the CAP exceeds
  U_max units; resident state is measured (+0.14 MiB per level, constant) and
  analytic (`accounting.scala_state_bytes`): 425 KiB at 8K context, 770 KiB
  at 1M. Celeritas L=3 at 1M is 25.5 MiB and O(T).
* **vs Celeritas L3 at the trained shape: unresolved** -- the pair-mean delta
  flips sign between slices (ja +0.056, en -0.013). Not called an effect.
* Generation at depth 3 is ~2x slower per token at 65M (launch-bound, one
  more sequential recursion level); the policy only asks for depth 3 past 16K
  context. Unoptimised.

**What it does not settle.** Transfer *working* and transfer *helping* are
different claims: Wikipedia asks <= 0.005 nats of anything past 1024 tokens
(section 16''s conclusion, unchanged by depth). The corpus question stays open.
Also untested: tie x FSDP (guarded), 250M, training at k>2.

One instrument correction along the way: `_attn_core` never counted the MLA
attention sink (n_heads params/layer); invisible at rel=1e-4 on 65M, caught
by `test_scala_analytic_param_count_matches` demanding exact equality.

---

## 21. Ultra-long context (2026-08-01): 1M tokens measured, flat, retrieving, in 0.6 GiB

Motivation: "supports ultra-long context" was a structural claim measured only
to 32K, because every instrument scores through the full training forward,
which at 131K tokens tries to allocate one 25 GiB boolean mask (measured) and
whose per-token equivalence harness would take ~10 h/sequence while hoarding
~200 GB of logits. The claim needed instruments before it needed anything else.

**The tiled exact scorer** (`scala/infer/scoring.py`, probe CLI
`scripts/longctx_probe.py`, 9 tests in `tests/test_scoring.py`): phase A
streams every windowed encoder through the generator''s certified rolling-cache
discipline in large write blocks and appends the NoPE CAP latent tile by tile
(4-8 MB at 1M); phase B re-decodes only a tail segment sized by the streaming
decoders'' receptive field (n layers x window w -> n*(w-1) positions of warm-up,
discarded), whole-stream fallback when a deep instantiation''s upper streams
are shorter than their warm-up. Exact vs model(x) at 2e-4 at every depth,
tiling, and fallback; at 1M no dense reference can exist and agreement is
inherited from the test-pinned kernels -- stated on every published row.
`RotaryEmbedding` gained a 131,072-position table cap (geometric growth below,
per-call fp64 phases above): kills ~0.5 GB of resident tables, the
rebuild-from-zero O(T^2) churn, and the ~0.1 rad fp32 phase quantization at
pos ~1.3e6. Byte-identical below the cap; suite green (85).

**Measured, probe-scala-k2 (trained at seq 4096), T = 131K..1M, held-out:**

* CE over identical scored tokens is FLAT TO FOUR DECIMALS from 131K to 1M --
  256x the training length -- at depths 2, 3, 4 and 6 (L=8), Japanese and
  English. Peak 0.6 GiB; 4.5-4.8 s per 1M-token sequence scored.
* Needle retrieval at T=1M: +0.44 / +0.94 / +0.82 nats at d = 131K / 262K /
  524K, identical to three decimals across CAP granularities of 256, 1024 and
  4096 tokens/unit. At depth 6 the bounded windows alone reach ~699K tokens,
  and past them the CAP retrieves +0.77 nats at d = 786K with 65,536-token
  units: the 16-token needle survives compression into a 65K-token unit. The
  granularity-flat result of section 16 now spans 256 -> 65,536 tokens/unit.
* Fixed-grid penalty at 1M: +0.0000 at every depth. Third survival, strongest
  conditions yet.
* Trained-vs-measured extrapolation: the CAP trained on 16 units; at depth 2
  and 1M it attends over 4,096 (256x). NoPE + sink carried it.

Corrections this session forced: (a) the copy_diag docstring''s "identical
filler" claim is false for i>=1 -- the control arm consumes one extra rng draw
per sequence, so the two arms share fillers in distribution, not per sequence;
the probe measures and prints the actual overlap structure instead of assuming
it. (b) `forced_logits` hoards (B,T,V) logits and is unusable past ~75K
tokens; it remains the small-T equivalence harness and nothing else.

Open: everything here is capability (synthetic needles), not demand -- Wikipedia
still asks ~nothing of the far channel (sections 16, 20), and no training run
has ever SEEN a context past 4096. Long-context midtraining, and a corpus
whose far channel earns its keep, remain the two named absences.

---

## 22. 0.4B validation (2026-08-02, GB10): the architecture holds at 6.5x scale, and the repo could not be cloned

A 422.3M-parameter SCALA (`scala_config(depth=2, d_token=1024, l1=(8,5),
mid=(3,3), cap=(8,3))`, dense, ctx 4096, llm-jp v3 tokenizer) trained on the
GB10 box for 330M tokens over an 8-source mixture (ja fineweb2-edu/fineweb2/
wikipedia/aozora, en fineweb-edu/Nemotron-HQ, swallow-code-v2,
swallow-math-v2; 378M tokens prepared). WSD cooldown completed; zero
spikes/NaNs; holdout CE 6.08 -> 3.39. Every lineage gate passed:

* Exactness: hiergen/recgen == the training forward at KL 0.000, cos(X_top)
  1.0000, at k=2 and k=3; greedy agreement 100.0% on real weights.
* Zero-shot depth transfer: strict-loaded at k=3/k=4 (untrained L=5/L=6),
  CE degradation <= +0.0071 nats across all 8 held-out sources -- slightly
  larger than the 65M probe''s +0.002 but still under a tenth of the seed
  spread.
* Length: 512 -> 16K worst +0.0031; CE flat to four decimals from 131K to 1M
  (256x the training length), also under k=4 re-expression. 1M-token scoring:
  11 s/sequence at 2.6 GiB.
* Retrieval GREW with scale: +1.83 nats in the CAP band past the 2,730-token
  bounded reach, and **+2.28 nats at d=524K in a 1M context** against the 65M
  probe''s +0.82. Fixed-grid penalty still zero.
* Generation is fluent in both languages; greedy code decoding loops, which
  is the expected failure at this token budget, not an architecture defect
  (recgen==hiergen exactness is proven separately).

Artifacts live on the GB10 box: `runs/scala-04b-gb10/final`,
`export/scala-04b` (bf16 safetensors, 0.79 GiB + tokenizer), verification
JSONs under `runs/scala-04b-gb10/verify/`, driven by `scripts/verify_04b.sh`;
new configs `scala_04b.yaml` / `train_04b_gb10.yaml` / `data_04b_mix.yaml`
and `scripts/eval_by_source.py`. (Not yet merged into this repo -- the GB10
tunnel was down when this entry was written.)

**Two defects the run surfaced, both now fixed in the canonical repo:**

1. **The public repo could not train from a clone.** `.gitignore` line 6 was
   the bare pattern `data/`, which matches a directory named `data` at any
   depth -- including the `scala/data/` *package* (dataset.py, chat.py). The
   package was silently untracked since the first commit; a fresh clone died
   on import. The GB10 session, unable to see the original, reconstructed
   the loader from its call sites (crc32 shard order, data_seed separation,
   holdout tail reservation) -- with one acknowledged divergence: the
   reconstruction samples random windows where the original reads a prefix
   per shard. The canonical repo now root-anchors the ignore patterns
   (`/data/`, `/runs/`, `/export/`) and ships the ORIGINAL prefix-reading
   implementation, which is what every published probe measurement used.
   The 0.4B run''s own measurements stand: it reserved explicit holdout
   tails rather than relying on the prefix convention.
2. **`resume: auto` could crash-loop forever.** `save_checkpoint` writes
   `state.pt` first and `meta.json` last, so a SIGTERM mid-save (the GB10
   sandbox kills long GPU processes on a schedule) leaves a `step-*` dir
   with a truncated `state.pt` and no meta. Auto-resume grabbed the newest
   dir unconditionally, crashed in `torch.load`, and the supervisor
   restarted into the same crash. `_maybe_resume` now treats `meta.json`
   (written last => a completion marker) as the integrity gate, skips
   incomplete checkpoints with a log line, and refuses an explicitly-named
   incomplete path with a clear error.

The scale conclusion, stated with its bounds: n=1 at 0.4B/330M tokens, but
no gate weakened and the retrieval channel strengthened. Nothing yet
suggests the architecture breaks with scale.
