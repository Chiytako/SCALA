# CELERITAS

> A standalone Japanese summary of the whole arc -- lineage, every
> measurement, the retraction list, and the Transformer/Mamba
> positioning -- is in [`celeritas_report.ja.md`](celeritas_report.ja.md).

A revision of PHOTON (arXiv:2512.20687) in which the local decoder stops paying
twice for its own context, and no component of the model evaluates a position
outside the range it was trained on.

Named for *c*: PHOTON travels at it.

**The headline, stated the way the ablation came out rather than the way it was
designed:**

| change | quality (held-out CE) | length @16× | what it does buy |
|---|---|---|---|
| **`decoder_stream`** | **−0.150 / −0.156 / −0.094 nats** vs FENESTRA, on three held-out slices against a two-seed spread of 0.065 | **removes the residual degradation entirely** | −44% layer-positions in the stack that is 72% of the budget, −9.9% activation memory |
| `pos: nope` + latent-pure MLA + sink | **neutral** (−0.1114 vs −0.1115) | **neutral** (+0.0254 vs +0.0262) | −20% on the only cache that grows with `T`; deletes `rope_theta` / YaRN / context-extension as decisions; `max_seq_len` stops being a limit |

So one of the two ideas this document is about worked on the measured axes and
one did not, and the one that did was not the one the design was named for. Both
ship — the second is free and simplifies the model — but they are reported for
what they are.

The one-line summary, against FENESTRA's:

> **FENESTRA**: make the training-time receptive field equal the
> generation-time cache budget. Exactness becomes free.
>
> **Celeritas**: stop recomputing the receptive field. The same positions, run
> as one windowed stream instead of blind blocks, reach twelve times further
> for 44% less — and every stack in the model then has a span fixed at build
> time, so the length limit goes too.

Every change is a **deletion**. Nothing is added to the forward pass.

---

## The three changes

### 1. `pos: nope` on the top encoder — and only there

**The observation.** A token's position in a PHOTON hierarchy is already a
mixed-radix number: (meta-unit `g`, chunk within group, slot within chunk).
The architecture encodes it a second time, with RoPE, in every stack. That
second copy is only *needed* where the sequence length is unbounded, and after
FENESTRA exactly one stack qualifies:

| stack | sequence it attends over | bounded? |
|---|---|---|
| level-1 encoder | `encoder_window` units | yes, at build time |
| level-1 decoder | `decoder_stream` positions (was `R + C`) | yes, at build time |
| level-2 decoder | same | yes |
| **top encoder** | **`T / C_<=L` units** | **no** |

RoPE is exactly relative: the score between a query and a key depends only on
their offset. So a stack whose span is bounded at `w` never evaluates RoPE at
an offset larger than `w`, wherever in the sequence it sits — its function is
the trained one at any `T`, not an extrapolation of it. Only the top encoder
is asked for something it was not trained on.

**Why NoPE is the thing to put there**, rather than a longer table or YaRN.
[RNoPE-SWA](https://arxiv.org/html/2501.18795v1) (Cohere, 2025) reports
that in a hybrid stack, *NoPE full-attention layers do the retrieval and RoPE
sliding-window layers supply recency bias*, with a 1:3 interleave; at 256K
against a 128K training length the hybrid held while the RoPE baseline lost
~41% on retrieval. That is a statement about **roles**, and PHOTON assigns
those roles to *stacks* instead of layers:

* the top encoder is the retrieval channel — everything a token knows about
  text outside its own meta-context arrives through it;
* the windowed lower stacks are the recency channel, and they keep RoPE.

The hierarchy gets this cheaper than a layer interleave does, because the
global stack runs once per `C_<=L` tokens. Here that is 1/16 of the token
rate: 0.375 layer-positions per token for a 6-layer top encoder.

**What it costs to state:** NoPE's known failure mode is attention dispersing
as the context grows. The standard cheap fix is a learned per-head attention
sink (StreamingLLM / GPT-OSS) — one extra logit per head that owns no value
vector, so a head can send mass nowhere instead of spreading it. It is on.

> **Measured, and it is the negative result of this document:** the sink does
> not cover it, and **this change buys no measurable length robustness at 65M**
> — the ablation is +0.0254 nats at 16× against the RoPE model's +0.0262. Keep
> it for what it does deliver, which is change (2), the sink on the absorbed
> path, and the deleted knobs. The length curve is bought by change (3).
> Details under *Measurements*.

### 2. `mla_qk_rope_head_dim: 0` — forced by (1), and the payoff of it

MLA carries a **decoupled RoPE key** for one reason: weight absorption
rewrites `q_nope · (c W^UK)` as `(q_nope W^UK,T) · c` so the latent `c` never
has to be decompressed, and RoPE does not survive that rewrite. So DeepSeek's
MLA keeps a second, un-absorbed key path purely to carry the rotation.

Under NoPE there is nothing to carry. The channels are deleted, the score
width is held constant by moving them into `qk_nope`, and the latent cache
becomes pure `kv_lora_rank`:

| | per unit per layer | probe | 250M |
|---|---|---:|---:|
| PHOTON / FENESTRA | `kv_lora + qk_rope` | 160 | 160 |
| **Celeritas** | `kv_lora` | **128** | **128** |

**−20% on the only cache that grows with `T`.** `AttentionConfig.__post_init__`
refuses `pos="nope"` with a nonzero `mla_qk_rope_head_dim` rather than
allocating channels that are provably uninformative.

A second thing falls out: the absorbed path now supports a learned sink. It
had refused one, on the grounds that a sink needs an explicit softmax — but
the absorbed path *is* an explicit softmax; it never reaches
`scaled_dot_product_attention`. So the one stack that wants NoPE, weight
absorption and a sink at once can have all three
(`test_learned_sink_survives_weight_absorption`).

### 3. `decoder_stream` — the decoder as a stream, not as blind blocks

*(Numbered third because it was designed last. It is the one that worked.)*

**The problem FENESTRA half-solved.** CE by slot inside a level-1 chunk showed
+0.119 nats for slot 0 against slot 3: the first token of a chunk has no
token-level context at all. `decoder_lookback: 1` fixed it by widening every
group's block from `R + C` to `R + 2C` positions — and **that pays for the
previous chunk twice**, because those `C` positions are recomputed at every
layer of every group, having already been computed as the *previous* group's
content.

**Celeritas.** Keep `R + C` positions per group and concatenate the groups
into one causal stream with a sliding window, instead of batching them into
`M` blocks that cannot see each other. The layout is unchanged — group `g`
still contributes `[u_g (R positions); content_g (C positions)]`, the readout
is still `out[R-1 : R-1+C]` — so this adds **no positions at all**:

| | span/group | layer-pos/token (L1 dec) | reach at slot 0 |
|---|---:|---:|---|
| PHOTON (`R=4`) | 8 | 6.000 | nothing |
| FENESTRA (`R=1, K=1`) | 9 | 6.750 | 1 chunk = 4 tokens |
| **Celeritas (`R=1, w=64`)** | **5** | **3.750** | **`w` positions ≈ 51 tokens** |

**−44% on the stack that is 72% of the layer-position budget, while reaching
12× further back.** `test_streaming_decoder_is_exact_and_reaches_past_one_chunk`
asserts the reach is real (perturbing a token 1, 2 and 4 chunks back all move
slot 0's logit) and that nothing leaks backwards.

Causality is unchanged and is worth stating precisely: `u_g` is a function of
`X_hat^(l)_{g-1}`, which has attended to exactly `t_{<gC}`; every stream
position before `u_g` belongs to an earlier group. So a token's receptive
field is still exactly its own prefix, and the model gains a second route to
information it was already permitted — which is the same shape of argument
`global_skip` made, except this one is also cheaper than what it replaces.

**Generation stays O(1) in `T`.** The per-group cache becomes one rolling
cache of `window + R + C` entries, written at `dec_fill` and read at
`dec_pos`, rolled back to `window` after each write — the discipline FENESTRA
introduced for the windowed encoder, one level down. Buffer-index differences
equal stream-position differences for every retained entry, so the window mask
selects exactly the keys the training forward selected. Under NoPE at the top
and RoPE-relative below, this is an identity, not an approximation.

The one thing it costs: **prefill must now run the decoders too.** A streaming
decoder's window reaches across the prompt boundary, so `_prefill_decoders`
replays the prompt's decoder stream — one decode pass over the prompt, which
is what "prefill is a forward pass" means everywhere else. Skipping it puts an
error exactly at the seam, which is the failure `_tail_content` fixes for
`decoder_lookback`.

---

## Fewer things to get right

Not a performance claim, but the reason the three changes are worth having
together. Each one *removes* a decision:

| gone | why |
|---|---|
| `max_seq_len` as a limit | it is now a training shape. No YaRN factor to pick, no `rope_theta` to retune for the top encoder, no context-extension phase. |
| `rope_theta` on the top encoder | there is no rotation to scale. |
| `mla_qk_rope_head_dim` | refused under NoPE, so it cannot be set wrong. |
| `decoder_lookback` | subsumed by `decoder_stream`, which is cheaper and reaches further; setting both is a config error. |
| `rec_loss_alpha`, `rec_loss_kind`, `rec_loss_min_level`, `chunk_cond_prob`, `self_cond_prob` | all 0 since FENESTRA — nothing is substituted at inference, so nothing needs correcting. |
| tuning `converter_width` | at `R=1` there is one conditioning position and the stream supplies the context the fan-out copies were padding for. |

And one thing added, for the same reason:

```python
from scala.model.celeritas import celeritas_config
cfg = celeritas_config(d_token=384)          # the measured probe geometry
```

`celeritas_config()` emits exactly `configs/celeritas_probe.yaml`, and
`test_celeritas_preset_matches_the_shipped_probe_yaml` fails if the two drift
apart — a preset that hands out a configuration none of the measurements are
about is worse than no preset.

## What Celeritas does not change

The hierarchy, the chunker, the converter, MLA, the MoE, the objective, and
the exactness discipline. It also inherits FENESTRA's `encoder_window`
unchanged, and with it the finding that made all of this possible: the paper's
RecGen substitution transmits exactly `0.000e+00` about what the model
generated (`findings.md` §4j-2), and bounding the intermediate encoders rather
than deleting them makes the exact protocol the cheap one. `rec_loss_alpha`,
`chunk_cond_prob` and `self_cond_prob` are all 0 — there is no substitution
left for them to correct.

---

## Cost, by construction

65M probe (`d_token=384`, 4×4), forward FLOPs per token:

| | layer-pos/tok | L1 decoder | stacks | attention | **total** |
|---|---:|---:|---:|---:|---:|
| PHOTON (`R=4`) | 8.375 | 6.000 | 26.5 M | 0.27 M | 103.2 M |
| FENESTRA full | 8.875 | 6.750 | 28.1 M | 0.10 M | 104.7 M |
| **Celeritas** | **5.875** | **3.750** | **18.6 M** | 0.49 M | **95.6 M** |

−34% of the stacks, −8.7% of the total — the LM head is 80% of the total at
`d_token=384` and no hierarchy knob touches it. At 250M
(`celeritas_small.yaml` against `base_small_p2.yaml`): layer-positions
12.125 → 7.875, stacks 48.7 → 31.8 M, total 150.9 → 134.7 M (**−10.7%**).
The head share falls with width, so the same change is worth more at 8B.

Attention grows 5× (0.10 → 0.49 MFLOP/token) because the decoder now scores a
64-position window instead of a 9-position block. It is **0.5% of the forward
pass** at these widths. Say the number rather than hiding it.

**Measured throughput does not see any of this**, and that is expected: at 65M
and batch 16 the step is launch-bound, which is exactly why `findings.md` §4m
had to retract a "−39% compute" claim. `scripts/probe_throughput.py`,
back-to-back on one 5060 Ti, seq 512, batch 16, three repeats each:

| | tok/s | peak alloc |
|---|---|---:|
| FENESTRA full | 12.4 – 12.9 K | 5.45 GiB |
| Celeritas | 12.7 – 13.7 K | **4.91 GiB** |

The real training runs agree and are a much larger sample (median over every
logged step past warmup, same card, same session): FENESTRA 20.0K, Celeritas
20.3K, `nope`-only 19.6K — interquartile ranges that overlap each other
completely.

So: **the throughput ranges overlap and the memory does not.** −9.9% peak
allocation reproduced on every single measurement. That is the honest
efficiency claim at this scale; the FLOP reduction is real arithmetic that this
hardware and this batch size cannot resolve, and saying otherwise is the
mistake §4m exists to prevent.

### KV cache

Bytes per generated token, `recgen` (only the top-level cache grows),
including the streaming decoders' rolling caches:

| ctx | 512 | 2048 | 8192 | 32768 |
|---|---:|---:|---:|---:|
| PHOTON HierGen (exact) | 0.617 | 0.617 | 0.617 | 0.617 |
| PHOTON RecGen (**approximate**) | 0.180 | 0.133 | 0.121 | 0.118 |
| FENESTRA (exact) | 0.195 | 0.137 | 0.122 | 0.118 |
| **Celeritas (exact)** | 0.511 | 0.198 | **0.120** | **0.100** |

Celeritas is **worse at short contexts and better asymptotically**, and the
reason is not subtle: the streaming decoders hold a fixed 173.5 KiB per
sequence that PHOTON's per-group scratch buffer did not. That is a constant,
so it is 0.34 KiB/token at 512 and 0.005 at 32768. Past ~4K it is paid for by
the −20% on the top-level latent cache. `decoder_cache_bytes_per_token` is
reported separately from `kv_cache_bytes_per_token` precisely so this trade is
visible and so every KV figure published before `decoder_stream` existed stays
comparable.

---

## What is verified, and how

**By construction, on any weights** (`tests/test_hierarchy.py`, 55 passing):

* `test_celeritas_generation_is_exact_under_both_protocols` — a NoPE top
  encoder read through the weight-absorbed latent cache *with a learned sink*,
  a sliding-window intermediate encoder, and two streaming decoders whose
  rolling caches survive the prefill seam, all at once: `recgen` reproduces the
  training forward to 2e-4 with flat per-position error.
* `test_streaming_decoder_is_exact_and_reaches_past_one_chunk` — **exhaustive**
  causality: every one of 64 positions is perturbed and the logits at or before
  it must not move, while the logits after it must. Sampling six offsets is not
  the right assurance here — the stream lets a position read across a group
  boundary for the first time in this architecture. Slot 0 moves when a token
  1, 2 or 4 chunks back moves.

  The same sweep on the **trained** `runs/probe-cel` checkpoint, all 128
  positions, fp32: worst-case backward influence **exactly 0.000e+00**, with
  forward influence of 3–6 logit units at `pos+1`. The quality gain below is
  not leakage.
* `test_celeritas_has_no_length_dependent_position` — enumerates every
  `TransformerStack` in the model and asserts each is either NoPE or
  span-bounded. **This is the architectural claim, and it is structural**: a
  PHOTON model fails it.
* `test_nope_mla_refuses_a_decoupled_rope_key`,
  `test_learned_sink_survives_weight_absorption`,
  `test_decoder_stream_rejects_configurations_it_cannot_honour`,
  `test_celeritas_survives_a_config_round_trip`.

**Measured** — see the next section. Ablations are one change at a time:
`celeritas_probe_nope.yaml` is (1)+(2) alone, `celeritas_probe_stream.yaml` is
(3) alone, against `base_probe_p2full.yaml`.

---

## Measurements

Four arms, one change at a time, identical data / seed / steps / LR, parameter
counts within 0.1%:

| arm | top encoder | level-1 decoder |
|---|---|---|
| `probe-p2full` | RoPE | block + `decoder_lookback: 1` |
| `probe-cel-nope` | **NoPE** + latent-pure MLA + sink | block + `decoder_lookback: 1` |
| `probe-cel-stream` | RoPE | **`decoder_stream: 64`** |
| `probe-cel` (Celeritas) | **NoPE** | **`decoder_stream: 64`** |

It comes out perfectly factorial, with no interaction, which makes the
attribution unambiguous.

### Quality, on data no arm read

Three disjoint held-out slices, two languages, 262,144 tokens each, as deltas
against the two-seed mean of each slice (`scripts/arm_table.py`):

| arm | ja@48M | ja@60M | en@40M | ctx nats |
|---|---:|---:|---:|---:|
| PHOTON (seed A) | +0.0323 | +0.0345 | +0.0021 | 0.323 / 0.327 / 0.325 |
| PHOTON (seed B) | −0.0323 | −0.0345 | −0.0021 | 0.372 / 0.386 / 0.306 |
| FENESTRA full | −0.1115 | −0.1579 | −0.1112 | 0.124 / 0.124 / 0.098 |
| Celeritas, `nope` only | −0.1114 | −0.1590 | −0.1083 | 0.128 / 0.123 / 0.100 |
| Celeritas, `stream` only | −0.2618 | −0.3121 | −0.2022 | 0.169 / 0.166 / 0.146 |
| **Celeritas** | **−0.2615** | **−0.3146** | **−0.2065** | 0.172 / 0.167 / 0.152 |
| two-seed spread | 0.0645 | 0.0689 | 0.0042 | |

*(The FENESTRA row reproduces the published −0.112 / −0.158 / −0.111 to the
fourth decimal, on a rebuilt measurement chain. That is the cross-check that
makes the rest of the table worth reading.)*

**`decoder_stream` is worth another −0.150 / −0.156 / −0.094 on top of
FENESTRA** — 2.3× the ja two-seed spread, same sign and similar magnitude on
three slices in two languages, and it is *cheaper* than what it replaces. It is
the second change in this project to clear the noise, and it clears it by more
than the first did.

**`pos: nope` is exactly neutral**: −0.1114 against −0.1115, −0.1590 against
−0.1579, −0.1083 against −0.1112. Train CE agrees to four decimals at every
logged step (4.7040 against 4.7028 at step 440), and the trajectories sit on top
of each other from step 100 onward.

`ctx nats` also partly repairs FENESTRA's one reproducible regression: 0.124 →
0.172 on ja, 0.098 → 0.152 on en. Still below PHOTON's 0.32–0.39, but moving
back the right way, and moving there while costing less.

### Length: does the model still work past its training shape?

`scripts/length_diag.py`. The same 256 tokens are scored at every setting and
only the amount of history in front of them changes, so the comparison is
between context lengths and not between slices. All arms trained at 512. CE
minus CE at the training length; **positive is degradation**:

| context | ×train | PHOTON | FENESTRA | `nope` only | `stream` only | **Celeritas** |
|---|---:|---:|---:|---:|---:|---:|
| 1024 | 2× | +0.0952 | +0.0049 | +0.0052 | 0.0000 | **−0.0001** |
| 2048 | 4× | +0.1739 | +0.0102 | +0.0105 | 0.0000 | **−0.0001** |
| 4096 | 8× | +0.2997 | +0.0194 | +0.0166 | 0.0000 | **−0.0001** |
| 8192 | 16× | **+0.4019** | **+0.0262** | **+0.0254** | **−0.0000** | **−0.0001** |

**PHOTON loses 0.40 nats at 16× its training length.** 0.376 of that is its
*global* level-1 encoder — RoPE over `T/4` = 2048 units against 128 trained.
FENESTRA bounded that stack with `encoder_window` for cache reasons, and
because RoPE is exactly relative, bounding the span also stopped the stack's
function depending on `T`. **A bounded span is a length-generalisation
mechanism, not only a memory one**, and FENESTRA got it without claiming it.

### The result that went the other way: NoPE does not fix the rest

`celeritas_probe_nope.yaml` isolates the positional change and was built to
explain FENESTRA's residual +0.0262. **It reads +0.0254.** The `stream`-only
arm, which keeps RoPE on the top encoder, is flat.

So, plainly: **at this scale, moving the global stack from RoPE to NoPE buys no
measurable length robustness, and no measurable quality.** "No position table to
extrapolate" is not the same as "length-invariant" — a NoPE causal stack still
learns its implicit sense of position from the length distribution it saw, and
attention over 512 units is out of distribution when it trained on 32. The
learned sink was the hedge against exactly that, and it did not cover it.

What change (1) *does* buy is real and independently verifiable, and none of it
is on either measured axis:

* the decoupled RoPE key is deleted — **−20% on the only cache that grows with
  `T`** (160 → 128 values per unit per layer);
* the learned sink becomes available on the weight-absorbed path at all;
* `rope_theta`, a YaRN factor and a context-extension phase stop being decisions
  for that stack, and `max_seq_len` stops being a limit — a *structural*
  property, asserted by a test.

Keep it for those, at zero measured cost. Do not sell it as the length fix.

### Where the context actually goes — and why the curve is flat

The unflattering explanation for a flat length curve is that the model stopped
depending on the far stream. `scripts/context_diag.py` looks like it refutes
that — nats lost when everything further back than `d` is replaced with noise:

| d | 512 | 256 | 128 | 64 | 32 | 16 |
|---|---:|---:|---:|---:|---:|---:|
| FENESTRA | 0.0000 | +0.0019 | +0.0099 | +0.0277 | +0.0741 | +0.1224 |
| `nope` only | 0.0000 | +0.0019 | +0.0112 | +0.0292 | +0.0773 | +0.1364 |
| **Celeritas** | 0.0000 | **−0.0001** | +0.0100 | +0.0316 | +0.0816 | **+0.1791** |

Celeritas leans on context **more** — 0.179 nats beyond 16 tokens against
FENESTRA's 0.122 — but read the bands, not the total. It is stronger everywhere
up to 128 tokens (0.0975 against 0.0483 in the 32→16 band) and uses **nothing at
all** between 256 and 512, where FENESTRA uses 0.0019. `decoder_stream` moved
the model's context consumption **inward**, into the window it can serve exactly
and cheaply.

The direct test settles it. Score the last 256 positions of one held-out
sequence with 512 tokens of history and again with 8192, and compare the
*distributions*:

| arm | max abs Δlogit | mean | KL(512 vs 8192) | argmax agreement |
|---|---:|---:|---:|---:|
| FENESTRA | 3.833 | 0.124 | 0.02105 | 91.4% |
| `nope` only | 3.354 | 0.170 | 0.03706 | 87.5% |
| **Celeritas** | **0.151** | **0.0032** | **0.00002** | **99.6%** |

**Celeritas' predictions barely move.** Essentially nothing from beyond ~512
tokens reaches the token prediction, so there is nothing out there to degrade.
**The flat length row is indifference, not superior generalisation**, and it
should be read that way.

Which reframes what the length column measures at this scale. FENESTRA's top
stream is very sensitive to sequence length — KL 0.021, 8.6% of argmaxes flip —
and that sensitivity is worth **+0.0019 nats of use against +0.0262 nats of
harm** at 16×. It is not a long-range channel doing work; it is an
out-of-distribution stack injecting noise. The length column is a
**noise-injection** measurement — which is still exactly what extrapolation
failure means operationally, and bounding the offending stack is still what
fixes it.

**Nothing here consumes meaningful context past a few hundred tokens *on ordinary text*** -- and 16 corrects the stronger reading this sentence used to carry. At 65M
parameters, 30M tokens and 512-token training contexts, nothing here consumes
meaningful context past a few hundred tokens, and nothing in this document
should be read as evidence that one of them would at scale.

> **Corrected by the copy probe.** That is true of *average* text and false of
> the model: plant something retrievable at 192-768 tokens and the hierarchy
> delivers **+0.85 nats**, in every arm, at distances beyond the training
> context. The corpus rarely asks; the channel answers when it does. See
> `findings.md` 16. This is the same
standing caveat FENESTRA carried, now measured sharply instead of suspected —
and it is why the roadmap below is about the hierarchy rather than about more
positional engineering.

### The `decoder_stream` sweep: the window *is* the length curve

The previous section left the mechanism behind the flat length row unestablished
and named the experiment that would test it — "sweep `decoder_stream` and watch
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

**4. And this refutes the hypothesis the previous section left standing.** That
section said the window was "the obvious first knob for anyone trying to get the
long-range channel back". It is not. The 256–512 band — the only context the
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
> same class of error as §4m's layer-positions: a statistic used one level past
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

### The confound control, in the weaker form that survives

These shards concatenate documents, so a longer context is also one more likely
to span an article boundary, and that inflates every row of every column equally
(the scored tokens are identical). So **any flat arm bounds it**: Celeritas
reads −0.0001, therefore the document-boundary effect is at most 0.0001 nats and
every number above is real model behaviour.

What the control can no longer do is attribute that behaviour to *positional
extrapolation*. The reading it was written for — "a model with nothing to
extrapolate measures the confound, so the remainder is extrapolation" — is
broken by the `nope` arm. For PHOTON's +0.40 the positional reading survives on
other grounds (bounding the offending stack removes 0.376 of it). For the +0.026
residual it is now known to be false.

## Fixing the five named disadvantages

The comparison against Transformers and Mamba named five. Three are fixed, and
two turned out not to need fixing — which is a result, not an evasion.

| # | disadvantage | outcome |
|---|---|---|
| D1 | still `O(T²)`, 1/256 constant | **fixed** — a third level makes it 1/4096: crossover 1.75M → **41.5M tokens**, KV@32k **−75%**, and a **4× larger local reach** nobody designed |
| D2 | granularity wall: 16 tokens → 1 vector | **measured; the wall is not where the argument put it.** 4× coarser costs **0.008 nats** of retrieval |
| D3 | fixed chunk grid cuts words | **priced at ~0 nats.** Do not build dynamic chunking at this scale |
| D4 | `T % C_<=L == 0` | **fixed** — right-pad and drop, exact by causality |
| D5 | no inference ecosystem | **partly fixed** — chunk-level self-speculative decoding, exact, mean **1.15×** at batch 1 |

### D1 — a third level

`configs/celeritas_probe_L3.yaml`. Built to make the comparison unambiguous
rather than favourable: **level 1 is byte-identical** to `celeritas_probe.yaml`,
the middle level is carved out of the old top level with `ffn_inter_size: 512`
to pay for itself, and **layer-positions per token land on exactly 5.875 in
both**. Parameters +0.44%.

**No code changed.** `_emit_group` recurses, `_alloc_caches` and `_prefill` loop
`1..L`, the accounting is per level — so a third level was a config file, and
`recgen` matched the training forward at 4.8e-06 the first time it ran.
`test_three_levels_need_no_new_code` keeps it that way.

| | L=2 | L=3 |
|---|---:|---:|
| held-out CE @2048 | 4.8128 | 4.8165 (**+0.0037**) |
| context used beyond 64 tokens | +0.0640 | +0.0632 |
| copy gain, d = 768…1536 | +1.2990 | +1.2906 |
| **attention overtakes everything else at** | 1.75M tokens | **41.5M** |
| **KV/token @ ctx 32768, exact `recgen`** | 0.1003 KiB | **0.0249 KiB** |
| non-global reach | 170 tokens | **682** |
| layer-positions / token | 5.875 | 5.875 |

+0.0037 against a two-seed spread of 0.065 is **unresolved**, and bounded far
below what the KV number would need to justify. The worry was the `c8x2`
precedent — fewer top units measurably weakened far-context use — and it does
not repeat: the far bands and the retrieval probe agree to three decimals.

**The part nobody designed.** A token's non-global reach is the maximum over
every span-bounded stack, and the deepest decoder dominates because its window
is measured in *coarser units*: 64 stream positions over level-1 units of 4
tokens is ~171 tokens; the same 64 over level-2 units of **16** tokens is ~683.
**Each level multiplies the local, exactly-cached, O(1)-per-token reach by
`C`** — for no extra layer-positions. It cuts both ways, and the numbers say
so: KL(2048 vs 8192) is 0.00052 at L=2 and **0.00000** at L=3. A deeper
hierarchy needs the global stream *less*.

### D2 / D3 — the retrieval probe, and a null result worth having

`scripts/copy_diag.py` plants a 16-token span, repeats it `d` tokens later, and
scores **only the second occurrence** against a control with an unrelated span
in the same slot. Everything but retrievability is held identical.

The control boundary is **not** the level-1 decoder window, and getting that
wrong invalidates the whole thing — see D1's last paragraph.
`bounded_reach_tokens()` computes it per architecture (16 / 64 / 171 / 682 for
PHOTON / FENESTRA / Celeritas / Celeritas L=3).

**Retrieval works, and past the training context.** Held-out ja, `seq_len` 1024,
mean over `d ∈ {192, 256, 384, 512, 768}`: PHOTON **+0.897**, FENESTRA
**+0.827**, Celeritas **+0.868** nats. At `seq_len` 4096 — which Celeritas
permits because it has no maximum context — a 512-trained checkpoint still
retrieves **+0.99 to +1.81 nats at 1024–3072 tokens**, i.e. at **6× its
training length**.

**The granularity wall is not measurable between 16 and 64 tokens per vector.**
L=3 makes one vector stand for 4× as much text and loses 0.0084 nats of
retrieval — 0.6% of the effect.

**The fixed grid costs -0.0000 to +0.0004 nats.** Each distance runs twice:
*aligned* (`d % C_<=L == 0`, so both occurrences are cut the same way) and
*offset* (`d % C_<=L = C/2`, same text on a different grid). Across five
arms the gap is at most 0.0018 on a single pair, against an effect size of 0.85.

> The script asserts that the offset arm **actually moves the span off the
> grid and actually changes the tokens** (96–124 positions differ). The headline
> here is a *null* result, and a silently-inert offset would look identical to
> one. It has already caught one bug — an over-strong alignment assertion that
> rejected a correct three-level sweep.

What this does **not** say: that dynamic chunking is worthless in general. One
tokenizer, 65M parameters, 16-token spans on a 16-token grid. A byte-level
model, where a chunk is a few characters and the grid cuts inside morphemes, is
a different question and this experiment does not touch it.

### D5 — the chunk is its own draft

`MTPModule` already learns `h_i, e(t_i) → t_{i+d}`, and the conditioning vector
already determines its chunk. So the drafter is free, has no attention in it,
and is exactly the right length; verification is **one** decoder pass, because
writing slots `0…C-2` in a single causal call yields the true slots `1…C-1` at
once. Two level-1 decoder calls per chunk instead of `C_1 = 4`.

Eight independent held-out prompts, greedy, batch 1: mean **1.15×**
(best 1.39×, worst 0.99×), acceptance **48.4%** (range 0–98%), and **every run
token-for-token identical to sequential decoding**.

Greedy speculation accepts only exact matches, so it is not an approximation of
sequential decoding — it *is* sequential decoding, and
`test_speculative_decoding_is_the_same_function_as_sequential` asserts equality
over 64 tokens rather than closeness. That is also what catches the real hazard:
rejected drafts are written into the rolling cache and must be un-written, and a
cache carrying drafted-but-wrong embeddings diverges several chunks later,
invisibly to any per-step tolerance.

**Report the spread, not the best row** — one prompt gave 1.50× and quoting it
would have been §4m's `--tail 5` mistake again.

Two structural limits: acceptance is **all-rows** (one shared cache cannot
advance rows by different amounts), so batch ≥ 4 gets 1.00–1.09× — this is a
*latency* optimisation. And it is **inert** under sampling, a repetition
penalty, or `forced_logits`, where exact-match acceptance would not be the
identity; `test_speculation_refuses_what_it_cannot_reproduce` pins each.

### D4 — any sequence length, exactly

`forward` right-pads to a multiple of `C_<=L` and drops the extra outputs.
Exact rather than approximate: the pad is *appended*, the chunker folds it into
units after every real one, and no position `< T` attends forward. Asserted to
2e-5, and to 0.000e+00 where the shapes coincide. Cost: at most `C_<=L - 1`
wasted positions, once. `PackedTokenDataset` keeps its own check — packing
efficiency is a different requirement, paid every window rather than once.

---

## What is not verified

n=1 per arm, 65M parameters, 30M tokens, 512-token training contexts. The
two-seed spread is one realisation of `|X1 − X2|` with no confidence attached,
so a delta smaller than it is **unresolved**, never equal — the discipline
`findings.md` §4m was written to enforce.

Specifically not run, and deliberately:

* an 8-run variance decomposition (init seed vs `data_seed`) to replace the
  two-seed gap with a real σ;
* a 250M iso-compute A/B — `celeritas_small.yaml` exists and is **untrained**;
* a *long-context training* run. Everything above trains at 512 and asks what
  happens at 8192. Whether Celeritas is better when *trained* long is a
  different question and this evidence does not touch it — and §"Where the
  context actually goes" says it is the question that matters, because no arm
  here has a long-range channel worth measuring;
* ~~a `decoder_stream` sweep~~ — **run**; see the sweep section. It refuted the
  reason it was listed for: narrowing the window does not push work back onto
  the hierarchy, it only makes the model worse and more length-fragile at once.
  What is still open is the same sweep at a training context where the far
  channel is worth something, which is the next item;
* whether `pos: nope` earns its keep at a scale where the top stream carries
  real information. It is neutral here on both measured axes, and here the top
  stream is nearly inert — those two facts are not independent.

## The roadmap this deliberately does not implement

**Dynamic chunking — priced, and not worth it here.**
[H-Net](https://arxiv.org/html/2507.07955v2) (Hwang, Wang & Gu, 2025) learns
chunk boundaries with a routing module and a straight-through estimator, and a
one-stage byte-level H-Net beats a BPE Transformer at matched compute. It is
the same shape as PHOTON and it breaks fixed-stride chunk-parallelism, the KV
accounting, and every exactness test in this repo.

`copy_diag.py` measures what it would buy before anyone builds it, by running
each retrieval distance on-grid and off-grid: **+0.0000 to +0.0004 nats across
five arms**, against an effect size of 0.85. At this scale the chunker is
robust to where the boundary falls — plausibly because the encoder above it
sees several chunks and can recompose. **So: no.**

That verdict is scoped. One tokenizer, 65M parameters, 16-token spans on a
16-token grid. At byte level, where a chunk is a few characters and the grid
cuts inside morphemes, the same experiment could easily come out differently —
and `copy_diag.py` is the thing to run first if anyone builds a byte-level
variant.

**What the roadmap is instead.** The measurements point somewhere else: every
arm retrieves ~0.9 nats through the hierarchy and ordinary text asks for it
almost never (`findings.md` 16). The open question is not how to chunk better,
it is **whether a corpus and a scale exist where the far channel earns its
keep** — long documents, code, multi-turn dialogue — and no amount of
architecture work answers that.
