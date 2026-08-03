# SCALA architecture notes

This document records the design decisions that are *not* obvious from the
paper summary, and the derivations that pin them down. Read it before changing
anything in `scala/model/hierarchy.py`.

---

## 1. Notation

| symbol | meaning |
|---|---|
| `T` | sequence length in tokens |
| `L` | number of hierarchy levels (2 here) |
| `C_l` | chunk size at level `l` (4 here, both levels) |
| `C_<=l` | `∏_{k<=l} C_k` — tokens summarised by one level-`l` unit (4, then 16) |
| `M_l` | `T / C_<=l` — number of level-`l` units |
| `D_l` | width at level `l`; `D_0` is the token width |
| `R_l` | Context Converter fan-out at level `l` |
| `X^(l)` | encoder states at level `l` (`X^(0)` = token embeddings) |
| `X̂^(l)` | decoder reconstructions at level `l` |

## 2. The four modules per level

```
A^(l) = C_θ^(l)( X^(l-1) )                       Context Chunker
X^(l) = F_θ^(l)( A^(l) )                         Context Encoder   (causal, M_l units)
U^(l)_g = U_θ^(l)( cond^(l)_g )                  Context Converter (1 vector -> R_l)
X̂^(l-1)_{I_g, j} = G_θ^(l)( U^(l)_g , X̂^(l-1)_{I_g, <j} )   Context Decoder
```

The decoder sees only `R_l + C_l` positions (8 for both levels here) and every
group `g` is decoded **in parallel** as a batch dimension. That bound — not the
sequence length — is what makes decoding cost `O(1)` in `T`.

> **The content input is hatted, and this file used to transcribe it without
> the hat.** That single character is the difference between "the decoder reads
> its own reconstructions" and "the decoder is teacher-forced with the encoder
> states", and it decides whether RecGen has a train/inference gap at all. The
> code follows the unhatted reading by default and the hatted one under
> `recursive_decoder_input`; §5 and §6b give the evidence for each.

## 3. The pipelining rule (the crux)

This is the part that is easy to get wrong, and getting it wrong produces a
model that either leaks the future or has a blind spot exactly one chunk wide.

**Rule.** The state a level-`l` decoder emits at slot `j` is the conditioning
for slot `j`'s contents one level down — consumed **unshifted**. Only the top
level shifts.

### Why unshifted below the top

The level-`l` decoder computes its slot-`j` output from

* `U^(l)_g`, derived from the level-`l` conditioning state for group `g`, and
* the level-`(l-1)` content units at slots `< j` of group `g`.

So slot `j`'s output has attended to everything through the end of group `g-1`
**plus** slots `0..j-1` of group `g` — which is exactly "everything strictly
before slot `j`". That is precisely the context needed to generate slot `j`'s
contents. No shift required; the shift is already baked into the causal mask.

### Why the top level shifts

`X^(L)_g` is an *encoder* output: it has already absorbed group `g`. Feeding it
back to decode group `g` would leak. So group `g` is conditioned on `X^(L)_{g-1}`,
with a learned start latent `X̂_0^(L)` at `g = 0`. That is `ScalaLevel.shift_cond`,
invoked only when `shift=True`.

### End-to-end check

Take token `i`, in level-1 unit `m = i // 4`, in meta-group `g = m // 4`, at
slot `j = m % 4`. The state that produces `logits_i` has seen:

* via the level-1 decoder's local window: tokens of unit `m` before `i`;
* via its conditioning `h_m` (the level-2 decoder's slot-`j` output): `X^(2)_{g-1}`
  (units `0 .. 4g-1`) and level-1 units `4g .. 4g+j-1`, i.e. units `0 .. m-1`.

Union = tokens `0 .. i-1`. Exactly `t_{<i}`. ∎

`tests/test_hierarchy.py::test_logits_do_not_depend_on_future_tokens` verifies both
halves numerically: perturbing token `p` must leave `logits[:, :p+1]` bit-stable
and must change something after it.

## 4. No shift in the loss

Because `X̂^(0)_i` has seen `t_{<i}`, the LM head on it predicts `t_i`:

```python
loss = cross_entropy(logits.flatten(0, 1), labels.flatten())   # no [:-1] / [1:]
```

Adding the usual `[:, :-1] / [:, 1:]` shift would double-shift and train the
model to predict two tokens ahead. If you ever see the loss plateau near
`ln(V)` while `L_rec` falls normally, check this first.

## 5. HierGen vs RecGen

All three protocols run the same recursive `emit_group`. They differ only in
what a finished group of level-`(l-1)` units becomes.

That group is consumed in **two** places, and the cheap protocols substitute in
both at once. `content` is what the level-`l` decoder reads at its next
position; `up` is what the level-`l` chunker folds for the encoder above.

| | content | up | encoder KV kept |
|---|---|---|---|
| `hiergen` | `F^(l-1)(C^(l-1)(group))` — the real encoder state | same | every level, unbounded |
| **`recgen`** | **the same real encoder state** | **same** | **every level, but bounded to a window** |
| `chunkgen` | `C^(l-1)(group)` — chunker only | same | top level only |
| `recgen_paper` | `X̂^(l-1)` — the decoder's own prediction | same | top level only |

HierGen is exact by construction: it hands the decoders the same encoder states
teacher forcing used. `test_hiergen_matches_the_training_forward` asserts this
to 2e-4 — though see `findings.md` §4g, where that assertion passed on every
run while the shipped inference path diverged, because the test config and the
zero-initialisation between them made the divergent module unreachable.

The paper's RecGen is exact **iff** `X̂^(L-1) = X^(L-1)` (Theorem A.6), and that
premise is not merely unmet but self-defeating — see below. Measured agreement
with the training forward is 31–54% across five matched arms, against HierGen's
98%.

**The shipped `recgen` therefore substitutes nothing.** RecGen's stated
contribution is a memory claim: the KV footprint drops from
`Σ_l O(T/C_≤l)` to `O(T/C_≤L)` because only the coarsest stream keeps growing.
That claim is satisfied by *bounding* the intermediate caches, not only by
deleting them, and bounding costs O(1) rather than a substitution. The window
is `enc_window_groups` blocks of `C_{l+1}` units; at 1 it is exactly the
`encoder_block_local` receptive field, and as it grows it converges to HierGen.
On a model trained with `encoder_block_local` the two coincide and `recgen`
reproduces the training forward bit-for-bit
(`test_recgen_is_exact_when_the_window_is_the_trained_receptive_field`).

### Why the substitution cannot be repaired by any objective

Under `recgen_paper`, `X̂^(L-1)_{I_{g+1},j}` is produced by the level-`L`
decoder from `U^(L)_g` and `X̂^(L-1)_{I_{g+1},<j}`, so the whole block — and
hence `Â^(L)_{g+1} = C^(L)(X̂^(L-1)_{I_{g+1}})` — is a deterministic function of
`X^(L)_g`, fixed *before* the meta-context is sampled. Under HierGen,
`A^(L)_{g+1}` is a function of the sampled tokens. Lemma A.5 asks these to be
equal, so it asks the top-level stream to be independent of everything the
model generates. Theorem A.6 is true; satisfying its premise empties the
hierarchy it is about.

Measured, on any weights: perturbing an emitted token moves later units'
logits by exactly `0.000e+00` under `recgen_paper`
(`test_recgen_generation_cannot_see_what_it_generated`).

**How much of that is the method and how much is this implementation is open,
and the paper does not settle it.** Equation (6) writes the decoder's content
input *hatted*:

```
X̂^(l-1)_{I_g,j} = G^(l)( U^(l)_{g-1} , X̂^(l-1)_{I_g,<j} ; M )
```

Read literally, no encoder state ever enters an upper decoder, so RecGen's
content substitution is not a substitution at all — it is what training did.
This repo instead teacher-forces those positions with `X^(l-1)`, computes the
slots in one parallel pass, and then patches the resulting gap from outside
with `L_rec`, `self_cond_prob` and `chunk_cond_prob`. The paper never says
which it means: Equation (6) sits in the *generation* section, and the
objective section says nothing about what the decoders consume during training.

`recursive_decoder_input: true` implements the literal reading (§6b). Under it
`test_recursive_input_makes_recgens_content_substitution_exact` shows the
`xhat_content` protocol reproducing the training forward exactly, which it
provably does not do under teacher forcing.

What survives either reading is the *upward* path: the top chunker consumes
`X^(L-1)` in training and `X̂^(L-1)` under RecGen. That is exactly Definition
A.4 and exactly what Theorem A.6 requires — nothing more. It is also the one
place the objective asks for something structurally hard: `X̂^(L-1)_m` is
produced before unit `m` is emitted, so matching it to the encoder summary of
tokens sampled afterwards is a prediction problem, and under sampling with
non-zero entropy it has no exact solution.

Three consequences worth internalising:

1. Of the two substitutions, `up` is the more damaging (36.1pp of lost
   agreement against 20.9pp under teacher forcing), and `chunk_cond_prob`
   cannot reach it at all — `encode_all` runs the real encoders bottom-up
   regardless. That is the signal to stop tuning substitutions.
2. `L_rec`, `self_cond_prob` and `chunk_cond_prob` exist only for the
   cache-free `chunkgen` path. Under `recgen` there is no substitution left for
   them to correct, and all three can be 0.
3. Choose by budget, not by protocol name: `recgen` for exact-or-near-exact
   generation at O(1) intermediate KV, `chunkgen` when even that is too much,
   `recgen_paper` only to reproduce the paper's rule. If you are training a new
   model, set `encoder_block_local: true` (§6) and `recgen` becomes exact.

## 6. Block-local intermediate encoders (an addition, not from the paper)

`LevelConfig.encoder_block_local` confines a level-`l` encoder (`l < L`) to the
`C_{l+1}` units that become one level-`(l+1)` unit: block-causal attention that
never crosses a meta-group boundary.

Nothing outside the current group is ever readable, so its KV cache holds one
group and is rewritten at every boundary — **O(1) in sequence length**, which is
exactly the saving RecGen was buying by discarding the cache. The difference is
that what gets handed up and across is still the real encoder state, so HierGen
stays bit-equal to the training forward.

Global context is not lost; it never lived in the intermediate encoders. The
top encoder keeps full attention, is amortised over `C_<=L` tokens, and reaches
the lower decoders through the converter.

On 250M v4 this moves HierGen from 0.945 to 0.207 KiB/token against the
paper-rule RecGen's 0.195 — exact generation for 6% more memory than
approximate generation, where it previously cost 4.8x more.

Generation rewinds the level-`l` cache at each group boundary. That is exact
rather than an approximation: attention is confined within the block and RoPE
is relative, so scoring a group at offsets `0..C-1` gives the same attention as
scoring it at its absolute offsets.

The same rewind is what `recgen` applies to a model that was *not* trained
block-local — there it truncates a receptive field the encoder really uses, so
it is an approximation, and `enc_window_groups` is the dial between
`encoder_block_local`'s cost and HierGen's fidelity. Trained block-local, the
dial does nothing because there is nothing left to truncate.

## 6b. Equation (6): hatted or not

`ScalaConfig.recursive_decoder_input` selects which reading of the paper's
Equation (6) the upper decoders are trained under.

`False` (default, and what every arm before round 4 used)
: the level-`l` decoder is teacher-forced with `X^(l-1)_{<j}` and all `C_l`
  slots are produced in one parallel pass. RecGen then feeds `X̂^(l-1)_{<j}`
  instead, which is a different function, and `L_rec` / `self_cond_prob` /
  `chunk_cond_prob` exist to close the difference.

`True`
: `ScalaLevel.decode_recursive` runs the recurrence the equation describes —
  `C_l` sequential decoder passes per group, the content positions holding the
  decoder's own outputs, gradient flowing through. RecGen's content
  substitution then *is* the training distribution.

Level 1 is unaffected either way: its content is the true token embeddings
under both readings, because generation feeds real embeddings there too.

Cost is small — the upper decoders are shallow, see at most `R_l + C_l`
positions, and the group axis stays fully parallel. Measured on GB10 at 250M:
10.0K tok/s against 10.7K.

Note what this does *not* do. `self_cond_prob` re-runs a decoder once on a
**detached** `X̂`, which is one step of this recurrence with the gradient cut —
it trains the decoder to tolerate its own output without ever training the
output it has to tolerate. That is why it is a patch and this is not.

### Why the default stays `False`

The literal reading has a consequence that is easy to miss and fatal once seen.

`X̂_0` is a function of `U^(l)_{g-1}` alone. `X̂_1` is a function of `U` and
`X̂_0`. By induction **every `X̂_j` in group `g` depends only on
`X^(L)_{g-1}`** — nothing emitted inside group `g` reaches any of them. So the
conditioning the level-1 decoder receives for unit `m` carries nothing about
units `4g .. m-1`, and a token is blind to the earlier units of its own
meta-group: `C_<=L - C_1` = **12 tokens** here, and 12 in the 8B too.

That is precisely the failure §3 calls "a blind spot exactly one chunk wide",
and `test_recursive_input_costs_within_group_context` pins it: perturbing a
token in an earlier unit of the same meta-group moves a later token's logits
under teacher forcing and does not move them under the recurrence, while the
next meta-group moves under both.

So the hats in Equation (6) are describing the *generation* recursion, not the
training rule. Teacher forcing is what makes the union in §3 come out to
exactly `t_{<i}`, and PHOTON's entire claim is that the hierarchy delivers full
context cheaply.

It also makes recursive consistency precise, and the news is bad.
Definition A.4 asks `X̂^(L-1)_{I_g} = X^(L-1)_{I_g}`. The left side is a
function of group `g-1`; the right side summarises tokens *inside* group `g`.
They can be equal only if every meta-group is determined by its predecessor —
zero conditional entropy over 16-token blocks. Theorem A.6 is therefore true
and vacuous: its premise is not merely unmet, it is unmeetable by any model
that has anything left to predict.

## 7. Scheduled sampling (an addition, not from the paper)

Training feeds the level-`L` decoder `X^(L-1)`; RecGen feeds it `X̂^(L-1)`. To
stop that substitution being a surprise at inference, `decode_all` optionally
re-runs the non-bottom decoders with a Bernoulli(`p`) mix of the two as content:

```python
mixed = torch.where(mask, xhat.detach(), content)
cond  = lvl.decode(cond, mixed, shift=(l == L))
```

Cost: one extra pass through the *upper* decoders only (3 layers over 8
positions, amortised ÷4). The level-1 decoder — the expensive per-token stack —
still runs exactly once. `L_rec` is always scored on the clean pass so the
target stays well defined.

`self_cond_prob` ramps 0 → 0.25 over the first half of training
(`self_cond_ramp_frac`).

## 8. Where the parameters go, and why

A level-`l` encoder runs once per `C_<=l` tokens. Its *cost* is therefore
divided by that factor while its *capacity* is not. So capacity belongs at the
top:

| stack | total | active | amortisation | amortised active |
|---|---|---|---|---|
| L1 encoder | 1055 M | 187 M | ÷4 | 47 M |
| L1 decoder | 399 M | 111 M | ÷1 | 111 M |
| L2 encoder | **5900 M** | 326 M | ÷16 | **20 M** |
| L2 decoder | 184 M | 78 M | ÷4 | 20 M |

The level-2 encoder holds 73% of the parameters and contributes 20 M to the
per-token cost. That trade is the entire point of the architecture.

The level-1 decoder is the opposite case — it runs every token with no
amortisation — which is why it is the shallowest MoE stack (5 layers, top-3).

### 8b. …and where the *compute* goes, which is not the same place

Parameters are amortised; layer-*positions* are what a token actually pays.
A level-`l` decoder layer runs over `R_l + C_l` positions to emit `C_l` units,
so it costs `(R_l + C_l) / C_l` layer-positions per level-`(l-1)` unit. On the
4x4 probe:

| stack | layer-positions / token | share |
|---|---:|---:|
| L1 encoder | 1.000 | 11.9% |
| **L1 decoder** | **6.000** | **71.6%** |
| L2 encoder | 0.375 | 4.5% |
| L2 decoder | 1.000 | 11.9% |

Two things follow, and neither is visible in the parameter table.

1. **`converter_width` is a multiplier on the most expensive stack.** The
   `R_l` converter positions go through every decoder layer and then
   `decode()` keeps only `out[:, R-1 : R-1+C]` — half the work at `R = C = 4`.
   They cannot simply be dropped (layer `n` attends to layer `n-1`'s values
   there, so the prefix evolves rather than being a static memory), but `R_l`
   itself has never been swept, and `4 -> 2` is -21% of the whole model's
   per-token compute for -0.9% parameters.

2. **Depth is 8x cheaper in the level-1 encoder than in the level-1 decoder**
   (0.25 against 2.0 layer-positions per token), and the encoder is the one
   with global context. Moving a layer up is parameter-matched and cheaper —
   the depth analogue of §1b of `findings.md`, "route where the tokens are".

Measured on the 65M probe, re-scored on held-out Japanese after the first
version of this table was retracted (`findings.md` §4m):

| change | stacks FLOP | total FLOP | KV/token | held-out CE vs 2-seed mean (2 slices) | verdict |
|---|---:|---:|---:|---:|---|
| `converter_width` 4 -> 2 | -21% | -5.4% | — | -0.009 / -0.044 | **unresolved** — inside the 0.065 spread, sign stable but small |
| 4x4 -> 8x2 (same 16-token meta-context) | -27% | -7.0% | **-40%** | -0.006 / +0.001 | **unresolved on CE**, but `ctx nats` falls 0.32 -> 0.21 on both slices |
| `rec_loss_alpha` 0.3 -> 0 | — | — | — | -0.029 / +0.009 | unresolved; zero-compute either way |
| one layer L1 decoder -> L1 encoder | -21% | -5.4% | — | +0.025 / +0.062 | **refuted** — worse on both slices |
| `global_skip` (direct top-level path) | — | — | — | -0.019 / +0.020 | refuted; the funnel is not the constraint |

Three things this table is not allowed to say any more, and why:

* **"-39% compute" was the stacks column.** `layer_positions_per_token`
  excludes the LM head, 74% of the forward FLOPs at `d_token=384`. Total FLOPs
  move -10.1% for all three changes together, and **measured throughput moves
  +1.1%** — at 65M the step is launch-bound. The head share falls with width
  (36% at the 8B config), so the same geometry projects to about -23% there.
* **"inside the seed floor" was a two-sample statistic** scored on 3,072
  English tokens the model had trained on. On held-out Japanese the two-seed
  spread is 0.065 and every arm is inside it on both slices, most flipping
  sign — nothing is resolved in either direction.
* **`c8x2` has one reproducible effect and it is negative.** Measured
  long-range context use drops from 0.32/0.37 nats to 0.210/0.211, identical
  to ±0.001 across two independent held-out slices. Halving the number of
  level-1 units feeding the top stream is exactly the mechanism.

What stands unconditionally: **-40% HierGen KV bytes/token** (an accounting
identity, confirmed by `gen.cache_bytes()`) and **-10.4% peak activation
memory** (measured). `configs/base_small_v5.yaml` applies the geometry and
is labelled unvalidated.

## 9. Attention choices per stack

* **Level-2 encoder → MLA.** It owns the long-lived global KV cache, and MLA
  stores `kv_lora_rank + qk_rope_head_dim = 320` elements/layer/unit versus
  GQA's `2 · 4 · 128 = 1024`. Over `T/16` units this is the cache that survives
  into RecGen, so it is worth the extra projection cost.
* **Level-1 encoder → GQA with 50% partial RoPE.** Global over `T/4` units;
  partial RoPE (ZAYA1 / Qwen3-Next) leaves half of each head position-free.
* **Both decoders → GQA with `rope_theta = 10_000`.** They only ever see 8
  positions. A million-scale theta would waste the entire rotary range on
  distances that never occur; small theta is the correct choice for a bounded
  local window.

## 10. Why there is no QK-Clip

Kimi K2 pairs Muon with QK-Clip because MLA without QK-Norm lets attention
logits drift, and Muon's orthogonalised updates can push them over. This model
enables **QK-Norm on every stack**, which bounds the logits directly rather than
repairing them post-hoc — the Qwen3 / MiniMax-M2 approach. Adding QK-Clip on top
would require tracking per-head max logits in the forward pass, which costs real
throughput for a problem QK-Norm has already solved.

The residual instability risk is covered by `GradNormGuard`, which drops a step
whose gradient norm exceeds 4× the running median (capped at 2% of steps).

## 11. Sequence-length constraint

**Note:** `max_seq_len` is *not* a maximum context length for a Celeritas
configuration; see §12. `T % C_<=L == 0` is enforced in `ScalaForCausalLM.forward`, in the dataloader
(`PackedTokenDataset`), and in the evaluation harness (which **left**-pads so the
scored continuation keeps its position at the end of the window). Anything that
feeds the model must respect it.

## 12. Celeritas: position is a property of a *span*, not of a model

Full design and measurements: `docs/celeritas.md`, `findings.md` §15.

§9 above lists the attention choice per stack. The observation Celeritas is
built on is that the same table already lists the *positional* choice, and one
row is unlike the others:

| stack | attends over | span bounded at build time? |
|---|---|---|
| level-1 encoder | `encoder_window` units (FENESTRA) | yes |
| level-1 decoder | `R_1 + C_1` positions, or `decoder_stream` | yes |
| level-2 decoder | same | yes |
| **level-2 (top) encoder** | **`T / C_<=L` units** | **no** |

RoPE is exactly relative — the score between a query and a key depends only on
their offset — so a stack whose span is bounded at `w` never evaluates RoPE at
an offset larger than `w`, wherever in the sequence it sits. Its function at
`T = 32768` is the *same function* it was trained with, not an extrapolation
of it. Only the top encoder is asked for something new, and it is asked for a
lot: 32 units at training against 2048 at 32K.

So there is exactly one stack whose positional scheme matters for length, and
it happens to be the retrieval stack — the role
[RNoPE-SWA](https://arxiv.org/html/2501.18795v1) assigns to its NoPE
full-attention layers, at a 1:3 interleave against RoPE sliding-window layers.
PHOTON supplies that same split one *stack* at a time instead of one *layer* at
a time, and does it more cheaply, because the global stack runs once per
`C_<=L` tokens: 0.375 layer-positions per token for a 6-layer top encoder.

`AttentionConfig.pos = "nope"` puts it there and nowhere else. Three
consequences, in decreasing order of how obvious they are:

1. **The model has no maximum context length.** `max_seq_len` becomes a
   *training shape* — it sizes the RoPE tables of the span-bounded stacks and
   nothing else. Measured: PHOTON +0.4019 nats at 16× its training length,
   FENESTRA +0.0262, Celeritas −0.0001 (§15a).
2. **MLA loses its decoupled RoPE key.** That key exists only because RoPE does
   not survive weight absorption; with no RoPE the channels are pure cache
   overhead, and `AttentionConfig.__post_init__` refuses them. The latent
   cache becomes `kv_lora_rank` alone: 160 → 128 values per unit per layer on
   the only cache that grows with `T`.
3. **A learned sink becomes available where it is needed.** The absorbed path
   used to refuse one; it softmaxes explicitly, so a sink is one extra column,
   and NoPE's documented failure mode — attention dispersing as the context
   grows — is exactly what a sink is for.

### 12b. The decoder was paying for its context twice

§8b measured that ~72% of the layer-position budget is in the level-1 decoder,
the one stack that sees eight positions, and that `decode()` computes `R + C`
positions and reads back `C`. FENESTRA's `decoder_lookback` then widened the
block to `R + (1+K)*C` to give slot 0 some real token history — which means the
previous chunk is recomputed at every layer of every group, having already been
computed as the previous group's content.

`decoder_stream` keeps `R + C` positions per group and concatenates the groups
into one causal stream with a sliding window, instead of batching them into
`M` blocks that cannot see each other. Same layout per group
(`[u_g; content_g]`), same readout (`out[R-1 : R-1+C]`), **no added positions**:

| | span/group | L1 decoder layer-pos/token | reach at slot 0 |
|---|---:|---:|---|
| PHOTON (`R=4`) | 8 | 6.000 | nothing |
| FENESTRA (`R=1, K=1`) | 9 | 6.750 | 1 chunk |
| Celeritas (`R=1, w=64`) | 5 | **3.750** | `w` positions ≈ 51 tokens |

Causality is unchanged: `u_g` is a function of `X_hat^(l)_{g-1}`, which has
attended to exactly `t_{<gC}`, and every stream position before it belongs to
an earlier group. Generation keeps one rolling cache of `w + R + C` entries per
streaming level — O(1) in `T` — and reproduces the training forward exactly,
by the argument §6 gives for the windowed encoder. The one new obligation is
that **prefill must run the decoders too** (`_prefill_decoders`), because the
window reaches across the prompt boundary.

Known limitation, shared with `encoder_window`: the windowed mask is built
dense, so a training-time forward at context `T` allocates a `(T(R+C)/C)²`
boolean mask for the level-1 decoder. Measured fine to 8192 tokens (1.9 GiB
peak at batch 2, fp32); a 128K training context would need a block-sparse or
FlexAttention mask instead. Generation is unaffected — its masks are
`window`-sized.
