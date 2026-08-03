# FENESTRA

> Naming note: this generation was originally called "PHOTON-2".  PHOTON is
> the name of the paper authors' architecture (arXiv:2512.20687), which this
> project only takes as a reference, so our own generations do not carry it.
> FENESTRA is Latin for *window* -- `encoder_window` is this generation's
> change -- continuing the lineage's Latin naming (Celeritas, SCALA).
> Internal identifiers ("p2" in config and run names) keep the old
> abbreviation so existing measurement artifacts stay traceable.

A revision of the reimplemented base architecture in which every change is
traceable to a measurement in `findings.md`, and every claim is stated with
what would falsify it.

The one-line summary: **FENESTRA makes the training-time receptive field equal
to the generation-time cache budget.** Once those are the same number, exact
generation at `O(T/C_<=L)` global KV is free, and the entire apparatus PHOTON
built to approximate it — `L_rec`, `chunk_cond_prob`, `self_cond_prob`, the
`X_hat` substitution — has nothing left to do.

---

## What PHOTON gets right, and FENESTRA keeps

Not everything needed changing. Verified this session:

* **The hierarchy delivers real long-range context.** Replacing everything
  further back than one meta-context with noise costs 0.34 nats
  (`scripts/context_diag.py`). It is not a 16-token-window LM with decorative
  upper levels.
* **HierGen is bit-exact with the training forward** and is machine-checked to
  be. That property is what made every other question answerable.
* **Amortisation is real.** A level-`l` encoder runs once per `C_<=l` tokens,
  so capacity at the top is nearly free per token, and the top-level KV is
  `O(T/C_<=L)`.
* **Chunk-parallel local decoding**, with an attention span independent of `T`.

FENESTRA changes nothing about any of these.

---

## Change 1 — sliding-window intermediate encoders (`encoder_window`)

**The problem.** The intermediate encoders' KV caches grow with `T`. That is
the *only* reason RecGen's substitution was ever proposed, and the substitution
does not work: the summary it hands upward is fixed before the meta-context is
sampled, so measured propagation from an emitted token to any later unit is
exactly `0.000e+00` on any weights (§4l). Theorem A.6 is true and its premise
is self-defeating — `A_hat` and `A` can only be equal if the top-level stream
ignores everything the model generates, and `L_rec` has already closed 94% of
the gap that *is* closable (`scripts/bottleneck_entropy.py`).

**The first repair, and its cost.** Bounding the cache instead of deleting it
kept the memory claim and dropped the substitution. But bounding by *tiling* —
recycling the buffer at block boundaries — gives a unit between 0 and `W` units
of history depending where it lands. Measured per-token KL across the tile at
`enc_window_groups=4`: **0.19 just after a boundary against 0.035 just before
the next, a 4.1x sawtooth** that widens as the window widens.

**FENESTRA.** Train the intermediate encoder with a causal *sliding* window of
`encoder_window` units (`AttentionConfig.window`, which already existed), and
generate with a cache that rolls instead of recycling. Every unit sees exactly
`encoder_window` units, at training and at generation, so:

* **generation is bit-exact under `recgen`**, the protocol whose only growing
  cache is the top level's
  (`test_hierarchy2_generation_is_exact_under_both_protocols`);
* there is no tile boundary, so no sawtooth;
* `rec_loss_alpha`, `chunk_cond_prob` and `self_cond_prob` can all be 0 —
  there is no substitution left for them to correct.

**How the roll stays exact.** Keys carry the RoPE phase of their absolute
position and the roll does not touch them; only the *write index* moves. So
`TransformerStack.forward` now takes `cache_start` separately from
`pos_offset`, and the cache holds `window + one write block` entries, rolled
back to `window` after each write. Buffer-index differences then equal
absolute-position differences, and the trained window mask lands on exactly the
right keys — which is why this is an identity rather than an approximation.

Against `encoder_block_local`, which FENESTRA supersedes: same O(1) footprint,
same exactness, but `W` units of history everywhere instead of between 0 and
`C_{l+1}`, and `W` is a free parameter rather than the chunk size.

## Change 2 — token lookback in the local decoder (`decoder_lookback`)

**The problem.** CE by slot inside a level-1 chunk (`scripts/position_diag.py`):

```
slot 0: 5.0152   slot 1: 4.9409   slot 2: 4.9116   slot 3: 4.8960
```

**+0.119 nats for the first token of a chunk against the last**, and flat
across chunks — so the loss is token-level detail, not global context. Slot 0
has no token-level context at all: everything it knows arrives through `R_l`
conditioning vectors derived from a single summary.

**Why more `converter_width` is not the fix.** `R_l` fan-out positions are a
deterministic expansion of one vector; they add decoder positions but no
information. That is also why they are the most expensive thing in the model
that can be removed: the level-1 decoder is ~72% of the layer-position budget
and `decode()` discards `R-1` of its `R+C` outputs.

**FENESTRA.** Spend those positions on the previous chunk's *actual tokens*.
`converter_width: 1` with `decoder_lookback: 1` gives a span of `1 + C + C = 9`
against the default `4 + C = 8` — one extra position, three fan-out copies
traded for `C` real tokens. Attention stays bounded and independent of `T`, so
chunk-parallel decoding and the KV story are untouched; generation keeps a
rolling buffer of the last `C_1` token embeddings, seeded from the prompt so
the prefill seam is exact too.

Causality is unchanged and machine-checked: the previous chunk is strictly
earlier text, and
`test_decoder_lookback_widens_the_window_without_leaking` asserts both that
nothing leaks backwards and that perturbing the last token of chunk `g-1` now
*does* move chunk `g`'s first logit — the blind spot is actually gone, not just
covered.

## Change 3 — the measurement foundation

Not architecture, but the reason the first version of §4m was wrong in three
independent ways. All three are now structural rather than a matter of
remembering:

| defect | fix |
|---|---|
| `hash(src.name)` seeded the shard shuffle, and Python salts it per process — the same `seed` drew a different shard on different runs, putting a 0.20-nat spread into what was read as seed noise | `zlib.crc32`, verified stable across interpreters |
| `seed` drove both initialisation and data order, so an A/B that varied the seed varied the text too | `TrainConfig.data_seed`, defaulting to `seed`; fix one and vary the other to separate the two variances |
| the corpus had no holdout at all, and `evaluate()` pulled from the training iterator | `PackedTokenDataset(holdout_frac=, split=)` reserves a tail of every shard; `TrainConfig.holdout_frac` defaults to 0.02 and `[eval]` now reads it |

Also: `flops_per_token` counts the LM head and the decoder's `R + (1+K)*C`
span, both of which it previously priced at zero, and
`layer_positions_per_token`'s docstring says in as many words that it is not a
compute metric.

---

## What is verified, and what is not

**Verified by construction, on any weights** (`tests/test_hierarchy.py`):

* `recgen` on a `encoder_window` model reproduces the training forward to
  2e-4 — the same tolerance HierGen meets — with a lower-level cache of
  `window + block` entries regardless of sequence length.
* Per-position error under that protocol is flat: no tile boundary.
* `decoder_lookback` leaks nothing backwards and does reach slot 0.
* The holdout split is disjoint from the training split.

**Measured on a trained checkpoint** (`runs/probe-p2/final`, sliding window,
16 units): `recgen` — the protocol whose only growing cache is the top level's
— scores **100.0% agreement and KL 0.0000** against the training forward, at
every window setting, with `cos(X_top) = 1.0000`. The paper's rule on the same
weights: 26.8% / KL 2.190. This is the claim RecGen was for, delivered.

**Quality, on data no arm read.** Three disjoint held-out slices, two languages,
262,144 tokens each, as deltas against the two-seed mean of each slice:

| arm | ja@48M | ja@60M | en@40M | ctx nats |
|---|---:|---:|---:|---:|
| probe (PHOTON, seed A) | +0.032 | +0.035 | +0.002 | 0.323 / 0.327 / 0.325 |
| probe (PHOTON, seed B) | -0.032 | -0.035 | -0.002 | 0.372 / 0.386 / 0.306 |
| FENESTRA minimal (window, R=2) | -0.016 | -0.051 | +0.001 | 0.162 / 0.156 / 0.126 |
| — same, but *tiled* (block-local) | +0.015 | -0.016 | +0.021 | 0.190 / 0.191 / 0.142 |
| **FENESTRA full (+ lookback, R=1)** | **-0.112** | **-0.158** | **-0.111** | 0.124 / 0.124 / 0.098 |
| two-seed spread | 0.065 | 0.069 | 0.004 | |

**`decoder_lookback` is the first architecture change in this project to clear
the noise.** -0.11 to -0.16 nats, same sign and similar magnitude on all three
slices in two languages, against a two-seed spread of 0.065. Everything else
measured — `global_skip`, `deepenc`, `c8x2`, `r2`, `norec`, and the window
alone — sat inside that spread and mostly changed sign between slices.

The mechanism check agrees. `scripts/position_diag.py` on held-out Japanese:

| | slot 0 | slot 1 | slot 2 | slot 3 | slot 0 - slot 3 | meta-group first vs last |
|---|---:|---:|---:|---:|---:|---:|
| PHOTON | 5.2719 | 5.1975 | 5.1683 | 5.2146 | **+0.057** | **+0.175** |
| FENESTRA full | 5.0553 | 5.0696 | 5.0652 | 5.0887 | **-0.033** | **+0.038** |

The chunk-start penalty is gone, and the meta-group-boundary penalty falls 4.6x.
That is the thing `decoder_lookback` was built to fix, fixed.

**Sliding beats tiled**, but not by enough to claim on CE alone: -0.031 /
-0.035 / -0.020 across the three slices, same sign every time, inside the ja
spread and outside the en one. The argument for it rests on the mechanism and
on the 4.1x sawtooth measurement, not on this column.

### The caveat, equally reproducible

**FENESTRA uses measurably less long-range context.** `ctx nats` falls from
0.32-0.39 to 0.098-0.124 — reproduced on all three slices — and the full
variant is lowest. Two causes, both structural: the level-1 encoder is now
confined to `encoder_window` units, so only the top stream carries anything
global; and `decoder_lookback` supplies locally what the hierarchy used to be
asked for.

At 512-token contexts that trade is clearly favourable — CE is 0.11 nats
better. **It need not stay favourable at 8K or 32K**, which is where PHOTON is
aimed, and nothing here measures that. The obvious guard is to widen
`encoder_window` (it is a free parameter, and the cache cost is linear in it)
and re-run the ablation; the obvious experiment is a long-context arm.

### What is still not verified

n=1 per arm, 65M parameters, 30M tokens, 512-token contexts. The two-seed
spread is one realisation of `|X1 - X2|` with no confidence attached, so
"1.7x the spread" is suggestive, not significant. Settling it needs what
`findings.md` §4m lists: an 8-run variance decomposition to replace that gap
with a real sigma, and a 250M iso-compute A/B — plus, specific to FENESTRA, a
long-context run to test whether the `ctx nats` drop matters.

## Cost

Probe config (65M, `d_token` 384, 4x4), forward FLOPs/token:

| | L1 decoder span | layer-pos/tok | stacks | total | note |
|---|---:|---:|---:|---:|---|
| PHOTON (`R=4`) | 8 | 8.375 | 26.5 M | 103.2 M | |
| FENESTRA minimal (`R=2`, window) | 6 | 6.625 | 21.0 M | 97.7 M | -5.4% |
| FENESTRA full (`R=1`, window, lookback) | 9 | 8.875 | 28.1 M | 104.8 M | **+1.5%** |

The full variant is *not* cheaper. It buys exact `O(T/C_<=L)` generation and
`C_1` real tokens of context at slot 0, for 1.5% more forward compute — and at
`d_token=384` that 1.5% is measured against a total the LM head owns 74% of, so
it is 5.7% of the stacks. State it that way; the retraction in §4m happened
because the stacks column was quoted as if it were the total.

---

## Superseded, in part, by Celeritas

`docs/celeritas.md`. Two of FENESTRA's three changes survive unchanged and one
is replaced:

* **`encoder_window` is kept**, and turned out to do more than it claimed.
  Bounding the intermediate encoder's span for *cache* reasons also bounded its
  RoPE offsets, and RoPE is exactly relative — so FENESTRA also removed the
  larger of PHOTON's two length-extrapolating stacks, without noticing.
  Measured at 16× the training length: PHOTON +0.4019 nats, FENESTRA +0.0262
  (`findings.md` §15a). That is a side effect worth naming.
* **The measurement foundation is kept** in full.
* **`decoder_lookback` is replaced by `decoder_stream`.** It was the right
  diagnosis — slot 0 has no token-level context and only real tokens fix it —
  and an expensive cure: widening the block to `R + (1+K)*C` recomputes the
  previous chunk at every layer of every group. Concatenating the groups into
  one windowed stream instead costs `R + C` positions per group, i.e. **1.25
  layer-positions per token against this document's 2.25**, and reaches
  `window` positions back instead of `C_1`.

The `ctx nats` caveat above (0.32–0.39 → 0.10–0.12) still stands and is not
resolved by Celeritas; see `docs/celeritas.md` for what is measured there.
