"""Correctness tests for the SCALA model: causality, generation, caches."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.infer.generate import (  # noqa: E402
    GenerationConfig, ScalaGenerator, sample_from_logits,
)
from scala.model.accounting import count_model  # noqa: E402
from scala.model.config import ScalaConfig  # noqa: E402
from scala.model.moe import (  # noqa: E402
    Router, update_expert_biases,
)
from scala.model.hierarchy import ScalaForCausalLM  # noqa: E402

TINY = Path(__file__).resolve().parents[1] / "configs" / "base_tiny.yaml"


@pytest.fixture(scope="module")
def model() -> ScalaForCausalLM:
    torch.manual_seed(0)
    cfg = ScalaConfig.load(TINY)
    m = ScalaForCausalLM(cfg).eval()
    return m


def _shake_zero_init_parameters(m: ScalaForCausalLM, seed: int = 11) -> None:
    """Give every zero-initialised parameter a random value: a zero parameter
    makes its module a no-op (e.g. ``2*sigmoid(0) == 1``), so equivalence
    tests could pass on a path that drops the module."""
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _, p in m.named_parameters():
            if p.abs().max() == 0:
                p.copy_(torch.randn(p.shape, generator=g) * 0.05)
    for mod in m.modules():
        if hasattr(mod, "invalidate_absorbed"):
            mod.invalidate_absorbed()


@pytest.fixture(scope="module")
def absorbed_model() -> ScalaForCausalLM:
    """Tiny model matching production top-encoder settings: ``qk_norm`` off
    (enables the absorbed latent cache) and ``output_gate`` on."""
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = ScalaConfig.load(TINY)
    top = cfg.levels[-1]
    cfg.levels[-1] = replace(
        top,
        encoder=replace(
            top.encoder,
            attention=replace(top.encoder.attention, qk_norm=False,
                              output_gate=True),
        ),
    )
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)
    return m


def test_absorbed_mla_cache_matches_the_uncached_path(absorbed_model):
    """The weight-absorbed latent cache is an algebraic identity, so it must
    reproduce the ordinary attention path to floating-point tolerance."""
    stack = absorbed_model.levels[-1].encoder
    assert stack.layers[0].attn.supports_latent_cache, \
        "fixture did not enable absorption -- the test would prove nothing"

    torch.manual_seed(5)
    a = torch.randn(2, 12, stack.cfg.d_model)

    ref = stack(a)                                   # training path, no cache
    cache = stack.alloc_cache(2, 16, torch.device("cpu"), torch.float32)
    inc = torch.cat([stack(a[:, i : i + 1], cache=cache, pos_offset=i)
                     for i in range(a.shape[1])], dim=1)

    torch.testing.assert_close(inc, ref, atol=2e-4, rtol=2e-4)


@pytest.fixture(scope="module")
def block_local_model() -> ScalaForCausalLM:
    """Tiny model whose intermediate encoder is confined to one meta-group."""
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = ScalaConfig.load(TINY)
    cfg.levels[0] = replace(cfg.levels[0], encoder_block_local=True)
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)
    return m


def test_block_local_encoder_ignores_earlier_groups(block_local_model):
    """The span restriction has to be real: editing a unit in an earlier
    meta-group must not move this group's encoder states."""
    lvl = block_local_model.levels[0]
    assert lvl.encoder_block == block_local_model.cfg.levels[1].chunk_size

    torch.manual_seed(7)
    a = torch.randn(1, 8, lvl.d_here)      # 2 groups of 4 units
    ref = lvl.encoder(a)

    a2 = a.clone()
    a2[:, 0] += 3.0                        # perturb group 0 only
    got = lvl.encoder(a2)

    assert not torch.allclose(got[:, :4], ref[:, :4]), "group 0 should move"
    torch.testing.assert_close(got[:, 4:], ref[:, 4:], atol=1e-6, rtol=1e-6)


@pytest.fixture(scope="module")
def recursive_model() -> ScalaForCausalLM:
    """Tiny model trained the way the paper's Equation (6) reads."""
    torch.manual_seed(0)
    cfg = ScalaConfig.load(TINY)
    cfg.recursive_decoder_input = True
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)
    return m


def test_recursive_input_makes_recgens_content_substitution_exact(recursive_model):
    """With `recursive_decoder_input` (Equation (6): no encoder state enters an
    upper decoder's input), `xhat_content` must reproduce the training forward;
    under teacher forcing it must not, or the flag does nothing."""
    cfg = recursive_model.cfg
    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (2, 64))

    ref = recursive_model(x).logits[:, 16:]
    gen = ScalaGenerator(recursive_model, device="cpu", dtype=torch.float32)
    got = gen.forced_logits(x, mode="xhat_content", prefix_meta=1)
    torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)

    # control: teacher forcing must not have this property
    torch.manual_seed(0)
    tf_cfg = ScalaConfig.load(TINY)
    tf = ScalaForCausalLM(tf_cfg).eval()
    _shake_zero_init_parameters(tf)
    tf_ref = tf(x).logits[:, 16:]
    tf_got = ScalaGenerator(tf, device="cpu", dtype=torch.float32).forced_logits(
        x, mode="xhat_content", prefix_meta=1)
    assert not torch.allclose(tf_got, tf_ref, atol=2e-4, rtol=2e-4)


def test_recursive_input_costs_within_group_context(recursive_model, model):
    """Under Equation (6) every ``X_hat_j`` in group ``g`` is a function of
    ``X^(L)_{g-1}`` only, so a token is blind to earlier units of its own
    meta-group (``C_<=L - C_1`` = 12 tokens here); teacher forcing is not."""
    cfg = recursive_model.cfg
    cp, c1 = cfg.chunk_product, cfg.levels[0].chunk_size
    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (1, 64))

    perturbed = 2 * cp                      # first token of meta-group 2
    later = 2 * cp + 3 * c1 + 1             # a later *unit* of that same group
    nxt = 3 * cp + 1                        # the following meta-group
    x2 = x.clone()
    x2[0, perturbed] = (x[0, perturbed] + 1234) % cfg.vocab_size

    def moves(m, i):
        return (m(x2).logits[0, i] - m(x).logits[0, i]).abs().max().item()

    # teacher forcing: the earlier unit reaches the later one
    assert moves(model, later) > 1e-4

    # Equation (6) literally: it does not
    assert moves(recursive_model, later) < 1e-5
    # ...while the next meta-group still moves, so the model is not simply dead
    assert moves(recursive_model, nxt) > 1e-4


def test_block_local_survives_a_config_round_trip(tmp_path):
    """`encoder_block_local` must survive the `model_config.json` round trip;
    exported checkpoints carry that file, not the YAML."""
    from dataclasses import replace

    cfg = ScalaConfig.load(TINY)
    cfg.levels[0] = replace(cfg.levels[0], encoder_block_local=True)
    block = ScalaForCausalLM(cfg).levels[0].encoder_block
    assert block == cfg.levels[1].chunk_size

    cfg.save(tmp_path / "model_config.json")
    reloaded = ScalaConfig.load(tmp_path / "model_config.json")
    assert ScalaForCausalLM(reloaded).levels[0].encoder_block == block


def test_block_local_hiergen_is_exact_and_bounded(block_local_model):
    """HierGen on a block-local model equals the training forward, and the
    level-1 encoder cache no longer grows with the sequence."""
    cfg = block_local_model.cfg
    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (2, 96))

    ref = block_local_model(x).logits[:, 16:]
    gen = ScalaGenerator(block_local_model, device="cpu", dtype=torch.float32)
    got = gen.forced_logits(x, mode="hiergen", prefix_meta=1)
    torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)

    prompt = torch.randint(0, cfg.vocab_size, (1, 32))
    sizes = []
    for n_new in (16, 64):
        gen.generate(prompt, GenerationConfig(max_new_tokens=n_new,
                                              mode="hiergen", greedy=True))
        sizes.append(gen.cache_bytes())
    # top encoder cache still grows with T; level 1 stays at one block
    assert sizes[1] > sizes[0]
    assert gen.state[1].enc_cache.k[0].shape[2] == lvl_block(block_local_model)


def lvl_block(model) -> int:
    return model.levels[0].encoder_block


@pytest.fixture(scope="module")
def windowed_model() -> ScalaForCausalLM:
    """Tiny model whose intermediate encoder slides over a fixed window."""
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = ScalaConfig.load(TINY)
    cfg.levels[0] = replace(cfg.levels[0], encoder_window=6)
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)
    return m


def test_sliding_window_encoder_is_exact_and_has_no_tile_boundary(windowed_model):
    """A sliding bound gives every unit exactly `encoder_window` units of
    history, and generation reproduces it bit-exactly: the cache rolls while
    RoPE stays absolute, so relative phases are preserved."""
    cfg = windowed_model.cfg
    lvl = windowed_model.levels[0]
    assert lvl.encoder_window == 6 and lvl.encoder_block is None
    assert lvl.encoder.cfg.attention.window == 6
    assert windowed_model.levels[-1].encoder_window is None, "top must stay global"

    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (2, 160))
    ref = windowed_model(x).logits[:, 16:]
    gen = ScalaGenerator(windowed_model, device="cpu", dtype=torch.float32)

    for mode in ("hiergen", "recgen"):
        got = gen.forced_logits(x, mode=mode, prefix_meta=1)
        torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)

    # per-position error must be flat: no tile boundary
    err = (gen.forced_logits(x, mode="recgen", prefix_meta=1) - ref).abs()
    per_unit = err.amax(-1).reshape(-1, cfg.chunk_product).mean(0)
    assert per_unit.max() < 2e-4

    # the cache is `window + one write block`, whatever the sequence length
    V = cfg.vocab_size
    widths = []
    for n_new in (16, 256):
        gen.generate(torch.randint(0, V, (1, 32)),
                     GenerationConfig(max_new_tokens=n_new, mode="recgen",
                                      greedy=True))
        widths.append(gen.state[1].enc_cache.k[0].shape[2])
    assert widths[0] == widths[1] == 6 + cfg.levels[1].chunk_size


@pytest.fixture(scope="module")
def photon2_model() -> ScalaForCausalLM:
    """Sliding-window encoder + token-level decoder lookback."""
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = ScalaConfig.load(TINY)
    cfg.levels[0] = replace(cfg.levels[0], encoder_window=6, decoder_lookback=1)
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)
    return m


def test_decoder_lookback_widens_the_window_without_leaking(photon2_model):
    """`decoder_lookback` gives slot 0 of a chunk the previous chunk's real
    tokens; causality and equality with the training forward must hold."""
    cfg = photon2_model.cfg
    lvl = photon2_model.levels[0]
    assert lvl.lookback == 1 and lvl.start_content is not None
    assert photon2_model.levels[-1].lookback == 0, "level 1 only"

    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (1, 160))
    base = photon2_model(x).logits
    for pos in (0, 5, 16, 31, 47, 96):
        y = x.clone()
        y[0, pos] = (y[0, pos] + 7) % cfg.vocab_size
        alt = photon2_model(y).logits
        assert torch.allclose(base[:, : pos + 1], alt[:, : pos + 1], atol=1e-4), \
            f"lookback leaked token {pos} backwards"
        assert not torch.allclose(base[:, pos + 1 :], alt[:, pos + 1 :], atol=1e-4)

    # the previous chunk must reach slot 0: perturbing the last token of
    # chunk g-1 has to move chunk g's first logit
    C = lvl.chunk
    g0 = 8 * C                                   # a chunk boundary, past prefill
    y = x.clone()
    y[0, g0 - 1] = (y[0, g0 - 1] + 11) % cfg.vocab_size
    moved = (photon2_model(x).logits[0, g0] - photon2_model(y).logits[0, g0]) \
        .abs().max().item()
    assert moved > 1e-3, "slot 0 still cannot see the previous chunk"


def test_hierarchy2_generation_is_exact_under_both_protocols(photon2_model):
    """`recgen` -- only the top-level cache growing with T -- reproduces the
    training forward exactly, with no substitution and no tile boundary."""
    cfg = photon2_model.cfg
    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (2, 160))
    ref = photon2_model(x).logits[:, 16:]
    gen = ScalaGenerator(photon2_model, device="cpu", dtype=torch.float32)
    for mode in ("hiergen", "recgen"):
        got = gen.forced_logits(x, mode=mode, prefix_meta=1)
        torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)

    # and the lower cache stays at window + one block whatever the length
    V = cfg.vocab_size
    widths = []
    for n_new in (16, 256):
        gen.generate(torch.randint(0, V, (1, 32)),
                     GenerationConfig(max_new_tokens=n_new, mode="recgen",
                                      greedy=True))
        widths.append(gen.state[1].enc_cache.k[0].shape[2])
    assert widths[0] == widths[1] == 6 + cfg.levels[1].chunk_size


# --------------------------------------------------------------------------- #
# Celeritas
# --------------------------------------------------------------------------- #
def _celeritas_cfg(**overrides):
    """Tiny geometry: NoPE top, windowed L1 encoder, streaming decoders."""
    from dataclasses import replace

    cfg = ScalaConfig.load(TINY)
    top = cfg.levels[-1]
    cfg.levels[-1] = replace(
        top, decoder_stream=24, converter_width=2,
        encoder=replace(
            top.encoder,
            attention=replace(top.encoder.attention, pos="nope",
                              mla_qk_rope_head_dim=0, mla_qk_nope_head_dim=32,
                              qk_norm=False, output_gate=True, attn_sink=True),
        ),
    )
    cfg.levels[0] = replace(cfg.levels[0], encoder_window=6, decoder_stream=32,
                            converter_width=1, **overrides)
    return cfg


@pytest.fixture(scope="module")
def celeritas_model() -> ScalaForCausalLM:
    torch.manual_seed(0)
    m = ScalaForCausalLM(_celeritas_cfg()).eval()
    _shake_zero_init_parameters(m)
    return m


def test_streaming_decoder_is_exact_and_reaches_past_one_chunk(celeritas_model):
    """`decoder_stream` concatenates each group's `R_l + C_l` positions into
    one windowed sequence: identical position count, `window` positions of
    reach.  Needs no leakage, reach past one chunk, and exact generation."""
    cfg = celeritas_model.cfg
    lvl = celeritas_model.levels[0]
    assert lvl.stream == 32 and lvl.decoder.cfg.attention.window == 32
    assert lvl.lookback == 0, "streaming replaces lookback, not augments it"
    V, C = cfg.vocab_size, lvl.chunk

    # exhaustive sweep: the stream reads across group boundaries, so every
    # position is checked for backward leakage
    torch.manual_seed(1)
    x = torch.randint(0, V, (1, 64))
    base = celeritas_model(x).logits
    worst, worst_pos = 0.0, -1
    for pos in range(x.shape[1]):
        y = x.clone()
        y[0, pos] = (y[0, pos] + 7) % V
        alt = celeritas_model(y).logits
        back = (base[:, : pos + 1] - alt[:, : pos + 1]).abs().max().item()
        if back > worst:
            worst, worst_pos = back, pos
        if pos + 1 < x.shape[1]:          # the last token has no future to move
            assert not torch.allclose(base[:, pos + 1 :], alt[:, pos + 1 :],
                                      atol=1e-5), f"token {pos} changed nothing"
    assert worst < 1e-5, \
        f"the decoder stream leaked token {worst_pos} backwards ({worst:.2e})"

    x = torch.randint(0, V, (1, 160))

    # slot 0 of a chunk must see real tokens several chunks back
    g0 = 12 * C
    reach = []
    for back in (1, 2, 4):
        y = x.clone()
        p = g0 - back * C
        y[0, p] = (y[0, p] + 11) % V
        reach.append((celeritas_model(x).logits[0, g0]
                      - celeritas_model(y).logits[0, g0]).abs().max().item())
    assert min(reach) > 1e-3, f"stream window does not reach back: {reach}"


def test_celeritas_generation_is_exact_under_both_protocols(celeritas_model):
    """NoPE top read through the absorbed latent cache with a learned sink,
    sliding-window intermediate encoder, streaming decoders across the prefill
    seam: `recgen` must reproduce the training forward."""
    cfg = celeritas_model.cfg
    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (2, 160))
    ref = celeritas_model(x).logits[:, 16:]
    gen = ScalaGenerator(celeritas_model, device="cpu", dtype=torch.float32)
    for mode in ("hiergen", "recgen"):
        got = gen.forced_logits(x, mode=mode, prefix_meta=1)
        torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)
        # flat per-position error: no seam at the prefill boundary and no tile
        err = (got - ref).abs().amax(-1).reshape(-1, cfg.chunk_product).mean(0)
        assert err.max() < 2e-4, f"{mode} has a position-dependent error"

    # every bounded cache stays bounded, whatever the sequence length
    V = cfg.vocab_size
    widths = []
    for n_new in (16, 512):
        gen.generate(torch.randint(0, V, (1, 32)),
                     GenerationConfig(max_new_tokens=n_new, mode="recgen",
                                      greedy=True))
        widths.append((gen.state[1].enc_cache.k[0].shape[2],
                       gen.state[1].dec_cache.k[0].shape[2],
                       gen.state[2].dec_cache.k[0].shape[2]))
    assert widths[0] == widths[1], f"a bounded cache grew with T: {widths}"
    assert widths[0][1] == 32 + max(1 + 4, 64)      # window + one write block


@pytest.mark.parametrize("window", [5, 9, 32, 200])
def test_streaming_decoder_is_exact_at_every_window_width(window):
    """Exactness at every window width: 5 = one group (`R + C`), roll fires on
    almost every write; 9 divides neither group size nor write block; 32 rolls
    periodically; 200 never rolls (plain growing buffer)."""
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = _celeritas_cfg()
    cfg.levels[0] = replace(cfg.levels[0], decoder_stream=window)
    cfg.levels[-1] = replace(cfg.levels[-1], decoder_stream=max(window, 6))
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)

    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (1, 160))
    ref = m(x).logits[:, 16:]
    gen = ScalaGenerator(m, device="cpu", dtype=torch.float32)
    for mode in ("hiergen", "recgen"):
        torch.testing.assert_close(gen.forced_logits(x, mode=mode, prefix_meta=1),
                                   ref, atol=2e-4, rtol=2e-4)

    # the rolling cache is `window + one write block`, at every width and every
    # sequence length -- otherwise a wide window would quietly be O(T)
    widths = []
    for n_new in (16, 256):
        gen.generate(torch.randint(0, cfg.vocab_size, (1, 32)),
                     GenerationConfig(max_new_tokens=n_new, mode="recgen",
                                      greedy=True))
        widths.append(gen.state[1].dec_cache.k[0].shape[2])
    assert widths[0] == widths[1] == window + max(1 + 4, 64)


def test_speculative_decoding_is_the_same_function_as_sequential():
    """Speculative decoding with exact-match acceptance must be token-for-token
    identical to sequential stepping, and rejected cache writes must be undone
    or later chunks silently diverge."""
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = replace(_celeritas_cfg(), mtp_depth=3, mtp_loss_weight=0.3)
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)
    assert len(m.mtp) >= cfg.levels[0].chunk_size - 1

    torch.manual_seed(5)
    prompt = torch.randint(0, cfg.vocab_size, (2, 32))
    gc = GenerationConfig(max_new_tokens=64, mode="recgen", greedy=True)

    ref = ScalaGenerator(m, device="cpu", dtype=torch.float32,
                          speculative=False).generate(prompt, gc)
    gen = ScalaGenerator(m, device="cpu", dtype=torch.float32, speculative=True)
    got = gen.generate(prompt, gc)
    assert torch.equal(ref, got), "speculative decoding changed the output"
    assert gen.stats["spec_chunks"] > 0, "speculation never actually ran"

    # a fully rejected chunk still costs <= C_1 level-1 decoder calls: the
    # rejected slot is corrected from the verify pass, not re-stepped
    C = cfg.levels[0].chunk_size
    assert gen.stats["dec_calls_per_chunk"] <= C + 0.5


def test_speculation_refuses_what_it_cannot_reproduce():
    """Speculation is refused wherever exact-match acceptance is not the
    identity: sampling, repetition penalty, non-level-1 stacks, and
    `forced_logits`."""
    from dataclasses import replace

    torch.manual_seed(0)
    cfg = replace(_celeritas_cfg(), mtp_depth=3)
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)
    gen = ScalaGenerator(m, device="cpu", dtype=torch.float32, speculative=True)
    lvl = m.levels[0]

    ctx_of = lambda **kw: _ctx_stub(GenerationConfig(**kw))
    assert gen._can_speculate(ctx_of(greedy=True), 1, lvl)
    assert not gen._can_speculate(ctx_of(greedy=False), 1, lvl)
    assert not gen._can_speculate(
        ctx_of(greedy=True, repetition_penalty=1.1), 1, lvl)
    assert not gen._can_speculate(ctx_of(greedy=True), 2, m.levels[-1])

    # forced_logits must stay unspeculated
    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (1, 160))
    ref = m(x).logits[:, 16:]
    torch.testing.assert_close(gen.forced_logits(x, mode="recgen", prefix_meta=1),
                               ref, atol=2e-4, rtol=2e-4)


def _ctx_stub(cfg: GenerationConfig):
    """A `_GenContext` with nothing forced and nothing captured."""
    from scala.infer.generate import _GenContext

    return _GenContext(cfg, 1, torch.device("cpu"),
                       torch.zeros(1, 0, dtype=torch.long), None)


def test_celeritas_has_no_length_dependent_position(celeritas_model):
    """No component evaluates a position outside its trained range: every RoPE
    stack is span-bounded (window/block), and the one stack that grows with
    ``T`` -- the top encoder -- is NoPE."""
    from scala.model.layers import TransformerStack

    unbounded = []
    for name, mod in celeritas_model.named_modules():
        if not isinstance(mod, TransformerStack):
            continue
        a = mod.cfg.attention
        if mod.rope is None:                      # NoPE: no position at all
            assert a.pos == "nope" and mod.rot_dim == 0
            continue
        if a.window or a.block:                   # bounded relative offsets
            continue
        unbounded.append(name)

    # per-group decoders are bounded by construction (R_l + C_l positions)
    for i, lvl in enumerate(celeritas_model.levels, start=1):
        if lvl.stream:
            assert f"levels.{i-1}.decoder" not in unbounded
    assert unbounded == [], (
        f"these stacks grow their position range with T: {unbounded}")


def test_nope_mla_refuses_a_decoupled_rope_key():
    """With `pos="nope"` the decoupled RoPE key channels carry nothing, so the
    config refuses them; dropping them makes the latent cache pure
    `kv_lora_rank`."""
    from scala.model.config import AttentionConfig

    with pytest.raises(ValueError, match="mla_qk_rope_head_dim=0"):
        AttentionConfig(kind="mla", pos="nope", mla_qk_rope_head_dim=32)

    a = AttentionConfig(kind="mla", pos="nope", mla_qk_rope_head_dim=0,
                        mla_kv_lora_rank=128, qk_norm=False)
    assert a.resolve_head_dim(384) == a.mla_qk_nope_head_dim


def test_learned_sink_survives_weight_absorption():
    """The absorbed path softmaxes explicitly, so `attn_sink` is one extra
    column; absorbed and plain paths must agree."""
    from dataclasses import replace

    from scala.model.config import StackConfig
    from scala.model.layers import TransformerStack

    torch.manual_seed(7)
    cfg = ScalaConfig.load(TINY)
    att = replace(cfg.levels[-1].encoder.attention, pos="nope",
                  mla_qk_rope_head_dim=0, qk_norm=False, attn_sink=True,
                  output_gate=True)
    stack = TransformerStack(replace(cfg.levels[-1].encoder, attention=att), 64)
    with torch.no_grad():
        for p in stack.parameters():
            p.copy_(torch.randn(p.shape) * 0.05)
    for m in stack.modules():
        if hasattr(m, "invalidate_absorbed"):
            m.invalidate_absorbed()
    assert stack.supports_latent_cache
    assert stack.layers[0].attn.sink.abs().max() > 0, "sink must not be zero"

    a = torch.randn(2, 12, stack.cfg.d_model)
    ref = stack(a)                                       # plain, no cache
    cache = stack.alloc_cache(2, 16, torch.device("cpu"), torch.float32)
    inc = torch.cat([stack(a[:, i : i + 1], cache=cache, pos_offset=i)
                     for i in range(a.shape[1])], dim=1)
    torch.testing.assert_close(inc, ref, atol=2e-4, rtol=2e-4)


def test_decoder_stream_rejects_configurations_it_cannot_honour():
    """Two ways to ask for a decoder stream that is not O(1) in T, or not a
    stream at all.  Both are config errors rather than silent behaviour."""
    from dataclasses import replace

    from scala.model.config import LevelConfig

    with pytest.raises(ValueError, match="subsumes decoder_lookback"):
        LevelConfig(decoder_stream=32, decoder_lookback=1)
    with pytest.raises(ValueError, match="full_attn_every"):
        LevelConfig(decoder_stream=32,
                    decoder=replace(LevelConfig().decoder, full_attn_every=4))
    with pytest.raises(ValueError, match="narrower than one group"):
        LevelConfig(decoder_stream=4, converter_width=4, chunk_size=4)


def test_celeritas_survives_a_config_round_trip(tmp_path):
    """`pos`, `decoder_stream` and a zero-width RoPE key must survive
    save/load."""
    cfg = _celeritas_cfg()
    p = tmp_path / "celeritas.json"
    cfg.save(p)
    back = ScalaConfig.load(p)
    assert back.levels[-1].encoder.attention.pos == "nope"
    assert back.levels[-1].encoder.attention.mla_qk_rope_head_dim == 0
    assert back.levels[0].decoder_stream == 32
    assert back.levels[0].encoder_window == 6
    m = ScalaForCausalLM(back).eval()
    assert m.levels[-1].encoder.rope is None
    assert m.levels[0].stream == 32


def test_three_levels_need_no_new_code():
    """A third level must be a pure config change: the middle level becomes
    non-top (needs a window and its own rolling cache) and `_prefill_decoders`
    walks three cascades.  Each level multiplies the non-global reach by `C`."""
    from dataclasses import replace

    from scala.model.celeritas import celeritas_config

    cfg = celeritas_config(chunks=(4, 4, 4), enc_layers=(2, 2, 3),
                           dec_layers=(2, 2, 2), converter_widths=(1, 2, 2),
                           d_token=128, vocab_size=512, max_seq_len=512,
                           n_heads=4, n_kv_heads=2, head_dim=32,
                           encoder_window=4, decoder_stream=16)
    assert cfg.chunk_product == 64
    torch.manual_seed(0)
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)

    # only the top level stays global; every other encoder is windowed
    assert [lv.encoder_window for lv in m.levels] == [4, 4, None]
    assert m.levels[-1].encoder.rope is None, "top level must still be NoPE"

    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (1, 4 * cfg.chunk_product))
    ref = m(x).logits[:, cfg.chunk_product :]
    gen = ScalaGenerator(m, device="cpu", dtype=torch.float32)
    for mode in ("hiergen", "recgen"):
        torch.testing.assert_close(
            gen.forced_logits(x, mode=mode, prefix_meta=1), ref,
            atol=2e-4, rtol=2e-4)

    # depth multiplies the local reach: each level's decoder windows units that
    # are C times coarser than the level below it
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    from copy_diag import bounded_reach_tokens

    two = replace(cfg, levels=cfg.levels[:2], max_seq_len=512)
    assert bounded_reach_tokens(cfg) > 3 * bounded_reach_tokens(two)


def test_celeritas_preset_matches_the_shipped_probe_yaml():
    """`celeritas_config()` must not drift from `configs/celeritas_probe.yaml`."""
    from scala.model.celeritas import celeritas_config

    yaml_cfg = ScalaConfig.load(
        Path(__file__).resolve().parents[1] / "configs" / "celeritas_probe.yaml")
    preset = celeritas_config(
        d_token=yaml_cfg.d_token, vocab_size=yaml_cfg.vocab_size,
        max_seq_len=yaml_cfg.max_seq_len,
        ffn_inter=yaml_cfg.levels[0].encoder.ffn_inter_size)

    for i, (a, b) in enumerate(zip(preset.levels, yaml_cfg.levels), start=1):
        for field in ("chunk_size", "converter_width", "encoder_window",
                      "decoder_stream", "decoder_lookback"):
            assert getattr(a, field) == getattr(b, field), \
                f"level {i}: preset {field}={getattr(a, field)} but the YAML " \
                f"says {getattr(b, field)}"
        for role in ("encoder", "decoder"):
            pa = getattr(a, role).attention
            pb = getattr(b, role).attention
            assert (pa.kind, pa.pos) == (pb.kind, pb.pos), f"level {i}.{role}"
            if pa.kind == "mla":
                assert pa.mla_qk_rope_head_dim == pb.mla_qk_rope_head_dim == 0
                assert pa.attn_sink and pb.attn_sink
    assert preset.rec_loss_alpha == yaml_cfg.rec_loss_alpha == 0.0
    assert preset.chunk_cond_prob == yaml_cfg.chunk_cond_prob == 0.0


def test_recgen_is_exact_when_the_window_is_the_trained_receptive_field(
        block_local_model):
    """When the model was trained with ``encoder_block_local``, RecGen's bound
    equals the trained receptive field, so RecGen reproduces the training
    forward with O(1) lower-cache memory."""
    cfg = block_local_model.cfg
    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (2, 96))

    ref = block_local_model(x).logits[:, 16:]
    gen = ScalaGenerator(block_local_model, device="cpu", dtype=torch.float32)
    got = gen.forced_logits(x, mode="recgen", prefix_meta=1)
    torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)

    # the paper's rule on the same weights must not match
    paper = gen.forced_logits(x, mode="recgen_paper", prefix_meta=1)
    assert (paper.argmax(-1) == ref.argmax(-1)).float().mean() < 0.9

    # and the level-1 cache is one block regardless of how much is generated
    prompt = torch.randint(0, cfg.vocab_size, (1, 32))
    widths = []
    for n_new in (16, 128):
        gen.generate(prompt, GenerationConfig(max_new_tokens=n_new,
                                              mode="recgen", greedy=True))
        widths.append(gen.state[1].enc_cache.k[0].shape[2])
    assert widths == [lvl_block(block_local_model)] * 2


def test_recgen_window_bounds_a_globally_trained_encoder(model):
    """On a model whose level-1 encoder was *not* trained block-local, the
    window is an approximation rather than an identity -- but it is still O(1),
    and widening it has to move the protocol towards HierGen, not away."""
    cfg = model.cfg
    torch.manual_seed(7)
    x = torch.randint(0, cfg.vocab_size, (1, 96))

    ref = ScalaGenerator(model, device="cpu", dtype=torch.float32) \
        .forced_logits(x, mode="hiergen", prefix_meta=1)

    def err(groups):
        g = ScalaGenerator(model, device="cpu", dtype=torch.float32,
                            enc_window_groups=groups)
        got = g.forced_logits(x, mode="recgen", prefix_meta=1)
        units = g.state[1].enc_cache.k[0].shape[2]
        return (got - ref).abs().max().item(), units

    narrow_err, narrow_units = err(1)
    wide_err, wide_units = err(16)          # >= the whole sequence here
    assert wide_units > narrow_units
    assert wide_err < narrow_err
    # widened past the sequence there is nothing left to truncate: it *is*
    # HierGen, which is the training forward
    assert wide_err < 2e-4


def test_hiergen_matches_the_training_forward_with_absorption(absorbed_model):
    """The equivalence that makes HierGen worth calling exact, on the cache
    layout every real config selects."""
    cfg = absorbed_model.cfg
    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (2, 64))

    ref = absorbed_model(x).logits[:, 16:]
    gen = ScalaGenerator(absorbed_model, device="cpu", dtype=torch.float32)
    got = gen.forced_logits(x, mode="hiergen", prefix_meta=1)

    torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)


# --------------------------------------------------------------------------- #
# shapes and the objective
# --------------------------------------------------------------------------- #
def test_forward_shapes(model):
    cfg = model.cfg
    x = torch.randint(0, cfg.vocab_size, (3, 64))
    out = model(x)
    assert out.logits.shape == (3, 64, cfg.vocab_size)
    assert len(out.enc_states) == cfg.n_levels + 1
    assert out.enc_states[0].shape == (3, 64, cfg.d_token)
    assert out.enc_states[1].shape == (3, 16, cfg.width(1))
    assert out.enc_states[2].shape == (3, 4, cfg.width(2))
    assert len(out.dec_states) == cfg.n_levels
    for l in range(cfg.n_levels):
        assert out.dec_states[l].shape == out.enc_states[l].shape


def test_initial_loss_is_near_uniform(model):
    cfg = model.cfg
    x = torch.randint(0, cfg.vocab_size, (4, 64))
    out = model(x)
    # an untrained model on random tokens must sit near ln(V)
    assert abs(out.loss_token.item() - math.log(cfg.vocab_size)) < 0.6


def test_any_sequence_length_is_accepted_exactly(model):
    """Right-padding is exact: pad units come after every real one and no
    position < T attends forward, so a ragged forward returns the aligned
    forward's logits."""
    cfg = model.cfg
    cp = cfg.chunk_product
    torch.manual_seed(4)
    x = torch.randint(0, cfg.vocab_size, (2, 8 * cp))
    full = model(x).logits

    for T in (8 * cp - 1, 7 * cp + 1, 6 * cp + cp // 2, 8 * cp):
        out = model(x[:, :T])
        assert out.logits.shape[1] == T, "ragged forward changed the output length"
        torch.testing.assert_close(out.logits, full[:, :T], atol=2e-5, rtol=2e-5)
        assert torch.isfinite(out.loss_token)

    # the dataloader keeps its own alignment check (packing efficiency); see
    # `test_dataset_rejects_misaligned_seq_len`




def test_analytic_param_count_matches(model):
    real = sum(p.numel() for p in model.parameters())
    analytic = count_model(model.cfg).total
    # the analytic count includes the router bias *buffers*, which are not
    # parameters; allow that small, exactly-known slack
    n_bias = sum(m.expert_bias.numel() for m in model.modules()
                 if isinstance(m, Router))
    assert analytic - n_bias == pytest.approx(real, rel=1e-4)


# --------------------------------------------------------------------------- #
# causality
# --------------------------------------------------------------------------- #
def test_logits_do_not_depend_on_future_tokens(model):
    """logits[:, i] predicts token i, so it must be a function of t_<i only."""
    cfg = model.cfg
    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (1, 64))
    base = model(x).logits

    for pos in (0, 5, 16, 31, 47):
        y = x.clone()
        y[0, pos] = (y[0, pos] + 7) % cfg.vocab_size
        alt = model(y).logits
        # positions <= pos must be untouched ...
        assert torch.allclose(base[:, : pos + 1], alt[:, : pos + 1], atol=1e-4), \
            f"changing token {pos} leaked backwards"
        # ... and at least one later position must react
        assert not torch.allclose(base[:, pos + 1 :], alt[:, pos + 1 :], atol=1e-4)


@pytest.fixture(scope="module")
def skip_model() -> ScalaForCausalLM:
    """Tiny model with the direct top-level path into every local decoder."""
    torch.manual_seed(0)
    cfg = ScalaConfig.load(TINY)
    cfg.global_skip = True
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)
    return m


def test_global_skip_is_causal_and_still_exact(skip_model):
    """The skip hands every local decoder ``X^(L)_{g-1}`` directly.  It must
    not widen any receptive field, and generation must still reproduce the
    training forward."""
    cfg = skip_model.cfg
    assert skip_model.levels[0].d_skip == cfg.levels[-1].encoder.d_model
    assert skip_model.levels[-1].d_skip == 0, "the top level conditions on it already"

    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (1, 96))
    base = skip_model(x).logits
    for pos in (0, 5, 16, 31, 47, 80):
        y = x.clone()
        y[0, pos] = (y[0, pos] + 7) % cfg.vocab_size
        alt = skip_model(y).logits
        assert torch.allclose(base[:, : pos + 1], alt[:, : pos + 1], atol=1e-4), \
            f"the skip leaked token {pos} backwards"
        assert not torch.allclose(base[:, pos + 1 :], alt[:, pos + 1 :], atol=1e-4)

    ref = skip_model(x).logits[:, 16:]
    gen = ScalaGenerator(skip_model, device="cpu", dtype=torch.float32)
    for mode in ("hiergen", "recgen"):
        got = gen.forced_logits(x, mode=mode, prefix_meta=1)
        if mode == "hiergen":
            torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)
        else:
            assert torch.isfinite(got).all()


def test_diagnostic_token_loader_honours_the_shard_dtype(tmp_path):
    """The loader takes the token width from the manifest and range-checks:
    a `uint32` corpus read as `uint16` yields plausible-looking interleaved
    halves, not an error."""
    import json

    import numpy as np

    from scripts.protocol_diag import load_tokens  # noqa: PLC0415

    ids = np.arange(1024, 1024 + 512, dtype=np.uint32) * 97 % 99_584
    (tmp_path / "src").mkdir()
    ids.tofile(tmp_path / "src" / "s-00000.bin")
    (tmp_path / "manifest.json").write_text(json.dumps({
        "dtype": "uint32",
        "sources": [{"name": "src", "weight": 1.0,
                     "shards": [{"path": "src/s-00000.bin",
                                 "n_tokens": int(ids.size)}]}],
    }), encoding="utf-8")

    class A:
        data_root, batch, seq_len, offset = str(tmp_path), 2, 16, 0
        data_source = None

    cfg = ScalaConfig.load(TINY)
    cfg.vocab_size = 99_584
    got = load_tokens(cfg, A(), torch.device("cpu"))
    assert got.shape == (2, 16)
    assert got.flatten().tolist() == ids[:32].tolist()

    # `data_source` selects the scored corpus; an unknown name must exit
    a = A(); a.data_source = "src"
    assert load_tokens(cfg, a, torch.device("cpu")).flatten().tolist() \
        == ids[:32].tolist()
    a = A(); a.data_source = "nope"
    with pytest.raises(SystemExit):
        load_tokens(cfg, a, torch.device("cpu"))

    # reading past the shard end must be loud, not silently short
    a = A(); a.offset = 500
    with pytest.raises(SystemExit):
        load_tokens(cfg, a, torch.device("cpu"))


def test_encoder_states_are_causal(model):
    """X^(l)_g may only depend on units up to g."""
    cfg = model.cfg
    torch.manual_seed(2)
    x = torch.randint(0, cfg.vocab_size, (1, 64))
    base = model.encode_all(x)
    y = x.clone()
    y[0, 40] = (y[0, 40] + 3) % cfg.vocab_size          # inside level-1 unit 10
    alt = model.encode_all(y)

    assert torch.allclose(base[1][:, :10], alt[1][:, :10], atol=1e-4)
    assert not torch.allclose(base[1][:, 10:], alt[1][:, 10:], atol=1e-4)
    # level-2 unit index 40 // 16 == 2
    assert torch.allclose(base[2][:, :2], alt[2][:, :2], atol=1e-4)


# --------------------------------------------------------------------------- #
# generation
# --------------------------------------------------------------------------- #
def test_hiergen_matches_the_training_forward(model):
    """HierGen conditions the decoders on exactly the encoder states that
    teacher forcing uses, so replaying a sequence must reproduce its logits."""
    cfg = model.cfg
    torch.manual_seed(3)
    x = torch.randint(0, cfg.vocab_size, (2, 64))

    ref = model(x).logits[:, 16:]                    # skip the prefilled meta-ctx
    gen = ScalaGenerator(model, device="cpu", dtype=torch.float32)
    got = gen.forced_logits(x, mode="hiergen", prefix_meta=1)

    assert got.shape == ref.shape
    torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-4)


def test_recgen_runs_on_an_unwindowed_encoder_without_erroring(model):
    """RecGen's cache-windowing (``enc_window_groups`` blocks) is only an
    identity when it matches the level's trained receptive field -- see
    ``test_recgen_is_exact_when_the_window_is_the_trained_receptive_field``.
    This fixture's level 1 trained with a *global* (unwindowed) encoder, so
    the window here is a real approximation, not an identity, regardless of
    training amount: the mismatch is structural (span-bounded window vs. a
    globally-attending trained encoder), not something more L_rec / more
    training would close -- ``rec_loss_alpha=0`` in every shipped preset that
    *does* pass the exactness tests confirms L_rec is not load-bearing for
    this. How the approximation error shrinks as the window widens is covered
    quantitatively by ``test_recgen_window_bounds_a_globally_trained_encoder``;
    this test only guards that RecGen still runs and returns well-formed,
    finite output in this non-ideal regime."""
    cfg = model.cfg
    torch.manual_seed(4)
    x = torch.randint(0, cfg.vocab_size, (1, 48))

    gen = ScalaGenerator(model, device="cpu", dtype=torch.float32)
    hier = gen.forced_logits(x, mode="hiergen", prefix_meta=1)
    rec = gen.forced_logits(x, mode="recgen", prefix_meta=1)

    assert rec.shape == hier.shape
    assert torch.isfinite(rec).all()


def test_recgen_generation_cannot_see_what_it_generated(model):
    """Under the paper's rule every `X_hat_j` in meta-group `g` is a function
    of `X^(L)_{g-1}` only, so no emitted token influences anything beyond its
    own `C_1`-token unit -- for any weights.  HierGen/ChunkGen must propagate."""
    cfg = model.cfg
    gen = ScalaGenerator(model, device="cpu", dtype=torch.float32)
    cp, c1 = cfg.chunk_product, cfg.levels[0].chunk_size

    torch.manual_seed(1)
    x = torch.randint(0, cfg.vocab_size, (1, 64))
    perturbed = 2 * cp                    # first unit of meta-group 2
    x2 = x.clone()
    x2[0, perturbed] = (x[0, perturbed] + 1234) % cfg.vocab_size

    later_unit = 2 * cp + 3 * c1 + 1      # later unit, same meta-group
    next_group = 3 * cp + 1               # the following meta-group
    off = cp                              # forced_logits skips prefix_meta=1

    def moved(mode, i):
        a = gen.forced_logits(x, mode=mode, prefix_meta=1).float()
        b = gen.forced_logits(x2, mode=mode, prefix_meta=1).float()
        return (b - a).abs()[0, i - off].max().item()

    # the paper's rule: the channel carries nothing
    assert moved("recgen_paper", later_unit) == 0.0
    assert moved("recgen_paper", next_group) == 0.0

    # the shipped protocols all propagate
    for mode in ("hiergen", "recgen", "chunkgen"):
        assert moved(mode, later_unit) > 1e-4, f"{mode} must propagate"
        assert moved(mode, next_group) > 1e-4, f"{mode} must propagate"


def test_recgen_grows_only_the_top_cache(model):
    """Only the top level's cache may grow with T: quadruple the sequence and
    the lower caches must not move while the top one does."""
    gen = ScalaGenerator(model, device="cpu", dtype=torch.float32)
    V = model.cfg.vocab_size

    from scala.model.layers import LatentKVCache

    def numel(cache):
        if cache is None:
            return 0
        ts = (cache.c + cache.k_rope) if isinstance(cache, LatentKVCache) \
            else (cache.k + cache.v)
        return sum(t.numel() for t in ts)

    def sizes(mode, n_new):
        prompt = torch.randint(0, V, (1, 32))
        gen.generate(prompt, GenerationConfig(max_new_tokens=n_new, mode=mode,
                                              greedy=True))
        lower = sum(numel(st.enc_cache) for st in gen.state[1:-1])
        return lower, numel(gen.state[-1].enc_cache)

    rec_short, top_short = sizes("recgen", 16)
    rec_long, top_long = sizes("recgen", 256)
    assert rec_short == rec_long > 0        # bounded, and actually present
    assert top_long > top_short             # the one stream that may grow

    hier_short, _ = sizes("hiergen", 16)
    hier_long, _ = sizes("hiergen", 256)
    assert hier_long > hier_short           # HierGen's does grow ...
    assert rec_long < hier_long             # ... and RecGen's is smaller for it

    # `chunkgen` is the cache-free fallback: no lower cache at all.
    gen.generate(torch.randint(0, V, (1, 32)),
                 GenerationConfig(max_new_tokens=16, mode="chunkgen", greedy=True))
    assert gen.state[1].enc_cache is None


def test_mla_kv_cache_handles_asymmetric_head_dims():
    """Under MLA a key is (qk_nope + qk_rope) wide and a value is v_head_dim
    wide.  Those differ in every realistic config, so the cache must not assume
    one head_dim for both."""
    from scala.model.config import AttentionConfig, StackConfig
    from scala.model.layers import TransformerStack

    att = AttentionConfig(
        kind="mla", n_heads=4, mla_q_lora_rank=32, mla_kv_lora_rank=32,
        mla_qk_nope_head_dim=24, mla_qk_rope_head_dim=8, mla_v_head_dim=16,
    )
    assert att.mla_qk_nope_head_dim + att.mla_qk_rope_head_dim != att.mla_v_head_dim
    stack = TransformerStack(StackConfig(d_model=64, n_layers=2, attention=att,
                                         moe={"enabled": False}), 32).eval()
    cache = stack.alloc_cache(2, 16, torch.device("cpu"), torch.float32)
    assert cache.k[0].shape[-1] == 32      # 24 + 8
    assert cache.v[0].shape[-1] == 16      # v_head_dim

    x = torch.randn(2, 8, 64)
    with torch.no_grad():
        prefill = stack(x, cache=cache, pos_offset=0)
        step = stack(torch.randn(2, 1, 64), cache=cache, pos_offset=8)
    assert prefill.shape == (2, 8, 64) and step.shape == (2, 1, 64)


def test_mla_weight_absorption_matches_the_plain_path():
    """Absorbed MLA caches only (kv_lora + qk_rope) per unit yet must produce
    exactly the same attention output as decompressing K and V."""
    from scala.model.config import AttentionConfig, StackConfig
    from scala.model.layers import TransformerStack

    torch.manual_seed(30)
    att = AttentionConfig(
        kind="mla", n_heads=4, mla_q_lora_rank=48, mla_kv_lora_rank=32,
        mla_qk_nope_head_dim=24, mla_qk_rope_head_dim=8, mla_v_head_dim=16,
        qk_norm=False,          # required: QK-Norm needs the decompressed K
    )
    stack = TransformerStack(
        StackConfig(d_model=64, n_layers=2, attention=att,
                    moe={"enabled": False}, ffn_inter_size=128), 64).eval()
    assert stack.supports_latent_cache

    x = torch.randn(2, 12, 64)
    with torch.no_grad():
        plain_cache = stack.alloc_cache(2, 24, torch.device("cpu"),
                                        torch.float32, latent=False)
        plain = stack(x, cache=plain_cache, pos_offset=0)

        lat_cache = stack.alloc_cache(2, 24, torch.device("cpu"), torch.float32)
        absorbed = stack(x, cache=lat_cache, pos_offset=0)
    torch.testing.assert_close(absorbed, plain, atol=2e-5, rtol=2e-5)

    # ... and it must keep matching when decoding incrementally
    step = torch.randn(2, 1, 64)
    with torch.no_grad():
        p = stack(step, cache=plain_cache, pos_offset=12)
        a = stack(step, cache=lat_cache, pos_offset=12)
    torch.testing.assert_close(a, p, atol=2e-5, rtol=2e-5)

    # the latent cache must be much smaller
    lat_bytes = sum(t.numel() for t in lat_cache.c + lat_cache.k_rope)
    plain_bytes = sum(t.numel() for t in plain_cache.k + plain_cache.v)
    assert lat_bytes * 3 < plain_bytes, (lat_bytes, plain_bytes)


def test_qk_norm_blocks_the_latent_cache():
    from scala.model.config import AttentionConfig, StackConfig
    from scala.model.layers import TransformerStack

    att = AttentionConfig(kind="mla", n_heads=4, mla_kv_lora_rank=32,
                          mla_qk_nope_head_dim=24, mla_qk_rope_head_dim=8,
                          mla_v_head_dim=16, qk_norm=True)
    stack = TransformerStack(StackConfig(d_model=64, n_layers=1, attention=att,
                                         moe={"enabled": False}), 32)
    assert not stack.supports_latent_cache
    with pytest.raises(ValueError, match="qk_norm"):
        stack.alloc_cache(1, 8, torch.device("cpu"), torch.float32, latent=True)


def test_generate_lengths_and_determinism(model):
    prompt = torch.randint(0, model.cfg.vocab_size, (2, 19))   # deliberately unaligned
    cfg = GenerationConfig(max_new_tokens=24, greedy=True, mode="hiergen")
    gen = ScalaGenerator(model, device="cpu", dtype=torch.float32)
    a = gen.generate(prompt, cfg)
    b = gen.generate(prompt, cfg)
    assert a.shape == (2, 19 + 24)
    torch.testing.assert_close(a, b)
    torch.testing.assert_close(a[:, :19], prompt)


def test_unaligned_prompt_is_teacher_forced_not_padded(model):
    """The prompt remainder must flow through the decoders unchanged."""
    gen = ScalaGenerator(model, device="cpu", dtype=torch.float32)
    prompt = torch.randint(0, model.cfg.vocab_size, (1, 21))
    out = gen.generate(prompt, GenerationConfig(max_new_tokens=8, greedy=True))
    torch.testing.assert_close(out[:, :21], prompt)


# --------------------------------------------------------------------------- #
# MoE
# --------------------------------------------------------------------------- #
def test_expert_bias_update_balances_load():
    from scala.model.config import MoEConfig
    from scala.model.moe import MoELayer

    torch.manual_seed(5)
    cfg = MoEConfig(n_routed_experts=8, top_k=2, expert_inter_size=16,
                    n_shared_experts=1, shared_inter_size=16,
                    bias_update_rate=0.002)
    layer = MoELayer(cfg, 32).train()
    x = torch.randn(2, 64, 32)

    vios = []
    for _ in range(400):
        layer(x)
        load = layer.router.load_counter
        vios.append(float((load.max() - load.mean()) / load.mean()))
        update_expert_biases(layer, gamma=cfg.bias_update_rate)

    head = sum(vios[:50]) / 50
    tail = sum(vios[-50:]) / 50
    assert tail < 0.6 * head, f"MaxVio did not improve: {head:.4f} -> {tail:.4f}"
    assert layer.router.expert_bias.abs().sum() > 0


def test_bias_update_rate_must_be_small_relative_to_scores():
    """The sign update is a control loop: too large a gamma overshoots and the
    load oscillates instead of settling.  Guards the config default."""
    from scala.model.config import MoEConfig
    from scala.model.moe import MoELayer

    def final_maxvio(gamma: float) -> float:
        torch.manual_seed(5)
        cfg = MoEConfig(n_routed_experts=8, top_k=2, expert_inter_size=16,
                        n_shared_experts=1, shared_inter_size=16)
        layer = MoELayer(cfg, 32).train()
        x = torch.randn(2, 64, 32)
        vios = []
        for _ in range(400):
            layer(x)
            load = layer.router.load_counter
            vios.append(float((load.max() - load.mean()) / load.mean()))
            update_expert_biases(layer, gamma=gamma)
        return sum(vios[-50:]) / 50

    assert final_maxvio(0.002) < final_maxvio(0.05)


def test_moe_output_matches_a_dense_reference():
    """The sorted grouped-GEMM dispatch must equal a naive per-token loop."""
    from scala.model.config import MoEConfig
    from scala.model.moe import MoELayer

    torch.manual_seed(6)
    cfg = MoEConfig(n_routed_experts=6, top_k=2, expert_inter_size=16,
                    n_shared_experts=0, routed_scaling_factor=1.0)
    layer = MoELayer(cfg, 24).eval()
    x = torch.randn(1, 12, 24)
    fast = layer(x)

    flat = x.reshape(-1, 24)
    idx, gate, _ = layer.router(flat, tokens_per_seq=12)
    ref = torch.zeros_like(flat)
    e = layer.experts
    for t in range(flat.shape[0]):
        for s in range(cfg.top_k):
            k = idx[t, s]
            h = torch.nn.functional.silu(flat[t] @ e.w_gate[k]) * (flat[t] @ e.w_up[k])
            ref[t] += gate[t, s] * (h @ e.w_down[k])
    torch.testing.assert_close(fast.reshape(-1, 24), ref, atol=1e-4, rtol=1e-4)


def test_pid_controller_settles_where_the_sign_rule_oscillates():
    """The sign rule is bang-bang: its steady-state MaxVio is set by gamma, so
    a too-large gamma never settles.  The PID controller scales its correction
    by the imbalance magnitude and should converge regardless."""
    from scala.model.config import MoEConfig
    from scala.model.moe import MoELayer

    def run(controller: str, gamma: float) -> float:
        torch.manual_seed(5)
        cfg = MoEConfig(n_routed_experts=8, top_k=2, expert_inter_size=16,
                        n_shared_experts=1, shared_inter_size=16,
                        bias_controller=controller)
        layer = MoELayer(cfg, 32).train()
        x = torch.randn(2, 64, 32)
        vios = []
        for _ in range(400):
            layer(x)
            load = layer.router.load_counter
            vios.append(float((load.max() - load.mean()) / load.mean()))
            update_expert_biases(layer, gamma=gamma)
        return sum(vios[-50:]) / 50

    # at a deliberately oversized gain the sign rule hunts; PID still settles
    assert run("pid", 0.05) < run("sign", 0.05)
    # and PID is no worse at a well-tuned gain
    assert run("pid", 0.05) < 0.15


def test_padded_dispatch_matches_the_reference_loop():
    """The batched-GEMM path must be numerically identical to the loop; it is
    the path devices without a grouped-GEMM kernel take."""
    from scala.model.config import MoEConfig
    from scala.model.moe import MoELayer

    torch.manual_seed(20)
    cfg = MoEConfig(n_routed_experts=12, top_k=3, expert_inter_size=32,
                    n_shared_experts=1, shared_inter_size=32)
    layer = MoELayer(cfg, 48).eval()
    x = torch.randn(2, 40, 48)

    layer.dispatch = "loop"
    ref = layer(x)
    layer.dispatch = "padded"
    got = layer(x)
    torch.testing.assert_close(got, ref, atol=1e-5, rtol=1e-5)


def test_padded_dispatch_capacity_cap_drops_only_the_overflow():
    """With capacity_factor set, over-capacity tokens are zeroed, not corrupted."""
    from scala.model.config import MoEConfig
    from scala.model.moe import MoELayer

    torch.manual_seed(21)
    cfg = MoEConfig(n_routed_experts=8, top_k=2, expert_inter_size=16,
                    n_shared_experts=0, capacity_factor=1.0)
    layer = MoELayer(cfg, 32).eval()
    x = torch.randn(1, 64, 32)

    layer.dispatch = "loop"
    ref = layer(x)
    layer.dispatch = "padded"
    got = layer(x)
    assert torch.isfinite(got).all()
    # a capacity of exactly the mean load must still route most tokens intact
    close = torch.isclose(got, ref, atol=1e-5, rtol=1e-5).all(-1).float().mean()
    assert close > 0.5, f"only {close:.1%} of tokens survived the capacity cap"


def test_router_group_limited_routing_respects_groups():
    from scala.model.config import MoEConfig

    torch.manual_seed(7)
    cfg = MoEConfig(n_routed_experts=8, n_groups=4, topk_groups=1, top_k=2,
                    expert_inter_size=8)
    r = Router(cfg, 16).eval()
    idx, _, _ = r(torch.randn(32, 16))
    groups = idx // (cfg.n_routed_experts // cfg.n_groups)
    assert (groups == groups[:, :1]).all(), "tokens escaped their selected group"


# --------------------------------------------------------------------------- #
# optimiser
# --------------------------------------------------------------------------- #
def test_newton_schulz_orthogonalises():
    from scala.train.optim import zeropower_via_newtonschulz5

    torch.manual_seed(8)
    g = torch.randn(64, 32)
    o = zeropower_via_newtonschulz5(g, steps=5).float()
    s = torch.linalg.svdvals(o)
    assert s.min() > 0.6 and s.max() < 1.4, f"singular values {s.min()}..{s.max()}"


def test_newton_schulz_is_batched_for_expert_weights():
    from scala.train.optim import zeropower_via_newtonschulz5

    torch.manual_seed(9)
    g = torch.randn(5, 32, 16)
    o = zeropower_via_newtonschulz5(g, steps=5)
    assert o.shape == g.shape
    for e in range(5):
        s = torch.linalg.svdvals(o[e].float())
        assert s.min() > 0.6 and s.max() < 1.4


def test_parameter_classification(model):
    from scala.train.optim import classify_parameter

    kinds = {n: classify_parameter(n, p) for n, p in model.named_parameters()}
    assert kinds["embed.weight"] == "adamw"
    for n, k in kinds.items():
        if "norm" in n or n.endswith("start_latent"):
            assert k == "adamw", n
        if "router.weight" in n:
            assert k == "adamw", n
        if ".experts.w_gate" in n or ".attn.wq.weight" in n:
            assert k == "muon", n


def test_wsd_schedule_shape():
    from scala.train.optim import WSDSchedule

    s = WSDSchedule(total_steps=1000, warmup_steps=100, decay_frac=0.2,
                    min_lr_ratio=0.1, peak_lr=1.0)
    assert s.factor(0) == pytest.approx(0.01)
    assert s.factor(99) == pytest.approx(1.0)
    assert s.factor(500) == pytest.approx(1.0)
    assert s.factor(799) == pytest.approx(1.0)
    assert s.factor(999) < 0.2
    assert s.factor(1000) == pytest.approx(0.1)


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
def test_top_p_and_top_k_restrict_support():
    torch.manual_seed(10)
    logits = torch.tensor([[10.0, 9.0, 1.0, 0.0, -5.0]])
    g = torch.Generator().manual_seed(0)
    cfg = GenerationConfig(temperature=1.0, top_k=2, top_p=1.0)
    draws = {int(sample_from_logits(logits.clone(), cfg, None, g)) for _ in range(50)}
    assert draws <= {0, 1}

    cfg = GenerationConfig(temperature=1.0, top_k=0, top_p=0.5)
    draws = {int(sample_from_logits(logits.clone(), cfg, None, g)) for _ in range(50)}
    assert draws == {0}


def test_greedy_is_argmax():
    logits = torch.tensor([[1.0, 5.0, 2.0]])
    cfg = GenerationConfig(greedy=True)
    assert int(sample_from_logits(logits, cfg, None, None)) == 1


# --------------------------------------------------------------------------- #
# training-loop plumbing
# --------------------------------------------------------------------------- #
def test_scheduled_sampling_changes_the_cascade_only_in_training(model):
    """self_cond_prob must be a no-op at eval, and must bite while training."""
    cfg = model.cfg
    torch.manual_seed(11)
    x = torch.randint(0, cfg.vocab_size, (1, 64))

    model.eval()
    model.self_cond_prob = 0.0
    eval_off = model(x).logits
    model.self_cond_prob = 1.0
    eval_on = model(x).logits
    torch.testing.assert_close(eval_on, eval_off)   # eval must ignore it

    model.train()
    model.self_cond_prob = 0.0
    train_off = model(x).logits
    model.self_cond_prob = 1.0                      # every slot substituted
    train_on = model(x).logits
    assert not torch.allclose(train_on, train_off, atol=1e-5), \
        "self-conditioning had no effect while training"

    model.self_cond_prob = 0.0
    model.eval()


def test_train_config_coerces_yaml11_scientific_notation(tmp_path):
    """PyYAML is YAML 1.1: `4.0e6` parses as a *string*, not a float."""
    from scala.train.trainer import TrainConfig

    p = tmp_path / "t.yaml"
    p.write_text(
        "train:\n  total_tokens: 4.0e6\n  bs_warmup_tokens: 1.5e9\n"
        "  global_batch_tokens: 16384\n  lr: 3.0e-4\n",
        encoding="utf-8",
    )
    cfg = TrainConfig.load(p)
    assert cfg.total_tokens == pytest.approx(4.0e6)
    assert cfg.bs_warmup_tokens == pytest.approx(1.5e9)
    assert isinstance(cfg.global_batch_tokens, int)
    assert cfg.total_tokens // cfg.global_batch_tokens > 0


def test_grad_norm_guard_skips_spikes_and_nans():
    from scala.train.optim import GradNormGuard

    g = GradNormGuard(window=50, threshold=4.0, warmup=20)
    for _ in range(40):
        assert not g.should_skip(1.0)
    assert g.should_skip(100.0)
    assert g.should_skip(float("nan"))
    assert not g.should_skip(1.1)


def test_moe_bias_update_returns_stats_before_clearing():
    """MaxVio must be read off the live counters, not the cleared ones."""
    from scala.model.config import MoEConfig
    from scala.model.moe import MoELayer, expert_load_stats

    torch.manual_seed(14)
    layer = MoELayer(MoEConfig(n_routed_experts=8, top_k=2,
                               expert_inter_size=16), 32).train()
    layer(torch.randn(1, 64, 32))
    stats = update_expert_biases(layer, gamma=1e-3)
    assert "moe/maxvio_mean" in stats and stats["moe/maxvio_mean"] > 0
    assert expert_load_stats(layer) == {}          # counters cleared afterwards


def test_dataset_rejects_misaligned_seq_len():
    from scala.data.dataset import MixtureSpec, PackedTokenDataset

    spec = MixtureSpec(sources=[])
    with pytest.raises(ValueError, match="multiple of"):
        PackedTokenDataset(spec, seq_len=100, chunk_product=16)
