"""SCALA (`tie_mid_levels`) tests: at every instantiated depth, generation
reproduces that depth's own training forward, causality holds, every non-CAP
cache is bounded, and one state_dict serves all depths."""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.infer.generate import (  # noqa: E402
    GenerationConfig, ScalaGenerator,
)
from scala.model.accounting import (  # noqa: E402
    count_model, kv_cache_bytes_per_token, scala_depth_for_context,
    scala_state_bytes,
)
from scala.model.layers import LatentKVCache  # noqa: E402
from scala.model.config import ScalaConfig  # noqa: E402
from scala.model.hierarchy import ScalaForCausalLM  # noqa: E402
from scala.model.scala import scala_config, scala_config_at_depth  # noqa: E402
from test_hierarchy import _shake_zero_init_parameters  # noqa: E402

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def _tiny(depth: int, **over) -> ScalaConfig:
    """Small enough that depth-3 exactness runs in seconds: chunk 2, so
    C_<=L is 2**(depth+2) and four meta-groups are 4 * C_<=L tokens."""
    return scala_config(
        depth=depth, d_token=64, vocab_size=512, chunk=2,
        n_heads=4, n_kv_heads=2, head_dim=16,
        l1_layers=(2, 2), mid_layers=(1, 1), cap_layers=(2, 1),
        encoder_window=4, decoder_stream=16, cap_units=4, **over)


def _build(depth: int, seed: int = 0) -> ScalaForCausalLM:
    torch.manual_seed(seed)
    m = ScalaForCausalLM(_tiny(depth)).eval()
    _shake_zero_init_parameters(m)
    return m


# --------------------------------------------------------------------------- #
# config surface
# --------------------------------------------------------------------------- #
def test_scala_config_validation_rejects_bad_ties():
    cfg = _tiny(2)

    with pytest.raises(ValueError, match=">= 3 levels"):
        replace(cfg, levels=[cfg.levels[0], cfg.levels[-1]], max_seq_len=32)
    het = replace(cfg.levels[1], converter_width=1)
    with pytest.raises(ValueError, match="differs from level 2"):
        replace(cfg, levels=[cfg.levels[0], cfg.levels[1], het, cfg.levels[-1]])
    unwindowed = replace(cfg.levels[1], encoder_window=None)
    with pytest.raises(ValueError, match="windowed"):
        replace(cfg, levels=[cfg.levels[0], unwindowed, cfg.levels[-1]],
                max_seq_len=64)
    unstreamed = replace(cfg.levels[1], decoder_stream=None)
    with pytest.raises(ValueError, match="stream"):
        replace(cfg, levels=[cfg.levels[0], unstreamed, cfg.levels[-1]],
                max_seq_len=64)
    with pytest.raises(ValueError, match="global_skip"):
        replace(cfg, global_skip=True)
    wide_mid = replace(
        cfg.levels[1], encoder=replace(cfg.levels[1].encoder, d_model=128))
    with pytest.raises(ValueError, match="width"):
        replace(cfg, levels=[cfg.levels[0], wide_mid, cfg.levels[-1]],
                max_seq_len=64)
    unwindowed_l1 = replace(cfg.levels[0], encoder_window=None,
                            encoder_block_local=False)
    with pytest.raises(ValueError, match="level 1"):
        replace(cfg, levels=[unwindowed_l1, cfg.levels[1], cfg.levels[-1]],
                max_seq_len=64)


def test_scala_survives_a_config_round_trip():
    cfg = _tiny(3)
    back = ScalaConfig.from_dict(cfg.to_dict())
    assert back.tie_mid_levels
    assert back.n_levels == cfg.n_levels
    assert back.levels == cfg.levels
    m = ScalaForCausalLM(back)
    assert m.levels[1] is m.levels[2] is m.levels[3]


def test_scala_preset_matches_the_shipped_yaml():
    """`scala_config()` must equal `configs/scala_probe_k2.yaml`, and depth=1
    must equal the `celeritas_probe_L3.yaml` geometry."""
    yaml_cfg = ScalaConfig.load(CONFIGS / "scala_probe_k2.yaml")
    preset = scala_config()
    assert yaml_cfg.tie_mid_levels
    assert preset.max_seq_len == yaml_cfg.max_seq_len
    assert len(preset.levels) == len(yaml_cfg.levels) == 4
    assert preset.levels == yaml_cfg.levels
    assert preset.rec_loss_alpha == yaml_cfg.rec_loss_alpha == 0.0

    l3 = ScalaConfig.load(CONFIGS / "celeritas_probe_L3.yaml")
    s1 = scala_config(depth=1, max_seq_len=2048)
    assert s1.levels == l3.levels, \
        "scala_config(depth=1) drifted from the Celeritas L=3 geometry"


# --------------------------------------------------------------------------- #
# the tie itself
# --------------------------------------------------------------------------- #
def test_scala_mid_levels_are_one_module():
    m = _build(3)
    assert m.levels[1] is m.levels[2] is m.levels[3]
    assert m.levels[1] is m.level_mid
    assert m.levels[0] is m.level_token and m.levels[-1] is m.level_cap
    n1 = sum(p.numel() for p in _build(1).parameters())
    n3 = sum(p.numel() for p in m.parameters())
    assert n1 == n3, "parameter count must not depend on depth"


def test_scala_state_dict_keys_are_depth_invariant():
    m1, m3 = _build(1), _build(3)
    sd1, sd3 = m1.state_dict(), m3.state_dict()
    assert set(sd1) == set(sd3)
    assert all(not k.startswith("levels.") for k in sd1), \
        "tied registration must not serialise per-level copies"
    m3.load_state_dict(sd1, strict=True)


def test_scala_depth_change_matches_direct_build():
    """`scala_config_at_depth` must be the identity on everything but depth:
    weights moved through it compute exactly what a directly-built model of
    the same depth computes."""
    m2 = _build(2)
    via = ScalaForCausalLM(scala_config_at_depth(m2.cfg, 3)).eval()
    via.load_state_dict(m2.state_dict(), strict=True)
    direct = ScalaForCausalLM(_tiny(3)).eval()
    direct.load_state_dict(m2.state_dict(), strict=True)
    torch.manual_seed(5)
    x = torch.randint(0, 512, (2, 4 * via.cfg.chunk_product))
    torch.testing.assert_close(via(x).logits, direct(x).logits,
                               atol=0.0, rtol=0.0)


def test_scala_depth_change_matches_direct_build_to_L6():
    """Same identity as `test_scala_depth_change_matches_direct_build`, but
    all the way to depth=4 (L=6) -- the deeper of the two untrained depths
    the README's zero-shot-transfer headline names explicitly ("L=5, L=6").
    No test in this suite previously built a SCALA model past depth=3 (L=5)."""
    m2 = _build(2)
    via = ScalaForCausalLM(scala_config_at_depth(m2.cfg, 4)).eval()
    via.load_state_dict(m2.state_dict(), strict=True)
    direct = ScalaForCausalLM(_tiny(4)).eval()
    direct.load_state_dict(m2.state_dict(), strict=True)
    torch.manual_seed(5)
    x = torch.randint(0, 512, (2, 4 * via.cfg.chunk_product))
    torch.testing.assert_close(via(x).logits, direct(x).logits,
                               atol=0.0, rtol=0.0)


def test_scala_analytic_param_count_matches():
    for k in (1, 2, 3):
        cfg = _tiny(k)
        real = sum(p.numel() for p in ScalaForCausalLM(cfg).parameters())
        assert count_model(cfg).total == real, f"depth {k}"
    assert count_model(_tiny(1)).total == count_model(_tiny(3)).total


# --------------------------------------------------------------------------- #
# function-level properties, per depth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("depth", [1, 2, 3, 4])
def test_scala_generation_is_exact_at_every_depth(depth):
    """At each instantiated depth, `hiergen` and `recgen` equal that depth's
    own training forward."""
    m = _build(depth)
    cp = m.cfg.chunk_product
    torch.manual_seed(3)
    x = torch.randint(0, m.cfg.vocab_size, (2, 4 * cp))
    ref = m(x).logits[:, cp:]
    gen = ScalaGenerator(m, device="cpu", dtype=torch.float32)
    for mode in ("hiergen", "recgen"):
        torch.testing.assert_close(
            gen.forced_logits(x, mode=mode, prefix_meta=1), ref,
            atol=2e-4, rtol=2e-4)


def test_scala_causality_and_no_backward_leak():
    m = _build(2)
    V = m.cfg.vocab_size
    torch.manual_seed(1)
    x = torch.randint(0, V, (1, 4 * m.cfg.chunk_product))
    base = m(x).logits
    worst = 0.0
    for pos in range(x.shape[1]):
        y = x.clone()
        y[0, pos] = (y[0, pos] + 7) % V
        alt = m(y).logits
        worst = max(worst, (base[:, : pos + 1]
                            - alt[:, : pos + 1]).abs().max().item())
        if pos + 1 < x.shape[1]:
            assert not torch.allclose(base[:, pos + 1:], alt[:, pos + 1:],
                                      atol=1e-5), f"token {pos} changed nothing"
    assert worst < 1e-5, f"backward leak through a tied level: {worst:.2e}"


def test_scala_bounded_caches_and_top_growth():
    """Every MID encoder window and every decoder stream is O(1) in T;
    only the CAP's cache is sized by the sequence."""
    m = _build(3)
    V, L = m.cfg.vocab_size, m.cfg.n_levels
    gen = ScalaGenerator(m, device="cpu", dtype=torch.float32)
    widths = {}
    for n_new in (16, 256):
        gen.generate(torch.randint(0, V, (1, 2 * m.cfg.chunk_product)),
                     GenerationConfig(max_new_tokens=n_new, mode="recgen",
                                      greedy=True))
        bounded = []
        for l in range(1, L):
            bounded.append(gen.state[l].enc_cache.k[0].shape[2])
        for l in range(1, L + 1):
            bounded.append(gen.state[l].dec_cache.k[0].shape[2])
        widths[n_new] = (bounded, gen.state[L].enc_cache.k[0].shape[2]
                         if hasattr(gen.state[L].enc_cache, "k")
                         else gen.state[L].enc_cache.c[0].shape[1])
    assert widths[16][0] == widths[256][0], \
        f"a bounded cache grew with T: {widths}"
    assert widths[256][1] > widths[16][1], \
        "the CAP cache is the one thing that may grow, and it did not"


def test_scala_has_no_length_dependent_position():
    """Structural walk at depth 3: the CAP is NoPE, everything else is
    span-bounded, so no component evaluates a position outside its trained
    range at any context length or any depth."""
    m = _build(3)
    assert m.levels[-1].encoder.rope is None, "the CAP must be NoPE"
    for i, lvl in enumerate(m.levels, start=1):
        if i < m.cfg.n_levels:
            assert lvl.encoder_window, f"level {i} encoder is unbounded RoPE"
        assert lvl.stream, f"level {i} decoder is not a bounded stream"


# --------------------------------------------------------------------------- #
# the bounded-top policy
# --------------------------------------------------------------------------- #
def test_scala_depth_policy_state_is_logarithmic():
    probe = scala_config()          # the shipped probe geometry, C=4, depth 2
    for t, want in [(4096, 2), (8192, 2), (16384, 3), (32768, 3),
                    (65536, 4), (131072, 4), (524288, 5)]:
        assert scala_depth_for_context(probe, t, u_max=32) == want, t

    s = [scala_state_bytes(probe, t, u_max=32)
         for t in (8192, 32768, 131072)]
    assert [x["depth"] for x in s] == [2, 3, 4]
    d1 = s[1]["bytes_total"] - s[0]["bytes_total"]
    d2 = s[2]["bytes_total"] - s[1]["bytes_total"]
    assert d1 == pytest.approx(d2), \
        "state must grow by exactly one MID block per 4x of context"
    # and the policy refuses configs whose weights cannot change depth
    with pytest.raises(ValueError, match="untied|tie"):
        scala_depth_for_context(replace(probe, tie_mid_levels=False), 8192)


def test_scala_state_bytes_matches_a_real_generator_for_bounded_caches():
    """`scala_state_bytes`'s bounded-cache byte estimate was previously only
    checked for internal self-consistency (the 4x-context growth-per-step
    assertion above); it was never cross-validated against what
    `ScalaGenerator` actually allocates, so a formula/allocator drift (e.g.
    the decoder-side `max(width + chunk, 64)` floor `_alloc_caches` applies,
    which this formula silently missed until fixed alongside this test) had
    nothing to catch it."""
    m = _build(3)                       # L = 5
    cp = m.cfg.chunk_product
    P, max_new = 2 * cp, 3 * cp
    gen = ScalaGenerator(m, device="cpu", dtype=torch.float32)
    gen.generate(torch.randint(0, m.cfg.vocab_size, (1, P)),
                 GenerationConfig(max_new_tokens=max_new, mode="recgen",
                                  greedy=True))

    def _bytes(c):
        if c is None:
            return 0
        tensors = (c.c + c.k_rope) if isinstance(c, LatentKVCache) else (c.k + c.v)
        return sum(t.numel() * t.element_size() for t in tensors)

    L = m.cfg.n_levels
    real_bounded = (sum(_bytes(gen.state[l].enc_cache) for l in range(1, L))
                    + sum(_bytes(gen.state[l].dec_cache) for l in range(1, L + 1)))

    # u_max huge so the policy does not re-instantiate at a different depth
    # than the one actually running above
    analytic = scala_state_bytes(m.cfg, P + max_new, u_max=10_000,
                                 dtype_bytes=4)  # dtype_bytes=4: fp32, matches `gen`
    assert real_bounded == analytic["bytes_windows"] + analytic["bytes_decoders"], \
        "the analytic bounded-cache formula drifted from what the generator allocates"


def test_kv_cache_bytes_per_token_recgen_matches_hiergen_when_structurally_windowed():
    """On a SCALA-tied config every lower level trains span-bounded
    (`encoder_window`), so `ScalaGenerator` allocates byte-identical
    lower-level caches for `hiergen` and `recgen` -- `_window_units` checks
    the structural window before ever consulting the protocol. The formula
    must agree: it previously skipped every lower level whenever
    `recgen=True` and `window_groups` was not given explicitly, silently
    undercounting every naive caller (`make_model_card.py`,
    `export_checkpoint.py`, `count_params.py`) that just asked for
    ``recgen=True`` without knowing to also pass `window_groups`."""
    cfg = _tiny(2)
    hier = kv_cache_bytes_per_token(cfg, recgen=False)
    rec = kv_cache_bytes_per_token(cfg, recgen=True)
    assert rec == hier, \
        "RecGen and HierGen must report identical bytes when every lower " \
        "level is structurally windowed"

    # `lower_encoder=False` (chunkgen/recgen_paper: no lower cache at all,
    # see PROTOCOLS in scala.infer.generate) must zero every level below the
    # top, independent of `window_groups`
    no_lower = kv_cache_bytes_per_token(cfg, recgen=True, lower_encoder=False)
    assert no_lower < rec
    assert no_lower == kv_cache_bytes_per_token(
        cfg, recgen=True, lower_encoder=False, window_groups=1)
