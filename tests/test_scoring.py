"""Tiled long-context scorer tests: ``TiledScorer.score_span`` equals the
training forward's per-position CE at every depth and tiling, the warm-up
bound is load-bearing, and RoPE's table cap keeps cached rows byte-identical."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.infer.scoring import TiledScorer  # noqa: E402
from scala.model.layers import RotaryEmbedding  # noqa: E402
from scala.model.hierarchy import ScalaForCausalLM  # noqa: E402
from scala.model.scala import scala_config  # noqa: E402
from test_hierarchy import _shake_zero_init_parameters  # noqa: E402


def _tiny(depth: int) -> ScalaForCausalLM:
    torch.manual_seed(0)
    cfg = scala_config(
        depth=depth, d_token=64, vocab_size=512, chunk=2,
        n_heads=4, n_kv_heads=2, head_dim=16,
        l1_layers=(2, 2), mid_layers=(1, 1), cap_layers=(2, 1),
        encoder_window=4, decoder_stream=16, cap_units=4)
    m = ScalaForCausalLM(cfg).eval()
    _shake_zero_init_parameters(m)
    return m


def _reference_ce(m, x, span):
    """Per-position CE of the last ``span`` rows from the full forward."""
    logits = m(x).logits[:, -span:].float()
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), x[:, -span:].reshape(-1),
        reduction="none").view(x.shape[0], span)


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_tiled_scorer_matches_the_training_forward(depth):
    m = _tiny(depth)
    cp = m.cfg.chunk_product
    T = 16 * cp
    torch.manual_seed(3)
    x = torch.randint(0, m.cfg.vocab_size, (2, T))
    for span in (7, 16, 2 * cp):
        ref = _reference_ce(m, x, span)
        scorer = TiledScorer(m, "cpu", torch.float32,
                             tile_tokens=4 * cp, enc_block_units=8)
        mean_ce, per_pos = scorer.score_span(x, span)
        torch.testing.assert_close(per_pos, ref, atol=2e-4, rtol=2e-4)
        assert mean_ce == pytest.approx(float(ref.mean()), abs=2e-4)


def test_tiled_scorer_is_tile_invariant():
    m = _tiny(2)
    cp = m.cfg.chunk_product
    T = 32 * cp
    torch.manual_seed(5)
    x = torch.randint(0, m.cfg.vocab_size, (1, T))
    outs = []
    for tile, blk in ((cp, 2), (4 * cp, 8), (T, 64)):
        s = TiledScorer(m, "cpu", torch.float32,
                        tile_tokens=tile, enc_block_units=blk)
        outs.append(s.score_span(x, 16)[1])
    for o in outs[1:]:
        torch.testing.assert_close(o, outs[0], atol=1e-5, rtol=1e-5)


def test_warmup_bound_is_load_bearing():
    """+groups must not change the answer; -1 group must (the receptive-field
    bound is used, not vestigial)."""
    m = _tiny(2)
    cp = m.cfg.chunk_product
    T = 64 * cp
    torch.manual_seed(7)
    x = torch.randint(0, m.cfg.vocab_size, (1, T))
    scorer = TiledScorer(m, "cpu", torch.float32, tile_tokens=8 * cp,
                         enc_block_units=8)
    base = scorer.score_span(x, 16)[1]

    import math
    warm = {}
    for l, lvl in enumerate(m.levels, start=1):
        n = lvl.decoder.cfg.n_layers
        warm[l] = math.ceil(n * (lvl.stream - 1) / (lvl.width + lvl.chunk))

    more = scorer.score_span(x, 16,
                             warm_override={k: v + 4 for k, v in warm.items()})[1]
    torch.testing.assert_close(more, base, atol=1e-5, rtol=1e-5)

    # level 1 is the tightest bound at these shapes; one group short must move
    # at least one scored position beyond tolerance
    less = scorer.score_span(x, 16,
                             warm_override={**warm, 1: warm[1] - 1})[1]
    assert (less - base).abs().max() > 2e-4, \
        "removing warm-up changed nothing -- the bound is vestigial"


def test_whole_stream_fallback_matches_the_training_forward():
    """When a level's warm-up would reach past the stream start, that level is
    decoded whole, which must match the training forward exactly (including
    the CAP shift path)."""
    for depth, mult in ((1, 2), (2, 4), (3, 8)):
        m = _tiny(depth)
        cp = m.cfg.chunk_product
        T = mult * cp
        torch.manual_seed(11 + depth)
        x = torch.randint(0, m.cfg.vocab_size, (2, T))
        scorer = TiledScorer(m, "cpu", torch.float32,
                             tile_tokens=2 * cp, enc_block_units=4)
        for span in (T, T // 2, 5):
            ref = _reference_ce(m, x, span)
            _, per_pos = scorer.score_span(x, span)
            torch.testing.assert_close(per_pos, ref, atol=2e-4, rtol=2e-4)


def test_scorer_refuses_misaligned_lengths():
    m = _tiny(1)
    cp = m.cfg.chunk_product
    scorer = TiledScorer(m, "cpu", torch.float32)
    with pytest.raises(AssertionError, match="multiple"):
        scorer.score_span(torch.randint(0, 512, (1, cp + 1)), 4)


def test_rope_table_cap_keeps_cached_rows_and_extends_exactly():
    r = RotaryEmbedding(head_dim=32, theta=10_000.0, max_seq_len=64)
    cos_a, sin_a = r(48, 0)
    # growth below the cap: geometric, rows unchanged
    cos_b, sin_b = r(48, 100)          # forces growth to >= 148
    cos_a2, _ = r(48, 0)
    torch.testing.assert_close(cos_a2, cos_a, atol=0.0, rtol=0.0)

    # beyond-table rows use fp64 phases; the fp32 table quantizes phase by up
    # to ~pos*eps (~8e-3 rad at pos~131K), which sets this tolerance
    off = RotaryEmbedding.MAX_TABLE_LEN - 8
    cos_c, sin_c = r._rows_beyond_table(8, off - 8)
    r._build_cache(off)
    torch.testing.assert_close(cos_c, r.cos_cached[off - 8 : off],
                               atol=1e-2, rtol=1e-2)
    # and at small positions the two are tight
    cos_s, _ = r._rows_beyond_table(8, 64)
    torch.testing.assert_close(cos_s, r.cos_cached[64:72],
                               atol=1e-3, rtol=1e-3)

    # a beyond-cap request must not allocate a giant table
    cos_d, _ = r(16, RotaryEmbedding.MAX_TABLE_LEN + 10_000)
    assert r._cache_len <= RotaryEmbedding.MAX_TABLE_LEN
    assert cos_d.shape[0] == 16


def test_rope_forward_is_granularity_invariant_across_the_table_cap():
    """The fp32-cached-vs-fp64-per-call routing decision must be a pure
    function of each row's own absolute position, never of how the caller
    chunked its call -- otherwise `TiledScorer`'s small per-block calls and a
    one-shot ``model(tokens)`` reference call can silently disagree for the
    same absolute positions once a level's unit count crosses
    ``MAX_TABLE_LEN`` (reproduced against the project's own published
    1M-token benchmark config, where level 1's unit count is ~2x the real
    cap). Shrinking the cap here brings the boundary into a tiny model's
    reach without needing anywhere near 131072 real positions."""
    r = RotaryEmbedding(head_dim=32, theta=10_000.0, max_seq_len=16)
    r.MAX_TABLE_LEN = 100  # instance-only override

    one_shot_cos, one_shot_sin = r(160, 0)

    block_cos, block_sin = [], []
    for off in range(0, 160, 8):
        c, s = r(8, off)
        block_cos.append(c)
        block_sin.append(s)
    block_cos, block_sin = torch.cat(block_cos), torch.cat(block_sin)

    torch.testing.assert_close(one_shot_cos, block_cos, atol=0.0, rtol=0.0)
    torch.testing.assert_close(one_shot_sin, block_sin, atol=0.0, rtol=0.0)

    # a single call straddling the boundary must equal the two halves
    # computed as separate calls on either side of it
    straddle_cos, straddle_sin = r(20, 90)
    lo_cos, lo_sin = r(10, 90)
    hi_cos, hi_sin = r(10, 100)
    torch.testing.assert_close(straddle_cos, torch.cat([lo_cos, hi_cos]),
                               atol=0.0, rtol=0.0)
    torch.testing.assert_close(straddle_sin, torch.cat([lo_sin, hi_sin]),
                               atol=0.0, rtol=0.0)


def test_rope_growth_is_geometric_not_per_call():
    r = RotaryEmbedding(head_dim=32, theta=10_000.0, max_seq_len=64)
    builds = 0
    orig = r._build_cache

    def counting(seq_len):
        nonlocal builds
        if seq_len > r._cache_len:
            builds += 1
        return orig(seq_len)

    r._build_cache = counting
    for off in range(0, 4096, 16):
        r(16, off)
    assert builds <= 8, f"{builds} rebuilds for a linear scan -- not geometric"
