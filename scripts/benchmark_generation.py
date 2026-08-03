#!/usr/bin/env python
"""Measure HierGen vs RecGen decode throughput and KV-cache footprint.
Per protocol: tokens/s, KV bytes per context token, peak allocated memory, and
TPM (throughput / KV-cache GiB, the paper's figure).  ``--check-agreement``
adds RecGen-vs-HierGen greedy agreement -- the recursive-consistency check.

    python scripts/benchmark_generation.py --config configs/base_tiny.yaml --prompt-len 4096 --new-tokens 512
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.infer.generate import (  # noqa: E402
    _NO_LOWER_ENCODER, GenerationConfig, ScalaGenerator,
)
from scala.model.accounting import (  # noqa: E402
    count_model, kv_cache_bytes_per_token,
)
from scala.model.config import ScalaConfig  # noqa: E402
from scala.model.hierarchy import ScalaForCausalLM  # noqa: E402


def load_model(cfg_path: str, ckpt: str | None, device: str, dtype: torch.dtype,
               depth: int | None = None):
    cfg = ScalaConfig.load(cfg_path)
    if depth is not None:
        from scala.model.scala import scala_config_at_depth

        cfg = scala_config_at_depth(cfg, depth)
        print(f"re-expressed at depth {depth}: L={cfg.n_levels}, "
              f"C_<=L={cfg.chunk_product}")
    model = ScalaForCausalLM(cfg)
    if ckpt:
        p = Path(ckpt)
        # export dir holds model.safetensors; a training checkpoint holds
        # state.pt; `--ckpt` may also name a .pt file directly
        if (p / "model.safetensors").exists():
            from safetensors.torch import load_file

            src = p / "model.safetensors"
            sd = load_file(str(src))
        else:
            src = p / "state.pt" if (p / "state.pt").exists() else p
            blob = torch.load(src, map_location="cpu", weights_only=False)
            sd = blob.get("model", blob)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # strict=False allows tied heads / re-derivable buffers, but it will
        # just as happily load nothing -- report what happened
        print(f"loaded weights from {src} ({len(sd)} tensors, "
              f"{len(missing)} missing, {len(unexpected)} unexpected)")
        if len(missing) > 8:
            print(f"  [warn] many missing keys, e.g. {missing[:5]}")
    return model.to(device=device, dtype=dtype).eval(), cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_tiny.yaml")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--depth", type=int, default=None,
                    help="re-express a tied (SCALA) checkpoint at this many "
                         "MID applications")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--new-tokens", type=int, default=128)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--check-agreement", action="store_true")
    ap.add_argument("--no-speculative", action="store_true",
                    help="disable chunk-level self-speculative decoding, which "
                         "is otherwise on whenever the model has an MTP head "
                         "and the request is greedy")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, cfg = load_model(args.config, args.ckpt, args.device, dtype,
                            depth=args.depth)
    mc = count_model(cfg)

    print("=" * 74)
    print(f"{mc.total/1e9:.2f}B total / {mc.active/1e9:.2f}B active | "
          f"C_<=L={cfg.chunk_product} | prompt={args.prompt_len} "
          f"new={args.new_tokens} batch={args.batch} {args.device}/{args.dtype}")
    print("=" * 74)

    torch.manual_seed(0)
    prompt = torch.randint(0, cfg.vocab_size,
                           (args.batch, args.prompt_len), device=args.device)
    gen = ScalaGenerator(model, args.device, dtype,
                          speculative=not args.no_speculative)
    results = {}

    for mode in ("hiergen", "recgen", "chunkgen", "recgen_paper"):
        gcfg = GenerationConfig(max_new_tokens=args.new_tokens, greedy=True,
                                mode=mode)
        for _ in range(args.warmup):
            gen.generate(prompt, gcfg)
        if args.device == "cuda":
            torch.cuda.reset_peak_memory_stats()

        best = None
        for _ in range(args.repeats):
            gen.spec_stats = dict.fromkeys(gen.spec_stats, 0)
            gen.generate(prompt, gcfg)
            s = dict(gen.stats)
            if best is None or s["decode_tok_per_s"] > best["decode_tok_per_s"]:
                best = s
        peak = (torch.cuda.max_memory_allocated() / 2**30
                if args.device == "cuda" else float("nan"))
        kv_gib = best["kv_cache_bytes"] / 2**30
        best["peak_gib"] = peak
        best["tpm"] = best["decode_tok_per_s"] / max(kv_gib, 1e-9)
        # `recgen` bounds the lower encoders rather than deleting them, so the
        # analytic figure counts those windows; `chunkgen`/`recgen_paper`
        # keep nothing below the top level.
        best["analytic_kv_bytes_per_token"] = kv_cache_bytes_per_token(
            cfg, recgen=(mode != "hiergen"),
            lower_encoder=(mode not in _NO_LOWER_ENCODER),
            window_groups=gen.enc_window_groups,
            context_tokens=args.prompt_len + args.new_tokens,
        )
        results[mode] = best

        print(f"\n{mode.upper()}")
        print(f"  prefill          {best['prefill_s']*1e3:9.2f} ms")
        print(f"  decode           {best['decode_s']*1e3:9.2f} ms "
              f"for {best['new_tokens']} tokens")
        print(f"  throughput       {best['decode_tok_per_s']:9.1f} tok/s")
        print(f"  KV cache         {kv_gib*1024:9.2f} MiB "
              f"({best['kv_cache_bytes_per_token']:.1f} B/token measured, "
              f"{best['analytic_kv_bytes_per_token']:.1f} analytic)")
        print(f"  peak allocated   {peak:9.2f} GiB")
        print(f"  TPM              {best['tpm']:9.1f} tok/s per GiB of KV")
        if "spec_accept_rate" in best:
            # C_1 level-1 decoder calls per chunk is what sequential costs;
            # anything below that is what the drafter bought.
            print(f"  speculation      {100*best['spec_accept_rate']:8.1f}% of "
                  f"drafts accepted, {best['dec_calls_per_chunk']:.2f} L1-decoder "
                  f"calls/chunk (sequential = {cfg.levels[0].chunk_size})")

    h, r = results["hiergen"], results["recgen"]
    print("\n" + "=" * 74)
    print(f"RecGen vs HierGen: {r['decode_tok_per_s']/h['decode_tok_per_s']:.2f}x "
          f"throughput, {h['kv_cache_bytes']/max(r['kv_cache_bytes'],1):.2f}x less "
          f"KV, {r['tpm']/max(h['tpm'],1e-9):.2f}x TPM")

    if args.check_agreement:
        # HierGen is the reference: it matches the training forward.
        tok = torch.randint(0, cfg.vocab_size,
                            (1, args.prompt_len + args.new_tokens),
                            device=args.device)
        lh = gen.forced_logits(tok, mode="hiergen")
        for cheap in ("recgen", "chunkgen"):
            lc = gen.forced_logits(tok, mode=cheap)
            agree = (lh.argmax(-1) == lc.argmax(-1)).float().mean().item()
            kl = torch.nn.functional.kl_div(
                torch.log_softmax(lc.float(), -1),
                torch.log_softmax(lh.float(), -1),
                log_target=True, reduction="batchmean").item()
            print(f"greedy agreement {cheap} vs HierGen: {agree:.1%}   "
                  f"KL: {kl:.4f}")
            results[f"agreement/{cheap}"] = agree
            results[f"kl/{cheap}"] = kl

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(results, indent=2),
                                       encoding="utf-8")
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
