#!/usr/bin/env python
"""Generate Japanese text from a trained SCALA checkpoint. Modes (cheapest KV
last): hiergen (full cache, exact), recgen (top-level cache only, bounded
lower-encoder window; default), chunkgen (no cache below top level),
recgen_paper (paper's Eq. 6 rule, reproduction only).

    python scripts/generate.py --ckpt runs/scala-8b-a1b/final --prompt "日本の首都は" --mode recgen --max-new-tokens 200
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.infer.generate import GenerationConfig, ScalaGenerator  # noqa: E402
from scala.model.config import ScalaConfig  # noqa: E402
from scala.model.loading import load_state_dict_compat  # noqa: E402
from scala.model.hierarchy import ScalaForCausalLM  # noqa: E402


def load(ckpt: str, config: str | None, device: str, dtype: torch.dtype):
    p = Path(ckpt)
    cfg_path = config or (p / "model_config.json")
    if not Path(cfg_path).exists():
        cfg_path = p.parent / "model_config.json"
    cfg = ScalaConfig.load(cfg_path)
    model = ScalaForCausalLM(cfg)

    if (p / "state.pt").exists():
        sd = torch.load(p / "state.pt", map_location="cpu", weights_only=False)
        load_state_dict_compat(model, sd.get("model", sd))
    elif (p / "model.safetensors").exists():
        from safetensors.torch import load_file

        sd = load_file(p / "model.safetensors")
        # lm_head is absent from the export when tied; __init__ already tied it
        extra = ("lm_head.weight",) if cfg.tie_word_embeddings else ()
        load_state_dict_compat(model, sd, extra_optional=extra)
    else:  # distributed checkpoint
        import torch.distributed.checkpoint as dcp

        sd = {"model": model.state_dict()}
        dcp.load(sd, checkpoint_id=str(p))
        load_state_dict_compat(model, sd["model"])
    return model.to(device=device, dtype=dtype).eval(), cfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer", default="llm-jp/llm-jp-3-1.8b")
    ap.add_argument("--prompt", default="日本の首都は")
    ap.add_argument("--prompt-file", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=0)
    ap.add_argument("--min-p", type=float, default=0.0)
    ap.add_argument("--repetition-penalty", type=float, default=1.05)
    ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--mode",
                    choices=["hiergen", "recgen", "chunkgen", "recgen_paper"],
                    default="hiergen")
    ap.add_argument("--compare-modes", action="store_true")
    ap.add_argument("--enc-window-groups", type=int, default=None,
                    help="recgen only: blocks of C_{l+1} units the bounded "
                         "lower encoders may read (default 4)")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    model, _ = load(args.ckpt, args.config, args.device,
                    getattr(torch, args.dtype))

    text = (Path(args.prompt_file).read_text(encoding="utf-8")
            if args.prompt_file else args.prompt)
    ids = torch.tensor([tok(text, add_special_tokens=False)["input_ids"]])

    kw = ({} if args.enc_window_groups is None
          else {"enc_window_groups": args.enc_window_groups})
    gen = ScalaGenerator(model, args.device, getattr(torch, args.dtype), **kw)
    modes = ["hiergen", "recgen"] if args.compare_modes else [args.mode]

    for mode in modes:
        cfg = GenerationConfig(
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            top_p=args.top_p, top_k=args.top_k, min_p=args.min_p,
            repetition_penalty=args.repetition_penalty, greedy=args.greedy,
            eos_token_id=tok.eos_token_id, mode=mode, seed=args.seed,
        )
        out = gen.generate(ids, cfg)
        s = gen.stats
        print(f"\n{'='*70}\n{mode.upper()}  "
              f"{s['decode_tok_per_s']:.1f} tok/s, "
              f"KV {s['kv_cache_bytes']/2**20:.1f} MiB "
              f"({s['kv_cache_bytes_per_token']:.0f} B/token)\n{'='*70}")
        print(tok.decode(out[0].tolist(), skip_special_tokens=True))


if __name__ == "__main__":
    main()
