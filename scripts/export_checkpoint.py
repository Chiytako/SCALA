#!/usr/bin/env python
"""Consolidate a training checkpoint (DCP or state.pt) into a single-file
export: model.safetensors, model_config.json, tokenizer files, and a model
card with the exact parameter accounting. Optimiser state is dropped.

    python scripts/export_checkpoint.py runs/scala-8b-a1b/final --out export/scala-8b-a1b
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.model.accounting import count_model, format_report  # noqa: E402
from scala.model.config import ScalaConfig  # noqa: E402
from scala.model.loading import load_state_dict_compat  # noqa: E402
from scala.model.hierarchy import ScalaForCausalLM  # noqa: E402

CARD = """---
license: apache-2.0
language: [ja, en]
library_name: scala
tags: [photon, hierarchical, mixture-of-experts, japanese, pretrained]
---

# Unofficial PHOTON reproduction -- Japanese {total:.1f}B-A{active:.1f}B

> **Unofficial.** This is an independent, hobby reimplementation of the PHOTON
> architecture described in [arXiv:2512.20687](https://arxiv.org/abs/2512.20687),
> trained from scratch by a private individual. It is **not** released by,
> affiliated with, or endorsed by the paper's authors or any organisation, and no
> weights, data or code from any official PHOTON release were used.

> **Read this before using it.** It was trained on roughly 0.36B tokens against
> 8.06B parameters -- about 0.3 tokens per *active* parameter, where the
> Chinchilla-optimal figure is ~20. It is two orders of magnitude undertrained.
> This checkpoint demonstrates that the architecture and the training recipe work
> at this scale; it is not a competitive model and should not be compared to one.

A Japanese language model on the **PHOTON** hierarchical autoregressive
architecture, with the transformer stacks replaced by fine-grained
Mixture-of-Experts using auxiliary-loss-free load balancing.

## What is different about this version

* **Expert capacity follows the token flow.** Rows a stack sees per micro-batch
  order as L1 decoder (2S) > L2 decoder (S/2) > L1 encoder (S/4) > L2 encoder
  (S/16), and expert counts now follow that order. The previous 8B config put
  256 experts on the *last* one -- 68% of the model -- and its load-imbalance
  metric (MaxVio) climbed monotonically past 30, meaning that capacity was inert.
* **llm-jp-tokenizer v4 (196,608).** Adopted because logit-level distillation
  needs the teacher's exact vocabulary, and because v4 carries the
  openai-harmony control tokens that make the agent format possible. It is
  *not* better at compressing Japanese -- measured on real Japanese Wikipedia
  it needs 1.7915 chars/token against v3's 1.8320. `d_token` dropped
  2048 -> 1024 to pay for the doubled vocabulary at the same embed+head cost.
* **Distillation, in two forms.** Sequence-level from frontier open models via
  their published transcripts (Kimi K3, DeepSeek-V4-pro, poolside Laguna S 2.1,
  DeepSeek-V3.2, GLM-5.1) plus llm-jp-4's own SFT corpus. Logit-level KD needs
  an exact tokenizer match, which is the reason the vocabulary is llm-jp v4.
* **Japanese is protected on purpose.** Every frontier agentic corpus measured
  **0.0% Japanese**, so the mid-training mixture caps English agentic data at
  0.18 and carries 0.74 Japanese. See `docs/distillation.md`.

## How good is it, honestly

Held-out Japanese Wikipedia (evaluated beyond record 700,000, which is past
everything this run consumed), against the two 250M models from the same code
base. Perplexity is per token and these models use different vocabularies, so
the only column that compares across rows is **bits per character**:

| model | train tokens | ja-wiki ppl | chars/token | **BPC** |
|---|---:|---:|---:|---:|
| 250M v4 | 200M | 51.61 | 1.8320 | **3.1056** |
| 250M v3-900m | 900M | 67.60 | 1.8320 | **3.3181** |
| **this model (8B-A1B)** | 344M | 69.00 | 1.7915 | **3.4098** |

This model is the worst of the three. Scaling parameters 32x while scaling
tokens 1.7x buys nothing: 0.04 tokens per parameter against the 250M v4's 0.8.
The architecture work in this version is real and the routing measurements hold,
but the binding constraint on this project is the token budget, and no amount of
architectural care substitutes for it.

## Size

| | |
|---|---|
| total parameters | {total:.2f} B |
| active per token | {active:.2f} B |
| FLOP-equivalent dense size | {amort:.2f} B |
| hierarchy | L={levels}, C_<=L={cp} tokens per top-level unit |
| context | {ctx} tokens |
| vocabulary | {vocab:,} (llm-jp-tokenizer v4, harmony chat format) |

The FLOP-equivalent figure is the one that governs speed: a level-l encoder runs
once per C_<=l tokens, so its cost is amortised. This model *thinks* with 8B
parameters and *costs* about what a {amort:.1f}B dense model costs.

## Generation protocols

* **HierGen** keeps encoder state at every level. Exact -- it reproduces the
  training-time distribution.
* **RecGen** keeps only the top-level KV cache and feeds the decoder cascade its
  own reconstructions, cutting KV traffic by roughly {kvratio:.0f}x. It is exact
  when recursive consistency holds, which is what the training objective
  `L_token + alpha * L_rec` optimises for.

## Usage

```python
from scala.model.hierarchy import ScalaForCausalLM
from scala.model.config import ScalaConfig
from scala.infer.generate import ScalaGenerator, GenerationConfig

cfg = ScalaConfig.load("model_config.json")
model = ScalaForCausalLM(cfg)
# load model.safetensors, then:
gen = ScalaGenerator(model, "cuda")
out = gen.generate(input_ids, GenerationConfig(mode="recgen", max_new_tokens=256))
```

## Agent use

The bundled tokenizer speaks openai-harmony, so tool calls round-trip through
the chat template. Two things a harness must get right:

1. **Stop on `<|call|>` as well as `<|return|>`.** `generation_config.json`
   lists both. Stopping only at `<|return|>` lets the model run past its own
   tool call and invent the tool's reply.
2. **Parse the channels.** `analysis` is private reasoning, `commentary` with a
   `to=` recipient is a tool call, `final` is the user-visible answer.

```python
from scala.infer.agent import parse_harmony, append_tool_result

turn = parse_harmony(generated_text)
if turn.wants_tool:
    call = turn.tool_calls[0]                  # .name, .arguments (parsed JSON)
    messages = append_tool_result(messages, call, my_tools[call.name](**call.arguments))
else:
    print(turn.final)                          # turn.reasoning holds the analysis
```

## Quantised variants

| repo suffix | format | bits/weight | weight RMS error | size |
|---|---|---:|---:|---:|
| `-fp8` | E4M3, per-channel scale | 8 | 0.026 | 1.9x |
| `-nvfp4` | E2M1, block 16, E4M3 scale | 4.5 | 0.096 | 3.3x |
| `-nvfp4-hier` | as above, level-2 encoder kept at FP8 | ~4.6 | 0.093 | 3.2x |
| `-mxfp4` | E2M1, block 32, E8M0 scale (OCP) | 4.25 | 0.117 | 3.4x |

Routers, embeddings, the LM head and all 1-D tensors stay in bf16 in every
variant: top-k routing over 192 experts turns on margins finer than 4-bit
resolution, and a wrong pick runs a different expert rather than degrading
gracefully. `-hier` additionally protects the level-2 encoder, which runs once
per 16 tokens (so 8-bit costs almost nothing in FLOPs) but whose MLA weights get
*multiplied together* by weight absorption for the latent KV cache.

## Accounting

```
{report}
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer", default="llm-jp/llm-jp-4-8b-base")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--push-to", default=None)
    args = ap.parse_args()

    src = Path(args.ckpt)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    cfg_path = args.config or (src / "model_config.json")
    if not Path(cfg_path).exists():
        cfg_path = src.parent / "model_config.json"
    cfg = ScalaConfig.load(cfg_path)

    model = ScalaForCausalLM(cfg)
    if (src / "state.pt").exists():
        sd = torch.load(src / "state.pt", map_location="cpu", weights_only=False)
        load_state_dict_compat(model, sd.get("model", sd))
        print(f"loaded single-file checkpoint {src/'state.pt'}")
    else:
        import torch.distributed.checkpoint as dcp

        state = {"model": model.state_dict()}
        dcp.load(state, checkpoint_id=str(src))
        load_state_dict_compat(model, state["model"])
        print(f"consolidated distributed checkpoint {src}")

    model = model.to(getattr(torch, args.dtype))

    from safetensors.torch import save_file

    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    if cfg.tie_word_embeddings:
        # safetensors cannot store two names for one storage; the loader
        # re-ties in __init__, so the duplicate is pure waste.
        tensors.pop("lm_head.weight", None)
    save_file(tensors, str(out / "model.safetensors"),
              metadata={"format": "pt",
                        "tie_word_embeddings": str(cfg.tie_word_embeddings)})
    cfg.save(out / "model_config.json")

    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                            trust_remote_code=True)
        tok.save_pretrained(out)

        # An agent harness needs the stop set, not just the EOS token.
        # Generation has to halt at <|call|> as well as <|return|>: stopping
        # only at <|return|> means the model runs past its own tool call and
        # invents the tool's reply, which is the usual way a tool loop breaks.
        from scala.infer.agent import STOP_TOKENS, stop_token_ids

        gen_cfg = {
            "bos_token_id": tok.bos_token_id,
            "eos_token_id": stop_token_ids(tok),
            "pad_token_id": tok.pad_token_id,
            "stop_strings": list(STOP_TOKENS),
            "temperature": 0.7,
            "top_p": 0.95,
            "_comment": "eos_token_id lists <|return|> AND <|call|>; see "
                        "scala/infer/agent.py",
        }
        (out / "generation_config.json").write_text(
            json.dumps(gen_cfg, indent=2), encoding="utf-8")
        print(f"bundled tokenizer {args.tokenizer} "
              f"(vocab {len(tok):,}, chat_template="
              f"{bool(getattr(tok, 'chat_template', None))}), "
              f"stop ids {gen_cfg['eos_token_id']}")
    except Exception as e:  # noqa: BLE001
        print(f"[warn] could not bundle the tokenizer: {e}")

    mc = count_model(cfg)
    from scala.model.accounting import kv_cache_bytes_per_token

    (out / "README.md").write_text(
        CARD.format(
            total=mc.total / 1e9, active=mc.active / 1e9,
            amort=mc.amortised_active / 1e9, levels=cfg.n_levels,
            cp=cfg.chunk_product, ctx=cfg.max_seq_len, vocab=cfg.vocab_size,
            kvratio=kv_cache_bytes_per_token(cfg)
            / max(kv_cache_bytes_per_token(cfg, recgen=True), 1e-9),
            report=format_report(cfg),
        ),
        encoding="utf-8",
    )
    for extra in ("meta.json",):
        if (src / extra).exists():
            shutil.copy(src / extra, out / extra)

    size = (out / "model.safetensors").stat().st_size / 2**30
    print(f"wrote {out}  ({size:.2f} GiB, {mc.total/1e9:.2f}B params)")

    if args.push_to:
        from huggingface_hub import HfApi

        api = HfApi()
        api.create_repo(args.push_to, exist_ok=True)
        api.upload_folder(folder_path=str(out), repo_id=args.push_to)
        print(f"pushed to https://huggingface.co/{args.push_to}")


if __name__ == "__main__":
    main()
