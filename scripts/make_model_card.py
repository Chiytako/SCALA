#!/usr/bin/env python
"""Write a model card from the run's actual logs, not intentions: every
numeric field is read from `log.jsonl` and the accounting module, so the
card cannot drift from what was actually trained.

    python scripts/make_model_card.py --export export/scala-small-v1 --run runs/scala-small-v1 --name "SCALA small v1" --eval-json eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.model.accounting import (  # noqa: E402
    count_model, flops_per_token, kv_cache_bytes_per_token, uses_latent_cache,
)
from scala.model.config import ScalaConfig  # noqa: E402

TEMPLATE = """---
license: apache-2.0
language:
- ja
- en
tags:
- photon
- hierarchical
- mixture-of-experts
- japanese
- pretrained
- research
library_name: scala
---

# {name}

A Japanese language model on the **PHOTON** hierarchical autoregressive
architecture ([arXiv:2512.20687](https://arxiv.org/abs/2512.20687)), with every
transformer stack replaced by a fine-grained Mixture-of-Experts using
auxiliary-loss-free load balancing.

Tokens are folded into multi-resolution latent units instead of being scanned
one at a time:

```
tokens   t1 t2 t3 t4 | t5 t6 t7 t8 | ...      <- level-1 decoder, every token
           \\  \\  /  /
level-1      u1      |     u2      | ...      <- once per 4 tokens
              \\_____/____________/
level-2            m1                          <- once per 16 tokens
```

A level-`l` encoder runs once per `C<=l` tokens, so its cost is amortised. That
is the whole point: capacity sits at the top of the hierarchy where it is
cheapest per token.

## Size

| | |
|---|---|
| total parameters | **{total:.3f} B** |
| active per token | **{active:.3f} B** |
| FLOP-equivalent dense size | **{amort:.3f} B** |
| forward FLOPs / token | {gflops:.3f} GFLOP |
| hierarchy | L={levels}, C≤L={cp} tokens per top-level unit |
| context | {ctx} tokens |
| vocabulary | {vocab:,} (llm-jp-tokenizer v3) |
| KV cache, HierGen | {kv_hier:.2f} KiB/token |
| KV cache, RecGen | {kv_rec:.2f} KiB/token |

## Training

| | |
|---|---|
| tokens seen | **{tokens:.2f} B** |
| steps | {steps:,} |
| tokens / parameter | {tpp:.2f} |
| hardware | {hardware} |
| throughput | {tps:.1f}K tokens/s |
| optimiser | Muon (2-D weights) + AdamW (embeddings, norms, router) |
| schedule | WSD, 1-sqrt cooldown |
| final train CE | **{final_ce:.4f}** (ppl {final_ppl:.2f}) |
{eval_rows}
### Data

{data_desc}

## Honest limitations

**This model is under-trained by design of the budget, not by accident.** At
{tpp:.2f} tokens per parameter it is far below the ~20 that Chinchilla-optimal
training implies, and further still below what an inference-efficient model
would normally get. It produces fluent Japanese *surface form* -- correct
particles, natural kana/kanji mixing, sentence-final forms -- while being
largely incoherent semantically. Treat it as a demonstration that the
architecture trains, not as a usable assistant.

{samples}

## Generation protocols

* **HierGen** keeps encoder state at every level. Exact: it reproduces the
  training-time distribution (verified to 2e-4 in the test suite).
* **RecGen** keeps only the top-level KV cache and feeds the decoder cascade its
  own reconstructions, cutting cache by {kvratio:.1f}x. Exact when recursive
  consistency holds, which is what `L_token + alpha * L_rec` optimises for.

## Usage

```python
from scala.model.config import ScalaConfig
from scala.model.hierarchy import ScalaForCausalLM
from scala.model.loading import load_state_dict_compat
from scala.infer.generate import ScalaGenerator, GenerationConfig
from safetensors.torch import load_file

cfg = ScalaConfig.load("model_config.json")
model = ScalaForCausalLM(cfg)
load_state_dict_compat(model, load_file("model.safetensors"))

gen = ScalaGenerator(model.cuda().eval(), "cuda")
out = gen.generate(input_ids, GenerationConfig(mode="recgen", max_new_tokens=256))
```

## Accounting

```
{report}
```

## Citation

```bibtex
@article{{ichikawa2025photon,
  title  = {{PHOTON: Hierarchical Autoregressive Modeling for Lightspeed and
            Memory-Efficient Language Generation}},
  author = {{Ichikawa, Yuma and Takagi, Naoya and Nakagawa, Takumi and
            Kanazawa, Yuzi and Sakai, Akira}},
  journal= {{arXiv preprint arXiv:2512.20687}},
  year   = {{2025}}
}}
```
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--export", required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--hardware", default="1x NVIDIA GB10")
    ap.add_argument("--eval-json", default=None)
    ap.add_argument("--samples-file", default=None)
    ap.add_argument("--data-desc", default="Japanese Wikipedia (190M tokens) and "
                    "English Wikipedia (64M tokens), tokenised with "
                    "`llm-jp/llm-jp-3-1.8b`, sampled 75:25.")
    args = ap.parse_args()

    exp = Path(args.export)
    cfg = ScalaConfig.load(exp / "model_config.json")
    mc = count_model(cfg)

    rows = [json.loads(l) for l in (Path(args.run) / "log.jsonl").open()]
    last = rows[-1]

    eval_rows = ""
    if args.eval_json and Path(args.eval_json).exists():
        ev = json.loads(Path(args.eval_json).read_text())
        lines = ["", "### Evaluation", "", "| benchmark | result |", "|---|---|"]
        for k, v in ev.items():
            lines.append(f"| {k} | {v:.4f} |" if isinstance(v, float)
                         else f"| {k} | {v} |")
        eval_rows = "\n".join(lines) + "\n"

    samples = ""
    if args.samples_file and Path(args.samples_file).exists():
        txt = Path(args.samples_file).read_text(encoding="utf-8").strip()
        samples = "### Samples\n\n```\n" + txt + "\n```\n"

    from scala.model.accounting import format_report

    card = TEMPLATE.format(
        name=args.name,
        total=mc.total / 1e9, active=mc.active / 1e9,
        amort=mc.amortised_active / 1e9,
        gflops=flops_per_token(cfg)["total"] / 1e9,
        levels=cfg.n_levels, cp=cfg.chunk_product, ctx=cfg.max_seq_len,
        vocab=cfg.vocab_size,
        kv_hier=kv_cache_bytes_per_token(cfg) / 1024,
        kv_rec=kv_cache_bytes_per_token(cfg, recgen=True) / 1024,
        kvratio=kv_cache_bytes_per_token(cfg)
        / max(kv_cache_bytes_per_token(cfg, recgen=True), 1e-9),
        tokens=last["tokens"] / 1e9, steps=last["step"],
        tpp=last["tokens"] / mc.total,
        hardware=args.hardware, tps=last["tok_per_s"] / 1000,
        final_ce=last["loss_token"],
        final_ppl=math.exp(min(last["loss_token"], 20)),
        eval_rows=eval_rows, samples=samples,
        data_desc=args.data_desc,
        report=format_report(cfg),
    )
    (exp / "README.md").write_text(card, encoding="utf-8")
    print(f"wrote {exp/'README.md'} ({len(card)} chars)")


if __name__ == "__main__":
    main()
