#!/usr/bin/env python
"""Scores substitution protocols (`content`/`up` token swaps, `recgen` window
sweep) against the teacher-forced training forward, not HierGen.

    python scripts/protocol_diag.py --ckpt runs/recgen-nolrec/final
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.infer.generate import PROTOCOLS, ScalaGenerator  # noqa: E402
from scala.model.config import ScalaConfig  # noqa: E402
from scala.model.hierarchy import ScalaForCausalLM  # noqa: E402

ORDER = ["hiergen", "xhat_content", "content_only", "up_only", "chunkgen",
         "recgen_paper"]
#: window widths (in C_{l+1}-unit groups) to sweep `recgen` over
WINDOWS = (1, 2, 4, 8)


def load_model(ckpt: str, config: str | None, device: str, dtype: torch.dtype,
               depth: int | None = None):
    """`depth` re-expresses a tied (SCALA) checkpoint at a different
    MID-application depth before loading; untied configs are rejected."""
    p = Path(ckpt)
    cfg_path = config or (p / "model_config.json")
    if not Path(cfg_path).exists():
        cfg_path = p.parent / "model_config.json"
    cfg = ScalaConfig.load(cfg_path)
    if depth is not None:
        from scala.model.scala import scala_config_at_depth

        cfg = scala_config_at_depth(cfg, depth)
        print(f"re-expressed at depth {depth}: L={cfg.n_levels}, "
              f"C_<=L={cfg.chunk_product}")
    model = ScalaForCausalLM(cfg)

    if (p / "model.safetensors").exists():
        from safetensors.torch import load_file

        sd = load_file(str(p / "model.safetensors"))
    elif (p / "state.pt").exists():
        blob = torch.load(p / "state.pt", map_location="cpu", weights_only=False)
        sd = blob.get("model", blob)
    else:
        import torch.distributed.checkpoint as dcp

        sd = {"model": model.state_dict()}
        dcp.load(sd, checkpoint_id=str(p))
        sd = sd["model"]
    missing, unexpected = model.load_state_dict(sd, strict=False)
    print(f"loaded {len(sd)} tensors ({len(missing)} missing, "
          f"{len(unexpected)} unexpected)")
    if len(missing) > 8:
        raise SystemExit(f"checkpoint does not match the config: {missing[:6]}")
    return model.to(device=device, dtype=dtype).eval(), cfg


def load_tokens(cfg: ScalaConfig, args, device) -> torch.Tensor:
    """Real tokens from a data shard if available, else random ids.

    Shard dtype must match manifest.json (a mismatch still decodes, silently
    wrong); the default offset sits inside training's read prefix, not a holdout.
    """
    root = Path(args.data_root) if args.data_root else None
    if root and (root / "manifest.json").exists():
        import numpy as np

        man = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        dtype = {"uint16": np.uint16, "uint32": np.uint32}[man.get("dtype", "uint32")]
        want = getattr(args, "data_source", None)
        srcs = [s for s in man["sources"] if not want or s["name"] == want]
        if want and not srcs:
            raise SystemExit(
                f"--data-source {want!r} not in {root/'manifest.json'}; have "
                + ", ".join(s["name"] for s in man["sources"]))
        shards = [root / sh["path"] for s in srcs for sh in s["shards"]]
        if shards:
            arr = np.memmap(shards[0], dtype=dtype, mode="r")
            need = args.batch * args.seq_len
            # `offset` gives disjoint batches per call; without it every call
            # returns the same tokens.
            off = getattr(args, "offset", 1024)
            if arr.size < need + off:
                raise SystemExit(
                    f"{shards[0].name} holds {arr.size} tokens; offset {off} + "
                    f"{need} needed.  Lower --data-offset, --batch or --batches.")
            buf = np.asarray(arr[off : off + need]).astype(np.int64)
            if buf.max() >= cfg.vocab_size:
                raise SystemExit(
                    f"{shards[0].name}: token id {buf.max()} >= vocab "
                    f"{cfg.vocab_size} -- the shard dtype in manifest.json "
                    f"({man.get('dtype')}) does not match the data")
            print(f"tokens: {shards[0].name} ({man.get('dtype', 'uint32')}) "
                  f"@{off}")
            return torch.tensor(buf, device=device).view(args.batch,
                                                         args.seq_len)
    print("tokens: RANDOM (no data shard found) -- results are a weak proxy")
    g = torch.Generator(device="cpu").manual_seed(0)
    return torch.randint(0, cfg.vocab_size, (args.batch, args.seq_len),
                         generator=g).to(device)


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--depth", type=int, default=None,
                    help="re-express a tied (SCALA) checkpoint at this many "
                         "MID applications before scoring")
    ap.add_argument("--data-root", default="data/tokens_ja")
    ap.add_argument("--data-source", default=None,
                    help="manifest source to score on; default is the first, "
                         "which for data/tokens_ja is English")
    ap.add_argument("--offset", type=int, default=1024,
                    help="token offset into the shard.  The default sits in "
                         "the prefix every training run reads first; pass a "
                         "value past total_tokens*weight for a real holdout")
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--prefix-meta", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    dtype = getattr(torch, args.dtype)
    model, cfg = load_model(args.ckpt, args.config, args.device, dtype,
                            depth=args.depth)
    tokens = load_tokens(cfg, args, args.device)
    cp = cfg.chunk_product
    split = args.prefix_meta * cp
    print(f"batch {tuple(tokens.shape)}  C_<=L={cp}  scoring positions "
          f"{split}..{tokens.shape[1]}\n")

    # ---- reference: one teacher-forced training forward -------------------- #
    ref = model(tokens, return_logits=True).logits[:, split:].float()
    ref_lp = F.log_softmax(ref, dim=-1)
    ref_tok = ref.argmax(-1)

    rows = []
    top_ref = None
    n_tokens = tokens.shape[0] * tokens.shape[1]

    def score(name: str, window: int | None = None):
        nonlocal top_ref
        gen = ScalaGenerator(model, args.device, dtype,
                              **({"enc_window_groups": window} if window else {}))
        lg = gen.forced_logits(tokens, mode=name,
                               prefix_meta=args.prefix_meta).float()
        n = min(lg.shape[1], ref.shape[1])
        lp = F.log_softmax(lg[:, :n], dim=-1)
        kl = F.kl_div(lp, ref_lp[:, :n], log_target=True,
                      reduction="none").sum(-1).mean().item()
        agree = (lg[:, :n].argmax(-1) == ref_tok[:, :n]).float().mean().item()

        # drift of the top-level state; `up` substitution corrupts this and
        # no loss term watches it
        tops = torch.stack(gen.top_states, dim=1).float()   # (B, n_groups, D_L)
        if top_ref is None:
            top_ref = tops
            top_cos = top_cos_last = 1.0
        else:
            c = F.cosine_similarity(tops, top_ref[:, : tops.shape[1]], dim=-1)
            top_cos, top_cos_last = c.mean().item(), c[:, -1].mean().item()

        p = PROTOCOLS[name]
        label = name if window is None else f"{name}(w={window})"
        row = {"protocol": label, "content": p.content, "up": p.up,
               "window": window, "agree_vs_train": agree, "kl_vs_train": kl,
               "top_cos": top_cos, "top_cos_last": top_cos_last,
               "kib_per_token": gen.cache_bytes() / n_tokens / 1024}
        rows.append(row)
        print(f"  {label:<16} agree {agree*100:5.1f}%   KL {kl:8.3f}   "
              f"cos(X_top) {top_cos:.4f}   KV {row['kib_per_token']:.3f} KiB/tok")
        return row

    for name in ORDER:
        score(name)
    # shipped protocol, swept over the bounded lower encoders' window
    win_rows = [score("recgen", w) for w in WINDOWS]

    print(f"\n{'protocol':<16}{'content':>9}{'up':>9}{'agree vs train':>16}"
          f"{'KL':>10}{'cos(X_top)':>12}{'KiB/token':>12}")
    print("-" * 84)
    for r in rows:
        print(f"{r['protocol']:<16}{r['content']:>9}{r['up']:>9}"
              f"{r['agree_vs_train']*100:>15.1f}%{r['kl_vs_train']:>10.3f}"
              f"{r['top_cos']:>12.4f}{r['kib_per_token']:>12.3f}")

    by = {r["protocol"]: r for r in rows}
    print(f"\nhiergen must be ~100% / KL ~0 -- it is the training forward.  "
          f"Measured {by['hiergen']['agree_vs_train']*100:.1f}% / "
          f"{by['hiergen']['kl_vs_train']:.4f}")
    a = 1.0 - by["content_only"]["agree_vs_train"]
    b = 1.0 - by["up_only"]["agree_vs_train"]
    both = 1.0 - by["chunkgen"]["agree_vs_train"]
    print(f"error attribution (1 - agreement): content {a*100:.1f}pp, "
          f"up {b*100:.1f}pp, both {both*100:.1f}pp")
    if b > a:
        print("  -> the UP path dominates.  chunk_cond_prob does not train for "
              "it, and no loss term does.  Substituting is the wrong lever; "
              "bounding the encoder removes the substitution entirely.")

    best = max(win_rows, key=lambda r: r["agree_vs_train"])
    print(f"\nbounded RecGen at its best window ({best['protocol']}): "
          f"{best['agree_vs_train']*100:.1f}% at {best['kib_per_token']:.3f} "
          f"KiB/token, against {by['recgen_paper']['agree_vs_train']*100:.1f}% "
          f"at {by['recgen_paper']['kib_per_token']:.3f} for the paper's rule "
          f"and {by['hiergen']['agree_vs_train']*100:.1f}% at "
          f"{by['hiergen']['kib_per_token']:.3f} for HierGen.")

    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.json_out}")


if __name__ == "__main__":
    main()
