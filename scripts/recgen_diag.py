#!/usr/bin/env python
"""Per-level RecGen diagnostics.  A = cos(X_hat_j, X_j): what L_rec scores --
decoder output at position R-1+j, a next-unit *prediction* (has not seen slot
j).  B = cos(post_j, X_j): the *reconstruction* at R+j.  C = cos(chunker
(content), X): what the encoder adds over the chunker.  `_emit_group`
substitutes only for l > 1; at l == 1 it emits real embeddings, so L_rec's
l == 1 term scores a substitution that never happens at inference.
"""
import argparse
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate import load as load_model  # noqa: E402

PROMPTS = [
    "日本の首都は東京であり、人口はおよそ1400万人である。東京は政治と経済の中心地として"
    "機能しており、多くの企業の本社が集まっている。交通網も発達しているため、周辺の県から"
    "通勤する人々も非常に多い。歴史的には江戸と呼ばれ、幕府が置かれた都市であった。",
    "機械学習とは、データから規則性を自動的に見つけ出す技術の総称である。近年は深層学習が"
    "中心となり、画像認識や自然言語処理の分野で大きな成果を上げている。学習には大量の"
    "データと計算資源が必要であり、モデルの規模が性能に強く影響することが知られている。",
    "The attention mechanism computes a weighted sum over value vectors, where the "
    "weights come from a compatibility function between the query and the "
    "corresponding keys. This lets the model route information between arbitrary "
    "positions in constant path length, which is what made deep transformers "
    "trainable at scale and displaced recurrent architectures for sequence tasks.",
]


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--config", default=None)
    ap.add_argument("--tokenizer", default="llm-jp/llm-jp-4-8b-base")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    ap.add_argument("--dtype", default="bfloat16")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.tokenizer,
                                        trust_remote_code=True)
    model, cfg = load_model(args.ckpt, args.config, args.device,
                            getattr(torch, args.dtype))
    model.eval()
    L = model.n_levels
    print(f"levels={L}  chunks={[lv.chunk_size for lv in cfg.levels]}")

    ids = [tok(p, add_special_tokens=False)["input_ids"] for p in PROMPTS]
    n = min(len(x) for x in ids)
    n -= n % cfg.chunk_product            # must be a multiple of C_<=L
    # n < chunk_product silently truncates to zero and the MoE router then
    # reshapes an empty tensor -- fail loudly instead
    assert n >= 2 * cfg.chunk_product, (
        f"prompts too short: {n} tokens after rounding to a multiple of "
        f"{cfg.chunk_product}")
    x = torch.tensor([i[:n] for i in ids], device=args.device,
                     dtype=torch.long)
    print(f"batch {tuple(x.shape)}\n")

    enc = model.encode_all(x)             # enc[0]=embeddings .. enc[L]=X^(L)

    stats = {}
    cond = enc[L]
    for l in range(L, 0, -1):
        lvl = model.levels[l - 1]
        content = enc[l - 1]
        R, C = lvl.width, lvl.chunk
        B, Mu = content.shape[0], content.shape[1] // C

        src = lvl.shift_cond(cond) if l == L else cond
        u = lvl.converter(src)
        c = content.view(B, Mu, C, lvl.d_below)
        seq = torch.cat([u, c], dim=2).reshape(B * Mu, R + C, lvl.d_below)
        out = lvl.decoder(seq)

        pre = out[:, R - 1: R - 1 + C].reshape(B, Mu * C, lvl.d_below)   # A
        post = out[:, R: R + C].reshape(B, Mu * C, lvl.d_below)          # B
        tgt = content

        a = F.cosine_similarity(pre.float(), tgt.float(), dim=-1).mean().item()
        b = F.cosine_similarity(post.float(), tgt.float(), dim=-1).mean().item()
        stats[l] = {"A_pred": a, "B_recon": b}

        # C: what the chunker alone gives, one level down
        if l > 1:
            below = model.levels[l - 2]
            ch = below.chunker(enc[l - 2])
            cc = F.cosine_similarity(ch.float(), enc[l - 1].float(),
                                     dim=-1).mean().item()
            stats[l]["C_chunker"] = cc

        cond = pre.detach()

    print(f"{'level':<8}{'A cos(X_hat, X)':>18}{'B cos(post, X)':>17}"
          f"{'C cos(chunk, X)':>18}{'RecGen substitutes?':>22}")
    print("-" * 83)
    tot_a = 0.0
    for l in sorted(stats):
        s = stats[l]
        subs = "YES" if l > 1 else "NO -- uses real embeddings"
        print(f"L{l:<7}{s['A_pred']:>18.4f}{s['B_recon']:>17.4f}"
              f"{s.get('C_chunker', float('nan')):>18.4f}{subs:>22}")
        tot_a += 1.0 - s["A_pred"]
    print(f"\nsum over levels of (1 - A) = {tot_a:.4f}"
          f"   <- this is what `loss_rec` reports")


if __name__ == "__main__":
    main()
