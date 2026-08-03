#!/usr/bin/env python
"""Search vast.ai for a suitable instance and estimate the cost of the run.

    # what will 120B tokens cost, and on what?
    python vastai/plan_run.py --budget-tokens 120e9

    # search live offers (needs `pip install vastai` and an API key)
    python vastai/plan_run.py --search --gpus 8 --gpu-name H200

    # print the ready-to-paste create command for the cheapest viable offer
    python vastai/plan_run.py --search --create

The throughput model is analytic: PHOTON's forward FLOPs/token come from
``scala.model.accounting``, training costs ~3x forward (fwd + bwd), and we
assume a Model-FLOPs-Utilisation you can override.  Treat the result as a
planning estimate, then replace ``--mfu`` with the tok/s the first hundred
steps actually report.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scala.model.accounting import count_model, flops_per_token  # noqa: E402
from scala.model.config import ScalaConfig  # noqa: E402

# Dense bf16 throughput, TFLOP/s, and typical vast.ai on-demand $/hr per GPU.
# Prices move constantly -- these are only for the offline estimate; --search
# reads live numbers.
GPUS = {
    "H100_SXM":  {"tflops": 990,  "vram": 80,  "usd_hr": 2.30},
    "H100_PCIE": {"tflops": 756,  "vram": 80,  "usd_hr": 1.95},
    "H200":      {"tflops": 990,  "vram": 141, "usd_hr": 2.60},
    "B200":      {"tflops": 2250, "vram": 180, "usd_hr": 4.80},
    "A100_SXM4": {"tflops": 312,  "vram": 80,  "usd_hr": 1.20},
    "RTX_4090":  {"tflops": 165,  "vram": 24,  "usd_hr": 0.35},
    "RTX_5090":  {"tflops": 210,  "vram": 32,  "usd_hr": 0.55},
}


def estimate(cfg: ScalaConfig, budget: float, gpu: str, n_gpus: int,
             mfu: float) -> dict:
    spec = GPUS[gpu]
    fwd = flops_per_token(cfg)["total"]
    train_flops_per_token = 3.0 * fwd          # fwd + bwd
    total_flops = train_flops_per_token * budget

    achieved = spec["tflops"] * 1e12 * mfu * n_gpus
    seconds = total_flops / achieved
    hours = seconds / 3600
    cost = hours * spec["usd_hr"] * n_gpus
    return {
        "gpu": gpu, "n_gpus": n_gpus,
        "fwd_gflops_per_token": fwd / 1e9,
        "train_pflops_total": total_flops / 1e15,
        "tok_per_s": budget / seconds,
        "hours": hours, "days": hours / 24,
        "usd": cost, "usd_per_hr": spec["usd_hr"] * n_gpus,
    }


def memory_check(cfg: ScalaConfig, n_gpus: int, vram_gb: int) -> dict:
    """Rough per-GPU memory for FSDP2 + bf16 params + Muon momentum."""
    mc = count_model(cfg)
    p = mc.total
    # bf16 params + bf16 grads + fp32 Muon momentum (2D) + fp32 AdamW m,v (rest)
    bytes_total = p * 2 + p * 2 + p * 4 + p * 0.6 * 4
    per_gpu = bytes_total / n_gpus / 2**30
    return {
        "params_B": p / 1e9,
        "state_gib_per_gpu": per_gpu,
        "headroom_gib": vram_gb - per_gpu,
        "fits": per_gpu < vram_gb * 0.55,   # leave >45% for activations
    }


# --------------------------------------------------------------------------- #
def search_offers(gpu_name: str | None, n_gpus: int, max_usd: float,
                  min_vram: int) -> list[dict]:
    if not shutil.which("vastai"):
        sys.exit("`vastai` CLI not found.  pip install vastai && "
                 "vastai set api-key <KEY>")
    q = [
        f"num_gpus={n_gpus}",
        f"gpu_ram>={min_vram}",
        "reliability>0.98",
        "inet_down>=500",
        "disk_space>=1000",
        "rentable=true",
        "verified=true",
        f"dph_total<{max_usd}",
    ]
    if gpu_name:
        q.append(f"gpu_name={gpu_name}")
    cmd = ["vastai", "search", "offers", " ".join(q), "--raw", "-o", "dph_total"]
    out = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if out.returncode != 0:
        sys.exit(f"vastai search failed:\n{out.stderr}")
    try:
        return json.loads(out.stdout)
    except json.JSONDecodeError:
        sys.exit(f"could not parse vastai output:\n{out.stdout[:500]}")


def create_command(offer: dict, image: str, disk: int, repo: str | None) -> str:
    env = "-e TOKENIZERS_PARALLELISM=true -e PYTHONUNBUFFERED=1"
    if repo:
        env += f" -e REPO_URL={repo}"
    return (
        f"vastai create instance {offer['id']} "
        f"--image {image} --disk {disk} {env} "
        f"--ssh --direct --jupyter-lab false "
        f"--onstart-cmd 'bash -c \"curl -fsSL $PROVISION_URL | bash\"'"
    )


# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/base_8b_a1b.yaml")
    ap.add_argument("--budget-tokens", type=float, default=120e9)
    ap.add_argument("--mfu", type=float, default=0.35,
                    help="model-FLOPs utilisation; MoE at this scale typically "
                         "lands 0.25-0.40")
    ap.add_argument("--gpus", type=int, default=8)
    ap.add_argument("--gpu-name", default=None,
                    help=f"one of {', '.join(GPUS)} (offline table) or any "
                         "vast.ai gpu_name (with --search)")
    ap.add_argument("--search", action="store_true", help="query live offers")
    ap.add_argument("--create", action="store_true",
                    help="print the create command for the cheapest offer")
    ap.add_argument("--max-usd-hr", type=float, default=30.0)
    ap.add_argument("--min-vram", type=int, default=80)
    ap.add_argument("--image",
                    default="pytorch/pytorch:2.9.0-cuda12.8-cudnn9-devel")
    ap.add_argument("--disk", type=int, default=1500)
    ap.add_argument("--repo", default=None)
    args = ap.parse_args()

    cfg = ScalaConfig.load(args.config)
    mc = count_model(cfg)

    print("=" * 78)
    print(f"SCALA  {mc.total/1e9:.2f}B total / {mc.active/1e9:.2f}B active "
          f"/ {mc.amortised_active/1e9:.2f}B FLOP-equivalent")
    print(f"budget     {args.budget_tokens/1e9:.0f}B tokens "
          f"({args.budget_tokens/mc.total:.1f} tok/param, "
          f"{args.budget_tokens/mc.active:.1f} tok/active-param)")
    print(f"assumed MFU {args.mfu:.0%}")
    print("=" * 78)

    names = [args.gpu_name] if args.gpu_name in GPUS else list(GPUS)
    print(f"\n{'gpu':<12}{'n':>3}{'tok/s':>12}{'days':>8}{'$/hr':>9}{'total $':>11}"
          f"{'state GiB/gpu':>15}{'fits':>6}")
    print("-" * 78)
    for g in names:
        e = estimate(cfg, args.budget_tokens, g, args.gpus, args.mfu)
        m = memory_check(cfg, args.gpus, GPUS[g]["vram"])
        print(f"{g:<12}{args.gpus:>3}{e['tok_per_s']:>12,.0f}{e['days']:>8.2f}"
              f"{e['usd_per_hr']:>9.2f}{e['usd']:>11,.0f}"
              f"{m['state_gib_per_gpu']:>15.1f}{'yes' if m['fits'] else 'NO':>6}")

    print("\n'state GiB/gpu' = sharded bf16 params + grads + Muon/AdamW state;")
    print("'fits' requires it to stay under ~55% of VRAM so activations fit.")

    if not args.search:
        print("\n(pass --search to query live vast.ai offers)")
        return

    offers = search_offers(args.gpu_name, args.gpus, args.max_usd_hr,
                           args.min_vram)
    if not offers:
        print("\nno offers matched")
        return
    print(f"\n{len(offers)} live offers (cheapest 10):")
    print(f"{'id':>10}{'gpu':<20}{'n':>3}{'$/hr':>9}{'vram':>7}"
          f"{'net Mbps':>10}{'rel':>7}")
    print("-" * 70)
    for o in offers[:10]:
        print(f"{o['id']:>10}{o['gpu_name']:<20}{o['num_gpus']:>3}"
              f"{o['dph_total']:>9.2f}{o.get('gpu_ram', 0):>7.0f}"
              f"{o.get('inet_down', 0):>10.0f}{o.get('reliability2', 0):>7.3f}")

    best = offers[0]
    hours = estimate(cfg, args.budget_tokens,
                     args.gpu_name if args.gpu_name in GPUS else "H100_SXM",
                     best["num_gpus"], args.mfu)["hours"]
    print(f"\ncheapest offer {best['id']}: ${best['dph_total']:.2f}/hr "
          f"-> ~${best['dph_total']*hours:,.0f} for the run "
          f"({hours/24:.1f} days)")

    if args.create:
        print("\n" + create_command(best, args.image, args.disk, args.repo))


if __name__ == "__main__":
    main()
