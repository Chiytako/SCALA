#!/usr/bin/env python
"""Quantize a SCALA export to NVFP4: block-scaled 4-bit float (E2M1 elements,
block-16 shared E4M3 scale, one FP32 global scale per tensor). Router/gate,
embedding/LM-head, and 1-D tensors stay in higher precision. `--verify`
dequantizes and reports the RMS error against the original weights.

    python scripts/quantize_nvfp4.py export/scala-8b-a1b-v3 --out export/scala-8b-a1b-v3-nvfp4
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

E2M1_MAX = 6.0
E4M3_MAX = 448.0
BLOCK = 16

#: the eight magnitudes E2M1 can represent
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])
#: midpoints, for round-to-nearest by bucketing
_E2M1_EDGES = (_E2M1_LEVELS[1:] + _E2M1_LEVELS[:-1]) / 2

#: tensors kept in high precision: embeddings/LM-head, router/gate, norms, biases
SKIP = re.compile(
    r"(^embed\.|^lm_head\.|"          # 196,608-way softmax at both ends
    r"\.gate\.weight$|\.router|"      # top-k selection is discrete
    r"_norm\.|\.norm\d*\.|"           # per-channel gains
    r"start_latent|res_scale|bias)"
)


def to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Round |x| to the nearest E2M1 level and return 4-bit codes.

    Code layout is the usual E2M1 ordering: bit 3 is the sign, bits 2..0 index
    the magnitude in ``_E2M1_LEVELS``.
    """
    sign = (x < 0).to(torch.uint8)
    mag = x.abs()
    idx = torch.bucketize(mag, _E2M1_EDGES.to(mag.device))
    return (idx.to(torch.uint8) | (sign << 3))


def from_e2m1_codes(codes: torch.Tensor) -> torch.Tensor:
    levels = _E2M1_LEVELS.to(codes.device)
    mag = levels[(codes & 0x7).long()]
    return torch.where((codes & 0x8) != 0, -mag, mag)


def pack_nibbles(codes: torch.Tensor) -> torch.Tensor:
    """Pack the last dim (even length) two-per-byte, low nibble first."""
    flat = codes.reshape(*codes.shape[:-1], -1, 2)
    return (flat[..., 0] | (flat[..., 1] << 4)).contiguous()


def unpack_nibbles(packed: torch.Tensor) -> torch.Tensor:
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    return torch.stack([lo, hi], dim=-1).reshape(*packed.shape[:-1], -1)


def quantize_tensor(w: torch.Tensor):
    """Return (packed_uint8, scales_e4m3, global_scale_fp32)."""
    orig = w.shape
    w32 = w.detach().to(torch.float32)
    amax = w32.abs().max()
    if amax == 0:
        amax = torch.tensor(1.0)
    # Put the block scales where E4M3 can hold them.
    global_scale = (E4M3_MAX * E2M1_MAX) / amax

    blocks = w32.reshape(orig[0], -1, BLOCK)
    block_amax = blocks.abs().amax(dim=-1, keepdim=True)
    s_b = block_amax / E2M1_MAX
    # quantise weights against the rounded (fp8) scale, not the ideal one, or
    # dequantisation reintroduces the scale's own rounding error
    s_fp8 = (s_b * global_scale).to(torch.float8_e4m3fn)
    s_eff = s_fp8.to(torch.float32).clamp_min(1e-12)

    q = blocks * global_scale / s_eff
    codes = to_e2m1_codes(q.clamp(-E2M1_MAX, E2M1_MAX))
    packed = pack_nibbles(codes.reshape(orig[0], -1))
    return packed, s_fp8.squeeze(-1).contiguous(), global_scale.to(torch.float32)


def dequantize_tensor(packed, s_fp8, global_scale, out_features: int):
    codes = unpack_nibbles(packed).reshape(out_features, -1, BLOCK)
    vals = from_e2m1_codes(codes)
    s = s_fp8.to(torch.float32).unsqueeze(-1)
    return (vals * s / global_scale).reshape(out_features, -1)


# --------------------------------------------------------------------------- #
# MXFP4 (OCP microscaling): E2M1 elements, shared E8M0 (power-of-two) scale
# over 32 elements instead of 16 -- coarser than NVFP4, portable across
# toolchains that lack Blackwell-specific NVFP4 support.
# --------------------------------------------------------------------------- #
MX_BLOCK = 32
_E8M0_BIAS = 127


def quantize_tensor_mxfp4(w: torch.Tensor):
    """Return (packed_uint8, shared_exponents_uint8)."""
    w32 = w.detach().to(torch.float32)
    blocks = w32.reshape(w32.shape[0], -1, MX_BLOCK)
    amax = blocks.abs().amax(dim=-1, keepdim=True)
    # OCP MX: shared exponent X = floor(log2(amax)) - emax_elem, and for E2M1
    # emax_elem = 2 (the largest power of two it represents is 4).
    e = torch.floor(torch.log2(amax.clamp_min(1e-30))) - 2.0
    e = e.clamp(-_E8M0_BIAS, 255 - _E8M0_BIAS)
    e = torch.where(amax == 0, torch.full_like(e, -_E8M0_BIAS), e)
    scale = torch.pow(2.0, e)
    codes = to_e2m1_codes((blocks / scale).clamp(-E2M1_MAX, E2M1_MAX))
    packed = pack_nibbles(codes.reshape(w32.shape[0], -1))
    exps = (e.squeeze(-1) + _E8M0_BIAS).to(torch.uint8)
    return packed, exps.contiguous()


def dequantize_tensor_mxfp4(packed, exps, out_features: int):
    codes = unpack_nibbles(packed).reshape(out_features, -1, MX_BLOCK)
    vals = from_e2m1_codes(codes)
    scale = torch.pow(2.0, exps.to(torch.float32) - _E8M0_BIAS).unsqueeze(-1)
    return (vals * scale).reshape(out_features, -1)


# --------------------------------------------------------------------------- #
# FP8 E4M3, per-output-channel scale: ~2x compression, near-lossless; use
# when preserving behaviour matters more than size.
# --------------------------------------------------------------------------- #
def quantize_tensor_fp8(w: torch.Tensor):
    w32 = w.detach().to(torch.float32)
    amax = w32.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scale = amax / E4M3_MAX
    q = (w32 / scale).clamp(-E4M3_MAX, E4M3_MAX).to(torch.float8_e4m3fn)
    return q, scale.squeeze(-1).to(torch.float32).contiguous()


def dequantize_tensor_fp8(q, scale):
    return q.to(torch.float32) * scale.unsqueeze(-1)


_FMT_CODE = {"nvfp4": 0, "mxfp4": 1, "fp8": 2}
_CODE_FMT = {v: k for k, v in _FMT_CODE.items()}

#: level-2 encoder: runs once per 16 tokens (8-bit costs ~0 FLOPs), but
#: conditions decoding of all 16 and its MLA weights are multiplied together
#: by weight absorption, turning additive quantisation error multiplicative.
_L2_ENCODER = re.compile(r"^levels\.1\.encoder\.")


def pick_format(name: str, args) -> str:
    if args.policy == "hierarchy" and _L2_ENCODER.search(name):
        return "fp8"
    return args.format


_SIDECAR = ("packed", "scale", "global_scale", "shape", "red_axis", "fmt")


def load_quantized_state_dict(path: str | Path, dtype=torch.bfloat16) -> dict:
    """Reference loader: quantised safetensors -> an ordinary state dict.

    Dequantises on the host, so it is for correctness checks and for running on
    hardware without 4-bit kernels -- it hands back the *quantised* weights (the
    rounding is already baked in), materialised in ``dtype``.  It saves no
    memory at runtime; that needs real kernels.
    """
    from safetensors.torch import load_file

    raw = load_file(str(Path(path) / "model.safetensors"))
    names = {k[: -len(".packed")] for k in raw if k.endswith(".packed")}
    out: dict[str, torch.Tensor] = {
        k: v for k, v in raw.items()
        if not (k.rsplit(".", 1)[-1] in _SIDECAR
                and k.rsplit(".", 1)[0] in names)
    }
    for n in names:
        shape = tuple(int(x) for x in raw[n + ".shape"].tolist())
        red = int(raw[n + ".red_axis"][0])
        fmt = _CODE_FMT[int(raw[n + ".fmt"][0])] if n + ".fmt" in raw else "nvfp4"
        moved_shape = list(shape)
        moved_shape.append(moved_shape.pop(red % len(shape)))
        rows = int(torch.tensor(moved_shape[:-1]).prod())
        if fmt == "nvfp4":
            w = dequantize_tensor(raw[n + ".packed"], raw[n + ".scale"],
                                  raw[n + ".global_scale"], rows)
        elif fmt == "mxfp4":
            w = dequantize_tensor_mxfp4(raw[n + ".packed"],
                                        raw[n + ".scale"], rows)
        else:
            w = dequantize_tensor_fp8(raw[n + ".packed"], raw[n + ".scale"])
        out[n] = w.reshape(moved_shape).movedim(-1, red).to(dtype).contiguous()
    return out


#: kept for callers written against the first version of this script
load_nvfp4_state_dict = load_quantized_state_dict


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("src", help="directory holding model.safetensors")
    ap.add_argument("--out", required=True)
    ap.add_argument("--verify", action="store_true",
                    help="dequantise and report per-tensor relative error")
    ap.add_argument("--format", default="nvfp4",
                    choices=["nvfp4", "mxfp4", "fp8"],
                    help="nvfp4: E2M1 + E4M3 scales, block 16 (Blackwell). "
                         "mxfp4: E2M1 + E8M0 power-of-two scales, block 32 "
                         "(portable, coarser). fp8: E4M3 per-channel, ~2x, "
                         "near-lossless.")
    ap.add_argument("--policy", default="uniform",
                    choices=["uniform", "hierarchy"],
                    help="hierarchy: keep the level-2 encoder at fp8 whatever "
                         "--format says. That stack runs once per 16 tokens so "
                         "8-bit costs almost nothing in FLOPs, but its output "
                         "conditions all 16 and its MLA weights get multiplied "
                         "together by weight absorption.")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available()
                    else "cpu")
    args = ap.parse_args()

    from safetensors.torch import load_file, save_file

    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sd = load_file(str(src / "model.safetensors"))

    new: dict[str, torch.Tensor] = {}
    quantised, kept = [], []
    err_num = err_den = 0.0
    worst = []

    for name, t in sd.items():
        # reduction axis is dim -1 for nn.Linear [out,in], dim -2 for MoE
        # w_gate_up/w_down (see scala/model/moe.py); blocking the wrong axis
        # makes the block scale meaningless, so it's moved to last before
        # quantising and back on load.
        red_axis = -2 if t.ndim == 3 else -1

        skip_reason = None
        if SKIP.search(name):
            skip_reason = "excluded by policy"
        elif t.ndim not in (2, 3):
            skip_reason = f"{t.ndim}-D"
        elif t.shape[red_axis] % BLOCK != 0:
            skip_reason = (f"reduction dim {t.shape[red_axis]} not a multiple "
                           f"of {BLOCK}")

        if skip_reason:
            new[name] = t
            kept.append((name, t.numel(), skip_reason))
            continue

        fmt = pick_format(name, args)
        blk = MX_BLOCK if fmt == "mxfp4" else BLOCK
        if fmt != "fp8" and t.shape[red_axis] % blk != 0:
            new[name] = t
            kept.append((name, t.numel(),
                         f"reduction dim {t.shape[red_axis]} not a multiple "
                         f"of {blk} ({fmt})"))
            continue

        w = t.to(args.device)
        moved = w.movedim(red_axis, -1).contiguous()   # reduction axis last
        w2 = moved.reshape(-1, moved.shape[-1])

        if fmt == "nvfp4":
            packed, scales, gscale = quantize_tensor(w2)
            new[name + ".packed"] = packed.cpu()
            new[name + ".scale"] = scales.cpu()
            new[name + ".global_scale"] = gscale.cpu()
            deq = (dequantize_tensor(packed, scales, gscale, w2.shape[0])
                   if args.verify else None)
        elif fmt == "mxfp4":
            packed, exps = quantize_tensor_mxfp4(w2)
            new[name + ".packed"] = packed.cpu()
            new[name + ".scale"] = exps.cpu()
            deq = (dequantize_tensor_mxfp4(packed, exps, w2.shape[0])
                   if args.verify else None)
        else:  # fp8
            q, scale = quantize_tensor_fp8(w2)
            new[name + ".packed"] = q.cpu()
            new[name + ".scale"] = scale.cpu()
            deq = dequantize_tensor_fp8(q, scale) if args.verify else None

        new[name + ".shape"] = torch.tensor(list(t.shape), dtype=torch.int32)
        new[name + ".red_axis"] = torch.tensor([red_axis], dtype=torch.int32)
        new[name + ".fmt"] = torch.tensor([_FMT_CODE[fmt]], dtype=torch.int32)
        quantised.append((name, t.numel(), fmt))

        if args.verify:
            deq = deq.reshape(moved.shape).movedim(-1, red_axis)
            d = (deq - w.float())
            err_num += float(d.pow(2).sum())
            err_den += float(w.float().pow(2).sum())
            rel = float(d.norm() / max(w.float().norm(), 1e-12))
            worst.append((rel, name, fmt))

    import collections

    nq = sum(n for _, n, _ in quantised)
    nk = sum(n for _, n, _ in kept)
    by_fmt = collections.Counter()
    for _, n, f in quantised:
        by_fmt[f] += n
    save_file(new, str(out / "model.safetensors"), metadata={"format": "pt"})

    qcfg = {
        "quant_method": args.format,
        "policy": args.policy,
        "parameters_by_format": {f: int(n) for f, n in by_fmt.items()},
        "formats": {
            "nvfp4": "E2M1 elements, block 16, E4M3 block scale, FP32 "
                     "per-tensor global scale",
            "mxfp4": "E2M1 elements, block 32, E8M0 (power-of-two) block "
                     "scale -- OCP microscaling, coarser but portable",
            "fp8": "E4M3 elements, one FP32 scale per output channel",
        },
        "weight_bits": {"nvfp4": 4, "mxfp4": 4, "fp8": 8}[args.format],
        "block_size": {"nvfp4": BLOCK, "mxfp4": MX_BLOCK,
                       "fp8": None}[args.format],
        "packing": "4-bit formats pack two codes per uint8, low nibble first, "
                   "along the reduction axis; fp8 stores e4m3 directly",
        "dequantize": "w = code_value * scale_e4m3 / global_scale, then "
                      "reshape to <name>.shape and movedim(-1, <name>.red_axis)",
        "per_tensor_entries": ["<name>.packed uint8", "<name>.scale e4m3",
                               "<name>.global_scale fp32",
                               "<name>.shape int32", "<name>.red_axis int32"],
        "reduction_axis_note": "-2 for 3-D MoE expert stacks (torch.bmm sums "
                               "over dim -2), -1 for 2-D nn.Linear weights",
        "ignored_modules": ["embed", "lm_head", "*.gate", "*norm*",
                            "1-D tensors"],
        "quantized_parameters": nq,
        "kept_parameters": nk,
    }
    for extra in ("model_config.json", "tokenizer.json",
                  "tokenizer_config.json", "special_tokens_map.json",
                  "generation_config.json", "chat_template.jinja"):
        p = src / extra
        if p.exists():
            (out / extra).write_bytes(p.read_bytes())
    (out / "quantization_config.json").write_text(
        json.dumps(qcfg, indent=2), encoding="utf-8")

    size_src = (src / "model.safetensors").stat().st_size / 2**30
    size_out = (out / "model.safetensors").stat().st_size / 2**30
    print(f"quantised {len(quantised)} tensors ({nq/1e9:.3f}B params)")
    for f, n in by_fmt.most_common():
        print(f"    {f:<8}{n/1e9:>8.3f}B")
    print(f"kept      {len(kept)} tensors ({nk/1e9:.3f}B params) in high precision")
    for name, n, why in sorted(kept, key=lambda k: -k[1])[:8]:
        print(f"    {name:<44}{n/1e6:>9.1f}M  {why}")
    print(f"size      {size_src:.2f} GiB -> {size_out:.2f} GiB "
          f"({size_src/max(size_out,1e-9):.2f}x)")
    if args.verify:
        print(f"weight RMS relative error: "
              f"{(err_num/max(err_den,1e-12))**0.5:.5f}")
        worst.sort(reverse=True)
        print("  worst tensors:")
        for rel, name, f in worst[:6]:
            print(f"    {rel:.5f}  [{f}] {name}")


if __name__ == "__main__":
    main()
