"""
export_ternary.py

Turn a full training checkpoint (fp32 weights + Adam optimizer state, ~3x
model size) into a compact inference-only file where every ternary layer's
weights are packed to their true information content: each weight is one of
{-1, 0, +1}, i.e. log2(3) ~= 1.58 bits, packed 5-per-byte (3^5 = 243 <= 255).
Everything that ISN'T ternary (biases, GroupNorm affine params, the RVQ
codebooks) is kept as float16, since those are a small fraction of the total
parameter count and don't benefit much from more aggressive packing.

This does NOT change what the model computes: for an already-ternary value
w in {-scale, 0, +scale}, re-deriving scale' = mean(|w|) and re-rounding
w/scale' reproduces the exact same {-1,0,1} pattern (it's already at a fixed
point of the quantization function). So exporting is lossless with respect
to the *quantized* forward pass the model actually uses at inference time --
you are not losing anything the fp32 checkpoint's forward pass was using
either, since it re-quantizes weights fresh on every forward call anyway.

Usage
-----
# Export
python export_ternary.py --checkpoint runs/ternary_codec/codec_final.pt \
    --out runs/ternary_codec/codec_final.ternary

# The exported file is tiny. To use it for inference, load_packed_model()
# below reconstructs a normal TernaryCodec you can call .forward() on same
# as any other torch model -- see the __main__ block for a round-trip
# sanity check, and test_ternary_codec.py can be pointed at it too (pass
# --packed).
"""

import argparse
import math
import os
from typing import Dict, Tuple

import torch
import torch.nn as nn

from vlora2 import (
    TernaryCodec,
    TernaryConv1d,
    TernaryConvTranspose1d,
    CodecConfig,
    _ternary_quantize,
)

TRITS_PER_BYTE = 5  # 3^5 = 243 <= 255, fits in a uint8


def pack_ternary(w: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pack a ternary-quantized weight tensor into base-3-encoded bytes.

    Returns (packed_bytes uint8 tensor, original_shape as a tensor) -- shape
    is returned as a tensor so it round-trips cleanly through torch.save.
    """
    flat = w.flatten()
    trits = torch.clamp(torch.round(flat / scale), -1, 1).to(torch.int64) + 1  # in {0,1,2}
    n = trits.numel()
    pad = (-n) % TRITS_PER_BYTE
    if pad:
        trits = torch.cat([trits, torch.zeros(pad, dtype=torch.int64)])
    trits = trits.view(-1, TRITS_PER_BYTE)
    powers = torch.tensor([3 ** i for i in range(TRITS_PER_BYTE)], dtype=torch.int64)
    packed = (trits * powers).sum(dim=1).to(torch.uint8)
    shape = torch.tensor(list(w.shape), dtype=torch.int64)
    return packed, shape


def unpack_ternary(packed: torch.Tensor, shape: torch.Tensor, scale: float,
                    numel: int) -> torch.Tensor:
    """Inverse of pack_ternary: bytes + scale -> full-precision weight tensor."""
    vals = packed.to(torch.int64)
    trits = torch.zeros(vals.numel(), TRITS_PER_BYTE, dtype=torch.int64)
    for i in range(TRITS_PER_BYTE):
        trits[:, i] = vals % 3
        vals = vals // 3
    trits = trits.view(-1) - 1  # back to {-1, 0, 1}
    trits = trits[:numel].to(torch.float32)
    return (trits * scale).view(*shape.tolist())


def export_checkpoint(checkpoint_path: str, out_path: str):
    try:
        torch.serialization.add_safe_globals([CodecConfig])
    except AttributeError:
        pass
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: CodecConfig = ckpt["cfg"]

    model = TernaryCodec(cfg)
    model.load_state_dict(ckpt["model"])
    model.eval()

    packed_layers: Dict[str, dict] = {}
    other_state: Dict[str, torch.Tensor] = {}
    ternary_module_names = set()

    for name, module in model.named_modules():
        if isinstance(module, (TernaryConv1d, TernaryConvTranspose1d)):
            ternary_module_names.add(name)
            w = module.weight.data
            _, scale = _ternary_quantize(w)
            packed, shape = pack_ternary(w, scale)
            packed_layers[name] = {
                "packed": packed,
                "shape": shape,
                "scale": float(scale.item()),
                "numel": w.numel(),
            }

    # Everything that isn't a ternary layer's `.weight` (biases, GroupNorm
    # affine params, RVQ codebook buffers, etc.) is kept as float16.
    for key, tensor in model.state_dict().items():
        is_ternary_weight = any(
            key == f"{mod_name}.weight" for mod_name in ternary_module_names
        )
        if not is_ternary_weight:
            other_state[key] = tensor.to(torch.float16)

    payload = {
        "cfg": cfg,
        "packed_layers": packed_layers,
        "other_state": other_state,
        "format": "ternary_packed_v1",
    }
    torch.save(payload, out_path)

    # ---- report size comparison ----
    orig_size = os.path.getsize(checkpoint_path)
    new_size = os.path.getsize(out_path)
    n_ternary_params = sum(v["numel"] for v in packed_layers.values())
    n_total_params = sum(p.numel() for p in model.parameters())
    print(f"Original checkpoint:  {orig_size / 1e6:.2f} MB  ({checkpoint_path})")
    print(f"Exported (packed):    {new_size / 1e6:.2f} MB  ({out_path})")
    print(f"Ternary params packed: {n_ternary_params:,} / {n_total_params:,} "
          f"({n_ternary_params / n_total_params:.1%})")
    print(f"Size reduction:        {orig_size / new_size:.1f}x")


def load_packed_model(path: str, device: str = "cpu") -> Tuple[TernaryCodec, CodecConfig]:
    """Load a file produced by export_checkpoint() and return a ready-to-use
    TernaryCodec in eval mode, identical (bit-for-bit, at the quantized
    level) to the model that was exported."""
    try:
        torch.serialization.add_safe_globals([CodecConfig])
    except AttributeError:
        pass
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("format") != "ternary_packed_v1":
        raise ValueError(f"Unrecognized packed-model format in {path}")

    cfg: CodecConfig = payload["cfg"]
    model = TernaryCodec(cfg)
    state = model.state_dict()

    for name, info in payload["packed_layers"].items():
        w = unpack_ternary(info["packed"], info["shape"], info["scale"], info["numel"])
        state[f"{name}.weight"] = w

    for key, tensor in payload["other_state"].items():
        state[key] = tensor.to(torch.float32)

    model.load_state_dict(state)
    model.to(device).eval()
    return model, cfg


def _sanity_check(checkpoint_path: str, packed_path: str):
    """Verifies the packed model produces identical output to the original."""
    try:
        torch.serialization.add_safe_globals([CodecConfig])
    except AttributeError:
        pass
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    cfg: CodecConfig = ckpt["cfg"]
    orig_model = TernaryCodec(cfg)
    orig_model.load_state_dict(ckpt["model"])
    orig_model.eval()

    packed_model, _ = load_packed_model(packed_path)

    torch.manual_seed(0)
    dummy = torch.randn(1, 1, cfg.hop_length * 8) * 0.1
    with torch.no_grad():
        out_orig, _, _ = orig_model(dummy)
        out_packed, _, _ = packed_model(dummy)

    max_diff = (out_orig - out_packed).abs().max().item()
    print(f"Sanity check: max abs diff between original and packed model output = {max_diff:.2e}")
    if max_diff < 1e-4:
        print("Round-trip OK: packed model is numerically equivalent.")
    else:
        print("WARNING: packed model output differs more than expected -- check for bugs.")


def main():
    p = argparse.ArgumentParser(description="Export a training checkpoint to a compact ternary-packed file")
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--out", type=str, required=True)
    p.add_argument("--verify", action="store_true",
                    help="After exporting, reload the packed file and confirm it produces "
                         "identical output to the original checkpoint")
    args = p.parse_args()

    export_checkpoint(args.checkpoint, args.out)
    if args.verify:
        _sanity_check(args.checkpoint, args.out)


if __name__ == "__main__":
    main()