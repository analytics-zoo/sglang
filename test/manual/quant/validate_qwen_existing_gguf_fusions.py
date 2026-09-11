#!/usr/bin/env python3
"""Numerically validate the existing Qwen GGUF XPU decode fusions.

This is a standalone diagnostic; it neither loads a model nor starts a server.
It uses the production GGUF repack/dequant helpers to build canonical packed
Q4_K, Q5_K, and Q6_K representations, then compares the registered XPU ops
with a dense FP32 reference that preserves the kernels' FP16 rounding points.

Run only on the intended TP=2 pair, for example:
  ZE_AFFINITY_MASK=6,7 python test/manual/quant/validate_qwen_existing_gguf_fusions.py

The three operator names and signatures are read from the installed extension:
  esimd_resadd_norm_gemv_kq
  esimd_resadd_norm_gemv_q4k_silu
  esimd_norm_gemv_q5k
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Any

import torch
import torch.nn.functional as F

K_HIDDEN = 5120
K_GDN = 3072
V_GDN = 128
N_Q4 = 64
N_Q6 = 64
N_BA = 32
N_Q5 = 64
I_MLP = 64
EPS = 1.0e-6
SENTINEL = -123.0


def _require_xpu() -> None:
    if os.environ.get("ZE_AFFINITY_MASK") != "6,7":
        raise RuntimeError(
            "This validator is reserved for the TP=2 validation pair; "
            "set ZE_AFFINITY_MASK=6,7."
        )
    if not torch.xpu.is_available():
        raise RuntimeError("torch.xpu is unavailable; this validator must run on XPU.")


def _finite_raw(
    rows: int, k: int, kind: str, generator: torch.Generator
) -> torch.Tensor:
    """Build valid GGUF block bytes, then rely on production repack/dequant.

    Random raw bytes exercise the quant payload and scale packing.  The only
    fields constrained here are the FP16 block scales: random bit patterns can
    encode NaN/Inf, which would make a numerical test meaningless.
    """
    specs = {"q4": (144, 256), "q5": (176, 256), "q6": (210, 256)}
    block_bytes, block_k = specs[kind]
    if k % block_k:
        raise ValueError(f"{kind}: K={k} is not block aligned")
    raw = torch.randint(
        0,
        256,
        (rows, k // block_k, block_bytes),
        dtype=torch.uint8,
        generator=generator,
    )
    if kind in ("q4", "q5"):
        # GGML block_q{4,5}_K begins with dall,dmin.
        # Small block scales keep random 6/8-bit subscales in a
        # weight range suited to FP16 activation/error checks.
        dm = torch.tensor([2.0**-15, 2.0**-16], dtype=torch.float16).view(torch.uint8)
        raw[:, :, :4] = dm
    else:
        # GGML block_q6_K ends with its FP16 scale.
        d = torch.tensor([2.0**-15], dtype=torch.float16).view(torch.uint8)
        raw[:, :, 208:210] = d
    return raw.reshape(rows, -1).contiguous()


def _reps(generator: torch.Generator) -> dict[str, tuple[torch.Tensor, ...]]:
    # Importing this module is intentional: these are the exact production
    # canonicalizers and dense dequantizers, not a hand-written mirror.
    from sglang.srt.layers.quantization.gguf import (
        _xpu_dequant_q4_k,
        _xpu_dequant_q5_k,
        _xpu_dequant_q6_k,
        _xpu_repack_q4_k,
        _xpu_repack_q5_k,
        _xpu_repack_q6_k,
    )

    q4 = _xpu_repack_q4_k(_finite_raw(N_Q4, K_HIDDEN, "q4", generator))
    q6 = _xpu_repack_q6_k(_finite_raw(N_Q6, K_HIDDEN, "q6", generator))
    q4_mlp = _xpu_repack_q4_k(_finite_raw(2 * I_MLP, K_HIDDEN, "q4", generator))
    q5 = _xpu_repack_q5_k(_finite_raw(N_Q5, K_GDN, "q5", generator))

    # Dequantization is deliberately performed by the same canonical helpers
    # used by the prefill/reference path. Keep it FP32 for accumulation.
    return {
        "q4": (*q4, _xpu_dequant_q4_k(*q4, torch.float32)),
        "q6": (*q6, _xpu_dequant_q6_k(*q6, torch.float32)),
        "q4_mlp": (*q4_mlp, _xpu_dequant_q4_k(*q4_mlp, torch.float32)),
        "q5": (*q5, _xpu_dequant_q5_k(*q5, torch.float32)),
    }


def _to_xpu(rep: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(x.to("xpu") for x in rep)


def _gemma_norm(
    h: torch.Tensor, residual: torch.Tensor, nw: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference the exact FP16 residual and normed-activation stores."""
    nr = (h + residual).to(torch.float16)
    rstd = torch.rsqrt(nr.float().square().mean(dim=1, keepdim=True) + EPS)
    xn = (nr.float() * rstd * nw.float()).to(torch.float16)
    return nr, xn


def _max_abs(actual: torch.Tensor, expected: torch.Tensor) -> float:
    if not (torch.isfinite(actual).all() and torch.isfinite(expected).all()):
        raise AssertionError("non-finite result or reference")
    return float((actual.float() - expected.float()).abs().max().cpu())


def _check(
    name: str,
    actual: torch.Tensor,
    expected: torch.Tensor,
    atol: float,
    out: dict[str, Any],
) -> None:
    err = _max_abs(actual, expected)
    out[name] = {"max_abs": err, "atol": atol, "shape": list(actual.shape)}
    if err > atol:
        raise AssertionError(
            f"{name}: max_abs={err:.8g} exceeds requested atol={atol:g}"
        )


def _kq_case(
    m: int,
    packed: dict[str, tuple[torch.Tensor, ...]],
    generator: torch.Generator,
    atol: float,
) -> dict[str, Any]:
    op = torch.ops.custom_esimd_kernels_sglang.esimd_resadd_norm_gemv_kq
    q4w, q4sc, q4mn, q4dense = _to_xpu(packed["q4"])
    q6ql, q6qh, q6sc, q6dense = _to_xpu(packed["q6"])
    h = (torch.randn((m, K_HIDDEN), generator=generator) * 0.02).to(
        device="xpu", dtype=torch.float16
    )
    residual = (torch.randn((m, K_HIDDEN), generator=generator) * 0.02).to(
        device="xpu", dtype=torch.float16
    )
    residual_before = residual.clone()
    # Qwen's GemmaRMSNorm uses weight+1 for this attention-input fusion.
    nw = (1.0 + torch.randn(K_HIDDEN, generator=generator) * 0.02).to(
        device="xpu", dtype=torch.float16
    )
    ba = (torch.randn((N_BA, K_HIDDEN), generator=generator) * 0.002).to(
        device="xpu", dtype=torch.float16
    )
    nr = torch.empty_like(h)  # Must be distinct from residual.
    xn = torch.empty_like(h)
    # Exercise the production mixed-Q4/Q6 offsets in one output buffer.
    off4, off6 = 3, 3 + N_Q4 + 5
    out = torch.full((m, off6 + N_Q6 + 2), SENTINEL, device="xpu", dtype=torch.float16)
    oba = torch.empty((m, N_BA), device="xpu", dtype=torch.float16)
    empty = torch.empty(0, device="xpu", dtype=torch.float16)
    op(
        h,
        residual,
        nw,
        EPS,
        nr,
        xn,
        q4w,
        q4sc,
        q4mn,
        out,
        off4,
        q6ql,
        q6qh,
        q6sc,
        out,
        off6,
        ba,
        oba,
    )
    torch.xpu.synchronize()
    if not torch.equal(residual, residual_before):
        raise AssertionError("kq: kernel modified the input residual")
    nr_ref, xn_ref = _gemma_norm(h, residual_before, nw)
    q4_ref = (xn_ref.float() @ q4dense.t()).to(torch.float16)
    q6_ref = (xn_ref.float() @ q6dense.t()).to(torch.float16)
    ba_ref = (xn_ref.float() @ ba.float().t()).to(torch.float16)
    result: dict[str, Any] = {}
    _check("new_residual", nr, nr_ref, atol, result)
    _check("normed", xn, xn_ref, atol, result)
    _check("q4_offset", out[:, off4 : off4 + N_Q4], q4_ref, atol, result)
    _check("q6_offset", out[:, off6 : off6 + N_Q6], q6_ref, atol, result)
    _check("fp16_ba", oba, ba_ref, atol, result)
    untouched = torch.cat(
        (out[:, :off4], out[:, off4 + N_Q4 : off6], out[:, off6 + N_Q6 :]), 1
    )
    if not torch.equal(untouched, torch.full_like(untouched, SENTINEL)):
        raise AssertionError("kq: mixed-output gap or guard columns were written")
    return result


def _mlp_case(
    m: int,
    packed: dict[str, tuple[torch.Tensor, ...]],
    generator: torch.Generator,
    atol: float,
) -> dict[str, Any]:
    op = torch.ops.custom_esimd_kernels_sglang.esimd_resadd_norm_gemv_q4k_silu
    q4w, q4sc, q4mn, q4dense = _to_xpu(packed["q4_mlp"])
    h = (torch.randn((m, K_HIDDEN), generator=generator) * 0.02).to(
        device="xpu", dtype=torch.float16
    )
    residual = (torch.randn((m, K_HIDDEN), generator=generator) * 0.02).to(
        device="xpu", dtype=torch.float16
    )
    residual_before = residual.clone()
    nw = (1.0 + torch.randn(K_HIDDEN, generator=generator) * 0.02).to(
        device="xpu", dtype=torch.float16
    )
    nr = torch.empty_like(h)
    y = torch.empty((m, I_MLP), device="xpu", dtype=torch.float16)
    op(h, residual, nw, EPS, nr, q4w, q4sc, q4mn, y)
    torch.xpu.synchronize()
    if not torch.equal(residual, residual_before):
        raise AssertionError("mlp: kernel modified the input residual")
    nr_ref, xn_ref = _gemma_norm(h, residual_before, nw)
    # The kernel rounds both gate/up GEMVs to FP16 before FP32 SiLU/multiply.
    gate_up = (xn_ref.float() @ q4dense.t()).to(torch.float16)
    y_ref = (F.silu(gate_up[:, :I_MLP].float()) * gate_up[:, I_MLP:].float()).to(
        torch.float16
    )
    result: dict[str, Any] = {}
    _check("new_residual", nr, nr_ref, atol, result)
    _check("silu_mul", y, y_ref, atol, result)
    return result


def _q5_case(
    packed: dict[str, tuple[torch.Tensor, ...]],
    generator: torch.Generator,
    atol: float,
) -> dict[str, Any]:
    # qwen3_5.py registers/uses this as esimd_norm_gemv_q5k (not the older
    # informal name esimd_rms_norm_gated_gemv_q5k).
    op = torch.ops.custom_esimd_kernels_sglang.esimd_norm_gemv_q5k
    ql, qh, sc, mn, dense = _to_xpu(packed["q5"])
    hv = K_GDN // V_GDN
    x = (torch.randn((1, K_GDN), generator=generator) * 0.02).to(
        device="xpu", dtype=torch.float16
    )
    z = (torch.randn((1, K_GDN), generator=generator) * 0.02).to(
        device="xpu", dtype=torch.float16
    )
    nw = (1.0 + torch.randn(V_GDN, generator=generator) * 0.02).to(
        device="xpu", dtype=torch.float16
    )
    y = torch.empty((1, N_Q5), device="xpu", dtype=torch.float16)
    op(x, z, nw, ql, qh, sc, mn, y, V_GDN, EPS)
    torch.xpu.synchronize()
    # RMSNormGated is per value head and writes its FP16 intermediate to SLM.
    xv = x.reshape(1, hv, V_GDN).float()
    zv = z.reshape(1, hv, V_GDN).float()
    inv = torch.rsqrt(xv.square().mean(dim=-1, keepdim=True) + EPS)
    normed = (xv * inv * nw.float().view(1, 1, V_GDN) * F.silu(zv)).to(torch.float16)
    y_ref = (normed.reshape(1, K_GDN).float() @ dense.t()).to(torch.float16)
    result: dict[str, Any] = {}
    _check("rms_norm_gated_q5k", y, y_ref, atol, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--atol",
        type=float,
        default=0.01,
        help="strict max absolute error threshold (default: 0.01)",
    )
    args = parser.parse_args()
    _require_xpu()
    try:
        import custom_esimd_kernels_sglang  # noqa: F401 - registers torch.ops
    except Exception as exc:
        raise RuntimeError("custom_esimd_kernels_sglang import failed") from exc
    required = (
        "esimd_resadd_norm_gemv_kq",
        "esimd_resadd_norm_gemv_q4k_silu",
        "esimd_norm_gemv_q5k",
    )
    namespace = torch.ops.custom_esimd_kernels_sglang
    missing = [name for name in required if not hasattr(namespace, name)]
    if missing:
        raise RuntimeError(f"missing required registered op(s): {missing}")

    generator = torch.Generator(device="cpu").manual_seed(20260910)
    packed = _reps(generator)
    report: dict[str, Any] = {
        "seed": 20260910,
        "ze_affinity_mask": os.environ["ZE_AFFINITY_MASK"],
        "atol": args.atol,
        "shapes": {"kq_mlp_k": K_HIDDEN, "q5_k": K_GDN, "q5_v": V_GDN},
        "kq": {},
        "mlp_silu": {},
        "q5_norm_gemv": {},
    }
    # KQ supports the regular per-token fallback beyond M=4; M=2/4 uses the
    # M-tiled implementation. MLP is intentionally limited to the production
    # default M<=4; Q5K production defaults to M=1.
    for m in (1, 2, 4, 8, 16):
        report["kq"][str(m)] = _kq_case(m, packed, generator, args.atol)
    for m in (1, 2, 4):
        report["mlp_silu"][str(m)] = _mlp_case(m, packed, generator, args.atol)
    report["q5_norm_gemv"]["1"] = _q5_case(packed, generator, args.atol)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
