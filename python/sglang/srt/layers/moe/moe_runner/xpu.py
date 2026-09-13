"""Intel XPU MoE Runner — ESIMD kernels first, triton fallback.

The XPU backend tries device-specific ESIMD kernels for FP8 silu MoE decode
(small T), and falls back to the triton runner for everything else:
  - bf16 / fp16 weights (no ESIMD kernel)
  - fp8 with T>8 (ESIMD kernel is decode-optimized)
  - non-silu activations (ESIMD kernel is silu-only)

Triton remains the last tier so coverage is complete.
"""
from __future__ import annotations

import functools
import logging
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

logger = logging.getLogger(__name__)

# Prefill (large-T) ESIMD MoE gate. Mirrors downstream fused_moe's
# `_ESIMD_MOE_PREFILL`. On this rebase env the upstream sgl-kernel refuses fp8
# (`assert use_fp8_w8a8 is False`), so the M-tiled DPAS fp8 prefill kernel from
# custom-esimd-kernels is the ONLY fp8-capable native MoE path for T>8 — the
# alternative is the Triton fused_moe_kernel, which needs 325 KB scratch (> BMG's
# 262 KB PTSS limit) and fails to compile. Routed-only, which is correct here:
# qwen2_moe adds the shared expert separately (`final_hidden_states +=
# shared_output`), so this kernel need only produce the routed contribution.
_ESIMD_MOE_PREFILL = os.environ.get("SGL_XPU_ESIMD_MOE_PREFILL", "0") == "1"

# Decode-tier gate (M=1 DPAS silu kernel, moe_forward_full_silu_routed). Default
# on. Set SGL_XPU_ESIMD_MOE_DECODE=0 to route decode (T<=8) through the M-tiled
# prefill kernel instead — the correctness bisect that isolates whether the
# decode kernel or the prefill kernel is the source of numerically-wrong output.
_ESIMD_MOE_DECODE = os.environ.get("SGL_XPU_ESIMD_MOE_DECODE", "1") == "1"

# --- prefill-kernel white-box diagnostic (correctness bisect) -----------------
# SGL_XPU_MOE_PREFILL_COMPARE=1: for SMALL prefill batches (<= _CMP_MAX_M, where
# the Triton small-M config compiles under BMG's 262 KB PTSS so there is no
# scratch crash), recompute the SAME MoE through the Triton fp8 fallback and log
# the max/mean abs error vs the ESIMD prefill kernel output. A large error means
# moe_prefill_full_fp8 (as fed by this call site) is the numerically-wrong path;
# a ~0 error exonerates the kernel and points at core code.
# SGL_XPU_MOE_PREFILL_USE_TRITON=1: additionally RETURN the Triton output for
# those small batches, so a full probe can confirm correctness returns when the
# ESIMD prefill kernel is bypassed. Large batches still use ESIMD (no crash).
_MOE_PREFILL_COMPARE = os.environ.get("SGL_XPU_MOE_PREFILL_COMPARE", "0") == "1"
_MOE_PREFILL_USE_TRITON = os.environ.get("SGL_XPU_MOE_PREFILL_USE_TRITON", "0") == "1"
_CMP_MAX_M = int(os.environ.get("SGL_XPU_MOE_PREFILL_CMP_MAX_M", "256"))

# WIRE P8 diag: one-shot log of the tier-2 (ESIMD fp8 MoE prefill) decision
# inputs. The path is fully wired and the gate is on, yet serve logs still show
# the Triton fused_moe fallback firing (see reports/pipeline-architecture-diff.md
# gap B) with NO kernel-failure warning -> a guard is declining silently. This
# logs the guard state on the first prefill-tier call so the log names the cause.
_P8_DIAG_DONE = False

# WIRE decode-drill: one-shot per (M-bucket, tier) log of WHICH MoE runner tier
# actually executes, so the orig-vs-rebase wiring table is proven from logs, not
# assumed. M<=8 is the decode regime; the fp8 variant (e4m3/e5m2) names the exact
# decode kernel (orig ships e5m2, rebase locked e4m3 -> different .so entry).
_MOE_TIER_DIAG_SEEN = set()


def _moe_tier_diag(m, tier, quant_info):
    try:
        wdt = getattr(getattr(quant_info, "w13_weight", None), "dtype", None)
        variant = "e5m2" if wdt == torch.float8_e5m2 else ("e4m3" if wdt == torch.float8_e4m3fn else str(wdt))
        regime = "decode" if m <= 8 else "prefill"
        key = (regime, tier, variant)
        if key not in _MOE_TIER_DIAG_SEEN:
            _MOE_TIER_DIAG_SEEN.add(key)
            logger.warning("[MOE-tier-diag] regime=%s M=%d TIER=%s wdtype=%s fp8=%s",
                           regime, m, tier, variant, getattr(quant_info, "use_fp8_w8a8", None))
    except Exception:
        pass


@functools.cache
def _load_esimd_moe_prefill_op():
    """Lazy-load the M-tiled FP8 MoE prefill op (moe_prefill_full_fp8)."""
    try:
        from custom_esimd_kernels_sglang import moe_fp8_prefill_ops  # noqa: F401

        return torch.ops.moe_fp8_prefill_ops.moe_prefill_full_fp8
    except Exception:
        return None

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


@functools.cache
def _load_esimd_moe_silu_op(fp8_variant: str = "e4m3"):
    """Load the ESIMD fp8 silu MoE DECODE kernel for the given fp8 variant.

    The kernel is variant-specific — reading e5m2 weights through the e4m3
    kernel (or vice versa) mis-decodes the mantissa/exponent split and produces
    numerically-wrong-but-fluent output. Downstream selects the op by the WEIGHT
    dtype (`fp8_variant = "e5m2" if w13.dtype == float8_e5m2 else "e4m3"`):
        e5m2 -> moe_forward_full_silu_routed_e5m2
        e4m3 -> moe_forward_full_silu_routed_sglang
    The bare `moe_forward_full_silu_routed` is NOT what downstream fires; binding
    it was a porting bug that corrupted every e5m2 decode step.
    """
    try:
        from custom_esimd_kernels_sglang import moe_ops as _moe_mod
        if fp8_variant == "e5m2":
            return _moe_mod.moe_forward_full_silu_routed_e5m2
        return _moe_mod.moe_forward_full_silu_routed_sglang
    except (ImportError, AttributeError):
        return None


@dataclass
class XpuMoeQuantInfo(MoeQuantInfo):
    """Quant info for XPU MoE — carries both XPU-specific and triton-fallback fields."""
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    b13: Optional[torch.Tensor] = None
    b2: Optional[torch.Tensor] = None
    use_fp8_w8a8: bool = False
    use_int8_w8a8: bool = False
    use_int8_w8a16: bool = False
    use_int4_w4a16: bool = False
    per_channel_quant: bool = False
    w13_scale: Optional[torch.Tensor] = None
    w2_scale: Optional[torch.Tensor] = None
    w13_zp: Optional[torch.Tensor] = None
    w2_zp: Optional[torch.Tensor] = None
    a13_scale: Optional[torch.Tensor] = None
    a2_scale: Optional[torch.Tensor] = None
    block_shape: Optional[List[int]] = None
    block_quant: bool = False
    fuse_swiglu_interleaved: bool = False


@dataclass
class XpuRunnerInput(RunnerInput):
    """Input for XPU MoE runner — carries data needed for both ESIMD and triton paths."""
    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.INTEL_XPU


@dataclass
class XpuRunnerOutput(RunnerOutput):
    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.INTEL_XPU


class XpuRunnerCore(MoeRunnerCore):
    """XPU MoE runner — ESIMD kernel for fp8 silu decode, triton fallback otherwise."""

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)
        # The decode op is variant-specific (e5m2 vs e4m3) and the weight dtype
        # is only known per-call, so it is loaded lazily in _try_esimd_fp8_silu.

    def _try_esimd_fp8_silu(
        self,
        runner_input: XpuRunnerInput,
        quant_info: XpuMoeQuantInfo,
    ) -> Optional[torch.Tensor]:
        """Try ESIMD fp8 silu MoE kernel; return None to fall back to triton."""
        if not _ESIMD_MOE_DECODE:
            return None
        if not quant_info.use_fp8_w8a8:
            return None
        if runner_input.hidden_states.shape[0] > 8:
            return None
        if self.config.activation != "silu":
            return None
        if self.config.gemm1_alpha is not None:
            return None
        if self.config.gemm1_clamp_limit is not None:
            return None
        if self.config.swiglu_limit is not None:
            return None
        if quant_info.b13 is not None or quant_info.b2 is not None:
            return None

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        _fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        if w13.dtype not in _fp8_dtypes or w2.dtype not in _fp8_dtypes:
            return None
        if w13.dtype != w2.dtype:
            return None

        # Select the decode op by WEIGHT dtype, exactly as downstream does. The
        # e5m2 weights this model ships (SGLANG_FP8_DTYPE=e5m2) MUST go through
        # the _e5m2 kernel; the bare/e4m3 kernel mis-decodes them.
        fp8_variant = "e5m2" if w13.dtype == torch.float8_e5m2 else "e4m3"
        esimd_op = _load_esimd_moe_silu_op(fp8_variant)
        if esimd_op is None:
            return None

        w13_scale = quant_info.w13_scale
        w2_scale = quant_info.w2_scale
        if w13_scale is None or w2_scale is None:
            return None

        def _per_expert_pt_scale(scale: torch.Tensor) -> torch.Tensor:
            cached = getattr(scale, "_xpu_pt_per_expert", None)
            if cached is not None:
                return cached
            s = scale.to(torch.float32).reshape(scale.shape[0], -1).mean(dim=-1)
            # e4m3-lock guard (decode twin of _f32c's): a 0/NaN/inf per-expert
            # scale (empty/padded expert -> amax 0, or an upstream requantize
            # that overflowed the e4m3 448 range) would feed the DPAS dequant a
            # poison multiplier -> NaN output -> garbage token ids -> the ungated
            # (values>=0) invariant assert. No-op on valid positive scales.
            s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-12)
            s = s.contiguous()
            try:
                scale._xpu_pt_per_expert = s
            except Exception:
                pass
            return s

        s13 = _per_expert_pt_scale(w13_scale)
        s2 = _per_expert_pt_scale(w2_scale)

        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids
        if topk_weights.dtype != torch.float16:
            topk_weights = topk_weights.to(torch.float16)
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        x = runner_input.hidden_states
        orig_dtype = x.dtype
        x_in = x if x.dtype == torch.float16 else x.to(torch.float16)

        top_k = topk_ids.shape[-1]
        n_routed = w13.shape[0]

        try:
            out = esimd_op(x_in, topk_weights, topk_ids, w13, s13, w2, s2, top_k, n_routed)
        except Exception:
            return None

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out

    def _run_esimd_prefill(
        self,
        runner_input: XpuRunnerInput,
        quant_info: XpuMoeQuantInfo,
    ) -> Optional[torch.Tensor]:
        """Large-T (prefill) fp8 MoE via the M-tiled DPAS kernel moe_prefill_full_fp8.

        The decode kernel (_try_esimd_fp8_silu) is DPAS M=1, gated T<=8; prefill
        would be GEMV-bound there. This routes T>8 fp8 batches to the M-tiled
        (MAX_M=32) DPAS GEMM instead — the ONLY fp8-capable native MoE path on
        this rebase env, since upstream sgl-kernel refuses fp8 and the Triton
        fused_moe_kernel needs 325 KB scratch (> BMG 262 KB PTSS) and won't
        compile. Ported from downstream `_maybe_esimd_moe_silu_prefill`.

        Routed-only, which is CORRECT here: qwen2_moe adds the shared expert
        separately (`final_hidden_states += shared_output` in
        `_forward_router_experts`), so this kernel need produce only the routed
        contribution. (Downstream's copy is gated OFF because its call site
        fuses the shared expert; this rebase's does not.)

        Returns the routed MoE output [T, hidden], or None to fall through.
        """
        global _P8_DIAG_DONE
        if not _P8_DIAG_DONE:
            _P8_DIAG_DONE = True
            try:
                _w13 = getattr(quant_info, "w13_weight", None)
                _w2 = getattr(quant_info, "w2_weight", None)
                logger.warning(
                    "[P8-diag] esimd_prefill entry: T=%s gate=%s fp8=%s act=%s "
                    "gemm1_alpha=%s clamp=%s swiglu=%s bias=%s w13.dtype=%s "
                    "w2.dtype=%s w13_scale=%s w2_scale=%s block_shape=%s op=%s",
                    tuple(getattr(runner_input.hidden_states, "shape", ()) or ()),
                    _ESIMD_MOE_PREFILL,
                    quant_info.use_fp8_w8a8,
                    self.config.activation,
                    self.config.gemm1_alpha,
                    self.config.gemm1_clamp_limit,
                    self.config.swiglu_limit,
                    (quant_info.b13 is not None or quant_info.b2 is not None),
                    getattr(_w13, "dtype", None),
                    getattr(_w2, "dtype", None),
                    quant_info.w13_scale is not None,
                    quant_info.w2_scale is not None,
                    quant_info.block_shape,
                    _load_esimd_moe_prefill_op() is not None,
                )
            except Exception as _e:
                logger.warning("[P8-diag] failed: %s", _e)

        if not _ESIMD_MOE_PREFILL:
            return None
        if not quant_info.use_fp8_w8a8:
            return None
        if self.config.activation != "silu":
            return None
        if self.config.gemm1_alpha is not None:
            return None
        if self.config.gemm1_clamp_limit is not None:
            return None
        if self.config.swiglu_limit is not None:
            return None
        if quant_info.b13 is not None or quant_info.b2 is not None:
            return None

        w13 = quant_info.w13_weight
        w2 = quant_info.w2_weight
        _fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
        if w13.dtype not in _fp8_dtypes or w2.dtype not in _fp8_dtypes:
            return None
        if w13.dtype != w2.dtype:
            return None
        if quant_info.w13_scale is None or quant_info.w2_scale is None:
            return None
        # Only online per-expert per-tensor scales (block_shape is None). A
        # block-quant checkpoint needs a block-scale kernel this op does not have.
        if quant_info.block_shape is not None:
            return None

        op = _load_esimd_moe_prefill_op()
        if op is None:
            return None

        # The prefill kernel keeps gate/up scales SEPARATE: w13_scale is [E,2]
        # (gate=w1, up=w3). Online fp8 here uses requantize_with_max_scale, so w1
        # and w3 share ONE per-expert scale -> w13_scale may arrive as [E]; expand
        # to [E,2] (exact, gate==up). w2_scale is [E]. Averaging w13 to a scalar
        # is too lossy (garbage), so pass the 2-wide tensor straight through.
        def _f32c(scale: torch.Tensor, want_2d: bool) -> Optional[torch.Tensor]:
            key = "_esimd_prefill_w13" if want_2d else "_esimd_prefill_w2"
            cached = getattr(scale, key, None)
            if cached is not None:
                return cached
            s = scale.to(torch.float32).reshape(scale.shape[0], -1)
            if want_2d:
                if s.shape[1] == 1:
                    s = s.expand(s.shape[0], 2)  # [E] -> [E,2], gate==up
                elif s.shape[1] != 2:
                    return None
            else:
                s = s.mean(dim=-1)  # [E]
            # e4m3-lock guard: a 0/NaN/inf per-expert scale (an empty/padded
            # expert whose amax collapsed to 0, or an upstream requantize that
            # overflowed the e4m3 448 range) would feed the DPAS dequant a poison
            # multiplier -> NaN in the routed MoE output -> garbage token ids ->
            # the ungated (values>=0) invariant assert ("input_[0] != 0"). No-op
            # on valid positive scales; an empty expert contributes ~0 anyway
            # (its topk weight is ~0), so clamping its dead scale changes nothing.
            s = torch.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0).clamp_min(1e-12)
            s = s.contiguous()
            try:
                setattr(scale, key, s)
            except Exception:
                pass
            return s

        s13 = _f32c(quant_info.w13_scale, want_2d=True)
        s2 = _f32c(quant_info.w2_scale, want_2d=False)
        if s13 is None or s2 is None:
            return None

        topk_weights = runner_input.topk_weights
        topk_ids = runner_input.topk_ids
        if topk_weights.dtype != torch.float16:
            topk_weights = topk_weights.to(torch.float16)
        if topk_ids.dtype != torch.int32:
            topk_ids = topk_ids.to(torch.int32)

        x = runner_input.hidden_states
        orig_dtype = x.dtype
        x_in = x if x.dtype == torch.float16 else x.to(torch.float16)

        top_k = topk_ids.shape[-1]
        n_routed = w13.shape[0]

        try:
            out = op(x_in, topk_weights, topk_ids, w13, s13, w2, s2, top_k, n_routed)
        except Exception as e:
            logger.warning("ESIMD moe_prefill_full_fp8 failed (%s); falling through", e)
            return None

        if out.dtype != orig_dtype:
            out = out.to(orig_dtype)
        return out

    def _run_triton_fallback(
        self,
        runner_input: XpuRunnerInput,
        quant_info: XpuMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> torch.Tensor:
        """Triton fallback for bf16, large-T fp8, non-silu activations, etc."""
        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
            _fused_moe_kernel_sequence,
        )

        filter_expert = (
            self.config.num_experts is None
            or self.config.num_experts != self.config.num_local_experts
        )

        out = _fused_moe_kernel_sequence(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            runner_input.topk_weights,
            runner_input.topk_ids,
            runner_input.sorted_token_ids,
            runner_input.expert_ids,
            runner_input.num_tokens_post_padded,
            running_state["config"],
            running_state.get("down_config"),
            running_state.get("down_moe_use_tma", False),
            running_state.get("up_moe_use_tma", False),
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            use_int8_w8a8=quant_info.use_int8_w8a8,
            use_int8_w8a16=quant_info.use_int8_w8a16,
            use_int4_w4a16=quant_info.use_int4_w4a16,
            per_channel_quant=quant_info.per_channel_quant,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
            activation=self.config.activation,
            is_gated=self.config.is_gated,
            no_combine=self.config.no_combine,
            inplace=self.config.inplace,
            apply_router_weight_on_input=self.config.apply_router_weight_on_input,
            routed_scaling_factor=self.config.routed_scaling_factor,
            gemm1_alpha=self.config.gemm1_alpha,
            gemm1_limit=self.config.gemm1_clamp_limit,
            filter_expert=filter_expert,
            hooks=hooks,
            swiglu_limit=self.config.swiglu_limit,
            fuse_swiglu_interleaved=quant_info.fuse_swiglu_interleaved,
        )
        return out

    def _run_sgl_native(
        self,
        runner_input: XpuRunnerInput,
        quant_info: XpuMoeQuantInfo,
    ) -> Optional[torch.Tensor]:
        """Native sgl-kernel SYCL MoE — the tier DOWNSTREAM uses for all XPU MoE.

        Downstream `fused_moe.py` dispatches every XPU MoE to
        `sgl_kernel.fused_experts` (`if _use_sgl_xpu: return sgl_fused_experts(...)`)
        and NEVER compiles the Triton `fused_moe_kernel` on XPU. This rebase's
        refactored runner dropped that native tier, leaving only ESIMD-decode
        (T<=8) + a Triton fallback whose prefill config needs 325 KB scratch —
        over BMG's 262 KB PTSS limit — so it fails to compile
        (ZE_RESULT_ERROR_MODULE_BUILD_FAILURE) and crashes the scheduler on the
        first extend batch. Restore the native tier: route XPU MoE (T>8 prefill
        and any decode the ESIMD kernel declined) to the same sgl-kernel
        downstream fires. Signature mirrors downstream's call site exactly.

        Returns the MoE output, or None if sgl_kernel is unavailable OR this is
        an fp8 MoE (the upstream sgl-kernel-xpu build asserts `use_fp8_w8a8 is
        False`); fp8 is handled by the ESIMD prefill/decode kernels above.
        """
        # Upstream sgl-kernel-xpu has no fp8 MoE (`assert use_fp8_w8a8 is False`
        # in sgl_kernel/moe.py). Don't call it for fp8 — that path is the ESIMD
        # kernels'. Only bf16/fp16 MoE reaches the native sgl kernel here.
        if quant_info.use_fp8_w8a8:
            return None
        try:
            from sgl_kernel import fused_experts as sgl_fused_experts
        except Exception as e:  # kernel not built into this env -> let triton try
            logger.warning("sgl_kernel.fused_experts unavailable (%s); using triton", e)
            return None

        return sgl_fused_experts(
            runner_input.hidden_states,
            quant_info.w13_weight,
            quant_info.w2_weight,
            runner_input.topk_weights,
            runner_input.topk_ids,
            b1=quant_info.b13,
            b2=quant_info.b2,
            use_fp8_w8a8=quant_info.use_fp8_w8a8,
            w1_scale=quant_info.w13_scale,
            w2_scale=quant_info.w2_scale,
            w1_zp=quant_info.w13_zp,
            w2_zp=quant_info.w2_zp,
            a1_scale=quant_info.a13_scale,
            a2_scale=quant_info.a2_scale,
            block_shape=quant_info.block_shape,
        )

    def run(
        self,
        runner_input: XpuRunnerInput,
        quant_info: XpuMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> XpuRunnerOutput:
        # Tier 1: ESIMD fp8 silu DECODE (M=1 DPAS, gated T<=8).
        _M = int(runner_input.hidden_states.shape[0])
        out = self._try_esimd_fp8_silu(runner_input, quant_info)
        if out is not None:
            _moe_tier_diag(_M, "tier1_esimd_decode", quant_info)
            return XpuRunnerOutput(hidden_states=out)

        # Tier 2: ESIMD fp8 silu PREFILL (M-tiled DPAS, T>8). The ONLY fp8-capable
        # native MoE path for prefill on this rebase env — upstream sgl-kernel
        # refuses fp8 and the Triton fused_moe_kernel exceeds BMG's PTSS limit.
        out = self._run_esimd_prefill(runner_input, quant_info)
        if out is not None:
            # White-box prefill-kernel diagnostic: on SMALL batches only (Triton
            # small-M config fits BMG PTSS -> no 325 KB scratch crash), compare
            # the ESIMD prefill output against the Triton fp8 fallback, and
            # optionally return Triton instead to confirm correctness.
            if (_MOE_PREFILL_COMPARE or _MOE_PREFILL_USE_TRITON) and \
                    runner_input.hidden_states.shape[0] <= _CMP_MAX_M:
                try:
                    tri = self._run_triton_fallback(
                        runner_input, quant_info, running_state, hooks
                    )
                    if _MOE_PREFILL_COMPARE:
                        a = out.float()
                        b = tri.float()
                        diff = (a - b).abs()
                        denom = b.abs().clamp_min(1e-4)
                        logger.warning(
                            "[MOE_PREFILL_CMP] M=%d esimd_vs_triton max_abs=%.4g "
                            "mean_abs=%.4g max_rel=%.4g mean_rel=%.4g "
                            "esimd|mean|=%.4g triton|mean|=%.4g",
                            runner_input.hidden_states.shape[0],
                            diff.max().item(), diff.mean().item(),
                            (diff / denom).max().item(), (diff / denom).mean().item(),
                            a.abs().mean().item(), b.abs().mean().item(),
                        )
                    if _MOE_PREFILL_USE_TRITON:
                        return XpuRunnerOutput(hidden_states=tri)
                except Exception as e:
                    logger.warning("[MOE_PREFILL_CMP] triton compare failed: %s", e)
            _moe_tier_diag(_M, "tier2_esimd_prefill", quant_info)
            return XpuRunnerOutput(hidden_states=out)

        # Tier 3: native sgl-kernel SYCL (downstream parity) for NON-fp8 MoE,
        # BEFORE the Triton fallback (the Triton fused_moe_kernel cannot compile
        # on BMG — scratch > 262 KB PTSS). fp8 returns None here (upstream sgl
        # kernel asserts use_fp8_w8a8 is False), so fp8 that reached this point
        # (prefill op unavailable/declined) falls through to triton as a last try.
        out = self._run_sgl_native(runner_input, quant_info)
        if out is not None:
            _moe_tier_diag(_M, "tier3_sgl_native", quant_info)
            return XpuRunnerOutput(hidden_states=out)

        _moe_tier_diag(_M, "tier4_triton", quant_info)
        out = self._run_triton_fallback(runner_input, quant_info, running_state, hooks)
        return XpuRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.INTEL_XPU


@register_pre_permute("standard", "intel_xpu")
def pre_permute_standard_to_xpu(
    dispatch_output: StandardDispatchOutput,
    quant_info: XpuMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> XpuRunnerInput:
    """Standard dispatcher -> XPU runner input. Same shape as triton pre-permute."""
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import (
        _prepare_fused_moe_run,
    )
    from sglang.srt.layers.moe.topk import TopKOutputChecker

    hidden_states, topk_output = (
        dispatch_output.hidden_states,
        dispatch_output.topk_output,
    )

    assert TopKOutputChecker.format_is_standard(topk_output)

    (
        config,
        down_config,
        down_moe_use_tma,
        up_moe_use_tma,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
    ) = _prepare_fused_moe_run(
        hidden_states,
        quant_info.w13_weight,
        quant_info.w2_weight,
        topk_output.topk_ids,
        use_fp8_w8a8=quant_info.use_fp8_w8a8,
        use_int8_w8a8=quant_info.use_int8_w8a8,
        use_int8_w8a16=quant_info.use_int8_w8a16,
        use_int4_w4a16=quant_info.use_int4_w4a16,
        per_channel_quant=quant_info.per_channel_quant,
        block_shape=quant_info.block_shape,
    )

    running_state["config"] = config
    running_state["down_config"] = down_config
    running_state["down_moe_use_tma"] = down_moe_use_tma
    running_state["up_moe_use_tma"] = up_moe_use_tma

    return XpuRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("intel_xpu", "standard")
def post_permute_xpu_to_standard(
    runner_output: XpuRunnerOutput,
    quant_info: XpuMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(hidden_states=runner_output.hidden_states)
