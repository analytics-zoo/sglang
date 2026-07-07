from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, List, Optional

import torch

from sglang.srt.layers.moe.moe_runner.base import (
    MoeQuantInfo,
    MoeRunnerConfig,
    MoeRunnerCore,
    RunnerInput,
    RunnerOutput,
    register_fused_func,
    register_post_permute,
    register_pre_permute,
)
from sglang.srt.layers.moe.utils import MoeRunnerBackend

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher.standard import (
        StandardCombineInput,
        StandardDispatchOutput,
    )


# Lazy-loaded esimd MoE op: registers torch.ops.moe_ops.moe_forward_full_silu_routed.
# Returns the op handle on success, None on failure (so we always fall back to Triton).
def _load_esimd_moe_op(fp8_variant: str = "e4m3"):
    """Load the right ESIMD MoE silu-routed kernel for the given fp8 variant.

    fp8_variant: "e4m3" → moe_forward_full_silu_routed_sglang
                 "e5m2" → moe_forward_full_silu_routed_e5m2
    """
    cache = getattr(_load_esimd_moe_op, "_cached", None)
    if cache is None:
        cache = {}
        _load_esimd_moe_op._cached = cache
    if fp8_variant in cache:
        return cache[fp8_variant]
    op = None
    try:
        from custom_esimd_kernels_sglang import moe_ops as _moe_mod
        if fp8_variant == "e5m2":
            op = _moe_mod.moe_forward_full_silu_routed_e5m2
        else:
            op = _moe_mod.moe_forward_full_silu_routed_sglang
    except Exception:
        op = None
    cache[fp8_variant] = op
    return op


def _try_esimd_moe_silu_routed(
    runner_input: "TritonRunnerInput",
    quant_info: "TritonMoeQuantInfo",
    config,
) -> Optional[torch.Tensor]:
    """XPU fast path for FP8 e4m3 silu MoE: dispatch to custom ESIMD kernel.

    Returns the MoE forward output tensor [T, hidden] on success, or None to
    fall back to the Triton path. Conditions for hitting this path:
      - hidden_states on XPU
      - FP8 W8A8 (e4m3) quant
      - silu activation (Qwen3-style)
      - small T (decode + small chunked-prefill); large T falls back since the
        kernel was tuned for small batch
      - non-gated, non-clamp config (vanilla SwiGLU)
    """
    _DEBUG = os.environ.get("SGLANG_ESIMD_MOE_DEBUG", "0") == "1"
    def _dbg(reason):
        if _DEBUG:
            print(f"[esimd-moe-runner] skip: {reason}", flush=True)

    if not quant_info.use_fp8_w8a8:
        _dbg(f"not fp8_w8a8")
        return None
    h = runner_input.hidden_states
    if h.device.type != "xpu":
        _dbg(f"device={h.device.type}")
        return None
    orig_dtype = h.dtype
    if h.dtype != torch.float16:
        # Kernel only supports fp16 — cast input/output once around the call.
        h = h.to(torch.float16)
    w13 = quant_info.w13_weight
    w2 = quant_info.w2_weight
    _fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if w13.dtype not in _fp8_dtypes or w2.dtype not in _fp8_dtypes:
        _dbg(f"weight dtype: w13={w13.dtype} w2={w2.dtype}")
        return None
    if w13.dtype != w2.dtype:
        _dbg(f"w13/w2 dtype mismatch: {w13.dtype} vs {w2.dtype}")
        return None
    # Only "silu" activation supported; gelu_tanh would need a separate wrapper.
    if getattr(config, "activation", "silu") != "silu":
        _dbg(f"act={getattr(config, 'activation', '?')}")
        return None
    # No gated/clamp transforms.
    if getattr(config, "gemm1_alpha", None) is not None:
        _dbg("gemm1_alpha set")
        return None
    if getattr(config, "gemm1_clamp_limit", None) is not None:
        _dbg("gemm1_clamp_limit set")
        return None
    if getattr(config, "swiglu_limit", None) is not None:
        _dbg("swiglu_limit set")
        return None
    T = h.shape[0]
    if T > 8:  # Tuned for decode + tiny prefill chunks
        _dbg(f"T={T} > 8")
        return None
    fp8_variant = "e5m2" if w13.dtype == torch.float8_e5m2 else "e4m3"
    op = _load_esimd_moe_op(fp8_variant)
    if op is None:
        _dbg(f"op not loaded ({fp8_variant})")
        return None
    if _DEBUG:
        print(f"[esimd-moe] HIT T={T} hidden={h.shape[1]} top_k={runner_input.topk_ids.shape[-1]} fp8={fp8_variant}", flush=True)

    # Collapse per-block weight_scale to per-expert per-tensor (mean), cached
    # on the parameter tensor to amortise across calls.
    def _per_expert_pt_scale(scale: torch.Tensor) -> torch.Tensor:
        cached = getattr(scale, "_esimd_pt_per_expert", None)
        if cached is not None:
            return cached
        # scale: [E, *block_dims] → per-expert mean → [E] fp32
        s = scale.to(torch.float32).reshape(scale.shape[0], -1).mean(dim=-1)
        s = s.contiguous()
        try:
            scale._esimd_pt_per_expert = s
        except Exception:
            pass
        return s

    if quant_info.w13_scale is None or quant_info.w2_scale is None:
        return None
    s13 = _per_expert_pt_scale(quant_info.w13_scale)
    s2 = _per_expert_pt_scale(quant_info.w2_scale)

    # We use the `_sglang` kernel variant which accepts w13 directly in
    # sglang's [E, 2*intermediate, hidden] layout — no transpose / copy
    # required, no extra memory cost, and the Triton fallback can still read
    # the same parameter unchanged.
    w13_kernel = w13

    topk_w = runner_input.topk_weights
    topk_i = runner_input.topk_ids
    if topk_w.dtype != torch.float16:
        topk_w = topk_w.to(torch.float16)
    if topk_i.dtype != torch.int32:
        topk_i = topk_i.to(torch.int32)

    top_k = topk_i.shape[-1]
    n_routed = w13.shape[0]
    try:
        out = op(h, topk_w, topk_i, w13_kernel, s13, w2, s2, top_k, n_routed)
    except Exception:
        return None
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out


@dataclass
class TritonRunnerInput(RunnerInput):

    hidden_states: torch.Tensor
    topk_weights: torch.Tensor
    topk_ids: torch.Tensor
    sorted_token_ids: torch.Tensor
    expert_ids: torch.Tensor
    num_tokens_post_padded: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonRunnerOutput(RunnerOutput):

    hidden_states: torch.Tensor

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


@dataclass
class TritonMoeQuantInfo(MoeQuantInfo):
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


class TritonRunnerCore(MoeRunnerCore):

    def __init__(self, config: MoeRunnerConfig):
        super().__init__(config)

    def run(
        self,
        runner_input: TritonRunnerInput,
        quant_info: TritonMoeQuantInfo,
        running_state: dict,
        hooks: Optional[Any] = None,
    ) -> TritonRunnerOutput:
        # XPU fast path: gated by SGLANG_ENABLE_ESIMD_MOE=1. The fused-func
        # path (`fused_experts_none_to_triton`) is the one that actually fires
        # under the default runner config; this branch is here for the future
        # case where a runner_input gets routed straight into the runner_core.
        if os.environ.get("SGLANG_ENABLE_ESIMD_MOE", "0") == "1":
            out = _try_esimd_moe_silu_routed(runner_input, quant_info, self.config)
            if out is not None:
                return TritonRunnerOutput(hidden_states=out)

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
        )

        return TritonRunnerOutput(hidden_states=out)

    @property
    def runner_backend(self) -> MoeRunnerBackend:
        return MoeRunnerBackend.TRITON


def _maybe_esimd_moe_silu_fused(
    dispatch_output,
    quant_info,
    runner_config,
):
    """Small-T fast path for the `fused_experts_none_to_triton` entry point.

    Returns the MoE output tensor on success, or None to fall back. Mirrors
    the conditions in `_try_esimd_moe_silu_routed` plus extracts the topk
    weights/ids from the StandardDispatchOutput.
    """
    if not quant_info.use_fp8_w8a8:
        return None
    h = dispatch_output.hidden_states
    if h.device.type != "xpu":
        return None
    w13 = quant_info.w13_weight
    w2 = quant_info.w2_weight
    _fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if w13.dtype not in _fp8_dtypes or w2.dtype not in _fp8_dtypes:
        return None
    if w13.dtype != w2.dtype:
        return None
    if getattr(runner_config, "activation", "silu") != "silu":
        return None
    if getattr(runner_config, "gemm1_alpha", None) is not None:
        return None
    if getattr(runner_config, "gemm1_clamp_limit", None) is not None:
        return None
    if getattr(runner_config, "swiglu_limit", None) is not None:
        return None
    if quant_info.b13 is not None or quant_info.b2 is not None:
        return None
    if h.shape[0] > 8:
        return None
    fp8_variant = "e5m2" if w13.dtype == torch.float8_e5m2 else "e4m3"
    op = _load_esimd_moe_op(fp8_variant)
    if op is None:
        return None

    topk_weights, topk_ids, _ = dispatch_output.topk_output

    def _per_expert_pt_scale(scale):
        cached = getattr(scale, "_esimd_pt_per_expert", None)
        if cached is not None:
            return cached
        s = scale.to(torch.float32).reshape(scale.shape[0], -1).mean(dim=-1)
        s = s.contiguous()
        try:
            scale._esimd_pt_per_expert = s
        except Exception:
            pass
        return s

    if quant_info.w13_scale is None or quant_info.w2_scale is None:
        return None
    s13 = _per_expert_pt_scale(quant_info.w13_scale)
    s2 = _per_expert_pt_scale(quant_info.w2_scale)

    # Use the `_sglang` kernel variant: accepts w13 in [E, 2*inter, hidden]
    # directly, no transpose required.
    w13_kernel = w13

    if topk_weights.dtype != torch.float16:
        topk_weights = topk_weights.to(torch.float16)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    orig_dtype = h.dtype
    x_in = h if h.dtype == torch.float16 else h.to(torch.float16)
    top_k = topk_ids.shape[-1]
    n_routed = w13.shape[0]
    try:
        out = op(x_in, topk_weights, topk_ids, w13_kernel, s13, w2, s2, top_k, n_routed)
    except Exception:
        return None
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out


def _load_esimd_moe_prefill_op():
    """Lazy-load the M-tiled FP8 MoE prefill op (moe_prefill_full_fp8)."""
    cache = getattr(_load_esimd_moe_prefill_op, "_cached", "unset")
    if cache != "unset":
        return cache
    op = None
    try:
        from custom_esimd_kernels_sglang import moe_fp8_prefill_ops  # noqa: F401

        op = torch.ops.moe_fp8_prefill_ops.moe_prefill_full_fp8
    except Exception:
        op = None
    _load_esimd_moe_prefill_op._cached = op
    return op


def _maybe_esimd_moe_silu_prefill(
    dispatch_output,
    quant_info,
    runner_config,
):
    """Large-T (prefill) fast path: M-tiled DPAS FP8 MoE GEMM kernel.

    The decode kernel (_maybe_esimd_moe_silu_fused) is DPAS M=1 and gated to
    T<=8; for prefill (T up to chunked_prefill_size) it would be GEMV-bound.
    This path routes prefill batches to moe_prefill_full_fp8, which sorts tokens
    by expert and runs a real M-tiled (MAX_M=32) DPAS GEMM. Online-fp8 weights
    carry a per-expert per-tensor scale (w13_scale=[E,2], w2_scale=[E]); we
    collapse w13's two scales to a per-expert mean (minor vs the block-collapse
    that breaks block-quant checkpoints — verified numerically cos=1.0).

    Returns the MoE output [T, hidden] on success, or None to fall back.
    """
    if not quant_info.use_fp8_w8a8:
        return None
    h = dispatch_output.hidden_states
    if h.device.type != "xpu":
        return None
    w13 = quant_info.w13_weight
    w2 = quant_info.w2_weight
    _fp8_dtypes = (torch.float8_e4m3fn, torch.float8_e5m2)
    if w13.dtype not in _fp8_dtypes or w2.dtype not in _fp8_dtypes:
        return None
    if w13.dtype != w2.dtype:
        return None
    if getattr(runner_config, "activation", "silu") != "silu":
        return None
    if getattr(runner_config, "gemm1_alpha", None) is not None:
        return None
    if getattr(runner_config, "gemm1_clamp_limit", None) is not None:
        return None
    if getattr(runner_config, "swiglu_limit", None) is not None:
        return None
    if quant_info.b13 is not None or quant_info.b2 is not None:
        return None
    if quant_info.w13_scale is None or quant_info.w2_scale is None:
        return None
    # Only block_shape==None (online per-expert per-tensor) is supported by the
    # per-expert scalar kernel; block-quant checkpoints need a block-scale kernel.
    if getattr(quant_info, "block_shape", None) is not None:
        return None
    op = _load_esimd_moe_prefill_op()
    if op is None:
        return None

    # The prefill kernel keeps gate/up scales SEPARATE (g_acc*=s_gate,
    # u_acc*=s_up): online fp8 w13_scale is [E,2] (gate=w1, up=w3) and
    # averaging the two to one scalar is too lossy (gsm8k garbage). Pass the
    # [E,2] tensor straight through; w2_scale is [E].
    # Online fp8 here uses requantize_with_max_scale, so w1 and w3 are
    # requantized to a SINGLE common per-expert scale: w13_scale is [E] meaning
    # the same scale for gate and up. Accept [E] (expand to [E,2], exact) and
    # [E,2] (separate). w2_scale is [E] scalar.
    def _f32c(scale, want_2d):
        key = "_esimd_prefill_w13" if want_2d else "_esimd_prefill_w2"
        cached = getattr(scale, key, None)
        if cached is not None:
            return cached
        s = scale.to(torch.float32).reshape(scale.shape[0], -1)
        if want_2d:
            if s.shape[1] == 1:
                s = s.expand(s.shape[0], 2)  # [E] -> [E,2], gate==up (max-scale requant)
            elif s.shape[1] != 2:
                return None
        else:
            s = s.mean(dim=-1)  # [E]
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
    # NOTE: this routed-only path does NOT yet handle Qwen3.6 shared-expert
    # fusion (shared_expert_intermediate_size == moe_intermediate_size), so the
    # full MoE output is incomplete -> SGLANG_ENABLE_ESIMD_MOE_PREFILL must stay
    # off until the shared expert contribution is added. Kept wired + scale-fixed
    # for when that lands.

    topk_weights, topk_ids, _ = dispatch_output.topk_output
    if topk_weights.dtype != torch.float16:
        topk_weights = topk_weights.to(torch.float16)
    if topk_ids.dtype != torch.int32:
        topk_ids = topk_ids.to(torch.int32)
    orig_dtype = h.dtype
    x_in = h if h.dtype == torch.float16 else h.to(torch.float16)
    top_k = topk_ids.shape[-1]
    n_routed = w13.shape[0]
    try:
        out = op(x_in, topk_weights, topk_ids, w13, s13, w2, s2, top_k, n_routed)
    except Exception:
        return None
    if out.dtype != orig_dtype:
        out = out.to(orig_dtype)
    return out


@register_fused_func("none", "triton")
def fused_experts_none_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
) -> StandardCombineInput:
    from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe import fused_experts
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    # XPU fast path: small-T (decode) FP8 e4m3 silu MoE → custom ESIMD kernel.
    # DISABLED — collapsing the per-block weight scale to a per-expert mean
    # (the trick that worked for the GEMM fast path) destroys accuracy on
    # MoE: gsm8k fell from 70% to 0% (output degraded to garbage tokens).
    # The 256 experts span a wider scale range than the per-block scales of
    # a single linear layer, so a single scalar per expert is too lossy.
    # TODO: migrate the per-block FP8 MoE GEMM kernel from llm-scaler/vllm
    # (or feed 2D scale through a future kernel variant) before re-enabling.
    if os.environ.get("SGLANG_ENABLE_ESIMD_MOE", "0") == "1":
        esimd_out = _maybe_esimd_moe_silu_fused(
            dispatch_output, quant_info, runner_config
        )
        if esimd_out is not None:
            return StandardCombineInput(hidden_states=esimd_out)

    # Large-T (prefill) M-tiled DPAS FP8 MoE kernel. Separate env gate so it can
    # be enabled independently of the decode kernel.
    if os.environ.get("SGLANG_ENABLE_ESIMD_MOE_PREFILL", "0") == "1":
        esimd_out = _maybe_esimd_moe_silu_prefill(
            dispatch_output, quant_info, runner_config
        )
        if esimd_out is not None:
            return StandardCombineInput(hidden_states=esimd_out)

    output = fused_experts(
        hidden_states=dispatch_output.hidden_states,
        w1=quant_info.w13_weight,
        w2=quant_info.w2_weight,
        topk_output=dispatch_output.topk_output,
        moe_runner_config=runner_config,
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
    )

    return StandardCombineInput(
        hidden_states=output,
    )


@register_pre_permute("standard", "triton")
def pre_permute_standard_to_triton(
    dispatch_output: StandardDispatchOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> TritonRunnerInput:

    # NOTE: this is dead code as a fused func for standard format is registered.
    # This is left here for testing and examples.

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

    return TritonRunnerInput(
        hidden_states=hidden_states,
        topk_weights=topk_output.topk_weights,
        topk_ids=topk_output.topk_ids,
        sorted_token_ids=sorted_token_ids,
        expert_ids=expert_ids,
        num_tokens_post_padded=num_tokens_post_padded,
    )


@register_post_permute("triton", "standard")
def post_permute_triton_to_standard(
    runner_output: TritonRunnerOutput,
    quant_info: TritonMoeQuantInfo,
    runner_config: MoeRunnerConfig,
    running_state: dict,
) -> StandardCombineInput:

    # NOTE: this is dead code as a fused func for standard format is registered.
    # This is left here for testing and examples.

    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput

    return StandardCombineInput(
        hidden_states=runner_output.hidden_states,
    )
