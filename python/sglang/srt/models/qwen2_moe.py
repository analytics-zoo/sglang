# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2023-2024 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Adapted from
# https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen2_moe.py
"""Inference-only Qwen2MoE model compatible with HuggingFace weights."""

import logging
import os
from contextlib import nullcontext
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.batch_overlap.two_batch_overlap import model_forward_maybe_tbo
from sglang.srt.distributed import (
    get_moe_data_parallel_world_size,
    get_moe_expert_parallel_world_size,
    get_pp_group,
    get_pp_indices,
    get_tensor_model_parallel_world_size,
    attention_tensor_model_parallel_all_reduce,
    tensor_model_parallel_all_reduce,
)
from sglang.srt.distributed.parallel_state import (
    get_attn_context_model_parallel_world_size,
)
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation
from sglang.srt.eplb.expert_location_dispatch import ExpertLocationDispatchInfo
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.communicator import (
    LayerCommunicator,
    LayerScatterModes,
    ScatterMode,
)
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)
from sglang.srt.layers.layernorm import RMSNorm
from sglang.srt.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    ReplicatedLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor
from sglang.srt.layers.moe import (
    get_moe_a2a_backend,
    should_skip_post_experts_all_reduce,
)
from sglang.srt.layers.moe.ep_moe.layer import get_moe_impl_class
from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
from sglang.srt.layers.moe.topk import StandardTopKOutput, TopK, TopKOutputChecker
from sglang.srt.layers.moe.utils import (
    RoutingMethodType,
    filter_moe_weight_param_global_expert,
    is_deepep_class_backend,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.utils.cp_utils import (
    cp_all_gather_rerange_output,
    cp_split_and_rebuild_data,
    cp_split_and_rebuild_position,
    is_prefill_context_parallel_enabled,
)
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.server_args import get_global_server_args
from sglang.srt.utils import (
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_hip,
    is_npu,
    make_layers,
    use_intel_amx_backend,
)

if is_npu():
    from sglang.srt.hardware_backend.npu.cmo import (
        shared_expert_on_independent_stream,
        wait_share_stream,
    )

from sglang.srt.environ import envs
from sglang.srt.utils.hf_transformers_utils import get_rope_config

_SGLANG_EXPERIMENTAL_LORA_OPTI = envs.SGLANG_EXPERIMENTAL_LORA_OPTI.get()

logger = logging.getLogger(__name__)

# ── NaN/Inf probe (env-gated: SGLANG_NAN_PROBE=1) ───────────────────────────
# Isolates whether a non-finite MoE output originates in the routed experts
# (e.g. GGUF grouped prefill GGEMV at batch>8) or the shared expert. Zero
# overhead when SGLANG_NAN_PROBE is unset. See qwen3_5.py for the layer-level
# probe that pins down the origin layer / prefill-vs-decode / token count.
import os as _os_np_moe

_NAN_PROBE_MOE = _os_np_moe.environ.get("SGLANG_NAN_PROBE", "0") == "1"


def _nan_probe_moe(tag, t, layer_id=None, forward_batch=None):
    if not _NAN_PROBE_MOE or not torch.is_tensor(t) or t.numel() == 0:
        return
    try:
        if not t.dtype.is_floating_point or bool(torch.isfinite(t).all()):
            return
        mode = "?"
        if forward_batch is not None:
            fm = getattr(forward_batch, "forward_mode", None)
            mode = getattr(fm, "name", str(fm)) if fm is not None else "?"
        logger.error(
            "[NANPROBE] tag=%s layer=%s mode=%s ntok=%d shape=%s nan=%d inf=%d",
            tag,
            layer_id,
            mode,
            int(t.shape[0]) if t.dim() > 0 else -1,
            tuple(t.shape),
            int(torch.isnan(t).sum().item()),
            int(torch.isinf(t).sum().item()),
        )
    except Exception:
        pass

_is_cuda = is_cuda()
_is_cpu = is_cpu()
_is_cpu_amx_available = cpu_has_amx_support()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip


# ---------------------------------------------------------------------------
# ESIMD MoE router (fp8) fast path.
#
# vLLM's BMG path runs the MoE router logits through an ESIMD fp8 GEMV
# (`custom_esimd_kernels_sglang.moe_ops.moe_router_forward`) instead of the
# default fp16 `aten::mm`, saving ~1 host-dispatch/layer during decode. The
# kernel ONLY accepts fp8 e4m3/e5m2 weights (no fp16 variant), so the fp16 gate
# weight is quantized once (per-tensor amax/448) and cached on the module.
#
# Gated by SGL_XPU_MOE_ROUTER_FP8=1 (default OFF: keeps the accurate fp16 gate).
# Quantizing the router to per-tensor fp8 perturbs ~top-8 routing on a fraction
# of tokens, so this must be gsm8k A/B validated before enabling in production.
# ---------------------------------------------------------------------------
_ESIMD_MOE_ROUTER_OP = "unset"
# Set once at launch; constant for the process. Cached to avoid an os.environ
# read on every MoE layer's router call during decode.
_MOE_ROUTER_FP8 = os.environ.get("SGL_XPU_MOE_ROUTER_FP8", "0") == "1"
logger.warning(
    "[router_fp8] module loaded: file=%s SGL_XPU_MOE_ROUTER_FP8=%s (_MOE_ROUTER_FP8=%s)",
    __file__, os.environ.get("SGL_XPU_MOE_ROUTER_FP8"), _MOE_ROUTER_FP8,
)


def _load_esimd_moe_router_op():
    global _ESIMD_MOE_ROUTER_OP
    if _ESIMD_MOE_ROUTER_OP != "unset":
        return _ESIMD_MOE_ROUTER_OP
    op = None
    try:
        from custom_esimd_kernels_sglang import moe_ops as _moe_mod

        op = getattr(_moe_mod, "moe_router_forward", None)
    except Exception:
        op = None
    _ESIMD_MOE_ROUTER_OP = op
    return op


_ROUTER_DEBUG = os.environ.get("SGL_XPU_ROUTER_DEBUG", "0") == "1"
_ROUTER_DEBUG_N = 0


def _router_dbg(reason, **kw):
    global _ROUTER_DEBUG_N
    if not _ROUTER_DEBUG or _ROUTER_DEBUG_N >= 12:
        return
    _ROUTER_DEBUG_N += 1
    logger.warning("[router_fp8] fallback=%s %s", reason, kw)


def _esimd_router_wq_scale(gate, hidden_states: torch.Tensor):
    """Return (wq_e4m3 [E,H], scale [1] fp32) for the router gate weight, lazily
    quantized + cached on the weight, or None on any shape/dtype/layout mismatch.
    Shared by _esimd_router_logits (split path) and the rtfused MoE path.
    """
    if not _MOE_ROUTER_FP8:
        _router_dbg("env_off")
        return None
    x = hidden_states
    if x.device.type != "xpu" or x.dim() != 2:
        _router_dbg("device_or_dim", dev=str(x.device), dim=x.dim(), shape=tuple(x.shape))
        return None
    # Kernel is tuned for decode + tiny prefill chunks (M=1 GEMV).
    if x.shape[0] > 8:
        return None
    weight = getattr(gate, "weight", None)
    if weight is None or weight.dim() != 2:
        _router_dbg("weight_missing", gate_type=type(gate).__name__,
                    has_w=weight is not None)
        return None
    N, K = weight.shape  # [num_experts, hidden]
    if x.shape[1] != K:
        _router_dbg("K_mismatch", xK=x.shape[1], wK=K, N=N)
        return None
    # A bias'd gate would need to be added post-GEMV; keep it simple & safe.
    if getattr(gate, "bias", None) is not None:
        _router_dbg("has_bias", bias_type=type(getattr(gate, "bias")).__name__)
        return None
    # Lazy per-tensor fp8 quant of the gate weight, cached on the parameter.
    wq = getattr(weight, "_esimd_router_wq", None)
    sc = getattr(weight, "_esimd_router_scale", None)
    if wq is None or sc is None:
        try:
            wf = weight.detach().float()
            amax = wf.abs().max()
            if not torch.isfinite(amax) or amax <= 0:
                return None
            scale = (amax / 448.0)
            wq = torch.clamp(wf / scale, -448.0, 448.0).to(torch.float8_e4m3fn).contiguous()
            sc = scale.reshape(1).to(torch.float32).contiguous()
            weight._esimd_router_wq = wq
            weight._esimd_router_scale = sc
        except Exception as e:
            _router_dbg("quant_exc", err=repr(e))
            return None
    return wq, sc


def _esimd_router_logits(gate, hidden_states: torch.Tensor):
    """Compute router logits via the ESIMD fp8 GEMV, or return None to fall back.

    Returns a [T, num_experts] fp16 logits tensor on success. Falls back (None)
    on any shape/dtype/layout mismatch so the standard fp16 gate runs instead.
    """
    op = _load_esimd_moe_router_op()
    if op is None:
        _router_dbg("op_none")
        return None
    wqsc = _esimd_router_wq_scale(gate, hidden_states)
    if wqsc is None:
        return None
    wq, sc = wqsc
    x = hidden_states
    x_in = x if x.dtype == torch.float16 else x.to(torch.float16)
    try:
        logits = op(x_in, wq, sc)
    except Exception as e:
        _router_dbg("kernel_exc", err=repr(e))
        return None
    if _ROUTER_DEBUG:
        _router_dbg("SUCCESS", out_shape=tuple(logits.shape))
    return logits


# ---------------------------------------------------------------------------
# ESIMD full MoE fusion (e5m2) fast path.
#
# `moe_forward_full_v2` fuses the whole decode MoE block into ONE dispatch:
#   router topk (softmax, renorm) + routed experts (silu) + shared expert (silu)
#   + shared_expert_gate (sigmoid) + weighted accumulate.
# This collapses the unfused decode path (routed silu kernel + shared-expert
# gate_up/act/down linears + gate linear/sigmoid/mul + adds ≈ 300 dispatch/step)
# down to a single op. e5m2-only (the kernel dequantises fp8_e5m2 weights);
# activations stay fp16. Gated by SGL_XPU_ESIMD_MOE_FULL=1.
#
# Weight layouts the kernel expects (all validated numerically, cos≈1.0):
#   routed gate_up : [E, 2*inter, hidden]  (native sglang w13, no transpose)
#   routed down    : [E, hidden, inter]    (native sglang w2, no transpose)
#   shared gate_up : [NS, 2*inter, hidden] (natural)  -> transpose of stored [hidden,2*inter]
#   shared down    : [NS, hidden, inter]   (natural)  -> transpose of stored [inter,hidden]
#   shared_gate    : [NS, hidden] fp16 (not quantised)
#   all scales     : per-expert per-tensor fp32 (dequant scale = amax/fp8_max)
# ---------------------------------------------------------------------------
_ESIMD_MOE_FULL = os.environ.get("SGL_XPU_ESIMD_MOE_FULL", "0") == "1"
_ESIMD_MOE_FULL_OP = "unset"
_MOE_FULL_DEBUG = os.environ.get("SGL_XPU_MOE_FULL_DEBUG", "0") == "1"
_MOE_FULL_DEBUG_N = 0


def _moe_full_dbg(reason, **kw):
    global _MOE_FULL_DEBUG_N
    if not _MOE_FULL_DEBUG or _MOE_FULL_DEBUG_N >= 16:
        return
    _MOE_FULL_DEBUG_N += 1
    logger.warning("[moe_full] fallback=%s %s", reason, kw)


def _load_esimd_moe_full_op():
    global _ESIMD_MOE_FULL_OP
    if _ESIMD_MOE_FULL_OP != "unset":
        return _ESIMD_MOE_FULL_OP
    ops = None
    try:
        from custom_esimd_kernels_sglang import moe_ops as _moe_mod

        v2 = getattr(_moe_mod, "moe_forward_full_v2", None)
        # BSZ=1 decode: moe_forward_full uses the fused down_finalize kernel
        # (shared-expert down + gate sigmoid + accumulate in ONE dispatch),
        # so it launches 5 internal kernels vs v2's 7 (gate_precompute +
        # down_shared + accumulate collapse into down_finalize). Prefer it when
        # n_tokens==1; fall back to v2 for multi-token.
        full = getattr(_moe_mod, "moe_forward_full", None)
        # rtfused: BSZ=1 path that also fuses router GEMV + topk into one
        # dispatch (takes the e4m3 gate weight + scale instead of logits).
        rtfused = getattr(_moe_mod, "moe_forward_full_rtfused", None)
        # rtfused_norm: BSZ=1 path that ALSO folds the pre-MoE GemmaRMSNorm
        # (resadd + rmsnorm) into the fused router kernel head and returns the
        # new residual, removing the standalone gemma_fused_add_rmsnorm dispatch.
        rtfused_norm = getattr(_moe_mod, "moe_forward_full_rtfused_norm", None)
        if v2 is not None:
            ops = {"v2": v2, "full": full, "rtfused": rtfused,
                   "rtfused_norm": rtfused_norm}
    except Exception:
        ops = None
    _ESIMD_MOE_FULL_OP = ops
    return ops


def _pt_scale_1d(scale):
    """Collapse a per-expert weight scale to a contiguous 1-D fp32 [E] tensor."""
    if scale is None:
        return None
    s = scale.to(torch.float32)
    if s.dim() == 0:
        s = s.reshape(1)
    elif s.dim() > 1:
        # [E, *block] -> per-expert scalar (these are already per-tensor scales,
        # so any trailing dims are size-1; mean is a safe collapse).
        s = s.reshape(s.shape[0], -1).mean(dim=-1)
    return s.contiguous()


def _gather_moe_full_weights(block, x: torch.Tensor):
    """Collect + validate all tensors needed by the fused decode MoE ops
    (moe_forward_full / rtfused / rtfused_norm). Returns a dict of the routed +
    shared expert weights/scales, the gate module, and routing dims, or None to
    fall back. Layout/dtype checks here guarantee a bad tensor never reaches the
    kernel."""
    experts = getattr(block, "experts", None)
    shared = getattr(block, "shared_expert", None)
    sgate = getattr(block, "shared_expert_gate", None)
    gate = getattr(block, "gate", None)
    topk = getattr(block, "topk", None)
    if experts is None or shared is None or sgate is None or gate is None or topk is None:
        _moe_full_dbg("missing_submodule", experts=experts is not None,
                      shared=shared is not None, sgate=sgate is not None)
        return None

    # Fused topk uses softmax + renorm + plain (ungrouped, no-bias) selection.
    tc = getattr(topk, "topk_config", None)
    if tc is None:
        return None
    if getattr(tc, "scoring_func", "softmax") != "softmax":
        return None
    if not getattr(tc, "renormalize", True):
        return None
    if getattr(tc, "use_grouped_topk", False):
        return None
    if getattr(tc, "correction_bias", None) is not None:
        return None
    if getattr(tc, "custom_routing_function", None) is not None:
        return None
    top_k = getattr(tc, "top_k", None)
    if top_k is None:
        return None

    # --- routed weights (e5m2) + load-time transposed caches ---
    w13 = getattr(experts, "w13_weight", None)
    w2 = getattr(experts, "w2_weight", None)
    if w13 is None or w2 is None:
        return None
    if w13.dtype != torch.float8_e5m2 or w2.dtype != torch.float8_e5m2:
        _moe_full_dbg("routed_not_e5m2", w13=str(w13.dtype), w2=str(w2.dtype))
        return None
    gate_up_routed = w13                                   # native [E, 2*inter, hidden]
    down_routed = w2                                        # natural [E, hidden, inter]
    s13 = _pt_scale_1d(getattr(experts, "w13_weight_scale", None))
    s2 = _pt_scale_1d(getattr(experts, "w2_weight_scale", None))
    if s13 is None or s2 is None:
        return None

    # --- shared expert (dense Fp8, weights stored transposed [hidden, out]) ---
    gu_s = getattr(shared, "gate_up_proj", None)
    dn_s = getattr(shared, "down_proj", None)
    if gu_s is None or dn_s is None:
        return None
    gw = getattr(gu_s, "weight", None)   # stored [hidden, 2*inter] e5m2
    dw = getattr(dn_s, "weight", None)   # stored [inter, hidden]   e5m2
    if gw is None or dw is None:
        return None
    if gw.dtype != torch.float8_e5m2 or dw.dtype != torch.float8_e5m2:
        _moe_full_dbg("shared_not_e5m2", gw=str(gw.dtype), dw=str(dw.dtype))
        return None
    shared_gate_up = getattr(gw, "_esimd_moe_full_nat", None)
    if shared_gate_up is None:
        try:
            shared_gate_up = gw.t().contiguous().unsqueeze(0)  # [1, 2*inter, hidden]
            gw._esimd_moe_full_nat = shared_gate_up
        except Exception:
            return None
    shared_down = getattr(dw, "_esimd_moe_full_nat", None)
    if shared_down is None:
        try:
            shared_down = dw.t().contiguous().unsqueeze(0)      # [1, hidden, inter]
            dw._esimd_moe_full_nat = shared_down
        except Exception:
            return None
    ss13 = _pt_scale_1d(getattr(gu_s, "weight_scale", None))
    ss2 = _pt_scale_1d(getattr(dn_s, "weight_scale", None))
    if ss13 is None or ss2 is None:
        return None

    # --- shared_expert_gate weight (fp16 [NS, hidden], not quantised) ---
    sgw = getattr(sgate, "weight", None)
    if sgw is None or sgw.dim() != 2:
        return None
    sgw16 = (sgw if sgw.dtype == torch.float16 else sgw.to(torch.float16)).contiguous()

    return {
        "gate": gate,
        "top_k": int(top_k),
        "n_routed": int(w13.shape[0]),
        "num_shared": int(shared_gate_up.shape[0]),
        "gate_up_routed": gate_up_routed, "s13": s13,
        "shared_gate_up": shared_gate_up, "ss13": ss13,
        "down_routed": down_routed, "s2": s2,
        "shared_down": shared_down, "ss2": ss2,
        "sgw16": sgw16,
    }


_GGUF_INTROSPECT_DONE = set()
def _load_shared_q8_op():
    try:
        import custom_esimd_kernels_sglang.custom_esimd_kernels  # noqa: F401 (registers ops)
    except Exception:
        pass
    ns = getattr(torch.ops, "custom_esimd_kernels_sglang", None)
    return getattr(ns, "esimd_shared_expert_q8", None) if ns is not None else None


_GGUF_MOE_SHARED = bool(
    os.environ.get("SGL_XPU_GGUF_MOE_SHARED")
    or os.environ.get("SGLANG_XPU_GGUF_MOE_SHARED")
)
_SHARED_Q8_OP = None


def _gather_gguf_shared_q8(block):
    """Collect + validate the Q8_0 shared-expert tensors for the fused decode op.
    Returns (gu_qs, gu_sc, d_qs, d_sc, wg, inter_s) or False to disable."""
    se = getattr(block, "shared_expert", None)
    sg = getattr(block, "shared_expert_gate", None)
    if se is None or sg is None:
        return False
    gu = getattr(se, "gate_up_proj", None)
    dn = getattr(se, "down_proj", None)
    if gu is None or dn is None:
        return False

    # gate_up: prefer the merged rep (rows gate then up); else concat per-shard.
    merged = getattr(gu, "_xpu_merged", None)
    if isinstance(merged, tuple) and merged and merged[0] == "q8_0":
        gu_qs, gu_sc = merged[1].contiguous(), merged[2].contiguous()
    else:
        reps = getattr(gu, "_xpu_reps", None)
        order = getattr(gu, "_xpu_shard_order", None)
        if not isinstance(reps, dict) or not order:
            return False
        parts = [reps[i] for i in order]
        if any((not isinstance(p, tuple)) or p[0] != "q8_0" for p in parts):
            return False
        gu_qs = torch.cat([p[1] for p in parts], dim=0).contiguous()
        gu_sc = torch.cat([p[2] for p in parts], dim=0).contiguous()

    dreps = getattr(dn, "_xpu_reps", None)
    drep = dreps.get("_single") if isinstance(dreps, dict) else None
    if not (isinstance(drep, tuple) and drep[0] == "q8_0"):
        return False
    d_qs, d_sc = drep[1].contiguous(), drep[2].contiguous()

    w = getattr(sg, "weight", None)
    if w is None or w.dim() != 2 or w.shape[0] != 1:
        return False
    wg = (w[0] if w.dtype == torch.float16 else w[0].to(torch.float16)).contiguous()

    two_inter, hidden = gu_qs.shape
    inter_s = two_inter // 2
    # shape/dtype guards: a bad tensor never reaches the kernel.
    if gu_qs.dtype != torch.int8 or d_qs.dtype != torch.int8:
        return False
    if (hidden % 32) or (inter_s % 32) or (two_inter % 2):
        return False
    if tuple(d_qs.shape) != (hidden, inter_s):
        return False
    if tuple(gu_sc.shape) != (two_inter, hidden // 32):
        return False
    if tuple(d_sc.shape) != (hidden, inter_s // 32):
        return False
    if wg.shape[0] != hidden:
        return False
    return (gu_qs, gu_sc, d_qs, d_sc, wg, int(inter_s))


def _maybe_gguf_shared_q8(block, x):
    """One-dispatch fused GGUF Q8_0 shared expert (gate_up+silu+down+gate*sigmoid).
    Returns the shared-expert partial [M, hidden] fp16, or None to fall back to
    the unfused ``_forward_shared_experts`` path. Env-gated by
    SGL_XPU_GGUF_MOE_SHARED=1."""
    if not _GGUF_MOE_SHARED or x.device.type != "xpu" or x.dim() != 2:
        return None
    global _SHARED_Q8_OP
    if _SHARED_Q8_OP is None:
        _SHARED_Q8_OP = _load_shared_q8_op()
        if _SHARED_Q8_OP is None:
            return None
    cache = getattr(block, "_gguf_shared_q8_cache", None)
    if cache is None:
        cache = _gather_gguf_shared_q8(block)
        block._gguf_shared_q8_cache = cache
    if not cache:
        return None
    gu_qs, gu_sc, d_qs, d_sc, wg, inter_s = cache
    xf = x if x.dtype == torch.float16 else x.to(torch.float16)
    try:
        return _SHARED_Q8_OP(xf, gu_qs, gu_sc, d_qs, d_sc, wg, inter_s)
    except Exception:
        block._gguf_shared_q8_cache = False
        return None


# ═══════════════ GGUF FULL MoE fusion (topk + routed + shared -> 1 op) ═══════════════
_GGUF_MOE_FULL = bool(
    os.environ.get("SGL_XPU_GGUF_MOE_FULL")
    or os.environ.get("SGLANG_XPU_GGUF_MOE_FULL")
)
_GGUF_FULL_OP = None
_GGUF_FULL_NORM_OP = None


def _env_int(name, default):
    try:
        v = os.environ.get(name)
        return default if v is None or v == "" else int(v)
    except Exception:
        return default


# Largest decode batch the norm-fused GGUF MoE op is allowed to serve. Every
# kernel stage behind it (topk / up_q4k / shared_up_q8 / down / finalize) is
# already M-generic and the C++ side sizes all scratch from x.size(0), so this
# is purely a policy cap: it keeps the fused path away from prefill-sized M
# where the per-token GEMV shape stops paying off. Set to 1 to A/B the fusion.
_GGUF_MOE_FUSE_MAX_M = _env_int("SGL_XPU_GGUF_MOE_FUSE_MAX_M", 64)


def _load_gguf_moe_full_norm_op():
    """Norm-fused GGUF MoE op: absorbs the post-attention GemmaRMSNorm and the
    fp16 router GEMV, cutting two host dispatches per layer on the (host-bound)
    decode path. Returns None on older kernel builds."""
    try:
        import custom_esimd_kernels_sglang.custom_esimd_kernels  # noqa: F401
    except Exception:
        pass
    ns = getattr(torch.ops, "custom_esimd_kernels_sglang", None)
    if ns is None:
        return None
    return getattr(ns, "esimd_moe_forward_full_gguf_norm", None)


def _gguf_router_weight(gate):
    """The dense fp16 router weight [E, hidden] behind the MoE gate linear, or
    None when it cannot be expressed that way.

    Two layouts occur for the 35B GGUF checkpoint: ``ffn_gate_inp`` is stored
    unquantized (F32), so depending on the quant config the gate is either a
    plain ``UnquantizedLinearMethod`` (weight on the module) or a GGUF XPU
    linear holding a single fp16 resident rep. Handle both; the fp16 copy is
    cached on the module so the (possible) cast happens once, not per step.
    """
    w = getattr(gate, "_esimd_router_w16", None)
    if w is not None:
        return w
    reps = getattr(gate, "_xpu_reps", None)
    if reps is None:
        qm = getattr(gate, "quant_method", None)
        reps = getattr(qm, "_xpu_reps", None)
    if isinstance(reps, dict) and list(reps.keys()) == ["_single"]:
        rep = reps["_single"]
        w = rep[1] if rep[0] == "fp16" else None
    else:
        w = getattr(gate, "weight", None)
        w = getattr(w, "data", w)
    if w is None or w.dim() != 2:
        return None
    if w.dtype != torch.float16:
        w = w.to(torch.float16)
    w = w.contiguous()
    gate._esimd_router_w16 = w
    return w


def _load_gguf_moe_full_op():
    try:
        import custom_esimd_kernels_sglang.custom_esimd_kernels  # noqa: F401
    except Exception:
        pass
    ns = getattr(torch.ops, "custom_esimd_kernels_sglang", None)
    return getattr(ns, "esimd_moe_forward_full_gguf", None) if ns is not None else None


def _gather_gguf_moe_full(block):
    """Collect + validate everything the fused GGUF full-MoE op needs:
    routed Q4_K gate/up + Q5_K/Q6_K down reps (from experts.quant_method),
    the Q8_0 shared-expert reps, top_k, renorm, and dims. Returns a dict or
    False. Every layout/dtype check here keeps a bad tensor off the kernel."""
    experts = getattr(block, "experts", None)
    topk = getattr(block, "topk", None)
    gate = getattr(block, "gate", None)
    if experts is None or topk is None or gate is None:
        return False
    qm = getattr(experts, "quant_method", None)
    if qm is None or not hasattr(qm, "gate_ql") or not hasattr(qm, "down_ql"):
        return False

    # topk must be plain softmax + renorm (matches the fused kernel selection).
    tc = getattr(topk, "topk_config", None)
    if tc is None:
        return False
    if getattr(tc, "scoring_func", "softmax") != "softmax":
        return False
    if getattr(tc, "use_grouped_topk", False):
        return False
    if getattr(tc, "correction_bias", None) is not None:
        return False
    if getattr(tc, "custom_routing_function", None) is not None:
        return False
    top_k = getattr(tc, "top_k", None)
    if top_k is None:
        return False
    renorm = bool(getattr(tc, "renormalize", True))

    shared = _gather_gguf_shared_q8(block)
    if not shared:
        return False
    gu_qs, gu_sc, d_qs, d_sc, wg, inter_s = shared

    down_mn = getattr(qm, "down_mn", None)
    down_is_q6 = bool(getattr(qm, "_down_is_q6", False))
    # q5k needs down_mn; q6k ignores it (pass down_sc as a valid placeholder).
    if down_mn is None:
        if not down_is_q6:
            return False
        down_mn = qm.down_sc

    hidden = int(getattr(qm, "hidden"))
    inter = int(getattr(qm, "intermediate"))
    E = int(getattr(qm, "E"))
    # shared and routed share the same hidden; inter_s (shared) may differ.
    if wg.shape[0] != hidden:
        return False
    return {
        "gate": gate, "top_k": int(top_k), "renorm": renorm,
        "E": E, "hidden": hidden, "inter": inter, "inter_s": int(inter_s),
        "down_is_q6": down_is_q6,
        "gate_ql": qm.gate_ql, "gate_sc": qm.gate_sc, "gate_mn": qm.gate_mn,
        "up_ql": qm.up_ql, "up_sc": qm.up_sc, "up_mn": qm.up_mn,
        "down_ql": qm.down_ql, "down_qh": qm.down_qh_plain,
        "down_sc": qm.down_sc, "down_mn": down_mn,
        "gu_qs": gu_qs, "gu_sc": gu_sc, "d_qs": d_qs, "d_sc": d_sc, "wg": wg,
    }


def _maybe_gguf_moe_full_norm(block, h, residual, nw, eps):
    """Norm-fused GGUF MoE: GemmaRMSNorm(resadd) + router GEMV + experts in ONE
    dispatch. Returns ``(moe_out, new_residual)`` or None to fall back.

    ``residual`` is updated in place by the kernel (same as
    ``gemma_fused_add_rmsnorm``), so a None return must happen BEFORE the call.
    """
    if not _GGUF_MOE_FULL or h.device.type != "xpu" or h.dim() != 2:
        return None
    if not 1 <= h.shape[0] <= _GGUF_MOE_FUSE_MAX_M:
        return None
    global _GGUF_FULL_NORM_OP
    if _GGUF_FULL_NORM_OP is None:
        _GGUF_FULL_NORM_OP = _load_gguf_moe_full_norm_op()
        if _GGUF_FULL_NORM_OP is None:
            return None
    cache = getattr(block, "_gguf_moe_full_cache", None)
    if cache is None:
        cache = _gather_gguf_moe_full(block)
        block._gguf_moe_full_cache = cache
    if not cache:
        return None
    if getattr(block, "_gguf_moe_full_norm_off", False):
        return None
    rw = cache.get("router_w", None)
    if rw is None:
        rw = _gguf_router_weight(cache["gate"])
        if rw is None or rw.shape != (int(cache["E"]), int(cache["hidden"])):
            logger.warning(
                "[gguf_moe_full_norm] disabled: router weight %s, want (%s, %s)",
                None if rw is None else tuple(rw.shape),
                cache["E"], cache["hidden"])
            block._gguf_moe_full_norm_off = True
            return None
        cache["router_w"] = rw
    hf = h if h.dtype == torch.float16 else h.to(torch.float16)
    hf = hf if hf.is_contiguous() else hf.contiguous()
    if residual.dtype != torch.float16 or not residual.is_contiguous():
        if not getattr(block, "_gguf_norm_res_warned", False):
            block._gguf_norm_res_warned = True
            logger.warning("[gguf_moe_full_norm] disabled: residual dtype=%s contig=%s",
                           residual.dtype, residual.is_contiguous())
        return None
    try:
        out, res_out = _GGUF_FULL_NORM_OP(
            hf, residual, nw, float(eps), rw,
            cache["gate_ql"], cache["gate_sc"], cache["gate_mn"],
            cache["up_ql"], cache["up_sc"], cache["up_mn"],
            cache["down_ql"], cache["down_qh"], cache["down_sc"], cache["down_mn"],
            cache["gu_qs"], cache["gu_sc"], cache["d_qs"], cache["d_sc"], cache["wg"],
            int(cache["E"]), int(cache["top_k"]), int(cache["inter"]),
            int(cache["inter_s"]), bool(cache["down_is_q6"]), bool(cache["renorm"]),
        )
    except Exception as e:
        logger.warning("[gguf_moe_full_norm] disabled: kernel raised %r", e)
        block._gguf_moe_full_norm_off = True
        return None
    if not getattr(block, "_gguf_norm_ok_logged", False):
        block._gguf_norm_ok_logged = True
        logger.warning("[gguf_moe_full_norm] ACTIVE (norm + router folded into MoE op)")
    return out, res_out


def _maybe_gguf_moe_full(block, x):
    """One-dispatch fused GGUF MoE: router topk + routed experts (Q4_K/Q5_K) +
    Q8_0 shared expert -> the final [M, hidden] fp16 PARTIAL (routed + gate*shared
    summed; caller does the all_reduce). Returns None to fall back to the unfused
    router/shared path. Env-gated by SGL_XPU_GGUF_MOE_FULL=1."""
    if not _GGUF_MOE_FULL or x.device.type != "xpu" or x.dim() != 2 or x.shape[0] > 8:
        return None
    global _GGUF_FULL_OP
    if _GGUF_FULL_OP is None:
        _GGUF_FULL_OP = _load_gguf_moe_full_op()
        if _GGUF_FULL_OP is None:
            _moe_full_dbg("gguf_op_none")
            return None
    cache = getattr(block, "_gguf_moe_full_cache", None)
    if cache is None:
        cache = _gather_gguf_moe_full(block)
        block._gguf_moe_full_cache = cache
    if not cache:
        return None

    gate = cache["gate"]
    logits = _esimd_router_logits(gate, x)
    if logits is None:
        logits, _ = gate(x)
    if logits.dtype != torch.float16:
        logits = logits.to(torch.float16)
    logits = logits if logits.dim() == 2 else logits.reshape(x.shape[0], -1)
    logits = logits.contiguous()
    xf = x if x.dtype == torch.float16 else x.to(torch.float16)
    try:
        out = _GGUF_FULL_OP(
            xf, logits,
            cache["gate_ql"], cache["gate_sc"], cache["gate_mn"],
            cache["up_ql"], cache["up_sc"], cache["up_mn"],
            cache["down_ql"], cache["down_qh"], cache["down_sc"], cache["down_mn"],
            cache["gu_qs"], cache["gu_sc"], cache["d_qs"], cache["d_sc"], cache["wg"],
            int(cache["E"]), int(cache["top_k"]), int(cache["inter"]),
            int(cache["inter_s"]), bool(cache["down_is_q6"]), bool(cache["renorm"]),
        )
    except Exception as e:
        _moe_full_dbg("gguf_kernel_exc", err=repr(e))
        block._gguf_moe_full_cache = False
        return None
    if _MOE_FULL_DEBUG:
        _moe_full_dbg("SUCCESS", T=int(x.shape[0]), E=int(cache["E"]),
                      top_k=int(cache["top_k"]), out=tuple(out.shape), path="gguf_full")
    return out


def _maybe_esimd_moe_full(block, hidden_states: torch.Tensor):
    """One-dispatch decode MoE via moe_forward_full_v2. Returns the final
    [T, hidden] fp16 tensor (routed + gate*shared, already summed), or None to
    fall back to the unfused router/shared path.
    """
    if not _ESIMD_MOE_FULL:
        return None
    ops = _load_esimd_moe_full_op()
    if ops is None:
        _moe_full_dbg("op_none")
        return None
    x = hidden_states
    if x.device.type != "xpu" or x.dim() != 2 or x.shape[0] > 8:
        return None

    W = _gather_moe_full_weights(block, x)
    if W is None:
        return None
    gate = W["gate"]
    top_k = W["top_k"]; num_shared = W["num_shared"]; n_routed = W["n_routed"]
    gate_up_routed = W["gate_up_routed"]; s13 = W["s13"]
    shared_gate_up = W["shared_gate_up"]; ss13 = W["ss13"]
    down_routed = W["down_routed"]; s2 = W["s2"]
    shared_down = W["shared_down"]; ss2 = W["ss2"]
    sgw16 = W["sgw16"]
    x_in = x if x.dtype == torch.float16 else x.to(torch.float16)

    # BSZ=1 decode: prefer moe_forward_full_rtfused, which folds the router GEMV
    # + softmax-topk into ONE dispatch (needs the e4m3-quantized gate weight).
    # Fall back to moe_forward_full (fused down_finalize, separate router+topk)
    # when the quant isn't available, and to moe_forward_full_v2 for multi-token.
    rt = ops.get("rtfused")
    if x.shape[0] == 1 and rt is not None:
        wqsc = _esimd_router_wq_scale(gate, x)
        if wqsc is not None:
            wq, sc = wqsc
            try:
                out = rt(
                    x_in, wq, sc,
                    gate_up_routed, s13,
                    shared_gate_up, ss13,
                    down_routed, s2,
                    shared_down, ss2,
                    sgw16,
                    int(top_k), int(num_shared), int(n_routed),
                )
            except Exception as e:
                _moe_full_dbg("rtfused_exc", err=repr(e))
                return None
            if _MOE_FULL_DEBUG:
                _moe_full_dbg("SUCCESS", T=int(x.shape[0]), E=int(n_routed),
                              top_k=int(top_k), out=tuple(out.shape), path="rtfused")
            return out

    # --- router logits [T, E] fp16 (full/v2 do topk internally) ---
    logits = _esimd_router_logits(gate, x)
    if logits is None:
        logits, _ = gate(x)
    if logits.dtype != torch.float16:
        logits = logits.to(torch.float16)

    # BSZ=1 decode -> moe_forward_full (fused down_finalize, 5 internal kernels).
    # Multi-token or missing op -> moe_forward_full_v2 (7 internal kernels).
    op = ops.get("full") if (x.shape[0] == 1 and ops.get("full") is not None) else ops["v2"]
    try:
        out = op(
            x_in, logits,
            gate_up_routed, s13,
            shared_gate_up, ss13,
            down_routed, s2,
            shared_down, ss2,
            sgw16,
            int(top_k), int(num_shared), int(n_routed),
        )
    except Exception as e:
        _moe_full_dbg("kernel_exc", err=repr(e))
        return None
    if _MOE_FULL_DEBUG:
        _moe_full_dbg("SUCCESS", T=int(x.shape[0]), E=int(n_routed),
                      top_k=int(top_k), out=tuple(out.shape),
                      path=("full" if op is ops.get("full") else "v2"))
    return out


def _maybe_esimd_moe_full_norm(block, hidden_states, residual, norm_weight_folded, eps):
    """BSZ=1 decode MoE that ALSO folds the pre-MoE GemmaRMSNorm (residual add +
    rmsnorm) into the fused router kernel head. ``hidden_states`` is the pre-norm
    attention output (already all-reduced across the attn-TP group by the
    caller), ``residual`` the residual stream, ``norm_weight_folded`` the Gemma
    (1 + weight) in fp16, and ``eps`` the norm epsilon.

    Returns ``(moe_out, new_residual)`` where new_residual = hidden + residual,
    or None to fall back to the standard prepare_mlp + mlp path.
    """
    if not _ESIMD_MOE_FULL:
        return None
    ops = _load_esimd_moe_full_op()
    if ops is None:
        _moe_full_dbg("op_none")
        return None
    rtn = ops.get("rtfused_norm")
    if rtn is None:
        return None
    x = hidden_states
    if x.device.type != "xpu" or x.dim() != 2 or x.shape[0] != 1:
        return None
    if residual is None or residual.dim() != 2 or residual.shape != x.shape:
        return None

    W = _gather_moe_full_weights(block, x)
    if W is None:
        return None

    wqsc = _esimd_router_wq_scale(W["gate"], x)
    if wqsc is None:
        return None
    wq, sc = wqsc

    h_in = x if x.dtype == torch.float16 else x.to(torch.float16)
    h_in = h_in.contiguous()
    res = residual if residual.dtype == torch.float16 else residual.to(torch.float16)
    res = res if res.is_contiguous() else res.contiguous()
    try:
        out = rtn(
            h_in, res, norm_weight_folded, float(eps),
            wq, sc,
            W["gate_up_routed"], W["s13"],
            W["shared_gate_up"], W["ss13"],
            W["down_routed"], W["s2"],
            W["shared_down"], W["ss2"],
            W["sgw16"],
            W["top_k"], W["num_shared"], W["n_routed"],
        )
    except Exception as e:
        _moe_full_dbg("rtfused_norm_exc", err=repr(e))
        return None
    if not isinstance(out, (list, tuple)) or len(out) != 2:
        return None
    if _MOE_FULL_DEBUG:
        _moe_full_dbg("SUCCESS", T=1, E=W["n_routed"], top_k=W["top_k"],
                      out=tuple(out[0].shape), path="rtfused_norm")
    return out[0], out[1]


def can_fuse_shared_expert(
    config: PretrainedConfig,
    quant_config: Optional[QuantizationConfig],
) -> bool:
    """Whether the shared expert may be fused as an extra MoE expert (Qwen3.5 + Aiter).

    Caller must still gate on ``support_shared_expert_fusion`` and ``_use_aiter``.
    """
    if (
        get_global_server_args().disable_shared_experts_fusion is True
        or getattr(config, "shared_expert_intermediate_size", 0) <= 0
        or config.shared_expert_intermediate_size != config.moe_intermediate_size
        or get_moe_a2a_backend().is_deepep()
    ):
        return False

    # If the shared expert is excluded from quantization (stored as FP32 in the
    # checkpoint), fusing it into the quantized MoE weight tensor requires online
    # quantization which is not supported. Disable fusion in this case.
    if quant_config is not None:
        exclude_layers = getattr(quant_config, "exclude_layers", [])
        if any(
            "shared_expert" in layer
            and "shared_expert_gate" not in layer
            and not layer.startswith("mtp.")
            for layer in exclude_layers
        ):
            return False

    return True


class Qwen2MoeMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: Optional[QuantizationConfig] = None,
        reduce_results: bool = True,
        prefix: str = "",
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            prefix=add_prefix("down_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(
        self,
        x,
        should_allreduce_fusion: bool = False,
        use_reduce_scatter: bool = False,
    ):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(
            x, skip_all_reduce=should_allreduce_fusion or use_reduce_scatter
        )
        return x


class Qwen2MoeSparseMoeBlock(nn.Module):
    _prep_mlp_skip_seen = set()
    def __init__(
        self,
        layer_id: int,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
        is_nextn: bool = False,
        support_shared_expert_fusion: bool = False,
    ):
        super().__init__()
        self.tp_size = get_tensor_model_parallel_world_size()
        self.layer_id = layer_id
        self.alt_stream = alt_stream
        if self.tp_size > config.num_experts:
            raise ValueError(
                f"Tensor parallel size {self.tp_size} is greater than "
                f"the number of experts {config.num_experts}."
            )
        self.num_experts = config.num_experts
        self.num_shared_experts = 0
        self.num_fused_shared_experts = 0
        if hasattr(config, "n_shared_experts"):
            # config defines the number of shared experts
            self.num_shared_experts = config.n_shared_experts
        elif (
            hasattr(config, "shared_expert_intermediate_size")
            and config.shared_expert_intermediate_size > 0
        ):
            # n_shared_experts is not defined, but shared_expert_intermediate_size is defined, so we use 1 as the number of shared experts
            self.num_shared_experts = 1

        self.enable_shared_expert_fusion = False  # default to False
        if _use_aiter:
            # enable shared expert fusion when use aiter
            self.enable_shared_expert_fusion = (
                support_shared_expert_fusion
                and can_fuse_shared_expert(config, quant_config)
            )
        if self.enable_shared_expert_fusion:
            self.num_fused_shared_experts = self.num_shared_experts

        self.topk = TopK(
            top_k=config.num_experts_per_tok,
            renormalize=config.norm_topk_prob,
            layer_id=layer_id,
        )

        self.experts = get_moe_impl_class(quant_config)(
            layer_id=self.layer_id,
            top_k=(
                config.num_experts_per_tok
                if not self.enable_shared_expert_fusion
                else config.num_experts_per_tok + self.num_fused_shared_experts
            ),
            num_experts=(
                config.num_experts + get_global_server_args().ep_num_redundant_experts
                if not self.enable_shared_expert_fusion
                else config.num_experts
                + get_global_server_args().ep_num_redundant_experts
                + self.num_fused_shared_experts
            ),
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            quant_config=quant_config,
            prefix=add_prefix("experts", prefix),
            routing_method_type=RoutingMethodType.RenormalizeNaive,
            num_fused_shared_experts=self.num_fused_shared_experts,
        )

        self.gate = ReplicatedLinear(
            config.hidden_size,
            config.num_experts,
            bias=False,
            quant_config=None,
            prefix=add_prefix("gate", prefix),
        )
        # When enable_shared_expert_fusion, the shared expert runs inside the MoE kernel
        # (via _append_shared_to_topk_output); a separate shared_expert MLP would
        # double-count. If fusion is off (num_fused_shared_experts == 0), keep shared_expert.
        if (
            config.shared_expert_intermediate_size > 0
            and not self.enable_shared_expert_fusion
        ):
            self.shared_expert = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.shared_expert_intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                reduce_results=False,
                prefix=add_prefix("shared_expert", prefix),
                **(
                    dict(tp_rank=0, tp_size=1)
                    if (
                        get_moe_a2a_backend().is_deepep()
                        or get_moe_a2a_backend().is_flashinfer()
                    )
                    else {}
                ),
            )
        else:
            self.shared_expert = None
        if _is_cpu and _is_cpu_amx_available:
            self.shared_expert_gate = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
                quant_config=None,
                prefix=add_prefix("shared_expert_gate", prefix),
            )
        else:
            self.shared_expert_gate = torch.nn.Linear(config.hidden_size, 1, bias=False)

        if get_moe_a2a_backend().is_deepep():
            # TODO: we will support tp < ep in the future
            self.ep_size = get_moe_expert_parallel_world_size()
            self.num_experts = (
                config.num_experts + get_global_server_args().ep_num_redundant_experts
            )
            self.top_k = config.num_experts_per_tok
        self.is_nextn = is_nextn

    def get_moe_weights(self):
        return [
            x.data
            for name, x in self.experts.named_parameters()
            if name not in ["correction_bias"]
            and filter_moe_weight_param_global_expert(
                name, x, self.experts.num_local_experts
            )
        ]

    def _get_shared_expert_weights(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Return sigmoid(shared_expert_gate) for fused shared expert weights."""
        if not self.enable_shared_expert_fusion or self.shared_expert_gate is None:
            return None
        shared_out = self.shared_expert_gate(hidden_states)
        shared_logits = shared_out[0] if isinstance(shared_out, tuple) else shared_out
        w = F.sigmoid(shared_logits)
        # This block runs only on the AMD AITER shared_expert_fusion path
        # Allreduce-EP path: the fused shared expert occupies a single global
        # slot loaded onto every EP rank (see FusedMoE.__init__: num_shared_slots
        # == num_fused_shared_experts when not is_deepep_class_backend()). Every
        # rank therefore computes the same full shared output, and the
        # post-experts all_reduce sums it ep_size times. Pre-scale the per-token
        # routing weight by 1/ep_size to cancel this, mirroring DeepSeek-V2's
        # fused_shared_experts_scaling_factor pattern.
        moe_ep_size = get_moe_expert_parallel_world_size()
        if moe_ep_size > 1 and not is_deepep_class_backend():
            w = w / float(moe_ep_size)
        return w

    def _append_shared_to_topk_output(
        self,
        topk_output: StandardTopKOutput,
        hidden_states: torch.Tensor,
    ) -> StandardTopKOutput:
        """Append shared expert ids and weights to topk output before fused MoE."""
        if not self.enable_shared_expert_fusion:
            return topk_output
        shared_weights = self._get_shared_expert_weights(hidden_states)
        if shared_weights is None:
            return topk_output

        from sglang.srt.layers.moe.moe_runner.triton_utils.fused_moe_triton_kernels import (
            fused_append_shared_experts_with_weights,
        )

        fused_topk_ids, fused_topk_weights = fused_append_shared_experts_with_weights(
            topk_output.topk_ids,
            topk_output.topk_weights,
            shared_weights,
            self.num_fused_shared_experts,
            N=self.num_experts,
        )
        return StandardTopKOutput(
            topk_weights=fused_topk_weights,
            topk_ids=fused_topk_ids,
            router_logits=topk_output.router_logits,
        )

    def _forward_shared_experts(self, hidden_states: torch.Tensor):
        shared_output = None
        if self.shared_expert is not None:
            shared_output = self.shared_expert(hidden_states)
            if self.shared_expert_gate is not None:
                if use_intel_amx_backend(self.shared_expert_gate):
                    shared_output = torch.ops.sgl_kernel.fused_linear_sigmoid_mul(
                        hidden_states,
                        self.shared_expert_gate.weight,
                        self.shared_expert_gate.bias,
                        True,
                        shared_output,
                    )
                else:
                    shared_output = (
                        F.sigmoid(self.shared_expert_gate(hidden_states))
                        * shared_output
                    )

        return shared_output

    def _forward_deepep(self, hidden_states: torch.Tensor, forward_batch: ForwardBatch):
        enable_dual_stream = (
            is_npu()
            and envs.SGLANG_NPU_USE_MULTI_STREAM.get()
            and forward_batch.forward_mode.is_cuda_graph()
        )
        shared_output = None
        if hidden_states.shape[0] > 0:
            # router_logits: (num_tokens, n_experts)
            router_logits, _ = self.gate(hidden_states)
            if enable_dual_stream:
                shared_output = shared_expert_on_independent_stream(
                    hidden_states.clone(), self._forward_shared_experts
                )
            else:
                shared_output = self._forward_shared_experts(hidden_states)
            topk_output = self.topk(
                hidden_states,
                router_logits,
                num_token_non_padded=forward_batch.num_token_non_padded,
                expert_location_dispatch_info=(
                    ExpertLocationDispatchInfo.init_new(
                        layer_id=self.layer_id,
                    )
                    if not self.is_nextn
                    else None
                ),
            )
        else:
            topk_output = self.topk.empty_topk_output(hidden_states.device)
        final_hidden_states = self.experts(
            hidden_states=hidden_states,
            topk_output=topk_output,
        )
        if enable_dual_stream:
            wait_share_stream()

        if shared_output is not None:
            final_hidden_states.add_(shared_output)

        return final_hidden_states

    def _forward_router_experts(self, hidden_states: torch.Tensor):
        # router_logits: (num_tokens, n_experts)
        router_logits = _esimd_router_logits(self.gate, hidden_states)
        if router_logits is None:
            router_logits, _ = self.gate(hidden_states)
        topk_output = self.topk(hidden_states, router_logits)
        if self.enable_shared_expert_fusion and TopKOutputChecker.format_is_standard(
            topk_output
        ):
            topk_output = self._append_shared_to_topk_output(topk_output, hidden_states)
        return self.experts(hidden_states, topk_output)

    def forward_normal_dual_stream(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        current_stream = torch.cuda.current_stream()
        self.alt_stream.wait_stream(current_stream)
        shared_output = self._forward_shared_experts(hidden_states.clone())

        # ===== TO BE REFACTORED ====
        # Shared-add overlap (SGLANG_OPT_LORA_SHARED_ADD_OVERLAP): hand the add to the LoRA
        # MoE dispatch so it overlaps the down-LoRA shrink on the alt stream.
        staged = False
        if shared_output is not None and _SGLANG_EXPERIMENTAL_LORA_OPTI:
            from sglang.srt.lora.trtllm_lora_temp.shared_add_overlap import (
                shared_add_overlap_enabled,
                stage_shared_expert_add,
                unstage_shared_expert_add,
            )

            if shared_add_overlap_enabled():
                stage_shared_expert_add(shared_output, current_stream)
                staged = True
        # ===== END TO BE REFACTORED ====

        with torch.cuda.stream(self.alt_stream):
            router_output = self._forward_router_experts(hidden_states)

        current_stream.wait_stream(self.alt_stream)

        if staged and unstage_shared_expert_add() is None:
            # The dispatch consumed the staging (add already enqueued); skip the caller's add.
            shared_output = None

        return router_output, shared_output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: Optional[ForwardBatch] = None,
        use_reduce_scatter: bool = False,
        should_allreduce_fusion: bool = False,
    ) -> torch.Tensor:
        num_tokens, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)

        if get_moe_a2a_backend().is_deepep():
            return self._forward_deepep(hidden_states, forward_batch)

        if hidden_states.shape[0] == 0:
            # M=0 guard for idle DP ranks: skip shared_experts and gate
            # (which crash on empty tensors in FP4 GEMM), but still call
            # self.experts() to participate in alltoall collective.
            shared_output = None
            topk_output = self.topk.empty_topk_output(hidden_states.device)
            final_hidden_states = self.experts(hidden_states, topk_output)
        elif self.alt_stream is not None and get_is_capture_mode():
            final_hidden_states, shared_output = self.forward_normal_dual_stream(
                hidden_states
            )
        else:
            fused_full = _maybe_esimd_moe_full(self, hidden_states)
            if fused_full is None:
                fused_full = _maybe_gguf_moe_full(self, hidden_states)
            if fused_full is not None:
                # Fused op returns routed + gate*shared already summed → skip the
                # separate shared path and its add below.
                final_hidden_states = fused_full
                shared_output = None
            else:
                shared_output = _maybe_gguf_shared_q8(self, hidden_states)
                if shared_output is None:
                    shared_output = self._forward_shared_experts(hidden_states)
                _nan_probe_moe(
                    "moe_shared_out",
                    shared_output,
                    layer_id=getattr(self, "layer_id", None),
                    forward_batch=forward_batch,
                )
                final_hidden_states = self._forward_router_experts(hidden_states)
                _nan_probe_moe(
                    "moe_routed_out",
                    final_hidden_states,
                    layer_id=getattr(self, "layer_id", None),
                    forward_batch=forward_batch,
                )

        if shared_output is not None:
            final_hidden_states += shared_output
        if (
            self.tp_size > 1
            and not should_skip_post_experts_all_reduce(
                is_tp_path=True,
                use_reduce_scatter=use_reduce_scatter,
                should_allreduce_fusion=should_allreduce_fusion,
            )
            and not get_moe_a2a_backend().is_flashinfer()
        ):
            final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

        # Debug removed - was causing issues during CUDA graph capture

        return final_hidden_states.view(num_tokens, hidden_dim)

    def esimd_prepare_mlp_moe(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        norm_module,
        forward_batch,
        use_reduce_scatter: bool,
        should_allreduce_fusion: bool,
    ):
        """Phase 3: fused replacement for ``prepare_mlp`` + ``mlp.forward`` on the
        plain-TP single-token decode path.

        Baseline does, in two dispatched pieces:
          1. prepare_mlp: all-reduce(attn output) then GemmaRMSNorm(resadd+norm)
          2. mlp.forward: MoE (router+experts+shared) then all-reduce(MoE output)

        Here the GemmaRMSNorm resadd+norm is folded into the fused MoE router
        kernel head (``moe_forward_full_rtfused_norm``), removing the standalone
        ``gemma_fused_add_rmsnorm`` dispatch and returning the new residual so no
        separate python add is needed. The two all-reduces are reproduced
        explicitly (identical to the baseline collectives).

        Returns ``(hidden_out, residual_out)`` on success, or ``None`` to signal
        the caller to run the standard ``prepare_mlp`` + ``mlp`` path. All guards
        that could invalidate the manual collectives are checked BEFORE any
        communication, so a ``None`` return never leaves a stray all-reduce.
        """
        def _skip(why):
            # One-shot per reason: tells us which guard blocks the fusion
            # without spamming a 40-layer x N-token decode loop.
            seen = Qwen2MoeSparseMoeBlock._prep_mlp_skip_seen
            if why not in seen:
                seen.add(why)
                logger.warning("[esimd_prepare_mlp_moe] disabled: %s", why)
            return None

        if not _ESIMD_MOE_FULL and not _GGUF_MOE_FULL:
            return _skip("no MOE_FULL env")
        ops = _load_esimd_moe_full_op()
        # Either the fp8 ESIMD norm-fused op or the GGUF norm-fused op must be
        # available, otherwise there is nothing to fold the norm into.
        have_fp8 = (
            _ESIMD_MOE_FULL and ops is not None and ops.get("rtfused_norm") is not None
        )
        have_gguf = _GGUF_MOE_FULL and _load_gguf_moe_full_norm_op() is not None
        if not have_fp8 and not have_gguf:
            return _skip("no norm-fused op (fp8=%s gguf=%s)" % (have_fp8, have_gguf))
        if not (
            hidden_states.device.type == "xpu"
            and forward_batch is not None
            and forward_batch.forward_mode.is_decode()
        ):
            return _skip("not xpu decode")
        # M>1 is only serviceable by the GGUF norm-fused op; the fp8 twin
        # (_maybe_esimd_moe_full_norm) is still single-token. Without GGUF we
        # keep the original M==1 gate so fp8-only setups are untouched.
        max_m = _GGUF_MOE_FUSE_MAX_M if have_gguf else 1
        if hidden_states.dim() != 2 or not (1 <= hidden_states.shape[0] <= max_m):
            return _skip("shape %s" % (tuple(hidden_states.shape),))
        if residual is None or residual.shape != hidden_states.shape:
            return _skip("residual mismatch")
        # Only the plain-TP path (no input-scatter, no DP-attention, no
        # all-reduce fusion, no reduce-scatter) matches the manual collectives.
        if use_reduce_scatter or should_allreduce_fusion:
            return _skip("reduce_scatter=%s allreduce_fusion=%s"
                         % (use_reduce_scatter, should_allreduce_fusion))
        if getattr(hidden_states, "_sglang_needs_allreduce_fusion", False):
            return _skip("needs_allreduce_fusion flag")
        try:
            from sglang.srt.layers.communicator import get_attn_tp_context
            from sglang.srt.layers.dp_attention import get_attention_dp_size

            if get_attn_tp_context().input_scattered:
                return _skip("input_scattered")
            if get_attention_dp_size() != 1:
                return _skip("dp_size != 1")
        except Exception as e:
            return _skip("ctx probe exc %r" % (e,))
        # Fold GemmaRMSNorm (1 + weight) once per layer.
        nw = getattr(norm_module, "_esimd_moe_nw", None)
        if nw is None:
            w = getattr(norm_module, "weight", None)
            eps = getattr(norm_module, "variance_epsilon", None)
            if w is None or eps is None:
                return None
            nw = (w.data.to(torch.float32) + 1.0).to(torch.float16).contiguous()
            norm_module._esimd_moe_nw = nw
        eps = float(norm_module.variance_epsilon)

        # ── Commit: reproduce prepare_mlp's attention-output all-reduce ──
        h_ar = attention_tensor_model_parallel_all_reduce(hidden_states)

        fused = None
        if have_gguf:
            fused = _maybe_gguf_moe_full_norm(self, h_ar, residual, nw, eps)
        if fused is None and have_fp8:
            fused = _maybe_esimd_moe_full_norm(self, h_ar, residual, nw, eps)
        if fused is not None:
            moe_out, new_residual = fused
            # Reproduce mlp.forward's post-experts all-reduce (kernel is per-rank).
            if self.tp_size > 1 and not get_moe_a2a_backend().is_flashinfer():
                moe_out = tensor_model_parallel_all_reduce(moe_out)
            return moe_out, new_residual

        # Fallback (kernel unavailable / guard miss inside the MoE helper): run
        # the baseline norm on the already-reduced hidden, then the standard MoE
        # forward (which performs its own post-experts all-reduce). This keeps
        # correctness without duplicating the attention all-reduce above.
        normed, new_residual = norm_module(h_ar, residual)
        moe_out = self.forward(
            normed, forward_batch, use_reduce_scatter, should_allreduce_fusion
        )
        return moe_out, new_residual


class Qwen2MoeAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        qkv_bias: int = True,
        quant_config: Optional[QuantizationConfig] = None,
        dual_chunk_attention_config: Optional[dict[str, Any]] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size

        attn_tp_rank = get_attention_tp_rank()
        attn_tp_size = get_attention_tp_size()

        self.total_num_heads = num_heads
        assert self.total_num_heads % attn_tp_size == 0
        self.num_heads = self.total_num_heads // attn_tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= attn_tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False,
            prefix=add_prefix("o_proj", prefix),
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("attn", prefix),
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2MoeDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        start_layer: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.start_layer = start_layer
        rope_theta, rope_scaling = get_rope_config(config)
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        qkv_bias = getattr(config, "qkv_bias", True)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )
        self.self_attn = Qwen2MoeAttention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qkv_bias=qkv_bias,
            prefix=add_prefix("self_attn", prefix),
        )

        self.layer_id = layer_id

        self.attn_tp_size = get_attention_tp_size()
        self.attn_tp_rank = get_attention_tp_rank()

        # Qwen2MoE all layers are sparse and have no nextn now
        self.is_layer_sparse = True
        is_previous_layer_sparse = True
        is_next_layer_sparse = True

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=self.is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        if self.is_layer_sparse:
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix),
            )
        else:
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix),
            )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
        captured_last_layer_outputs: Optional[List[torch.Tensor]] = None,
        **kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        hidden_states, residual = (
            self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs=captured_last_layer_outputs,
                **kwargs,
            )
        )

        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        # For DP with padding, reduce scatter can be used instead of all-reduce.
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        hidden_states = self.mlp(hidden_states, forward_batch, use_reduce_scatter)

        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )

        return hidden_states, residual


class Qwen2MoeModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2MoeDecoderLayer,
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.pp_group = get_pp_group()

        self.moe_dp_size = get_moe_data_parallel_world_size()
        self.attn_cp_size = get_attn_context_model_parallel_world_size()

        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                use_attn_tp_group=is_dp_attention_enabled(),
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Use the provided decoder layer type or default to Qwen2MoeDecoderLayer
        decoder_layer_type = decoder_layer_type or Qwen2MoeDecoderLayer
        pp_start_layer, _ = get_pp_indices(
            config.num_hidden_layers,
            self.pp_group.rank_in_group,
            self.pp_group.world_size,
        )
        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: decoder_layer_type(
                layer_id=idx,
                start_layer=pp_start_layer,
                config=config,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )
        if self.pp_group.is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer(return_tuple=True)

        # For EAGLE3 support
        self.layers_to_capture = []

    def set_eagle3_layers_to_capture(self, layers_to_capture: List[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            residual = None
        else:
            assert pp_proxy_tensors is not None
            hidden_states = pp_proxy_tensors["hidden_states"]
            residual = pp_proxy_tensors["residual"]

        if (
            is_prefill_context_parallel_enabled()
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            if self.pp_group.is_first_rank:
                hidden_states = cp_split_and_rebuild_data(forward_batch, hidden_states)
            positions = cp_split_and_rebuild_position(forward_batch, positions)

        aux_hidden_states = []
        if forward_batch.can_run_tbo:
            hidden_states, residual = model_forward_maybe_tbo(
                layers=self.layers,
                enable_tbo=True,
                input_data_scatter_mode=ScatterMode.model_input_output(),
                positions=positions,
                forward_batch=forward_batch,
                hidden_states=hidden_states,
                residual=residual,
            )
        else:
            for i in range(self.start_layer, self.end_layer):
                ctx = (
                    nullcontext()
                    if not get_global_server_args().disable_piecewise_cuda_graph
                    else get_global_expert_distribution_recorder().with_current_layer(i)
                )
                with ctx:
                    layer = self.layers[i]
                    hidden_states, residual = layer(
                        positions,
                        hidden_states,
                        forward_batch,
                        residual,
                        captured_last_layer_outputs=(
                            aux_hidden_states
                            if getattr(layer, "_is_layer_to_capture", False)
                            else None
                        ),
                    )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )
        else:
            if hidden_states.shape[0] != 0:
                if residual is None:
                    hidden_states = self.norm(hidden_states)
                else:
                    hidden_states, _ = self.norm(hidden_states, residual)

        if (
            self.pp_group.is_last_rank
            and is_prefill_context_parallel_enabled()
            and forward_batch.forward_mode.is_context_parallel_extend()
            and forward_batch.attn_cp_metadata is not None
        ):
            hidden_states = cp_all_gather_rerange_output(
                hidden_states,
                self.attn_cp_size,
                forward_batch,
                torch.cuda.current_stream(),
            )

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states


class Qwen2MoeForCausalLM(nn.Module):
    fall_back_to_pt_during_load = False

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.pp_group = get_pp_group()
        self.config = config
        self.quant_config = quant_config
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        self.model = Qwen2MoeModel(
            config,
            quant_config,
            prefix=add_prefix("model", prefix),
            alt_stream=alt_stream,
        )
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=add_prefix("lm_head", prefix),
            use_attn_tp_group=get_global_server_args().enable_dp_lm_head,
        )
        self.logits_processor = LogitsProcessor(config)
        # For EAGLE3 support
        self.capture_aux_hidden_states = False

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states
        if self.pp_group.is_last_rank:
            return self.logits_processor(
                input_ids, hidden_states, self.lm_head, forward_batch, aux_hidden_states
            )
        else:
            return hidden_states

    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        start, end = split_interval
        # embed
        if start == 0:
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            else:
                forward_batch.hidden_states = input_embeds

        # decoder layer
        for i in range(start, end):
            with get_global_expert_distribution_recorder().with_current_layer(i):
                layer = self.model.layers[i]
                forward_batch.hidden_states, forward_batch.residual = layer(
                    positions,
                    forward_batch.hidden_states,
                    forward_batch,
                    forward_batch.residual,
                )

        if end == self.model.config.num_hidden_layers:
            # norm
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # logits process
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result

    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self.model, "start_layer")
                and (
                    layer_id < self.model.start_layer
                    or layer_id >= self.model.end_layer
                )
            ):
                continue
            if "rotary_emb.inv_freq" in name:
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                # We have mlp.experts[0].gate_proj in the checkpoint.
                # Since we handle the experts below in expert_params_mapping,
                # we need to skip here BEFORE we update the name, otherwise
                # name will be updated to mlp.experts[0].gate_up_proj, which
                # will then be updated below in expert_params_mapping
                # for mlp.experts[0].gate_gate_up_proj, which breaks load.
                if "mlp.experts" in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    name = name.replace(weight_name, param_name)
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(
                        param,
                        loaded_weight,
                        name,
                        shard_id=shard_id,
                        expert_id=expert_id,
                    )
                    break
                else:
                    # Skip loading extra bias for GPTQ models.
                    if name.endswith(".bias") and name not in params_dict:
                        continue
                    if name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )

    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.set_eagle3_layers_to_capture(
                [
                    2,
                    num_layers // 2,
                    num_layers - 3,
                ]
            )  # Specific layers for EAGLE3 support
        else:
            self.model.set_eagle3_layers_to_capture([val + 1 for val in layer_ids])


EntryClass = Qwen2MoeForCausalLM
