# Copyright 2025 Qwen Team
# Copyright 2025 SGLang Team
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
"""Inference-only Qwen3.5 model and Qwen3.5 MoE model compatible with HuggingFace weights."""

import logging
import os
from functools import lru_cache
from typing import Iterable, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
import triton

from sglang.jit_kernel.triton.gdn_fused_proj import (
    fused_qkvzba_split_reshape_cat_contiguous,
)

# Configs
from sglang.srt.configs.qwen3_5 import (
    Qwen3_5Config,
    Qwen3_5MoeConfig,
    Qwen3_5TextConfig,
)

# Distributed
from sglang.srt.distributed import get_pp_group
from sglang.srt.eplb.expert_distribution import get_global_expert_distribution_recorder
from sglang.srt.eplb.expert_location import ModelConfigForExpertLocation

# Layers - Attention
from sglang.srt.layers.attention.fla.layernorm_gated import RMSNorm as RMSNormGated
from sglang.srt.layers.attention.mamba.mamba import mamba_v2_sharded_weight_loader
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
from sglang.srt.layers.dp_attention import (
    get_attention_tp_rank,
    get_attention_tp_size,
    is_dp_attention_enabled,
)

# Layers - Others
from sglang.srt.layers.layernorm import GemmaRMSNorm

# Layers - Linear
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.moe.fused_moe_triton.layer import FusedMoE
from sglang.srt.layers.parameter import (
    BlockQuantScaleParameter,
    PerTensorScaleParameter,
)
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.layers.radix_linear_attention import RadixLinearAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import (
    default_weight_loader,
    sharded_weight_loader,
)
from sglang.srt.models.qwen2_moe import Qwen2MoeMLP, Qwen2MoeSparseMoeBlock

# Models
from sglang.srt.models.qwen3_vl import Qwen3VLForConditionalGeneration
from sglang.srt.models.utils import fused_qk_gemma_rmsnorm
from sglang.srt.server_args import get_global_server_args

# Utils
from sglang.srt.utils import (
    LazyValue,
    add_prefix,
    cpu_has_amx_support,
    get_bool_env_var,
    is_cpu,
    is_cuda,
    is_gfx95_supported,
    is_hip,
    is_npu,
    is_xpu,
    make_layers,
    set_weight_attrs,
)
from sglang.srt.utils.hf_transformers_utils import get_processor, get_rope_config

logger = logging.getLogger(__name__)

# ── NaN/Inf probe (env-gated: SGLANG_NAN_PROBE=1) ───────────────────────────
# Debug instrumentation to locate the first non-finite tensor in the forward
# pass (symptom: "！！！" garbage generations, more frequent at batch>8/16).
# Logs the ORIGIN stage (input clean -> output NaN/Inf) with layer id, forward
# mode (EXTEND/DECODE), and token count so we can tell which kernel/batch size
# introduces the NaN. Zero overhead when SGLANG_NAN_PROBE is unset.
import os as _os_np

_NAN_PROBE = _os_np.environ.get("SGLANG_NAN_PROBE", "0") == "1"

_nan_probe_state = {"origin_found": False, "fwd": -1}


def _np_bad(t):
    """Return (n_nan, n_inf) as python ints; (0, 0) if finite / not applicable."""
    if t is None or not torch.is_tensor(t) or t.numel() == 0:
        return (0, 0)
    if not t.dtype.is_floating_point:
        return (0, 0)
    if bool(torch.isfinite(t).all()):
        return (0, 0)
    return (int(torch.isnan(t).sum().item()), int(torch.isinf(t).sum().item()))


def _nan_probe(tag, t, layer_id=None, forward_batch=None, residual=None):
    if not _NAN_PROBE:
        return
    try:
        nan_h, inf_h = _np_bad(t)
        nan_r, inf_r = _np_bad(residual)
        if (nan_h + inf_h + nan_r + inf_r) == 0:
            return
        mode = "?"
        if forward_batch is not None:
            fm = getattr(forward_batch, "forward_mode", None)
            mode = getattr(fm, "name", str(fm)) if fm is not None else "?"
        ntok = int(t.shape[0]) if (torch.is_tensor(t) and t.dim() > 0) else -1
        first = not _nan_probe_state["origin_found"]
        if first:
            _nan_probe_state["origin_found"] = True
        extra = ""
        if tag == "embed_out" and torch.is_tensor(t) and t.dim() == 2:
            bad = ~torch.isfinite(t)
            rows = bad.any(dim=1).nonzero(as_tuple=False).flatten()
            full = int(bad.all(dim=1).sum().item())
            ids = getattr(forward_batch, "input_ids", None)
            idh = []
            if torch.is_tensor(ids) and ids.numel() >= t.shape[0]:
                idh = (
                    ids.flatten()[: t.shape[0]].index_select(0, rows[:16]).tolist()
                )
            extra = (
                " | bad_rows=%d full_nan_rows=%d rows_head=%s ids_head=%s"
                " ids_shape=%s elem_head=%s"
                % (
                    int(rows.numel()),
                    full,
                    rows[:16].tolist(),
                    idh,
                    (tuple(ids.shape) if torch.is_tensor(ids) else None),
                    bad.nonzero(as_tuple=False)[:8].tolist(),
                )
            )
        logger.error(
            "[NANPROBE]%s fwd=%d tag=%s layer=%s mode=%s ntok=%d | "
            "hidden nan=%d inf=%d | residual nan=%d inf=%d%s",
            " ORIGIN" if first else "",
            _nan_probe_state["fwd"],
            tag,
            layer_id,
            mode,
            ntok,
            nan_h,
            inf_h,
            nan_r,
            inf_r,
            extra,
        )
    except Exception:
        pass


def _nan_probe_new_forward():
    if not _NAN_PROBE:
        return
    _nan_probe_state["origin_found"] = False
    _nan_probe_state["fwd"] += 1


_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_is_gfx95 = is_gfx95_supported()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_hip_use_alt_stream = get_bool_env_var("SGLANG_ALT_STREAM") and _is_hip
_gdn_use_alt_stream = (
    get_bool_env_var("SGLANG_GDN_QKVZ_BA_ALT_STREAM", "False") and _hip_use_alt_stream
)
_qknorm_use_alt_stream = (
    get_bool_env_var("SGLANG_QK_NORM_ALT_STREAM", "False") and _hip_use_alt_stream
)
_is_amx_available = cpu_has_amx_support()
# XPU ESIMD fast-path gates. These SGL_XPU_* env vars are set once at launch and
# are constant for the process lifetime, so cache them at import instead of
# re-reading os.environ on every decode-step call (30-60 reads/step otherwise).
_XPU_GDN_NORM_GEMV = os.environ.get("SGL_XPU_GDN_NORM_GEMV", "0") == "1"
_XPU_GDN_RESADD_NORM = os.environ.get("SGL_XPU_GDN_RESADD_NORM", "0") == "1"
_XPU_GDN_FAST_PATH = os.environ.get("SGLANG_XPU_GDN_FAST_PATH", "0") == "1"
_XPU_GDN_ESIMD = os.environ.get("SGL_XPU_GDN_ESIMD", "0") == "1"
_XPU_FA_ESIMD_QKV = os.environ.get("SGL_XPU_FA_ESIMD_QKV", "0") == "1"
# Fuse the GDN in_proj_qkvz + in_proj_ba GEMVs (same hidden input, both fp8) into a
# single esimd_gemv_fp8_pert_fused2 call, removing one GEMM dispatch + its glue per
# GDN layer (~30/step). Decode-only (M=1). Falls back to two separate linears.
_XPU_GDN_INPROJ_FUSED2 = os.environ.get("SGL_XPU_GDN_INPROJ_FUSED2", "0") == "1"
# Phase 5b: fuse the full-attention layers' input_layernorm (GemmaRMSNorm resadd +
# rmsnorm) into the qkv_proj GEMV via the single deployed
# esimd_resadd_norm_gemv2_fp8_pert op (qkv as matrix-0, a 1-row dummy as matrix-1),
# with kernel-written new_residual. Removes the per-full-attn-layer
# gemma_fused_add_rmsnorm dispatch (~10/step). Decode single-token, fp8 qkv only.
_XPU_FA_RESADD_NORM = os.environ.get("SGL_XPU_FA_RESADD_NORM", "0") == "1"

# Phase 5c (GGUF): the Phase-5b resadd-norm fusions above are fp8-only, but the
# 35B GGUF build runs its attention projections as ESIMD q8_0 GEMVs, so they
# never fire. `esimd_resadd_norm_gemv_q8_ba` is the GGUF counterpart: it folds
# GemmaRMSNorm(input_layernorm) plus the q8_0 in_proj/qkv GEMV plus (for GDN)
# the unquantised fp16 in_proj_ba GEMV into ONE op call. Decode at bs=1 is
# host-bound (~13-25us of torch dispatch per call vs ~4us of actual enqueue), so
# collapsing 70 op calls/step is worth far more than any kernel-level tuning.
# On by default whenever the GGUF MoE full fusion is on; SGL_XPU_GGUF_RESADD_NORM=0
# forces the unfused fallback.
_XPU_GGUF_RESADD_NORM = (
    os.environ.get("SGL_XPU_GGUF_MOE_FULL", "0") == "1"
    and os.environ.get("SGL_XPU_GGUF_RESADD_NORM", "1") == "1"
)
# DEPRECATED. First iteration of the GGUF norm+proj fusion, superseded by
# _gguf_norm_gemv() / _XPU_GGUF_RESADD_NORM above. It drove a single-matrix
# `esimd_resadd_norm_gemv_q8_0` op that was never landed in
# custom-esimd-kernels, so the path always ImportErrors and falls back; the
# shipped design instead uses `esimd_resadd_norm_gemv_q8_ba`, which folds the
# unquantised fp16 in_proj_ba GEMV into the SAME op call (one launch instead of
# two) and keeps the q/k/v-merged rep. Kept for reference behind its own opt-in
# flag so it can never shadow the supported path; remove once the q8_0-only
# variant is confirmed unnecessary.
_XPU_GGUF_RESADD_NORM_LEGACY = (
    os.environ.get("SGL_XPU_GGUF_RESADD_NORM_LEGACY", "0") == "1"
)


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


# Largest decode batch the GGUF resadd-norm fusion will handle.
#
# The kernel (Moe_norm_q8_kernel) has always been M-generic: its grid is
# M*blocks*K_SPLIT and every access is indexed off `token = gid / blocks`. The
# bs=1 restriction lived purely in this file, because the cached output buffers
# were allocated as [1, N]. Real serving decode batches are almost never 1 (a
# BFCL multi_turn run spends 87% of its decode steps at batch > 8), so that
# restriction meant the fusion was bypassed for the large majority of steps.
#
# Buffers are now allocated per distinct M, so the only reason for an upper
# bound is to stop an unbounded prefill-sized M from allocating one buffer set
# per batch size ever seen. Set to 1 to restore the old single-token behaviour.
_GGUF_FUSE_MAX_M = _env_int("SGL_XPU_GGUF_FUSE_MAX_M", 64)
_GGUF_NORM_Q8_OP = None


def _load_gguf_norm_q8_op():
    global _GGUF_NORM_Q8_OP
    if _GGUF_NORM_Q8_OP is None:
        try:
            import custom_esimd_kernels_sglang  # noqa: F401  (registers the lib)

            _GGUF_NORM_Q8_OP = (
                torch.ops.custom_esimd_kernels_sglang.esimd_resadd_norm_gemv_q8_ba
            )
        except Exception as e:
            logger.warning("[gguf_resadd_norm] op unavailable: %r", e)
            _GGUF_NORM_Q8_OP = False
    return _GGUF_NORM_Q8_OP or None


def _gguf_xpu_rep(lin, kind):
    """The single ESIMD weight rep of a GGUF XPU linear, if it is `kind`.

    A GGUF linear keeps one rep per loaded shard; when every shard shares a
    quant type they are pre-merged into ``_xpu_merged`` (row-cat in output
    order), which is exactly the ``[N, K]`` matrix the fused kernel needs. A
    single-shard linear has no merge, so fall back to its only rep.
    """
    if lin is None or getattr(lin, "bias", None) is not None:
        return None
    merged = getattr(lin, "_xpu_merged", None)
    rep = merged[0] if merged is not None else None
    if rep is None:
        reps = getattr(lin, "_xpu_reps", None)
        order = getattr(lin, "_xpu_shard_order", None)
        if isinstance(reps, dict) and order is not None and len(order) == 1:
            rep = reps[order[0]]
    if rep is None or rep[0] != kind:
        return None
    return rep


def _gguf_resadd_norm_guard(hidden_states, residual, forward_batch):
    """True when the plain-TP decode path is what will run, i.e. prepare_attn
    reduces to a bare input_layernorm and the fused op is a drop-in."""
    if not _XPU_GGUF_RESADD_NORM or not _is_xpu:
        return False
    if forward_batch is None or not forward_batch.forward_mode.is_decode():
        return False
    if forward_batch.forward_mode.is_target_verify():
        return False
    if hidden_states.dim() != 2:
        return False
    # The kernel handles any M; the bound only caps how many distinct output
    # buffer sets we are willing to cache (see _GGUF_FUSE_MAX_M).
    if not 1 <= hidden_states.shape[0] <= _GGUF_FUSE_MAX_M:
        return False
    if hidden_states.dtype != torch.float16:
        return False
    # Layer 0 has no residual yet. The kernel always does the add, so it is fed
    # a cached zero buffer: ``h + 0.0`` is exact in fp16 and reproduces the
    # plain path's ``residual = hidden_states; hidden = norm(hidden)``. Leaving
    # layer 0 on the unfused path meant its in_proj_ba ran a oneDNN fp16 GEMV
    # whose split-K reduction order is not reproducible under load, which made
    # decode non-deterministic (identical operands, two possible results).
    if residual is not None:
        if residual.shape != hidden_states.shape:
            return False
        if residual.dtype != torch.float16:
            return False
    if getattr(hidden_states, "_sglang_needs_allreduce_fusion", False):
        return False
    try:
        from sglang.srt.layers.communicator import get_attn_tp_context

        if get_attn_tp_context().input_scattered:
            return False
    except Exception:
        return False
    return True


def _gguf_norm_gemv(layer, norm, lin0, lin1, hidden_states, residual, tag):
    """Run the fused resadd-norm + q8_0 GEMV (+ optional fp16 GEMV).

    ``lin1`` is the GDN ``in_proj_ba`` (unquantised fp16) or None for the
    full-attention layers. Returns ``(out0, out1_or_None, new_residual)`` or
    None to fall back. Weights and norm constants are resolved once and cached
    on the layer; output buffers are cached per distinct token count M, so the
    steady-state cost stays exactly one op dispatch at any batch size.
    """
    op = _load_gguf_norm_q8_op()
    if op is None or getattr(layer, "_gguf_norm_q8_off", False):
        return None
    cache = getattr(layer, "_gguf_norm_q8_cache", None)
    if cache is None:
        rep0 = _gguf_xpu_rep(lin0, "q8_0")
        if rep0 is None:
            logger.warning("[gguf_resadd_norm] %s: proj is not a single q8_0 rep", tag)
            layer._gguf_norm_q8_off = True
            return None
        qs, sc = rep0[1], rep0[2]
        hidden = hidden_states.shape[1]
        if qs.dim() != 2 or qs.shape[1] != hidden:
            logger.warning("[gguf_resadd_norm] %s: q8_0 K=%s != hidden=%s",
                           tag, tuple(qs.shape), hidden)
            layer._gguf_norm_q8_off = True
            return None
        dev = qs.device
        w1 = torch.empty(0, dtype=torch.float16, device=dev)
        if lin1 is not None:
            rep1 = _gguf_xpu_rep(lin1, "fp16")
            if rep1 is None or rep1[1].dim() != 2 or rep1[1].shape[1] != hidden:
                logger.warning("[gguf_resadd_norm] %s: ba is not a single fp16 [N,%s] rep",
                               tag, hidden)
                layer._gguf_norm_q8_off = True
                return None
            w1 = rep1[1].contiguous()
        nw = ((norm.weight.data.to(torch.float32) + 1.0)
              .to(torch.float16).contiguous())
        cache = {
            "nw": nw,
            "eps": float(norm.variance_epsilon),
            "qs": qs,
            "sc": sc,
            "w1": w1,
            "hidden": hidden,
            "dev": dev,
            "has_ba": lin1 is not None,
            # Output buffers, keyed by token count. Decode batch sizes repeat,
            # so this is allocated a handful of times and then only looked up.
            "bufs": {},
        }
        layer._gguf_norm_q8_cache = cache
        logger.warning("[gguf_resadd_norm] %s ACTIVE (norm + proj folded, ba=%s)",
                       tag, cache["has_ba"])

    M = int(hidden_states.shape[0])
    buf = cache["bufs"].get(M)
    if buf is None:
        dev, hidden = cache["dev"], cache["hidden"]
        w1 = cache["w1"]
        buf = {
            "o0": torch.empty((M, cache["qs"].shape[0]), dtype=torch.float16, device=dev),
            "o1": (torch.empty((M, w1.shape[0]), dtype=torch.float16, device=dev)
                   if cache["has_ba"]
                   else torch.empty(0, dtype=torch.float16, device=dev)),
            "xn": torch.empty((M, hidden), dtype=torch.float16, device=dev),
            # Post-add residual goes to its own buffer: the kernel's block 0
            # stores it while the other row-blocks still read the OLD residual.
            # Per-layer, so it is never the same buffer the kernel reads (the
            # incoming residual belongs to the previous layer).
            "nr": torch.empty((M, hidden), dtype=torch.float16, device=dev),
            # Zero residual for layer 0 (residual is None there).
            "zr": torch.zeros((M, hidden), dtype=torch.float16, device=dev),
        }
        cache["bufs"][M] = buf

    h = hidden_states if hidden_states.is_contiguous() else hidden_states.contiguous()
    res = buf["zr"] if residual is None else (
        residual if residual.is_contiguous() else residual.contiguous())
    try:
        op(h, res, cache["nw"], cache["eps"], buf["xn"], buf["nr"],
           cache["qs"], cache["sc"], buf["o0"], cache["w1"], buf["o1"])
    except Exception as e:
        logger.warning("[gguf_resadd_norm] %s: kernel raised %r", tag, e)
        layer._gguf_norm_q8_off = True
        return None
    return buf["o0"], (buf["o1"] if cache["has_ba"] else None), buf["nr"]


cached_get_processor = lru_cache(get_processor)


def _disable_shared_experts_fusion() -> bool:
    # Resolved lazily: the global server args is not set at module import time
    # (e.g. when this module is imported by unit tests).
    return get_global_server_args().disable_shared_experts_fusion


def _esimd_fp8_weight_nk_scale(lin):
    """Resolve an fp8 Linear's weight as row-major ``[N, K]`` plus a per-tensor
    scalar scale, for the ESIMD GEMV fusion kernels.

    Handles the two fp8 weight layouts SGLang produces:

    * **block-quant** (``weight_scale_inv`` present): ``layer.weight`` is stored
      ``[N, K]`` already; the block scale is collapsed to its mean.
    * **online / dynamic per-tensor** (``weight_scale`` present): the loader
      stores ``layer.weight`` **transposed** as ``[K, N]`` (see
      ``Fp8LinearMethod.process_weights_after_loading``). ESIMD needs ``[N, K]``,
      so we transpose (reusing the ``_esimd_t`` cache shared with the dense fp8
      fast path) and take the per-tensor scalar.

    Returns ``(weight_nk, scale_pt_fp32_1d)`` on success, or ``None`` to force a
    safe fallback. The result is validated against the layer's known
    ``output_size_per_partition`` / ``input_size_per_partition`` so a layout we
    did not anticipate never reaches the kernel (which reads raw pointers and
    would otherwise fault with UR_RESULT_ERROR_DEVICE_LOST).
    """
    w = getattr(lin, "weight", None)
    if w is None or w.dtype not in (torch.float8_e4m3fn, torch.float8_e5m2):
        return None
    # Weight and its fp8 scale are static after loading, so the transposed
    # [N, K] layout and the scalar per-tensor scale never change across decode
    # steps. Cache the fully-resolved (w_nk, scale_pt) on the weight so we skip
    # the per-step t()/contiguous()/to(fp32)/mean()/contiguous() device ops
    # (~5 dispatched ops x 60 calls/step of pure redundant recompute).
    _cached = getattr(w, "_esimd_nk_scale", None)
    if _cached is not None:
        return _cached
    if w.dim() != 2:
        return None
    N = getattr(lin, "output_size_per_partition", None)
    K = getattr(lin, "input_size_per_partition", None)
    if N is None or K is None:
        return None

    ws_inv = getattr(lin, "weight_scale_inv", None)
    if ws_inv is not None:
        # Block-quant: weight already [N, K].
        if w.shape[0] != N or w.shape[1] != K:
            return None
        w_nk = w
        scale_src = ws_inv
    else:
        ws = getattr(lin, "weight_scale", None)
        if ws is None:
            return None
        # Online per-tensor: stored [K, N] -> transpose to [N, K].
        if w.shape[0] != K or w.shape[1] != N:
            return None
        w_nk = getattr(w, "_esimd_t", None)
        if w_nk is None:
            w_nk = w.t().contiguous()
            try:
                w._esimd_t = w_nk
            except Exception:
                pass
        scale_src = ws

    if w_nk.shape[0] != N or w_nk.shape[1] != K:
        return None
    scale_pt = (
        scale_src.data.to(torch.float32).reshape(-1).mean().reshape(1).contiguous()
    )
    result = (w_nk, scale_pt)
    try:
        w._esimd_nk_scale = result
    except Exception:
        pass
    return result


def _esimd_q8_0_weight_scale(lin):
    """DEPRECATED (see _XPU_GGUF_RESADD_NORM_LEGACY).

    Resolve a GGUF-q8_0 Linear's quant weight as ``(qs [N, K] int8, scale
    [N, K/32] fp16)`` for the legacy single-matrix ESIMD q8_0 GEMV kernel.

    Superseded by ``_gguf_xpu_rep()``, which returns the packed rep tuple
    directly and additionally handles the fp16 ``in_proj_ba`` shard so both
    projections can be folded into one ``esimd_resadd_norm_gemv_q8_ba`` call.

    The GGUF XPU linear method stores its resident quant reps on the module:
      * ``layer._xpu_merged`` = ``(merged_rep, sizes)`` when q/k/v shards were
        row-concatenated into one big-N GEMV (the D1 optimization); or
      * ``layer._xpu_reps`` / ``layer._xpu_shard_order`` for the per-shard reps.
    A single-matrix fused GEMV needs the *whole* projection as one q8_0 matrix,
    so this returns a value only when the projection is a lone q8_0 shard or a
    merged q8_0 rep; anything else (multi-shard unmerged, non-q8_0) yields
    ``None`` and the caller safely falls back. Cached on the module.
    """
    cached = getattr(lin, "_esimd_q8_0_ws", None)
    if cached is not None:
        return cached
    rep = None
    merged = getattr(lin, "_xpu_merged", None)
    if merged is not None:
        rep = merged[0]
    else:
        order = getattr(lin, "_xpu_shard_order", None)
        reps = getattr(lin, "_xpu_reps", None)
        if order is not None and reps is not None and len(order) == 1:
            rep = reps.get(order[0])
    if rep is None or rep[0] != "q8_0":
        return None
    _, qs, scale = rep
    # Validate the exact [N, K] int8 + [N, K/32] fp16 layout the kernel reads
    # from raw pointers (a mismatch would fault with DEVICE_LOST).
    if qs.dim() != 2 or scale.dim() != 2:
        return None
    if qs.dtype != torch.int8 or scale.dtype != torch.float16:
        return None
    if qs.shape[1] % 32 != 0 or scale.shape[0] != qs.shape[0]:
        return None
    if scale.shape[1] != qs.shape[1] // 32:
        return None
    if not (qs.is_contiguous() and scale.is_contiguous()):
        return None
    result = (qs, scale)
    try:
        lin._esimd_q8_0_ws = result
    except Exception:
        pass
    return result


if _is_npu:
    from sgl_kernel_npu.norm.split_qkv_rmsnorm_rope import (
        split_qkvgate_gemma_rmsnorm_rope,
    )


class Qwen3_5GatedDeltaNet(nn.Module):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        alt_stream: Optional[torch.cuda.Stream] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.hidden_size = config.hidden_size
        self.num_v_heads = (
            config.linear_num_value_heads
            if not _is_cpu
            else config.linear_num_value_heads_cpu
        )
        self.num_k_heads = (
            config.linear_num_key_heads
            if not _is_cpu
            else config.linear_num_key_heads_cpu
        )
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.alt_stream = alt_stream

        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.layer_id = layer_id
        self.activation = config.hidden_act
        self.output_gate_type = config.output_gate_type
        self.layer_norm_epsilon = config.rms_norm_eps

        # Conv1d layer
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv1d = ColumnParallelLinear(
            input_size=self.conv_kernel_size,
            output_size=self.conv_dim,
            bias=False,
            quant_config=None,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("conv1d", prefix),
        )
        self.conv1d.weight.data = self.conv1d.weight.data.unsqueeze(1)

        # projection of the input hidden states
        self.in_proj_qkvz = self.create_qkvz_proj(
            hidden_size=self.hidden_size,
            key_dim=self.key_dim,
            value_dim=self.value_dim,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_qkvz", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        self.in_proj_ba = self.create_ba_proj(
            hidden_size=self.hidden_size,
            num_v_heads=self.num_v_heads,
            quant_config=quant_config,
            prefix=add_prefix("in_proj_ba", prefix),
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
        )

        # Override weight loaders for packed checkpoint format.
        # Important: for FP8, this must cover not only `.weight` but also
        # `weight_scale_inv` / `weight_scale` / `input_scale` if present.
        self._bind_packed_weight_loaders(self.in_proj_qkvz)
        self._bind_packed_weight_loaders(self.in_proj_ba)

        # Conv1d weight loader setup
        query_key_settings = (self.key_dim, 0, False)
        value_settings = (self.value_dim, 0, False)

        self._override_weight_loader(
            self.conv1d.weight,
            mamba_v2_sharded_weight_loader(
                [
                    query_key_settings,
                    query_key_settings,
                    value_settings,
                ],
                self.attn_tp_size,
                self.attn_tp_rank,
            ),
        )

        # State parameters
        self.dt_bias = nn.Parameter(
            torch.ones(self.num_v_heads // self.attn_tp_size),
        )
        self.A_log = nn.Parameter(
            torch.empty(self.num_v_heads // self.attn_tp_size, dtype=torch.float32),
        )

        set_weight_attrs(self.A_log, {"weight_loader": sharded_weight_loader(0)})
        set_weight_attrs(self.dt_bias, {"weight_loader": sharded_weight_loader(0)})

        conv_weights = self.conv1d.weight.view(
            self.conv1d.weight.size(0), self.conv1d.weight.size(2)
        )
        # conv_weights aliases conv1d.weight's storage; loaders that swap that
        # storage on a device move must call rebind_device_views() afterwards.
        self.attn = RadixLinearAttention(
            layer_id=layer_id,
            num_q_heads=self.num_k_heads // self.attn_tp_size,
            num_k_heads=self.num_k_heads // self.attn_tp_size,
            num_v_heads=self.num_v_heads // self.attn_tp_size,
            head_q_dim=self.head_k_dim,
            head_k_dim=self.head_k_dim,
            head_v_dim=self.head_v_dim,
            conv_weights=conv_weights,
            bias=self.conv1d.bias,
            activation=self.activation,
            A_log=self.A_log,
            dt_bias=self.dt_bias,
        )

        self.norm = RMSNormGated(
            self.head_v_dim,
            eps=self.layer_norm_epsilon,
            group_size=None,
            norm_before_gate=True,
            device=torch.get_device_module().current_device(),
            dtype=config.torch_dtype,
            **(
                {"activation": self.output_gate_type}
                if self.output_gate_type is not None
                else {}
            ),
        )

        self.out_proj = RowParallelLinear(
            self.value_dim,
            self.hidden_size,
            bias=False,
            input_is_parallel=True,
            reduce_results=False,
            quant_config=quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("out_proj", prefix),
        )
        # NOTE (Qwen3.6 GGUF): out_proj's input (value-head) columns are
        # stored by GGUF in [ratio, num_k] order but HF/core_attn_out expects
        # [num_k, ratio]. This is an INPUT-dim (column) permute, and under TP the
        # value-head grouping crosses the RowParallel input-shard boundary
        # (rank0's HF heads map to GGUF cols in BOTH ratio halves), so it cannot
        # be done entirely per-rank. It is applied to the GLOBAL pre-shard weight
        # in `_gguf_gdn_transform`: as a single raw-byte per-head permute when
        # head_v_dim is a multiple of the quant block (the 35B's Q8_0, block=32),
        # or otherwise split into a coarse pre-shard group permute plus a
        # per-rank element-order permute carried by `_gguf_gdn_col_perm` (the
        # 27B's Q5_K, whose 256-elem super-block is twice head_v_dim).
        # At ratio=1 the layouts coincide.

    def rebind_device_views(self):
        """Re-derive tensors that alias conv1d.weight's storage.

        ``conv_weights`` (passed to RadixLinearAttention) is a *view* of
        ``self.conv1d.weight`` captured at construction time. Loaders that
        load on CPU and then swap ``conv1d.weight.data`` for a device tensor
        (e.g. ``--load-format layered_fp8``) leave that view pointing at the
        freed CPU storage, so the conv1d kernel later dereferences an invalid
        pointer. Rebuild the view from the current weight; the lazily-built
        ESIMD copy (``_esimd_conv_weights``) self-heals on next forward, so
        just drop it here.
        """
        w = self.conv1d.weight
        self.attn.conv_weights = w.view(w.size(0), w.size(2))
        self.attn.bias = self.conv1d.bias
        self._esimd_conv_weights = None

    @staticmethod
    def _override_weight_loader(param, loader):
        """Robustly override loader for:
        1) BasevLLMParameter subclasses: real storage is `_weight_loader`
        2) regular Parameters that already have mutable `weight_loader`
        3) regular Parameters without `weight_loader` yet
        """
        if hasattr(param, "_weight_loader"):
            # FP8 / quantized BasevLLMParameter path
            param._weight_loader = loader
            return

        if hasattr(param, "weight_loader"):
            # Regular parameter/tensor that already has a mutable attr.
            # Do NOT call set_weight_attrs here, because it asserts when
            # overwriting an existing attribute.
            param.weight_loader = loader
            return

        # Fresh attribute on a normal tensor/Parameter
        set_weight_attrs(param, {"weight_loader": loader})

    def _bind_packed_weight_loaders(self, module):
        """Bind packed-checkpoint-aware loaders to all relevant params of a merged module."""
        # "qweight" / "qweight_type" cover the GGUF path: its merged params are
        # named qweight (not weight), and its native weight_loader only accepts
        # int shard ids. The packed wrapper splits a fused checkpoint tensor
        # (e.g. GGUF attn_qkv = q|k|v) by the tuple shard id (0,1,2) into int
        # shards before delegating, so GGUF GDN projections load correctly.
        for attr_name in (
            "weight",
            "weight_scale_inv",
            "weight_scale",
            "input_scale",
            "qweight",
            "qweight_type",
        ):
            param = getattr(module, attr_name, None)
            if param is None:
                continue
            original_loader = getattr(param, "weight_loader", None)
            if original_loader is None:
                continue
            wrapped_loader = self._make_packed_weight_loader(module, original_loader)
            self._override_weight_loader(param, wrapped_loader)

    @staticmethod
    def _get_split_sizes_for_param(module, param, loaded_shard_id):
        """Return checkpoint-side split sizes for this param type."""
        if isinstance(param, BlockQuantScaleParameter):
            # Split by output blocks, not raw output sizes.
            block_n, _ = module.quant_method.quant_config.weight_block_size
            block_n = 1 if getattr(param, "format_ue8m0", False) else block_n
            return [
                (module.output_sizes[idx] + block_n - 1) // block_n
                for idx in loaded_shard_id
            ]

        if isinstance(param, PerTensorScaleParameter):
            # One logical scale per logical shard.
            return [1 for _ in loaded_shard_id]

        # Normal weight / non-block quant tensor
        return [module.output_sizes[idx] for idx in loaded_shard_id]

    @classmethod
    def _make_packed_weight_loader(cls, module, original_weight_loader):
        """Wrap the param's original loader so split checkpoints:
          - in_proj_qkv + in_proj_z -> merged in_proj_qkvz
          - in_proj_b + in_proj_a   -> merged in_proj_ba
        can load correctly for both normal and FP8 params.
        """

        def weight_loader(param, loaded_weight, loaded_shard_id=None):
            # Only intercept split-checkpoint tuple shards.
            # int shard_id and None should preserve original behavior.
            if isinstance(loaded_shard_id, tuple):
                split_sizes = cls._get_split_sizes_for_param(
                    module, param, loaded_shard_id
                )

                if len(loaded_weight.shape) == 0:
                    # Scalar shard payload. Two cases:
                    #  - single logical shard: load as-is (original behavior).
                    #  - GGUF qweight_type: one scalar quant-type for the whole
                    #    fused tensor, replicated to each int shard so the GGUF
                    #    loader records the type per shard slot.
                    if len(split_sizes) == 1 and split_sizes[0] == 1:
                        chunks = [loaded_weight.reshape(1)]
                    else:
                        chunks = [loaded_weight for _ in loaded_shard_id]
                else:
                    split_dim = getattr(param, "output_dim", 0)
                    if _is_cpu:
                        cpu_split_sizes = []
                        split_size_sum = sum(split_sizes)
                        target_size_sim = loaded_weight.size(split_dim)
                        for i in range(len(split_sizes)):
                            cpu_split_sizes.append(
                                int(target_size_sim * split_sizes[i] / split_size_sum)
                            )
                        assert (
                            sum(cpu_split_sizes) == target_size_sim
                        ), f"Padding the loaded weight failed due to sizes are not divisible cleanly from {cpu_split_sizes} to {target_size_sim}"
                        chunks = loaded_weight.split(cpu_split_sizes, dim=split_dim)
                    else:
                        chunks = loaded_weight.split(split_sizes, dim=split_dim)

                assert len(chunks) == len(loaded_shard_id), (
                    f"Chunk/shard mismatch: {len(chunks)=}, "
                    f"{len(loaded_shard_id)=}, {split_sizes=}"
                )

                for idx, chunk in zip(loaded_shard_id, chunks):
                    # Delegate each chunk to the param's original int-shard loader.
                    original_weight_loader(param, chunk, idx)
                return

            return original_weight_loader(param, loaded_weight, loaded_shard_id)

        return weight_loader

    def create_qkvz_proj(
        self,
        hidden_size: int,
        key_dim: int,
        value_dim: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> MergedColumnParallelLinear:
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[key_dim, key_dim, value_dim, value_dim],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def create_ba_proj(
        self,
        hidden_size: int,
        num_v_heads: int,
        quant_config: QuantizationConfig | None,
        prefix: str,
        tp_rank: Optional[int] = None,
        tp_size: Optional[int] = None,
    ) -> MergedColumnParallelLinear:
        # Qwen3.5 has separate in_proj_b and in_proj_a weights in the
        # checkpoint, which are loaded into the fused in_proj_ba parameter
        # via stacked_params_mapping with shard_id 0 and 1 respectively.
        return MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[num_v_heads, num_v_heads],
            bias=False,
            quant_config=quant_config,
            prefix=prefix,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )

    def fix_query_key_value_ordering(
        self,
        mixed_qkvz: torch.Tensor,
        mixed_ba: torch.Tensor,
    ):
        """
        Derives `query`, `key` and `value` tensors from `mixed_qkvzba`.
        """
        k_tp = self.key_dim // self.attn_tp_size
        v_tp = self.value_dim // self.attn_tp_size
        nv_tp = self.num_v_heads // self.attn_tp_size

        # Directly split, no head group reshape
        query, key, value, z = mixed_qkvz.split([k_tp, k_tp, v_tp, v_tp], dim=-1)
        b, a = mixed_ba.split([nv_tp, nv_tp], dim=-1)

        # value / z reshape to (seq, num_v_heads/tp, head_v_dim)
        value = value.reshape(value.size(0), -1, self.head_v_dim)
        z = z.reshape(z.size(0), -1, self.head_v_dim)

        return query, key, value, z, b, a

    def _esimd_fused_input_proj(self, hidden_states: torch.Tensor):
        """Fuse in_proj_qkvz + in_proj_ba into one esimd_gemv_fp8_pert_fused2 call.

        Both projections read the same ``hidden_states`` and share the contraction
        dim K, so the two fp8 GEMVs collapse into a single kernel dispatch. Returns
        ``(qkvz, ba)`` on success or ``None`` to fall back to two separate linears.
        Guarded to decode single-token (M=1), fp16 input, both weights fp8.
        """
        if not _XPU_GDN_INPROJ_FUSED2:
            return None
        if hidden_states.dim() != 2 or hidden_states.shape[0] != 1:
            return None
        # A bias'd projection would need a post-GEMV add; keep it simple.
        if getattr(self.in_proj_qkvz, "bias", None) is not None:
            return None
        if getattr(self.in_proj_ba, "bias", None) is not None:
            return None
        r0 = _esimd_fp8_weight_nk_scale(self.in_proj_qkvz)
        r1 = _esimd_fp8_weight_nk_scale(self.in_proj_ba)
        if r0 is None or r1 is None:
            return None
        w0, s0 = r0
        w1, s1 = r1
        if w0.shape[1] != w1.shape[1] or w0.shape[1] != hidden_states.shape[1]:
            return None
        try:
            from custom_esimd_kernels_sglang import esimd_gemv_fp8_pert_fused2
        except Exception:
            return None
        x = (
            hidden_states
            if hidden_states.dtype == torch.float16
            else hidden_states.to(torch.float16)
        )
        x = x if x.is_contiguous() else x.contiguous()
        scratch = getattr(self, "_esimd_inproj_scratch", None)
        if scratch is None or scratch[0].shape[1] != w0.shape[0] or scratch[1].shape[1] != w1.shape[0]:
            o0 = torch.empty((1, w0.shape[0]), dtype=torch.float16, device=x.device)
            o1 = torch.empty((1, w1.shape[0]), dtype=torch.float16, device=x.device)
            self._esimd_inproj_scratch = (o0, o1)
        else:
            o0, o1 = scratch
        try:
            esimd_gemv_fp8_pert_fused2(x, w0, s0, o0, w1, s1, o1)
        except Exception:
            return None
        return o0, o1

    def _forward_input_proj(self, hidden_states: torch.Tensor):
        fused = self._esimd_fused_input_proj(hidden_states)
        if fused is not None:
            return fused
        if (
            _is_cpu
            or _is_npu
            or not get_global_server_args().disable_piecewise_cuda_graph
        ):
            DUAL_STREAM_TOKEN_THRESHOLD = 0
        else:
            DUAL_STREAM_TOKEN_THRESHOLD = 1024

        seq_len, _ = hidden_states.shape
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and seq_len < DUAL_STREAM_TOKEN_THRESHOLD
            and _gdn_use_alt_stream
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            with torch.cuda.stream(self.alt_stream):
                projected_states_ba, _ = self.in_proj_ba(hidden_states)
            current_stream.wait_stream(self.alt_stream)
        else:
            projected_states_qkvz, _ = self.in_proj_qkvz(hidden_states)
            projected_states_ba, _ = self.in_proj_ba(hidden_states)
        return projected_states_qkvz, projected_states_ba

    def _gdn_seq_to_interleaved(
        self,
        projected_states_qkvz: torch.Tensor,
        projected_states_ba: torch.Tensor,
    ):
        """Reorder sglang's sequential qkvz/ba into the GQA-interleaved layout
        the native ``sgl_kernel.gdn_attention`` kernel reads.

        Input (sglang, per-tp columns):
          qkvz = [q_all(nk*hk) | k_all(nk*hk) | v_all(nv*hv) | z_all(nv*hv)]
          ba   = [b_all(nv)    | a_all(nv)]
        Output (kernel, per-k-head interleaved blocks):
          qkvz = [ q(hk) k(hk) v(ratio*hv) z(ratio*hv) ] x nk
          ba   = [ b(ratio) a(ratio) ] x nk
        where ratio = nv // nk and v-head v belongs to k-head v // ratio
        (contiguous GQA grouping). Returns contiguous tensors.
        """
        T = projected_states_qkvz.shape[0]
        nk = self.num_k_heads // self.attn_tp_size
        nv = self.num_v_heads // self.attn_tp_size
        ratio = nv // nk
        hk = self.head_k_dim
        hv = self.head_v_dim
        key_dim = nk * hk
        val_dim = nv * hv

        qkvz = projected_states_qkvz
        q = qkvz[:, 0:key_dim].reshape(T, nk, hk)
        k = qkvz[:, key_dim : 2 * key_dim].reshape(T, nk, hk)
        v = qkvz[:, 2 * key_dim : 2 * key_dim + val_dim].reshape(T, nk, ratio * hv)
        z = qkvz[:, 2 * key_dim + val_dim :].reshape(T, nk, ratio * hv)
        qkvz_il = torch.cat([q, k, v, z], dim=2).reshape(T, -1).contiguous()

        ba = projected_states_ba
        b = ba[:, 0:nv].reshape(T, nk, ratio)
        a = ba[:, nv : 2 * nv].reshape(T, nk, ratio)
        ba_il = torch.cat([b, a], dim=2).reshape(T, -1).contiguous()

        return qkvz_il, ba_il

    def _forward_xpu_fast_path(
        self,
        projected_states_qkvz: torch.Tensor,
        projected_states_ba: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Run conv1d + GDN + RMSNormGated + out_proj using the vendored
        torch.ops.sgl_kernel.gdn_attention kernel. Returns None if the op is
        unusable (non-contiguous input, unexpected shape, ...) so the caller
        falls back to the default Triton/PyTorch path.

        Layout assumption: num_v_heads // num_k_heads == 1 — then sglang's
        MergedColumnParallelLinear sequential `[Q|K|V|Z]` layout and the
        GQA-interleaved layout gdn_attention expects are bit-identical.
        """
        if not hasattr(torch.ops, "sgl_kernel") or not hasattr(
            torch.ops.sgl_kernel, "gdn_attention"
        ):
            return None

        # Pull the backend metadata the kernel needs. This mirrors
        # GDNAttnBackend.forward_extend.
        attn_backend = forward_batch.attn_backend
        linear_backend = getattr(attn_backend, "linear_attn_backend", attn_backend)
        fwd_md = getattr(linear_backend, "forward_metadata", None)
        if fwd_md is None:
            return None

        query_start_loc = fwd_md.query_start_loc
        cache_indices = fwd_md.mamba_cache_indices
        layer_id = self.layer_id
        mamba_cache_params = linear_backend.req_to_token_pool.mamba2_layer_cache(
            layer_id
        )
        # Layout adapter. sglang's MambaPool stores the conv state as
        # (cache_batch, conv_dim, W-1) (see configs/mamba_utils.py:161 —
        # Mamba2StateShape.conv_state_shape), but the sgl-kernel-xpu
        # gdn_attention kernel expects (cache_batch, W-1, conv_dim) and
        # asserts per-batch contiguity on it. A raw .transpose() on the pool
        # view produces a non-contiguous tensor; .contiguous() would copy the
        # whole pool and would NOT write the kernel's updates back to the
        # real pool.
        #
        # Instead, gather the rows indexed by `cache_indices` into a small
        # scratch tensor in the kernel layout, run the kernel (which writes
        # into scratch), then scatter the (conv_dim, W-1)-laid-out updates
        # back into the pool. Cost is O(bs * conv_dim * (W-1)) per call —
        # negligible for bs=1 decode and still cheap for extend.
        pool_conv = mamba_cache_params.conv[0]  # (cache, conv_dim, W-1)
        pool_ssm = mamba_cache_params.temporal  # (cache, Hv, head_v, head_k)
        scratch_conv = (
            pool_conv.index_select(0, cache_indices).transpose(-1, -2).contiguous()
        )  # (bs, W-1, conv_dim) — kernel layout
        # ssm_state's layout already matches the kernel expectation, so just
        # gather the active rows.
        scratch_ssm = pool_ssm.index_select(0, cache_indices).contiguous()

        # num_prefills / num_decodes. During pure extend it's bs/0; during
        # pure decode it's 0/bs; we approximate by inspecting forward_mode.
        batch_size = cache_indices.shape[0]
        if forward_batch.forward_mode.is_decode():
            num_prefills = 0
            num_decodes = batch_size
        else:
            num_prefills = batch_size
            num_decodes = 0

        # has_initial_state: True for each seq whose ssm state was warmed up
        # by a previous extend. During pure prefill all false; during decode
        # all true; during mixed, check extend_prefix_lens.
        extend_prefix_lens = forward_batch.extend_prefix_lens
        if extend_prefix_lens is None:
            has_initial_state = torch.ones(
                batch_size, dtype=torch.bool, device=cache_indices.device
            )
        else:
            has_initial_state = extend_prefix_lens > 0

        num_actual_tokens = projected_states_qkvz.shape[0]

        # Layout adapter: sglang produces projected_states_qkvz in SEQUENTIAL
        # [q_all | k_all | v_all | z_all] order and projected_states_ba as
        # [b_all | a_all] (see fix_query_key_value_ordering). The native
        # gdn_attention kernel instead reads qkvz as GQA-INTERLEAVED per-k-head
        # blocks [q(head_k), k(head_k), v(head_v*ratio), z(head_v*ratio)] and ba
        # as per-k-head [b(ratio), a(ratio)] (see chunk_causal_conv1d_xe2.hpp:
        # qkvz_elems_offset = k_head_id*qkvz_dim+off; chunk_reorder_zba step =
        # (token*num_v + k_head*ratio)*2). Reorder here so the kernel sees the
        # layout it expects. conv_weights/conv_state/ssm_state stay sequential
        # (the kernel reads those via reordered_elems_offset), so they are
        # untouched. All dims are per-tp (the projection is column-parallel).
        projected_states_qkvz, projected_states_ba = self._gdn_seq_to_interleaved(
            projected_states_qkvz, projected_states_ba
        )

        # Output buffers (kernel writes into these).
        nv_tp = self.num_v_heads // self.attn_tp_size
        core_attn_out = projected_states_qkvz.new_empty(
            (num_actual_tokens, nv_tp, self.head_v_dim)
        )
        z = torch.empty_like(core_attn_out)

        # The kernel takes `cache_indices` so it indexes into `conv_state` as
        # `conv_state[cache_indices[b]]`. Our scratch is densely packed 0..bs-1,
        # so pass an identity index vector to the kernel and scatter back
        # using the real cache_indices afterwards.
        scratch_indices = torch.arange(
            batch_size, device=cache_indices.device, dtype=cache_indices.dtype
        )

        torch.ops.sgl_kernel.gdn_attention(
            core_attn_out,
            z,
            projected_states_qkvz,
            projected_states_ba,
            self.num_k_heads,
            self.num_v_heads,
            self.head_k_dim,
            self.head_v_dim,
            scratch_conv,
            scratch_ssm,
            self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            ),
            self.conv1d.bias,
            self.activation,
            self.A_log,
            self.dt_bias,
            num_prefills,
            num_decodes,
            has_initial_state,
            query_start_loc,
            scratch_indices,
            num_actual_tokens,
            self.attn_tp_size,
        )

        # Scatter kernel writeback into the real MambaPool slots:
        #   conv: (bs, W-1, conv_dim) → (cache, conv_dim, W-1)
        #   ssm:  (bs, Hv, head_v, head_k) — already matches pool layout
        cache_indices_long = cache_indices.to(torch.long)
        pool_conv.index_copy_(
            0, cache_indices_long, scratch_conv.transpose(-1, -2).contiguous()
        )
        pool_ssm.index_copy_(0, cache_indices_long, scratch_ssm)

        # Post: RMSNormGated(core_attn_out, z) then out_proj. Mirrors the
        # default path lines 504-519 below.
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if core_attn_out.shape != z.shape:
            core_attn_out_pad = torch.zeros_like(z)
            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = core_attn_out_pad
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
        output, _ = self.out_proj(core_attn_out)
        return output

    def _gguf_norm_out_proj(
        self, core_attn_out: torch.Tensor, z_out: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """RMSNormGated + GGUF q8_0 out_proj in one op dispatch.

        The GGUF counterpart of ``_esimd_norm_out_proj`` (which is fp8-only).
        Removes the standalone ``gdn_rms_norm_gated`` launch plus the reshape /
        cast glue around it, ~30 dispatches per decode step.

        The out_proj rep already carries the GGUF->HF value-head column permute
        (baked into the global pre-shard weight, see __init__), so feeding the
        raw rep is exactly what the unfused ``out_proj(...)`` would do.
        """
        op = getattr(torch.ops.custom_esimd_kernels_sglang,
                     "esimd_norm_gemv_q8_0", None)
        if op is None or getattr(self, "_gguf_outproj_off", False):
            return None
        cache = getattr(self, "_gguf_outproj_const", None)
        if cache is None:
            rep = _gguf_xpu_rep(self.out_proj, "q8_0")
            if rep is None:
                logger.warning("[gguf_norm_out_proj] out_proj is not a single q8_0 rep")
                self._gguf_outproj_off = True
                return None
            qs, sc = rep[1], rep[2]
            HV, V = int(core_attn_out.shape[1]), int(core_attn_out.shape[2])
            if qs.shape[1] != HV * V:
                logger.warning("[gguf_norm_out_proj] K=%s != HV*V=%s",
                               qs.shape[1], HV * V)
                self._gguf_outproj_off = True
                return None
            nw = self.norm.weight
            nw = nw.to(torch.float16).contiguous() if nw.dtype != torch.float16 \
                else nw.contiguous()
            if nw.numel() != V:
                logger.warning("[gguf_norm_out_proj] norm weight %s != V=%s",
                               nw.numel(), V)
                self._gguf_outproj_off = True
                return None
            cache = {
                "qs": qs, "sc": sc, "nw": nw, "HV": HV, "V": V,
                "eps": float(self.layer_norm_epsilon),
                # Buffers keyed by token count; the kernel takes any M.
                "bufs": {},
            }
            self._gguf_outproj_const = cache
            logger.warning("[gguf_norm_out_proj] ACTIVE (gated norm + out_proj folded)")
        HV, V = cache["HV"], cache["V"]
        M = int(core_attn_out.shape[0])
        buf = cache["bufs"].get(M)
        if buf is None:
            dev = cache["qs"].device
            buf = {
                "y": torch.empty((M * HV, V), dtype=torch.float16, device=dev),
                "out": torch.empty((M, cache["qs"].shape[0]),
                                   dtype=torch.float16, device=dev),
            }
            cache["bufs"][M] = buf
        x = core_attn_out.reshape(M * HV, V)
        z = z_out.reshape(M * HV, V)
        if not x.is_contiguous():
            x = x.contiguous()
        if not z.is_contiguous():
            z = z.contiguous()
        try:
            op(x, z, cache["nw"], buf["y"],
               cache["qs"], cache["sc"], buf["out"],
               HV, V, cache["eps"])
        except Exception as e:
            logger.warning("[gguf_norm_out_proj] kernel raised %r", e)
            self._gguf_outproj_off = True
            return None
        return buf["out"]

    def _esimd_norm_out_proj(
        self, core_attn_out: torch.Tensor, z_out: torch.Tensor
    ) -> Optional[torch.Tensor]:
        """Fused RMSNormGated + FP8 out_proj GEMV for the GDN decode path.

        Replaces ``self.norm(x, z)`` (a standalone RMSNormGated launch) +
        ``self.out_proj(...)`` (a separate fp8 GEMV) + the surrounding
        cast/reshape/empty glue with a single ESIMD launch
        (``esimd_norm_gemv_fp8_pert``). Gated by ``SGL_XPU_GDN_NORM_GEMV=1``.

        Semantics matched to RMSNormGated(norm_before_gate=True,
        activation="swish"): ``out = (rmsnorm(x) * norm_weight * silu(z)) @
        out_proj.weight^T * scale``. The per-block out_proj weight_scale is
        collapsed to a single per-tensor scalar (mean), mirroring the dense
        ESIMD fp8 GEMV fast path.

        Returns the layer output ``[M, hidden]`` on success, or ``None`` to fall
        back to the eager norm + out_proj path. The GGUF q8_0 branch handles any
        decode batch size; the fp8 branch below is still single-token only
        (``esimd_norm_gemv_fp8_pert`` emits ``[1, N]``).
        """
        if core_attn_out.dim() != 3:
            return None
        if core_attn_out.shape != z_out.shape:
            return None
        if getattr(self.norm, "activation", "swish") not in ("swish", "silu"):
            return None
        if not getattr(self.norm, "norm_before_gate", True):
            return None
        # Phase 5c: GGUF q8_0 out_proj (the fp8 path below never matches it).
        # M-generic: x/z are [M, HV, V] and the kernel derives M from the row
        # count, so batched decode is still a single launch.
        if (
            _XPU_GGUF_RESADD_NORM
            and core_attn_out.dtype == torch.float16
            and 1 <= core_attn_out.shape[0] <= _GGUF_FUSE_MAX_M
        ):
            r = self._gguf_norm_out_proj(core_attn_out, z_out)
            if r is not None:
                return r
        # fp8 kernel assumes a single decode token: x/z are [HV, V].
        if core_attn_out.shape[0] != 1:
            return None
        if not _XPU_GDN_NORM_GEMV:
            return None
        try:
            from custom_esimd_kernels_sglang import esimd_norm_gemv_fp8_pert
        except ImportError:
            return None
        resolved = _esimd_fp8_weight_nk_scale(self.out_proj)
        if resolved is None:
            return None
        w_nk, scale_pt = resolved
        HV = core_attn_out.shape[1]
        V = core_attn_out.shape[2]
        # Kernel contracts along HV*V; must equal out_proj in_features (K).
        if HV * V != w_nk.shape[1]:
            return None
        x = core_attn_out.reshape(HV, V)
        z = z_out.reshape(HV, V)
        if x.dtype != torch.float16:
            x = x.to(torch.float16)
        if z.dtype != torch.float16:
            z = z.to(torch.float16)
        x = x.contiguous()
        z = z.contiguous()
        # Cache fp16 norm weight, collapsed per-tensor scale, and the output
        # buffer once per layer (constants across calls; the buffer is reused so
        # its data ptr stays stable across XPU-graph replays).
        cache = getattr(self, "_esimd_outproj_const", None)
        if cache is None:
            nw = self.norm.weight
            nw = (
                nw.to(torch.float16).contiguous()
                if nw.dtype != torch.float16
                else nw.contiguous()
            )
            out_buf = torch.empty(
                (1, w_nk.shape[0]), dtype=torch.float16, device=w_nk.device
            )
            cache = {"nw": nw, "scale": scale_pt, "w": w_nk, "out": out_buf}
            self._esimd_outproj_const = cache
        try:
            esimd_norm_gemv_fp8_pert(
                x,
                z,
                cache["nw"],
                cache["w"],
                cache["scale"],
                cache["out"],
                HV,
                V,
                float(self.layer_norm_epsilon),
            )
        except Exception:
            return None
        return cache["out"]

    def _forward_xpu_esimd_gdn_decode(
        self,
        projected_states_qkvz: torch.Tensor,
        projected_states_ba: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> Optional[torch.Tensor]:
        """ESIMD fast path for GDN decode.

        Calls custom_esimd_kernels_sglang.esimd_gdn_conv_fused_seq, which
        fuses conv1d + the GDN recurrence into one ESIMD launch. Accepts
        sequential [q|k|v|z] layout directly (no gather).

        Kernel is fp16-only; on a bf16 model the implicit cast loses
        precision in the GDN recurrence and accuracy drops. Restrict to
        fp16 inputs.

        Returns the layer output on success, or None to fall back.
        """
        if projected_states_qkvz.dtype != torch.float16:
            return None
        try:
            from custom_esimd_kernels_sglang import esimd_gdn_conv_fused_seq
        except ImportError:
            return None

        from sglang.srt.model_executor.forward_context import get_attn_backend
        try:
            attn_backend = get_attn_backend()
        except Exception:
            return None
        linear_backend = getattr(attn_backend, "linear_attn_backend", attn_backend)
        fwd_md = getattr(linear_backend, "forward_metadata", None)
        if fwd_md is None:
            return None
        cache_indices = fwd_md.mamba_cache_indices
        if cache_indices is None:
            return None

        mamba_cache_params = linear_backend.req_to_token_pool.mamba2_layer_cache(
            self.layer_id
        )
        # sglang pool conv native layout: (cache, conv_dim, W-1). The updated
        # ESIMD kernel reads AND writes conv_state directly in this native
        # layout (auto-detected in C++ by the small trailing dim W-1), so when
        # the pool is fp16 we pass it in-place and skip both the whole-pool
        # transpose copy and the per-layer index_copy_ writeback (~3 dispatched
        # ops/layer * 30 GDN layers eliminated). Falls back to the legacy
        # transposed-copy path when the pool dtype is not fp16.
        pool_conv = mamba_cache_params.conv[0]
        pool_ssm = mamba_cache_params.temporal
        _conv_native = (pool_conv.dtype == torch.float16)
        if _conv_native:
            conv_state_view = pool_conv
        else:
            # The kernel reads conv state in (cache, W-1, conv_dim) layout, which
            # the pool does not store, so we materialize a transposed-contiguous
            # copy. Reuse a fixed buffer (copy_ rather than a fresh
            # transpose().contiguous() each call) so its data ptr stays stable
            # across XPU-graph replays.
            cvcache = getattr(self, "_esimd_gdn_conv_view", None)
            if cvcache is None or cvcache.shape != pool_conv.shape[:1] + pool_conv.shape[1:][::-1]:
                cvcache = torch.empty(
                    (pool_conv.size(0), pool_conv.size(2), pool_conv.size(1)),
                    dtype=pool_conv.dtype, device=pool_conv.device,
                )
                self._esimd_gdn_conv_view = cvcache
            cvcache.copy_(pool_conv.transpose(-1, -2))
            conv_state_view = cvcache

        # Cache the conv1d.weight view + zeros bias once per layer.
        if getattr(self, "_esimd_conv_weights", None) is None:
            w = self.conv1d.weight
            self._esimd_conv_weights = w.view(w.size(0), w.size(2)).contiguous()
            self._esimd_conv_bias_zeros = torch.zeros(
                w.size(0),
                dtype=w.dtype,
                device=w.device,
            )

        # The ESIMD kernel is fp16-only — cast inputs/outputs around the call.
        # qkvz/ba already on XPU; cast to fp16.
        orig_dtype = projected_states_qkvz.dtype
        # Cache the constant fp16 casts of the per-layer weights once. These
        # never change between calls, so recomputing .to(fp16) each time both
        # wastes time and allocates fresh tensors mid-graph-capture.
        wcache = getattr(self, "_esimd_gdn_wconst", None)
        if wcache is None:
            wcache = {
                "conv_w": self._esimd_conv_weights.to(torch.float16),
                "conv_b": self._esimd_conv_bias_zeros.to(torch.float16),
                "A_log": (
                    self.A_log.to(torch.float16)
                    if self.A_log.dtype != torch.float16
                    else self.A_log
                ),
                "dt_bias": (
                    self.dt_bias.to(torch.float16)
                    if self.dt_bias.dtype != torch.float16
                    else self.dt_bias
                ),
            }
            self._esimd_gdn_wconst = wcache
        conv_w = wcache["conv_w"]
        conv_b = wcache["conv_b"]
        A_log = wcache["A_log"]
        dt_bias = wcache["dt_bias"]
        if orig_dtype != torch.float16:
            qkvz = projected_states_qkvz.to(torch.float16).contiguous()
            ba = projected_states_ba.to(torch.float16).contiguous()
            conv_state_view = conv_state_view.to(torch.float16)
            ssm_state_view = pool_ssm.to(torch.float16).contiguous()
        else:
            qkvz = projected_states_qkvz.contiguous()
            ba = projected_states_ba.contiguous()
            ssm_state_view = pool_ssm

        N = qkvz.shape[0]
        nk_tp = self.num_k_heads // self.attn_tp_size
        nv_tp = self.num_v_heads // self.attn_tp_size
        scale = float(self.head_k_dim ** -0.5)

        # Pre-allocate the conv/recurrence output scratch, keyed by token count,
        # and reuse across XPU-graph replays so the buffers' data ptrs stay
        # stable. Decode graphs are captured per batch size, so a given replay
        # always sees the N it was captured with.
        ocache = getattr(self, "_esimd_gdn_scratch", None)
        if ocache is None or ocache[0] != N:
            core_attn_out = torch.empty(
                (N, nv_tp, self.head_v_dim),
                dtype=torch.float16, device=qkvz.device,
            )
            z_out = torch.empty_like(core_attn_out)
            self._esimd_gdn_scratch = (N, core_attn_out, z_out)
        else:
            _, core_attn_out, z_out = ocache

        try:
            esimd_gdn_conv_fused_seq(
                qkvz, conv_state_view, conv_w, conv_b, cache_indices,
                A_log, dt_bias, ba,
                ssm_state_view, cache_indices, core_attn_out, z_out,
                N, nk_tp, nv_tp, self.head_k_dim, self.head_v_dim, scale,
            )
        except Exception:
            return None

        # conv_state writeback. In the native fp16 path the ESIMD kernel already
        # shifted conv_state in-place into pool_conv, so no python writeback is
        # needed. Only the legacy transposed-copy path (or a non-fp16 ssm pool)
        # needs the index_copy_ round-trip.
        # cache_indices is a step-level (batch) tensor identical for all 30 GDN
        # layers; ``.to(torch.long)`` is a real dispatched copy (int32->int64).
        # Memoize the long view on the per-step fwd_md object so only the first
        # GDN layer that needs it pays the cast and the other layers reuse it.
        need_conv_wb = not _conv_native
        need_ssm_wb = pool_ssm.dtype != torch.float16
        if need_conv_wb or need_ssm_wb:
            cache_indices_long = getattr(fwd_md, "_cache_indices_long", None)
            if cache_indices_long is None or cache_indices_long.numel() != cache_indices.numel():
                cache_indices_long = cache_indices.to(torch.long)
                try:
                    fwd_md._cache_indices_long = cache_indices_long
                except Exception:
                    pass
        if need_conv_wb:
            # (cache, W-1, conv_dim) → (cache, conv_dim, W-1); only touched slots.
            pool_conv.index_copy_(
                0, cache_indices_long,
                conv_state_view.index_select(0, cache_indices_long).transpose(-1, -2).contiguous().to(pool_conv.dtype),
            )
        if need_ssm_wb:
            pool_ssm.index_copy_(
                0, cache_indices_long,
                ssm_state_view.index_select(0, cache_indices_long).to(pool_ssm.dtype),
            )

        # Norm + out_proj. Mirrors the default path.
        # Fast path: fuse RMSNormGated + fp8 out_proj into one ESIMD launch,
        # eliminating the standalone norm kernel + separate GEMV + cast/reshape
        # glue (decode single-token, SGL_XPU_GDN_NORM_GEMV=1). Falls back on None.
        fused_out = self._esimd_norm_out_proj(core_attn_out, z_out)
        if fused_out is not None:
            return fused_out

        core_attn_out = core_attn_out.to(orig_dtype)
        z_out = z_out.to(orig_dtype)
        z_shape_og = z_out.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z_out = z_out.reshape(-1, z_out.shape[-1])
        if core_attn_out.shape != z_out.shape:
            pad = torch.zeros_like(z_out)
            pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = pad
        core_attn_out = self.norm(core_attn_out, z_out)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)
        output, _ = self.out_proj(core_attn_out)
        return output

    def forward(
        self,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """
        Forward pass with three parts:
        1. Input projection
        2. Core attention (custom op)
        3. Output projection
        """
        projected_states_qkvz, projected_states_ba = self._forward_input_proj(
            hidden_states
        )
        return self._forward_from_projected(
            projected_states_qkvz, projected_states_ba, forward_batch
        )

    def _forward_from_projected(
        self,
        projected_states_qkvz: torch.Tensor,
        projected_states_ba: torch.Tensor,
        forward_batch: ForwardBatch,
    ):
        """Core attention + output path starting from the in_proj outputs.

        Split out of ``forward`` so the fused input_layernorm + in_proj decode
        path (esimd_resadd_norm_gemv2_fp8_pert) can supply the projected states
        directly, skipping the standalone in_proj GEMV launches.
        """
        # --- XPU native conv1d+GDN fast path (sgl_kernel.gdn_attention) ---
        # Cherry-picked from origin/dev 7680aecdd4. Env-gated; falls back to
        # the default Triton path if the kernel isn't usable on this shape.
        # Prefill(extend)-only: decode keeps the tuned ESIMD recurrent path
        # below (esimd_gdn_conv_fused_seq); the native Xe2 kernel is a chunked
        # (parallel) algorithm that only wins on long prefill sequences.
        _ENABLE_XPU_FAST_PATH = _XPU_GDN_FAST_PATH
        if (
            _ENABLE_XPU_FAST_PATH
            and _is_xpu
            and not forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
            and self.num_v_heads % self.num_k_heads == 0
        ):
            output = self._forward_xpu_fast_path(
                projected_states_qkvz,
                projected_states_ba,
                forward_batch,
            )
            if output is not None:
                return output

        # --- XPU ESIMD GDN decode fast path (esimd_gdn_conv_fused_seq) ---
        # Calls the BMG-validated ESIMD kernel that fuses conv1d + the GDN
        # recurrence in one launch. Sequential [q|k|v|z] layout — no gather.
        # Decode-only; prefill stays on the Triton/PyTorch path.
        _ENABLE_XPU_GDN_ESIMD = _XPU_GDN_ESIMD
        if (
            _ENABLE_XPU_GDN_ESIMD
            and _is_xpu
            and forward_batch.forward_mode.is_decode()
            and not forward_batch.forward_mode.is_target_verify()
        ):
            output = self._forward_xpu_esimd_gdn_decode(
                projected_states_qkvz,
                projected_states_ba,
                forward_batch,
            )
            if output is not None:
                return output

        if (
            self.num_v_heads // self.num_k_heads in [1, 2, 4]
            and not _is_cpu
            and not _is_npu
        ):
            mixed_qkv, z, b, a = fused_qkvzba_split_reshape_cat_contiguous(
                projected_states_qkvz,
                projected_states_ba,
                triton.cdiv(self.num_k_heads, self.attn_tp_size),
                triton.cdiv(self.num_v_heads, self.attn_tp_size),
                self.head_k_dim,
                self.head_v_dim,
            )
        elif _is_cpu and _is_amx_available:
            mixed_qkv, z, b, a = (
                torch.ops.sgl_kernel.fused_qkvzba_split_reshape_cat_contiguous_cpu(
                    projected_states_qkvz,
                    projected_states_ba,
                    self.num_k_heads // self.attn_tp_size,
                    self.num_v_heads // self.attn_tp_size,
                    self.head_k_dim,
                    self.head_v_dim,
                )
            )
        else:
            query, key, value, z, b, a = self.fix_query_key_value_ordering(
                projected_states_qkvz, projected_states_ba
            )
            b = b.contiguous()
            a = a.contiguous()

            query, key, value = map(
                lambda x: x.reshape(x.shape[0], -1), (query, key, value)
            )
            mixed_qkv = torch.cat((query, key, value), dim=-1)

        core_attn_out = self.attn(
            forward_batch,
            mixed_qkv=mixed_qkv,
            a=a,
            b=b,
        )

        z_shape_og = z.shape
        # reshape input data into 2D tensor
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])

        # Add padding for DP-Attn
        if core_attn_out.shape != z.shape:
            core_attn_out_pad = torch.zeros_like(z)
            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = core_attn_out_pad

        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(z_shape_og)
        core_attn_out = core_attn_out.reshape(*core_attn_out.shape[:-2], -1)

        output, _ = self.out_proj(core_attn_out)
        return output


class Qwen3_5LinearDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Linear Attention (GatedDeltaNet)."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.layer_id = layer_id

        linear_attn_quant_config = (
            None
            if quant_config and quant_config.get_name() == "modelopt_fp4"
            else quant_config
        )
        self.linear_attn = Qwen3_5GatedDeltaNet(
            config, layer_id, linear_attn_quant_config, alt_stream, prefix
        )

        # NOTE: Determine the MLP type based on the model type
        # Qwen3.5 use all layers for MLP / Qwen3.5-MoE use sparse MoE blocks
        if config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=(alt_stream if _disable_shared_experts_fusion() else None),
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
                is_nextn=is_nextn,
                support_shared_expert_fusion=not _disable_shared_experts_fusion(),
            )
            is_layer_sparse = True
            is_previous_layer_sparse = True
            is_next_layer_sparse = True
        elif config.model_type == "qwen3_5_text":
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
            )
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = False
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

    def _esimd_fused_input_norm_in_proj_q8_0(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        qkvz_lin,
        ba_lin,
        r0q,
        captured_last_layer_outputs: Optional[List[torch.Tensor]],
    ):
        """DEPRECATED (see _XPU_GGUF_RESADD_NORM_LEGACY).

        GGUF-q8_0 variant of the fused input_layernorm + GDN in_proj.

        ``in_proj_qkvz`` is q8_0; ``in_proj_ba`` is fp16 (GGUF keeps the tiny
        b/a tensors unquantized). A single-matrix q8_0 kernel cannot fuse both,
        so this fuses the GemmaRMSNorm into the q8_0 qkvz GEMV
        (``esimd_resadd_norm_gemv_q8_0``, which also writes ``normed_out`` and a
        separate ``new_residual``), then runs the fp16 ba projection on the
        kernel-written normed hidden (its fp16 matmul is already
        transpose-cached). Returns ``(qkvz, ba, new_residual)`` or ``None``.
        Shared guards are checked by the caller.

        Superseded by ``_gguf_norm_gemv()``: ``esimd_resadd_norm_gemv_q8_ba``
        folds the fp16 ba GEMV into the same launch, so the shipped path costs
        one op call here instead of two. ``esimd_resadd_norm_gemv_q8_0`` was
        never landed, so this always ImportErrors and falls back.
        """
        try:
            from custom_esimd_kernels_sglang import esimd_resadd_norm_gemv_q8_0
        except ImportError:
            return None
        qs, scale = r0q
        K = hidden_states.shape[1]
        if qs.shape[1] != K:
            return None
        cache = getattr(self, "_esimd_resadd_q8_const", None)
        if cache is None:
            nw = (
                (self.input_layernorm.weight.data.to(torch.float32) + 1.0)
                .to(torch.float16)
                .contiguous()
            )
            o0 = torch.empty((1, qs.shape[0]), dtype=torch.float16, device=qs.device)
            normed = torch.empty((1, K), dtype=torch.float16, device=qs.device)
            nr = torch.empty((1, K), dtype=torch.float16, device=qs.device)
            cache = {
                "nw": nw,
                "qs": qs,
                "scale": scale,
                "o0": o0,
                "normed": normed,
                "nr": nr,
            }
            self._esimd_resadd_q8_const = cache
        h = (
            hidden_states
            if hidden_states.dtype == torch.float16
            else hidden_states.to(torch.float16)
        )
        h = h.contiguous()
        res = residual if residual.is_contiguous() else residual.contiguous()
        try:
            esimd_resadd_norm_gemv_q8_0(
                h,
                res,
                cache["nw"],
                cache["qs"],
                cache["scale"],
                cache["o0"],
                cache["normed"],
                cache["nr"],
                float(self.input_layernorm.variance_epsilon),
            )
        except Exception:
            return None
        # ba projection reads the normed hidden (matches in_proj_ba(normed) in
        # the standard prepare_attn -> in_proj flow). fp16 GGUF matmul path.
        try:
            o1, _ = ba_lin(cache["normed"])
        except Exception:
            return None
        new_residual = cache["nr"]
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.append(new_residual.clone())
        return cache["o0"], o1, new_residual

    def _esimd_fused_input_norm_in_proj(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[List[torch.Tensor]],
    ):
        """Fused GemmaRMSNorm(input_layernorm) + GDN in_proj (qkvz + ba).

        Replaces the prepare_attn ``input_layernorm`` launch plus the two
        ``in_proj`` fp8 GEMV launches with a single
        ``esimd_resadd_norm_gemv2_fp8_pert`` launch. Gated by
        ``SGL_XPU_GDN_RESADD_NORM=1``; decode single-token only.

        Returns ``(projected_qkvz, projected_ba, new_residual)`` on success, or
        ``None`` to fall back to the standard prepare_attn + in_proj path. Only
        the plain TP decode path (no input-scatter, no allreduce-fusion) is
        intercepted; every other configuration falls back.
        """
        # Phase 5c: GGUF q8_0 build (the fp8 path below never matches it).
        if _gguf_resadd_norm_guard(hidden_states, residual, forward_batch):
            gdn = self.linear_attn
            r = _gguf_norm_gemv(
                self,
                self.input_layernorm,
                gdn.in_proj_qkvz,
                gdn.in_proj_ba,
                hidden_states,
                residual,
                "gdn_in_proj",
            )
            if r is not None:
                o0, o1, new_residual = r
                if captured_last_layer_outputs is not None:
                    captured_last_layer_outputs.append(new_residual.clone())
                return o0, o1, new_residual
        # DEPRECATED legacy GGUF q8_0 path, opt-in only and never reached unless
        # SGL_XPU_GGUF_RESADD_NORM_LEGACY=1 (see the flag's comment).
        if _XPU_GGUF_RESADD_NORM_LEGACY and residual is not None and _gguf_resadd_norm_guard(
            hidden_states, residual, forward_batch
        ):
            gdn = self.linear_attn
            r0q = _esimd_q8_0_weight_scale(gdn.in_proj_qkvz)
            if r0q is not None:
                q8out = self._esimd_fused_input_norm_in_proj_q8_0(
                    hidden_states,
                    residual,
                    gdn.in_proj_qkvz,
                    gdn.in_proj_ba,
                    r0q,
                    captured_last_layer_outputs,
                )
                if q8out is not None:
                    return q8out
        if not _XPU_GDN_RESADD_NORM:
            return None
        if not (_is_xpu and forward_batch.forward_mode.is_decode()):
            return None
        if forward_batch.forward_mode.is_target_verify():
            return None
        # Single decode token only (kernel emits [1, N]).
        if hidden_states.dim() != 2 or hidden_states.shape[0] != 1:
            return None
        # First layer has no residual yet; kernel requires the residual add.
        if residual is None:
            return None
        # Only interceptable when prepare_attn reduces to plain input_layernorm.
        try:
            from sglang.srt.layers.communicator import get_attn_tp_context

            if get_attn_tp_context().input_scattered:
                return None
        except Exception:
            return None
        if getattr(hidden_states, "_sglang_needs_allreduce_fusion", False):
            return None
        gdn = self.linear_attn
        qkvz_lin = gdn.in_proj_qkvz
        ba_lin = gdn.in_proj_ba
        try:
            from custom_esimd_kernels_sglang import esimd_resadd_norm_gemv2_fp8_pert
        except ImportError:
            return None

        # Cache row-major [N, K] fp8 weights + per-tensor scales, the
        # (1 + weight) Gemma norm weight, and the reusable output buffers once
        # per layer. _esimd_fp8_weight_nk_scale handles both the block-quant
        # ([N, K]) and online per-tensor (stored [K, N], transposed) layouts and
        # validates shapes so a bad layout never reaches the kernel.
        cache = getattr(self, "_esimd_resadd_const", None)
        if cache is None:
            r0 = _esimd_fp8_weight_nk_scale(qkvz_lin)
            r1 = _esimd_fp8_weight_nk_scale(ba_lin)
            if r0 is None or r1 is None:
                return None
            w0, s0 = r0
            w1, s1 = r1
            # Contraction dim (K) must match the hidden size for both.
            K = hidden_states.shape[1]
            if w0.shape[1] != K or w1.shape[1] != K:
                return None
            # GemmaRMSNorm scales by (1 + weight); the kernel expects the
            # pre-folded weight.
            nw = (
                (self.input_layernorm.weight.data.to(torch.float32) + 1.0)
                .to(torch.float16)
                .contiguous()
            )
            o0 = torch.empty((1, w0.shape[0]), dtype=torch.float16, device=w0.device)
            o1 = torch.empty((1, w1.shape[0]), dtype=torch.float16, device=w1.device)
            # Per-layer buffer for the kernel-written post-add residual
            # (hidden + residual). Safe: each layer object owns a distinct nr, so
            # the kernel never reads and writes the same buffer within one launch
            # (input residual belongs to the *previous* layer's nr).
            nr = torch.empty(
                (1, hidden_states.shape[1]), dtype=torch.float16, device=w0.device
            )
            cache = {
                "nw": nw,
                "w0": w0,
                "s0": s0,
                "w1": w1,
                "s1": s1,
                "o0": o0,
                "o1": o1,
                "nr": nr,
            }
            self._esimd_resadd_const = cache

        h = (
            hidden_states
            if hidden_states.dtype == torch.float16
            else hidden_states.to(torch.float16)
        )
        h = h.contiguous()
        # ESIMD kernels read raw contiguous pointers; guard residual layout.
        res = residual if residual.is_contiguous() else residual.contiguous()
        try:
            esimd_resadd_norm_gemv2_fp8_pert(
                h,
                res,
                cache["nw"],
                cache["w0"],
                cache["s0"],
                cache["o0"],
                cache["w1"],
                cache["s1"],
                cache["o1"],
                cache["nr"],
                float(self.input_layernorm.variance_epsilon),
            )
        except Exception:
            return None
        # The kernel now writes the post-add residual (hidden + residual, fp16)
        # into cache["nr"] via its gid==0 group, eliminating the separate
        # aten::add dispatch.
        new_residual = cache["nr"]
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.append(new_residual.clone())
        return cache["o0"], cache["o1"], new_residual

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)

        fused = None
        if not forward_batch.forward_mode.is_idle():
            fused = self._esimd_fused_input_norm_in_proj(
                hidden_states,
                residual,
                forward_batch,
                kwargs.get("captured_last_layer_outputs", None),
            )

        if fused is not None:
            projected_qkvz, projected_ba, residual = fused
            hidden_states = self.linear_attn._forward_from_projected(
                projected_qkvz, projected_ba, forward_batch
            )
        else:
            hidden_states, residual = (
                self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                    hidden_states,
                    residual,
                    forward_batch,
                    captured_last_layer_outputs=kwargs.get(
                        "captured_last_layer_outputs", None
                    ),
                )
            )

            if not forward_batch.forward_mode.is_idle():
                hidden_states = self.linear_attn(
                    hidden_states,
                    forward_batch,
                )

        if _NAN_PROBE: _nan_probe(
            "gdn_attn_out",
            hidden_states,
            layer_id=self.layer_id,
            forward_batch=forward_batch,
            residual=residual,
        )
        # Fully Connected
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # Phase 3: fold the post_attention_layernorm (resadd + rmsnorm) into the
        # fused MoE router kernel, removing a per-layer dispatch. Falls back to
        # the standard prepare_mlp + mlp path when not applicable.
        fused_mlp = None
        if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
            fused_mlp = self.mlp.esimd_prepare_mlp_moe(
                hidden_states,
                residual,
                self.post_attention_layernorm,
                forward_batch,
                use_reduce_scatter,
                should_allreduce_fusion,
            )

        if fused_mlp is not None:
            hidden_states, residual = fused_mlp
        else:
            hidden_states, residual = self.layer_communicator.prepare_mlp(
                hidden_states, residual, forward_batch
            )
            if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch,
                    use_reduce_scatter,
                    should_allreduce_fusion,
                )
            else:
                hidden_states = self.mlp(
                    hidden_states, should_allreduce_fusion, use_reduce_scatter
                )
        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


class Qwen3_5AttentionDecoderLayer(nn.Module):
    """Qwen3.5 Decoder Layer with Full Attention."""

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.attn_tp_rank = get_attention_tp_rank()
        self.attn_tp_size = get_attention_tp_size()
        self.total_num_heads = config.num_attention_heads
        assert self.total_num_heads % self.attn_tp_size == 0
        self.num_heads = self.total_num_heads // self.attn_tp_size
        self.total_num_kv_heads = config.num_key_value_heads
        if self.total_num_kv_heads >= self.attn_tp_size:
            assert self.total_num_kv_heads % self.attn_tp_size == 0
        else:
            assert self.attn_tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // self.attn_tp_size)
        self.head_dim = config.head_dim or (self.hidden_size // self.num_heads)
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.max_position_embeddings = getattr(config, "max_position_embeddings", 8192)

        self.rope_theta, rope_scaling = get_rope_config(config)
        self.partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
        self.layer_id = layer_id

        # If rope_scaling doesn't specify a scaling type, treat as no scaling
        if rope_scaling and not ("rope_type" in rope_scaling or "type" in rope_scaling):
            rope_scaling = None

        self.attn_output_gate = getattr(config, "attn_output_gate", True)
        if self.attn_output_gate:
            logger.warning_once("using attn output gate!")

        self.rotary_emb = get_rope(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position=self.max_position_embeddings,
            rope_scaling=rope_scaling,
            base=self.rope_theta,
            partial_rotary_factor=self.partial_rotary_factor,
            is_neox_style=True,
            dtype=torch.get_default_dtype(),
        )

        attn_quant_config = (
            None
            if quant_config and quant_config.get_name() == "modelopt_fp4"
            else quant_config
        )

        self.qkv_proj = QKVParallelLinear(
            config.hidden_size,
            self.head_dim,
            self.total_num_heads * (1 + self.attn_output_gate),
            self.total_num_kv_heads,
            bias=False,
            quant_config=attn_quant_config,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix),
        )

        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            config.hidden_size,
            bias=False,
            quant_config=attn_quant_config,
            reduce_results=False,
            tp_rank=self.attn_tp_rank,
            tp_size=self.attn_tp_size,
            prefix=add_prefix("o_proj", prefix),
        )

        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=f"{prefix}.attn",
        )

        # Dense MLP for non-MoE variant
        if config.model_type == "qwen3_5_text":
            self.mlp = Qwen2MoeMLP(
                hidden_size=config.hidden_size,
                intermediate_size=config.intermediate_size,
                hidden_act=config.hidden_act,
                quant_config=quant_config,
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
            )
            is_layer_sparse = False
            is_previous_layer_sparse = False
            is_next_layer_sparse = False
        elif config.model_type == "qwen3_5_moe_text":
            self.mlp = Qwen2MoeSparseMoeBlock(
                layer_id=layer_id,
                config=config,
                quant_config=quant_config,
                alt_stream=(alt_stream if _disable_shared_experts_fusion() else None),
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
                is_nextn=is_nextn,
                support_shared_expert_fusion=not _disable_shared_experts_fusion(),
            )
            is_layer_sparse = True
            is_previous_layer_sparse = True
            is_next_layer_sparse = True
        else:
            raise ValueError(f"Invalid model type: {config.model_type}")

        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=is_layer_sparse,
            is_previous_layer_sparse=is_previous_layer_sparse,
            is_next_layer_sparse=is_next_layer_sparse,
        )

        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
            allow_reduce_scatter=True,
            is_last_layer=(layer_id == config.num_hidden_layers - 1),
        )

        self.alt_stream = alt_stream

    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply Q/K normalization with optional alt_stream overlap."""
        if (
            self.alt_stream is not None
            and get_is_capture_mode()
            and _qknorm_use_alt_stream
        ):
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        elif _is_hip:
            q_by_head, k_by_head = fused_qk_gemma_rmsnorm(
                q,
                k,
                self.q_norm.weight.data,
                self.k_norm.weight.data,
                self.q_norm.variance_epsilon,
                self.head_dim,
            )
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k

    def forward_prepare_native(self, positions, hidden_states):
        qkv, _ = self.qkv_proj(hidden_states)
        if self.attn_output_gate:
            q_gate, k, v = qkv.split(
                [self.q_size * 2, self.kv_size, self.kv_size], dim=-1
            )
            orig_shape = q_gate.shape[:-1]
            q_gate = q_gate.view(*orig_shape, self.num_heads, -1)
            q, gate = torch.chunk(q_gate, 2, dim=-1)
            q = q.reshape(*orig_shape, -1)
            gate = gate.reshape(*orig_shape, -1)
        else:
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        return q, k, v, gate

    def forward_prepare_npu(self, positions, hidden_states, forward_batch):
        qkv, _ = self.qkv_proj(hidden_states)
        # Calculate first full attention layer ID based on config
        if self.attn.layer_id == (self.config.full_attention_interval - 1):
            self.rotary_emb.get_cos_sin_with_position(positions)

        q, k, v, gate = split_qkvgate_gemma_rmsnorm_rope(
            qkv,
            self.rotary_emb.position_sin,
            self.rotary_emb.position_cos,
            self.q_size,
            self.kv_size,
            self.head_dim,
            int(self.head_dim * self.partial_rotary_factor),
            eps=self.q_norm.variance_epsilon,
            q_weight=self.q_norm.weight,
            k_weight=self.k_norm.weight,
        )
        return q, k, v, gate

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
        precomputed_qkv: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Full attention forward pass.

        When ``precomputed_qkv`` is provided (Phase 5b fused input-norm + qkv
        path), the ``qkv_proj`` GEMV is skipped and the supplied ``[1, N]`` fp16
        qkv tensor is fed straight into the ESIMD split/norm/rope path.
        """
        # vllm parity: fuse split + qk_norm + rope into single ESIMD call.
        # Hard-coded requirements: head_dim=256, fp16 model, GemmaRMSNorm
        # weight+1.0 convention. The kernel is fp16-only; on a bf16 model the
        # implicit cast loses dynamic range during RoPE and gsm8k drops from
        # 0.80 to 0.40 (verified). Restrict to fp16 inputs only.
        if (
            _XPU_FA_ESIMD_QKV
            and self.head_dim == 256
            and (
                precomputed_qkv is not None
                or (hidden_states.dim() == 2 and hidden_states.dtype == torch.float16)
            )
        ):
            try:
                # Prefer the BMG sglang variant; fall back to the vllm one
                # if installed in this env.
                try:
                    from custom_esimd_kernels_sglang import (
                        esimd_qkv_split_norm_rope,
                    )
                except ImportError:
                    from custom_esimd_kernels_vllm import esimd_qkv_split_norm_rope
            except ImportError:
                esimd_qkv_split_norm_rope = None
            if esimd_qkv_split_norm_rope is not None:
                if precomputed_qkv is not None:
                    qkv = precomputed_qkv
                else:
                    qkv, _ = self.qkv_proj(hidden_states)
                nTokens = qkv.shape[0]
                orig_dtype = qkv.dtype
                qkv_fp16 = qkv.to(torch.float16).contiguous()
                # Cache the constant fp16 casts (norm weights, cos/sin cache)
                # once. Recomputing .to(fp16).contiguous() every call both wastes
                # time and allocates fresh tensors mid-graph-capture.
                cache = getattr(self, "_esimd_qkv_const", None)
                if cache is None:
                    cs = self.rotary_emb.cos_sin_cache
                    cache = {
                        "q_norm": self.q_norm.weight.to(torch.float16).contiguous(),
                        "k_norm": self.k_norm.weight.to(torch.float16).contiguous(),
                        "cos_sin": cs.to(torch.float16) if cs.dtype != torch.float16 else cs,
                        "rotary_dim": int(
                            self.head_dim
                            * getattr(self.config, "partial_rotary_factor", 1.0)
                        ),
                    }
                    self._esimd_qkv_const = cache
                # Pre-allocate the q/k/v/gate output scratch, keyed by token
                # count, and reuse across XPU-graph replays so the buffers' data
                # ptrs stay stable. Graphs are captured per batch size, so a
                # given replay always sees the nTokens it was captured with.
                scratch = getattr(self, "_esimd_qkv_scratch", None)
                if scratch is None or scratch[0] != nTokens:
                    q_out = torch.empty(
                        (nTokens, self.num_heads * 256),
                        device=qkv.device, dtype=torch.float16,
                    )
                    gate_out = (
                        torch.empty(
                            (nTokens, self.num_heads * 256),
                            device=qkv.device, dtype=torch.float16,
                        )
                        if self.attn_output_gate
                        else torch.empty(0, device=qkv.device, dtype=torch.float16)
                    )
                    k_out = torch.empty(
                        (nTokens, self.num_kv_heads * 256),
                        device=qkv.device, dtype=torch.float16,
                    )
                    v_out = torch.empty(
                        (nTokens, self.num_kv_heads * 256),
                        device=qkv.device, dtype=torch.float16,
                    )
                    self._esimd_qkv_scratch = (
                        nTokens, q_out, gate_out, k_out, v_out
                    )
                else:
                    _, q_out, gate_out, k_out, v_out = scratch
                # Phase 5c: positions is identical across all full-attention
                # layers within a decode step, so the int32 conversion (a real
                # dispatched copy) is redundant after the first layer. Memoize it
                # on the per-step forward_batch (fresh each step -> auto-invalidates)
                # keyed by the positions object identity. Saves ~9/step copy_.
                pos_i32 = getattr(forward_batch, "_esimd_pos_i32", None)
                if pos_i32 is None or getattr(
                    forward_batch, "_esimd_pos_id", None
                ) != id(positions):
                    pos_i32 = positions.to(torch.int32).contiguous()
                    try:
                        forward_batch._esimd_pos_i32 = pos_i32
                        forward_batch._esimd_pos_id = id(positions)
                    except Exception:
                        pass
                esimd_qkv_split_norm_rope(
                    qkv_fp16,
                    q_out, gate_out, k_out, v_out,
                    cache["q_norm"],
                    cache["k_norm"],
                    pos_i32,
                    self.num_heads, self.num_kv_heads,
                    self.attn_output_gate,
                    cache["rotary_dim"], cache["cos_sin"],
                )
                q = q_out if q_out.dtype == orig_dtype else q_out.to(orig_dtype)
                k = k_out if k_out.dtype == orig_dtype else k_out.to(orig_dtype)
                v = v_out if v_out.dtype == orig_dtype else v_out.to(orig_dtype)
                gate = (
                    (gate_out if gate_out.dtype == orig_dtype else gate_out.to(orig_dtype))
                    if self.attn_output_gate
                    else None
                )
                attn_output = self.attn(q, k, v, forward_batch)
                if self.attn_output_gate:
                    # ESIMD kernel already applies sigmoid; don't re-sigmoid.
                    attn_output = attn_output * gate
                output, _ = self.o_proj(attn_output)
                return output

        if (
            not _is_npu
            or forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed()
            or not self.attn_output_gate
        ):
            q, k, v, gate = self.forward_prepare_native(
                positions=positions,
                hidden_states=hidden_states,
            )
        else:
            q, k, v, gate = self.forward_prepare_npu(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        attn_output = self.attn(q, k, v, forward_batch)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output

    def _esimd_fused_input_norm_qkv_q8_0(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        r0q,
        captured_last_layer_outputs: Optional[List[torch.Tensor]],
    ):
        """DEPRECATED (see _XPU_GGUF_RESADD_NORM_LEGACY).

        GGUF-q8_0 variant of the fused input_layernorm + qkv_proj.

        Uses the single-matrix ``esimd_resadd_norm_gemv_q8_0`` op (int8 weight +
        per-32-block fp16 scale). ``normed_out`` is a scratch buffer here (the
        full-attn path feeds ``o0`` straight into ``esimd_qkv_split_norm_rope``
        and does not reuse the normed hidden). Returns ``(qkv, new_residual)`` or
        ``None`` to fall back. Shared guards are checked by the caller.

        Superseded by ``_gguf_norm_gemv()`` on ``esimd_resadd_norm_gemv_q8_ba``.
        ``esimd_resadd_norm_gemv_q8_0`` was never landed in
        custom-esimd-kernels, so this always ImportErrors and falls back.
        """
        try:
            from custom_esimd_kernels_sglang import esimd_resadd_norm_gemv_q8_0
        except ImportError:
            return None
        qs, scale = r0q
        K = hidden_states.shape[1]
        if qs.shape[1] != K:
            return None
        cache = getattr(self, "_esimd_fa_norm_q8_const", None)
        if cache is None:
            nw = (
                (self.input_layernorm.weight.data.to(torch.float32) + 1.0)
                .to(torch.float16)
                .contiguous()
            )
            o0 = torch.empty((1, qs.shape[0]), dtype=torch.float16, device=qs.device)
            normed = torch.empty((1, K), dtype=torch.float16, device=qs.device)
            nr = torch.empty((1, K), dtype=torch.float16, device=qs.device)
            cache = {
                "nw": nw,
                "qs": qs,
                "scale": scale,
                "o0": o0,
                "normed": normed,
                "nr": nr,
            }
            self._esimd_fa_norm_q8_const = cache
        h = (
            hidden_states
            if hidden_states.dtype == torch.float16
            else hidden_states.to(torch.float16)
        )
        h = h.contiguous()
        res = residual if residual.is_contiguous() else residual.contiguous()
        try:
            esimd_resadd_norm_gemv_q8_0(
                h,
                res,
                cache["nw"],
                cache["qs"],
                cache["scale"],
                cache["o0"],
                cache["normed"],
                cache["nr"],
                float(self.input_layernorm.variance_epsilon),
            )
        except Exception:
            return None
        new_residual = cache["nr"]
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.append(new_residual.clone())
        return cache["o0"], new_residual

    def _esimd_fused_input_norm_qkv(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[List[torch.Tensor]],
    ):
        """Fused GemmaRMSNorm(input_layernorm) + qkv_proj (Phase 5b).

        Replaces the prepare_attn ``input_layernorm`` launch
        (gemma_fused_add_rmsnorm) plus the ``qkv_proj`` fp8 GEMV with a single
        ``esimd_resadd_norm_gemv2_fp8_pert`` launch (qkv as matrix-0, a 1-row
        dummy as matrix-1), with the post-add residual written back by the
        kernel. Gated by ``SGL_XPU_FA_RESADD_NORM=1``; decode single-token, fp8
        qkv only. Requires the ESIMD qkv split path (head_dim==256, fp16) since
        the resulting qkv feeds ``esimd_qkv_split_norm_rope``.

        Returns ``(qkv, new_residual)`` on success, or ``None`` to fall back to
        the standard prepare_attn + self_attention path.
        """
        # Phase 5c: GGUF q8_0 build (the fp8 path below never matches it).
        if _gguf_resadd_norm_guard(hidden_states, residual, forward_batch):
            r = _gguf_norm_gemv(
                self,
                self.input_layernorm,
                # self_attention is a method on this layer, not a submodule:
                # qkv_proj hangs off the layer itself.
                self.qkv_proj,
                None,
                hidden_states,
                residual,
                "fa_qkv",
            )
            if r is not None:
                o0, _, new_residual = r
                if captured_last_layer_outputs is not None:
                    captured_last_layer_outputs.append(new_residual.clone())
                return o0, new_residual
        # DEPRECATED legacy GGUF q8_0 path, opt-in only and never reached unless
        # SGL_XPU_GGUF_RESADD_NORM_LEGACY=1 (see the flag's comment).
        if (
            _XPU_GGUF_RESADD_NORM_LEGACY
            and _XPU_FA_ESIMD_QKV
            and residual is not None
            and _gguf_resadd_norm_guard(hidden_states, residual, forward_batch)
        ):
            r0q = _esimd_q8_0_weight_scale(self.qkv_proj)
            if r0q is not None:
                q8out = self._esimd_fused_input_norm_qkv_q8_0(
                    hidden_states, residual, r0q, captured_last_layer_outputs
                )
                if q8out is not None:
                    return q8out
        if not (_XPU_FA_RESADD_NORM and _XPU_FA_ESIMD_QKV):
            return None
        if not (_is_xpu and forward_batch.forward_mode.is_decode()):
            return None
        if forward_batch.forward_mode.is_target_verify():
            return None
        if self.head_dim != 256:
            return None
        # Single decode token only (kernel emits [1, N]).
        if hidden_states.dim() != 2 or hidden_states.shape[0] != 1:
            return None
        # First layer has no residual yet; kernel requires the residual add.
        if residual is None:
            return None
        # Only interceptable when prepare_attn reduces to plain input_layernorm.
        try:
            from sglang.srt.layers.communicator import get_attn_tp_context

            if get_attn_tp_context().input_scattered:
                return None
        except Exception:
            return None
        if getattr(hidden_states, "_sglang_needs_allreduce_fusion", False):
            return None
        try:
            from custom_esimd_kernels_sglang import esimd_resadd_norm_gemv2_fp8_pert
        except ImportError:
            return None

        # Cache row-major [N, K] fp8 qkv weight + per-tensor scale, the
        # (1 + weight) Gemma norm weight, the reusable output/residual buffers,
        # and a 1-row dummy second matrix (the deployed op fuses two matrices;
        # the dummy contributes a discarded zero column but lets us reuse the op
        # without a new kernel). All static after load.
        cache = getattr(self, "_esimd_fa_norm_const", None)
        if cache is None:
            r0 = _esimd_fp8_weight_nk_scale(self.qkv_proj)
            if r0 is None:
                return None
            w0, s0 = r0
            K = hidden_states.shape[1]
            if w0.shape[1] != K:
                return None
            nw = (
                (self.input_layernorm.weight.data.to(torch.float32) + 1.0)
                .to(torch.float16)
                .contiguous()
            )
            o0 = torch.empty((1, w0.shape[0]), dtype=torch.float16, device=w0.device)
            # 1-row dummy second matrix (zeros -> contributes nothing).
            w1 = torch.zeros((1, K), dtype=w0.dtype, device=w0.device)
            s1 = torch.ones((1,), dtype=torch.float32, device=w0.device)
            o1 = torch.empty((1, 1), dtype=torch.float16, device=w0.device)
            nr = torch.empty((1, K), dtype=torch.float16, device=w0.device)
            cache = {
                "nw": nw,
                "w0": w0,
                "s0": s0,
                "w1": w1,
                "s1": s1,
                "o0": o0,
                "o1": o1,
                "nr": nr,
            }
            self._esimd_fa_norm_const = cache

        h = (
            hidden_states
            if hidden_states.dtype == torch.float16
            else hidden_states.to(torch.float16)
        )
        h = h.contiguous()
        res = residual if residual.is_contiguous() else residual.contiguous()
        try:
            esimd_resadd_norm_gemv2_fp8_pert(
                h,
                res,
                cache["nw"],
                cache["w0"],
                cache["s0"],
                cache["o0"],
                cache["w1"],
                cache["s1"],
                cache["o1"],
                cache["nr"],
                float(self.input_layernorm.variance_epsilon),
            )
        except Exception:
            return None
        new_residual = cache["nr"]
        if captured_last_layer_outputs is not None:
            captured_last_layer_outputs.append(new_residual.clone())
        return cache["o0"], new_residual

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ):
        # Phase 5b: try to fuse input_layernorm (resadd + rmsnorm) into the
        # qkv_proj GEMV, replacing prepare_attn's gemma_fused_add_rmsnorm launch.
        # On a hit we get qkv + new_residual directly and feed qkv straight into
        # self_attention, skipping prepare_attn and qkv_proj. Falls back to the
        # standard prepare_attn path otherwise.
        fused_qkv = None
        if not forward_batch.forward_mode.is_idle():
            fused_qkv = self._esimd_fused_input_norm_qkv(
                hidden_states,
                residual,
                forward_batch,
                captured_last_layer_outputs,
            )

        if fused_qkv is not None:
            qkv, residual = fused_qkv
            hidden_states = self.self_attention(
                positions=positions,
                hidden_states=None,
                forward_batch=forward_batch,
                precomputed_qkv=qkv,
            )
        else:
            hidden_states, residual = (
                self.layer_communicator.prepare_attn_and_capture_last_layer_outputs(
                    hidden_states,
                    residual,
                    forward_batch,
                    captured_last_layer_outputs=captured_last_layer_outputs,
                )
            )

            if not forward_batch.forward_mode.is_idle():
                hidden_states = self.self_attention(
                    positions=positions,
                    hidden_states=hidden_states,
                    forward_batch=forward_batch,
                )

        if _NAN_PROBE: _nan_probe(
            "full_attn_out",
            hidden_states,
            layer_id=self.layer_id,
            forward_batch=forward_batch,
            residual=residual,
        )
        # Fully Connected
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )
        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
        )

        # Phase 3: fold the post_attention_layernorm (resadd + rmsnorm) into the
        # fused MoE router kernel, removing a per-layer dispatch. Falls back to
        # the standard prepare_mlp + mlp path when not applicable.
        fused_mlp = None
        if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
            fused_mlp = self.mlp.esimd_prepare_mlp_moe(
                hidden_states,
                residual,
                self.post_attention_layernorm,
                forward_batch,
                use_reduce_scatter,
                should_allreduce_fusion,
            )

        if fused_mlp is not None:
            hidden_states, residual = fused_mlp
        else:
            hidden_states, residual = self.layer_communicator.prepare_mlp(
                hidden_states, residual, forward_batch
            )
            if isinstance(self.mlp, Qwen2MoeSparseMoeBlock):
                hidden_states = self.mlp(
                    hidden_states,
                    forward_batch,
                    use_reduce_scatter,
                    should_allreduce_fusion,
                )
            else:
                hidden_states = self.mlp(
                    hidden_states, should_allreduce_fusion, use_reduce_scatter
                )
        if should_allreduce_fusion:
            hidden_states._sglang_needs_allreduce_fusion = True
        else:
            hidden_states, residual = self.layer_communicator.postprocess_layer(
                hidden_states, residual, forward_batch
            )

        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3_5AttentionDecoderLayer,
    "linear_attention": Qwen3_5LinearDecoderLayer,
}


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 Model with support for dense variant."""

    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
        "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
        "in_proj_ba": ["in_proj_b", "in_proj_a"],
    }

    supported_lora_modules = [
        "qkv_proj",
        "o_proj",
        "out_proj",
        "in_proj_qkvz",
        "gate_up_proj",
        "down_proj",
        "lm_head",
    ]

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        config = self.config
        head_dim = config.head_dim or (config.hidden_size // config.num_attention_heads)

        if module_name == "qkv_proj":
            attn_output_gate = getattr(config, "attn_output_gate", True)
            q_heads = config.num_attention_heads * (2 if attn_output_gate else 1)
            return (
                config.hidden_size,
                head_dim * (q_heads + config.num_key_value_heads * 2),
            )
        elif module_name == "o_proj":
            return config.num_attention_heads * head_dim, config.hidden_size
        elif module_name == "out_proj":
            value_dim = config.linear_value_head_dim * config.linear_num_value_heads
            return value_dim, config.hidden_size
        elif module_name == "in_proj_qkvz":
            key_dim = config.linear_key_head_dim * config.linear_num_key_heads
            value_dim = config.linear_value_head_dim * config.linear_num_value_heads
            return config.hidden_size, key_dim * 2 + value_dim * 2
        elif module_name == "gate_up_proj":
            # MoE: shared expert uses shared_expert_intermediate_size
            # Dense: regular MLP uses intermediate_size
            is_moe = "moe" in getattr(config, "model_type", "")
            if is_moe:
                inter = config.shared_expert_intermediate_size
            else:
                inter = config.intermediate_size
            return config.hidden_size, inter * 2
        elif module_name == "down_proj":
            is_moe = "moe" in getattr(config, "model_type", "")
            if is_moe:
                inter = config.shared_expert_intermediate_size
            else:
                inter = config.intermediate_size
            return inter, config.hidden_size
        elif module_name == "gate_up_proj_moe":
            return config.hidden_size, config.moe_intermediate_size * 2
        elif module_name == "down_proj_moe":
            return config.moe_intermediate_size, config.hidden_size
        elif module_name == "embed_tokens":
            return config.vocab_size, config.hidden_size
        elif module_name == "lm_head":
            return config.hidden_size, config.vocab_size
        else:
            raise NotImplementedError(
                f"get_hidden_dim not implemented for {module_name}"
            )

    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        is_nextn: bool = False,
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.pp_group = get_pp_group()

        alt_stream = torch.cuda.Stream() if _is_cuda or _hip_use_alt_stream else None

        # Embedding layer
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                enable_tp=not is_dp_attention_enabled(),
                # Pass quant_config so a GGUF checkpoint routes embed_tokens
                # through the GGUF embedding method; without it the layer is
                # Unquantized and its dense .weight is never filled by the GGUF
                # loader (which provides qweight), yielding all-zero embeddings.
                quant_config=quant_config,
                prefix=add_prefix("embed_tokens", prefix),
            )
        else:
            self.embed_tokens = PPMissingLayer()

        # Decoder layers
        def get_layer(idx: int, prefix: str):
            layer_type = config.layers_block_type[idx]
            layer_class = ALL_DECODER_LAYER_TYPES[layer_type]
            if layer_type == "attention":
                prefix = add_prefix("self_attn", prefix)
            else:
                prefix = add_prefix("linear_attn", prefix)
            return layer_class(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
                alt_stream=alt_stream,
                is_nextn=is_nextn,
            )

        self.layers, self._start_layer, self._end_layer = make_layers(
            config.num_hidden_layers,
            get_layer,
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=f"{prefix}.layers",
        )

        # Final normalization
        if self.pp_group.is_last_rank:
            self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.layers_to_capture = []

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_dflash_layers_to_capture(self, layers_to_capture: list[int]):
        self.layers_to_capture = layers_to_capture
        for layer_id in self.layers_to_capture:
            setattr(self.layers[layer_id], "_is_layer_to_capture", True)

    @property
    def start_layer(self) -> int:
        return self._start_layer

    @property
    def end_layer(self) -> int:
        return self._end_layer

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
        input_deepstack_embeds: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, PPProxyTensors]:
        # Initialize hidden states
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

        aux_hidden_states = []
        if _NAN_PROBE:
            _nan_probe_new_forward()
            _nan_probe("embed_out", hidden_states, forward_batch=forward_batch)
        # Pass through decoder layers
        for layer_idx in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_idx]
            with get_global_expert_distribution_recorder().with_current_layer(
                layer_idx
            ):
                hidden_states, residual = layer(
                    positions=positions,
                    hidden_states=hidden_states,
                    residual=residual,
                    forward_batch=forward_batch,
                    captured_last_layer_outputs=(
                        aux_hidden_states
                        if getattr(layer, "_is_layer_to_capture", False)
                        else None
                    ),
                )
            if _NAN_PROBE: _nan_probe(
                "layer_out",
                hidden_states,
                layer_id=layer_idx,
                forward_batch=forward_batch,
                residual=residual,
            )

            # Process deepstack embeddings if provided
            if (
                input_deepstack_embeds is not None
                and input_deepstack_embeds.numel() > 0
                and layer_idx < 3
            ):
                sep = self.hidden_size * layer_idx
                hidden_states.add_(
                    input_deepstack_embeds[:, sep : sep + self.hidden_size]
                )

        # Return intermediate tensors for pipeline parallelism
        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": residual,
                }
            )

        # Apply final normalization
        if hidden_states.shape[0] != 0:
            if residual is None:
                hidden_states = self.norm(hidden_states)
            else:
                hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) == 0:
            return hidden_states

        return hidden_states, aux_hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        return ModelConfigForExpertLocation(
            num_layers=config.num_hidden_layers,
            num_logical_experts=config.num_experts,
            num_groups=None,
        )


class Qwen3_5MoeForCausalLM(Qwen3_5ForCausalLM):
    def __init__(
        self,
        config: Qwen3_5TextConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=self.config.num_experts,
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            ".weight_scale",
            "_weight_scale",
            ".input_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        num_experts = self.config.num_experts

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if "visual" in name:
                continue
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if "experts.gate_up_proj" in name or "experts.down_proj" in name:
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

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
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        if "experts.gate_up_proj" in name:
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        else:
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                    else:
                        # Skip loading extra parameters for GPTQ/modelopt models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)

        return loaded_params


class Qwen3_5ForConditionalGeneration(Qwen3VLForConditionalGeneration):

    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    hf_to_sglang_mapper = None

    supported_lora_modules = Qwen3_5ForCausalLM.supported_lora_modules

    def __init__(
        self,
        config: Qwen3_5Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5ForCausalLM,
    ):
        super().__init__(config, quant_config, prefix, language_model_cls)

        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config

        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        return self.model.get_hidden_dim(module_name, layer_idx)

    def should_apply_lora(self, module_name: str) -> bool:
        return module_name.startswith("model.layers.")

    @property
    def start_layer(self) -> int:
        return getattr(getattr(self, "model", None), "start_layer", 0)

    @property
    def end_layer(self) -> int:
        model = getattr(self, "model", None)
        end_layer = getattr(model, "end_layer", None)
        if end_layer is not None:
            return end_layer
        cfg = getattr(model, "config", None)
        return int(getattr(cfg, "num_hidden_layers", 0))

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
        head = self.lm_head.weight if self.pp_group.is_last_rank else None
        return embed, head

    def set_embed_and_head(self, embed, head):
        if self.pp_group.is_first_rank and embed is not None:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        if self.pp_group.is_last_rank and head is not None:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN fused projections
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        # GGUF load-path detection. A GGUF checkpoint stores weights in
        # llama.cpp conventions that differ from the HF safetensors this model
        # code expects; the fixups below mirror the MoE class one-for-one,
        # minus shared_expert_gate (no MoE layers in the dense arch):
        #   * GemmaRMSNorm weights are stored standard (~1.0), but GemmaRMSNorm
        #     computes x*(1+w) so the param must be (standard-1) -> subtract 1.
        #   * GDN linear_attn.* needs the value-head permute / A_log / dt_bias
        #     transform (_gguf_gdn_transform).
        #   * conv1d is stored 2-D [ch, kernel] but the param is 3-D.
        _is_gguf = (
            getattr(self, "quant_config", None) is not None
            and getattr(self.quant_config, "get_name", lambda: "")() == "gguf"
        )
        # The GDN linear_attn.norm uses plain RMSNormGated (no offset) -- exclude.
        _gemma_norm_suffixes = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        )

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if _is_gguf and (
                name.endswith(_gemma_norm_suffixes)
                or name == "model.language_model.norm.weight"
                or name == "model.norm.weight"
            ):
                loaded_weight = loaded_weight - 1.0
            if _is_gguf and ".linear_attn." in name:
                loaded_weight = self._gguf_gdn_transform(name, loaded_weight)
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)
            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "visual" in name or "mlp.experts" in name:
                    continue

                name = name.replace(weight_name, param_name)
                # GGUF F32 GDN gate shards (ssm_beta/ssm_alpha -> in_proj_b/a)
                # are yielded as `.weight`: the gguf iterator only renames
                # non-F32 tensors to `.qweight`. But the fused `in_proj_ba` is a
                # GGUF quantized module whose merged param is `.qweight`, so the
                # F32 `...in_proj_ba.weight` target is absent from params_dict
                # and the shard would otherwise fall through to the non-stacked
                # branch, losing its shard_id (-> shard_id=[None, None] and an
                # unsortable merge in GGUFLinearXPUMethod). Redirect here,
                # inside the stacked loop, so the shard_id is preserved.
                if (
                    _is_gguf
                    and name.endswith(".weight")
                    and name not in params_dict
                    and (name[: -len(".weight")] + ".qweight") in params_dict
                ):
                    name = name[: -len(".weight")] + ".qweight"
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Skip layers on other devices.
                # if is_pp_missing_parameter(name, self):
                #     continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                if "visual" in name:
                    # adapt to VisionAttention
                    name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                    name = name.replace(r"model.visual.", r"visual.")

                # print(name, loaded_weight.shape)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if name not in params_dict:
                    logger.warning(f"Parameter {name} not found in params_dict")
                    continue
                param = params_dict[name]

                # GGUF stores conv1d 2-D [ch, kernel] but the param is
                # 3-D [ch, 1, kernel]; insert the singleton middle dim.
                if (
                    _is_gguf
                    and "conv1d.weight" in name
                    and loaded_weight.dim() == 2
                    and param.dim() == 3
                ):
                    loaded_weight = loaded_weight.unsqueeze(1)

                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
                if (
                    self.config.tie_word_embeddings
                    and name == "model.embed_tokens.weight"
                    and (_is_cpu and _is_amx_available)
                ):
                    param_lm_head = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        param_lm_head, "weight_loader", default_weight_loader
                    )
                    weight_loader(param_lm_head, loaded_weight)
            loaded_params.add(name)
        return loaded_params


class Qwen3_5MoeForConditionalGeneration(Qwen3VLForConditionalGeneration):
    """Qwen3.5 MoE Vision-Language Model."""

    packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
    hf_to_sglang_mapper = None

    supported_lora_modules = Qwen3_5ForCausalLM.supported_lora_modules

    def __init__(
        self,
        config: Qwen3_5MoeConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        language_model_cls=Qwen3_5MoeForCausalLM,
    ) -> None:
        super().__init__(config, quant_config, prefix, language_model_cls)
        rope_config = getattr(self.config, "rope_parameters", None) or getattr(
            self.config, "rope_scaling", {}
        )
        self.is_mrope_enabled = "mrope_section" in rope_config

        self.deepstack_visual_indexes = self.visual.deepstack_visual_indexes
        self.num_fused_shared_experts = 0
        if _use_aiter and not _disable_shared_experts_fusion():
            self.num_fused_shared_experts = self._get_num_fused_shared_experts()

        self.enable_shared_expert_fusion = self.num_fused_shared_experts > 0

    def get_hidden_dim(self, module_name: str, layer_idx: int):
        return self.model.get_hidden_dim(module_name, layer_idx)

    def should_apply_lora(self, module_name: str) -> bool:
        # Accept all language model layer modules (attention, linear_attn, mlp).
        return module_name.startswith("model.layers.")

    def _get_num_fused_shared_experts(self):
        if not (
            hasattr(self.model, "layers")
            and len(self.model.layers) > 0
            and hasattr(self.model.layers[0].mlp, "num_fused_shared_experts")
        ):
            return 0
        return self.model.layers[0].mlp.num_fused_shared_experts

    def get_embed_and_head(self):
        embed = self.model.embed_tokens.weight if self.pp_group.is_first_rank else None
        head = self.lm_head.weight if self.pp_group.is_last_rank else None
        return embed, head

    def set_embed_and_head(self, embed, head):
        if self.pp_group.is_first_rank and embed is not None:
            del self.model.embed_tokens.weight
            self.model.embed_tokens.weight = embed
        if self.pp_group.is_last_rank and head is not None:
            del self.lm_head.weight
            self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    def _gguf_gdn_transform(
        self, name: str, w: torch.Tensor
    ) -> torch.Tensor:
        """Convert a GGUF GDN linear_attn weight to HF layout.

        GGUF orders the value-head dimension as [ratio, num_k_heads] whereas HF
        expects [num_k_heads, ratio] (ratio = num_v_heads // num_k_heads), so the
        value-head axis is re-permuted via reshape(ratio, num_k, ...).transpose.
        GGUF also stores the SSM decay as A (= -exp(A_log)); HF stores A_log.

        This runs on the GGUF tensor as delivered by the weight iterator, which
        is RAW QUANTIZED BYTES for quantized layers (name ends in ``.qweight``,
        shape ``[out_rows, block_bytes]``) and the real F32 values otherwise.
        Permuting whole *rows* (dim 0) is bit-identical on quantized bytes since
        GGUF packs each output row contiguously (verified: dequant∘rowperm ==
        rowperm∘dequant, max diff 0). Therefore every transform here is a dim-0
        row permutation only, EXCEPT ``out_proj``, which needs an INPUT-dim
        (column) permute; see that branch for how it avoids splitting q-blocks.
        The key-head q/k slices of
        in_proj_qkv / conv1d are NOT permuted. (notes §3.6.)
        """
        # The weight iterator also yields per-tensor ``qweight_type`` scalars
        # (0-dim) for quantized layers; those carry no head layout and must pass
        # through untouched.
        if w.dim() == 0:
            return w
        tc = getattr(self.config, "text_config", self.config)
        nk = tc.linear_num_key_heads
        nv = tc.linear_num_value_heads
        if nv % nk != 0:
            return w
        ratio = nv // nk
        # A_log is F32 (no .qweight); GGUF stores A, HF stores log(-A).
        if name.endswith("linear_attn.A_log"):
            w = torch.log(-w)
            return self._perm_value_rows(w, ratio, nk) if ratio > 1 else w
        if ratio == 1:
            return w  # value/key head layouts coincide; nothing else to do

        if name.endswith("linear_attn.dt_bias"):
            return self._perm_value_rows(w, ratio, nk)
        # out_proj: value-head columns are the INPUT dim. GGUF stores them in
        # [ratio, num_k] order; HF/core_attn_out expects [num_k, ratio]. This is
        # a dim-1 permute and MUST be done here on the GLOBAL pre-shard weight:
        # under TP the RowParallel input shard cuts value_dim contiguously in
        # GGUF order, but the HF value-head grouping crosses that boundary, so a
        # per-rank permute in the XPU method is impossible. The weight arrives as
        # RAW quantized bytes [out_rows, block_bytes] (Q8_0 on the 35B) or real
        # F32 [out_rows, value_dim]; either way each value head owns an equal,
        # contiguous span of the last axis (block_bytes // nv bytes, or head_v_dim
        # elems). Reordering whole per-head spans is bit-identical to a
        # post-dequant column permute (verified vs HF golden, maxdiff = quant
        # error) ONLY IF head_v_dim is a multiple of the quant block, so that no
        # packed block is split. That holds for the 35B (Q8_0 block=32 divides
        # head_v_dim=128) but NOT for k-quants whose super-block is 256 elems
        # (the 27B's ssm_out is Q5_K: head_v_dim=128 is HALF a super-block, and
        # a super-block's shared d/dmin/scales header plus its de-interleaved
        # qh/qs payload means no byte range maps to a contiguous element range
        # at all). See the two-level path below for that case.
        if ".linear_attn.out_proj." in name:
            assert w.shape[1] % nv == 0, (
                f"out_proj last dim {w.shape[1]} not divisible by nv={nv}"
            )
            span = w.shape[1] // nv  # bytes-per-head (raw) or head_v_dim (F32)
            hvd = tc.linear_value_head_dim
            block_elems = self._gguf_block_elems(w.shape[1], nv * hvd)
            if hvd % block_elems == 0:
                # Block-safe: whole per-head spans are whole quant blocks.
                return (
                    w.reshape(w.shape[0], ratio, nk, span)
                    .transpose(1, 2)
                    .reshape(w.shape)
                    .contiguous()
                )
            # Not block-safe (k-quant with head_v_dim < super-block). Split the
            # permute into two levels so neither one ever cuts a packed block:
            #   (1) HERE, on the GLOBAL pre-shard weight, permute at the
            #       coarser (nk // tp)-head GROUP granularity. That is all the
            #       cross-rank regrouping there is: it just moves each rank's
            #       columns into its own contiguous RowParallel slice, and a
            #       group spans (nk // tp) * head_v_dim elems, a whole number of
            #       super-blocks. After it, each rank's slice holds exactly its
            #       columns in [ratio, nk_loc] order.
            #   (2) PER-RANK, in element order, via layer._gguf_gdn_col_perm:
            #       _xpu_repack_* unpacks -> permutes -> repacks, turning
            #       [ratio, nk_loc] into HF's [nk_loc, ratio]. Never splits a
            #       block because it works on unpacked elements.
            # The two compose to exactly the full [ratio, nk] -> [nk, ratio]
            # value-head permute.
            mod = self._resolve_gdn_out_proj(name)
            tp = int(getattr(mod, "tp_size", 1) or 1)
            if nk % tp != 0:
                raise ValueError(
                    f"GGUF GDN out_proj col-permute: linear_num_key_heads={nk} "
                    f"is not divisible by out_proj tp_size={tp}."
                )
            nk_loc = nk // tp
            if (nk_loc * hvd) % block_elems != 0:
                raise ValueError(
                    f"GGUF GDN out_proj col-permute: (nk/tp)*head_v_dim = "
                    f"{nk_loc}*{hvd} = {nk_loc * hvd} is not a multiple of the "
                    f"quant block ({block_elems} elems) for {name}; the "
                    f"pre-shard group permute would split a packed block. "
                    f"Use a smaller tp_size."
                )
            mod._gguf_gdn_col_perm = (ratio, nk_loc, hvd)
            return (
                w.reshape(w.shape[0], ratio, tp, nk_loc * span)
                .transpose(1, 2)
                .reshape(w.shape)
                .contiguous()
            )
        # in_proj_a / in_proj_b: value-head rows. Match both quantized
        # (.qweight) and unquantized (.weight) deliveries.
        if (
            ".linear_attn.in_proj_a." in name
            or ".linear_attn.in_proj_b." in name
        ):
            return self._perm_value_rows(w, ratio, nk)
        if ".linear_attn.in_proj_z." in name:
            return self._perm_value_rows(w, ratio, nk)
        # in_proj_qkv: rows are [q | k | v] output dims; only the v block (the
        # value heads) permutes. q/k are key heads (no ratio).
        if ".linear_attn.in_proj_qkv." in name:
            kdim = nk * tc.linear_key_head_dim
            vdim = nv * tc.linear_value_head_dim
            q, k, v = torch.split(w, [kdim, kdim, vdim], dim=0)
            return torch.cat(
                [q, k, self._perm_value_rows(v, ratio, nk)], dim=0
            ).contiguous()
        # conv1d (F32): same [q | k | v] row layout on dim 0.
        if name.endswith("linear_attn.conv1d.weight"):
            kdim = nk * tc.linear_key_head_dim
            vdim = nv * tc.linear_value_head_dim
            q, k, v = torch.split(w, [kdim, kdim, vdim], dim=0)
            return torch.cat(
                [q, k, self._perm_value_rows(v, ratio, nk)], dim=0
            ).contiguous()
        return w

    @staticmethod
    def _gguf_block_elems(nbytes_last: int, n_elems: int) -> int:
        """Recover the GGUF quant block size, in ELEMENTS, of a raw-byte last
        axis of ``nbytes_last`` bytes that encodes ``n_elems`` values.

        Returns 1 when the tensor carries real (unquantized) values, i.e. the
        weight iterator already dequantized it, so any element-granular permute
        is exact. Used to decide whether a raw-byte column permute would split
        a packed block.
        """
        if nbytes_last == n_elems:
            return 1
        import gguf as _gguf

        cands = [
            be
            for be, ts in _gguf.GGML_QUANT_SIZES.values()
            if be and n_elems % be == 0 and (n_elems // be) * ts == nbytes_last
        ]
        if not cands:
            raise ValueError(
                f"cannot infer GGUF block size: last axis of {nbytes_last} "
                f"bytes encoding {n_elems} elements matches no GGML quant type"
            )
        # Ambiguity is only possible between types with identical bytes/elem;
        # take the largest block, which is the conservative choice (it can only
        # push us onto the safe two-level path, never off it).
        return max(cands)

    def _resolve_gdn_out_proj(self, name: str) -> torch.nn.Module:
        """Resolve the ``linear_attn.out_proj`` module that a GGUF weight named
        ``name`` belongs to, so its per-rank column permute can be recorded."""
        marker = ".linear_attn.out_proj."
        path = name.replace("model.language_model.", "model.")
        path = path[: path.index(marker)] + ".linear_attn.out_proj"
        try:
            return self.get_submodule(path)
        except AttributeError as exc:
            raise AttributeError(
                f"GGUF GDN out_proj col-permute: cannot resolve module "
                f"'{path}' for weight '{name}'"
            ) from exc

    @staticmethod
    def _perm_value_rows(t: torch.Tensor, ratio: int, nk: int) -> torch.Tensor:
        """Reorder the value-head axis (dim 0) from GGUF [ratio, nk, per_head]
        to HF [nk, ratio, per_head]. Works on real values and on raw quantized
        bytes alike (whole-row permutation)."""
        nv = ratio * nk
        per_head = t.shape[0] // nv
        return (
            t.reshape(ratio, nk, per_head, *t.shape[1:])
            .transpose(0, 1)
            .reshape(t.shape)
            .contiguous()
        )

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
            # GDN fused projections
            ("in_proj_qkvz.", "in_proj_qkv.", (0, 1, 2)),
            ("in_proj_qkvz.", "in_proj_z.", 3),
            ("in_proj_ba.", "in_proj_b.", 0),
            ("in_proj_ba.", "in_proj_a.", 1),
        ]

        # GGUF load-path detection (mirror the dense Qwen3_5 load_weights).
        # A GGUF checkpoint stores weights in llama.cpp conventions that differ
        # from the HF safetensors the model code expects:
        #   * GemmaRMSNorm weights are stored standard (~1.0), but GemmaRMSNorm
        #     computes x*(1+w) so the param must be (standard-1) -> subtract 1.
        #   * GDN linear_attn.* needs the value-head permute / A_log / dt_bias
        #     transform (_gguf_gdn_transform), same as the dense path.
        #   * shared_expert_gate is stored 1-D [hidden] but the param is
        #     [1, hidden]; conv1d is stored 2-D but the param is 3-D.
        # The GDN linear_attn.norm uses plain RMSNormGated (no offset) -- exclude.
        _is_gguf = (
            getattr(self, "quant_config", None) is not None
            and getattr(self.quant_config, "get_name", lambda: "")() == "gguf"
        )
        _gemma_norm_suffixes = (
            "input_layernorm.weight",
            "post_attention_layernorm.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_norm.weight",
        )

        num_experts = self.config.num_experts

        # Params for weights, fp8 weight scales, fp8 activation scales
        # (param_name, weight_name, expert_id, shard_id)
        expert_params_mapping = FusedMoE.make_expert_params_mapping(
            ckpt_gate_proj_name="gate_proj",
            ckpt_down_proj_name="down_proj",
            ckpt_up_proj_name="up_proj",
            num_experts=(
                num_experts
                if not self.enable_shared_expert_fusion
                else num_experts + self.num_fused_shared_experts
            ),
        )

        # Skip loading extra parameters for GPTQ/modelopt models.
        ignore_suffixes = (
            ".bias",
            "_bias",
            ".k_scale",
            "_k_scale",
            ".v_scale",
            "_v_scale",
            "_weight_scale",
            "_input_scale",
        )

        is_fused_expert = False
        fused_expert_params_mapping = [
            ("experts.w13_weight", "experts.gate_up_proj", 0, "w1"),
            ("experts.w2_weight", "experts.down_proj", 0, "w2"),
        ]

        if self.enable_shared_expert_fusion:
            """
            When shared experts are fused, we need to map the shared experts to routed experts.

            mlp.share_expert.gate_up_proj.weight  --> experts.512.gate_up_proj.weight -> experts.w13_weight, expert_id = 512
            mlp.share_expert.down_proj.weight  --> experts.512.down_proj.weight -> experts.w2_weight, expert_id = 512
            """
            fused_expert_params_mapping += [
                (
                    "experts.w13_",
                    f"experts.{num_experts}.gate_up_proj.",
                    num_experts,
                    "w1",
                ),
                (
                    "experts.w2_",
                    f"experts.{num_experts}.down_proj.",
                    num_experts,
                    "w2",
                ),
                ## shared experts may contain gate_proj and up_proj instead of gate_up_proj
                (
                    "experts.w13_",
                    f"experts.{num_experts}.gate_proj.",
                    num_experts,
                    "w1",
                ),
                (
                    "experts.w13_",
                    f"experts.{num_experts}.up_proj.",
                    num_experts,
                    "w3",
                ),
            ]

        def load_fused_expert_weights(
            name: str,
            params_dict: dict,
            loaded_weight: torch.Tensor,
            shard_id: str,
            num_experts: int,
        ):
            if name not in params_dict:
                return False
            param = params_dict[name]
            weight_loader = param.weight_loader
            # let ep moe layer to gracefully handle expert_ids that do not belong to local moe rank
            for expert_id in range(num_experts):
                curr_expert_weight = loaded_weight[expert_id]
                weight_loader(
                    param,
                    curr_expert_weight,
                    name,
                    shard_id,
                    expert_id,
                )
            return True

        loaded_params: Set[str] = set()
        params_dict = dict(self.named_parameters(remove_duplicate=False))

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "mtp" in name:
                continue
            if _is_gguf and (
                name.endswith(_gemma_norm_suffixes)
                or name == "model.language_model.norm.weight"
                or name == "model.norm.weight"
            ):
                loaded_weight = loaded_weight - 1.0
            if _is_gguf and ".linear_attn." in name:
                loaded_weight = self._gguf_gdn_transform(name, loaded_weight)
            if "language_model" in name:
                name = name.replace(r"model.language_model.", r"model.")
            if ".self_attn." in name:
                name = name.replace(".self_attn", "")
            if (
                self.config.tie_word_embeddings
                and self.pp_group.is_last_rank
                and "model.embed_tokens.weight" in name
            ):
                if "lm_head.weight" in params_dict:
                    lm_head_param = params_dict["lm_head.weight"]
                    weight_loader = getattr(
                        lm_head_param, "weight_loader", default_weight_loader
                    )
                    weight_loader(lm_head_param, loaded_weight)

            layer_id = get_layer_id(name)
            if (
                layer_id is not None
                and hasattr(self, "start_layer")
                and (layer_id < self.start_layer or layer_id >= self.end_layer)
            ):
                continue

            if self.enable_shared_expert_fusion:
                if "mlp.shared_expert." in name:
                    # Firstly map mlp.shared_expert.xx_proj to mlp.experts.512.xx_proj
                    name = name.replace(
                        "mlp.shared_expert.",
                        f"mlp.experts.{num_experts}.",
                    )

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if name.endswith("experts.gate_up_proj") or name.endswith(
                    "experts.down_proj"
                ):
                    is_fused_expert = True
                    expert_params_mapping = fused_expert_params_mapping

                # Skip non-stacked layers and experts (experts handled below).
                if weight_name not in name:
                    continue
                if "visual" in name:
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
                # GGUF F32 GDN gate shards (35B ssm_beta/ssm_alpha -> in_proj_b/a)
                # are yielded as `.weight`: the gguf iterator only renames non-F32
                # tensors to `.qweight`. But the fused `in_proj_ba` is a GGUF
                # quantized module whose merged param is `.qweight`, so the F32
                # `...in_proj_ba.weight` target is absent from params_dict and the
                # shard is silently dropped -> empty rep -> 0-wide matmul crash
                # (Qwen3.6 ratio=2; the ref only exercised the quantized ratio=1
                # ba path). Redirect the F32 `.weight` merged-GDN target to its
                # `.qweight` param so the GGUF weight_loader records it into the
                # shard data_container (F32 shards take the fp16 rep in the XPU
                # method, which defaults shard_weight_type to F32). GGUF only.
                if (
                    _is_gguf
                    and name.endswith(".weight")
                    and name not in params_dict
                    and (name[: -len(".weight")] + ".qweight") in params_dict
                ):
                    name = name[: -len(".weight")] + ".qweight"
                # Skip loading extra parameters for GPTQ/modelopt models.
                if name.endswith(ignore_suffixes) and name not in params_dict:
                    continue

                if name not in params_dict:
                    continue

                param = params_dict[name]
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Track if this is an expert weight to enable early skipping
                is_expert_weight = False

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in name:
                        continue
                    if "visual" in name or self.config.encoder_only:
                        continue
                    # Anyway, this is an expert weight and should not be
                    # attempted to load as other weights later
                    is_expert_weight = True
                    name_mapped = name.replace(weight_name, param_name)
                    if is_fused_expert:
                        # is_fused_expert is True, the checkpoint contains gate_up_proj and down_proj for each expert
                        if "experts.gate_up_proj" in name:
                            # experts.gate_up_proj contains all 512 routed experts, excluding shared experts
                            # split into w1 and w3
                            loaded_weight = loaded_weight.chunk(2, dim=-2)
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[0],
                                "w1",
                                num_experts,
                            )
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight[1],
                                "w3",
                                num_experts,
                            )
                        elif "experts.down_proj" in name:
                            # experts.down_proj contains all 512 routed experts, excluding shared experts
                            load_fused_expert_weights(
                                name_mapped,
                                params_dict,
                                loaded_weight,
                                shard_id,
                                num_experts,
                            )
                        elif self.enable_shared_expert_fusion:
                            # shared experts should be loaded to experts.w13_weight and experts.w2_weight
                            param = params_dict[name_mapped]
                            weight_loader = getattr(
                                param, "weight_loader", default_weight_loader
                            )
                            param = params_dict[name_mapped]
                            if f"{num_experts}.gate_up_proj" in name:
                                # split into w1 and w3
                                loaded_weight = loaded_weight.chunk(2, dim=-2)
                                # load to experts.w13_weight, shard_id = w1, expert_id = 512
                                weight_loader(
                                    param,
                                    loaded_weight[0],
                                    name_mapped,
                                    "w1",
                                    expert_id,
                                )
                                # load to experts.w13_weight, shard_id = w3, expert_id = 512
                                weight_loader(
                                    param,
                                    loaded_weight[1],
                                    name_mapped,
                                    "w3",
                                    expert_id,
                                )
                            else:
                                # load down_proj to experts.w2_weight, shard_id = w2, expert_id = 512
                                # Or load gate_proj and up_proj to experts.w13_weight, shard_id = w1/w3, expert_id = 512
                                weight_loader(
                                    param,
                                    loaded_weight,
                                    name_mapped,
                                    shard_id,
                                    expert_id,
                                )
                    else:
                        # Skip loading extra parameters for GPTQ models.
                        if (
                            name_mapped.endswith(ignore_suffixes)
                            and name_mapped not in params_dict
                        ):
                            continue
                        param = params_dict[name_mapped]
                        # We should ask the weight loader to return success or
                        # not here since otherwise we may skip experts with
                        # # other available replicas.
                        weight_loader = param.weight_loader
                        weight_loader(
                            param,
                            loaded_weight,
                            name_mapped,
                            shard_id=shard_id,
                            expert_id=expert_id,
                        )
                    name = name_mapped
                    break
                else:
                    if is_expert_weight:
                        # This is an expert weight but not mapped to this rank, skip all remaining processing
                        continue

                    if "visual" in name:
                        # adapt to VisionAttention
                        name = name.replace(r"attn.qkv.", r"attn.qkv_proj.")
                        name = name.replace(r"model.visual.", r"visual.")

                    # Skip loading extra parameters for GPTQ/modelopt models.
                    if name.endswith(ignore_suffixes) and name not in params_dict:
                        continue

                    if name in params_dict.keys():
                        param = params_dict[name]
                        # GGUF stores conv1d 2-D [ch, kernel] but the param is
                        # 3-D [ch, 1, kernel]; insert the singleton middle dim.
                        if (
                            _is_gguf
                            and "conv1d.weight" in name
                            and loaded_weight.dim() == 2
                            and param.dim() == 3
                        ):
                            loaded_weight = loaded_weight.unsqueeze(1)
                        # GGUF stores shared_expert_gate 1-D [hidden] but the
                        # param is 2-D [1, hidden]; add the leading dim.
                        if (
                            _is_gguf
                            and name.endswith("shared_expert_gate.weight")
                            and loaded_weight.dim() == 1
                            and param.dim() == 2
                        ):
                            loaded_weight = loaded_weight.unsqueeze(0)
                        weight_loader = getattr(
                            param, "weight_loader", default_weight_loader
                        )
                        weight_loader(param, loaded_weight)
                    else:
                        logger.warning(f"Parameter {name} not found in params_dict")
            loaded_params.add(name)

        self._routed_experts_weights_of_layer = LazyValue(
            lambda: {
                layer_id: layer.mlp.get_moe_weights()
                for layer_id, layer in enumerate(self.model.layers)
                if isinstance(layer.mlp, Qwen2MoeSparseMoeBlock)
            }
        )

        return loaded_params

    @property
    def routed_experts_weights_of_layer(self):
        return self._routed_experts_weights_of_layer.value

    @classmethod
    def get_model_config_for_expert_location(cls, config):
        text_config = getattr(config, "text_config", config)
        return ModelConfigForExpertLocation(
            num_layers=text_config.num_hidden_layers,
            num_logical_experts=text_config.num_experts,
            num_groups=None,
        )


# The GGUF GDN layout transform is arch-independent (it only reads
# linear_num_{key,value}_heads / linear_{key,value}_head_dim), so the dense
# class reuses the MoE implementation rather than duplicating it. Bound here
# because Qwen3_5MoeForConditionalGeneration is defined after the dense class.
Qwen3_5ForConditionalGeneration._gguf_gdn_transform = (
    Qwen3_5MoeForConditionalGeneration._gguf_gdn_transform
)
Qwen3_5ForConditionalGeneration._perm_value_rows = staticmethod(
    Qwen3_5MoeForConditionalGeneration._perm_value_rows
)
Qwen3_5ForConditionalGeneration._gguf_block_elems = staticmethod(
    Qwen3_5MoeForConditionalGeneration._gguf_block_elems
)
Qwen3_5ForConditionalGeneration._resolve_gdn_out_proj = (
    Qwen3_5MoeForConditionalGeneration._resolve_gdn_out_proj
)

EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]
