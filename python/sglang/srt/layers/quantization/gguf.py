# SPDX-License-Identifier: Apache-2.0
# Adapted from: https://github.com/vllm-project/vllm/blob/ab3e80042eac24dd362408e6d63ad98768046359/vllm/model_executor/layers/quantization/gguf.py
from __future__ import annotations

import logging
import os
import warnings
from typing import TYPE_CHECKING, Any, List, Optional

import gguf
import torch
from gguf import GGMLQuantizationType as WeightType
from torch.nn.parameter import Parameter, UninitializedParameter

from sglang.srt.layers.linear import LinearBase
from sglang.srt.layers.moe import MoeRunnerConfig
from sglang.srt.layers.quantization.base_config import (
    FusedMoEMethodBase,
    LinearMethodBase,
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.unquant import UnquantizedLinearMethod
from sglang.srt.utils import is_cuda, is_hip, is_musa, is_npu, is_xpu, set_weight_attrs

if TYPE_CHECKING:
    from sglang.srt.layers.moe.token_dispatcher import (
        CombineInput,
        StandardDispatchOutput,
    )

_is_cuda = is_cuda()
_is_hip = is_hip()
_is_xpu = is_xpu()
_is_musa = is_musa()
_is_npu = is_npu()

if _is_cuda:
    from sgl_kernel import moe_align_block_size, moe_sum
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )

    from sglang.jit_kernel.activation import gelu_and_mul, silu_and_mul
elif _is_musa:
    from sgl_kernel import gelu_and_mul, moe_align_block_size, moe_sum, silu_and_mul
    from sgl_kernel.quantization import (
        ggml_dequantize,
        ggml_moe_a8,
        ggml_moe_a8_vec,
        ggml_moe_get_block_size,
        ggml_mul_mat_a8,
        ggml_mul_mat_vec_a8,
    )
elif _is_npu:
    from gguf import dequantize as gguf_dequantize
elif _is_xpu:
    # XPU GGUF: q4_0 + q8_0 linear run on ESIMD GEMV (quant resident,
    # bandwidth-optimal at decode); remaining types (Q4_1/Q5_K/Q6_K/...) are
    # CPU-dequantized once to fp16 at load via the gguf lib and kept resident.
    from gguf import dequantize as gguf_dequantize

    # SGL-OWNED kernel package (custom_esimd_kernels_sglang) — a full rename-copy
    # of custom-esimd-kernels-vllm living at llm-scaler/vllm/custom-esimd-kernels-sglang,
    # so sglang can evolve these kernels (e.g. the D1 small-N q8_0 BW opt) without
    # touching the vllm-shared package. Every GGUF kernel import below prefers the
    # sgl package and falls back to the vllm package if the sgl .so is absent.
    # Toggle SGLANG_GGUF_XPU_USE_VLLM_KERNELS=1 to force the legacy vllm package.
    # (notes §10bg/§10bh/§10bi)
    _force_vllm_k = os.environ.get("SGLANG_GGUF_XPU_USE_VLLM_KERNELS") == "1"

    def _imp_kernels(names):
        """Import `names` from the sgl package, else the vllm package, else Nones."""
        mods = ("custom_esimd_kernels_vllm",) if _force_vllm_k else (
            "custom_esimd_kernels_sglang", "custom_esimd_kernels_vllm")
        for mod in mods:
            try:
                m = __import__(mod, fromlist=list(names))
                return [getattr(m, n) for n in names]
            except (ImportError, AttributeError):
                continue
        return [None] * len(names)

    esimd_gemv_q4_0, esimd_gemm_q4_0 = _imp_kernels(("esimd_gemv_q4_0", "esimd_gemm_q4_0"))
    # oneDNN u4 fused-dequant matmul for q4_0 PREFILL (M>1): keeps the weight in
    # int4, no fp16 DRAM round-trip. ~6x faster than dequant+matmul and 3.7x over
    # the hand-written DPAS GEMM on PTL (notes §10v). Bit-exact for q4_0 because
    # GGUF q4_0 is offset-binary (value=nibble-8) == oneDNN u4 with zp=8.
    try:
        import onednn_gguf_xpu as _onednn_gguf
        if os.environ.get("SGLANG_GGUF_XPU_NO_ONEDNN") == "1":
            _onednn_gguf = None  # debug toggle: force dequant+matmul baseline
    except ImportError:
        _onednn_gguf = None
    # Grouped Q4_K/Q5_K MoE prefill GGEMV (doubleGRF DPAS): replaces the per-route
    # GEMV at prefill (M>1), which is ~161x off the compute floor (notes §10x).
    # ~17x faster (§10ad). Falls back to per-route GEMV if absent / SGLANG_GGUF_XPU_NO_GROUPED_MOE=1.
    try:
        import moe_grouped_gguf_xpu as _moe_grouped
        if os.environ.get("SGLANG_GGUF_XPU_NO_GROUPED_MOE") == "1":
            _moe_grouped = None
    except ImportError:
        _moe_grouped = None
    (esimd_gemv_q8_0,) = _imp_kernels(("esimd_gemv_q8_0",))
    (esimd_gemv_q4_k,) = _imp_kernels(("esimd_gemv_q4_k",))
    esimd_gemv_q5_k, esimd_gemv_q6_k = _imp_kernels(("esimd_gemv_q5_k", "esimd_gemv_q6_k"))
    esimd_moe_up_q4k, esimd_moe_down_q5k, esimd_moe_down_q6k = _imp_kernels(
        ("esimd_moe_up_q4k", "esimd_moe_down_q5k", "esimd_moe_down_q6k"))
else:
    if not _is_hip:
        warnings.warn(f"Only CUDA, MUSA and NPU support GGUF quantization currently.")

logger = logging.getLogger(__name__)


class GGUFConfig(QuantizationConfig):
    """Config class for GGUF."""

    def __init__(self, modules_to_not_convert: list[str] | None = None) -> None:
        super().__init__()
        if _is_hip:
            warnings.warn(f"Only CUDA and MUSA support GGUF quantization currently.")
        self.modules_to_not_convert = modules_to_not_convert or []

    def __repr__(self) -> str:
        return "GGUFConfig()"

    def get_scaled_act_names(self) -> List[str]:
        return []

    def get_name(self) -> "str":
        return "gguf"

    def get_supported_act_dtypes(self) -> list[torch.dtype]:
        return [torch.half, torch.bfloat16, torch.float32]

    @classmethod
    def get_min_capability(cls) -> int:
        return 60 if not _is_musa else 21

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []  # no extra configs.

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFConfig":
        modules_to_not_convert = cls.get_from_keys_or(
            config, ["modules_to_not_convert"], None
        )
        return cls(modules_to_not_convert)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.moe.fused_moe_triton import FusedMoE
        from sglang.srt.layers.vocab_parallel_embedding import VocabParallelEmbedding

        if isinstance(layer, LinearBase):
            if is_layer_skipped_gguf(prefix, self.modules_to_not_convert):
                return UnquantizedLinearMethod()
            if _is_npu:
                return GGUFLinearAscendMethod(self)
            if _is_xpu:
                return GGUFLinearXPUMethod(self)
            return GGUFLinearMethod(self)
        elif isinstance(layer, VocabParallelEmbedding):
            if _is_npu:
                return GGUFEmbeddingAscendMethod(self)
            if _is_xpu:
                return GGUFEmbeddingXPUMethod(self)
            return GGUFEmbeddingMethod(self)
        elif isinstance(layer, FusedMoE):
            if _is_npu:
                return GGUFMoEAscendMethod(self)
            if _is_xpu:
                return GGUFMoEXPUMethod(self)
            return GGUFMoEMethod(self)
        return None


def is_layer_skipped_gguf(prefix: str, modules_to_not_convert: list[str]):
    return any(module_name in prefix for module_name in modules_to_not_convert)


UNQUANTIZED_TYPES = {WeightType.F32, WeightType.F16, WeightType.BF16}
STANDARD_QUANT_TYPES = {
    WeightType.Q4_0,
    WeightType.Q4_1,
    WeightType.Q5_0,
    WeightType.Q5_1,
    WeightType.Q8_0,
    WeightType.Q8_1,
}
KQUANT_TYPES = {
    WeightType.Q2_K,
    WeightType.Q3_K,
    WeightType.Q4_K,
    WeightType.Q5_K,
    WeightType.Q6_K,
}
IMATRIX_QUANT_TYPES = {
    WeightType.IQ1_M,
    WeightType.IQ1_S,
    WeightType.IQ2_XXS,
    WeightType.IQ2_XS,
    WeightType.IQ2_S,
    WeightType.IQ3_XXS,
    WeightType.IQ3_S,
    WeightType.IQ4_XS,
    WeightType.IQ4_NL,
}
# TODO(Isotr0py): Currently, we don't have MMQ kernel for I-Matrix quantization.
# Consolidate DEQUANT_TYPES, MMVQ_QUANT_TYPES and MMQ_QUANT_TYPES after we add
# MMQ kernel for I-Matrix quantization.
DEQUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMVQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES | IMATRIX_QUANT_TYPES
MMQ_QUANT_TYPES = STANDARD_QUANT_TYPES | KQUANT_TYPES


def fused_mul_mat_gguf(
    x: torch.Tensor, qweight: torch.Tensor, qweight_type: int
) -> torch.Tensor:
    if qweight_type in IMATRIX_QUANT_TYPES:
        mmvq_safe = 8 if qweight.shape[0] > 5120 else 16
    else:
        mmvq_safe = 2 if qweight.shape[0] > 5120 else 6
    # HACK: when doing chunked prefill we don't generate output tokens
    # so input to logits generator is empty which causes invalid parameter
    if x.shape[0] == 0:
        return torch.empty(x.shape[0], qweight.shape[0], dtype=x.dtype, device=x.device)
    # there is no need to call any kernel for fp16/bf16
    if qweight_type in UNQUANTIZED_TYPES:
        return x @ qweight.T
    # enable MMVQ in contiguous batching with batch_size=1
    if x.shape[0] <= mmvq_safe and qweight_type in MMVQ_QUANT_TYPES:
        y = ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
    # Use MMQ Kernel if it's available (standard + k-quants)
    elif qweight_type in MMQ_QUANT_TYPES:
        y = ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
    # If there is no available MMQ kernel, fallback to dequantize
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
        weight = ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
        y = x @ weight.T
    else:
        # Raise an error if the quantization type is not supported.
        # Might be useful if llama.cpp adds a new quantization type.
        # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")
    return y


def fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    activation: str,
) -> torch.Tensor:
    def act(x: torch.Tensor):
        if activation == "silu":
            return silu_and_mul(x)
        elif activation == "gelu":
            return gelu_and_mul(x)
        raise ValueError(f"Unsupported activation: {activation}")

    out_hidden_states = torch.empty_like(x)
    # unless we decent expert reuse we are better off running moe_vec kernel
    if (
        qweight_type2 in MMQ_QUANT_TYPES
        and qweight_type in MMQ_QUANT_TYPES
        and x.shape[0] > 64
    ):
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]
        BLOCK_SIZE = ggml_moe_get_block_size(qweight_type)

        sorted_token_ids, expert_ids, num_tokens_post_padded = moe_align_block_size(
            topk_ids, BLOCK_SIZE, E
        )
        out = ggml_moe_a8(
            x,
            w1,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type,
            N,
            top_k,
            num_tokens,
        )
        out = act(out)
        out = ggml_moe_a8(
            out,
            w2,
            sorted_token_ids,
            expert_ids,
            num_tokens_post_padded,
            qweight_type2,
            w2.shape[1],
            1,
            num_tokens * top_k,
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        # TODO(FlamingoPg): maybe we can use moe_sum_reduce here?
        moe_sum(out, out_hidden_states)
    elif qweight_type2 in MMVQ_QUANT_TYPES and qweight_type in MMVQ_QUANT_TYPES:
        num_tokens, _ = x.shape
        E, N, _ = w1.shape
        top_k = topk_ids.shape[1]

        out = ggml_moe_a8_vec(x, w1, topk_ids, top_k, qweight_type, N, num_tokens)
        out = act(out)

        out = ggml_moe_a8_vec(
            out, w2, topk_ids, 1, qweight_type2, w2.shape[1], num_tokens * top_k
        )
        out = out.reshape(num_tokens, top_k, w2.shape[1]).mul_(
            topk_weights.view(num_tokens, top_k, 1)
        )
        moe_sum(out, out_hidden_states)
    else:
        logger.warning_once(
            "There is no support for fast MoE kernel "
            "for current quantization method. "
            "Falling back to slow implementation. "
        )
        for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
            inp = x[tok].reshape((1,) + x.shape[1:])
            current_hidden_state = None
            for ww, ii in zip(w, idx):
                expert_up = w1[ii]

                out = fused_mul_mat_gguf(inp, expert_up, qweight_type)
                out = act(out)

                expert_down = w2[ii]
                current_state = fused_mul_mat_gguf(
                    out, expert_down, qweight_type2
                ).mul_(ww)
                if current_hidden_state is None:
                    current_hidden_state = current_state
                else:
                    current_hidden_state.add_(current_state)
            out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


def apply_gguf_embedding(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    elif qweight_type in DEQUANT_TYPES:
        block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
        x_flat = x.flatten()
        assert hidden_size == qweight.shape[1] // type_size * block_size
        quant = torch.index_select(qweight, dim=0, index=x_flat)
        dequant = ggml_dequantize(
            quant, qweight_type, hidden_size, x_flat.shape[0], dtype
        )
        return dequant.view(*x.shape, hidden_size)
    else:
        qweight_type = WeightType(qweight_type)
        raise NotImplementedError(f"Unsupported GGUF quantization type: {qweight_type}")


class GGUFLinearMethod(LinearMethodBase):
    """Linear method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            qweight_type = WeightType(qweight_type)
            raise ValueError(
                f"Unsupported GGUF quantization type {qweight_type} in layer {layer}."
            )
        # For MergedColumnParallelLinear and QKVParallelLinear, we need to
        # materialize the padded weight parameter for CUDA Graph compatibility.
        self._create_padded_weight_param(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1, ValueError(
                f"Data container has mixed dtypes: {dtype}"
            )
            dtype = next(iter(dtype))
            # concat dim0 and pad dim1
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            # Pad the quantized weights to dense tensor, and create a map
            # with the location of each shard in the padded tensor.
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            # (dim0_start, dim0_end, dim1_size)
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        shard_id = layer.qweight.shard_id

        if shard_id:
            # dequantize shard weights respectively
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            qweight = layer.qweight
            result = []
            for idx in shard_id:
                start, end, offset = layer.qweight.shard_offset_map[idx]
                qweight_type = layer.qweight_type.shard_weight_type[idx]
                result.append(
                    fused_mul_mat_gguf(
                        x, qweight[start:end, :offset].contiguous(), qweight_type
                    )
                )
            out = torch.cat(result, axis=1)
        else:
            qweight = layer.qweight
            qweight_type = layer.qweight_type.weight_type
            out = fused_mul_mat_gguf(x, qweight, qweight_type)
        if bias is not None:
            out.add_(bias)
        return out


# =============================================================================
# XPU (Intel PTL Xe3) GGUF: q4_0 on ESIMD kernel, everything else fp16-resident
# =============================================================================
_Q4_0_BLOCK_BYTES = 18  # GGML q4_0: {fp16 d; uint8 qs[16]} per 32 elements


def _xpu_repack_q4_0(qweight: torch.Tensor):
    """GGUF q4_0 raw blocks -> ESIMD interleaved (qweight [N,K/2] u8, scale [N,K/32] f16).

    qweight in: [N, blocks*18] uint8 (per-output-row, K/32 contiguous 18-byte
    q4_0 blocks). GGML packs split-half (byte j: low->elem j, high->elem j+16);
    the ESIMD kernel wants interleaved (byte j: low->2j, high->2j+1). The repack
    is a value-preserving nibble permutation (bit-exact, validated in
    custom-esimd-kernels-vllm/tests/test_q4_0_repack.py).
    """
    N = qweight.shape[0]
    blocks = qweight.shape[1] // _Q4_0_BLOCK_BYTES
    buf = qweight.reshape(N, blocks, _Q4_0_BLOCK_BYTES)
    scale = buf[:, :, 0:2].contiguous().view(torch.float16).view(N, blocks)
    qs = buf[:, :, 2:18].contiguous()                 # [N, blocks, 16] uint8
    lo = qs & 0x0F                                    # nib of elems 0..15
    hi = (qs >> 4) & 0x0F                             # nib of elems 16..31
    nib = torch.cat([lo, hi], dim=2)                  # [N, blocks, 32], nib[...,i]=elem i
    even = nib[:, :, 0::2]                            # elems 0,2,...,30
    odd = nib[:, :, 1::2]                             # elems 1,3,...,31
    packed = (even | (odd << 4)).to(torch.uint8).view(N, blocks * 16)  # [N, K/2]
    return packed.contiguous(), scale.contiguous()


def _xpu_dequant_q4_0_packed(packed: torch.Tensor, scale: torch.Tensor,
                             out_dtype: torch.dtype) -> torch.Tensor:
    """Dequantize the interleaved q4_0 rep back to a dense [N, K] tensor on XPU.

    Inverse of _xpu_repack_q4_0's nibble interleaving: packed[N, K/2] holds the
    even element in the low nibble and the odd element in the high nibble;
    scale[N, K/32] is one fp16 scale per 32-element block. value = (nibble-8)*d.
    Used for the prefill (M>1) path, where one dequant + a dense matmul is far
    cheaper than the per-row ESIMD GEMM (which degrades ~linearly in M).
    """
    N, half = packed.shape
    blocks = scale.shape[1]
    even = (packed & 0x0F).to(torch.int16)            # [N, K/2] -> elems 0,2,...
    odd = ((packed >> 4) & 0x0F).to(torch.int16)      # elems 1,3,...
    nib = torch.stack([even, odd], dim=2).view(N, half * 2)  # interleave back
    vals = (nib - 8).to(out_dtype)                    # [N, K]
    vals = vals.view(N, blocks, -1) * scale.to(out_dtype).unsqueeze(-1)
    return vals.view(N, half * 2).contiguous()


def _xpu_dequant_to_fp16(qweight: torch.Tensor, qweight_type: int,
                         params_dtype: torch.dtype) -> torch.Tensor:
    """CPU-dequantize a non-q4_0 GGUF weight to a dense [N, K] tensor on XPU."""
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    rows = qweight.shape[0]
    cols = qweight.shape[1] // type_size * block_size
    dq = gguf_dequantize(qweight.cpu().numpy(), qweight_type)
    return (
        torch.from_numpy(dq)
        .to(dtype=params_dtype, device=qweight.device)
        .reshape(rows, cols)
    )


def apply_gguf_embedding_xpu(
    x: torch.Tensor,
    qweight: torch.Tensor,
    qweight_type: int,
    hidden_size: int,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Row-level GGUF embedding lookup for XPU.

    Mirrors apply_gguf_embedding but uses the gguf library (CPU) instead of the
    CUDA-only ggml_dequantize, which is not imported on XPU. Only the rows for
    the actual token ids are dequantized, so the quantized table stays resident
    (no full-table fp16 copy). The selected quant rows are gathered, moved to
    CPU, dequantized by the gguf lib, then returned on the original device.
    """
    if qweight_type in UNQUANTIZED_TYPES:
        return torch.embedding(qweight, x)
    if qweight_type not in DEQUANT_TYPES:
        raise NotImplementedError(
            f"Unsupported GGUF embedding quant type {WeightType(qweight_type)} on XPU."
        )
    block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    assert hidden_size == qweight.shape[1] // type_size * block_size
    x_flat = x.flatten()
    quant = torch.index_select(qweight, dim=0, index=x_flat)
    dq = gguf_dequantize(quant.cpu().numpy(), qweight_type)
    dequant = (
        torch.from_numpy(dq)
        .to(dtype=dtype or torch.float16, device=qweight.device)
        .reshape(x_flat.shape[0], hidden_size)
    )
    return dequant.view(*x.shape, hidden_size)


class GGUFEmbeddingXPUMethod(GGUFLinearMethod):
    """GGUF embedding for Intel XPU (PTL Xe3).

    The base GGUFEmbeddingMethod calls ggml_dequantize (a CUDA/MUSA-only sgl
    kernel) which is absent on XPU. Earlier this routed through
    apply_gguf_embedding_xpu, but that does a per-step .cpu() gguf-lib dequant
    which is illegal under XPUGraph capture ("wait method cannot be used for an
    event associated with a command graph"). This method instead repacks the
    table once into the resident packed k-quant rep (zero extra memory, same as
    the linear layers) and gathers + dequants the looked-up rows entirely on-XPU
    (graph-capturable, no host sync).
    """

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if qweight_type in UNQUANTIZED_TYPES:
            layer._xpu_emb_rep = ("fp16", layer.qweight.to(self.params_dtype), None)
            return
        # Repack the [vocab, K] GGUF table to the resident packed rep (q6_k /
        # q8_0 / q4_k ...). _xpu_prepare_shard handles every supported type and
        # keeps the weight quantized (no fp16 full-table copy).
        layer._xpu_emb_rep = _xpu_prepare_shard(
            layer.qweight.data, int(qweight_type), self.params_dtype
        )
        if hasattr(layer, "qweight"):
            del layer.qweight

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        # Eagle/NEXTN embed-share: set_embed_and_head may have replaced this
        # layer's table with the TARGET's shared dequantized `.weight`
        # ([vocab, hidden] fp16). When present it takes priority over the GGUF
        # rep — a plain embedding lookup (graph-safe, no dequant).
        shared_w = getattr(layer, "weight", None)
        if isinstance(shared_w, torch.Tensor):
            x_flat = x.flatten()
            out = torch.nn.functional.embedding(x_flat, shared_w.to(self.params_dtype))
            return out.view(*x.shape, shared_w.shape[1])
        rep = getattr(layer, "_xpu_emb_rep", None)
        if rep is None:  # not yet processed (shouldn't happen post-load)
            qweight = layer.qweight
            return apply_gguf_embedding_xpu(
                x, qweight, layer.qweight_type.weight_type,
                qweight.tensor_shape[1], dtype=self.params_dtype)
        kind = rep[0]
        x_flat = x.flatten()
        # Gather the looked-up rows of the packed rep (index_select is graph-safe
        # on XPU), then dequant ONLY those rows fully on-device — no .cpu().
        if kind == "fp16":
            out = torch.nn.functional.embedding(x_flat, rep[1].to(self.params_dtype))
            return out.view(*x.shape, rep[1].shape[1])
        if kind == "q6_k":
            ql, qh, sc = rep[1], rep[2], rep[3]
            g = _xpu_dequant_q6_k(ql.index_select(0, x_flat),
                                  qh.index_select(0, x_flat),
                                  sc.index_select(0, x_flat), self.params_dtype)
        elif kind == "q8_0":
            qs, sc = rep[1], rep[2]
            g = _xpu_dequant_q8_0(qs.index_select(0, x_flat),
                                  sc.index_select(0, x_flat), self.params_dtype)
        elif kind == "q5_k":
            ql, qh, sc, mn = rep[1], rep[2], rep[3], rep[4]
            g = _xpu_dequant_q5_k(ql.index_select(0, x_flat),
                                  qh.index_select(0, x_flat),
                                  sc.index_select(0, x_flat),
                                  mn.index_select(0, x_flat), self.params_dtype)
        elif kind == "q4_k":
            ql, sc, mn = rep[1], rep[2], rep[3]
            g = _xpu_dequant_q4_k(ql.index_select(0, x_flat),
                                  sc.index_select(0, x_flat),
                                  mn.index_select(0, x_flat), self.params_dtype)
        elif kind == "q4_0":
            packed, sc = rep[1], rep[2]
            g = _xpu_dequant_q4_0_packed(packed.index_select(0, x_flat),
                                         sc.index_select(0, x_flat), self.params_dtype)
        else:
            raise NotImplementedError(f"GGUF embedding rep '{kind}' on XPU")
        return g.view(*x.shape, g.shape[-1])

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute logits for a tied lm_head (lm_head IS this embedding module).

        _compute_lm_head routes here because a GGUF embedding has no dense
        .weight. The base GGUFLinearMethod.apply would call fused_mul_mat_gguf ->
        ggml_dequantize (CUDA-only, undefined on XPU). Instead dequantize the
        vocab table once via the gguf lib (cached on the layer) and matmul.
        """
        rep = getattr(layer, "_xpu_emb_rep", None)
        x2 = x.reshape(-1, x.shape[-1])
        M = x2.shape[0]
        # Hot path: single-token decode. Run the packed k-quant GEMV directly
        # over the full vocab (lm_head was 23.6% of 35B graph decode — it had
        # been dequantizing the Q6_K output.weight to a ~1GB fp16 table and
        # doing a full-vocab fp16 GEMM every step; see notes §10q). The GEMV
        # reads the resident packed rep, so no fp16 table and no dense matmul.
        # esimd_gemv_* require M==1; M>1 (prefill last-token logits, rare) falls
        # through to the dense path below.
        if (
            rep is not None
            and M == 1
            and rep[0] in ("q6_k", "q5_k", "q4_k", "q8_0", "q4_0")
        ):
            xf = x2.to(torch.float16).contiguous()
            out = _xpu_rep_gemv(xf, rep).to(x.dtype)  # [1, vocab]
            if bias is not None:
                out = out + bias
            return out.reshape(*x.shape[:-1], out.shape[-1])

        w = getattr(layer, "_xpu_lmhead_dense", None)
        if w is None:
            # process_weights_after_loading deletes layer.qweight and keeps the
            # packed rep; rebuild the dense vocab table from the rep (the M>1
            # prefill-logits path needs a full [vocab, hidden] matmul). Falls
            # back to qweight if the rep is somehow absent (pre-process call).
            if rep is not None:
                w = _xpu_dequant_rep_to_fp16(rep, self.params_dtype)
            else:
                qweight = layer.qweight
                qweight_type = layer.qweight_type.weight_type
                if qweight_type in UNQUANTIZED_TYPES:
                    w = qweight.to(self.params_dtype)
                else:
                    w = _xpu_dequant_to_fp16(qweight, qweight_type, self.params_dtype)
            layer._xpu_lmhead_dense = w  # [vocab, hidden] resident dense
        out = x.to(w.dtype) @ w.t()
        if bias is not None:
            out = out + bias
        return out


# GGML quant enum values (avoid importing the enum at call sites).
_Q4_0_TYPE = int(WeightType.Q4_0)
_Q8_0_TYPE = int(WeightType.Q8_0)
_Q4_K_TYPE = int(WeightType.Q4_K)

_Q4_K_SB = 256          # q4_K super-block elements
_Q4_K_BYTES = 144       # half2 dm(4) + scales[12] + qs[128]


def _xpu_repack_q4_k(qweight: torch.Tensor):
    """GGUF q4_K super-blocks -> ESIMD interleaved (ql [N,K/2] u8, scale + min
    [N,K/32] f16, pre-computed from the 6-bit sub-fields). Same interleaved
    nibble layout as q4_0 so the GEMV deinterleave is identical.

    GGML block_q4_K = {half2 dm(dall,dmin); u8 scales[12]; u8 qs[128]}, 8
    sub-blocks of 32. get_scale_min_k4 unpacks the 6-bit sub-scale/min; scale =
    dall*sc6, min = dmin*mn6 (skill Stage 1, zero extra memory). dequant on GPU:
    w = scale*nibble - min. Validated vs gguf-lib (q4_k_repack_ref.py, ~2.5e-4).
    """
    N = qweight.shape[0]
    nsb = qweight.shape[1] // _Q4_K_BYTES
    buf = qweight.reshape(N, nsb, _Q4_K_BYTES)
    dm = buf[:, :, 0:4].contiguous().view(torch.float16).view(N, nsb, 2)
    dall = dm[:, :, 0].float()                       # [N, nsb]
    dmin = dm[:, :, 1].float()
    sc12 = buf[:, :, 4:16].to(torch.int32)           # [N, nsb, 12]
    qs = buf[:, :, 16:144]                           # [N, nsb, 128] uint8

    # get_scale_min_k4 vectorized over j=0..7
    sc6 = torch.empty(N, nsb, 8, dtype=torch.int32, device=qweight.device)
    mn6 = torch.empty_like(sc6)
    for j in range(8):
        if j < 4:
            sc6[:, :, j] = sc12[:, :, j] & 63
            mn6[:, :, j] = sc12[:, :, j + 4] & 63
        else:
            sc6[:, :, j] = (sc12[:, :, j + 4] & 0xF) | ((sc12[:, :, j - 4] >> 6) << 4)
            mn6[:, :, j] = (sc12[:, :, j + 4] >> 4) | ((sc12[:, :, j] >> 6) << 4)
    scale = (dall[:, :, None] * sc6.float()).to(torch.float16)   # [N,nsb,8]
    minv = (dmin[:, :, None] * mn6.float()).to(torch.float16)

    # element-order nibbles: group il (0..3): elem 64il+p (p<32) = qs[32il+p]&0xF,
    # +32 = >>4. Then interleave into ql[N,K/2].
    K = nsb * _Q4_K_SB
    nib = torch.empty(N, nsb, _Q4_K_SB, dtype=torch.uint8, device=qweight.device)
    for il in range(4):
        qg = qs[:, :, 32 * il:32 * il + 32]
        nib[:, :, 64 * il:64 * il + 32] = qg & 0x0F
        nib[:, :, 64 * il + 32:64 * il + 64] = qg >> 4
    nib = nib.view(N, K)
    even = nib[:, 0::2]
    odd = nib[:, 1::2]
    ql = (even | (odd << 4)).to(torch.uint8)
    return (ql.contiguous(),
            scale.view(N, K // 32).contiguous(),
            minv.view(N, K // 32).contiguous())


def _xpu_dequant_q4_k(ql: torch.Tensor, scale: torch.Tensor, minv: torch.Tensor,
                      out_dtype: torch.dtype) -> torch.Tensor:
    """Dequant the interleaved q4_K rep -> dense [N, K]. w = scale*nibble - min."""
    N, half = ql.shape
    K = half * 2
    nb = scale.shape[1]
    even = (ql & 0x0F).to(torch.int16)
    odd = ((ql >> 4) & 0x0F).to(torch.int16)
    nib = torch.stack([even, odd], dim=2).view(N, K).to(out_dtype)
    w = (nib.view(N, nb, 32) * scale.to(out_dtype).unsqueeze(-1)
         - minv.to(out_dtype).unsqueeze(-1))
    return w.view(N, K).contiguous()


_Q5_K_TYPE = int(WeightType.Q5_K)
_Q6_K_TYPE = int(WeightType.Q6_K)
_Q5_K_BYTES = 2 + 2 + 12 + 32 + 128   # dm + scales[12] + qh[32] + qs[128] = 176
_Q6_K_BYTES = 128 + 64 + 16 + 2       # ql[128] + qh[64] + scales[16] + d = 210


_Q5Q6_VL = 512  # K-tile matching the ESIMD kernel + host pre-shuffle chunk


def _q5q6_col_perm_elems(t, perm):
    """Permute the K (last) dim of an element-order tensor [N,K] from GGUF
    [ratio,num_k] value-head order to HF [num_k,ratio], head_v_dim-granular."""
    ratio, nk, hvd = perm
    N, K = t.shape
    return t.reshape(N, ratio, nk, hvd).transpose(1, 2).reshape(N, K).contiguous()


def _pack_nibble_interleaved(u_elem):
    """element-order low-nibble [N,K] (0..15) -> packed [N,K/2] (low->2j, high->2j+1)."""
    even = u_elem[:, 0::2] & 0x0F
    odd = u_elem[:, 1::2] & 0x0F
    return ((even | (odd << 4)).to(torch.uint8)).contiguous()


def _preshuffle_qh1(high, K):
    """1-bit high [N,K] (0/1) -> pre-shuffled qh [N,K/8]. Per 512-tile: bit b of
    shuffled byte t holds element b*64+t (so the GPU stride-1 add works)."""
    N = high.shape[0]
    VL, VL8 = _Q5Q6_VL, _Q5Q6_VL // 8       # 512, 64
    ntile = K // VL
    h = high.to(torch.int32).view(N, ntile, 8, VL8)   # [N,tile,b,t] = elem tile*512+b*64+t
    byte = torch.zeros(N, ntile, VL8, dtype=torch.int32, device=high.device)
    for b in range(8):
        byte |= (h[:, :, b, :] & 1) << b
    return byte.to(torch.uint8).view(N, K // 8).contiguous()


def _preshuffle_qh2(high, K):
    """2-bit high [N,K] (0..3) -> pre-shuffled qh [N,K/4]. Per 512-tile: field p
    of shuffled byte t holds element p*128+t."""
    N = high.shape[0]
    VL, VLQ = _Q5Q6_VL, _Q5Q6_VL // 4       # 512, 128
    ntile = K // VL
    h = high.to(torch.int32).view(N, ntile, 4, VLQ)   # [N,tile,p,t] = elem tile*512+p*128+t
    byte = torch.zeros(N, ntile, VLQ, dtype=torch.int32, device=high.device)
    for p in range(4):
        byte |= (h[:, :, p, :] & 3) << (2 * p)
    return byte.to(torch.uint8).view(N, K // 4).contiguous()


def _xpu_q5_k_elem(qweight):
    """GGUF q5_K -> element-order val5 [N,K] (0..31) + scale,min [N,K/32] f16."""
    N = qweight.shape[0]
    nsb = qweight.shape[1] // _Q5_K_BYTES
    buf = qweight.reshape(N, nsb, _Q5_K_BYTES)
    dm = buf[:, :, 0:4].contiguous().view(torch.float16).view(N, nsb, 2)
    dall = dm[:, :, 0].float(); dmin = dm[:, :, 1].float()
    sc12 = buf[:, :, 4:16].to(torch.int32)
    qh = buf[:, :, 16:48].to(torch.int32)
    qs = buf[:, :, 48:176].to(torch.int32)
    sc6 = torch.empty(N, nsb, 8, dtype=torch.int32, device=qweight.device)
    mn6 = torch.empty_like(sc6)
    for j in range(8):
        if j < 4:
            sc6[:, :, j] = sc12[:, :, j] & 63
            mn6[:, :, j] = sc12[:, :, j + 4] & 63
        else:
            sc6[:, :, j] = (sc12[:, :, j + 4] & 0xF) | ((sc12[:, :, j - 4] >> 6) << 4)
            mn6[:, :, j] = (sc12[:, :, j + 4] >> 4) | ((sc12[:, :, j] >> 6) << 4)
    scale = (dall[:, :, None] * sc6.float()).to(torch.float16)
    minv = (dmin[:, :, None] * mn6.float()).to(torch.float16)
    K = nsb * _Q4_K_SB
    u5 = torch.empty(N, nsb, _Q4_K_SB, dtype=torch.int32, device=qweight.device)
    for il in range(4):
        hm0 = 1 << (2 * il); hm1 = hm0 << 1
        ql0 = qs[:, :, 32 * il + 0:32 * il + 32:2]
        ql1 = qs[:, :, 32 * il + 1:32 * il + 32:2]
        h0 = qh[:, :, 0:32:2]; h1 = qh[:, :, 1:32:2]
        u5[:, :, 64 * il + 0:64 * il + 32:2] = (ql0 & 0xF) + ((h0 & hm0) > 0).to(torch.int32) * 16
        u5[:, :, 64 * il + 1:64 * il + 32:2] = (ql1 & 0xF) + ((h1 & hm0) > 0).to(torch.int32) * 16
        u5[:, :, 64 * il + 32:64 * il + 64:2] = (ql0 >> 4) + ((h0 & hm1) > 0).to(torch.int32) * 16
        u5[:, :, 64 * il + 33:64 * il + 64:2] = (ql1 >> 4) + ((h1 & hm1) > 0).to(torch.int32) * 16
    return u5.view(N, K), scale.view(N, K // 32), minv.view(N, K // 32)


def _xpu_repack_q5_k(qweight: torch.Tensor, col_perm=None):
    """GGUF q5_K -> PACKED (ql [N,K/2] nibble + pre-shuffled qh [N,K/8] + scale,
    min [N,K/32] f16). Zero extra memory. Optional col_perm (GDN out_proj) applied
    in element order before packing. Validated vs gguf-lib (q5q6_packed_repack_ref.py)."""
    u5, scale, minv = _xpu_q5_k_elem(qweight)
    N, K = u5.shape
    if col_perm is not None:
        u5 = _q5q6_col_perm_elems(u5, col_perm)
        g = col_perm[2] // 32
        scale = _q5q6_col_perm_elems(scale, (col_perm[0], col_perm[1], g))
        minv = _q5q6_col_perm_elems(minv, (col_perm[0], col_perm[1], g))
    ql = _pack_nibble_interleaved(u5)
    qh = _preshuffle_qh1((u5 >> 4) & 1, K)
    return ql, qh, scale.contiguous(), minv.contiguous()


def _xpu_repack_q5_k_plain(qweight: torch.Tensor):
    """Q5_K rep for the GROUPED prefill down kernel: ql [N,K/2] interleaved nibble
    + PLAIN element-order qh [N,K/8] (byte j bit b = elem 8j+b) + scale,min [N,K/32].
    Differs from _xpu_repack_q5_k only in qh layout (no 512-tile pre-shuffle), so the
    grouped DPAS down kernel can index the 5th bit with simple per-element arithmetic.
    """
    u5, scale, minv = _xpu_q5_k_elem(qweight)
    N, K = u5.shape
    ql = _pack_nibble_interleaved(u5)
    hbit = ((u5 >> 4) & 1).to(torch.uint8).view(N, K // 8, 8)
    weights = (1 << torch.arange(8, dtype=torch.int32, device=u5.device))
    qh = (hbit.to(torch.int32) * weights).sum(dim=2).to(torch.uint8)   # [N, K/8]
    return ql.contiguous(), qh.contiguous(), scale.contiguous(), minv.contiguous()


def _xpu_repack_q6_k_plain(qweight: torch.Tensor):
    """Q6_K rep for the GROUPED prefill down kernel: ql [N,K/2] interleaved nibble
    + PLAIN element-order 2-bit qh [N,K/4] (byte j = elems 4j..4j+3, 2 bits each) +
    scale [N,K/16]. Symmetric (w = scale*(v6-32), no min). Mirrors moe_q6k_down_ggemv.h.
    """
    u6, scale = _xpu_q6_k_elem(qweight)
    N, K = u6.shape
    ql = _pack_nibble_interleaved(u6)
    h2 = ((u6 >> 4) & 3).to(torch.int32).view(N, K // 4, 4)
    shifts = (2 * torch.arange(4, dtype=torch.int32, device=u6.device))
    qh = (h2 << shifts).sum(dim=2).to(torch.uint8)   # [N, K/4]
    return ql.contiguous(), qh.contiguous(), scale.contiguous()


def _xpu_moe_grouped_prefill(xf, topk_ids, topk_weights, E, hidden, inter,
                             gate_ql, gate_sc, gate_mn, up_ql, up_sc, up_mn,
                             d_ql, d_qh, d_sc, d_mn, down_is_q6=False):
    """Grouped MoE prefill: sort tokens by expert -> Q4_K up GGEMV -> silu*mul ->
    Q5_K down GGEMV -> weighted accumulate. Returns out [M, hidden] fp32.
    xf [M, hidden] fp16; topk_ids/weights [M, top_k]. (notes §10ad-§10af)."""
    dev = xf.device
    M, top_k = topk_ids.shape
    flat_exp = topk_ids.reshape(-1).to(torch.int64)
    route_tok = torch.arange(M, device=dev).repeat_interleave(top_k)
    order = torch.argsort(flat_exp)
    exp_sorted = flat_exp[order]
    tok_sorted = route_tok[order].to(torch.int64)
    n_route = M * top_k
    # T2 (notes §10be): the up GGEMV folds the per-expert input gather into its
    # load via tok_sorted (reads xf directly at the sorted token id), so the
    # explicit `es = xf.index_select(0, tok_sorted)` round-trip (the IndexKernel,
    # ~10% of prefill) is gone. tok_sorted as int32 for the kernel.
    tok_sorted_i32 = tok_sorted.to(torch.int32).contiguous()
    counts = torch.bincount(exp_sorted, minlength=E)
    offs = torch.cat([torch.zeros(1, dtype=torch.long, device=dev), counts.cumsum(0)]).cpu().tolist()
    # The GGEMV kernel processes at most MAX_M=64 tokens per chunk (hardcoded
    # tiling); split any expert with >64 routed tokens into contiguous ≤64
    # sub-chunks, else tokens beyond 64 are silently dropped (wrong output).
    MAX_M = 64
    chunks = []
    for e in range(E):
        s, n = offs[e], offs[e + 1] - offs[e]
        while n > 0:
            c = min(MAX_M, n)
            chunks.append([e, s, c])
            s += c; n -= c
    chunks_t = torch.tensor(chunks, dtype=torch.int32, device=dev)
    gate_buf = torch.zeros(n_route, 2 * inter, dtype=torch.float16, device=dev)
    # read xf directly (xf must be row-contiguous in hidden); tok_sorted_i32 folds
    # the gather in. xf is [M, hidden] fp16 contiguous from the caller.
    _moe_grouped.moe_up_q4k_ggemv(xf, gate_ql, gate_sc, gate_mn, up_ql, up_sc, up_mn,
                                  gate_buf, chunks_t, hidden, inter, tok_sorted_i32)
    # P-ELEM-silu (notes §10bl): do silu*mul in fp16 directly. The old explicit
    # .float() on both halves materialized two [n_route, inter] fp32 intermediates
    # (the UnrolledElementwise/VectorizedElementwise flood in the prefill trace);
    # fp16 silu+mul is 4.3x faster (1412→332us @ n_route=8192) and bit-identical
    # (cos=1.0) since the kernels already produced fp16. (esimd_moe_silu_mul exists
    # as a binding but its op is in the [skip-ptl] moe module, not built — so the
    # fp16-lean torch path is the zero-build win.)
    inter_states = (torch.nn.functional.silu(gate_buf[:, :inter])
                    * gate_buf[:, inter:]).contiguous()
    out_route = torch.zeros(n_route, hidden, dtype=torch.float16, device=dev)
    if down_is_q6:
        _moe_grouped.moe_down_q6k_ggemv(inter_states, d_ql, d_qh, d_sc, out_route,
                                        chunks_t, inter, hidden)
    else:
        _moe_grouped.moe_down_q5k_ggemv(inter_states, d_ql, d_qh, d_sc, d_mn, out_route,
                                        chunks_t, inter, hidden)
    # Combine: each token has exactly top_k routes. Instead of an atomic
    # index_add_ scatter (fp32 atomics over 32k routes = the prefill IndexKernel
    # hot spot, ~19ms/layer; notes §10ar/§10as), un-sort the per-route outputs
    # back to [M, top_k] order then do a contiguous weighted reduce. ~1.7x faster,
    # bit-exact (cos=1.0). order is the expert-sort permutation; its argsort is
    # the inverse (sorted-order -> original route order).
    w_sorted = topk_weights.reshape(-1)[order].to(out_route.dtype)
    contrib = out_route * w_sorted.unsqueeze(1)                # [n_route, hidden] fp16, sorted
    inv = torch.argsort(order)
    contrib = contrib.index_select(0, inv)                     # back to route order
    out = contrib.view(M, top_k, hidden).sum(dim=1).float()    # contiguous reduce
    return out


def _xpu_dequant_q5_k(ql, qh, scale, minv, out_dtype):
    """Dequant packed q5_K -> dense [N,K] (for prefill M>1). Mirrors the kernel."""
    N = ql.shape[0]; K = ql.shape[1] * 2
    even = (ql & 0x0F).to(torch.int16); odd = ((ql >> 4) & 0x0F).to(torch.int16)
    v = torch.stack([even, odd], dim=2).view(N, K).to(torch.int32)
    # add 5th bit by inverting the pre-shuffle: shuffled byte t bit b -> elem b*64+t
    VL, VL8 = _Q5Q6_VL, _Q5Q6_VL // 8
    ntile = K // VL
    qhb = qh.to(torch.int32).view(N, ntile, VL8)
    high = torch.zeros(N, ntile, 8, VL8, dtype=torch.int32, device=ql.device)
    for b in range(8):
        high[:, :, b, :] = (qhb >> b) & 1
    v = v.view(N, ntile, VL) + (high.view(N, ntile, VL) << 4)
    v = v.view(N, K)
    nb = scale.shape[1]
    w = (v.to(out_dtype).view(N, nb, 32) * scale.to(out_dtype).unsqueeze(-1)
         - minv.to(out_dtype).unsqueeze(-1))
    return w.view(N, K).contiguous()


def _xpu_q6_k_elem(qweight):
    """GGUF q6_K -> element-order val6 [N,K] (0..63) + scale [N,K/16] f16 (sym)."""
    N = qweight.shape[0]
    nsb = qweight.shape[1] // _Q6_K_BYTES
    buf = qweight.reshape(N, nsb, _Q6_K_BYTES)
    ql = buf[:, :, 0:128].to(torch.int32)
    qh = buf[:, :, 128:192].to(torch.int32)
    sc = buf[:, :, 192:208].contiguous().view(torch.int8).float()
    d = buf[:, :, 208:210].contiguous().view(torch.float16).view(N, nsb).float()
    K = nsb * _Q4_K_SB
    u6 = torch.empty(N, nsb, _Q4_K_SB, dtype=torch.int32, device=qweight.device)
    for ip in range(2):
        q0 = ql[:, :, 64 * ip + 0:64 * ip + 32]
        q32 = ql[:, :, 64 * ip + 32:64 * ip + 64]
        h = qh[:, :, 32 * ip:32 * ip + 32]
        u6[:, :, 128 * ip + 0:128 * ip + 32] = (q0 & 0xF) | (((h >> 0) & 3) << 4)
        u6[:, :, 128 * ip + 32:128 * ip + 64] = (q32 & 0xF) | (((h >> 2) & 3) << 4)
        u6[:, :, 128 * ip + 64:128 * ip + 96] = (q0 >> 4) | (((h >> 4) & 3) << 4)
        u6[:, :, 128 * ip + 96:128 * ip + 128] = (q32 >> 4) | (((h >> 6) & 3) << 4)
    scale16 = (d[:, :, None] * sc).to(torch.float16)
    return u6.view(N, K), scale16.view(N, K // 16)


def _xpu_repack_q6_k(qweight: torch.Tensor, col_perm=None):
    """GGUF q6_K -> PACKED (ql [N,K/2] nibble + pre-shuffled qh [N,K/4] + scale
    [N,K/16] f16, symmetric). Zero extra memory. Optional col_perm in elem order."""
    u6, scale = _xpu_q6_k_elem(qweight)
    N, K = u6.shape
    if col_perm is not None:
        u6 = _q5q6_col_perm_elems(u6, col_perm)
        g = col_perm[2] // 16
        scale = _q5q6_col_perm_elems(scale, (col_perm[0], col_perm[1], g))
    ql = _pack_nibble_interleaved(u6)
    qh = _preshuffle_qh2((u6 >> 4) & 3, K)
    return ql, qh, scale.contiguous()


def _xpu_dequant_q6_k(ql, qh, scale16, out_dtype):
    """Dequant packed q6_K -> dense [N,K] (prefill M>1). Symmetric w=scale*(v6-32)."""
    N = ql.shape[0]; K = ql.shape[1] * 2
    even = (ql & 0x0F).to(torch.int16); odd = ((ql >> 4) & 0x0F).to(torch.int16)
    v = torch.stack([even, odd], dim=2).view(N, K).to(torch.int32)
    VL, VLQ = _Q5Q6_VL, _Q5Q6_VL // 4
    ntile = K // VL
    qhb = qh.to(torch.int32).view(N, ntile, VLQ)
    high = torch.zeros(N, ntile, 4, VLQ, dtype=torch.int32, device=ql.device)
    for p in range(4):
        high[:, :, p, :] = (qhb >> (2 * p)) & 3
    v = v.view(N, ntile, VL) + (high.view(N, ntile, VL) << 4)
    v = v.view(N, K)
    nb = scale16.shape[1]
    w = (v.to(out_dtype).view(N, nb, 16) - 32) * scale16.to(out_dtype).unsqueeze(-1)
    return w.view(N, K).contiguous()


_Q8_0_BLOCK_BYTES = 34  # GGML q8_0: {fp16 d; int8 qs[32]} per 32 elements


def _xpu_repack_q8_0(qweight: torch.Tensor):
    """GGUF q8_0 raw blocks -> ESIMD split buffer (qs [N,K] int8, scale [N,K/32] f16).

    qweight in: [N, blocks*34] uint8 (per-output-row, K/32 contiguous 34-byte
    q8_0 blocks: {fp16 d; int8 qs[32]}). GGML q8_0 is symmetric: w = d * qs,
    qs signed int8, no min. Bit-exact repack (validated in
    cc_workspace/tools/q8_0_repack_ref.py vs gguf-lib).
    """
    N = qweight.shape[0]
    blocks = qweight.shape[1] // _Q8_0_BLOCK_BYTES
    buf = qweight.reshape(N, blocks, _Q8_0_BLOCK_BYTES)
    scale = buf[:, :, 0:2].contiguous().view(torch.float16).view(N, blocks)
    qs = buf[:, :, 2:34].contiguous().view(torch.int8).view(N, blocks * 32)  # [N,K]
    return qs.contiguous(), scale.contiguous()


def _xpu_dequant_q8_0(qs: torch.Tensor, scale: torch.Tensor,
                      out_dtype: torch.dtype) -> torch.Tensor:
    """Dequant the q8_0 split rep -> dense [N, K]. w = scale[k/32] * qs."""
    N, K = qs.shape
    blocks = scale.shape[1]
    vals = qs.to(out_dtype).view(N, blocks, -1) * scale.to(out_dtype).unsqueeze(-1)
    return vals.view(N, K).contiguous()


def _xpu_prepare_shard(qweight: torch.Tensor, qweight_type: int,
                       params_dtype: torch.dtype, col_perm=None):
    """Return a resident per-shard rep: q4_0 -> ('q4_0', packed_u8, scale_f16),
    q5_k/q6_k -> packed (ql, qh, scale[, min]), other quant -> ('fp16', dense),
    unquantized -> ('fp16', w, None).

    col_perm (GDN out_proj value-head reorder) is applied INSIDE the quant repack
    in element order before packing (so it never splits a packed block); for the
    fp16 rep it is applied by _xpu_permute_gdn_out_cols at the call site.
    """
    # Debug switch: force q4_0 through the CPU gguf-lib dequant -> fp16 dense
    # path (bypassing the ESIMD kernels) to isolate kernel numerics from the
    # rest of the GGUF integration.
    _force_dq = os.environ.get("SGLANG_GGUF_XPU_FORCE_DEQUANT") == "1"
    if qweight_type == _Q4_0_TYPE and esimd_gemv_q4_0 is not None and not _force_dq:
        packed, scale = _xpu_repack_q4_0(qweight)
        # q4_0 is interleaved nibble [N,K/2]; a head_v_dim col-perm is not used
        # by any q4_0 GDN layer in Qwen3.5/3.6 (out_proj is Q5_K/Q8_0). Guard.
        assert col_perm is None, "q4_0 GDN out_proj col-perm unsupported"
        return ("q4_0", packed, scale)
    if qweight_type == _Q8_0_TYPE and esimd_gemv_q8_0 is not None and not _force_dq:
        qs, scale = _xpu_repack_q8_0(qweight)  # qs [N,K] int8, scale [N,K/32] f16
        if col_perm is not None and qs.numel() > 0:
            # GDN out_proj (35B ssm_out is Q8_0): permute K (input) columns in
            # element order, head_v_dim-granular; scale follows at hvd//32.
            ratio, nk, hvd = col_perm
            qs = _q5q6_col_perm_elems(qs, col_perm)
            scale = _q5q6_col_perm_elems(scale, (ratio, nk, hvd // 32))
        return ("q8_0", qs, scale)
    if qweight_type == _Q4_K_TYPE and esimd_gemv_q4_k is not None and not _force_dq:
        ql, scale, minv = _xpu_repack_q4_k(qweight)
        return ("q4_k", ql, scale, minv)
    _no_q5 = os.environ.get("SGLANG_GGUF_XPU_NO_Q5K") == "1"
    _no_q6 = os.environ.get("SGLANG_GGUF_XPU_NO_Q6K") == "1"
    if (qweight_type == _Q5_K_TYPE and esimd_gemv_q5_k is not None
            and not _force_dq and not _no_q5):
        ql, qh, scale, minv = _xpu_repack_q5_k(qweight, col_perm=col_perm)
        return ("q5_k", ql, qh, scale, minv)
    if (qweight_type == _Q6_K_TYPE and esimd_gemv_q6_k is not None
            and not _force_dq and not _no_q6):
        ql, qh, scale = _xpu_repack_q6_k(qweight, col_perm=col_perm)
        return ("q6_k", ql, qh, scale)
    if qweight_type in UNQUANTIZED_TYPES:
        rep = ("fp16", qweight.to(params_dtype), None)
    else:
        rep = ("fp16", _xpu_dequant_to_fp16(qweight, qweight_type, params_dtype), None)
    if col_perm is not None:
        rep = _xpu_permute_gdn_out_cols(rep, col_perm)
    return rep


def _xpu_dequant_rep_to_fp16(rep, out_dtype: torch.dtype) -> torch.Tensor:
    """Dense [N, K] from a packed rep tuple (inverse of _xpu_prepare_shard).

    Used by the tied lm_head path, which needs a full vocab matmul after
    process_weights_after_loading has discarded the raw qweight.
    """
    kind = rep[0]
    if kind == "fp16":
        return rep[1].to(out_dtype)
    if kind == "q8_0":
        return _xpu_dequant_q8_0(rep[1], rep[2], out_dtype)
    if kind == "q4_k":
        return _xpu_dequant_q4_k(rep[1], rep[2], rep[3], out_dtype)
    if kind == "q5_k":
        return _xpu_dequant_q5_k(rep[1], rep[2], rep[3], rep[4], out_dtype)
    if kind == "q6_k":
        return _xpu_dequant_q6_k(rep[1], rep[2], rep[3], out_dtype)
    if kind == "q4_0":
        return _xpu_dequant_q4_0_packed(rep[1], rep[2], out_dtype)
    raise NotImplementedError(f"GGUF rep '{kind}' -> dense on XPU")


# Cache of per-group scales transposed to oneDNN's [num_groups, N] layout,
# keyed by the q4_0 scale tensor's data_ptr (one-time transpose per shard).
_onednn_scale_cache = {}


def _onednn_scale_t(scale: torch.Tensor) -> torch.Tensor:
    """GGUF q4_0 scale [N, K/32] f16 -> oneDNN [num_groups=K/32, N] f16 (cached)."""
    key = scale.data_ptr()
    st = _onednn_scale_cache.get(key)
    if st is None or st.shape[1] != scale.shape[0]:
        st = scale.t().contiguous()
        _onednn_scale_cache[key] = st
    return st


def _xpu_shard_matmul(x: torch.Tensor, rep) -> torch.Tensor:
    """x [M,K] fp16 @ shard^T -> [M,N] fp16. rep from _xpu_prepare_shard."""
    kind = rep[0]
    if kind == "q4_0":
        _, packed, scale = rep
        N = packed.shape[0]
        M = x.shape[0]
        xf = x.to(torch.float16).contiguous()
        if M == 1:
            # Decode: the ESIMD GEMV is bandwidth-optimal (~3x faster than a
            # dense fp16 matmul at M=1).
            out = torch.empty(M, N, dtype=torch.float16, device=x.device)
            esimd_gemv_q4_0(xf, packed, scale, out)
            return out
        # Prefill (M>1): oneDNN u4 fused-dequant matmul — keeps the weight int4
        # (no fp16 DRAM round-trip), ~6x faster than dequant+matmul (notes §10v).
        # The ESIMD packed rep ([N,K/2] interleaved low=2j/high=2j+1) IS the u4
        # layout oneDNN wants; GGUF offset-binary nibble n == u4 with zp=8 so the
        # result is bit-exact. Falls back to dequant+matmul if the ext is absent.
        if _onednn_gguf is not None:
            return _onednn_gguf.onednn_q4_gemm(xf, packed, _onednn_scale_t(scale))
        w = _xpu_dequant_q4_0_packed(packed, scale, torch.float16)  # [N, K]
        return xf @ w.t()
    if kind == "q8_0":
        _, qs, scale = rep
        N = qs.shape[0]
        M = x.shape[0]
        xf = x.to(torch.float16).contiguous()
        if M == 1:
            # Decode: ESIMD q8_0 GEMV (int8 resident, bandwidth-optimal).
            out = torch.empty(M, N, dtype=torch.float16, device=x.device)
            esimd_gemv_q8_0(xf, qs, scale, out)
            return out
        # Prefill (M>1): oneDNN s8 fused-dequant matmul (symmetric, no zero-point;
        # qs [N,K] int8 IS the s8 weight, scale transposed to [K/32,N]). Keeps the
        # weight int8, no fp16 DRAM round-trip. Covers 35B dense prefill (notes
        # §10x). Falls back to dequant+matmul if the ext is absent.
        if _onednn_gguf is not None:
            return _onednn_gguf.onednn_q8_gemm(xf, qs, _onednn_scale_t(scale))
        w = _xpu_dequant_q8_0(qs, scale, torch.float16)  # [N, K]
        return xf @ w.t()
    if kind == "q4_k":
        _, ql, scale, minv = rep
        N = ql.shape[0]
        M = x.shape[0]
        xf = x.to(torch.float16).contiguous()
        if M == 1:
            # Decode: ESIMD q4_K GEMV (4.5-bit resident, asymmetric scale+min).
            out = torch.empty(M, N, dtype=torch.float16, device=x.device)
            esimd_gemv_q4_k(xf, ql, scale, minv, out)
            return out
        # Prefill (M>1): dequant 4-bit -> fp16 once + dense matmul.
        w = _xpu_dequant_q4_k(ql, scale, minv, torch.float16)  # [N, K]
        return xf @ w.t()
    if kind == "q5_k":
        _, ql, qh, scale, minv = rep
        N = ql.shape[0]
        M = x.shape[0]
        xf = x.to(torch.float16).contiguous()
        if M == 1:
            out = torch.empty(M, N, dtype=torch.float16, device=x.device)
            esimd_gemv_q5_k(xf, ql, qh, scale, minv, out)
            return out
        w = _xpu_dequant_q5_k(ql, qh, scale, minv, torch.float16)  # [N, K]
        return xf @ w.t()
    if kind == "q6_k":
        _, ql, qh, scale = rep
        N = ql.shape[0]
        M = x.shape[0]
        xf = x.to(torch.float16).contiguous()
        if M == 1:
            out = torch.empty(M, N, dtype=torch.float16, device=x.device)
            esimd_gemv_q6_k(xf, ql, qh, scale, out)
            return out
        w = _xpu_dequant_q6_k(ql, qh, scale, torch.float16)  # [N, K]
        return xf @ w.t()
    # fp16-resident dense weight [N, K]
    _, w, _ = rep
    return x.to(w.dtype) @ w.t()


def _xpu_permute_gdn_out_cols(rep, perm):
    """Permute the K (input/value-head) columns of a GDN out_proj rep from GGUF
    [ratio, num_k] order to HF [num_k, ratio], head_v_dim-granular.

    Applies to whatever resident rep _xpu_prepare_shard produced. The permute
    moves contiguous head_v_dim-element column blocks, so for the quant reps the
    per-group scale/min tensors are permuted at head_v_dim//group granularity.
    Requires head_v_dim % group == 0 (asserted) so a block is never split.
    """
    ratio, nk, hvd = perm

    def _perm_last(t, blk):
        # t [N, C], reorder C in blocks of `blk` as [ratio,nk]->[nk,ratio]
        N, C = t.shape
        assert C == ratio * nk * blk, (
            f"GDN col-perm shape mismatch: C={C} != {ratio}*{nk}*{blk}")
        return t.reshape(N, ratio, nk, blk).transpose(1, 2).reshape(N, C).contiguous()

    kind = rep[0]
    if kind == "fp16":
        # Empty / not-loaded weight (e.g. an out_proj param on a layer that has
        # no ssm_out in this checkpoint): nothing to permute.
        if rep[1].dim() < 2 or rep[1].numel() == 0:
            return rep
        return ("fp16", _perm_last(rep[1], hvd), None)
    # Quant reps (q5_k/q6_k/...) apply col_perm INSIDE _xpu_repack_* in element
    # order before packing, so a quant rep must never reach here.
    raise NotImplementedError(
        f"GDN out_proj col-perm reached for non-fp16 rep '{kind}'; quant col-perm "
        f"is done in _xpu_repack_* (pass col_perm to _xpu_prepare_shard instead).")


def _xpu_perm_rep_rows(rep, perm: torch.Tensor):
    """Permute the OUTPUT-N rows (axis 0) of a resident rep. N is the output
    dim, so out[:, perm] == (rep with rows permuted) @ x. Bit-exact (rows are
    independent). Handles every rep kind by permuting each row-indexed tensor.
    perm is a LongTensor of length N on the rep's device."""
    kind = rep[0]
    perm = perm.to(rep[1].device)
    if kind == "fp16":
        return ("fp16", rep[1].index_select(0, perm).contiguous(), None)
    if kind == "q8_0":
        # ("q8_0", qs[N,K] int8, scale[N,K/32] f16) — both row-indexed by N
        return ("q8_0",
                rep[1].index_select(0, perm).contiguous(),
                rep[2].index_select(0, perm).contiguous())
    if kind == "q4_0":
        return ("q4_0",
                rep[1].index_select(0, perm).contiguous(),
                rep[2].index_select(0, perm).contiguous())
    # k-quant reps (q4_k/q5_k/q6_k) are not expected for the GDN in_proj on the
    # 35B (q8_0); fall back to an explicit error rather than silently mis-permute.
    raise NotImplementedError(
        f"_xpu_perm_rep_rows: output-row perm not implemented for rep kind {kind}"
    )


def _xpu_bake_out_row_perm(reps: dict, merged, ids: list, perm: torch.Tensor):
    """OPT-1: bake the GDN qkvz/ba interleave (a static output-feature
    permutation) into the rep ROWS so the GEMV emits the kernel's interleaved
    layout directly, eliminating the per-step repack cat.

    Builds ONE assembled rep in shard-order row layout (= the [q|k|v|z] /
    [b|a] concatenation the fast path used to cat), permutes its rows by `perm`,
    and returns (reps={'_single': rep}, merged=(rep, sizes), order=['_single'])
    so apply() runs a single GEMV with no per-step cat/split.

    Requires all shards share one rep kind (true for 35B: qkvz all q8_0, ba all
    fp16). perm length must equal the assembled N (sum of per-shard N)."""
    if merged is not None:
        # q8_0/q4_0 already row-cat in shard order -> just permute its rows.
        merged_rep, sizes = merged
        N = merged_rep[1].shape[0]
        assert perm.numel() == N, f"perm {perm.numel()} != merged N {N}"
        pr = _xpu_perm_rep_rows(merged_rep, perm)
        return {"_single": pr}, (pr, [N]), ["_single"]
    # not merged (e.g. ba = 2 fp16 shards): row-cat the per-shard reps in shard
    # order into one rep, then permute. All must be the same kind.
    kinds = {reps[i][0] for i in ids}
    assert len(kinds) == 1, f"_xpu_bake_out_row_perm: mixed kinds {kinds}"
    kind = next(iter(kinds))
    if kind == "fp16":
        w = torch.cat([reps[i][1] for i in ids], dim=0).contiguous()  # [sumN, K]
        N = w.shape[0]
        assert perm.numel() == N, f"perm {perm.numel()} != cat N {N}"
        pr = ("fp16", w.index_select(0, perm.to(w.device)).contiguous(), None)
        return {"_single": pr}, (pr, [N]), ["_single"]
    # quant + unmerged (shouldn't happen for GDN in_proj): cat row-indexed tensors
    qs = torch.cat([reps[i][1] for i in ids], dim=0)
    sc = torch.cat([reps[i][2] for i in ids], dim=0)
    pr = _xpu_perm_rep_rows((kind, qs, sc), perm)
    return {"_single": pr}, (pr, [pr[1].shape[0]]), ["_single"]


def _xpu_try_merge_shards(reps: dict, ids: list):
    """D1: if every shard in `ids` is the SAME GEMV rep kind (q8_0 or q4_0) with
    the same K, build one merged rep by row-concatenating the per-shard weights,
    plus the per-shard N sizes (to slice the output). Returns
    (merged_rep, [N0, N1, ...]) or None if not mergeable (mixed kinds / fp16 /
    k-quant — those keep the per-shard path). Bit-exact: q8_0/q4_0 rows are
    independent, so cat-then-GEMV == per-shard-GEMV-then-cat.
    notes §10bj."""
    kinds = {reps[i][0] for i in ids}
    if len(ids) < 2 or len(kinds) != 1:
        return None
    kind = next(iter(kinds))
    if kind == "q8_0":
        # rep = ("q8_0", qs[N,K] int8, scale[N,K/32] f16)
        Ks = {reps[i][1].shape[1] for i in ids}
        if len(Ks) != 1:
            return None
        sizes = [reps[i][1].shape[0] for i in ids]
        qs = torch.cat([reps[i][1] for i in ids], dim=0).contiguous()
        sc = torch.cat([reps[i][2] for i in ids], dim=0).contiguous()
        return (("q8_0", qs, sc), sizes)
    if kind == "q4_0":
        # rep = ("q4_0", packed[N,K/2] u8, scale[N,K/32] f16); same K -> same K/2
        Ks = {reps[i][1].shape[1] for i in ids}
        if len(Ks) != 1:
            return None
        sizes = [reps[i][1].shape[0] for i in ids]
        packed = torch.cat([reps[i][1] for i in ids], dim=0).contiguous()
        sc = torch.cat([reps[i][2] for i in ids], dim=0).contiguous()
        return (("q4_0", packed, sc), sizes)
    return None


class GGUFLinearXPUMethod(GGUFLinearMethod):
    """GGUF linear for Intel XPU (PTL Xe3).

    Mirrors GGUFLinearMethod's create_weights / padded-weight handling, but in
    process_weights_after_loading converts every (shard of every) weight to a
    resident representation: q4_0 -> repacked INT4 (ESIMD kernel), other types
    -> fp16 dense. apply then dispatches per shard with no per-call branching on
    raw GGUF bytes.
    """

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            raise ValueError(
                f"Unsupported GGUF quantization type {WeightType(qweight_type)} on XPU."
            )
        self._create_padded_weight_param(layer)

        qweight = layer.qweight
        shard_id = getattr(qweight, "shard_id", None)
        reps = {}
        if shard_id and hasattr(qweight, "shard_offset_map"):
            # shard_id is in checkpoint *load* (yield) order, which for GGUF is
            # the file's tensor order — NOT the fused parameter's logical order.
            # The fused output must be concatenated by ascending shard index
            # (q,k,v / 0,1,2,3), else e.g. in_proj_ba comes out as [a,b] instead
            # of [b,a] (ssm_alpha is yielded before ssm_beta) and the GDN
            # recurrence gets a<->b swapped. Sort to the logical order here.
            if "q" in shard_id:
                ids = ["q", "k", "v"]
            else:
                ids = sorted(shard_id)
            for idx in ids:
                start, end, offset = qweight.shard_offset_map[idx]
                # F32 shards (e.g. the 35B's ssm_alpha/beta -> in_proj_a/b) carry
                # no qweight_type from the gguf iterator; default to F32 (0) so
                # _xpu_prepare_shard takes the unquantized fp16 path.
                stype = layer.qweight_type.shard_weight_type.get(idx, 0)
                w = qweight[start:end, :offset].contiguous()
                reps[idx] = _xpu_prepare_shard(w, stype, self.params_dtype)
            layer._xpu_shard_order = ids
            # D1 (notes §10bf/§10bj): the per-shard q8_0 GEMVs at decode are small-N
            # (k/v [512,2048] hit only ~48 GB/s = 2.31x BW floor — small N starves
            # the LSC). If ALL shards share the SAME q8_0/q4_0 rep kind and the same
            # K, merge their weights into ONE big-N rep (cat rows) so decode runs a
            # SINGLE large-N GEMV (better LSC util), then slice the output back per
            # shard. Bit-exact (row-independent). Only for the GEMV reps; fp16 /
            # k-quant shards keep the per-shard path.
            layer._xpu_merged = _xpu_try_merge_shards(reps, ids)
            # OPT-1 (notes §11/§12): GDN in_proj_qkvz/ba feed the gdn_attention
            # kernel, which wants a per-k-head-group INTERLEAVED feature layout.
            # The model normally produces that with a per-step torch.cat repack
            # (_repack_qkvz_ba_for_gdn_attention) — ~90 CatArrayBatchedCopy/step,
            # launch-bound at M=1 decode (~0.5ms/step, notes §12). Since the repack
            # is a STATIC output-feature permutation (POC bit-exact,
            # tools/verify_gdn_repack_as_weight_perm.py), bake it into the rep ROWS
            # ONCE here so the GEMV emits interleaved directly and the cat is gone.
            # The model sets layer._gguf_gdn_out_row_perm at init.
            out_perm = getattr(layer, "_gguf_gdn_out_row_perm", None)
            if out_perm is not None:
                layer._xpu_reps, layer._xpu_merged, layer._xpu_shard_order = (
                    _xpu_bake_out_row_perm(
                        reps, layer._xpu_merged, ids, out_perm
                    )
                )
                reps = layer._xpu_reps
        else:
            # GDN out_proj: permute the input (value-head) columns from GGUF
            # [ratio, num_k] order to HF [num_k, ratio]. For quant reps this is
            # applied inside _xpu_repack_* in element order before packing (so it
            # never splits a packed block); for fp16 it is applied post-dequant.
            # The permute is head_v_dim-granular along K (the input dim).
            perm = getattr(layer, "_gguf_gdn_col_perm", None)
            rep = _xpu_prepare_shard(
                qweight.data, qweight_type, self.params_dtype, col_perm=perm
            )
            reps["_single"] = rep
            layer._xpu_shard_order = ["_single"]
            layer._xpu_merged = None
        layer._xpu_reps = reps
        # free the raw GGUF bytes
        if hasattr(layer, "qweight"):
            del layer.qweight

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x2 = x.reshape(-1, x.shape[-1])
        merged = getattr(layer, "_xpu_merged", None)
        if merged is not None:
            # D1: one big-N GEMV over the merged q8_0/q4_0 shards (§10bj). The
            # merged rep is row-cat in shard_order (q,k,v), so its [M, sum(N)]
            # output IS already the fused-output concatenation — no split needed.
            merged_rep, _sizes = merged
            out = _xpu_shard_matmul(x2, merged_rep)
        else:
            parts = [_xpu_shard_matmul(x2, layer._xpu_reps[idx])
                     for idx in layer._xpu_shard_order]
            out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=1)
        # The q4_0 ESIMD kernels are fp16-only (PTL has no bf16 ESIMD), so a
        # bf16 network would otherwise get an fp16 tensor back here. Cast the
        # result to the input activation dtype to keep the graph type-consistent.
        out = out.to(x.dtype)
        if bias is not None:
            out = out + bias
        return out.reshape(*x.shape[:-1], out.shape[-1])


class GGUFMoEMethod(FusedMoEMethodBase):
    """MoE method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        # gate up proj
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        # gate down proj
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )

        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        assert self.fused_experts is None

        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert (
            self.moe_runner_config.activation == "silu"
        ), "Only SiLU activation is supported."

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output

        moe_runner_config = self.moe_runner_config

        topk_weights, topk_ids, _ = topk_output
        output = fused_moe_gguf(
            x=x,
            w1=layer.w13_qweight,
            w2=layer.w2_qweight,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            qweight_type=layer.w13_qweight_type.weight_type,
            qweight_type2=layer.w2_qweight_type.weight_type,
            activation=moe_runner_config.activation,
        )
        return StandardCombineInput(hidden_states=output)


def _xpu_dequant_rep(rep, out_dtype=torch.float16):
    """Dequant any resident rep (from _xpu_prepare_shard) -> dense [N, K]."""
    kind = rep[0]
    if kind == "q4_0":
        return _xpu_dequant_q4_0_packed(rep[1], rep[2], out_dtype)
    if kind == "q8_0":
        return _xpu_dequant_q8_0(rep[1], rep[2], out_dtype)
    if kind == "q4_k":
        return _xpu_dequant_q4_k(rep[1], rep[2], rep[3], out_dtype)
    if kind == "q5_k":
        return _xpu_dequant_q5_k(rep[1], rep[2], rep[3], rep[4], out_dtype)
    if kind == "q6_k":
        return _xpu_dequant_q6_k(rep[1], rep[2], rep[3], out_dtype)
    # fp16 dense
    return rep[1].to(out_dtype)


def _xpu_rep_gemv(x_row, rep):
    """x_row [1,K] fp16 @ rep^T -> [1,N] fp16 via the matching ESIMD GEMV."""
    kind = rep[0]
    N = rep[1].shape[0]
    out = torch.empty(1, N, dtype=torch.float16, device=x_row.device)
    if kind == "q4_0":
        esimd_gemv_q4_0(x_row, rep[1], rep[2], out)
    elif kind == "q8_0":
        esimd_gemv_q8_0(x_row, rep[1], rep[2], out)
    elif kind == "q4_k":
        esimd_gemv_q4_k(x_row, rep[1], rep[2], rep[3], out)
    elif kind == "q5_k":
        esimd_gemv_q5_k(x_row, rep[1], rep[2], rep[3], rep[4], out)
    elif kind == "q6_k":
        esimd_gemv_q6_k(x_row, rep[1], rep[2], rep[3], out)
    else:  # fp16
        return x_row.to(rep[1].dtype) @ rep[1].t()
    return out


class GGUFMoEXPUMethod(FusedMoEMethodBase):
    """GGUF FusedMoE for Intel XPU (PTL Xe3).

    Mirrors GGUFMoEMethod.create_weights (w13/w2 GGUFUninitializedParameter with
    data_container), but in process_weights_after_loading repacks each expert's
    gate/up (w13) and down (w2) GGUF bytes into the resident ESIMD k-quant rep
    (Q4_K/Q5_K/Q6_K/Q8_0 — reuses _xpu_prepare_shard / _xpu_repack_*). apply
    routes M=1 decode through per-expert ESIMD GEMV and M>1 prefill through
    per-expert dequant + matmul (functional baseline, mirrors AWQMoEXPUMethod).

    The shared expert (ffn_*_shexp) is a separate dense Linear handled by
    GGUFLinearXPUMethod; only the routed experts come through here.
    """

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        # Mirror GGUFMoEMethod: w13 (gate+up fused) + w2 (down), GGUF-uninit.
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {"input_dim": 1, "output_dim": 0, "tensor_shape": tensor_shape,
             "is_gguf_weight": True, "data_container": []},
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)
        w13_qweight_type = Parameter(torch.empty(1, dtype=torch.uint8),
                                     requires_grad=False)
        set_weight_attrs(w13_qweight_type,
                         {"is_gguf_weight_type": True, "weight_type": 0,
                          "ignore_warning": True})
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {"input_dim": 1, "output_dim": 0, "tensor_shape": tensor_shape,
             "is_gguf_weight": True, "data_container": []},
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)
        w2_qweight_type = Parameter(torch.empty(1, dtype=torch.uint8),
                                    requires_grad=False)
        set_weight_attrs(w2_qweight_type,
                         {"is_gguf_weight_type": True, "weight_type": 0,
                          "ignore_warning": True})
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

    def create_moe_runner(self, layer, moe_runner_config):
        self.moe_runner_config = moe_runner_config

    def process_weights_after_loading(self, layer: torch.nn.Module):
        # Materialize the per-expert GGUF byte tensors from data_container.
        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()
        w13 = layer.w13_qweight            # [E, 2*inter_bytes_rows, hidden_bytes]
        w2 = layer.w2_qweight              # [E, inter_rows, hidden_bytes]
        w13_type = int(layer.w13_qweight_type.weight_type)
        w2_type = int(layer.w2_qweight_type.weight_type)
        E = w13.shape[0]
        half = w13.shape[1] // 2           # gate rows | up rows
        # Per-expert repack to packed k-quant reps, then STACK into [E, ...]
        # contiguous buffers for the fused MoE kernels (one launch over all
        # routed pairs, vs the old per-expert Python GEMV loop). Q4_K gate/up;
        # Q5_K/Q6_K down. Both packed -> zero extra resident memory.
        self._w13_type, self._w2_type = w13_type, w2_type
        assert w13_type == _Q4_K_TYPE, (
            f"GGUFMoEXPUMethod fused path expects Q4_K gate/up, got {w13_type}")
        assert w2_type in (_Q5_K_TYPE, _Q6_K_TYPE), (
            f"GGUFMoEXPUMethod fused path expects Q5_K/Q6_K down, got {w2_type}")
        gq, gs, gm, uq, us, um = [], [], [], [], [], []
        dq, dh, ds, dm = [], [], [], []
        self._down_is_q6 = (w2_type == _Q6_K_TYPE)
        for e in range(E):
            ql, sc, mn = _xpu_repack_q4_k(w13[e, :half, :].contiguous())
            gq.append(ql); gs.append(sc); gm.append(mn)
            ql, sc, mn = _xpu_repack_q4_k(w13[e, half:, :].contiguous())
            uq.append(ql); us.append(sc); um.append(mn)
            if self._down_is_q6:
                ql, qh, sc = _xpu_repack_q6_k(w2[e].contiguous())
                dq.append(ql); dh.append(qh); ds.append(sc)
            else:
                ql, qh, sc, mn = _xpu_repack_q5_k(w2[e].contiguous())
                dq.append(ql); dh.append(qh); ds.append(sc); dm.append(mn)
        dev = w13.device
        self.gate_ql = torch.stack(gq).contiguous(); self.gate_sc = torch.stack(gs).contiguous(); self.gate_mn = torch.stack(gm).contiguous()
        self.up_ql = torch.stack(uq).contiguous(); self.up_sc = torch.stack(us).contiguous(); self.up_mn = torch.stack(um).contiguous()
        self.down_ql = torch.stack(dq).contiguous(); self.down_qh = torch.stack(dh).contiguous(); self.down_sc = torch.stack(ds).contiguous()
        self.down_mn = (torch.stack(dm).contiguous() if not self._down_is_q6
                        else torch.zeros(1, dtype=torch.float16, device=dev))
        # dims: w13 gate rows = intermediate; hidden from gate K (=ql cols*2).
        self.intermediate = half
        self.hidden = self.gate_ql.shape[2] * 2
        self.E = E

        # Grouped prefill (M>1): reuse the decode reps with ZERO extra resident
        # memory. The up GGEMV takes gate_ql/up_ql separately (no [E,2*inter,...]
        # concat copy). The down GGEMV reuses down_ql/down_sc/down_mn and only
        # needs a small PLAIN element-order qh (the decode qh is pre-shuffled and
        # can't be reused). Both Q5_K (1-bit qh) and Q6_K (2-bit qh) down supported.
        self._grouped_ok = _moe_grouped is not None
        if self._grouped_ok:
            dh2 = []
            for e in range(E):
                if self._down_is_q6:
                    _, qh_plain, _ = _xpu_repack_q6_k_plain(w2[e].contiguous())
                else:
                    _, qh_plain, _, _ = _xpu_repack_q5_k_plain(w2[e].contiguous())
                dh2.append(qh_plain)
            self.down_qh_plain = torch.stack(dh2).contiguous()

        layer._xpu_moe_ready = True
        del layer.w13_qweight
        del layer.w2_qweight

    def apply(self, layer: torch.nn.Module, dispatch_output) -> "CombineInput":
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        assert self.moe_runner_config.activation == "silu", \
            "GGUFMoEXPUMethod only supports SiLU activation."
        x = dispatch_output.hidden_states
        topk_weights, topk_ids, _ = dispatch_output.topk_output

        x2 = x.reshape(-1, x.shape[-1])
        M = x2.shape[0]
        out = torch.zeros_like(x2)
        if M == 0:
            return StandardCombineInput(hidden_states=out.reshape_as(x))
        top_k = topk_ids.shape[1]
        hidden, inter = self.hidden, self.intermediate
        n_routed = M * top_k
        xf = x2.to(torch.float16).contiguous()
        sel = topk_ids.reshape(-1).to(torch.int32).contiguous()
        tw = topk_weights.reshape(-1).to(torch.float16).contiguous()

        # Prefill (M>1): grouped GGEMV (sort tokens by expert -> one DPAS GEMM per
        # expert group). ~17x over per-route GEMV which is 161x off the compute
        # floor at prefill (notes §10x/§10ad). M==1 decode keeps the per-route
        # fused GEMV below (BW-bound, GEMV is optimal there).
        if M > 1 and getattr(self, "_grouped_ok", False):
            out_g = _xpu_moe_grouped_prefill(
                xf, topk_ids, topk_weights, self.E, hidden, inter,
                self.gate_ql, self.gate_sc, self.gate_mn,
                self.up_ql, self.up_sc, self.up_mn,
                self.down_ql, self.down_qh_plain, self.down_sc, self.down_mn,
                down_is_q6=self._down_is_q6)
            return StandardCombineInput(hidden_states=out_g.to(out.dtype).reshape_as(x))

        # Fused: 1 up launch (gate/up Q4_K + silu*up) + 1 down launch (Q5_K/Q6_K
        # weighted) over ALL routed pairs, then sum the top_k partials. Replaces
        # the old top_k*3-GEMV-per-token Python loop (launch-bound at decode).
        inter_buf = torch.empty(n_routed, inter, dtype=torch.float16, device=x2.device)
        esimd_moe_up_q4k(xf, self.gate_ql, self.gate_sc, self.gate_mn,
                         self.up_ql, self.up_sc, self.up_mn, sel, inter_buf,
                         M, hidden, inter, top_k)
        out_partial = torch.empty(n_routed, hidden, dtype=torch.float16, device=x2.device)
        if self._down_is_q6:
            esimd_moe_down_q6k(inter_buf, self.down_ql, self.down_qh, self.down_sc,
                               sel, tw, out_partial, M, hidden, inter, top_k)
        else:
            esimd_moe_down_q5k(inter_buf, self.down_ql, self.down_qh, self.down_sc,
                               self.down_mn, sel, tw, out_partial, M, hidden, inter, top_k)
        # sum the top_k per-route partials back to per-token output (one op).
        out = out_partial.view(M, top_k, hidden).sum(dim=1).to(out.dtype)
        return StandardCombineInput(hidden_states=out.reshape_as(x))


class GGUFEmbeddingMethod(GGUFLinearMethod):
    """Embedding method for GGUF.

    Args:
        quant_config: The GGUF quantization config.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]

        return apply_gguf_embedding(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )


class GGUFUninitializedParameter(UninitializedParameter):
    cls_to_become = Parameter
    data_container: list[torch.Tensor]


# =============================================================================
# NPU-specific implementations for Ascend hardware
# =============================================================================
def ggml_dequantize_ascend(
    qweight: torch.Tensor,
    qweight_type: int,
    rows: int,
    cols: int,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Dequantize GGML quantized weights for NPU.

    Uses gguf library's reference implementation which supports all GGML formats
    and is guaranteed to be correct. The dequantization runs on CPU during model
    loading, then the dequantized weights are transferred to NPU for inference.
    """

    # Move to CPU for dequantization using gguf library
    qweight_cpu = qweight.cpu().numpy()

    # Use gguf library's dequantize (supports all GGML formats)
    dequant_np = gguf_dequantize(qweight_cpu, qweight_type)

    # Convert to torch and move to target device
    result = torch.from_numpy(dequant_np).to(dtype=dtype, device=qweight.device)
    result = result.reshape(rows, cols)

    return result


class GGUFLinearAscendMethod(LinearMethodBase):
    """Linear method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        self.params_dtype = params_dtype
        output_size_per_partition = sum(output_partition_sizes)

        tensor_shape = (output_size_per_partition, input_size_per_partition)
        qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
                "shard_id": [],
                "shard_id_map": {},
            },
        )
        set_weight_attrs(qweight, extra_weight_attrs)
        layer.register_parameter("qweight", qweight)

        qweight_type = Parameter(
            torch.empty(len(output_partition_sizes), dtype=torch.uint8),
            requires_grad=False,
        )
        set_weight_attrs(
            qweight_type,
            {
                "is_gguf_weight_type": True,
                "weight_type": 0,
                "shard_weight_type": {},
                "ignore_warning": True,
            },
        )
        set_weight_attrs(qweight_type, extra_weight_attrs)
        layer.register_parameter("qweight_type", qweight_type)

    def process_weights_after_loading(self, layer: torch.nn.Module):
        qweight_type = layer.qweight_type.weight_type
        if not (qweight_type in UNQUANTIZED_TYPES or qweight_type in DEQUANT_TYPES):
            raise ValueError(
                f"Unsupported GGUF quantization type {WeightType(qweight_type)} in layer."
            )
        self._create_padded_weight_param(layer)
        # Pre-dequantize weights for faster inference
        self._pre_dequantize_weights(layer)

    def _create_padded_weight_param(self, layer: torch.nn.Module):
        """Create padded weight parameter for GGUF MergedLinear layer."""
        qweight = layer.qweight
        shard_id_map = qweight.shard_id_map
        shard_id = qweight.shard_id
        if len(data_container := qweight.data_container) > 1:
            dtype = {data.dtype for data in data_container}
            assert len(dtype) == 1
            dtype = next(iter(dtype))
            padded_side = max(x.size(1) for x in data_container)
            concat_side = sum(x.size(0) for x in data_container)
            padded_data = torch.zeros(
                (concat_side, padded_side), dtype=dtype, device=qweight.device
            )
            shard_offset_map = dict[str, tuple[int, int, int]]()
            for idx in shard_id:
                id_in_container = shard_id_map[idx]
                start = sum(x.size(0) for x in data_container[:id_in_container])
                end = start + data_container[id_in_container].size(0)
                size = data_container[id_in_container].size(1)
                padded_data[start:end, :size] = data_container[id_in_container]
                shard_offset_map[idx] = (start, end, size)
            qweight.data_container.clear()
            padded_param = Parameter(padded_data, requires_grad=False)
            set_weight_attrs(padded_param, vars(qweight))
            set_weight_attrs(padded_param, {"shard_offset_map": shard_offset_map})
            layer.register_parameter("qweight", padded_param)

    def _pre_dequantize_weights(self, layer: torch.nn.Module):
        """Pre-dequantize GGML weights to FP16 for faster inference.

        This eliminates runtime dequantization overhead at the cost of more memory.
        """
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type

        if qweight_type in UNQUANTIZED_TYPES and qweight.dtype in (
            torch.float16,
            torch.bfloat16,
            torch.float32,
        ):
            layer.dequantized_weight = qweight
            return

        shard_id = getattr(qweight, "shard_id", None)
        has_shard_offset = hasattr(qweight, "shard_offset_map")

        if shard_id and has_shard_offset:
            # Handle sharded weights (QKV merged)
            shard_id = ["q", "k", "v"] if "q" in shard_id else shard_id
            dequant_shards = []
            for idx in shard_id:
                start, end, offset = qweight.shard_offset_map[idx]
                shard_qtype = layer.qweight_type.shard_weight_type[idx]
                shard_data = qweight[start:end, :offset].contiguous()

                block_size, type_size = gguf.GGML_QUANT_SIZES[shard_qtype]
                shape = (
                    shard_data.shape[0],
                    shard_data.shape[1] // type_size * block_size,
                )
                dequant = ggml_dequantize_ascend(
                    shard_data, shard_qtype, *shape, self.params_dtype
                )
                dequant_shards.append(dequant)

            dequant_weight = torch.cat(dequant_shards, dim=0)
        else:
            # Handle single weight
            block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
            shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
            dequant_weight = ggml_dequantize_ascend(
                qweight, qweight_type, *shape, self.params_dtype
            )

        layer.dequantized_weight = dequant_weight

        if hasattr(layer, "qweight"):
            del layer.qweight
        if hasattr(layer, "qweight_type"):
            del layer.qweight_type

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # Use pre-dequantized weight (always available after process_weights_after_loading)
        weight = layer.dequantized_weight
        out = x @ weight.T
        if bias is not None:
            out.add_(bias)
        return out


class GGUFMoEAscendMethod(FusedMoEMethodBase):
    """MoE method for GGUF on Ascend NPU."""

    def __init__(self, quant_config: GGUFConfig):
        self.quant_config = quant_config

    def create_weights(
        self,
        layer: torch.nn.Module,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ):
        tensor_shape = (num_experts, 2 * intermediate_size_per_partition, hidden_size)
        w13_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w13_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w13_qweight, extra_weight_attrs)
        layer.register_parameter("w13_qweight", w13_qweight)

        w13_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w13_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w13_qweight_type, extra_weight_attrs)
        layer.register_parameter("w13_qweight_type", w13_qweight_type)

        tensor_shape = (num_experts, intermediate_size_per_partition, hidden_size)
        w2_qweight = GGUFUninitializedParameter(requires_grad=False)
        set_weight_attrs(
            w2_qweight,
            {
                "input_dim": 1,
                "output_dim": 0,
                "tensor_shape": tensor_shape,
                "is_gguf_weight": True,
                "data_container": [],
            },
        )
        set_weight_attrs(w2_qweight, extra_weight_attrs)
        layer.register_parameter("w2_qweight", w2_qweight)

        w2_qweight_type = Parameter(
            torch.empty(1, dtype=torch.uint8), requires_grad=False
        )
        set_weight_attrs(
            w2_qweight_type,
            {"is_gguf_weight_type": True, "weight_type": 0, "ignore_warning": True},
        )
        set_weight_attrs(w2_qweight_type, extra_weight_attrs)
        layer.register_parameter("w2_qweight_type", w2_qweight_type)

        # Store params_dtype for pre-dequantization
        self.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module):
        """Pre-dequantize MoE weights to FP16 for faster inference."""

        if hasattr(layer, "materialize_gguf_weights"):
            layer.materialize_gguf_weights()

        # Check if weights are actually loaded (not still UninitializedParameter/empty)
        w13_qweight = layer.w13_qweight
        w13_qtype = layer.w13_qweight_type.weight_type

        # Pre-dequantize w13 weights (gate+up projections)
        if w13_qtype not in UNQUANTIZED_TYPES:
            num_experts = w13_qweight.shape[0]
            w13_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w13_qtype]

            for e in range(num_experts):
                qweight_cpu = w13_qweight[e].cpu().numpy()
                rows = w13_qweight[e].shape[0]
                cols = w13_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w13_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w13_qweight.device)
                    .reshape(rows, cols)
                    .transpose(-1, -2)
                    .contiguous()
                )
                w13_dequant_list.append(dequant)

            w13_full = torch.stack(w13_dequant_list, dim=0)

            layer.register_buffer("w13_dequant", w13_full, persistent=False)
        else:
            layer.register_buffer("w13_dequant", w13_qweight.data, persistent=False)

        # Pre-dequantize w2 weights (down projection)
        w2_qweight = layer.w2_qweight
        w2_qtype = layer.w2_qweight_type.weight_type

        if w2_qtype not in UNQUANTIZED_TYPES:
            num_experts = w2_qweight.shape[0]
            w2_dequant_list = []

            block_size, type_size = gguf.GGML_QUANT_SIZES[w2_qtype]

            for e in range(num_experts):
                qweight_cpu = w2_qweight[e].cpu().numpy()
                rows = w2_qweight[e].shape[0]
                cols = w2_qweight[e].shape[1] // type_size * block_size

                dequant_np = gguf_dequantize(qweight_cpu.flatten(), w2_qtype)
                dequant = (
                    torch.from_numpy(dequant_np)
                    .to(dtype=self.params_dtype, device=w2_qweight.device)
                    .reshape(rows, cols)
                    .transpose(-1, -2)
                    .contiguous()
                )
                w2_dequant_list.append(dequant)

            w2_full = torch.stack(w2_dequant_list, dim=0)

            layer.register_buffer("w2_dequant", w2_full, persistent=False)
        else:
            layer.register_buffer("w2_dequant", w2_qweight.data, persistent=False)

        if hasattr(layer, "w2_qweight"):
            del layer.w2_qweight
        if hasattr(layer, "w13_qweight"):
            del layer.w13_qweight

    def create_moe_runner(
        self, layer: torch.nn.Module, moe_runner_config: MoeRunnerConfig
    ):
        self.moe_runner_config = moe_runner_config

    def apply(
        self,
        layer: torch.nn.Module,
        dispatch_output: StandardDispatchOutput,
    ) -> CombineInput:
        """Apply MoE forward pass on NPU using npu_grouped_matmul for maximum performance."""
        from sglang.srt.distributed.communication_op import (
            tensor_model_parallel_all_gather,
        )
        from sglang.srt.layers.moe.token_dispatcher import StandardCombineInput

        x = dispatch_output.hidden_states
        topk_output = dispatch_output.topk_output
        topk_weights, topk_ids, _ = topk_output

        # Check if pre-dequantized weights are available
        use_pre_dequant = hasattr(layer, "w13_dequant") and hasattr(layer, "w2_dequant")

        if not use_pre_dequant:
            raise RuntimeError(
                "GGUF MoE on NPU requires pre-dequantization (FusedMoE fix). Please report if this occurs."
            )

        w13 = layer.w13_dequant
        w2 = layer.w2_dequant

        num_experts = w13.shape[0]

        tp_size = getattr(layer, "moe_tp_size", 1)

        original_dtype = x.dtype
        num_tokens = x.shape[0]
        top_k = topk_ids.shape[1]

        # Ensure correct dtypes for NPU ops
        topk_ids = topk_ids.to(torch.int32)
        topk_weights = topk_weights.to(x.dtype)

        #  MoE routing initialization - reorder tokens by expert
        row_idx_len = num_tokens * top_k
        row_idx = (
            torch.arange(0, row_idx_len, dtype=torch.int32, device=x.device)
            .view(top_k, -1)
            .permute(1, 0)
            .contiguous()
        )

        sorted_hidden_states, expanded_row_idx, expanded_expert_idx = (
            torch.ops.npu.npu_moe_init_routing(
                x, row_idx=row_idx, expert_idx=topk_ids, active_num=num_tokens
            )
        )

        # Compute tokens per expert
        expert_tokens = torch.ops.npu.npu_moe_compute_expert_tokens(
            expanded_expert_idx, num_experts
        )
        expert_tokens = expert_tokens.to(torch.int64)

        w13_gmm = w13  # No transpose needed

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[sorted_hidden_states],
            weight=[w13_gmm],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        #  Activation (SwiGLU)
        hidden_states = torch.ops.npu.npu_swiglu(hidden_states)

        # TP all-gather for intermediate dimension if needed
        if tp_size > 1:
            hidden_states = tensor_model_parallel_all_gather(hidden_states, dim=-1)

        w2_gmm = w2

        hidden_states = torch.ops.npu.npu_grouped_matmul(
            x=[hidden_states],
            weight=[w2_gmm],
            split_item=2,
            group_list_type=0,
            group_type=0,
            group_list=expert_tokens,
            output_dtype=original_dtype,
        )[0]

        # Finalize routing - reorder back and apply weights
        final_hidden_states = torch.ops.npu.npu_moe_finalize_routing(
            hidden_states,
            skip1=None,
            skip2=None,
            bias=None,
            scales=topk_weights,
            expanded_src_to_dst_row=expanded_row_idx,
            export_for_source_row=topk_ids,
        )

        if tp_size > 1:
            final_hidden_states = tensor_model_parallel_all_gather(
                final_hidden_states, dim=-1
            )

        # Ensure output matches input dtype
        final_hidden_states = final_hidden_states.to(dtype=original_dtype)

        return StandardCombineInput(hidden_states=final_hidden_states)


class GGUFEmbeddingAscendMethod(GGUFLinearAscendMethod):
    """Embedding method for GGUF on Ascend NPU."""

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        return torch.embedding(layer.dequantized_weight, x)
