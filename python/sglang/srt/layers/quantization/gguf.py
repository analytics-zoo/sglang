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

    try:
        from custom_esimd_kernels_vllm import esimd_gemv_q4_0, esimd_gemm_q4_0
    except ImportError:
        esimd_gemv_q4_0 = None
        esimd_gemm_q4_0 = None
    try:
        from custom_esimd_kernels_vllm import esimd_gemv_q8_0
    except ImportError:
        esimd_gemv_q8_0 = None
    try:
        from custom_esimd_kernels_vllm import esimd_gemv_q4_k
    except ImportError:
        esimd_gemv_q4_k = None
    try:
        from custom_esimd_kernels_vllm import esimd_gemv_q5_k, esimd_gemv_q6_k
    except ImportError:
        esimd_gemv_q5_k = None
        esimd_gemv_q6_k = None
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
    kernel) which is absent on XPU, so embedding lookups silently break and the
    model emits garbage. This method routes through apply_gguf_embedding_xpu
    (gguf-lib row dequant) instead.
    """

    def embedding(self, layer: torch.nn.Module, x: torch.Tensor) -> torch.Tensor:
        qweight = layer.qweight
        qweight_type = layer.qweight_type.weight_type
        hidden_size = qweight.tensor_shape[1]
        return apply_gguf_embedding_xpu(
            x, qweight, qweight_type, hidden_size, dtype=self.params_dtype
        )

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
        w = getattr(layer, "_xpu_lmhead_dense", None)
        if w is None:
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
        # Prefill (M>1): the per-row ESIMD GEMM degrades ~linearly in M
        # (131x slower than dense matmul at M=1024), so dequant the INT4 weight
        # to fp16 once and use a single dense matmul instead.
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
        # Prefill (M>1): dequant int8 -> fp16 once + dense matmul.
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
        # w13 row dim = gate rows (N) followed by up rows (N); split at half.
        n13 = w13.shape[1]
        half = n13 // 2
        # Per-expert repack -> resident k-quant reps. gate and up share dtype, so
        # repack each [N, K_bytes] slice via _xpu_prepare_shard.
        self.gate_reps, self.up_reps, self.down_reps = [], [], []
        for e in range(E):
            g = w13[e, :half, :].contiguous()
            u = w13[e, half:, :].contiguous()
            d = w2[e].contiguous()
            self.gate_reps.append(_xpu_prepare_shard(g, w13_type, self.params_dtype))
            self.up_reps.append(_xpu_prepare_shard(u, w13_type, self.params_dtype))
            self.down_reps.append(_xpu_prepare_shard(d, w2_type, self.params_dtype))
        layer._xpu_moe_ready = True
        # free the raw GGUF bytes
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

        if M == 1:
            # Decode: per-expert ESIMD GEMV over the top_k routed experts.
            xf = x2.to(torch.float16).contiguous()
            ids = topk_ids[0].tolist()
            ws = topk_weights[0].to(torch.float16)
            acc = torch.zeros(1, x2.shape[-1], dtype=torch.float16, device=x2.device)
            for j in range(top_k):
                e = int(ids[j])
                gate = _xpu_rep_gemv(xf, self.gate_reps[e])
                up = _xpu_rep_gemv(xf, self.up_reps[e])
                h = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.float16)
                d = _xpu_rep_gemv(h.contiguous(), self.down_reps[e])
                acc += ws[j] * d
            out[0] = acc.to(out.dtype)
            return StandardCombineInput(hidden_states=out.reshape_as(x))

        # Prefill (M>1): group tokens by expert, dequant once per expert + matmul.
        flat_ids = topk_ids.reshape(-1)
        flat_w = topk_weights.reshape(-1).to(x2.dtype)
        tok_idx = (torch.arange(M, device=x2.device).unsqueeze(1)
                   .expand(-1, top_k).reshape(-1))
        for e in torch.unique(flat_ids).tolist():
            e = int(e)
            mask = flat_ids == e
            sel = tok_idx[mask]
            sw = flat_w[mask]
            x_e = x2.index_select(0, sel).to(torch.float16)
            gw = _xpu_dequant_rep(self.gate_reps[e])       # [N, K]
            uw = _xpu_dequant_rep(self.up_reps[e])
            gate = x_e @ gw.t()
            up = x_e @ uw.t()
            h = (torch.nn.functional.silu(gate.float()) * up.float()).to(torch.float16)
            dw = _xpu_dequant_rep(self.down_reps[e])       # [hidden, N]
            d = h @ dw.t()
            out.index_add_(0, sel, (sw.unsqueeze(-1) * d).to(out.dtype))
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
