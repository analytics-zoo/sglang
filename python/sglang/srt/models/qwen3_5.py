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
from typing import Iterable, Optional, Set, Tuple, Union

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
    xpu_flag_on,
)
from sglang.srt.utils.hf_transformers_utils import get_processor, get_rope_config

logger = logging.getLogger(__name__)
_is_cuda = is_cuda()
_is_npu = is_npu()
_is_cpu = is_cpu()
_is_xpu = is_xpu()
_is_gfx95 = is_gfx95_supported()
_is_hip = is_hip()
_use_aiter = get_bool_env_var("SGLANG_USE_AITER") and _is_hip
_is_amx_available = cpu_has_amx_support()

cached_get_processor = lru_cache(get_processor)


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

        # OPT-1 (notes §11/§12): the GDN fast path (gdn_attention kernel) needs
        # qkvz/ba in a per-k-head-group INTERLEAVED layout. Instead of the
        # per-step torch.cat repack (~0.5ms/step, launch-bound at M=1 decode),
        # bake that static output-feature permutation into the GGUF rep rows at
        # load. We attach the perms here; GGUFLinearXPUMethod consumes
        # `_gguf_gdn_out_row_perm` in process_weights_after_loading and the fast
        # path then skips the repack. On non-XPU-GGUF backends the attrs are
        # simply ignored (the eager repack still runs). Enabled only when the
        # fast path is eligible (env + ratio); see _forward_xpu_fast_path.
        if _is_xpu and xpu_flag_on("GDN_BAKE_PERM", default=True):
            pq, pb = self._build_gdn_out_row_perms()
            self.in_proj_qkvz._gguf_gdn_out_row_perm = pq
            self.in_proj_ba._gguf_gdn_out_row_perm = pb
            self._gdn_out_row_perm_baked = True
        else:
            self._gdn_out_row_perm_baked = False

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
        # GGUF stores out_proj's input (value-head) columns in [ratio, num_k]
        # order; HF expects [num_k, ratio]. This is an INPUT-dim permute that
        # would break q-blocks if done on raw quantized bytes, so the GGUF XPU
        # method applies it post-dequant. Tag the layer with the perm params.
        if (
            quant_config is not None
            and getattr(quant_config, "get_name", lambda: "")() == "gguf"
            and self.num_v_heads % self.num_k_heads == 0
            and self.num_v_heads // self.num_k_heads > 1
        ):
            self.out_proj._gguf_gdn_col_perm = (
                self.num_v_heads // self.num_k_heads,  # ratio
                self.num_k_heads,
                self.head_v_dim,
            )

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

    def _build_gdn_out_row_perms(self):
        """OPT-1: derive the static output-feature permutations that
        _repack_qkvz_ba_for_gdn_attention applies, as index vectors, so they can
        be baked into the in_proj rep rows at load (eliminating the per-step cat).

        Each perm P satisfies repack(y)[:, i] == y[:, P[i]] — i.e. running the
        repack on a row whose values are their own column index yields P. Bake
        the rep rows by P so the GEMV emits the interleaved layout directly.
        Bit-exact, verified in tools/verify_gdn_repack_as_weight_perm.py."""
        hk = self.num_k_heads // self.attn_tp_size
        ratio = (self.num_v_heads // self.attn_tp_size) // hk
        dk = self.head_k_dim
        dv = self.head_v_dim
        qkvz_w = 2 * hk * dk + 2 * (hk * ratio) * dv
        ba_w = 2 * (hk * ratio)
        # Build the index vectors on CPU as integers. This runs once per GDN
        # layer inside `with torch.device("xpu")` (model init); a float64 arange
        # there would dispatch to the XPU and PTL Xe3 (LPG) has NO hardware fp64,
        # so it emulates — serializing the whole per-layer init loop into tens of
        # seconds of cold-start. The repack below is pure split/reshape/cat (no
        # arithmetic), so integer indices are exact; the consumers
        # (_xpu_perm_rep_rows / _xpu_bake_out_row_perm) `.to(w.device)` the perm
        # themselves, so building on CPU is correct and device-agnostic.
        idx_q = torch.arange(qkvz_w, dtype=torch.long, device="cpu").reshape(1, qkvz_w)
        idx_b = torch.arange(ba_w, dtype=torch.long, device="cpu").reshape(1, ba_w)
        out_q, out_b = self._repack_qkvz_ba_for_gdn_attention(idx_q, idx_b)
        perm_q = out_q.reshape(-1).long()
        perm_b = out_b.reshape(-1).long()
        # sanity: both are bijections of their width
        assert perm_q.numel() == qkvz_w and perm_b.numel() == ba_w
        assert int(perm_q.min()) == 0 and int(perm_q.max()) == qkvz_w - 1
        assert int(perm_b.min()) == 0 and int(perm_b.max()) == ba_w - 1
        assert torch.unique(perm_q).numel() == qkvz_w
        assert torch.unique(perm_b).numel() == ba_w
        return perm_q, perm_b

    def _repack_qkvz_ba_for_gdn_attention(self, mixed_qkvz, mixed_ba):
        """Convert sglang's CONTIGUOUS projection layout into the per-k-head-
        GROUP INTERLEAVED layout that torch.ops.sgl_kernel.gdn_attention's conv
        kernel expects (sgl-kernel-xpu/src/gdn_attn/causal_conv1d.hpp:100-177).

        sglang contiguous:  qkvz = [all_q | all_k | all_v | all_z]
                            ba   = [all_b | all_a]
        kernel interleaved: qkvz = [g0: q k v(ratio) z(ratio) | g1: ... ]
                            ba   = [g0: b(ratio) a(ratio) | g1: ... ]
        where ratio = num_v_heads / num_k_heads, group = per k-head.

        NOTE: contiguous != interleaved at EVERY ratio (incl. ratio 1) — verified
        in tools/verify_gdn_qkvz_repack.py (all q/k/v/z/b/a round-trip bit-exact).
        The prior "bit-identical at ratio 1" assumption was wrong; always repack.
        """
        hk = self.num_k_heads // self.attn_tp_size
        ratio = (self.num_v_heads // self.attn_tp_size) // hk
        dk = self.head_k_dim
        dv = self.head_v_dim
        T = mixed_qkvz.shape[0]

        all_q = hk * dk
        all_k = hk * dk
        all_v = hk * ratio * dv  # == num_v_heads/tp * dv
        all_z = all_v
        q, k, v, z = mixed_qkvz.split([all_q, all_k, all_v, all_z], dim=-1)
        q = q.reshape(T, hk, dk)
        k = k.reshape(T, hk, dk)
        v = v.reshape(T, hk, ratio * dv)
        z = z.reshape(T, hk, ratio * dv)
        qkvz_inter = torch.cat([q, k, v, z], dim=-1).reshape(T, -1).contiguous()

        nv_tp = self.num_v_heads // self.attn_tp_size
        b, a = mixed_ba.split([nv_tp, nv_tp], dim=-1)
        b = b.reshape(T, hk, ratio)
        a = a.reshape(T, hk, ratio)
        ba_inter = torch.cat([b, a], dim=-1).reshape(T, -1).contiguous()
        return qkvz_inter, ba_inter

    def _forward_input_proj(self, hidden_states: torch.Tensor):
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
        # DTYPE CONTRACT: gdn_attention dispatches scalar_t off mixed_qkvz /
        # core_attn_out (= projected_states_qkvz.dtype, here the run dtype) and
        # reinterpret_casts conv_state / ssm_state / A_log / dt_bias to scalar_t*
        # (causal_conv1d.hpp:610-619, gated_delta_rule.hpp:362-365,397-406). The
        # MambaPool defaults are conv=bf16 (SGLANG_MAMBA_CONV_DTYPE) and ssm=fp32
        # (SGLANG_MAMBA_SSM_DTYPE=None); A_log/dt_bias are fp32. Any tensor whose
        # native dtype != kdtype is read as garbage bytes (the !!!! bug) — so
        # cast every kernel-consumed state/param to kdtype here, and scatter
        # results back in each pool's NATIVE dtype.
        kdtype = projected_states_qkvz.dtype
        # OPT-2 (notes §11/§14): A_log / dt_bias are CONSTANT fp32 params, re-cast
        # to kdtype every decode step every GDN layer (2 launches/layer/step). Cache
        # the kdtype copy once (keyed by dtype, like _cos_sin_cache_fp16) — the cast
        # is dtype-deterministic so this is bit-exact vs the per-step .to(). Gated on
        # the SAME flag as OPT-1 (_gdn_out_row_perm_baked) so SGLANG_XPU_GDN_BAKE_PERM
        # toggles the OPT-1+OPT-2 bundle together (clean single-variable A/B).
        if getattr(self, "_gdn_out_row_perm_baked", False):
            if getattr(self, "_a_log_kdtype_cache", (None, None))[0] != kdtype:
                self._a_log_kdtype_cache = (
                    kdtype,
                    self.A_log.to(kdtype).contiguous(),
                    self.dt_bias.to(kdtype).contiguous(),
                )
            _, a_log_k, dt_bias_k = self._a_log_kdtype_cache
        else:
            a_log_k = self.A_log.to(kdtype)
            dt_bias_k = self.dt_bias.to(kdtype)
        pool_conv = mamba_cache_params.conv[0]  # (cache, conv_dim, W-1)
        pool_ssm = mamba_cache_params.temporal  # (cache, Hv, head_v, head_k)
        is_decode = forward_batch.forward_mode.is_decode()
        # Stage B (§15): for pure DECODE the kernel (NATIVE_LAUNCHER causal_conv1d)
        # indexes conv_state IN-PLACE by cache_indices with tensor-derived inner
        # strides, so we pass the WHOLE pool as a transposed VIEW (free, no copy)
        # + the real cache_indices — eliminating the per-step gather + transpose +
        # contiguous + scatter (~1.5ms scatter + casts, the +4.88ms adapter). The
        # conv pool dtype == run dtype (SGLANG_MAMBA_CONV_DTYPE tracks --dtype), so
        # the in-place write is type-safe (assert guards it). PREFILL still goes
        # through the XE2 chunk kernel which needs a contiguous [W-1,conv_dim]
        # scratch, so keep the gather/scatter adapter there (cheap: amortized over
        # many tokens, §11 prefill cat ~0.22ms).
        # Stage B in-place: for pure DECODE pass the WHOLE pools (conv as a free
        # transposed view, ssm as-is fp32) + the real cache_indices, so the kernel
        # reads/writes pool[cache_indices] in-place. Eliminates the entire adapter
        # (gather + transpose + cast + scatter, ~4.88ms, §15). The conv pool dtype
        # must == run dtype (it does: SGLANG_MAMBA_CONV_DTYPE tracks --dtype); the
        # gated_delta_rule kernel now reads the fp32 ssm pool directly (no cast).
        use_inplace = (
            is_decode
            and getattr(self, "_gdn_inplace", True)
            and pool_conv.dtype == kdtype
            and pool_ssm.dtype == torch.float32
        )
        if use_inplace:
            scratch_conv = pool_conv.transpose(-1, -2)  # (cache, W-1, conv_dim) view
            scratch_ssm = pool_ssm                      # (cache, Hv, hv, hk) fp32, in-place
            kernel_state_indices = cache_indices        # real slots; kernel writes in-place
        else:
            scratch_conv = (
                pool_conv.index_select(0, cache_indices)
                .transpose(-1, -2)
                .to(kdtype)
                .contiguous()
            )  # (bs, W-1, conv_dim) — kernel layout + kernel dtype
            # PREFILL (non-inplace) goes through the XE2 chunk_gated_delta_rule
            # kernel which reads ssm as kdtype (scalar_t) — NOT the fp32-decoupled
            # gated_delta_rule. So gather + cast to kdtype scratch (dense 0..bs-1),
            # scattered back to the fp32 pool below via .to(pool.dtype). Decode
            # (in-place) uses the fp32 gated_delta_rule; pool stays fp32 throughout.
            scratch_ssm = (
                pool_ssm.index_select(0, cache_indices).to(kdtype).contiguous()
            )
            kernel_state_indices = None  # set to scratch arange below

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

        # Repack sglang's CONTIGUOUS [all_q|all_k|all_v|all_z] / [all_b|all_a]
        # into the per-k-head-group INTERLEAVED layout the kernel de-interleaves
        # (causal_conv1d.hpp). Required at every ratio — verified bit-exact in
        # tools/verify_gdn_qkvz_repack.py. Also makes them contiguous.
        # OPT-1 (notes §11/§12): when the perm is baked into the GGUF rep rows at
        # load (_gdn_out_row_perm_baked), the GEMV already emits the interleaved
        # layout, so skip the per-step cat (~0.5ms/step, launch-bound at M=1).
        # The projection output is still contiguous (GEMV writes a fresh tensor).
        if not getattr(self, "_gdn_out_row_perm_baked", False):
            projected_states_qkvz, projected_states_ba = (
                self._repack_qkvz_ba_for_gdn_attention(
                    projected_states_qkvz, projected_states_ba
                )
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
        # using the real cache_indices afterwards. For the in-place decode path
        # the kernel uses the REAL cache_indices and writes the pools directly.
        if kernel_state_indices is None:
            kernel_state_indices = torch.arange(
                batch_size, device=cache_indices.device, dtype=cache_indices.dtype
            )

        is_verify = forward_batch.forward_mode.is_target_verify()
        if is_verify:
            # MTP target-verify GDN on XPU. The triton GDN path (conv +
            # fused_sigmoid recurrence) is numerically BROKEN on triton-XPU, so
            # run the verify conv+recurrence in ONE all-SYCL eagle_ops kernel
            # (gdn_fused_verify) per layer — instead of replaying gdn_attention
            # per draft token (~4 launches/layer + ~40 glue ops/layer, launch-
            # bound on PTL). The fused kernel takes the RAW interleaved
            # projection directly, does gdn_attention's internal conv channel
            # reorder + width-W causal conv + the gated_delta_rule recurrence,
            # and emits the per-step (post-each-draft-token) ssm + conv-window
            # intermediates the mamba rollback needs. NO commit to the main pool
            # (disable_state_update) — update_mamba_state_after_mtp_verify
            # scatters the accepted step's intermediate back. Layout + numerics
            # validated vs gdn_attention in tools/gdn_fused_sycl_check.py
            # (out rel 6.8e-4, inter_ssm rel 3.9e-4). projected_states_qkvz/_ba
            # are already in the interleaved layout the kernel expects (repacked
            # above, or GEMV-baked). A_log/dt_bias MUST be fp32 (self.*), not the
            # kdtype copies.
            if not getattr(self, "_gdn_fused_loaded", False):
                import custom_esimd_kernels_sglang  # noqa: F401 — registers eagle_ops
                self._gdn_fused_loaded = True
            draft_token_num = forward_batch.spec_info.draft_token_num
            inter_ssm = mamba_cache_params.intermediate_ssm  # [slots, draft, Hv, V, K]
            inter_conv = mamba_cache_params.intermediate_conv_window[0]  # [slots,draft,dim,W-1]
            verify_state_idx = linear_backend.verify_intermediate_state_indices[
                :batch_size
            ]
            conv_w_view = self.conv1d.weight.view(
                self.conv1d.weight.size(0), self.conv1d.weight.size(2)
            )
            act_i = 1 if self.activation in ("silu", "swish") else 0
            torch.ops.eagle_ops.gdn_fused_verify(
                core_attn_out,
                z,                                      # gate out: reordered z-block
                projected_states_qkvz.contiguous(),
                projected_states_ba.contiguous(),
                conv_w_view.contiguous(),
                self.conv1d.bias.contiguous() if self.conv1d.bias is not None else None,
                pool_conv,                              # [slots, conv_dim, W-1] pool
                self.A_log.float().contiguous(),
                self.dt_bias.float().contiguous(),
                pool_ssm,                               # [slots, Hv, V, K] fp32 pool
                inter_ssm,
                inter_conv,
                query_start_loc.to(torch.int32).contiguous(),
                cache_indices.to(torch.int32).contiguous(),
                verify_state_idx.to(torch.int32).contiguous(),
                self.num_k_heads,
                self.num_v_heads,
                self.head_k_dim,
                self.head_v_dim,
                int(act_i),
                int(draft_token_num),
            )
            # disable_state_update: main ssm/conv pools left untouched; the
            # mamba scatter commits the accepted step from inter_ssm/inter_conv.
        else:
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
                a_log_k,
                dt_bias_k,
                num_prefills,
                num_decodes,
                has_initial_state,
                query_start_loc,
                kernel_state_indices,
                num_actual_tokens,
                self.attn_tp_size,
            )

            # Scatter kernel writeback into the real MambaPool slots. SKIPPED for
            # the in-place decode path (kernel already wrote pool[cache_indices]).
            if not use_inplace:
                #   conv: (bs, W-1, conv_dim) → (cache, conv_dim, W-1)
                #   ssm:  (bs, Hv, head_v, head_k) — already matches pool layout
                cache_indices_long = cache_indices.to(torch.long)
                pool_conv.index_copy_(
                    0,
                    cache_indices_long,
                    scratch_conv.transpose(-1, -2).to(pool_conv.dtype).contiguous(),
                )
                pool_ssm.index_copy_(
                    0, cache_indices_long, scratch_ssm.to(pool_ssm.dtype)
                )

        # Post: RMSNormGated(core_attn_out, z) then out_proj. Mirrors the
        # default path lines 504-519 below.
        z_shape_og = z.shape
        core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
        z = z.reshape(-1, z.shape[-1])
        if core_attn_out.shape != z.shape:
            core_attn_out_pad = torch.zeros_like(z)
            core_attn_out_pad[: core_attn_out.shape[0], :] = core_attn_out
            core_attn_out = core_attn_out_pad
        _gn = self.norm(core_attn_out, z)
        core_attn_out = _gn
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

        # --- XPU native conv1d+GDN fast path (sgl_kernel.gdn_attention) ---
        # _forward_xpu_fast_path now handles the MambaPool conv_state layout
        # mismatch with a gather/scatter adapter around the kernel call.
        # Gate with an env var so we can A/B against the PyTorch fallback
        # (chunk_torch_xpu.py) per-run without a code change. Keep off by
        # default until end-to-end correctness has been confirmed on Qwen3.5.
        # XPU-default-on (the PTL GGUF prod config; the use-site below also
        # gates on _is_xpu + ratio). Set SGLANG_XPU_GDN_FAST_PATH=0 to disable.
        _ENABLE_XPU_FAST_PATH = xpu_flag_on("GDN_FAST_PATH", default=True)
        # Repack adapter (_repack_qkvz_ba_for_gdn_attention) handles any
        # num_v_heads/num_k_heads ratio (verified bit-exact for ratio 1 & 2),
        # so the fast path is eligible whenever the kernel's ratio constraint
        # (num_v_heads % num_k_heads == 0) holds.
        # MTP target-verify now routes through the SYCL fast path (per-token
        # gdn_attention replay + per-step intermediate-state capture, see
        # _forward_xpu_fast_path's is_verify branch). The triton GDN path is
        # numerically broken on triton-XPU, so this is the correct route and is
        # ON BY DEFAULT. SGL_XPU_VERIFY_FASTPATH=0 forces the legacy triton
        # verify path (GDNAttnBackend.target_verify) for A/B comparison.
        _allow_verify_fast = xpu_flag_on("VERIFY_FASTPATH", default=True)
        if (
            _ENABLE_XPU_FAST_PATH
            and _is_xpu
            and (_allow_verify_fast or not forward_batch.forward_mode.is_target_verify())
            and self.num_v_heads % self.num_k_heads == 0
        ):
            output = self._forward_xpu_fast_path(
                projected_states_qkvz,
                projected_states_ba,
                forward_batch,
            )
            if output is not None:
                return output

        if self.num_v_heads // self.num_k_heads in [1, 2, 4] and not _is_cpu:
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

        _gn = self.norm(core_attn_out, z)
        core_attn_out = _gn
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
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix.replace(".linear_attn", "")),
                is_nextn=is_nextn,
                support_shared_expert_fusion=True,
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

    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        **kwargs,
    ):
        forward_batch = kwargs.get("forward_batch", None)

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

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )

        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
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
        self.is_nextn = is_nextn  # used by the MTP NaN-trace debug probe

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
                alt_stream=alt_stream,
                prefix=add_prefix("mlp", prefix.replace(".self_attn", "")),
                is_nextn=is_nextn,
                support_shared_expert_fusion=True,
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
        if self.alt_stream is not None and get_is_capture_mode():
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

    def self_attention(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        """Full attention forward pass."""
        qkv, _ = self.qkv_proj(hidden_states)

        # vllm parity: fuse split + qk_norm + rope into single ESIMD call.
        # Hard-coded requirements: head_dim=256, fp16, GemmaRMSNorm weight+1.0
        # (matches Qwen3.5's q_norm/k_norm). XPU-default-on (the PTL GGUF prod
        # config); shape + ImportError checks below keep it safe. Set
        # SGLANG_XPU_FA_ESIMD_QKV=0 to disable. (legacy SGL_XPU_* honored.)
        if (
            _is_xpu
            and xpu_flag_on("FA_ESIMD_QKV", default=True)
            and self.head_dim == 256
            and hidden_states.dim() == 2
        ):
            try:
                from custom_esimd_kernels_sglang import esimd_qkv_split_norm_rope
            except ImportError:
                esimd_qkv_split_norm_rope = None
            if esimd_qkv_split_norm_rope is not None:
                nTokens = qkv.shape[0]
                orig_dtype = qkv.dtype

                # Persistent scratch buffers for XPUGraph capture stability.
                # Without these, every forward pass allocates fresh temporaries
                # and the captured graph references stale data_ptrs on replay,
                # yielding garbage output. Scratch grows if a larger nTokens
                # is seen; small replays slice into the prefix.
                scratch = getattr(self, "_esimd_qkv_scratch", None)
                need_resize = (
                    scratch is None
                    or scratch["nTokens"] < nTokens
                )
                if need_resize:
                    scratch = {
                        "nTokens": nTokens,
                        "qkv_in_dim": qkv.shape[1],
                        "qkv_fp16": torch.empty(
                            (nTokens, qkv.shape[1]),
                            device=qkv.device, dtype=torch.float16,
                        ),
                        "q_out": torch.empty(
                            (nTokens, self.num_heads * 256),
                            device=qkv.device, dtype=torch.float16,
                        ),
                        "gate_out": (
                            torch.empty(
                                (nTokens, self.num_heads * 256),
                                device=qkv.device, dtype=torch.float16,
                            )
                            if self.attn_output_gate
                            else torch.empty(
                                0, device=qkv.device, dtype=torch.float16,
                            )
                        ),
                        "k_out": torch.empty(
                            (nTokens, self.num_kv_heads * 256),
                            device=qkv.device, dtype=torch.float16,
                        ),
                        "v_out": torch.empty(
                            (nTokens, self.num_kv_heads * 256),
                            device=qkv.device, dtype=torch.float16,
                        ),
                        "q_norm_w_fp16": self.q_norm.weight.to(torch.float16).contiguous(),
                        "k_norm_w_fp16": self.k_norm.weight.to(torch.float16).contiguous(),
                    }
                    # Pre-convert cos_sin_cache to fp16 once (attribute so
                    # it's stable; rotary_emb is shared across layers).
                    cs = self.rotary_emb.cos_sin_cache
                    if cs.dtype != torch.float16:
                        if not hasattr(self.rotary_emb, "_cos_sin_cache_fp16"):
                            self.rotary_emb._cos_sin_cache_fp16 = cs.to(
                                torch.float16
                            ).contiguous()
                        scratch["cs_fp16"] = self.rotary_emb._cos_sin_cache_fp16
                    else:
                        scratch["cs_fp16"] = cs
                    self._esimd_qkv_scratch = scratch

                # Fill persistent buffers in place so the captured graph
                # picks up new values on each replay.
                qkv_fp16 = scratch["qkv_fp16"][:nTokens]
                qkv_fp16.copy_(qkv.to(torch.float16))
                q_out = scratch["q_out"][:nTokens]
                gate_out = scratch["gate_out"][:nTokens] if self.attn_output_gate else scratch["gate_out"]
                k_out = scratch["k_out"][:nTokens]
                v_out = scratch["v_out"][:nTokens]
                # positions shape can be [nTokens] or [3, nTokens] (mrope).
                # Keep the fresh cast here — it's only a few bytes and the
                # int32 cast doesn't hurt graph stability (allocation happens
                # in the graph pool and stays stable across replays because
                # allocation order is deterministic).
                pos_i32 = positions.to(torch.int32).contiguous()

                rotary_dim_arg = int(
                    self.head_dim
                    * getattr(self.config, "partial_rotary_factor", 1.0)
                )
                esimd_qkv_split_norm_rope(
                    qkv_fp16,
                    q_out, gate_out, k_out, v_out,
                    scratch["q_norm_w_fp16"],
                    scratch["k_norm_w_fp16"],
                    pos_i32,
                    self.num_heads, self.num_kv_heads,
                    self.attn_output_gate,
                    rotary_dim_arg, scratch["cs_fp16"],
                )
                q = q_out.to(orig_dtype)
                k = k_out.to(orig_dtype)
                v = v_out.to(orig_dtype)
                # ESIMD kernel already applies sigmoid to gate_out (see
                # qkv_split_norm_rope.h:159), so don't re-sigmoid here.
                gate = gate_out.to(orig_dtype) if self.attn_output_gate else None
                attn_output = self.attn(q, k, v, forward_batch)
                if self.attn_output_gate:
                    attn_output = attn_output * gate
                output, _ = self.o_proj(attn_output)
                return output

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

        q, k = self._apply_qk_norm(q, k)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, forward_batch)

        if self.attn_output_gate:
            gate = torch.sigmoid(gate)
            attn_output = attn_output * gate

        output, _ = self.o_proj(attn_output)
        return output

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: Optional[torch.Tensor],
        forward_batch: ForwardBatch,
        captured_last_layer_outputs: Optional[list[torch.Tensor]] = None,
        **kwargs,
    ):
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

        # Fully Connected
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states, residual, forward_batch
        )
        use_reduce_scatter = self.layer_communicator.should_use_reduce_scatter(
            forward_batch
        )

        should_allreduce_fusion = (
            self.layer_communicator.should_fuse_mlp_allreduce_with_next_layer(
                forward_batch
            )
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

    if _is_gfx95 or _is_npu:
        packed_modules_mapping = {
            "qkv_proj": ["q_proj", "k_proj", "v_proj"],
            "gate_up_proj": ["gate_proj", "up_proj"],
            "in_proj_qkvz": ["in_proj_qkv", "in_proj_z"],
            "in_proj_ba": ["in_proj_b", "in_proj_a"],
        }

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

        alt_stream = torch.cuda.Stream() if _is_cuda else None

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
    if _is_gfx95 or _is_npu:
        packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
        hf_to_sglang_mapper = None

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

    def _share_dequant(self, layer):
        """Return a dense fp16 [rows, hidden] weight for sharing target->draft
        (EAGLE/NEXTN). Under GGUF on XPU the embed/head are quantized and have NO
        plain `.weight`: the XPU embed method (GGUFEmbeddingXPUMethod) deletes
        `.qweight` post-load and keeps `layer._xpu_emb_rep`; a quantized linear
        head keeps `.qweight`+`.qweight_type`. Dequantize the FULL matrix. (GAP-B)"""
        w = getattr(layer, "weight", None)
        if w is not None:
            return w
        if hasattr(layer, "_xpu_emb_rep"):
            qm = layer.quant_method
            rep = layer._xpu_emb_rep
            rows = rep[1].shape[0]
            ids = torch.arange(rows, device=rep[1].device, dtype=torch.long)
            return qm.embedding(layer, ids)
        from sglang.srt.layers.quantization.gguf import _xpu_dequant_to_fp16
        if hasattr(layer, "qweight"):
            qtype = layer.qweight_type.weight_type
            return _xpu_dequant_to_fp16(layer.qweight, qtype, torch.float16)
        raise AttributeError(
            f"_share_dequant: {type(layer).__name__} has no weight/_xpu_emb_rep/qweight"
        )

    def _share_head_quant(self, layer):
        """Share the QUANTIZED lm_head (not a dense-dequant fp16 copy) so the
        draft's _compute_lm_head takes the GGUF branch (quant_method.apply ->
        Q6_K M-GEMV, the #86 kernel) instead of a dense [hidden, vocab] aten::mm
        (~14.7ms/draft-fwd, the #89 hog). Returns a dict of the prepared XPU
        quant reps (the SAME tensor objects -> bit-identical to the target,
        EAGLE-safe) + the quant_method, or None if the head isn't an XPU-GGUF
        quant layer (caller then falls back to the dense dequant share).
        SGLANG_XPU_SHARE_QUANT_HEAD=0 reverts to the dequant share."""
        if not xpu_flag_on("SHARE_QUANT_HEAD", default=True):
            return None
        # The lm_head is a VocabParallelEmbedding/ParallelLMHead loaded via
        # GGUFEmbeddingXPUMethod, whose apply() reads `_xpu_emb_rep` (the packed
        # Q6_K rep) and runs _xpu_rep_gemv (M=1) / esimd_gemv_q6_k_m (M>1). Share
        # that rep + the quant_method instance (SAME objects -> bit-identical).
        rep = getattr(layer, "_xpu_emb_rep", None)
        qm = getattr(layer, "quant_method", None)
        if rep is None or qm is None:
            return None
        return {
            "kind": "xpu_quant_head",
            "_xpu_emb_rep": rep,
            "quant_method": qm,
        }

    def get_embed_and_head(self):
        embed = self._share_dequant(self.model.embed_tokens) if self.pp_group.is_first_rank else None
        if self.pp_group.is_last_rank:
            head = self._share_head_quant(self.lm_head)
            if head is None:
                head = self._share_dequant(self.lm_head)
        else:
            head = None
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
        row permutation only. ``out_proj`` needs an INPUT-dim (column) permute
        that would break q-blocks, so it is handled post-dequant in the XPU
        method (see GGUFLinearXPUMethod), not here. The key-head q/k slices of
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

        # GGUF stores Gemma-style RMSNorm weights in the standard (≈1.0)
        # convention, but Qwen3.5 uses GemmaRMSNorm which computes x*(1+w) and
        # therefore expects the checkpoint weight to be (standard-1). HF
        # safetensors already store (standard-1); a GGUF checkpoint does not, so
        # subtract 1 from every GemmaRMSNorm weight on the GGUF load path. The
        # GDN linear_attn.norm uses plain RMSNormGated (no offset) — exclude it.
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

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue

                if "visual" in name or "mlp.experts" in name:
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

                # GGUF stores the depthwise conv1d weight 2D [channels, kernel],
                # but the model param (and mamba_v2_sharded_weight_loader) expect
                # 3D [channels, 1, kernel] like the safetensors checkpoint. Insert
                # the singleton middle dim for the GGUF case; no-op when already 3D.
                if (
                    "conv1d.weight" in name
                    and loaded_weight.dim() == 2
                    and getattr(param, "dim", lambda: 0)() == 3
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

    if _is_gfx95 or _is_npu:
        packed_modules_mapping = Qwen3_5ForCausalLM.packed_modules_mapping
        hf_to_sglang_mapper = None

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
        if _use_aiter:
            self.num_fused_shared_experts = self._get_num_fused_shared_experts()

        self.enable_shared_expert_fusion = self.num_fused_shared_experts > 0

    def _get_num_fused_shared_experts(self):
        if not (
            hasattr(self.model, "layers")
            and len(self.model.layers) > 0
            and hasattr(self.model.layers[0].mlp, "num_fused_shared_experts")
        ):
            return 0
        return self.model.layers[0].mlp.num_fused_shared_experts

    def _share_dequant(self, layer):
        """Return a dense fp16 [rows, hidden] weight for sharing target->draft
        (EAGLE/NEXTN). Under GGUF on XPU the embed/head are quantized and have NO
        plain `.weight`: the XPU embed method (GGUFEmbeddingXPUMethod) deletes
        `.qweight` post-load and keeps `layer._xpu_emb_rep`; a quantized linear
        head keeps `.qweight`+`.qweight_type`. Dequantize the FULL matrix. (GAP-B)"""
        w = getattr(layer, "weight", None)
        if w is not None:
            return w
        if hasattr(layer, "_xpu_emb_rep"):
            qm = layer.quant_method
            rep = layer._xpu_emb_rep
            rows = rep[1].shape[0]
            ids = torch.arange(rows, device=rep[1].device, dtype=torch.long)
            return qm.embedding(layer, ids)
        from sglang.srt.layers.quantization.gguf import _xpu_dequant_to_fp16
        if hasattr(layer, "qweight"):
            qtype = layer.qweight_type.weight_type
            return _xpu_dequant_to_fp16(layer.qweight, qtype, torch.float16)
        raise AttributeError(
            f"_share_dequant: {type(layer).__name__} has no weight/_xpu_emb_rep/qweight"
        )

    def _share_head_quant(self, layer):
        """Share the QUANTIZED lm_head (not a dense-dequant fp16 copy) so the
        draft's _compute_lm_head takes the GGUF branch (quant_method.apply ->
        Q6_K M-GEMV, the #86 kernel) instead of a dense [hidden, vocab] aten::mm
        (~14.7ms/draft-fwd, the #89 hog). Returns a dict of the prepared XPU
        quant reps (the SAME tensor objects -> bit-identical to the target,
        EAGLE-safe) + the quant_method, or None if the head isn't an XPU-GGUF
        quant layer (caller then falls back to the dense dequant share).
        SGLANG_XPU_SHARE_QUANT_HEAD=0 reverts to the dequant share."""
        if not xpu_flag_on("SHARE_QUANT_HEAD", default=True):
            return None
        # The lm_head is a VocabParallelEmbedding/ParallelLMHead loaded via
        # GGUFEmbeddingXPUMethod, whose apply() reads `_xpu_emb_rep` (the packed
        # Q6_K rep) and runs _xpu_rep_gemv (M=1) / esimd_gemv_q6_k_m (M>1). Share
        # that rep + the quant_method instance (SAME objects -> bit-identical).
        rep = getattr(layer, "_xpu_emb_rep", None)
        qm = getattr(layer, "quant_method", None)
        if rep is None or qm is None:
            return None
        return {
            "kind": "xpu_quant_head",
            "_xpu_emb_rep": rep,
            "quant_method": qm,
        }

    def get_embed_and_head(self):
        embed = self._share_dequant(self.model.embed_tokens) if self.pp_group.is_first_rank else None
        if self.pp_group.is_last_rank:
            head = self._share_head_quant(self.lm_head)
            if head is None:
                head = self._share_dequant(self.lm_head)
        else:
            head = None
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

        # --- GGUF load-path adaptations (mirror the dense Qwen3_5 load_weights) ---
        # A GGUF checkpoint stores weights in llama.cpp conventions that differ
        # from the HF safetensors the model code expects:
        #   * GemmaRMSNorm weights are stored standard (~1.0), but GemmaRMSNorm
        #     computes x*(1+w) so the param must be (standard-1) -> subtract 1.
        #   * GDN linear_attn.* needs the value-head permute / A_log / dt_bias
        #     transform (_gguf_gdn_transform), same as the dense path.
        #   * shared_expert_gate is stored 1-D [hidden] but the param is
        #     [1, hidden]; conv1d is stored 2-D but the param is 3-D.
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
                # GGUF fused GDN projections (in_proj_qkvz / in_proj_ba) register a
                # quantized `.qweight` param. When the source shard is F32 (the 35B
                # stores ssm_alpha/beta = in_proj_a/b as F32), the gguf weight iterator
                # yields it under `.weight` (no quant rename). Redirect to `.qweight`
                # so the shard lands in the fused param; qweight_type defaults to 0
                # (F32) -> handled as the fp16 rep.
                if (
                    name.endswith(".weight")
                    and name not in params_dict
                    and name[: -len(".weight")] + ".qweight" in params_dict
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


# The MoE GGUF load path reuses the dense model's GDN value-head transform
# (_gguf_gdn_transform / _perm_value_rows). Both reference only self.config, so
# bind the dense implementations onto the MoE class rather than duplicating them.
Qwen3_5MoeForConditionalGeneration._gguf_gdn_transform = (
    Qwen3_5ForConditionalGeneration._gguf_gdn_transform
)
Qwen3_5MoeForConditionalGeneration._perm_value_rows = staticmethod(
    Qwen3_5ForConditionalGeneration._perm_value_rows
)


EntryClass = [Qwen3_5MoeForConditionalGeneration, Qwen3_5ForConditionalGeneration]
