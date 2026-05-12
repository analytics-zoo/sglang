from __future__ import annotations

import os
from typing import TYPE_CHECKING, Optional

import torch
import torch.nn.functional as F

from sglang.srt.configs.model_config import AttentionArch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.attention.flashattention_backend import (
    FlashAttentionMetadata,
    make_local_attention_virtual_batches,
    merge_state_v2_wrapper,
    prepare_swa_spec_page_table_triton,
)
from sglang.srt.managers.schedule_batch import get_global_server_args
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.model_executor.model_runner import ModelRunner

from sgl_kernel import merge_state_v2
from sgl_kernel.flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache


class XPUAttentionBackend(AttentionBackend):
    """XPU FlashAttention backend, currently based on FlashAttentionBackend, will be refactored later.

    TODO:
    - Prefill and Decode disaggregation, currently only chunked prefill is supported
    - Speculative Decoding support
    - XPU Graph support, see https://github.com/pytorch/pytorch/issues/162143
    - MLA support
    """

    def __init__(
        self,
        model_runner: ModelRunner,
        skip_prefill: bool = False,
        speculative_step_id=0,
        topk=0,
        speculative_num_steps=0,
    ):
        super().__init__()

        assert not (
            model_runner.sliding_window_size is not None
            and model_runner.model_config.is_encoder_decoder
        ), "Sliding window and cross attention are not supported together"

        self.forward_metadata: FlashAttentionMetadata = None
        # extra metadata for handling speculative decoding topk > 1, extended draft decode and verify
        self.forward_metadata_spec_decode_expand: FlashAttentionMetadata = None
        self.max_context_len = model_runner.model_config.context_len
        self.device = model_runner.device
        self.decode_cuda_graph_metadata = {}
        self.target_verify_metadata = {}
        self.req_to_token = model_runner.req_to_token_pool.req_to_token
        self.kv_cache_dtype = model_runner.kv_cache_dtype
        self.kv_cache_dtype_str = model_runner.server_args.kv_cache_dtype
        self.page_size = model_runner.page_size
        self.use_mla = model_runner.model_config.attention_arch == AttentionArch.MLA
        assert (
            self.use_mla is False
        ), "XPUAttentionBackend doesn't support MLA yet, please use --attention-backend triton instead."
        self.skip_prefill = skip_prefill
        self.is_hybrid_swa = model_runner.is_hybrid_swa
        if self.is_hybrid_swa:
            self.full_to_swa_index_mapping = (
                model_runner.token_to_kv_pool.full_to_swa_index_mapping
            )
        self.topk = model_runner.server_args.speculative_eagle_topk or 0
        self.speculative_num_steps = speculative_num_steps
        self.speculative_num_draft_tokens = (
            model_runner.server_args.speculative_num_draft_tokens
        )
        self.speculative_step_id = speculative_step_id

        # Local attention settings
        self.attention_chunk_size = (
            model_runner.attention_chunk_size
            if hasattr(model_runner, "attention_chunk_size")
            else None
        )

        # For each layer, the sliding_window_size can be different. This is only used for preparing SWA metadata.
        # We use `layer.sliding_window_size` to decide whether to use SWA for each layer.
        self.sliding_window_size = model_runner.sliding_window_size
        self.has_swa = (
            self.sliding_window_size is not None and self.sliding_window_size > -1
        )

    def init_forward_metadata(self, forward_batch: ForwardBatch):
        """Initialize forward metadata hence all layers in the forward pass can reuse it."""
        metadata = FlashAttentionMetadata()
        seqlens_in_batch = forward_batch.seq_lens
        batch_size = forward_batch.batch_size
        device = seqlens_in_batch.device

        if forward_batch.forward_mode.is_decode_or_idle():
            # Draft Decode
            if forward_batch.spec_info is not None:
                assert (
                    False
                ), "XPUAttentionBackend doesn't support speculative decoding yet, please use --attention-backend triton instead."
                if self.topk <= 1:
                    metadata.cache_seqlens_int32 = (
                        seqlens_in_batch + (self.speculative_step_id + 1)
                    ).to(torch.int32)
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item() + (
                        self.speculative_step_id + 1
                    )
                    metadata.cu_seqlens_q = torch.arange(
                        0, batch_size + 1, dtype=torch.int32, device=device
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]
                else:
                    metadata.cache_seqlens_int32 = (seqlens_in_batch).to(torch.int32)
                    metadata.max_seq_len_q = self.topk
                    metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                    metadata.cu_seqlens_q = torch.arange(
                        0,
                        batch_size * self.topk + 1,
                        step=self.topk,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata.cu_seqlens_k = torch.nn.functional.pad(
                        torch.cumsum(
                            metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                        ),
                        (1, 0),
                    )
                    metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, : metadata.max_seq_len_k
                    ]

                    metadata_expand = FlashAttentionMetadata()
                    decode_length = self.speculative_step_id + 1
                    metadata_expand.cache_seqlens_int32 = torch.full(
                        (seqlens_in_batch.numel() * self.topk,),
                        decode_length,
                        device=device,
                        dtype=torch.int32,
                    )
                    metadata_expand.max_seq_len_q = 1
                    metadata_expand.cu_seqlens_q = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() + 1,
                        dtype=torch.int32,
                        device=device,
                    )
                    metadata_expand.cu_seqlens_k = torch.arange(
                        0,
                        metadata_expand.cache_seqlens_int32.numel() * decode_length + 1,
                        step=decode_length,
                        dtype=torch.int32,
                        device=device,
                    )
                    # shape: [bs, num_steps, topk] -> [bs x topk, num_steps]
                    cache_loc = forward_batch.out_cache_loc.view(
                        -1, self.speculative_num_steps
                    )
                    metadata_expand.page_table = (
                        cache_loc[:, :decode_length].contiguous().to(torch.int32)
                    )
                    self.forward_metadata_spec_decode_expand = metadata_expand
            else:
                # Normal Decode
                metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0, batch_size + 1, dtype=torch.int32, device=device
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]
            # TODO: we need to test this part for llama 4 eagle case
            self._init_local_attn_metadata(forward_batch, metadata, device)
        elif forward_batch.forward_mode.is_target_verify():
            if self.topk <= 1:
                metadata.cache_seqlens_int32 = (
                    forward_batch.seq_lens + self.speculative_num_draft_tokens
                ).to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = (
                    forward_batch.seq_lens_cpu.max().item()
                    + self.speculative_num_draft_tokens
                )
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                self._init_local_attn_metadata(forward_batch, metadata, device)
            else:
                metadata.cache_seqlens_int32 = forward_batch.seq_lens.to(torch.int32)
                metadata.max_seq_len_q = self.speculative_num_draft_tokens
                metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
                metadata.cu_seqlens_q = torch.arange(
                    0,
                    batch_size * self.speculative_num_draft_tokens + 1,
                    step=self.speculative_num_draft_tokens,
                    dtype=torch.int32,
                    device=device,
                )
                metadata.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                    forward_batch.req_pool_indices, : metadata.max_seq_len_k
                ]

                metadata_expand = FlashAttentionMetadata()

                metadata_expand.max_seq_len_q = 1
                metadata_expand.cu_seqlens_q = torch.arange(
                    0,
                    forward_batch.seq_lens.numel() * self.speculative_num_draft_tokens
                    + 1,
                    dtype=torch.int32,
                    device=device,
                )

                # create expand page table
                offsets = torch.arange(
                    self.speculative_num_draft_tokens, device=device
                ).unsqueeze(
                    0
                )  # shape: (1, self.speculative_num_draft_tokens)
                cols = offsets.expand(
                    forward_batch.seq_lens.numel(), -1
                ) + forward_batch.seq_lens.unsqueeze(1)
                cum_len = torch.nn.functional.pad(
                    torch.cumsum(
                        (
                            forward_batch.seq_lens + self.speculative_num_draft_tokens
                        ).repeat_interleave(self.speculative_num_draft_tokens),
                        dim=0,
                    ),
                    (1, 0),
                )[:-1]
                mask_extraction_indices = (
                    cols.repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                    + cum_len[:, None]
                ).view(1, -1)
                mask = forward_batch.spec_info.custom_mask[
                    mask_extraction_indices
                ].view(
                    -1, self.speculative_num_draft_tokens
                )  # (bsz * draft_num, draft_num)

                # shift table indices to avoid padding
                # non_masked_page_table [[8, 9, 10],   mask (display with int format) [[1, 0, 0],
                #                        [8, 9, 10],                                   [1, 1, 0],
                #                        [8, 9, 10]]                                   [1, 0, 1]]
                # if masked with padding [[8, 0, 0],   our mask without padding       [[8, 9, 10],
                #                        [8, 9, 0],                                    [8, 9, 10],
                #                        [8, 0, 10]]                                   [8, 10, 9]]
                # note here cache_seqlens_int32 is [1, 2, 2] so extra page indices will be ignored in each row
                col_indices = offsets.expand(
                    mask.shape[0], self.speculative_num_draft_tokens
                )
                # Build keys: if an entry is valid (mask==True), keep its original index;
                # if not, add self.speculative_num_draft_tokens so that it sorts after all valid entries.
                keys = torch.where(
                    mask, col_indices, col_indices + self.speculative_num_draft_tokens
                )
                _, sort_order = torch.sort(keys, dim=1)
                non_masked_page_table = (
                    forward_batch.req_to_token_pool.req_to_token[
                        forward_batch.req_pool_indices, :
                    ]
                    .gather(1, cols)
                    .repeat_interleave(self.speculative_num_draft_tokens, dim=0)
                )  # (bsz, draft_num)
                metadata_expand.page_table = non_masked_page_table.gather(1, sort_order)
                metadata_expand.cache_seqlens_int32 = mask.sum(dim=1).to(torch.int32)
                metadata_expand.cu_seqlens_k = torch.nn.functional.pad(
                    torch.cumsum(
                        metadata_expand.cache_seqlens_int32, dim=0, dtype=torch.int32
                    ),
                    (1, 0),
                )
                self.forward_metadata_spec_decode_expand = metadata_expand

                if self.has_swa:
                    self._init_sliding_window_attn_spec_metadata(
                        metadata, metadata_expand
                    )

        elif forward_batch.forward_mode.is_extend_or_draft_extend_or_mixed():
            metadata.cache_seqlens_int32 = seqlens_in_batch.to(torch.int32)
            metadata.max_seq_len_k = forward_batch.seq_lens_cpu.max().item()
            metadata.cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.int32), (1, 0)
            )
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.max_seq_len_k
            ]

            if (
                any(forward_batch.extend_prefix_lens_cpu)
                or forward_batch.forward_mode == ForwardMode.DRAFT_EXTEND
            ):
                extend_seq_lens = forward_batch.extend_seq_lens
                metadata.max_seq_len_q = max(forward_batch.extend_seq_lens_cpu)
                metadata.cu_seqlens_q = torch.nn.functional.pad(
                    torch.cumsum(extend_seq_lens, dim=0, dtype=torch.int32), (1, 0)
                )
            else:
                metadata.max_seq_len_q = metadata.max_seq_len_k
                metadata.cu_seqlens_q = metadata.cu_seqlens_k

            # Setup local attention if enabled
            if forward_batch.forward_mode == ForwardMode.EXTEND:
                self._init_local_attn_metadata(forward_batch, metadata, device)

        # Encoder metadata for cross attention
        if forward_batch.encoder_lens is not None:
            assert (
                forward_batch.encoder_lens.numel() == 1
            ), "Only encoder size 1 is supported for now"

            metadata.encoder_lens_int32 = forward_batch.encoder_lens.to(torch.int32)
            metadata.encoder_cu_seqlens_k = torch.nn.functional.pad(
                torch.cumsum(metadata.encoder_lens_int32, dim=0, dtype=torch.int32),
                (1, 0),
            )
            metadata.encoder_max_seq_len_k = metadata.encoder_lens_int32.max().item()
            metadata.encoder_page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices, : metadata.encoder_max_seq_len_k
            ]

            # Currently only support forward_batch.encoder_lens.numel() == 1
            metadata.page_table = forward_batch.req_to_token_pool.req_to_token[
                forward_batch.req_pool_indices,
                metadata.encoder_max_seq_len_k : (
                    metadata.encoder_max_seq_len_k + metadata.max_seq_len_k
                ),
            ]

        # Convert the page table to a strided format which is needed by FA3 API
        if self.page_size > 1:
            self.strided_indices = torch.arange(
                0, metadata.page_table.shape[1], self.page_size, device=self.device
            )
            metadata.page_table = (
                metadata.page_table[:, self.strided_indices] // self.page_size
            )

        self.forward_metadata = metadata

    def forward_extend(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ):
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
                # Required on PTL iGPU: set_kv_buffer's scatter into the KV
                # pool and the subsequent attention kernel's gather from the
                # pool are not stream-ordered on XPU; without this sync the
                # kernel reads stale/garbage KV entries. Env-gated so the
                # default path (other GPUs) is unchanged.
                if os.environ.get("SGLANG_XPU_FORCE_SYNC") == "1":
                    import torch as _tt
                    _tt.xpu.synchronize()

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        is_hybrid_swa = (
            layer.sliding_window_size is not None and layer.sliding_window_size > -1
        )
        window_size = (layer.sliding_window_size, 0) if is_hybrid_swa else (-1, -1)

        # currently no FP8 KV cache supported
        k_descale, v_descale = None, None
        # # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # # has corresponding quantization method so that layer.k_scale is not None,
        # # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        # if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
        #     if layer.k_scale is not None:
        #         descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
        #         k_descale = layer.k_scale.expand(descale_shape)
        #         v_descale = layer.v_scale.expand(descale_shape)
        #     q = q.to(self.kv_cache_dtype)
        #     q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
        #     k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        causal = not layer.is_cross_attention

        # Check if we should use local attention
        use_local_attn = (
            self.attention_chunk_size is not None
            and metadata.local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # We do cascade attention for Target Verify with topk > 1
        # We don't use cascade attention for Sliding Window Attention:
        # - Different window sizes should be passed in for each q in the first stage of cascade attention, but FA3 interface doesn't support pass in a list of window sizes.
        # - The overhead of duplicated computation of the common prefix part is small for sliding window layers (seq_len <= window_size), so we can just expand it.
        use_cascade_attn = (
            forward_batch.forward_mode.is_target_verify()
            and self.topk > 1
            and not is_hybrid_swa
        )

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        # Get the appropriate page table based on whether we're using local attention
        if use_local_attn:
            local_metadata = metadata.local_attn_metadata
            page_table = local_metadata.local_block_table
            cu_seqlens_q = local_metadata.local_query_start_loc
            cache_seqlens = local_metadata.local_seqused_k
            max_seqlen_q = local_metadata.local_max_query_len
        elif is_hybrid_swa and metadata.swa_spec_metadata is not None:
            swa_spec_metadata = metadata.swa_spec_metadata
            page_table = swa_spec_metadata.page_table
            cu_seqlens_q = swa_spec_metadata.cu_seqlens_q
            cache_seqlens = swa_spec_metadata.cache_seqlens_int32
            max_seqlen_q = swa_spec_metadata.max_seq_len_q
            cu_seqlens_k = swa_spec_metadata.cu_seqlens_k
        else:
            page_table = metadata.page_table
            cu_seqlens_q = metadata.cu_seqlens_q
            cache_seqlens = metadata.cache_seqlens_int32
            max_seqlen_q = metadata.max_seq_len_q
            cu_seqlens_k = metadata.cu_seqlens_k

        # Use Flash Attention for prefill
        if not self.use_mla:
            # Do multi-head attention
            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )
            if layer.is_cross_attention:
                page_table = metadata.encoder_page_table
                cache_seqlens = metadata.encoder_lens_int32
                cu_seqlens_k = metadata.encoder_cu_seqlens_k
                window_size = (-1, -1)

            # Prefill: SDPA fallback only when SGL_XPU_FA_FALLBACK=1.
            # SGL_XPU_ESIMD_DECODE does NOT force prefill fallback — prefill
            # keeps the native FMHA kernel.
            if (
                not use_cascade_attn
                and os.environ.get("SGL_XPU_FA_FALLBACK") == "1"
            ):
                result = self._sdpa_fallback(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    key_cache=key_cache,
                    value_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    softmax_scale=layer.scaling,
                    causal=causal,
                    window_size=window_size,
                    tp_q_head_num=layer.tp_q_head_num,
                    tp_k_head_num=layer.tp_k_head_num,
                    head_dim=layer.head_dim,
                )
            else:
                result = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                    **kwargs,
                )

            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                    **kwargs,
                )
                o, _ = merge_state_v2_wrapper(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result
        else:
            if (
                forward_batch.attn_attend_prefix_cache is not None
                and not forward_batch.forward_mode.is_target_verify()
                and not forward_batch.forward_mode.is_draft_extend()
            ):
                # Do multi-head attention with chunked prefix cache
                if forward_batch.attn_attend_prefix_cache:
                    assert not get_global_server_args().disable_chunked_prefix_cache
                    # MHA for chunked prefix kv cache when running model with MLA
                    assert forward_batch.prefix_chunk_idx is not None
                    assert forward_batch.prefix_chunk_cu_seq_lens is not None
                    assert forward_batch.prefix_chunk_max_seq_lens is not None

                    chunk_idx = forward_batch.prefix_chunk_idx
                    assert chunk_idx >= 0

                    assert forward_batch.mha_return_lse
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=forward_batch.prefix_chunk_cu_seq_lens[chunk_idx],
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=forward_batch.prefix_chunk_max_seq_lens[chunk_idx],
                        softmax_scale=layer.scaling,
                        causal=False,
                        return_softmax_lse=True,
                    )
                else:
                    # MHA for extend part of sequence without attending prefix kv cache
                    output = flash_attn_varlen_func(
                        q=q.view(-1, layer.tp_q_head_num, layer.head_dim),
                        k=k.view(-1, layer.tp_k_head_num, layer.head_dim).to(q.dtype),
                        v=v.view(-1, layer.tp_k_head_num, layer.v_head_dim).to(q.dtype),
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k=metadata.cu_seqlens_q,
                        max_seqlen_q=metadata.max_seq_len_q,
                        max_seqlen_k=metadata.max_seq_len_q,
                        softmax_scale=layer.scaling,
                        causal=True,
                        return_softmax_lse=forward_batch.mha_return_lse,
                    )
                if forward_batch.mha_return_lse:
                    output, lse, *rest = output
                    lse = torch.transpose(lse, 0, 1).contiguous()
                    return output, lse
                return output
            else:
                # Do absorbed multi-latent attention
                kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(
                    layer.layer_id
                ).to(q.dtype)
                k_rope = kv_cache[:, :, layer.v_head_dim :]
                c_kv = kv_cache[:, :, : layer.v_head_dim]
                k_rope_cache = k_rope.view(
                    -1,
                    self.page_size,
                    layer.tp_k_head_num,
                    layer.head_dim - layer.v_head_dim,
                )
                c_kv_cache = c_kv.view(
                    -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
                )
                if q_rope is not None:
                    q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                    q_rope = q_rope.view(
                        -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                    )
                else:
                    q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                    q_nope = q_all[:, :, : layer.v_head_dim]
                    q_rope = q_all[:, :, layer.v_head_dim :]

                result = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=page_table,
                    cache_seqlens=cache_seqlens,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k_new=cu_seqlens_k if not use_local_attn else None,
                    max_seqlen_q=max_seqlen_q,
                    softmax_scale=layer.scaling,
                    causal=False if use_cascade_attn else causal,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=use_cascade_attn,
                )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_rope,
                            k_cache=k_rope_cache,
                            v_cache=c_kv_cache,
                            qv=q_nope,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                        )
                    )
                    o, _ = merge_state_v2_wrapper(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def _sdpa_fallback(
        self,
        q: torch.Tensor,           # (total_q, num_q_heads, head_dim)
        key_cache: torch.Tensor,   # (num_pages, page_size, num_kv_heads, head_dim)
        value_cache: torch.Tensor, # (num_pages, page_size, num_kv_heads, head_dim)
        page_table: torch.Tensor,  # (batch, max_pages_per_seq) int32
        cache_seqlens: torch.Tensor,  # (batch,) int32 -- length of keys per batch
        cu_seqlens_q: torch.Tensor,   # (batch+1,) int32 -- cumulative query offsets
        softmax_scale: float,
        causal: bool,
        window_size,
        tp_q_head_num: int,
        tp_k_head_num: int,
        head_dim: int,
    ) -> torch.Tensor:
        """SDPA fallback for BOTH prefill and decode. Handles variable q length
        per batch via cu_seqlens_q. Each batch b has q length
        cu_seqlens_q[b+1] - cu_seqlens_q[b] and k length cache_seqlens[b]."""
        batch_size = cache_seqlens.numel()
        num_q_heads = tp_q_head_num
        num_kv_heads = tp_k_head_num
        page_size = key_cache.size(1)

        cu_q_cpu = cu_seqlens_q.to("cpu", dtype=torch.int64).tolist()
        seq_k_cpu = cache_seqlens.to("cpu", dtype=torch.int64).tolist()

        outputs = []
        for b in range(batch_size):
            q_start = cu_q_cpu[b]
            q_end = cu_q_cpu[b + 1]
            q_len = q_end - q_start
            if q_len == 0:
                continue
            k_len = seq_k_cpu[b]
            if k_len == 0:
                outputs.append(
                    torch.zeros(q_len, num_q_heads, head_dim, device=q.device, dtype=q.dtype)
                )
                continue

            # Gather this batch's KV pages.
            num_pages_needed = (k_len + page_size - 1) // page_size
            pages_b = page_table[b, :num_pages_needed].to(torch.long)
            k_b = key_cache.index_select(0, pages_b).reshape(
                -1, num_kv_heads, head_dim
            )[:k_len]
            v_b = value_cache.index_select(0, pages_b).reshape(
                -1, num_kv_heads, head_dim
            )[:k_len]

            # Expand KV heads for GQA.
            if num_kv_heads != num_q_heads:
                group = num_q_heads // num_kv_heads
                k_b = k_b.repeat_interleave(group, dim=1)
                v_b = v_b.repeat_interleave(group, dim=1)

            # (q_len, H, D) → (H, q_len, D); (k_len, H, D) → (H, k_len, D).
            q_b = q[q_start:q_end].transpose(0, 1)
            k_bh = k_b.transpose(0, 1)
            v_bh = v_b.transpose(0, 1)

            # Causal mask: q at position (k_len - q_len + i) attends to keys 0..(k_len - q_len + i).
            if causal and q_len > 1:
                row_idx = torch.arange(q_len, device=q.device).unsqueeze(1) + (k_len - q_len)
                col_idx = torch.arange(k_len, device=q.device).unsqueeze(0)
                attn_mask = col_idx <= row_idx  # (q_len, k_len)
            else:
                attn_mask = None

            out_b = F.scaled_dot_product_attention(
                q_b.unsqueeze(0),  # (1, H, q_len, D)
                k_bh.unsqueeze(0),
                v_bh.unsqueeze(0),
                attn_mask=attn_mask.unsqueeze(0).unsqueeze(0) if attn_mask is not None else None,
                dropout_p=0.0,
                is_causal=False,
                scale=softmax_scale,
            ).squeeze(0).transpose(0, 1)  # (q_len, H, D)
            outputs.append(out_b)

        return torch.cat(outputs, dim=0)

    def _esimd_fallback_decode(
        self,
        q: torch.Tensor,              # (total_q, num_q_heads, head_dim)
        key_cache: torch.Tensor,      # (num_pages, page_size, num_kv_heads, head_dim)
        value_cache: torch.Tensor,    # (num_pages, page_size, num_kv_heads, head_dim)
        page_table: torch.Tensor,     # (batch, max_pages_per_seq) int32
        cache_seqlens: torch.Tensor,  # (batch,) int32
        max_seq_len: int,
        tp_q_head_num: int,
        tp_k_head_num: int,
        head_dim: int,
    ) -> torch.Tensor:
        """ESIMD page_attn_decode fallback (hardcoded headDim=256, gqaRatio=4).

        ESIMD kernel requires:
          - fp16 OR bf16 for q/kv_cache/out. Keep the model's native dtype
            when it's already bf16 so we skip an extra cast on the critical
            path (saves ~6ms/tok on Qwen3.5-4B decode).
          - Merged KV cache shape [2, num_pages, page_size, num_kv_heads, head_dim]
            with stride[0] = 2 * stride[1]. We stack K and V along a new leading dim.
          - page_size is power-of-2 in [64, 1024].
          - num_q_heads / num_kv_heads must be a multiple of 4.
        Qwen3.5-0.8B (HV=2, H=8, D=256, page_size=64) satisfies all.
        """
        # Lazy import so the sglang scheduler subprocess (spawned via fork+spawn)
        # registers the eagle_ops torch.ops namespace.
        if not getattr(self, "_esimd_loaded", False):
            import custom_esimd_kernels_vllm  # noqa: F401 — registers torch.ops.eagle_ops
            self._esimd_loaded = True
        batch_size = cache_seqlens.numel()

        # Keep kernel dtype matched to KV cache dtype to avoid per-decode casts.
        kernel_dtype = key_cache.dtype if key_cache.dtype in (torch.float16, torch.bfloat16) else torch.float16
        q_kern = q.to(kernel_dtype).contiguous() if q.dtype != kernel_dtype else q.contiguous()
        # Gather only the pages this batch actually touches, then stack + cast.
        # Avoids O(num_pages) copy (which was ~800MB/call for Qwen3.5) and
        # produces a compact (2, batch*max_pages, page_size, H_kv, D) buffer.
        # bsz=1 decode: page_table has no duplicates — skip torch.unique() which
        # costs ~2ms/call of CPU dispatch and dominates the decode path.
        pages = page_table.to(torch.long)  # (B, max_pages_per_seq)
        flat = pages.reshape(-1)
        # Required on PTL iGPU: ensure preceding writes to key_cache /
        # value_cache have landed before the index_select gather runs.
        if os.environ.get("SGLANG_XPU_FORCE_SYNC") == "1":
            import torch as _tt
            _tt.xpu.synchronize()
        # PTL workaround: the ESIMD page_attn_decode kernel crashes the GPU
        # when given a kv_cache tensor produced by
        # `torch.stack([index_select(...), ...]).contiguous()` — even though
        # the tensor is contiguous with the right shape/dtype. Isolated stress
        # test (test_esimd_bisect.py `stack_only`) shows this is the only
        # per-call op that triggers the crash. Using a persistent pre-allocated
        # kv_merged buffer and `copy_`-ing each decode's gather into it is
        # stable (`index_copy` mode passes 20+ repeated calls cleanly).
        num_pages_sel = flat.numel()
        H_kv, head_dim_kv = tp_k_head_num, head_dim
        page_size = key_cache.size(1)
        cache_key = (num_pages_sel, page_size, H_kv, head_dim_kv, kernel_dtype)
        kv_merged_cache = getattr(self, "_kv_merged_cache", {})
        kv_merged = kv_merged_cache.get(cache_key)
        if kv_merged is None:
            kv_merged = torch.empty(
                (2, num_pages_sel, page_size, H_kv, head_dim_kv),
                dtype=kernel_dtype, device=q.device,
            )
            kv_merged_cache[cache_key] = kv_merged
            self._kv_merged_cache = kv_merged_cache
        # index_select into the persistent buffer. When KV cache dtype already
        # matches kernel_dtype (bf16 model -> bf16 cache -> bf16 kernel), this
        # is one contiguous copy with no intermediate cast.
        if key_cache.dtype == kernel_dtype:
            kv_merged[0].copy_(key_cache.index_select(0, flat))
            kv_merged[1].copy_(value_cache.index_select(0, flat))
        else:
            kv_merged[0].copy_(key_cache.index_select(0, flat).to(kernel_dtype))
            kv_merged[1].copy_(value_cache.index_select(0, flat).to(kernel_dtype))
        if os.environ.get("SGLANG_XPU_FORCE_SYNC") == "1":
            import torch as _tt
            _tt.xpu.synchronize()
        # Block table just maps each position to its own slot in the compact buffer.
        new_block_table = (
            torch.arange(flat.numel(), device=q.device, dtype=torch.int32)
            .view(pages.shape)
            .contiguous()
        )
        seq_lens_i32 = cache_seqlens.to(torch.int32).contiguous()

        out_kern = torch.empty(
            (batch_size, tp_q_head_num, head_dim),
            device=q.device,
            dtype=kernel_dtype,
        )
        torch.ops.eagle_ops.page_attn_decode(
            q_kern,
            kv_merged,
            new_block_table,
            seq_lens_i32,
            out_kern,
            1,            # max_query_len (decode: always 1)
            max_seq_len,  # max_seq_len across batch (from metadata)
        )
        # Cast back to original dtype only when needed.
        return out_kern if out_kern.dtype == q.dtype else out_kern.to(q.dtype)

    def _sdpa_fallback_decode(
        self,
        q: torch.Tensor,           # (total_q, num_q_heads, head_dim)
        key_cache: torch.Tensor,   # (num_pages, page_size, num_kv_heads, head_dim)
        value_cache: torch.Tensor, # (num_pages, page_size, num_kv_heads, head_dim)
        page_table: torch.Tensor,  # (batch, max_pages_per_seq) int32
        cache_seqlens: torch.Tensor,  # (batch,) int32
        softmax_scale: float,
        causal: bool,
        window_size,
        tp_q_head_num: int,
        tp_k_head_num: int,
        head_dim: int,
    ) -> torch.Tensor:
        """Gather the paged KV cache into contiguous tensors and delegate to
        torch SDPA. Used to isolate the FMHA decode kernel hang on XPU.

        Decode path only: assumes 1 query token per batch. Same-length batches
        are processed together; uneven batches loop per-sequence.
        """
        batch_size = cache_seqlens.numel()
        num_q_heads = tp_q_head_num
        num_kv_heads = tp_k_head_num
        page_size = key_cache.size(1)

        # Reshape q to (batch, num_q_heads, head_dim) — decode has 1 q per batch.
        q_bhd = q.view(batch_size, num_q_heads, head_dim)

        # Pull cache_seqlens to CPU once for control flow.
        seq_lens_cpu = cache_seqlens.to("cpu", dtype=torch.int64).tolist()
        max_k = max(seq_lens_cpu) if seq_lens_cpu else 0
        if max_k == 0:
            return torch.zeros_like(q_bhd)

        # Gather K/V per batch into a padded (B, max_k, H_kv, D) tensor.
        num_pages_needed = (max_k + page_size - 1) // page_size
        # page_table rows: (batch, max_pages_per_seq); use first num_pages_needed.
        pages = page_table[:, :num_pages_needed].to(torch.long)
        pages_flat = pages.reshape(-1).contiguous()  # (B*P,)
        # key/value_cache: (num_pages, page_size, H_kv, D)
        # Force key/value_cache to be contiguous before the gather; the pool
        # tensor may carry a non-trivial stride that XPU index_select dislikes.
        kc = key_cache.contiguous() if not key_cache.is_contiguous() else key_cache
        vc = value_cache.contiguous() if not value_cache.is_contiguous() else value_cache
        k_flat_pages = torch.index_select(kc, 0, pages_flat)
        v_flat_pages = torch.index_select(vc, 0, pages_flat)
        # k_flat_pages shape: (B*P, page_size, H_kv, D) → (B, P*page_size, H_kv, D)
        k_flat = k_flat_pages.reshape(
            batch_size, num_pages_needed * page_size, num_kv_heads, head_dim
        )
        v_flat = v_flat_pages.reshape(
            batch_size, num_pages_needed * page_size, num_kv_heads, head_dim
        )

        # Expand KV heads for GQA before calling SDPA (SDPA's enable_gqa is
        # newer-only; manual expand keeps compat).
        if num_kv_heads != num_q_heads:
            group = num_q_heads // num_kv_heads
            k_flat = k_flat.repeat_interleave(group, dim=2)
            v_flat = v_flat.repeat_interleave(group, dim=2)

        # Build per-batch mask: True where key index < seqlen and within window.
        dev = q.device
        k_pos = torch.arange(k_flat.size(1), device=dev)
        seqlens = cache_seqlens.to(dev, dtype=torch.int64).unsqueeze(1)  # (B, 1)
        attn_mask = k_pos.unsqueeze(0) < seqlens  # (B, max_k_padded)
        # Sliding window (left, right) — for decode q is at position seqlen-1,
        # so a key at position p is allowed if (seqlen-1 - p) <= window_left.
        wl, wr = window_size
        if wl is not None and wl >= 0:
            left_bound = seqlens - 1 - wl
            attn_mask = attn_mask & (k_pos.unsqueeze(0) >= left_bound)
        # Causal for decode reduces to: mask keys beyond seqlen-1, already handled above.
        # window_right >= 0 doesn't affect decode (no future keys).

        # SDPA expects (B, H, T, D). q_bhd is (B, H, D); add T=1 dim.
        q_sdpa = q_bhd.unsqueeze(2).transpose(1, 2)  # (B, 1, H, D) -> (B, H, 1, D) via next transpose
        # Actually want (B, H, 1, D):
        q_sdpa = q_bhd.unsqueeze(2)  # (B, H, 1, D)
        k_sdpa = k_flat.transpose(1, 2).contiguous()  # (B, H, max_k, D)
        v_sdpa = v_flat.transpose(1, 2).contiguous()  # (B, H, max_k, D)

        # attn_mask: (B, max_k) → broadcast to (B, 1, 1, max_k)
        attn_mask_bf = attn_mask.unsqueeze(1).unsqueeze(1)

        out = F.scaled_dot_product_attention(
            q_sdpa,
            k_sdpa,
            v_sdpa,
            attn_mask=attn_mask_bf,
            dropout_p=0.0,
            is_causal=False,
            scale=softmax_scale,
        )  # (B, H, 1, D)
        # Back to (total_q, num_q_heads, head_dim) with total_q == batch_size.
        return out.squeeze(2)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache=True,
        # For multi-head latent attention
        q_rope: Optional[torch.Tensor] = None,
        k_rope: Optional[torch.Tensor] = None,
        sinks: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if k is not None:
            assert v is not None
            if save_kv_cache:
                cache_loc = (
                    forward_batch.out_cache_loc
                    if not layer.is_cross_attention
                    else forward_batch.encoder_out_cache_loc
                )
                if not self.use_mla:
                    forward_batch.token_to_kv_pool.set_kv_buffer(
                        layer, cache_loc, k, v, layer.k_scale, layer.v_scale
                    )
                else:
                    forward_batch.token_to_kv_pool.set_mla_kv_buffer(
                        layer,
                        cache_loc,
                        k,
                        k_rope,
                    )
                # Required on PTL iGPU: same rationale as forward_extend.
                if os.environ.get("SGLANG_XPU_FORCE_SYNC") == "1":
                    import torch as _tt
                    _tt.xpu.synchronize()

        # Use precomputed metadata across all layers
        metadata = self.forward_metadata
        local_attn_metadata = getattr(metadata, "local_attn_metadata", None)
        use_local_attn = (
            self.attention_chunk_size is not None
            and local_attn_metadata is not None
            and (hasattr(layer, "use_irope") and layer.use_irope)
        )

        # When Spec Decode enabled, forward_decode would be called with two mode:
        # 1. DRAFT_DECODE: we enable cascade attention when top_k > 1
        # 2. IDLE: we don’t need cascade attention, spec_info will be none in this case
        use_cascade_attn = forward_batch.spec_info is not None and self.topk > 1

        # Calculate window size (can be moved to metadata if layer properties don't change)
        # we don't do layer.sliding_window_size - 1 since in model.get_attention_sliding_window_size() we already - 1
        # here is two side inclusive
        window_size = (
            (layer.sliding_window_size, 0)
            if layer.sliding_window_size is not None and layer.sliding_window_size > -1
            else (-1, -1)
        )
        causal = not layer.is_cross_attention

        # For fa3 interface version compatibility, we put new fields into conditional keyword args
        kwargs = {}
        if sinks is not None:
            kwargs["sinks"] = sinks

        k_descale, v_descale = None, None
        # only use kv scaling if: 1) fp8 kv is explicitly enabled, 2) RadixAttention
        # has corresponding quantization method so that layer.k_scale is not None,
        # 3) layer.head_dim <= 256 since fa3 kernel require fp16 and bf16 data type in this case.
        if self.kv_cache_dtype_str != "auto" and layer.head_dim <= 256:
            if layer.k_scale is not None:
                descale_shape = (forward_batch.batch_size, layer.tp_k_head_num)
                k_descale = layer.k_scale.expand(descale_shape)
                v_descale = layer.v_scale.expand(descale_shape)
            q = q.to(self.kv_cache_dtype)
            q_rope = q_rope.to(self.kv_cache_dtype) if q_rope is not None else None
            k_rope = k_rope.to(self.kv_cache_dtype) if k_rope is not None else None
        if not self.use_mla:
            # Do multi-head attention

            key_cache, value_cache = forward_batch.token_to_kv_pool.get_kv_buffer(
                layer.layer_id
            )
            key_cache = key_cache.view(
                -1, self.page_size, layer.tp_k_head_num, layer.head_dim
            )
            value_cache = value_cache.view(
                -1, self.page_size, layer.tp_v_head_num, layer.head_dim
            )

            if layer.is_cross_attention:
                # Always use non-chunked logic for cross-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=metadata.encoder_page_table,
                    cache_seqlens=metadata.encoder_lens_int32,
                    cu_seqlens_q=metadata.cu_seqlens_q,
                    cu_seqlens_k_new=metadata.encoder_cu_seqlens_k,
                    max_seqlen_q=1,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            elif use_local_attn:
                # Use chunked (local) attention batching for self-attention
                o = flash_attn_with_kvcache(
                    q=q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim),
                    k_cache=key_cache,
                    v_cache=value_cache,
                    page_table=local_attn_metadata.local_block_table,
                    cache_seqlens=local_attn_metadata.local_seqused_k,
                    cu_seqlens_q=local_attn_metadata.local_query_start_loc,
                    cu_seqlens_k_new=None,
                    max_seqlen_q=local_attn_metadata.local_max_query_len,
                    softmax_scale=layer.scaling,
                    causal=True,
                    window_size=(-1, -1),
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    **kwargs,
                )
            else:
                page_table = metadata.page_table
                cache_seqlens = metadata.cache_seqlens_int32
                cu_seqlens_k = metadata.cu_seqlens_k
                max_seqlen_q = metadata.max_seq_len_q
                q_reshaped = q.contiguous().view(
                    -1, layer.tp_q_head_num, layer.head_dim
                )

                if (
                    not use_cascade_attn
                    and os.environ.get("SGL_XPU_ESIMD_DECODE") == "1"
                ):
                    result = self._esimd_fallback_decode(
                        q=q_reshaped,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens,
                        max_seq_len=int(metadata.max_seq_len_k),
                        tp_q_head_num=layer.tp_q_head_num,
                        tp_k_head_num=layer.tp_k_head_num,
                        head_dim=layer.head_dim,
                    )
                elif (
                    not use_cascade_attn
                    and os.environ.get("SGL_XPU_FA_FALLBACK") == "1"
                ):
                    result = self._sdpa_fallback_decode(
                        q=q_reshaped,
                        key_cache=key_cache,
                        value_cache=value_cache,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens,
                        softmax_scale=layer.scaling,
                        causal=causal,
                        window_size=window_size,
                        tp_q_head_num=layer.tp_q_head_num,
                        tp_k_head_num=layer.tp_k_head_num,
                        head_dim=layer.head_dim,
                    )
                else:
                    # Default: single-token self-attention
                    result = flash_attn_with_kvcache(
                        q=q_reshaped,
                        k_cache=key_cache,
                        v_cache=value_cache,
                        page_table=page_table,
                        cache_seqlens=cache_seqlens,
                        cu_seqlens_q=metadata.cu_seqlens_q,
                        cu_seqlens_k_new=cu_seqlens_k,
                        max_seqlen_q=max_seqlen_q,
                        softmax_scale=layer.scaling,
                        causal=False if use_cascade_attn else causal,
                        window_size=window_size,
                        softcap=layer.logit_cap,
                        k_descale=k_descale,
                        v_descale=v_descale,
                        return_softmax_lse=use_cascade_attn,
                        **kwargs,
                    )
                if use_cascade_attn:
                    o, softmax_lse, *rest = result
                    o_expand, softmax_lse_expand, *rest_expand = (
                        flash_attn_with_kvcache(
                            q=q_reshaped,
                            k_cache=key_cache,
                            v_cache=value_cache,
                            page_table=self.forward_metadata_spec_decode_expand.page_table,
                            cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                            cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                            cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                            max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                            softmax_scale=layer.scaling,
                            causal=False,
                            window_size=window_size,
                            softcap=layer.logit_cap,
                            k_descale=k_descale,
                            v_descale=v_descale,
                            return_softmax_lse=True,
                            **kwargs,
                        )
                    )
                    o, _ = merge_state_v2(
                        o,
                        softmax_lse.T.contiguous(),
                        o_expand,
                        softmax_lse_expand.T.contiguous(),
                    )
                else:
                    o = result
        else:
            # Do absorbed multi-latent attention
            kv_cache = forward_batch.token_to_kv_pool.get_key_buffer(layer.layer_id).to(
                q.dtype
            )
            k_rope = kv_cache[:, :, layer.v_head_dim :]
            c_kv = kv_cache[:, :, : layer.v_head_dim]
            k_rope_cache = k_rope.view(
                -1,
                self.page_size,
                layer.tp_k_head_num,
                layer.head_dim - layer.v_head_dim,
            )
            c_kv_cache = c_kv.view(
                -1, self.page_size, layer.tp_v_head_num, layer.v_head_dim
            )

            if q_rope is not None:
                q_nope = q.view(-1, layer.tp_q_head_num, layer.v_head_dim)
                q_rope = q_rope.view(
                    -1, layer.tp_q_head_num, layer.head_dim - layer.v_head_dim
                )
            else:
                q_all = q.contiguous().view(-1, layer.tp_q_head_num, layer.head_dim)
                q_nope = q_all[:, :, : layer.v_head_dim]
                q_rope = q_all[:, :, layer.v_head_dim :]
            max_seqlen_q = metadata.max_seq_len_q

            result = flash_attn_with_kvcache(
                q=q_rope,
                k_cache=k_rope_cache,
                v_cache=c_kv_cache,
                qv=q_nope,
                page_table=metadata.page_table,
                cache_seqlens=metadata.cache_seqlens_int32,
                cu_seqlens_q=metadata.cu_seqlens_q,
                cu_seqlens_k_new=metadata.cu_seqlens_k,
                max_seqlen_q=max_seqlen_q,
                softmax_scale=layer.scaling,
                causal=False if use_cascade_attn else causal,
                softcap=layer.logit_cap,
                k_descale=k_descale,
                v_descale=v_descale,
                return_softmax_lse=use_cascade_attn,  # softmax_lse is needed for merge states
            )
            if use_cascade_attn:
                o, softmax_lse, *rest = result
                o_expand, softmax_lse_expand, *rest_expand = flash_attn_with_kvcache(
                    q=q_rope,
                    k_cache=k_rope_cache,
                    v_cache=c_kv_cache,
                    qv=q_nope,
                    page_table=self.forward_metadata_spec_decode_expand.page_table,
                    cache_seqlens=self.forward_metadata_spec_decode_expand.cache_seqlens_int32,
                    cu_seqlens_q=self.forward_metadata_spec_decode_expand.cu_seqlens_q,
                    cu_seqlens_k_new=self.forward_metadata_spec_decode_expand.cu_seqlens_k,
                    max_seqlen_q=self.forward_metadata_spec_decode_expand.max_seq_len_q,
                    softmax_scale=layer.scaling,
                    causal=False,
                    window_size=window_size,
                    softcap=layer.logit_cap,
                    k_descale=k_descale,
                    v_descale=v_descale,
                    return_softmax_lse=True,
                )
                o, _ = merge_state_v2(
                    o,
                    softmax_lse.T.contiguous(),
                    o_expand,
                    softmax_lse_expand.T.contiguous(),
                )
            else:
                o = result

        return o.view(-1, layer.tp_q_head_num * layer.v_head_dim)

    def get_cuda_graph_seq_len_fill_value(self):
        """Get the fill value for sequence length in CUDA graph."""
        return 1

    def _init_local_attn_metadata(
        self, forwardbatch: ForwardBatch, metadata: FlashAttentionMetadata, device
    ):
        """Centralized utility to initialize local_attn_metadata if chunked attention is enabled."""
        if self.attention_chunk_size is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q = metadata.cu_seqlens_q
        cache_seqlens_int32 = metadata.cache_seqlens_int32
        if self.is_hybrid_swa:
            page_table = self.full_to_swa_index_mapping[metadata.page_table].to(
                torch.int32
            )
        else:
            page_table = metadata.page_table
        if cu_seqlens_q is None or cache_seqlens_int32 is None or page_table is None:
            metadata.local_attn_metadata = None
            return

        cu_seqlens_q_np = cu_seqlens_q.cpu().numpy()
        seq_lens_np = cache_seqlens_int32.cpu().numpy()
        (
            seqlens_q_local_np,
            cu_seqlens_q_local_np,
            seqlens_k_local_np,
            block_table_local,
        ) = make_local_attention_virtual_batches(
            self.attention_chunk_size,
            cu_seqlens_q_np,
            seq_lens_np,
            page_table,
            self.page_size,
        )

        local_metadata = FlashAttentionMetadata.LocalAttentionMetadata(
            local_query_start_loc=torch.from_numpy(cu_seqlens_q_local_np).to(device),
            local_seqused_k=torch.from_numpy(seqlens_k_local_np).to(device),
            local_block_table=block_table_local.to(device),
            local_max_query_len=int(seqlens_q_local_np.max()),
            local_max_seq_len=int(seqlens_k_local_np.max()),
        )
        metadata.local_attn_metadata = local_metadata

    def _init_sliding_window_attn_spec_metadata(
        self,
        metadata: FlashAttentionMetadata,
        metadata_expand: FlashAttentionMetadata,
        metadata_swa: Optional[FlashAttentionMetadata] = None,
    ):
        # TODO: support page_size > 1 for swa spec
        assert (
            self.page_size == 1
        ), "FlashAttention backend doesn't support topk > 1 speculative decoding with page size > 1 sliding window attention"

        cache_seqlens_int32 = (
            metadata.cache_seqlens_int32.repeat_interleave(
                self.speculative_num_draft_tokens
            )
            + metadata_expand.cache_seqlens_int32
        )
        cu_seqlens_k = torch.nn.functional.pad(
            torch.cumsum(cache_seqlens_int32, dim=0, dtype=torch.int32), (1, 0)
        )
        bs = cache_seqlens_int32.shape[0]
        page_table = (
            metadata.page_table.new_zeros(
                (bs, metadata.max_seq_len_k + metadata_expand.page_table.shape[1])
            )
            if metadata_swa is None
            else metadata_swa.page_table
        )

        prepare_swa_spec_page_table_triton(
            page_table,
            metadata.page_table,
            metadata_expand.page_table,
            metadata.cache_seqlens_int32,
            metadata_expand.cache_seqlens_int32,
            self.speculative_num_draft_tokens,
        )

        if metadata_swa is None:
            metadata_swa = FlashAttentionMetadata()
            metadata_swa.max_seq_len_q = 1
            metadata_swa.cu_seqlens_q = metadata_expand.cu_seqlens_q
            metadata_swa.cache_seqlens_int32 = cache_seqlens_int32
            metadata_swa.cu_seqlens_k = cu_seqlens_k
            metadata_swa.page_table = page_table
        else:
            metadata_swa.cache_seqlens_int32.copy_(cache_seqlens_int32)
            metadata_swa.cu_seqlens_k.copy_(cu_seqlens_k)

        metadata.swa_spec_metadata = metadata_swa
