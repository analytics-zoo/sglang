# SPDX-License-Identifier: Apache-2.0

import logging
import math
from typing import Iterable, Optional, Tuple

import torch
from torch import nn
from transformers import PretrainedConfig

from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from sglang.srt.layers.activation import SiluAndMul
from sglang.srt.layers.layernorm import Gemma4RMSNorm
from sglang.srt.layers.linear import (
    ColumnParallelLinear,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from sglang.srt.layers.logits_processor import LogitsProcessor, LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.layers.radix_attention import AttentionType, RadixAttention
from sglang.srt.layers.rotary_embedding import get_rope
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
from sglang.srt.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.utils import add_prefix, empty_device_cache, make_layers

logger = logging.getLogger(__name__)

ONYX_QUERY_SCALE_FACTOR = 43.7840518911
ONYX_LOGITS_SCALE = 0.19611613513818404
ONYX_LOGITS_SOFT_CAP = 20.0
ONYX_VISION_WEIGHT_PREFIXES = (
    "model.vision_encoder.",
    "model.vision_adapter.",
    "model.vision_projection.",
    "model.perception_emb_norm.",
)


def get_onyx_query_scale(config: PretrainedConfig) -> float:
    scale_factor = getattr(
        config,
        "query_pre_attn_scalar",
        getattr(config, "qk_scale_factor", ONYX_QUERY_SCALE_FACTOR),
    )
    return scale_factor / math.sqrt(config.head_dim)


def get_onyx_layer_types(config: PretrainedConfig) -> list[str]:
    layer_types = getattr(config, "layer_types", None)
    if layer_types is not None:
        if len(layer_types) != config.num_hidden_layers:
            raise ValueError(
                "Onyx layer_types length must equal num_hidden_layers: "
                f"{len(layer_types)} != {config.num_hidden_layers}"
            )
        invalid_types = set(layer_types) - {"sliding_attention", "full_attention"}
        if invalid_types:
            raise ValueError(f"Unsupported Onyx layer types: {sorted(invalid_types)}")
        return list(layer_types)

    pattern = getattr(config, "sliding_window_pattern", None)
    if pattern is None:
        pattern = [
            config.sliding_window,
            config.sliding_window,
            config.sliding_window,
            0,
        ]
    if not pattern or not any(window <= 0 for window in pattern):
        raise ValueError(
            "Onyx sliding_window_pattern must include a full-attention layer"
        )

    nope_frequency = getattr(config, "every_n_layers_nope", None) or 1
    count_backward_offset = nope_frequency - config.num_hidden_layers % nope_frequency
    return [
        (
            "sliding_attention"
            if pattern[(layer_id + count_backward_offset) % len(pattern)] > 0
            else "full_attention"
        )
        for layer_id in range(config.num_hidden_layers)
    ]


def onyx_layer_uses_rope(config: PretrainedConfig, layer_id: int) -> bool:
    if not getattr(config, "use_rope", True):
        return False

    no_rope_layers = getattr(config, "no_rope_layers", None)
    if no_rope_layers is not None:
        if len(no_rope_layers) != config.num_hidden_layers:
            raise ValueError(
                "Onyx no_rope_layers length must equal num_hidden_layers: "
                f"{len(no_rope_layers)} != {config.num_hidden_layers}"
            )
        return bool(no_rope_layers[layer_id])

    every_n_layers_nope = getattr(config, "every_n_layers_nope", None)
    if every_n_layers_nope is None:
        return True
    return (config.num_hidden_layers - layer_id - 1) % every_n_layers_nope != 0


def get_onyx_sliding_window(
    config: PretrainedConfig, layer_id: int, layer_type: Optional[str] = None
) -> int:
    if layer_type is None:
        layer_type = get_onyx_layer_types(config)[layer_id]
    if layer_type == "full_attention":
        return -1

    pattern = getattr(config, "sliding_window_pattern", None)
    if pattern is not None:
        nope_frequency = getattr(config, "every_n_layers_nope", None) or 1
        count_backward_offset = (
            nope_frequency - config.num_hidden_layers % nope_frequency
        )
        window = pattern[(layer_id + count_backward_offset) % len(pattern)]
    else:
        window = config.sliding_window
    if window <= 0:
        raise ValueError(
            f"Onyx sliding-attention layer {layer_id} has invalid window {window}"
        )

    # HF/Onyx uses an inclusive window; RadixAttention stores the exclusive offset.
    return window - 1


class OnyxOffsetRMSNorm(Gemma4RMSNorm):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__(hidden_size, eps=eps, scale_shift=1.0)


class OnyxRMSNorm(Gemma4RMSNorm):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__(hidden_size, eps=eps, scale_shift=0.0)


class OnyxScalelessRMSNorm(Gemma4RMSNorm):
    def __init__(self, hidden_size: int, eps: float):
        super().__init__(hidden_size, eps=eps, with_scale=False)


class OnyxLogitsProcessor(LogitsProcessor):
    def _compute_lm_head(
        self,
        hidden_states: torch.Tensor,
        lm_head: VocabParallelEmbedding,
        embedding_bias: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        logits = super()._compute_lm_head(hidden_states, lm_head, embedding_bias)
        return logits.float()


class OnyxMLP(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        if getattr(config, "hidden_act", "silu") != "silu":
            raise ValueError("Onyx currently supports only the SiLU MLP activation")
        self.gate_up_proj = MergedColumnParallelLinear(
            config.hidden_size,
            [config.intermediate_size, config.intermediate_size],
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("gate_up_proj", prefix),
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("down_proj", prefix),
        )
        self.act_fn = SiluAndMul()

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(hidden_states)
        hidden_states = self.act_fn(gate_up)
        hidden_states, _ = self.down_proj(hidden_states)
        return hidden_states


class OnyxAttention(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.total_num_heads = config.num_attention_heads
        self.total_num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim

        tp_size = get_tensor_model_parallel_world_size()
        tp_rank = get_tensor_model_parallel_rank()
        if self.total_num_heads % tp_size != 0:
            raise ValueError(
                f"Onyx attention heads {self.total_num_heads} not divisible by TP {tp_size}"
            )
        self.num_heads = self.total_num_heads // tp_size
        if self.total_num_kv_heads >= tp_size:
            if self.total_num_kv_heads % tp_size != 0:
                raise ValueError(
                    "Onyx KV heads must divide TP when they are not replicated"
                )
            self.num_kv_heads = self.total_num_kv_heads // tp_size
        else:
            if tp_size % self.total_num_kv_heads != 0:
                raise ValueError(
                    "Onyx TP size must be divisible by KV heads when replicating KV"
                )
            self.num_kv_heads = 1

        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = 1.0 / math.sqrt(self.head_dim)
        self.query_scale = get_onyx_query_scale(config)

        self.qkv_proj = QKVParallelLinear(
            self.hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("qkv_proj", prefix),
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        self.use_output_gate = getattr(config, "use_attn_output_gate", True)
        if self.use_output_gate:
            self.output_gate_proj = ColumnParallelLinear(
                self.hidden_size,
                self.total_num_heads * self.head_dim,
                bias=False,
                quant_config=quant_config,
                prefix=add_prefix("output_gate_proj", prefix),
                tp_rank=tp_rank,
                tp_size=tp_size,
            )
        else:
            self.output_gate_proj = None
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            self.hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=add_prefix("o_proj", prefix),
        )

        self.use_qk_norm = getattr(config, "use_qk_norm", True)
        if self.use_qk_norm:
            qk_norm_eps = getattr(config, "qk_norm_eps", config.rms_norm_eps)
            self.q_norm = OnyxScalelessRMSNorm(self.head_dim, eps=qk_norm_eps)
            self.k_norm = OnyxScalelessRMSNorm(self.head_dim, eps=qk_norm_eps)
        else:
            self.q_norm = None
            self.k_norm = None

        self.use_rope = onyx_layer_uses_rope(config, layer_id)
        if self.use_rope:
            self.rotary_emb = get_rope(
                self.head_dim,
                rotary_dim=self.head_dim,
                max_position=config.max_position_embeddings,
                base=config.rope_theta,
                is_neox_style=False,
                dtype=getattr(config, "torch_dtype", None),
            )
        else:
            self.rotary_emb = None

        layer_type = get_onyx_layer_types(config)[layer_id]
        sliding_window_size = get_onyx_sliding_window(
            config, layer_id, layer_type=layer_type
        )
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            sliding_window_size=sliding_window_size,
            quant_config=quant_config,
            attn_type=AttentionType.DECODER,
            use_irope=self.use_rope,
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

        if self.use_qk_norm:
            q_shape = q.shape
            k_shape = k.shape
            q = self.q_norm(q.reshape(-1, self.head_dim)).reshape(q_shape)
            k = self.k_norm(k.reshape(-1, self.head_dim)).reshape(k_shape)
            q = q * self.query_scale

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)

        attn_output = self.attn(q, k, v, forward_batch)
        if self.output_gate_proj is not None:
            output_gate, _ = self.output_gate_proj(hidden_states)
            attn_output = torch.sigmoid(output_gate) * attn_output
        output, _ = self.o_proj(attn_output)
        return output


class OnyxDecoderLayer(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        layer_id: int,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.self_attn = OnyxAttention(
            config=config,
            layer_id=layer_id,
            quant_config=quant_config,
            prefix=add_prefix("self_attn", prefix),
        )
        self.mlp = OnyxMLP(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = OnyxOffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_attention_layernorm = OnyxOffsetRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        post_norm_eps = getattr(config, "post_norm_eps", config.rms_norm_eps)
        self.post_attn_norm = OnyxOffsetRMSNorm(config.hidden_size, eps=post_norm_eps)
        self.post_ffn_norm = OnyxOffsetRMSNorm(config.hidden_size, eps=post_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            forward_batch=forward_batch,
        )
        hidden_states = residual + self.post_attn_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + self.post_ffn_norm(hidden_states)


class OnyxModel(nn.Module):
    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.has_vision = False

        self.pp_group = get_pp_group()
        self.normalize_tok_embeddings = getattr(
            config, "normalize_tok_embeddings", True
        )
        if self.pp_group.is_first_rank:
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                prefix=add_prefix("embed_tokens", prefix),
            )
            if self.normalize_tok_embeddings:
                self.embed_norm = OnyxScalelessRMSNorm(
                    config.hidden_size, eps=config.rms_norm_eps
                )
            else:
                self.embed_norm = None
        else:
            self.embed_tokens = PPMissingLayer()
            self.embed_norm = PPMissingLayer()

        self.layers, self.start_layer, self.end_layer = make_layers(
            config.num_hidden_layers,
            lambda idx, prefix: OnyxDecoderLayer(
                config=config,
                layer_id=idx,
                quant_config=quant_config,
                prefix=prefix,
            ),
            pp_rank=self.pp_group.rank_in_group,
            pp_size=self.pp_group.world_size,
            prefix=add_prefix("layers", prefix),
        )

        if self.pp_group.is_last_rank:
            self.norm = OnyxRMSNorm(
                config.hidden_size,
                eps=getattr(config, "final_norm_eps", config.rms_norm_eps),
            )
        else:
            self.norm = PPMissingLayer()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor | PPProxyTensors:
        if self.pp_group.is_first_rank:
            if input_embeds is None:
                hidden_states = self.embed_tokens(input_ids)
            else:
                hidden_states = input_embeds
            if self.embed_norm is not None:
                hidden_states = self.embed_norm(hidden_states)
        else:
            if pp_proxy_tensors is None or "hidden_states" not in pp_proxy_tensors:
                raise ValueError("Pipeline rank requires hidden_states proxy tensor")
            hidden_states = pp_proxy_tensors["hidden_states"]

        for layer_id in range(self.start_layer, self.end_layer):
            layer = self.layers[layer_id]
            hidden_states = layer(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        if not self.pp_group.is_last_rank:
            return PPProxyTensors(
                {
                    "hidden_states": hidden_states,
                    "residual": torch.zeros_like(hidden_states),
                }
            )
        return self.norm(hidden_states)


class OnyxForCausalLM(nn.Module):
    packed_modules_mapping = {
        "qkv_proj": ["q_proj", "k_proj", "v_proj"],
        "gate_up_proj": ["gate_proj", "up_proj"],
    }

    def __init__(
        self,
        config: PretrainedConfig,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
        self.pp_group = get_pp_group()
        self.quant_config = quant_config
        self.model = OnyxModel(
            config=config,
            quant_config=quant_config,
            prefix=add_prefix("model", prefix),
        )
        if self.pp_group.is_last_rank:
            self.lm_head = ParallelLMHead(
                config.vocab_size,
                config.hidden_size,
                bias=False,
                quant_config=None,
                prefix=add_prefix("lm_head", prefix),
            )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_soft_cap = getattr(
            config,
            "final_logit_softcapping",
            getattr(config, "output_soft_cap_temp", ONYX_LOGITS_SOFT_CAP),
        )
        self.logits_scale = getattr(
            config,
            "logits_scaling",
            getattr(config, "output_multiplier", ONYX_LOGITS_SCALE),
        )
        self.logits_processor = OnyxLogitsProcessor(
            config, logit_scale=self.logits_scale
        )
        self.logits_processor.final_logit_softcapping = self.logits_soft_cap

    def validate_online_fp8_weights(self) -> None:
        from sglang.srt.layers.quantization.fp8 import Fp8LinearMethod

        expected_shapes = {
            "qkv_proj": (6656, 2304),
            "output_gate_proj": (6656, 2048),
            "o_proj": (2048, 6656),
            "gate_up_proj": (6656, 19968),
            "down_proj": (9984, 6656),
        }
        expected_per_type = self.config.num_hidden_layers
        counts = {module_type: 0 for module_type in expected_shapes}
        errors = []

        for name, module in self.named_modules():
            module_type = name.rsplit(".", 1)[-1]
            if module_type not in expected_shapes:
                continue
            counts[module_type] += 1
            if not isinstance(getattr(module, "quant_method", None), Fp8LinearMethod):
                errors.append(f"{name}: quant method is not Fp8LinearMethod")
                continue
            if module.weight.dtype not in (
                torch.float8_e4m3fn,
                torch.float8_e4m3fnuz,
            ):
                errors.append(f"{name}: unexpected weight dtype {module.weight.dtype}")
            if tuple(module.weight.shape) != expected_shapes[module_type]:
                errors.append(
                    f"{name}: weight shape {tuple(module.weight.shape)} != "
                    f"{expected_shapes[module_type]}"
                )
            weight_scale = getattr(module, "weight_scale", None)
            if (
                weight_scale is None
                or weight_scale.numel() != 1
                or not bool(torch.isfinite(weight_scale).all())
            ):
                errors.append(f"{name}: weight_scale is not one finite scalar")
            if getattr(module, "input_scale", None) is not None:
                errors.append(f"{name}: dynamic FP8 path persisted input_scale")

        for module_type, count in counts.items():
            if count != expected_per_type:
                errors.append(
                    f"{module_type}: found {count}, expected {expected_per_type}"
                )

        boundary_weights = {
            "model.embed_tokens": self.model.embed_tokens.weight,
            "lm_head": self.lm_head.weight,
        }
        for name, weight in boundary_weights.items():
            if weight.dtype != torch.float16:
                errors.append(f"{name}: expected float16 weight, got {weight.dtype}")

        for name, module in self.named_modules():
            if isinstance(module, (OnyxRMSNorm, OnyxOffsetRMSNorm)):
                if module.weight.dtype != torch.float16:
                    errors.append(
                        f"{name}: expected float16 norm weight, got {module.weight.dtype}"
                    )

        if errors:
            raise RuntimeError(
                "Onyx online-FP8 weight validation failed:\n- "
                + "\n- ".join(errors)
            )

        logger.info(
            "Onyx online-FP8 validation passed: 260 decoder linears use "
            "E4M3 weights with finite per-tensor scales; embedding, LM head, "
            "and norms remain FP16"
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: Optional[torch.Tensor] = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> LogitsProcessorOutput | PPProxyTensors:
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors,
        )

        if not self.pp_group.is_last_rank:
            return hidden_states

        return self.logits_processor(
            input_ids,
            hidden_states,
            self.lm_head,
            forward_batch,
        )

    @property
    def start_layer(self) -> int:
        return self.model.start_layer

    @property
    def end_layer(self) -> int:
        return self.model.end_layer

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def get_embed_and_head(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.model.embed_tokens.weight, self.lm_head.weight

    def set_embed_and_head(self, embed: torch.Tensor, head: torch.Tensor) -> None:
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        device_module = torch.get_device_module()
        device_module.synchronize()
        empty_device_cache(device_module)

    def get_attention_sliding_window_size(self) -> int:
        pattern = getattr(self.config, "sliding_window_pattern", None)
        if pattern is not None:
            return max(pattern) - 1
        return self.config.sliding_window - 1

    def dtype(self) -> torch.dtype:
        return next(self.parameters()).dtype

    @torch.no_grad()
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()

        for name, loaded_weight in weights:
            if name.endswith(("rotary_emb.freqs", "rotary_emb.inv_freq")):
                continue
            if name.startswith(ONYX_VISION_WEIGHT_PREFIXES):
                continue

            layer_id = get_layer_id(name)
            if layer_id is not None and not (
                self.model.start_layer <= layer_id < self.model.end_layer
            ):
                continue
            if not self.pp_group.is_first_rank and name.startswith(
                ("model.embed_tokens.", "model.embed_norm.")
            ):
                continue
            if not self.pp_group.is_last_rank and name.startswith(
                ("model.norm.", "lm_head.")
            ):
                continue

            matched = False
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                mapped_name = name.replace(weight_name, param_name)
                if mapped_name.endswith(".bias") and mapped_name not in params_dict:
                    matched = True
                    break
                if mapped_name not in params_dict:
                    raise KeyError(
                        f"Onyx checkpoint parameter {name} maps to unknown {mapped_name}"
                    )
                param = params_dict[mapped_name]
                weight_loader = getattr(param, "weight_loader")
                weight_loader(param, loaded_weight, shard_id)
                loaded_params.add(mapped_name)
                matched = True
                break
            if matched:
                continue

            if name.endswith(".bias") and name not in params_dict:
                continue
            if name not in params_dict:
                raise KeyError(f"Unexpected Onyx checkpoint parameter: {name}")
            param = params_dict[name]
            weight_loader = getattr(param, "weight_loader", default_weight_loader)
            weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


EntryClass = OnyxForCausalLM
