"""Config-time override declarations for gemma4.

Architectures: Gemma4ForCausalLM, Gemma4ForConditionalGeneration, Gemma4UnifiedForConditionalGeneration.
"""

import logging
from typing import Any, Dict

from sglang.srt.arg_groups.model_override_base import (
    _register_for,
    is_attention_backend_not_set,
    model_config_of,
    resolving_view,
)
from sglang.srt.runtime_context import get_platform

logger = logging.getLogger(__name__)


def is_gemma4_modelopt_fp4_moe(server_args: Any) -> bool:
    """Detect Gemma4 MoE with modelopt_fp4 quantization.
    
    SM10X trtllm_mha has accuracy issues with this combination, requiring
    fallback to triton attention backend.
    """
    model_config = model_config_of(server_args)
    return (
        model_config.quantization == "modelopt_fp4"
        and getattr(model_config.hf_text_config, "enable_moe_block", False)
    )


@_register_for(
    "Gemma4ForConditionalGeneration",
    "Gemma4ForCausalLM",
    "Gemma4UnifiedForConditionalGeneration",
)
def _gemma4_overrides(server_args: Any, hf_config: Any) -> dict:
    cfg = resolving_view(server_args)
    overrides: Dict[str, Any] = {}
    
    # SM10X trtllm_mha has accuracy issues with MoE + modelopt_fp4
    use_trtllm_mha = get_platform().is_sm100 and not is_gemma4_modelopt_fp4_moe(server_args)
    default_attention_backend = "trtllm_mha" if use_trtllm_mha else "triton"
    
    if is_attention_backend_not_set(cfg):
        if get_platform().is_sm100 and is_gemma4_modelopt_fp4_moe(server_args):
            logger.info(
                "Gemma4 MoE with modelopt_fp4 detected on SM100: "
                "falling back to triton attention backend (trtllm_mha accuracy issue)"
            )
        logger.info(
            f"Use {default_attention_backend} as default attention backend for Gemma4"
        )
        overrides["attention_backend"] = default_attention_backend
    # If only one split backend is set, keep the other side on a
    # Gemma4-compatible fallback instead of letting generic backend selection
    # choose an unsupported backend later.
    elif cfg.attention_backend is None:
        overrides["attention_backend"] = default_attention_backend
    if get_platform().is_sm100 and cfg.moe_runner_backend == "auto":
        if model_config_of(server_args).quantization == "modelopt_fp4":
            overrides["quantization"] = "modelopt_fp4"
            overrides["moe_runner_backend"] = "flashinfer_trtllm"
            logger.info(
                "Use flashinfer_trtllm as MoE runner backend on "
                "SM100 for Gemma-4 (modelopt_fp4)"
            )
    return overrides
