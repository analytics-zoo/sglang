from pathlib import Path
from types import SimpleNamespace

import bfcl_eval._llm_response_generation as generation
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)


REGISTRY = "onyx-complete-grammar"
MODEL_CONFIG_MAPPING[REGISTRY] = ModelConfig(
    model_name="/llm/workspace/model/onyx-hf",
    display_name="Onyx Complete Grammar",
    url="",
    org="Intel",
    license="",
    model_handler=OpenAICompletionsHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=False,
)

generation.TEST_IDS_TO_GENERATE_PATH = Path(
    "/llm/workspace/sgl_gemma/sglang/benchmark/onyx/"
    "bfcl_neutral_prompt_ids.json"
)
generation.main(
    SimpleNamespace(
        model=[REGISTRY],
        test_category=["multi_turn_base"],
        temperature=0.001,
        include_input_log=False,
        exclude_state_log=False,
        num_threads=4,
        num_samples=None,
        num_gpus=1,
        backend="sglang",
        gpu_memory_utilization=0.9,
        result_dir=None,
        run_ids=True,
        allow_overwrite=True,
        skip_server_setup=True,
        local_model_path=None,
    )
)
