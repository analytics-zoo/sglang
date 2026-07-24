import os
from pathlib import Path
from types import SimpleNamespace

import bfcl_eval._llm_response_generation as generation
from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)


REGISTRY = os.environ.get(
    "ONYX_BFCL_REGISTRY", "onyx-complete-grammar-full"
)
IDS_FILE = os.environ.get("ONYX_BFCL_IDS_FILE")
NUM_THREADS = int(os.environ.get("ONYX_BFCL_NUM_THREADS", "4"))

if IDS_FILE:
    generation.TEST_IDS_TO_GENERATE_PATH = Path(IDS_FILE)
MODEL_CONFIG_MAPPING[REGISTRY] = ModelConfig(
    model_name="/llm/workspace/model/onyx-hf",
    display_name="Onyx Complete Grammar Full",
    url="",
    org="Intel",
    license="",
    model_handler=OpenAICompletionsHandler,
    input_price=None,
    output_price=None,
    is_fc_model=True,
    underscore_to_dot=False,
)

generation.main(
    SimpleNamespace(
        model=[REGISTRY],
        test_category=["multi_turn_base"],
        temperature=0.001,
        include_input_log=False,
        exclude_state_log=False,
        num_threads=NUM_THREADS,
        num_samples=None,
        num_gpus=1,
        backend="sglang",
        gpu_memory_utilization=0.9,
        result_dir=None,
        run_ids=bool(IDS_FILE),
        allow_overwrite=True,
        skip_server_setup=True,
        local_model_path=None,
    )
)
