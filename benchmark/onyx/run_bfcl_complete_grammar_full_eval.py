import os

from bfcl_eval.constants.model_config import MODEL_CONFIG_MAPPING, ModelConfig
from bfcl_eval.eval_checker.eval_runner import main
from bfcl_eval.model_handler.api_inference.openai_completion import (
    OpenAICompletionsHandler,
)


REGISTRY = os.environ.get(
    "ONYX_BFCL_REGISTRY", "onyx-complete-grammar-full"
)
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

main(
    model=[REGISTRY],
    test_categories=["multi_turn_base"],
    result_dir=None,
    score_dir=None,
    partial_eval=os.environ.get("ONYX_BFCL_PARTIAL_EVAL", "").lower()
    in {"1", "true", "yes"},
)
