# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Union

from sglang.srt.managers.multimodal_processor import (
    BaseMultimodalProcessor as SGLangBaseProcessor,
)
from sglang.srt.managers.schedule_batch import MultimodalProcessorOutput
from sglang.srt.models.onyx import OnyxForCausalLM
from sglang.srt.multimodal.processors.base_processor import MultimodalSpecialTokens


class OnyxSGLangProcessor(SGLangBaseProcessor):
    models = [OnyxForCausalLM]

    def __init__(self, hf_config, server_args, _processor, *args, **kwargs):
        super().__init__(
            hf_config, server_args, _processor, *args, **kwargs
        )
        self.mm_tokens = MultimodalSpecialTokens(
            image_token="<|image|>",
            image_token_id=hf_config.patch_token_id,
            video_token="<|video|>",
            video_token_id=hf_config.video_token_id,
        ).build(_processor)

    async def process_mm_data_async(
        self,
        image_data: Optional[List[Union[str, bytes, Dict]]] = None,
        audio_data: Optional[List[Union[str, bytes, Dict]]] = None,
        input_text: str = "",
        request_obj=None,
        *args,
        **kwargs,
    ):
        if audio_data:
            raise ValueError("Onyx does not support audio inputs")
        if request_obj is not None and request_obj.video_data:
            raise NotImplementedError(
                "Onyx video inputs are not supported by the SGLang path yet"
            )
        base_output = await self.load_mm_data(
            prompt=input_text,
            image_data=image_data,
            multimodal_tokens=self.mm_tokens,
        )
        mm_items, input_ids, _ = self.process_and_combine_mm_data(
            base_output, self.mm_tokens
        )
        return MultimodalProcessorOutput(
            input_ids=input_ids.tolist(),
            mm_items=mm_items,
            im_token_id=self.mm_tokens.image_token_id,
            video_token_id=self.mm_tokens.video_token_id,
        )
