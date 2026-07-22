import unittest
from array import array
from unittest.mock import Mock

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import (
    STOP_ON_EOS_OUTPUT_PREFIX_KEY,
    SamplingParams,
)


class TestConditionalEosStop(unittest.TestCase):
    def _make_request(self, decoded_output: str, custom_params=None):
        sampling_params = SamplingParams(
            ignore_eos=True,
            custom_params=custom_params,
        )
        request = Req(
            rid="test",
            origin_input_text="",
            origin_input_ids=array("q", [1]),
            sampling_params=sampling_params,
            eos_token_ids={200008},
            vocab_size=202048,
        )
        request.output_ids = array("q", [10, 11, 200008])
        request.tokenizer = Mock(
            eos_token_id=200008,
            additional_stop_token_ids=None,
        )
        request.tokenizer.decode.return_value = decoded_output
        return request

    def test_honors_eos_for_matching_output_prefix(self):
        request = self._make_request(
            " to=user<|message|>A long answer.<|eot|>",
            {STOP_ON_EOS_OUTPUT_PREFIX_KEY: "to=user<|message|>"},
        )

        self.assertTrue(request._check_token_based_finish([200008]))
        self.assertEqual(request.finished_len, 3)
        self.assertEqual(request.finished_reason.to_json()["matched"], 200008)

    def test_ignores_eos_for_tool_call_output(self):
        request = self._make_request(
            ' to=get_weather<|message|>{"city":"Paris"}<|eot|>',
            {STOP_ON_EOS_OUTPUT_PREFIX_KEY: "to=user<|message|>"},
        )

        self.assertFalse(request._check_token_based_finish([200008]))
        self.assertIsNone(request.finished_reason)

    def test_preserves_unconditional_ignore_eos(self):
        request = self._make_request(
            " to=user<|message|>A long answer.<|eot|>",
        )

        self.assertFalse(request._check_token_based_finish([200008]))
        self.assertIsNone(request.finished_reason)


if __name__ == "__main__":
    unittest.main()
