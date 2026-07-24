import unittest
from array import array
from unittest.mock import Mock

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.constrained.base_grammar_backend import (
    _filter_onyx_grammar_stop_tokens,
)
from sglang.srt.sampling.sampling_params import (
    ONYX_PROTOCOL_TOKEN_IDS_KEY,
    SamplingParams,
)


class TestConditionalEosStop(unittest.TestCase):
    EOM = 200007
    EOT = 200008
    START = 200022
    MESSAGE = 200023
    SELF_PREFIX = [101, 102]
    TOOL_PREFIX = [103, 104]

    def _make_request(
        self, output_ids, *, with_onyx_protocol=True, ignore_eos=False
    ):
        custom_params = None
        if with_onyx_protocol:
            custom_params = {
                ONYX_PROTOCOL_TOKEN_IDS_KEY: {
                    "eom": self.EOM,
                    "eot": self.EOT,
                    "start": self.START,
                    "message": self.MESSAGE,
                    "self_recipient_prefixes": [self.SELF_PREFIX],
                }
            }
        sampling_params = SamplingParams(
            ignore_eos=ignore_eos,
            custom_params=custom_params,
        )
        request = Req(
            rid="test",
            origin_input_text="",
            origin_input_ids=array("q", [1]),
            sampling_params=sampling_params,
            eos_token_ids={200001, self.EOM, self.EOT},
            vocab_size=202048,
        )
        request.output_ids = array("q", output_ids)
        request.tokenizer = Mock(
            eos_token_id=200001,
            additional_stop_token_ids=None,
        )
        return request

    def test_self_eom_is_a_message_separator(self):
        request = self._make_request(
            [*self.SELF_PREFIX, self.MESSAGE, 55, self.EOM]
        )

        self.assertFalse(request._check_token_based_finish([self.EOM]))
        self.assertIsNone(request.finished_reason)

    def test_tool_eom_is_terminal(self):
        request = self._make_request(
            [*self.TOOL_PREFIX, self.MESSAGE, 56, self.EOM]
        )

        self.assertTrue(request._check_token_based_finish([self.EOM]))
        self.assertEqual(request.finished_reason.to_json()["matched"], self.EOM)

    def test_start_after_self_eom_is_a_message_separator(self):
        request = self._make_request(
            [*self.SELF_PREFIX, self.MESSAGE, 55, self.EOM, self.START]
        )
        request.tokenizer.additional_stop_token_ids = {self.START}

        self.assertFalse(request._check_token_based_finish([self.START]))
        self.assertIsNone(request.finished_reason)

    def test_standalone_start_is_terminal(self):
        request = self._make_request([55, self.START])
        request.tokenizer.additional_stop_token_ids = {self.START}

        self.assertTrue(request._check_token_based_finish([self.START]))
        self.assertEqual(request.finished_reason.to_json()["matched"], self.START)

    def test_self_then_tool_stops_at_tool_eom(self):
        request = self._make_request(
            [
                *self.SELF_PREFIX,
                self.MESSAGE,
                55,
                self.EOM,
                200022,
                *self.TOOL_PREFIX,
                self.MESSAGE,
                56,
                self.EOM,
            ]
        )

        self.assertTrue(request._check_token_based_finish([self.EOM]))
        self.assertEqual(request.finished_reason.to_json()["matched"], self.EOM)

    def test_eot_is_terminal(self):
        request = self._make_request(
            [*self.SELF_PREFIX, self.MESSAGE, 55, self.EOT]
        )

        self.assertTrue(request._check_token_based_finish([self.EOT]))
        self.assertEqual(request.finished_reason.to_json()["matched"], self.EOT)

    def test_malformed_eom_is_terminal(self):
        request = self._make_request([55, self.EOM])

        self.assertTrue(request._check_token_based_finish([self.EOM]))

    def test_onyx_protocol_overrides_client_ignore_eos(self):
        request = self._make_request(
            [*self.TOOL_PREFIX, self.MESSAGE, 56, self.EOM],
            ignore_eos=True,
        )

        self.assertTrue(request._check_token_based_finish([self.EOM]))

    def test_preserves_unconditional_ignore_eos(self):
        request = self._make_request(
            [55, self.EOT],
            with_onyx_protocol=False,
            ignore_eos=True,
        )

        self.assertFalse(request._check_token_based_finish([self.EOT]))
        self.assertIsNone(request.finished_reason)

    def test_onyx_grammar_does_not_own_message_separator_stops(self):
        server_args = Mock(tool_call_parser="onyx")
        tokenizer = Mock()
        tokenizer.get_vocab.return_value = {
            "<|eom|>": self.EOM,
            "<|eot|>": self.EOT,
            "<|start|>": self.START,
        }

        filtered = _filter_onyx_grammar_stop_tokens(
            server_args,
            tokenizer,
            [200001, self.EOM, self.EOT, self.START],
        )

        self.assertEqual(filtered, [200001])


if __name__ == "__main__":
    unittest.main()
