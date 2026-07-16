import json
import unittest

from sglang.srt.entrypoints.openai.protocol import Function, Tool
from sglang.srt.function_call.onyx_detector import OnyxDetector
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class TestOnyxDetector(unittest.TestCase):
    def setUp(self):
        self.tools = [
            Tool(
                type="function",
                function=Function(
                    name="get_weather",
                    description="Get weather",
                    parameters={"type": "object"},
                ),
            )
        ]

    def test_complete_recipient_call(self):
        result = OnyxDetector().detect_and_parse(
            ' to=get_weather{"city": "Paris"}', self.tools
        )
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})

    def test_call_with_protocol_markers(self):
        result = OnyxDetector().detect_and_parse(
            ' to=get_weather<|message|>{"city": "Paris"}<|eom|>', self.tools
        )
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})

    def test_unknown_recipient_is_normal_text(self):
        text = " to=userParis is the capital of France."
        result = OnyxDetector().detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, text)
        self.assertEqual(result.calls, [])

    def test_streaming_recipient_call(self):
        detector = OnyxDetector()
        first = detector.parse_streaming_increment(" to=get_", self.tools)
        second = detector.parse_streaming_increment(
            'weather{"city": "Paris"}', self.tools
        )
        self.assertEqual(first.calls, [])
        self.assertEqual(second.calls[0].name, "get_weather")

    def test_structural_constraints_are_not_advertised(self):
        self.assertFalse(OnyxDetector().supports_structural_tag())


if __name__ == "__main__":
    unittest.main()
