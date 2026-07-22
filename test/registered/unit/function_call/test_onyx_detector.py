import json
import unittest
from pathlib import Path

from jinja2 import Environment, StrictUndefined
from sglang.srt.entrypoints.openai.protocol import (
    Function,
    Tool,
    ToolChoice,
    ToolChoiceFuncName,
)
from sglang.srt.function_call.function_call_parser import FunctionCallParser
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
        self.assertEqual(result.normal_text, "Paris is the capital of France.")
        self.assertEqual(result.calls, [])

    def test_user_recipient_with_protocol_markers(self):
        result = OnyxDetector().detect_and_parse(
            " to=user<|message|>It is 18 degrees.<|eot|>", self.tools
        )
        self.assertEqual(result.normal_text, "It is 18 degrees.")
        self.assertEqual(result.calls, [])

    def test_parser_preserves_cleaned_user_recipient(self):
        parser = FunctionCallParser(self.tools, "onyx")
        text, calls = parser.parse_non_stream(
            " to=user<|message|>It is 18 degrees.<|eot|>"
        )
        self.assertEqual(text, "It is 18 degrees.")
        self.assertEqual(calls, [])

    def test_parser_preserves_empty_user_recipient(self):
        parser = FunctionCallParser(self.tools, "onyx")
        text, calls = parser.parse_non_stream(
            " to=user<|message|><|eot|>"
        )
        self.assertEqual(text, "")
        self.assertEqual(calls, [])

    def test_multiple_recipient_calls_use_response_ordinals(self):
        text = (
            ' to=get_weather<|message|>{"city":"Paris"}<|eot|>'
            'assistant to=get_weather<|message|>{"city":"Tokyo"}<|eot|>'
            "<|start|>assistant to=user<|message|>"
        )

        result = OnyxDetector().detect_and_parse(text, self.tools)

        self.assertEqual([call.tool_index for call in result.calls], [0, 1])
        self.assertEqual(
            [json.loads(call.parameters) for call in result.calls],
            [{"city": "Paris"}, {"city": "Tokyo"}],
        )
        self.assertEqual(result.normal_text, "")

    def test_streaming_recipient_call(self):
        detector = OnyxDetector()
        first = detector.parse_streaming_increment(" to=get_", self.tools)
        second = detector.parse_streaming_increment(
            'weather<|message|>{"city":', self.tools
        )
        third = detector.parse_streaming_increment(
            ' "Paris"}<|eom|>', self.tools
        )
        self.assertEqual(first.calls, [])
        self.assertEqual(second.calls[0].name, "get_weather")
        self.assertEqual(second.calls[0].parameters, "")
        self.assertEqual(second.calls[1].parameters, '{"city":')
        self.assertEqual(third.calls[0].parameters, ' "Paris"}')

    def test_streaming_user_recipient(self):
        detector = OnyxDetector()
        first = detector.parse_streaming_increment(
            " to=user<|message|>It is ", self.tools
        )
        second = detector.parse_streaming_increment("18 degrees.<|eot|>", self.tools)
        self.assertEqual(first.normal_text, "It is ")
        self.assertEqual(second.normal_text, "18 degrees.")
        self.assertEqual(first.calls + second.calls, [])

    def test_streaming_multiple_recipient_calls(self):
        detector = OnyxDetector()
        chunks = [
            ' to=get_weather<|message|>{"city":"Paris"}<|eot|><|start|>assi',
            'stant to=get_weather<|message|>{"city":"Tokyo"}<|eot|>',
            "<|start|>assistant to=user<|message|>",
        ]
        calls = []
        normal_text = ""

        for chunk in chunks:
            result = detector.parse_streaming_increment(chunk, self.tools)
            calls.extend(result.calls)
            normal_text += result.normal_text

        names = [call for call in calls if call.name]
        self.assertEqual(
            [(call.tool_index, call.name) for call in names],
            [(0, "get_weather"), (1, "get_weather")],
        )
        parameters = {0: "", 1: ""}
        for call in calls:
            parameters[call.tool_index] += call.parameters
        self.assertEqual(json.loads(parameters[0]), {"city": "Paris"})
        self.assertEqual(json.loads(parameters[1]), {"city": "Tokyo"})
        self.assertEqual(normal_text, "")

    def test_streaming_suppresses_generated_tool_result_header(self):
        detector = OnyxDetector()

        first = detector.parse_streaming_increment(
            ' to=get_weather<|message|>{"city":"Paris"}<|eot|><|start|>to',
            self.tools,
        )
        second = detector.parse_streaming_increment("ol get_weather", self.tools)

        self.assertEqual([call.name for call in first.calls if call.name], ["get_weather"])
        self.assertEqual(first.normal_text + second.normal_text, "")
        self.assertEqual(second.calls, [])

    def test_structure_info_uses_native_recipient_protocol(self):
        info = OnyxDetector().structure_info()("get_weather")
        self.assertEqual(info.begin, " to=get_weather<|message|>")
        self.assertEqual(info.end, "<|eom|>")
        self.assertEqual(info.trigger, " to=")
        self.assertTrue(OnyxDetector().supports_structural_tag())

    def test_required_uses_native_structural_constraint(self):
        parser = FunctionCallParser(self.tools, "onyx")
        constraint_type, constraint = parser.get_structure_constraint(
            "required", parallel_tool_calls=False
        )
        self.assertEqual(constraint_type, "structural_tag")
        structure = constraint.structures[0]
        self.assertEqual(structure.begin, " to=get_weather<|message|>")
        self.assertEqual(structure.end, "<|eom|>")

    def test_named_choice_constrains_only_the_selected_recipient(self):
        tools = self.tools + [
            Tool(
                type="function",
                function=Function(
                    name="search",
                    description="Search",
                    parameters={"type": "object"},
                ),
            )
        ]
        parser = FunctionCallParser(tools, "onyx")
        _, constraint = parser.get_structure_constraint(
            ToolChoice(
                type="function",
                function=ToolChoiceFuncName(name="search"),
            ),
            parallel_tool_calls=False,
        )

        self.assertEqual(len(constraint.structures), 1)
        self.assertEqual(
            constraint.structures[0].begin, " to=search<|message|>"
        )

    def test_template_renders_openai_tool_call_round_trip(self):
        template_path = (
            Path(__file__).resolve().parents[4]
            / "benchmark"
            / "onyx"
            / "onyx_tool_chat_template.jinja"
        )
        environment = Environment(undefined=StrictUndefined)

        def raise_exception(message):
            raise ValueError(message)

        environment.globals["raise_exception"] = raise_exception
        template = environment.from_string(template_path.read_text())
        rendered = template.render(
            bos_token="<s>",
            tools=[tool.model_dump() for tool in self.tools],
            messages=[
                {"role": "system", "content": "Be helpful."},
                {"role": "user", "content": "Weather in Paris?"},
                {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_weather",
                            "type": "function",
                            "function": {
                                "name": "get_weather",
                                "arguments": {"city": "Paris"},
                            },
                        }
                    ],
                },
                {
                    "role": "tool",
                    "tool_call_id": "call_weather",
                    "content": '{"temperature": 18}',
                },
                {"role": "assistant", "content": "It is 18 degrees."},
            ],
            add_generation_prompt=False,
        )

        self.assertIn(
            '<|start|>assistant to=get_weather<|message|>{"city": "Paris"}<|eom|>',
            rendered,
        )
        self.assertIn(
            '<|start|>tool get_weather<|message|>{"temperature": 18}<|eot|>',
            rendered,
        )
        self.assertTrue(
            rendered.endswith(
                "<|start|>assistant<|message|>It is 18 degrees.<|eot|>"
            )
        )

    def test_template_renders_auto_parallel_native_generation_prompt(self):
        template_path = (
            Path(__file__).resolve().parents[4]
            / "benchmark"
            / "onyx"
            / "onyx_tool_chat_template.jinja"
        )
        environment = Environment(undefined=StrictUndefined)
        template = environment.from_string(template_path.read_text())

        rendered = template.render(
            bos_token="<s>",
            tools=[tool.model_dump() for tool in self.tools],
            messages=[{"role": "user", "content": "Weather in Paris and Tokyo?"}],
            add_generation_prompt=True,
            tool_choice="auto",
            parallel_tool_calls=True,
        )

        self.assertIn("Decide whether functions are actually needed.", rendered)
        self.assertIn("one recipient message per call", rendered)
        self.assertTrue(rendered.endswith("<|start|>assistant"))


if __name__ == "__main__":
    unittest.main()
