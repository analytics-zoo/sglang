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
from sglang.srt.function_call.core_types import ToolCallParseError
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

    def test_user_recipient_without_protocol_markers(self):
        text = " to=userParis is the capital of France."
        result = OnyxDetector().detect_and_parse(text, self.tools)
        self.assertEqual(result.normal_text, "Paris is the capital of France.")
        self.assertEqual(result.calls, [])

    def test_self_recipient_is_suppressed(self):
        result = OnyxDetector().detect_and_parse(
            " to=self<|message|>hidden reasoning<|eot|>", self.tools
        )
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_self_eom_continues_to_tool_recipient(self):
        result = OnyxDetector().detect_and_parse(
            " to=self<|message|>hidden reasoning<|eom|>"
            '<|start|>assistant to=get_weather<|message|>{"city":"Paris"}',
            self.tools,
        )
        self.assertEqual(result.normal_text, "")
        self.assertEqual(len(result.calls), 1)
        self.assertEqual(result.calls[0].name, "get_weather")
        self.assertEqual(json.loads(result.calls[0].parameters), {"city": "Paris"})

    def test_multiple_self_eom_frames_continue_to_user_recipient(self):
        result = OnyxDetector().detect_and_parse(
            " to=self<|message|>plan one<|eom|>"
            "<|start|>assistant to=self<|message|>plan two<|eom|>"
            "<|start|>assistant to=user<|message|>Done.<|eot|>",
            self.tools,
        )
        self.assertEqual(result.normal_text, "Done.")
        self.assertEqual(result.calls, [])

    def test_unknown_tool_recipient_is_rejected(self):
        with self.assertRaisesRegex(
            ToolCallParseError, "unknown tool recipient: missing"
        ):
            OnyxDetector().detect_and_parse(
                ' to=missing<|message|>{"value": 1}<|eot|>', self.tools
            )

    def test_malformed_start_marker_is_suppressed(self):
        result = OnyxDetector().detect_and_parse(
            "<|start|>assistant malformed output", self.tools
        )
        self.assertEqual(result.normal_text, "")
        self.assertEqual(result.calls, [])

    def test_malformed_embedded_transition_suppresses_hidden_payload(self):
        result = OnyxDetector().detect_and_parse(
            "Visible preamble.<|start|>assistant "
            "to=self<|message|>hidden reasoning<|eot|>",
            self.tools,
        )
        self.assertEqual(result.normal_text, "Visible preamble.")
        self.assertEqual(result.calls, [])

    def test_user_recipient_with_protocol_markers(self):
        result = OnyxDetector().detect_and_parse(
            " to=user<|message|>It is 18 degrees.<|eot|>", self.tools
        )
        self.assertEqual(result.normal_text, "It is 18 degrees.")
        self.assertEqual(result.calls, [])

    def test_bare_user_message_with_protocol_markers(self):
        result = OnyxDetector().detect_and_parse(
            "<|message|>It is 18 degrees.<|eot|>", self.tools
        )
        self.assertEqual(result.normal_text, "It is 18 degrees.")
        self.assertEqual(result.calls, [])

    def test_self_eom_continues_to_bare_user_message(self):
        result = OnyxDetector().detect_and_parse(
            " to=self<|message|>hidden reasoning<|eom|>"
            "<|start|>assistant<|message|>Done.<|eot|>",
            self.tools,
        )
        self.assertEqual(result.normal_text, "Done.")
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

    def test_multiple_recipient_calls_are_rejected(self):
        text = (
            ' to=get_weather<|message|>{"city":"Paris"}<|eot|>'
            'assistant to=get_weather<|message|>{"city":"Tokyo"}<|eot|>'
            "<|start|>assistant to=user<|message|>"
        )

        with self.assertRaisesRegex(
            ToolCallParseError, "at most one tool call"
        ):
            OnyxDetector().detect_and_parse(text, self.tools)

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

    def test_streaming_bare_user_message(self):
        detector = OnyxDetector()
        first = detector.parse_streaming_increment("<|mess", self.tools)
        second = detector.parse_streaming_increment("age|>It is ", self.tools)
        third = detector.parse_streaming_increment("18 degrees.<|eot|>", self.tools)
        self.assertEqual(first.normal_text, "")
        self.assertEqual(second.normal_text, "It is ")
        self.assertEqual(third.normal_text, "18 degrees.")
        self.assertEqual(first.calls + second.calls + third.calls, [])

    def test_streaming_self_recipient_is_suppressed(self):
        detector = OnyxDetector()
        first = detector.parse_streaming_increment(" to=self<|mess", self.tools)
        second = detector.parse_streaming_increment(
            "age|>hidden reasoning<|eot|>", self.tools
        )
        third = detector.parse_streaming_increment("still hidden", self.tools)
        self.assertEqual(first.normal_text + second.normal_text + third.normal_text, "")
        self.assertEqual(first.calls + second.calls + third.calls, [])

    def test_streaming_self_eom_continues_to_tool_recipient(self):
        detector = OnyxDetector()
        first = detector.parse_streaming_increment(
            " to=self<|message|>hidden reasoning<|eo", self.tools
        )
        second = detector.parse_streaming_increment(
            "m|><|start|>assistant to=get_weather<|message|>{", self.tools
        )
        third = detector.parse_streaming_increment(
            '"city":"Paris"}<|eom|>', self.tools
        )

        calls = first.calls + second.calls + third.calls
        self.assertEqual(first.normal_text + second.normal_text + third.normal_text, "")
        self.assertEqual([call.name for call in calls if call.name], ["get_weather"])
        self.assertEqual(
            "".join(call.parameters for call in calls if call.parameters),
            '{"city":"Paris"}',
        )

    def test_streaming_split_protocol_marker_does_not_leak(self):
        detector = OnyxDetector()
        first = detector.parse_streaming_increment("Visible preamble.<|", self.tools)
        second = detector.parse_streaming_increment(
            "start|>assistant to=self<|message|>hidden<|eot|>", self.tools
        )
        self.assertEqual(first.normal_text, "Visible preamble.")
        self.assertEqual(second.normal_text, "")
        self.assertEqual(first.calls + second.calls, [])

    def test_parser_sanitizes_protocol_without_tools(self):
        parser = FunctionCallParser([], "onyx")
        text, calls = parser.parse_non_stream(
            " to=self<|message|>hidden reasoning<|eot|>"
        )
        self.assertEqual(text, "")
        self.assertEqual(calls, [])

    def test_streaming_unmarked_user_recipient_without_tools(self):
        parser = FunctionCallParser([], "onyx")
        first_text, first_calls = parser.parse_stream_chunk(" to=u")
        second_text, second_calls = parser.parse_stream_chunk("serHello")
        third_text, third_calls = parser.parse_stream_chunk("!")
        self.assertEqual(first_text + second_text + third_text, "Hello!")
        self.assertEqual(first_calls + second_calls + third_calls, [])

    def test_streaming_marked_user_recipient_without_tools(self):
        parser = FunctionCallParser([], "onyx")
        first_text, first_calls = parser.parse_stream_chunk(
            " to=user<|message|>Hello"
        )
        second_text, second_calls = parser.parse_stream_chunk("!<|eot|>")
        self.assertEqual(first_text + second_text, "Hello!")
        self.assertEqual(first_calls + second_calls, [])

    def test_streaming_finalize_validates_without_end_marker(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="select_color",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {
                            "color": {"type": "string", "enum": ["red", "blue"]}
                        },
                        "required": ["color"],
                        "additionalProperties": False,
                    },
                ),
            )
        ]
        parser = FunctionCallParser(tools, "onyx")
        parser.parse_stream_chunk(
            ' to=select_color<|message|>{"color":"green"}'
        )
        with self.assertRaisesRegex(ToolCallParseError, "violate its schema"):
            parser.finalize_stream()

    def test_streaming_user_suppresses_split_internal_transition(self):
        parser = FunctionCallParser(self.tools, "onyx")
        first_text, first_calls = parser.parse_stream_chunk(
            " to=user<|message|>Visible<|st"
        )
        second_text, second_calls = parser.parse_stream_chunk(
            "art|>assistant to=self<|message|>hidden"
        )
        third_text, third_calls = parser.parse_stream_chunk("MORE")
        self.assertEqual(first_text + second_text + third_text, "Visible")
        self.assertEqual(first_calls + second_calls + third_calls, [])

    def test_streaming_finalize_flushes_safe_trailing_less_than(self):
        parser = FunctionCallParser([], "onyx")
        text, calls = parser.parse_stream_chunk("2 <")
        final_text = parser.finalize_stream()
        self.assertEqual(text + final_text, "2 <")
        self.assertEqual(calls, [])

    def test_streaming_multiple_recipient_calls_are_rejected(self):
        detector = OnyxDetector()
        first = detector.parse_streaming_increment(
            ' to=get_weather<|message|>{"city":"Paris"}<|eot|><|start|>assi',
            self.tools,
        )
        self.assertEqual([call.name for call in first.calls if call.name], ["get_weather"])
        with self.assertRaisesRegex(
            ToolCallParseError, "at most one tool call"
        ):
            detector.parse_streaming_increment(
                'stant to=get_weather<|message|>{"city":"Tokyo"}<|eot|>',
                self.tools,
            )

    def test_strict_arguments_are_schema_validated(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="select_color",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {
                            "color": {"type": "string", "enum": ["red", "blue"]}
                        },
                        "required": ["color"],
                        "additionalProperties": False,
                    },
                ),
            )
        ]
        with self.assertRaisesRegex(ToolCallParseError, "violate its schema"):
            OnyxDetector().detect_and_parse(
                ' to=select_color<|message|>{"color":"green","note":"x"}<|eot|>',
                tools,
            )

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

    def test_auto_uses_complete_recipient_union(self):
        self.tools[0].function.strict = True
        self.tools[0].function.parameters = {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        }

        parser = FunctionCallParser(self.tools, "onyx")
        constraint_type, constraint = parser.get_structure_constraint(
            "auto", parallel_tool_calls=False
        )

        self.assertEqual(constraint_type, "structural_tag")
        structural_tag = constraint.model_dump()
        format_ = structural_tag["format"]
        self.assertEqual(format_["type"], "grammar")
        grammar = format_["grammar"]
        self.assertIn("onyx_self_sequence{0, 8}", grammar)
        self.assertIn('" to=self<|message|>"', grammar)
        self.assertIn('"<|eom|><|start|>assistant"', grammar)
        self.assertIn('" to=user<|message|>"', grammar)
        self.assertIn('"<|message|>"', grammar)
        self.assertIn('"<|eot|>"', grammar)
        self.assertIn('" to=get_weather<|message|>"', grammar)
        self.assertIn('"<|eom|>"', grammar)
        self.assertIn("onyx_final_onyx_tool_0_root", grammar)
        self.assertIn('"\\"city\\""', grammar)

    def test_auto_schema_grammar_uses_structural_tag_repeat_syntax(self):
        self.tools[0].function.strict = True
        self.tools[0].function.parameters = {
            "type": "object",
            "properties": {
                "city": {"type": "string", "minLength": 1},
            },
            "required": ["city"],
            "additionalProperties": False,
        }

        parser = FunctionCallParser(self.tools, "onyx")
        _, constraint = parser.get_structure_constraint(
            "auto", parallel_tool_calls=False
        )
        grammar = constraint.model_dump()["format"]["grammar"]

        self.assertIn("{1,}", grammar)
        self.assertNotIn("{1, -1}", grammar)

    def test_auto_tool_union_namespaces_each_schema(self):
        tools = [
            Tool(
                type="function",
                function=Function(
                    name="first",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {"root": {"type": "string"}},
                        "required": ["root"],
                        "additionalProperties": False,
                    },
                ),
            ),
            Tool(
                type="function",
                function=Function(
                    name="second",
                    strict=True,
                    parameters={
                        "type": "object",
                        "properties": {"value": {"type": "integer"}},
                        "required": ["value"],
                        "additionalProperties": False,
                    },
                ),
            ),
        ]

        _, constraint = FunctionCallParser(tools, "onyx").get_structure_constraint(
            "auto", parallel_tool_calls=False
        )
        grammar = constraint.model_dump()["format"]["grammar"]

        self.assertIn("onyx_final_onyx_tool_branch_0", grammar)
        self.assertIn("onyx_final_onyx_tool_branch_1", grammar)
        self.assertIn("onyx_final_onyx_tool_0_root", grammar)
        self.assertIn("onyx_final_onyx_tool_1_root", grammar)
        self.assertIn('" to=first<|message|>"', grammar)
        self.assertIn('" to=second<|message|>"', grammar)
        # Literal schema property names must not be rewritten as rule names.
        self.assertIn('"\\"root\\""', grammar)

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

    def test_template_renders_single_call_generation_prompt(self):
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

        self.assertIn("at most one short private self-recipient decision", rendered)
        self.assertIn("only the minimum functions required", rendered)
        self.assertIn("A successful tool result completes that action", rendered)
        self.assertIn("answer the user and stop calling functions", rendered)
        self.assertIn("do not repeat the same function with identical arguments", rendered)
        self.assertIn("Do not ask the user to repeat values", rendered)
        self.assertIn("supplied access token", rendered)
        self.assertIn("listed status function directly provides it", rendered)
        self.assertIn("Use a listed calculation or conversion function", rendered)
        self.assertIn("Function arguments must start with { and end with }", rendered)
        self.assertIn("never use XML for function arguments", rendered)
        self.assertIn("argument rule does not apply to normal user replies", rendered)
        self.assertIn("Call at most one function", rendered)
        self.assertNotIn("multiple independent calls", rendered)
        self.assertTrue(rendered.endswith("<|start|>assistant"))


if __name__ == "__main__":
    unittest.main()
