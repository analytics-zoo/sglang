import json
import re
from typing import List, Literal, Union

from jsonschema import Draft202012Validator

try:
    from xgrammar import Grammar, StructuralTag
except ImportError:
    Grammar = None
    StructuralTag = None

from sglang.srt.entrypoints.openai.protocol import Tool, ToolChoice
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StructureInfo,
    StreamingParseResult,
    ToolCallItem,
    ToolCallParseError,
    _GetInfoFunc,
)


class OnyxDetector(BaseFormatDetector):
    """Parse Onyx recipient tool calls: ``to=name{"arg": "value"}``."""

    _recipient = re.compile(r"^\s*to=(user|[A-Za-z_][A-Za-z0-9_.-]*)\s*")
    _sequence_header = re.compile(
        r"\s*(?:(?:<\|start\|>)?assistant\s+)?"
        r"to=(user|[A-Za-z_][A-Za-z0-9_.-]*)\s*(<\|message\|>|\{)"
    )
    _stream_header = re.compile(
        r"^\s*(?:(?:<\|start\|>)?assistant\s+)?"
        r"to=([A-Za-z_][A-Za-z0-9_.-]*)\s*(<\|message\|>|\{)"
    )
    _bare_user_header = re.compile(
        r"\s*(?:(?:<\|start\|>)?assistant\s*)?<\|message\|>"
    )
    _embedded_header = re.compile(
        r"(?:(?:<\|start\|>)?assistant\s+)?"
        r"to=[A-Za-z_][A-Za-z0-9_.-]*\s*<\|message\|>"
    )
    _end_tokens = ("<|eom|>", "<|eot|>")
    _protocol_tokens = ("<|start|>", "<|message|>", "<|eom|>", "<|eot|>")

    def __init__(self):
        super().__init__()
        self._stream_name: str | None = None
        self._stream_tool_index: int | None = None
        self._stream_is_user = False
        self._stream_is_self = False
        self._stream_user_visible = False
        self._stream_suppress_output = False

    def _strip_protocol_payload(self, payload: str) -> str:
        payload = payload.strip()
        if payload.startswith("<|message|>"):
            payload = payload[len("<|message|>") :].lstrip()
        for end_token in self._end_tokens:
            if payload.endswith(end_token):
                payload = payload[: -len(end_token)].rstrip()
        return payload

    def _sanitize_visible_text(self, text: str) -> str:
        positions = [
            position
            for token in self._protocol_tokens
            if (position := text.find(token)) != -1
        ]
        if match := self._embedded_header.search(text):
            positions.append(match.start())
        return text[: min(positions)] if positions else text

    def _split_safe_visible_text(self) -> str:
        sanitized = self._sanitize_visible_text(self._buffer)
        if len(sanitized) != len(self._buffer):
            self._buffer = ""
            self._stream_suppress_output = True
            return sanitized

        held_suffix = 0
        for token in self._protocol_tokens:
            for length in range(1, min(len(token), len(self._buffer)) + 1):
                if self._buffer.endswith(token[:length]):
                    held_suffix = max(held_suffix, length)
        emit_end = len(self._buffer) - held_suffix
        visible_text = self._buffer[:emit_end]
        self._buffer = self._buffer[emit_end:]
        return visible_text

    def _parse_arguments(self, payload: str) -> str | None:
        try:
            arguments, end = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError:
            return None
        if payload[end:].strip() or not isinstance(arguments, dict):
            return None
        return json.dumps(arguments, ensure_ascii=False)

    def _validate_arguments(
        self, name: str, parameters: str, tools: List[Tool]
    ) -> None:
        tool_indices = self._get_tool_indices(tools)
        tool = tools[tool_indices[name]]
        schema = tool.function.parameters
        if not tool.function.strict or schema is None:
            return

        arguments = json.loads(parameters)
        error = next(Draft202012Validator(schema).iter_errors(arguments), None)
        if error is not None:
            path = ".".join(str(part) for part in error.absolute_path)
            location = f" at '{path}'" if path else ""
            raise ToolCallParseError(
                f"Onyx tool '{name}' arguments violate its schema{location}: "
                f"{error.message}"
            )

    def _find_end_token(self, text: str, start: int) -> tuple[int, str]:
        end_pos = -1
        end_token = ""
        for candidate in self._end_tokens:
            candidate_pos = text.find(candidate, start)
            if candidate_pos != -1 and (
                end_pos == -1 or candidate_pos < end_pos
            ):
                end_pos = candidate_pos
                end_token = candidate
        return end_pos, end_token

    def has_tool_call(self, text: str) -> bool:
        return (
            self._recipient.match(text) is not None
            or self._sequence_header.match(text) is not None
        )

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        tool_indices = self._get_tool_indices(tools)
        pos = 0
        saw_self = False
        while pos < len(text):
            bare_user_match = self._bare_user_header.match(text, pos)
            if bare_user_match is not None:
                payload_start = bare_user_match.end()
                end_pos, _ = self._find_end_token(text, payload_start)
                payload_end = len(text) if end_pos == -1 else end_pos
                return StreamingParseResult(
                    normal_text=self._sanitize_visible_text(
                        text[payload_start:payload_end]
                    )
                )

            match = self._sequence_header.match(text, pos)
            if match is None:
                break
            name = match.group(1)
            payload_start = match.end()
            if match.group(2) == "{":
                payload_start -= 1
            end_pos, end_token = self._find_end_token(text, payload_start)
            payload_end = len(text) if end_pos == -1 else end_pos
            payload = text[payload_start:payload_end].strip()

            if name == "user":
                return StreamingParseResult(
                    normal_text=self._sanitize_visible_text(payload)
                )

            if name == "self":
                saw_self = True
                if end_pos == -1 or end_token != "<|eom|>":
                    return StreamingParseResult()
                pos = end_pos + len(end_token)
                continue
            if name not in tool_indices:
                raise ToolCallParseError(
                    f"Onyx generated an unknown tool recipient: {name}"
                )
            parameters = self._parse_arguments(payload)
            if parameters is None:
                raise ToolCallParseError(
                    f"Onyx tool '{name}' arguments are not a JSON object"
                )
            self._validate_arguments(name, parameters, tools)

            if end_pos != -1:
                remainder = text[end_pos + len(end_token) :]
                next_match = self._sequence_header.match(remainder)
                if next_match is not None and next_match.group(1) not in (
                    "user",
                    "self",
                ):
                    raise ToolCallParseError(
                        "Onyx supports at most one tool call per assistant response"
                    )

            return StreamingParseResult(
                calls=[
                    ToolCallItem(
                        tool_index=0,
                        name=name,
                        parameters=parameters,
                    )
                ]
            )

        if saw_self:
            return StreamingParseResult()

        match = self._recipient.match(text)
        if match is not None:
            if match.group(1) == "user":
                return StreamingParseResult(
                    normal_text=self._sanitize_visible_text(
                        self._strip_protocol_payload(text[match.end() :])
                    )
                )
            return StreamingParseResult()
        return StreamingParseResult(normal_text=self._sanitize_visible_text(text))

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        if self._stream_suppress_output:
            return StreamingParseResult()

        self._buffer += new_text
        calls = []

        if self._stream_name is None and not self._stream_is_user:
            stripped = self._buffer.lstrip()
            tool_result_prefix = "<|start|>tool"
            if self.current_tool_id >= 0:
                if tool_result_prefix.startswith(stripped):
                    return StreamingParseResult()
                if stripped.startswith(tool_result_prefix):
                    self._buffer = ""
                    self._stream_suppress_output = True
                    return StreamingParseResult()

            bare_user_match = self._bare_user_header.match(self._buffer)
            if bare_user_match is not None:
                self._stream_is_user = True
                self._stream_user_visible = self.current_tool_id < 0
                self._buffer = self._buffer[bare_user_match.end() :]
                stripped = self._buffer.lstrip()
            else:
                bare_user_prefixes = (
                    "<|message|>",
                    "assistant<|message|>",
                    "<|start|>assistant<|message|>",
                )
                if any(prefix.startswith(stripped) for prefix in bare_user_prefixes):
                    return StreamingParseResult()

            if not tools:
                unmarked_recipients = ("to=user", "to=self")
                if any(prefix.startswith(stripped) for prefix in unmarked_recipients):
                    return StreamingParseResult()
                for prefix in unmarked_recipients:
                    if stripped.startswith(prefix):
                        remainder = stripped[len(prefix) :]
                        message_token = "<|message|>"
                        if message_token.startswith(remainder):
                            return StreamingParseResult()
                        if remainder.startswith(message_token):
                            break
                        self._buffer = remainder
                        if prefix == "to=self":
                            self._stream_is_self = True
                        self._stream_is_user = True
                        self._stream_user_visible = not self._stream_is_self
                        break

            if self._stream_is_user:
                stripped = self._buffer.lstrip()

            if not self._stream_is_user:
                match = self._stream_header.match(self._buffer)
                if match is None:
                    header_prefixes = (
                        "to=",
                        "assistant to=",
                        "<|start|>assistant to=",
                    )
                    could_be_header = any(
                        prefix.startswith(stripped) or stripped.startswith(prefix)
                        for prefix in header_prefixes
                    )
                    if could_be_header:
                        return StreamingParseResult()
                    if self._buffer:
                        normal_text = self._split_safe_visible_text()
                        return StreamingParseResult(normal_text=normal_text)
                    return StreamingParseResult()

                name = match.group(1)
                tool_indices = self._get_tool_indices(tools)
                if name in ("user", "self"):
                    self._stream_is_user = True
                    self._stream_is_self = name == "self"
                    self._stream_user_visible = (
                        name == "user" and self.current_tool_id < 0
                    )
                    self._buffer = self._buffer[match.end() :]
                    if match.group(2) == "{":
                        self._buffer = "{" + self._buffer
                elif name not in tool_indices:
                    self._buffer = ""
                    self._stream_suppress_output = True
                    raise ToolCallParseError(
                        f"Onyx generated an unknown tool recipient: {name}"
                    )
                else:
                    if self.current_tool_id >= 0:
                        self._buffer = ""
                        self._stream_suppress_output = True
                        raise ToolCallParseError(
                            "Onyx supports at most one tool call per assistant response"
                        )
                    separator = match.group(2)
                    self._stream_name = name
                    self.current_tool_id += 1
                    self._stream_tool_index = self.current_tool_id
                    self.current_tool_name_sent = True
                    self.streamed_args_for_tool.append("")
                    self._buffer = self._buffer[match.end() :]
                    if separator == "{":
                        self._buffer = "{" + self._buffer
                    calls.append(
                        ToolCallItem(
                            tool_index=self._stream_tool_index,
                            name=name,
                            parameters="",
                        )
                    )

        end_pos = -1
        end_token = ""
        control_tokens = (
            self._protocol_tokens if self._stream_is_user else self._end_tokens
        )
        for candidate in control_tokens:
            candidate_pos = self._buffer.find(candidate)
            if candidate_pos != -1 and (end_pos == -1 or candidate_pos < end_pos):
                end_pos = candidate_pos
                end_token = candidate

        if end_pos != -1:
            argument_delta = self._buffer[:end_pos]
            self._buffer = self._buffer[end_pos + len(end_token) :]
            complete = True
        else:
            held_suffix = 0
            for candidate in control_tokens:
                for length in range(1, min(len(candidate), len(self._buffer)) + 1):
                    if self._buffer.endswith(candidate[:length]):
                        held_suffix = max(held_suffix, length)
            emit_end = len(self._buffer) - held_suffix
            argument_delta = self._buffer[:emit_end]
            self._buffer = self._buffer[emit_end:]
            complete = False

        if self._stream_is_user:
            was_self = self._stream_is_self
            if complete:
                self._stream_is_user = False
            normal_text = (
                self._sanitize_visible_text(argument_delta)
                if self._stream_user_visible
                else ""
            )
            if complete:
                self._stream_is_self = False
                self._stream_user_visible = False
                if was_self and end_token == "<|eom|>":
                    if self._buffer:
                        remainder = self.parse_streaming_increment("", tools)
                        return StreamingParseResult(
                            calls=remainder.calls,
                            normal_text=normal_text + remainder.normal_text,
                        )
                else:
                    self._buffer = ""
                    self._stream_suppress_output = True
            return StreamingParseResult(normal_text=normal_text)

        if argument_delta:
            assert self._stream_tool_index is not None
            calls.append(
                ToolCallItem(
                    tool_index=self._stream_tool_index,
                    parameters=argument_delta,
                )
            )
            self.streamed_args_for_tool[self._stream_tool_index] += argument_delta

        if complete:
            self.finalize_stream(tools)

            if self._buffer:
                remainder = self.parse_streaming_increment("", tools)
                calls.extend(remainder.calls)
                return StreamingParseResult(
                    calls=calls, normal_text=remainder.normal_text
                )

        return StreamingParseResult(calls=calls)

    def finalize_stream(self, tools: List[Tool]) -> str:
        if self._stream_suppress_output:
            self._buffer = ""
            return ""

        if self._stream_name is None:
            trailing_text = self._buffer
            self._buffer = ""
            hidden_self = self._stream_is_self
            self._stream_is_user = False
            self._stream_is_self = False
            self._stream_user_visible = False
            if hidden_self:
                return ""
            if len(trailing_text) > 1 and any(
                token.startswith(trailing_text) for token in self._protocol_tokens
            ):
                return ""
            return self._sanitize_visible_text(trailing_text)

        assert self._stream_tool_index is not None
        full_arguments = self.streamed_args_for_tool[self._stream_tool_index]
        try:
            parsed_arguments = json.loads(full_arguments)
        except json.JSONDecodeError as e:
            self._stream_suppress_output = True
            raise ToolCallParseError(
                f"Onyx tool '{self._stream_name}' arguments are invalid JSON: {e.msg}"
            ) from e
        if not isinstance(parsed_arguments, dict):
            self._stream_suppress_output = True
            raise ToolCallParseError(
                f"Onyx tool '{self._stream_name}' arguments are not a JSON object"
            )
        self._validate_arguments(self._stream_name, full_arguments, tools)
        self.prev_tool_call_arr.append(
            {"name": self._stream_name, "arguments": parsed_arguments}
        )
        self._stream_name = None
        self._stream_tool_index = None
        return ""

    def _schema_grammar(self, schema: dict) -> str:
        grammar = str(
            Grammar.from_json_schema(
                schema,
                any_whitespace=True,
                max_whitespace_cnt=1,
            )
        )
        # Grammar.__str__ renders an unbounded EBNF repeat as ``{n, -1}``,
        # while the structural-tag parser accepts the standard ``{n,}``
        # spelling. Both forms represent the same XGrammar repeat.
        return re.sub(r"\{(\d+), -1\}", r"{\1,}", grammar)

    @staticmethod
    def _namespace_grammar(grammar: str, prefix: str) -> tuple[str, str]:
        """Give every EBNF rule a branch-local name.

        XGrammar's structural ``or`` format can merge the states of sibling
        grammar/tag branches.  With multiple tools this can let a recipient
        selected for one function consume the JSON grammar of another and
        eventually reach a state with no valid next token.  A single EBNF
        union avoids that cross-branch dispatch, but its component grammars
        must not share rule names such as ``root`` and ``basic_string``.

        Only identifiers outside string literals and character classes are
        rewritten.  This preserves schemas containing literal enum values
        that happen to equal an EBNF rule name.
        """
        rule_names = {}
        for line in grammar.splitlines():
            match = re.match(r"^([A-Za-z_][A-Za-z0-9_]*)\s*::=", line)
            if match is not None:
                name = match.group(1)
                rule_names[name] = f"{prefix}{name}"

        if "root" not in rule_names:
            raise ValueError("XGrammar schema grammar has no root rule")

        def rewrite_line(line: str) -> str:
            output = []
            position = 0
            in_string = False
            in_character_class = False
            escaped = False

            while position < len(line):
                char = line[position]
                if in_string:
                    output.append(char)
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == '"':
                        in_string = False
                    position += 1
                    continue

                if in_character_class:
                    output.append(char)
                    if escaped:
                        escaped = False
                    elif char == "\\":
                        escaped = True
                    elif char == "]":
                        in_character_class = False
                    position += 1
                    continue

                if char == '"':
                    in_string = True
                    output.append(char)
                    position += 1
                    continue
                if char == "[":
                    in_character_class = True
                    output.append(char)
                    position += 1
                    continue
                if char.isalpha() or char == "_":
                    end = position + 1
                    while end < len(line) and (
                        line[end].isalnum() or line[end] == "_"
                    ):
                        end += 1
                    identifier = line[position:end]
                    output.append(rule_names.get(identifier, identifier))
                    position = end
                    continue

                output.append(char)
                position += 1

            return "".join(output)

        namespaced = "\n".join(rewrite_line(line) for line in grammar.splitlines())
        return namespaced, rule_names["root"]

    def _tool_union_grammar(self, tools: List[Tool]) -> str:
        """Build one recipient-discriminated EBNF union for all tools."""
        branch_rules = []
        schema_rules = []
        branch_names = []

        for index, tool in enumerate(tools):
            function = tool.function
            if not function.name:
                continue
            schema = function.parameters if function.strict else None
            schema = schema or {"type": "object"}
            namespaced, schema_root = self._namespace_grammar(
                self._schema_grammar(schema), f"onyx_tool_{index}_"
            )
            branch_name = f"onyx_tool_branch_{index}"
            begin = json.dumps(f" to={function.name}<|message|>")
            end = json.dumps("<|eom|>")
            branch_rules.append(
                f"{branch_name} ::= (({begin} {schema_root} {end}))"
            )
            schema_rules.append(namespaced)
            branch_names.append(branch_name)

        if not branch_names:
            raise ValueError("Onyx tool union requires at least one named tool")

        alternatives = ") | (".join(branch_names)
        return "\n".join(
            [
                f"root ::= (({alternatives}))",
                *branch_rules,
                *schema_rules,
            ]
        )

    def _complete_auto_grammar(self, tools: List[Tool]) -> str:
        """Compile the entire auto response into one isolated EBNF grammar.

        Keeping even the private-self prefix outside the final grammar can
        leave StructuralTag alternatives active after ``<|eom|>``. A single
        namespaced EBNF root makes the recipient/schema decision irreversible.

        Private reasoning excludes ``<`` so it cannot consume an Onyx protocol
        boundary as ordinary text. Final user text may contain arbitrary
        non-NUL characters; the scheduler owns its EOT/EOM termination.
        """
        tool_grammar, tool_root = self._namespace_grammar(
            self._tool_union_grammar(tools), "onyx_final_"
        )
        self_begin = json.dumps(" to=self<|message|>")
        self_end = json.dumps("<|eom|><|start|>assistant")
        explicit_user_begin = json.dumps(" to=user<|message|>")
        bare_user_begin = json.dumps("<|message|>")
        user_end = json.dumps("<|eot|>")
        return "\n".join(
            [
                "root ::= ((onyx_self_sequence{0, 8} onyx_final_response))",
                (
                    "onyx_self_sequence ::= "
                    f"(({self_begin} onyx_self_content {self_end}))"
                ),
                "onyx_self_content ::= (([^<]{0, 4096}))",
                (
                    "onyx_final_response ::= ((onyx_explicit_user) | "
                    f"(onyx_bare_user) | ({tool_root}))"
                ),
                (
                    "onyx_explicit_user ::= "
                    f"(({explicit_user_begin} onyx_user_content {user_end}))"
                ),
                (
                    "onyx_bare_user ::= "
                    f"(({bare_user_begin} onyx_user_content {user_end}))"
                ),
                "onyx_user_content ::= (([^\\0]*))",
                tool_grammar,
            ]
        )

    def get_structural_tag(
        self,
        tools: Union[List[Tool], None] = None,
        tool_choice: Union[ToolChoice, Literal["auto", "required"]] = "auto",
        thinking_mode: bool = False,
    ):
        """Build the complete native Onyx ``auto`` response union.

        An auto response is zero or more private ``to=self`` planning frames,
        followed by exactly one final branch: a free-form user response or one
        schema-constrained tool call.  User responses may use the explicit
        ``to=user`` recipient or the bare ``<|message|>`` form used when an SFT
        assistant message has no recipient.  Describing the whole response is
        important: a trigger-only grammar leaves the recipient prefix free, so
        the model can narrate a planned call to the user or invent a recipient
        before schema enforcement starts.

        A free-text EBNF branch preserves genuine ordinary answers while
        constraining the recipient from the first generated token. The explicit
        assistant separator after a self frame matches the Onyx SFT protocol and
        works with the scheduler's recipient-aware EOM continuation. Tool
        recipients and schemas are compiled into one namespaced EBNF union so a
        selected function cannot consume another function's argument grammar.
        """
        if (
            tool_choice != "auto"
            or Grammar is None
            or StructuralTag is None
            or not tools
        ):
            return None

        named_tools = [tool for tool in tools if tool.function.name]
        if not named_tools:
            return None

        auto_response = {
            "type": "grammar",
            "grammar": self._complete_auto_grammar(named_tools),
        }

        return StructuralTag.from_json(
            {
                "type": "structural_tag",
                "format": auto_response,
            }
        )

    def structure_info(self) -> _GetInfoFunc:
        def _info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f" to={name}<|message|>",
                end="<|eom|>",
                trigger=" to=",
            )

        return _info
