import json
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StructureInfo,
    StreamingParseResult,
    ToolCallItem,
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
    _end_tokens = ("<|eom|>", "<|eot|>")

    def __init__(self):
        super().__init__()
        self._stream_name: str | None = None
        self._stream_tool_index: int | None = None
        self._stream_is_user = False
        self._stream_user_visible = False

    def _strip_protocol_payload(self, payload: str) -> str:
        payload = payload.strip()
        if payload.startswith("<|message|>"):
            payload = payload[len("<|message|>") :].lstrip()
        for end_token in self._end_tokens:
            if payload.endswith(end_token):
                payload = payload[: -len(end_token)].rstrip()
        return payload

    def _parse_arguments(self, payload: str) -> str | None:
        try:
            arguments, end = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError:
            return None
        if payload[end:].strip() or not isinstance(arguments, dict):
            return None
        return json.dumps(arguments, ensure_ascii=False)

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
        return self._recipient.match(text) is not None

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        tool_indices = self._get_tool_indices(tools)
        calls = []
        pos = 0

        while pos < len(text):
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
                if calls:
                    return StreamingParseResult(calls=calls)
                return StreamingParseResult(normal_text=payload)

            if name not in tool_indices:
                break
            parameters = self._parse_arguments(payload)
            if parameters is None:
                break
            calls.append(
                ToolCallItem(
                    tool_index=len(calls),
                    name=name,
                    parameters=parameters,
                )
            )

            if end_pos == -1:
                return StreamingParseResult(calls=calls)
            pos = end_pos + len(end_token)

        if calls:
            return StreamingParseResult(calls=calls)

        match = self._recipient.match(text)
        if match is not None and match.group(1) == "user":
            return StreamingParseResult(
                normal_text=self._strip_protocol_payload(text[match.end() :])
            )
        return StreamingParseResult(normal_text=text)

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
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
                    return StreamingParseResult()

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
                    normal_text = self._buffer
                    self._buffer = ""
                    return StreamingParseResult(normal_text=normal_text)
                return StreamingParseResult()

            name = match.group(1)
            tool_indices = self._get_tool_indices(tools)
            if name == "user":
                self._stream_is_user = True
                self._stream_user_visible = self.current_tool_id < 0
                self._buffer = self._buffer[match.end() :]
                if match.group(2) == "{":
                    self._buffer = "{" + self._buffer
            elif name not in tool_indices:
                normal_text = self._buffer
                self._buffer = ""
                return StreamingParseResult(normal_text=normal_text)
            else:
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
        for candidate in self._end_tokens:
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
            for candidate in self._end_tokens:
                for length in range(1, min(len(candidate), len(self._buffer)) + 1):
                    if self._buffer.endswith(candidate[:length]):
                        held_suffix = max(held_suffix, length)
            emit_end = len(self._buffer) - held_suffix
            argument_delta = self._buffer[:emit_end]
            self._buffer = self._buffer[emit_end:]
            complete = False

        if self._stream_is_user:
            if complete:
                self._stream_is_user = False
            normal_text = argument_delta if self._stream_user_visible else ""
            if complete:
                self._stream_user_visible = False
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
            assert self._stream_tool_index is not None
            completed_tool_index = self._stream_tool_index
            full_arguments = self.streamed_args_for_tool[completed_tool_index]
            try:
                parsed_arguments = json.loads(full_arguments)
            except json.JSONDecodeError:
                parsed_arguments = {}
            self.prev_tool_call_arr.append(
                {"name": self._stream_name, "arguments": parsed_arguments}
            )
            self._stream_name = None
            self._stream_tool_index = None

            if self._buffer:
                remainder = self.parse_streaming_increment("", tools)
                calls.extend(remainder.calls)
                return StreamingParseResult(
                    calls=calls, normal_text=remainder.normal_text
                )

        return StreamingParseResult(calls=calls)

    def structure_info(self) -> _GetInfoFunc:
        def _info(name: str) -> StructureInfo:
            return StructureInfo(
                begin=f" to={name}<|message|>",
                end="<|eom|>",
                trigger=" to=",
            )

        return _info
