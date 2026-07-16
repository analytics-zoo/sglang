import json
import re
from typing import List

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.base_format_detector import BaseFormatDetector
from sglang.srt.function_call.core_types import (
    StreamingParseResult,
    ToolCallItem,
    _GetInfoFunc,
)


class OnyxDetector(BaseFormatDetector):
    """Parse Onyx recipient tool calls: ``to=name{"arg": "value"}``."""

    _recipient = re.compile(r"^\s*to=([A-Za-z_][A-Za-z0-9_.-]*)\s*")

    def _parse(self, text: str, tools: List[Tool]) -> ToolCallItem | None:
        match = self._recipient.match(text)
        if match is None:
            return None
        name = match.group(1)
        tool_indices = self._get_tool_indices(tools)
        if name not in tool_indices:
            return None
        payload = text[match.end() :].strip()
        if payload.startswith("<|message|>"):
            payload = payload[len("<|message|>") :].lstrip()
        for end_token in ("<|eom|>", "<|eot|>"):
            if payload.endswith(end_token):
                payload = payload[: -len(end_token)].rstrip()
        try:
            arguments, end = json.JSONDecoder().raw_decode(payload)
        except json.JSONDecodeError:
            return None
        if payload[end:].strip() or not isinstance(arguments, dict):
            return None
        return ToolCallItem(
            tool_index=tool_indices[name],
            name=name,
            parameters=json.dumps(arguments, ensure_ascii=False),
        )

    def has_tool_call(self, text: str) -> bool:
        return self._recipient.match(text) is not None

    def detect_and_parse(self, text: str, tools: List[Tool]) -> StreamingParseResult:
        call = self._parse(text, tools)
        if call is None:
            return StreamingParseResult(normal_text=text)
        return StreamingParseResult(calls=[call])

    def parse_streaming_increment(
        self, new_text: str, tools: List[Tool]
    ) -> StreamingParseResult:
        self._buffer += new_text
        call = self._parse(self._buffer, tools)
        if call is not None:
            self._buffer = ""
            return StreamingParseResult(calls=[call])

        if not new_text and self._buffer:
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)

        stripped = self._buffer.lstrip()
        if stripped and not "to=".startswith(stripped) and not stripped.startswith("to="):
            normal_text = self._buffer
            self._buffer = ""
            return StreamingParseResult(normal_text=normal_text)
        return StreamingParseResult()

    def structure_info(self) -> _GetInfoFunc:
        raise NotImplementedError

    def supports_structural_tag(self) -> bool:
        return False
