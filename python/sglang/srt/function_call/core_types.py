from dataclasses import dataclass
from typing import Callable, List, Optional

from pydantic import BaseModel


class ToolCallItem(BaseModel):
    """Simple encapsulation of the parsed ToolCall result for easier usage in streaming contexts."""

    tool_index: int
    name: Optional[str] = None
    parameters: str  # JSON string


class StreamingParseResult(BaseModel):
    """Result of streaming incremental parsing."""

    normal_text: str = ""
    calls: List[ToolCallItem] = []


class ToolCallParseError(ValueError):
    """Raised when model output violates the configured tool-call protocol."""


@dataclass
class StructureInfo:
    begin: str
    end: str
    trigger: str


"""
Helper alias of function
Usually it is a function that takes a name string and returns a StructureInfo object,
which can be used to construct a structural_tag object
"""
_GetInfoFunc = Callable[[str], StructureInfo]
