"""Tool-call intent parser for streamed LLM output.

The mock LLM can embed tool-call directives inside its response using a
simple line-based syntax that is easy to test without a real LLM:

    TOOL_CALL:<tool_name>:<json_args>

Example:
    TOOL_CALL:weather_lookup:{"location": "New York"}
    TOOL_CALL:order_status:{"order_id": "12345"}

The parser scans accumulated token output for these markers, extracts
tool name and arguments, and returns structured ToolCallIntent objects.
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# Pattern: TOOL_CALL:<name>:<json>
_TOOL_CALL_RE = re.compile(
    r"TOOL_CALL:([\w_]+):(.+)",
    re.MULTILINE,
)


@dataclass
class ToolCallIntent:
    """
    Represents a parsed tool-call directive extracted from LLM output.

    Attributes
    ----------
    tool_name : str
        Name of the tool to invoke.
    arguments : dict
        Parsed keyword arguments for the tool.
    raw : str
        The original raw string that was matched.
    """

    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    raw: str = ""


def parse_tool_calls(text: str) -> List[ToolCallIntent]:
    """
    Scan *text* for TOOL_CALL directives and return a list of intents.

    Parameters
    ----------
    text : str
        Accumulated LLM output (may include multiple lines).

    Returns
    -------
    list of ToolCallIntent
        Possibly empty if no directives were found.
    """
    intents: List[ToolCallIntent] = []
    for match in _TOOL_CALL_RE.finditer(text):
        tool_name = match.group(1).strip()
        args_str = match.group(2).strip()
        try:
            arguments = json.loads(args_str)
            if not isinstance(arguments, dict):
                arguments = {}
        except json.JSONDecodeError:
            arguments = {}
        intents.append(
            ToolCallIntent(
                tool_name=tool_name,
                arguments=arguments,
                raw=match.group(0),
            )
        )
    return intents


def has_tool_call(text: str) -> bool:
    """Return True if *text* contains at least one TOOL_CALL directive."""
    return bool(_TOOL_CALL_RE.search(text))


def strip_tool_calls(text: str) -> str:
    """
    Remove all TOOL_CALL directives from *text*, returning the
    remaining (human-facing) content.

    Parameters
    ----------
    text : str
        Raw LLM output containing possible TOOL_CALL markers.

    Returns
    -------
    str
        Cleaned text with markers removed and extraneous whitespace collapsed.
    """
    cleaned = _TOOL_CALL_RE.sub("", text)
    # Collapse multiple blank lines
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()
