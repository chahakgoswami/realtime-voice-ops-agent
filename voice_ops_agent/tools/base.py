"""Base class and result type for all tools."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import abc


@dataclass
class ToolResult:
    """
    Container returned by every tool invocation.

    Attributes
    ----------
    tool_name : str
        The name of the tool that produced this result.
    success : bool
        Whether the tool call succeeded.
    data : dict
        Structured output data from the tool.
    error : str or None
        Human-readable error message when success is False.
    text_summary : str
        A short natural-language summary suitable for injecting into LLM
        context or speaking aloud.
    """

    tool_name: str
    success: bool
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    text_summary: str = ""

    def to_context_dict(self) -> Dict[str, Any]:
        """Return a dict representation suitable for LLM context injection."""
        return {
            "role": "tool",
            "tool_name": self.tool_name,
            "success": self.success,
            "result": self.data if self.success else {"error": self.error},
            "summary": self.text_summary,
        }


class BaseTool(abc.ABC):
    """
    Abstract base class for all tools in the registry.

    Sub-classes must implement:
        - ``name`` property  (unique string identifier)
        - ``description`` property  (human-readable description)
        - ``execute(**kwargs)``  (tool logic)
    """

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Unique tool identifier used in tool-call detection."""
        ...

    @property
    @abc.abstractmethod
    def description(self) -> str:
        """Human-readable description of what the tool does."""
        ...

    @abc.abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """
        Run the tool with the given keyword arguments.

        Returns
        -------
        ToolResult
        """
        ...

    def __repr__(self) -> str:
        return f"<Tool name={self.name!r}>"
