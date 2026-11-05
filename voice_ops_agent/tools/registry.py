"""Tool registry: register, look up, and execute tools by name."""

from typing import Dict, List, Optional

from .base import BaseTool, ToolResult


class ToolRegistry:
    """
    A central registry for all available tools.

    Tools are registered by their ``name`` property.  The registry exposes
    methods to look up and execute tools, and to list all registered tools.

    Usage
    -----
    ::

        registry = ToolRegistry()
        registry.register(WeatherLookupTool())
        result = registry.execute("weather_lookup", location="New York")
    """

    def __init__(self) -> None:
        self._tools: Dict[str, BaseTool] = {}

    # ------------------------------------------------------------------
    # Registration
    # ------------------------------------------------------------------

    def register(self, tool: BaseTool) -> None:
        """
        Register a tool instance.

        Parameters
        ----------
        tool : BaseTool
            Tool to register.  Its ``name`` property is used as the key.

        Raises
        ------
        ValueError
            If a tool with the same name is already registered.
        """
        if tool.name in self._tools:
            raise ValueError(
                f"A tool named {tool.name!r} is already registered. "
                "Use replace() to overwrite."
            )
        self._tools[tool.name] = tool

    def replace(self, tool: BaseTool) -> None:
        """Register or overwrite a tool (no uniqueness check)."""
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        """
        Remove a tool from the registry.

        Parameters
        ----------
        name : str
            Name of the tool to remove.

        Raises
        ------
        KeyError
            If the tool is not registered.
        """
        if name not in self._tools:
            raise KeyError(f"No tool named {name!r} is registered.")
        del self._tools[name]

    # ------------------------------------------------------------------
    # Look-up / execution
    # ------------------------------------------------------------------

    def get(self, name: str) -> Optional[BaseTool]:
        """Return the tool with *name*, or None if not found."""
        return self._tools.get(name)

    def has(self, name: str) -> bool:
        """Return True if a tool with *name* is registered."""
        return name in self._tools

    def execute(self, name: str, **kwargs) -> ToolResult:
        """
        Execute the named tool with the given keyword arguments.

        Parameters
        ----------
        name : str
            Tool name.
        **kwargs
            Arguments forwarded to ``tool.execute()``.

        Returns
        -------
        ToolResult

        Raises
        ------
        KeyError
            If the tool is not registered.
        """
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(f"No tool named {name!r} is registered.")
        return tool.execute(**kwargs)

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def list_tools(self) -> List[Dict]:
        """Return a list of dicts describing each registered tool."""
        return [
            {"name": t.name, "description": t.description}
            for t in self._tools.values()
        ]

    def __len__(self) -> int:
        return len(self._tools)

    def __repr__(self) -> str:
        names = list(self._tools.keys())
        return f"<ToolRegistry tools={names}>"


def default_registry() -> ToolRegistry:
    """
    Build and return a ToolRegistry pre-loaded with all mock tools.

    Returns
    -------
    ToolRegistry
    """
    from .mock_tools import (
        WeatherLookupTool,
        CalendarCheckTool,
        OrderStatusTool,
        KnowledgeBaseSearchTool,
    )

    registry = ToolRegistry()
    registry.register(WeatherLookupTool())
    registry.register(CalendarCheckTool())
    registry.register(OrderStatusTool())
    registry.register(KnowledgeBaseSearchTool())
    return registry
