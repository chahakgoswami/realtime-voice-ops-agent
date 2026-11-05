"""Tool executor: bridges the agent loop and the tool registry.

Responsibilities
----------------
1. Receive a list of ToolCallIntent objects from the parser.
2. Execute each intent via the registry.
3. Inject ToolResult summaries back into the LLM conversation context.
4. Provide the enriched context for the next LLM call.
"""

from typing import Callable, List, Optional

from voice_ops_agent.tools.base import ToolResult
from voice_ops_agent.tools.parser import ToolCallIntent
from voice_ops_agent.tools.registry import ToolRegistry


class ToolExecutor:
    """
    Executes tool calls and manages context injection.

    Parameters
    ----------
    registry : ToolRegistry
        Registry containing all available tools.
    on_tool_call : callable, optional
        ``on_tool_call(intent: ToolCallIntent)`` – called before each tool
        execution (useful for logging / UI feedback).
    on_tool_result : callable, optional
        ``on_tool_result(result: ToolResult)`` – called after each tool
        execution.
    """

    def __init__(
        self,
        registry: ToolRegistry,
        on_tool_call: Optional[Callable[[ToolCallIntent], None]] = None,
        on_tool_result: Optional[Callable[[ToolResult], None]] = None,
    ) -> None:
        self.registry = registry
        self.on_tool_call = on_tool_call
        self.on_tool_result = on_tool_result
        # History of all executed tool calls and their results
        self.execution_history: List[dict] = []

    def execute_all(
        self,
        intents: List[ToolCallIntent],
        context: List[dict],
    ) -> List[dict]:
        """
        Execute every intent in *intents*, append results to *context*,
        and return the updated context list.

        Parameters
        ----------
        intents : list of ToolCallIntent
            Parsed tool-call directives from the LLM output.
        context : list of dict
            Current conversation context (will be mutated in-place and
            also returned).

        Returns
        -------
        list of dict
            Updated context including tool results.
        """
        for intent in intents:
            result = self._execute_one(intent)
            context.append(result.to_context_dict())

        return context

    def _execute_one(self, intent: ToolCallIntent) -> ToolResult:
        """Execute a single ToolCallIntent and return its ToolResult."""
        if self.on_tool_call:
            self.on_tool_call(intent)

        if not self.registry.has(intent.tool_name):
            result = ToolResult(
                tool_name=intent.tool_name,
                success=False,
                error=f"Tool '{intent.tool_name}' not found in registry.",
                text_summary=f"I could not find a tool named '{intent.tool_name}'.",
            )
        else:
            try:
                result = self.registry.execute(intent.tool_name, **intent.arguments)
            except Exception as exc:  # noqa: BLE001
                result = ToolResult(
                    tool_name=intent.tool_name,
                    success=False,
                    error=str(exc),
                    text_summary=f"The tool '{intent.tool_name}' encountered an error: {exc}",
                )

        self.execution_history.append(
            {"intent": intent, "result": result}
        )

        if self.on_tool_result:
            self.on_tool_result(result)

        return result
