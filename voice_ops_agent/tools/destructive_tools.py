"""Destructive mock tools: order cancellation and data deletion stubs.

These tools are registered separately from the standard mock tools because
they require human-in-the-loop confirmation before execution.  They are
stubs – no real data is modified.
"""

from .base import BaseTool, ToolResult


class OrderCancelTool(BaseTool):
    """
    Stub tool that simulates cancelling a customer order.

    Requires HITL confirmation before execution.
    Accepts ``order_id`` keyword argument.
    """

    @property
    def name(self) -> str:
        return "order_cancel"

    @property
    def description(self) -> str:
        return "Cancel a customer order by order ID. ⚠️  Destructive action."

    def execute(self, order_id: str = "00000", **kwargs) -> ToolResult:
        data = {"order_id": order_id, "status": "cancelled"}
        summary = f"Order {order_id} has been successfully cancelled."
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            text_summary=summary,
        )


class DataDeleteTool(BaseTool):
    """
    Stub tool that simulates deleting a user data record.

    Requires HITL confirmation before execution.
    Accepts ``record_id`` keyword argument.
    """

    @property
    def name(self) -> str:
        return "data_delete"

    @property
    def description(self) -> str:
        return "Permanently delete a user data record. ⚠️  Destructive action."

    def execute(self, record_id: str = "0", **kwargs) -> ToolResult:
        data = {"record_id": record_id, "deleted": True}
        summary = f"Data record {record_id} has been permanently deleted."
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            text_summary=summary,
        )
