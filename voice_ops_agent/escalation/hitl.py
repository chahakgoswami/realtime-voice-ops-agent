"""Human-in-the-loop confirmation for destructive tool actions.

Destructive tools (e.g. order cancellation, data deletion) must be approved
by an explicit human confirmation before execution.  In production this would
show a UI prompt or send an out-of-band message; in this mock the confirmation
is obtained via a configurable strategy (interactive stdin prompt, or an
auto-approve / auto-deny flag for testing).
"""

from typing import Any, Callable, Dict, List, Optional


# Names of tool actions considered destructive – used as the default list.
DESTRUCTIVE_TOOLS: List[str] = [
    "order_cancel",
    "data_delete",
    "account_delete",
    "subscription_cancel",
]


class HumanInTheLoopConfirmation:
    """
    Gate that requires explicit human approval before a destructive tool
    action is executed.

    Parameters
    ----------
    destructive_tools : list of str
        Tool names that require confirmation.  Defaults to
        :data:`DESTRUCTIVE_TOOLS`.
    auto_approve : bool
        When True, all confirmations are automatically approved (useful for
        integration tests that do not want an interactive prompt).
    auto_deny : bool
        When True, all confirmations are automatically denied (useful for
        testing the rejection path).
    prompt_fn : callable, optional
        ``prompt_fn(message: str) -> str`` – used to obtain user input.
        Defaults to Python's built-in ``input()``.
    on_approved : callable, optional
        ``on_approved(tool_name, arguments)`` callback fired after approval.
    on_denied : callable, optional
        ``on_denied(tool_name, arguments)`` callback fired after denial.
    """

    def __init__(
        self,
        destructive_tools: Optional[List[str]] = None,
        auto_approve: bool = False,
        auto_deny: bool = False,
        prompt_fn: Optional[Callable[[str], str]] = None,
        on_approved: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_denied: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> None:
        self.destructive_tools = destructive_tools if destructive_tools is not None else list(DESTRUCTIVE_TOOLS)
        self.auto_approve = auto_approve
        self.auto_deny = auto_deny
        self.prompt_fn = prompt_fn or input
        self.on_approved = on_approved
        self.on_denied = on_denied
        # History of all confirmation decisions
        self.history: List[Dict[str, Any]] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_destructive(self, tool_name: str) -> bool:
        """Return True if *tool_name* requires human confirmation."""
        return tool_name in self.destructive_tools

    def confirm(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """
        Request human approval to execute a destructive tool.

        Parameters
        ----------
        tool_name : str
            Name of the tool about to be executed.
        arguments : dict
            Arguments that will be passed to the tool.

        Returns
        -------
        bool
            True if the action is approved, False if denied.
        """
        if self.auto_approve:
            approved = True
        elif self.auto_deny:
            approved = False
        else:
            approved = self._interactive_confirm(tool_name, arguments)

        record = {
            "tool_name": tool_name,
            "arguments": arguments,
            "approved": approved,
        }
        self.history.append(record)

        if approved and self.on_approved:
            self.on_approved(tool_name, arguments)
        elif not approved and self.on_denied:
            self.on_denied(tool_name, arguments)

        return approved

    def confirm_if_destructive(
        self, tool_name: str, arguments: Dict[str, Any]
    ) -> bool:
        """
        Call :meth:`confirm` only when the tool is destructive; return True
        immediately for non-destructive tools.

        Parameters
        ----------
        tool_name : str
        arguments : dict

        Returns
        -------
        bool
            True if the tool may proceed (either non-destructive, or approved).
        """
        if not self.is_destructive(tool_name):
            return True
        return self.confirm(tool_name, arguments)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _interactive_confirm(self, tool_name: str, arguments: Dict[str, Any]) -> bool:
        """Show an interactive prompt and return the user's decision."""
        args_display = ", ".join(f"{k}={v!r}" for k, v in arguments.items())
        message = (
            f"\n[HITL] ⚠️  Destructive action requested\n"
            f"  Tool     : {tool_name}\n"
            f"  Arguments: {args_display}\n"
            f"  Type 'yes' to approve, anything else to deny: "
        )
        try:
            response = self.prompt_fn(message)
        except (EOFError, KeyboardInterrupt):
            response = ""
        return response.strip().lower() in ("yes", "y")
