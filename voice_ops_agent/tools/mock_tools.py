"""Mock tool implementations: weather, calendar, order status, KB search."""

import random
from typing import Optional

from .base import BaseTool, ToolResult


# ---------------------------------------------------------------------------
# Weather Lookup
# ---------------------------------------------------------------------------

class WeatherLookupTool(BaseTool):
    """
    Mock weather lookup tool.

    Accepts ``location`` keyword argument.  Returns a fake weather report.
    """

    _CONDITIONS = ["sunny", "partly cloudy", "overcast", "light rain", "thunderstorms"]

    @property
    def name(self) -> str:
        return "weather_lookup"

    @property
    def description(self) -> str:
        return "Look up current weather conditions for a given location."

    def execute(self, location: str = "Unknown", **kwargs) -> ToolResult:
        condition = random.choice(self._CONDITIONS)
        temp_f = random.randint(45, 95)
        humidity = random.randint(20, 90)
        data = {
            "location": location,
            "condition": condition,
            "temperature_f": temp_f,
            "humidity_pct": humidity,
        }
        summary = (
            f"The weather in {location} is currently {condition} "
            f"with a temperature of {temp_f}°F and {humidity}% humidity."
        )
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            text_summary=summary,
        )


# ---------------------------------------------------------------------------
# Calendar Check
# ---------------------------------------------------------------------------

class CalendarCheckTool(BaseTool):
    """
    Mock calendar check tool.

    Accepts ``date`` keyword argument (string, e.g. "tomorrow").
    Returns a fake list of events.
    """

    _SAMPLE_EVENTS = [
        {"time": "09:00", "title": "Team stand-up", "duration_min": 15},
        {"time": "14:00", "title": "Product review", "duration_min": 60},
        {"time": "16:00", "title": "1:1 with manager", "duration_min": 30},
    ]

    @property
    def name(self) -> str:
        return "calendar_check"

    @property
    def description(self) -> str:
        return "Check the user's calendar for a given date."

    def execute(self, date: str = "today", **kwargs) -> ToolResult:
        # Return a random subset of sample events
        num_events = random.randint(1, len(self._SAMPLE_EVENTS))
        events = self._SAMPLE_EVENTS[:num_events]
        data = {"date": date, "events": events}
        event_descriptions = "; ".join(
            f"{e['time']} – {e['title']} ({e['duration_min']} min)" for e in events
        )
        summary = f"You have {num_events} event(s) on {date}: {event_descriptions}."
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            text_summary=summary,
        )


# ---------------------------------------------------------------------------
# Order Status
# ---------------------------------------------------------------------------

class OrderStatusTool(BaseTool):
    """
    Mock order status tool.

    Accepts ``order_id`` keyword argument.
    """

    _STATUSES = [
        "processing",
        "shipped",
        "in transit",
        "out for delivery",
        "delivered",
    ]

    @property
    def name(self) -> str:
        return "order_status"

    @property
    def description(self) -> str:
        return "Retrieve the current status of a customer order by order ID."

    def execute(self, order_id: str = "00000", **kwargs) -> ToolResult:
        status = random.choice(self._STATUSES)
        eta_days = random.randint(0, 5) if status not in ("delivered",) else 0
        data = {
            "order_id": order_id,
            "status": status,
            "eta_days": eta_days,
        }
        if status == "delivered":
            summary = f"Order {order_id} has been delivered."
        else:
            summary = (
                f"Order {order_id} is currently {status} "
                f"and is expected to arrive in {eta_days} day(s)."
            )
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            text_summary=summary,
        )


# ---------------------------------------------------------------------------
# Knowledge Base Search
# ---------------------------------------------------------------------------

class KnowledgeBaseSearchTool(BaseTool):
    """
    Mock knowledge base search tool.

    Accepts ``query`` keyword argument.
    Returns fake KB articles matching the query.
    """

    _KB_ARTICLES = [
        {
            "id": "KB001",
            "title": "How to reset your password",
            "snippet": "Visit the account settings page and click 'Forgot Password'.",
        },
        {
            "id": "KB002",
            "title": "Return and refund policy",
            "snippet": "Items can be returned within 30 days of purchase with a receipt.",
        },
        {
            "id": "KB003",
            "title": "Product compatibility guide",
            "snippet": "Our products are compatible with all major operating systems.",
        },
        {
            "id": "KB004",
            "title": "Shipping information",
            "snippet": "Standard shipping takes 3-5 business days. Expedited options available.",
        },
    ]

    @property
    def name(self) -> str:
        return "kb_search"

    @property
    def description(self) -> str:
        return "Search the knowledge base for articles relevant to a query."

    def execute(self, query: str = "", **kwargs) -> ToolResult:
        # Simple mock: return 1-2 random articles
        num_results = random.randint(1, 2)
        results = random.sample(self._KB_ARTICLES, min(num_results, len(self._KB_ARTICLES)))
        data = {"query": query, "results": results}
        if results:
            top = results[0]
            summary = (
                f"Found {len(results)} article(s) for '{query}'. "
                f"Top result: {top['title']} – {top['snippet']}"
            )
        else:
            summary = f"No articles found for '{query}'."
        return ToolResult(
            tool_name=self.name,
            success=True,
            data=data,
            text_summary=summary,
        )
