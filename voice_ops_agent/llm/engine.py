"""Mocked LLM backend with simulated streaming responses."""

import time
import random
from typing import Iterator, List, Optional, Dict


# ---------------------------------------------------------------------------
# Pre-canned response templates keyed by rough intent keywords
# ---------------------------------------------------------------------------
_RESPONSE_TEMPLATES: Dict[str, str] = {
    "weather": "The weather in your area looks partly cloudy with a high of 72 degrees Fahrenheit. There is a 20 percent chance of rain in the afternoon.",
    "calendar": "You have three events scheduled for tomorrow: a team stand-up at 9 AM, a product review at 2 PM, and a one-on-one with your manager at 4 PM.",
    "order": "I found your order number 12345. It is currently in transit and expected to arrive within two business days.",
    "cancel": "I can help you with that cancellation. Could you please confirm the account details so I can proceed safely?",
    "hello": "Hello! I am your voice assistant. How can I help you today?",
    "human": "I understand you would like to speak with a human agent. Let me connect you now. Please hold for a moment.",
    "account": "I am sorry to hear you are having trouble. Let me look into your account. Could you verify your email address or phone number?",
    "product": "Our latest updates include improved performance, a refreshed interface, and several new integrations announced last week.",
    "default": "I understand your request. Let me assist you with that right away. Is there anything specific you would like me to focus on?",
}


def _choose_response(user_text: str) -> str:
    """Pick a canned response that loosely matches the user's text."""
    lower = user_text.lower()
    for keyword, response in _RESPONSE_TEMPLATES.items():
        if keyword in lower:
            return response
    return _RESPONSE_TEMPLATES["default"]


class MockLLMEngine:
    """
    A simulated LLM backend that returns streaming token chunks.

    Parameters
    ----------
    token_latency : float
        Delay in seconds between each streamed token (simulates generation
        speed).  Default is 0.02 s, giving ~50 tokens/second.
    first_token_latency : float
        Additional delay before the very first token is emitted (simulates
        time-to-first-token / prefill latency).  Default is 0.05 s.
    fixed_response : str or None
        If provided, every call returns this text regardless of input.
    """

    def __init__(
        self,
        token_latency: float = 0.02,
        first_token_latency: float = 0.05,
        fixed_response: Optional[str] = None,
    ) -> None:
        self.token_latency = token_latency
        self.first_token_latency = first_token_latency
        self.fixed_response = fixed_response
        self._cancelled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def cancel(self) -> None:
        """Signal that the current stream should be cancelled."""
        self._cancelled = True

    def reset(self) -> None:
        """Clear the cancellation flag before starting a new request."""
        self._cancelled = False

    def stream_response(
        self,
        user_text: str,
        context: Optional[List[Dict]] = None,
    ) -> Iterator[str]:
        """
        Yield response tokens one at a time for the given user text.

        Parameters
        ----------
        user_text : str
            Transcribed user utterance from STT.
        context : list of dict, optional
            Conversation history (ignored in mock, reserved for future use).

        Yields
        ------
        str
            Individual word tokens, including trailing spaces where appropriate.
        """
        self.reset()

        if self.fixed_response is not None:
            full_response = self.fixed_response
        else:
            full_response = _choose_response(user_text)

        tokens = full_response.split(" ")

        # Simulate time-to-first-token
        if self.first_token_latency > 0:
            time.sleep(self.first_token_latency)

        for idx, token in enumerate(tokens):
            if self._cancelled:
                return

            # Yield token with space (except the last)
            yield token + (" " if idx < len(tokens) - 1 else "")

            if self.token_latency > 0:
                time.sleep(self.token_latency)

    def complete(
        self,
        user_text: str,
        context: Optional[List[Dict]] = None,
    ) -> str:
        """
        Non-streaming convenience wrapper – collects all tokens into one string.

        Returns
        -------
        str
            The complete LLM response.
        """
        return "".join(self.stream_response(user_text, context))
