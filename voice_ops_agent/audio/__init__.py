"""Audio I/O sub-package."""

from .input import MockMicrophone
from .output import MockSpeaker

__all__ = ["MockMicrophone", "MockSpeaker"]
