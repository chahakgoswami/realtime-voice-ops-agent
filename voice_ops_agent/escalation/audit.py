"""Audit logger: persists escalation events to a local JSON file."""

import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Union


_DEFAULT_LOG_PATH = "escalation_audit.json"


class AuditLogger:
    """
    Thread-safe logger that appends escalation events to a local JSON file.

    The file is a JSON array.  On each write the array is re-serialised so
    the file is always valid JSON (suitable for small audit trails; for
    high-volume use a JSONL / append-only format would be preferred).

    Parameters
    ----------
    path : str or Path
        File path for the audit log (default ``escalation_audit.json`` in
        the current working directory).
    auto_create : bool
        If True (default), the file and any parent directories are created
        automatically on the first write.
    """

    def __init__(
        self,
        path: Union[str, Path] = _DEFAULT_LOG_PATH,
        auto_create: bool = True,
    ) -> None:
        self.path = Path(path)
        self.auto_create = auto_create
        self._lock = threading.Lock()
        self._records: List[Dict[str, Any]] = []
        # Load existing records if the file already exists
        if self.path.exists():
            try:
                with self.path.open("r", encoding="utf-8") as fh:
                    data = json.load(fh)
                    if isinstance(data, list):
                        self._records = data
            except (json.JSONDecodeError, OSError):
                self._records = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def log(self, event: Any) -> None:
        """
        Append *event* to the audit log.

        *event* may be an :class:`EscalationEvent` (which exposes
        ``to_dict()``) or any dict-serialisable object.

        Parameters
        ----------
        event : EscalationEvent or dict
        """
        if hasattr(event, "to_dict"):
            record = event.to_dict()
        elif isinstance(event, dict):
            record = event
        else:
            record = {"raw": str(event)}

        with self._lock:
            self._records.append(record)
            self._persist()

    def records(self) -> List[Dict[str, Any]]:
        """Return a copy of all logged records."""
        with self._lock:
            return list(self._records)

    def clear(self) -> None:
        """
        Remove all records from the in-memory log **and** delete the file.

        This is intended for testing only; production use should retain the
        audit trail.
        """
        with self._lock:
            self._records.clear()
            if self.path.exists():
                self.path.unlink()

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _persist(self) -> None:
        """Write the current records list to disk (must be called under lock)."""
        if self.auto_create:
            self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("w", encoding="utf-8") as fh:
            json.dump(self._records, fh, indent=2, default=str)
