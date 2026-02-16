"""
Logging service for centralized log buffer management and UI integration.
"""
import threading
from collections import deque
from datetime import datetime
from typing import List, Dict, Any, Optional
from enum import Enum


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO" 
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogEntry:
    """Represents a log entry with timestamp and categorization."""

    def __init__(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        category: str = "general",
        source: str = "openapi",
        *,
        timestamp: Optional[str] = None,
        sequence: Optional[int] = None,
        **kwargs,
    ):
        timestamp = timestamp or datetime.now().isoformat()
        self.message = message
        self.level = level
        self.category = category
        self.source = source
        self.timestamp = timestamp
        self.sequence = sequence
        self.metadata = kwargs

    def to_dict(self) -> Dict[str, Any]:
        """Convert log entry to dictionary format."""
        return {
            "message": self.message,
            "level": self.level.value,
            "category": self.category,
            "source": self.source,
            "timestamp": self.timestamp,
            "sequence": self.sequence,
            **self.metadata,
        }


class LogManager:
    """Manages application logs with categorization and UI integration."""

    def __init__(self, max_entries: int = 100):
        self.max_entries = max_entries
        self._log_buffer: deque = deque(maxlen=max_entries)
        self._lock = threading.Lock()
        self._sequence = 0

    def add_log(
        self,
        message: str,
        level: LogLevel = LogLevel.INFO,
        category: str = "general",
        source: str = "openapi",
        **kwargs,
    ) -> LogEntry:
        """Add a log entry to the buffer."""
        timestamp = kwargs.pop("timestamp", None)

        with self._lock:
            self._sequence += 1
            entry = LogEntry(
                message,
                level,
                category,
                source=source,
                timestamp=timestamp,
                sequence=self._sequence,
                **kwargs,
            )
            self._log_buffer.append(entry)
        return entry

    def get_logs(
        self,
        category: str = None,
        source: str = None,
        after: int | None = None,
        limit: int | None = None,
        exclude_logger: str = None,
    ) -> List[Dict[str, Any]]:
        """Get all logs, optionally filtered by category or source."""
        with self._lock:
            logs = list(self._log_buffer)
            if category:
                logs = [log for log in logs if log.category == category]
            if source:
                logs = [log for log in logs if log.source == source]
            if exclude_logger:
                logs = [log for log in logs if log.metadata.get("logger") != exclude_logger]
            if after is not None:
                logs = [log for log in logs if (log.sequence or 0) > after]
            if limit is not None and limit > 0:
                logs = logs[-limit:]
            return [log.to_dict() for log in logs]

    def get_logs_categorized(
        self,
        source: str = None,
        after: int | None = None,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Get logs grouped by category, optionally limited to a source."""
        categorized: Dict[str, List[Dict[str, Any]]] = {}
        with self._lock:
            for log in self._log_buffer:
                if source and log.source != source:
                    continue
                if after is not None and (log.sequence or 0) <= after:
                    continue
                categorized.setdefault(log.category, []).append(log.to_dict())
        return categorized

    def clear_logs(self, category: str = None, source: str = None) -> None:
        """Clear logs, optionally for a specific category and/or source."""
        with self._lock:
            if category is None and source is None:
                self._log_buffer.clear()
                return

            def keep(entry: LogEntry) -> bool:
                if category and entry.category == category:
                    if source is None or entry.source == source:
                        return False
                if source and entry.source == source and category is None:
                    return False
                return True

            self._log_buffer = deque(
                (log for log in self._log_buffer if keep(log)),
                maxlen=self.max_entries,
            )

    def get_log_count(self, source: str = None) -> int:
        """Get total number of log entries."""
        with self._lock:
            if source:
                return sum(1 for log in self._log_buffer if log.source == source)
            return len(self._log_buffer)

    def get_latest_sequence(self) -> int:
        with self._lock:
            return self._sequence

    def get_categories(self, source: str = None) -> List[str]:
        """Get list of all log categories."""
        with self._lock:
            categories = {
                log.category
                for log in self._log_buffer
                if not source or log.source == source
            }
            return sorted(categories)

    def get_sources(self) -> List[str]:
        """Get list of all log sources."""
        with self._lock:
            return sorted({log.source for log in self._log_buffer})


# Global instance for backward compatibility
_global_log_manager = None
_global_log_lock = threading.Lock()

def get_log_manager(max_entries: int = 2000) -> LogManager:
    """Get or create global log manager instance."""
    global _global_log_manager
    if _global_log_manager is None:
        with _global_log_lock:
            if _global_log_manager is None:
                _global_log_manager = LogManager(max_entries)
    elif max_entries and max_entries > _global_log_manager.max_entries:
        _global_log_manager.max_entries = max_entries
    return _global_log_manager



