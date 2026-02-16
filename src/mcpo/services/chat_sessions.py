from __future__ import annotations

import asyncio
import os
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


@dataclass
class ChatStep:
    """Represents a high-level step in an agentic exchange."""

    id: str
    type: str
    title: str
    detail: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "title": self.title,
            "detail": self.detail,
            "createdAt": self.created_at.isoformat(),
        }


@dataclass
class ChatSession:
    """In-memory representation of a chat session."""

    id: str
    model: str
    system_prompt: Optional[str]
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    messages: List[Dict[str, Any]] = field(default_factory=list)
    steps: List[ChatStep] = field(default_factory=list)
    tool_definitions: List[Dict[str, Any]] = field(default_factory=list)
    tool_index: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    server_allowlist: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "model": self.model,
            "createdAt": self.created_at.isoformat(),
            "lastAccessed": self.last_accessed.isoformat(),
            "systemPrompt": self.system_prompt,
            "messages": list(self.messages),
            "steps": [step.to_dict() for step in self.steps],
            "tools": list(self.tool_definitions),
            "serverAllowlist": list(self.server_allowlist) if self.server_allowlist else None,
        }

    def reset(self) -> None:
        preserved_system: List[Dict[str, Any]] = []
        if self.system_prompt:
            preserved_system.append({"role": "system", "content": self.system_prompt})
        self.messages = preserved_system
        self.steps.clear()
        self.tool_definitions.clear()
        self.tool_index.clear()


class ChatSessionManager:
    """Thread-safe session manager for chat interactions."""

    MAX_SESSIONS = int(os.environ.get("MCPO_MAX_SESSIONS", "100"))
    SESSION_TTL_SECONDS = int(os.environ.get("MCPO_SESSION_TTL_SECONDS", "3600"))

    def __init__(self) -> None:
        self._sessions: Dict[str, ChatSession] = {}
        self._lock = asyncio.Lock()

    def _cleanup_expired(self) -> None:
        """Remove sessions older than TTL. Must be called while holding self._lock."""
        now = datetime.now(timezone.utc)
        expired = [
            sid for sid, s in self._sessions.items()
            if (now - s.last_accessed).total_seconds() > self.SESSION_TTL_SECONDS
        ]
        for sid in expired:
            del self._sessions[sid]

    async def create_session(
        self,
        *,
        model: str,
        system_prompt: Optional[str] = None,
        tool_definitions: Optional[List[Dict[str, Any]]] = None,
        tool_index: Optional[Dict[str, Dict[str, Any]]] = None,
        server_allowlist: Optional[List[str]] = None,
    ) -> ChatSession:
        session_id = uuid.uuid4().hex
        messages: List[Dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        session = ChatSession(
            id=session_id,
            model=model,
            system_prompt=system_prompt,
            messages=messages,
            tool_definitions=tool_definitions or [],
            tool_index=tool_index or {},
            server_allowlist=list(server_allowlist) if server_allowlist else None,
        )

        async with self._lock:
            self._cleanup_expired()
            if len(self._sessions) >= self.MAX_SESSIONS:
                oldest_id = min(self._sessions, key=lambda sid: self._sessions[sid].last_accessed)
                del self._sessions[oldest_id]
            self._sessions[session_id] = session
        return session

    async def get_session(self, session_id: str) -> ChatSession:
        async with self._lock:
            try:
                session = self._sessions[session_id]
                session.last_accessed = datetime.now(timezone.utc)
                return session
            except KeyError as exc:
                raise KeyError(f"Chat session '{session_id}' not found") from exc

    async def delete_session(self, session_id: str) -> None:
        async with self._lock:
            self._sessions.pop(session_id, None)

    async def reset_session(self, session_id: str) -> ChatSession:
        async with self._lock:
            if session_id not in self._sessions:
                raise KeyError(f"Chat session '{session_id}' not found")
            self._sessions[session_id].reset()
            return self._sessions[session_id]

    async def list_sessions(self) -> List[ChatSession]:
        async with self._lock:
            return list(self._sessions.values())


_session_manager: Optional[ChatSessionManager] = None
_session_lock = asyncio.Lock()


async def get_chat_session_manager() -> ChatSessionManager:
    global _session_manager
    if _session_manager is None:
        async with _session_lock:
            if _session_manager is None:
                _session_manager = ChatSessionManager()
    return _session_manager
