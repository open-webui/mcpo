"""
Domain models for core entities: servers, tools, sessions, and state.
Centralizes all business logic models in one location.
"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Set
from enum import Enum
from pydantic import BaseModel, Field, field_validator


class ServerType(str, Enum):
    """Supported MCP server types."""
    STDIO = "stdio"
    SSE = "sse"
    STREAMABLE_HTTP = "streamable-http"


class ServerState(str, Enum):
    """Server connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    ERROR = "error"


class ToolState(str, Enum):
    """Tool execution states."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    ERROR = "error"


class ServerInfo(BaseModel):
    """MCP server information from initialization."""
    name: str
    version: Optional[str] = None
    description: Optional[str] = None
    instructions: Optional[str] = None


class ToolInfo(BaseModel):
    """MCP tool information."""
    name: str
    description: Optional[str] = None
    input_schema: Dict[str, Any] = Field(default_factory=dict)
    output_schema: Optional[Dict[str, Any]] = None


class ServerConfiguration(BaseModel):
    """Runtime server configuration."""
    name: str
    server_type: ServerType
    command: Optional[str] = None
    args: List[str] = Field(default_factory=list)
    url: Optional[str] = None
    env: Dict[str, str] = Field(default_factory=dict)
    headers: Dict[str, str] = Field(default_factory=dict)
    timeout: Optional[float] = None

    @field_validator('command')
    @classmethod
    def validate_stdio_command(cls, v, info):
        if info.data.get('server_type') == ServerType.STDIO and not v:
            raise ValueError('stdio servers require a command')
        return v

    @field_validator('url')
    @classmethod
    def validate_remote_url(cls, v, info):
        server_type = info.data.get('server_type')
        if server_type in (ServerType.SSE, ServerType.STREAMABLE_HTTP) and not v:
            raise ValueError('remote servers require a URL')
        return v


class ServerRuntimeState(BaseModel):
    """Runtime state of a server."""
    name: str
    config: ServerConfiguration
    state: ServerState = ServerState.DISCONNECTED
    enabled: bool = True
    tools: Dict[str, ToolState] = Field(default_factory=dict)
    server_info: Optional[ServerInfo] = None
    last_error: Optional[str] = None
    connected_at: Optional[int] = None
    metrics: Dict[str, Any] = Field(default_factory=dict)

    def enable_tool(self, tool_name: str) -> None:
        """Enable a specific tool."""
        self.tools[tool_name] = ToolState.ENABLED

    def disable_tool(self, tool_name: str) -> None:
        """Disable a specific tool."""
        self.tools[tool_name] = ToolState.DISABLED

    def set_connected(self, server_info: Optional[ServerInfo] = None) -> None:
        """Mark server as connected."""
        self.state = ServerState.CONNECTED
        self.connected_at = int(time.time() * 1000)
        if server_info:
            self.server_info = server_info

    def set_error(self, error_message: str) -> None:
        """Mark server as having an error."""
        self.state = ServerState.ERROR
        self.last_error = error_message

    def get_enabled_tools(self) -> List[str]:
        """Get list of enabled tools."""
        return [
            tool_name for tool_name, state in self.tools.items()
            if state == ToolState.ENABLED
        ]

    def get_tool_count(self) -> Dict[str, int]:
        """Get tool counts by state."""
        total = len(self.tools)
        enabled = len(self.get_enabled_tools())
        return {
            'total': total,
            'enabled': enabled,
            'disabled': total - enabled
        }


class LogEntry(BaseModel):
    """Log entry for UI display."""
    timestamp: int = Field(default_factory=lambda: int(time.time() * 1000))
    level: str
    message: str
    category: Optional[str] = None
    server: Optional[str] = None
    tool: Optional[str] = None


class SessionMetrics(BaseModel):
    """Session execution metrics."""
    tool_calls: Dict[str, int] = Field(default_factory=dict)
    latencies: Dict[str, List[float]] = Field(default_factory=dict)
    errors: Dict[str, int] = Field(default_factory=dict)
    start_time: int = Field(default_factory=lambda: int(time.time() * 1000))

    def record_call(self, tool_name: str, latency_ms: float, success: bool = True) -> None:
        """Record a tool call with metrics."""
        # Update call count
        self.tool_calls[tool_name] = self.tool_calls.get(tool_name, 0) + 1
        
        # Record latency
        if tool_name not in self.latencies:
            self.latencies[tool_name] = []
        self.latencies[tool_name].append(latency_ms)
        
        # Record error if unsuccessful
        if not success:
            self.errors[tool_name] = self.errors.get(tool_name, 0) + 1

    def get_avg_latency(self, tool_name: str) -> float:
        """Get average latency for a tool."""
        latencies = self.latencies.get(tool_name, [])
        return sum(latencies) / len(latencies) if latencies else 0.0

    def get_summary(self) -> Dict[str, Any]:
        """Get metrics summary."""
        total_calls = sum(self.tool_calls.values())
        total_errors = sum(self.errors.values())
        
        per_tool = {}
        for tool_name in self.tool_calls:
            per_tool[tool_name] = {
                'calls': self.tool_calls[tool_name],
                'avgLatencyMs': self.get_avg_latency(tool_name),
                'errors': self.errors.get(tool_name, 0)
            }
        
        return {
            'totalCalls': total_calls,
            'totalErrors': total_errors,
            'successRate': (total_calls - total_errors) / total_calls if total_calls > 0 else 0,
            'perTool': per_tool,
            'uptimeMs': int(time.time() * 1000) - self.start_time
        }
