from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from fastapi import FastAPI
from starlette.routing import Mount

from mcpo.services.state import get_state_manager

logger = logging.getLogger(__name__)


def collect_enabled_mcp_sessions(
    app: FastAPI,
    allowlist: Optional[List[str]] = None,
    connected_only: bool = True,
) -> List[Any]:
    """Collect active MCP ClientSession objects from mounted sub-apps."""
    return [session for _, session in collect_enabled_mcp_sessions_with_names(app, allowlist, connected_only)]


def collect_enabled_mcp_sessions_with_names(
    app: FastAPI,
    allowlist: Optional[List[str]] = None,
    connected_only: bool = True,
) -> List[Tuple[str, Any]]:
    """
    Collect (server_name, session) tuples from mounted sub-apps.

    - Iterates FastAPI mounts on the main app
    - Skips FastMCP proxy mounts (internal proxies)
    - Respects enabled/disabled state from the state manager
    - Optionally filters by 'allowlist' of server names
    - Optionally requires connected_only sessions

    Returns:
        List of (server_name, session) tuples.
    """
    out: List[Tuple[str, Any]] = []
    state_manager = get_state_manager()

    for route in app.router.routes:
        if not isinstance(route, Mount):
            continue
        sub_app = getattr(route, "app", None)
        # Accept any app-like object with a 'state' attribute (friendly to tests using MagicMock)
        if sub_app is None or not hasattr(sub_app, "state"):
            continue

        # Skip FastMCP proxy apps; they are not user-facing servers
        if getattr(sub_app.state, "is_fastmcp_proxy", False):
            continue

        # Identify server name from mount path
        server_name = route.path.strip("/")

        # Add logging for skipped servers
        if allowlist is not None and server_name not in allowlist:
            logger.info(f"MCP session for '{server_name}' skipped: not in allowlist.")
            continue

        # Respect server enabled state from state manager
        if not state_manager.is_server_enabled(server_name):
            logger.info(f"MCP session for '{server_name}' skipped: server disabled in state manager.")
            continue

        # Connected filter (default True)
        if connected_only and not getattr(sub_app.state, "is_connected", False):
            logger.warning(f"MCP session for '{server_name}' skipped: server not connected.")
            continue

        # Grab the MCP session if present
        session = getattr(sub_app.state, "session", None)
        if session is not None:
            out.append((server_name, session))

    return out


def sanitize_tool_name(name: str) -> str:
    """Return an OpenRouter-compliant tool name (alphanumeric, '_', '-')."""
    if not name:
        return "tool"
    sanitized = re.sub(r"[^0-9A-Za-z_-]", "_", name)
    return sanitized or "tool"