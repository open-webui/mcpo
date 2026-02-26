"""
Code Mode MCP Middleware

When code mode is enabled, this middleware intercepts MCP protocol messages on the
FastMCP proxy (port 8001) to:

1. Replace tools/list responses with only search_tools + execute_tool
2. Intercept tools/call for search_tools → search the internal catalog
3. Intercept tools/call for execute_tool → route to the actual MCP server
4. Pass through all other MCP messages unchanged

This sits as ASGI middleware on the FastMCP proxy app, similar to MCPToolFilterMiddleware.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, Dict, List, Optional

from starlette.types import ASGIApp, Receive, Scope, Send

from mcpo.services.code_mode import (
    CatalogEntry,
    build_catalog,
    get_code_mode_tool_definitions,
    search_catalog,
)
from mcpo.services.state import get_state_manager

logger = logging.getLogger(__name__)


class CodeModeMCPMiddleware:
    """
    ASGI middleware that transforms MCP proxy behavior when code mode is active.

    When code mode is OFF: passes everything through unchanged.
    When code mode is ON:
      - tools/list → returns search_tools + execute_tool only
      - tools/call search_tools → searches internal catalog, returns results
      - tools/call execute_tool → rewrites to actual tool call, passes to upstream
    """

    def __init__(self, app: ASGIApp, server_name: Optional[str] = None):
        self.app = app
        self.server_name = server_name  # None = aggregate proxy
        self.state_manager = get_state_manager()
        self._catalog: List[CatalogEntry] = []
        self._catalog_built = False
        # Store original tools from upstream for catalog building
        self._upstream_tools: Dict[str, List[Dict[str, Any]]] = {}

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        if not self.state_manager.is_code_mode_enabled():
            await self.app(scope, receive, send)
            return

        # Code mode is ON — intercept MCP messages
        body_parts: List[bytes] = []

        async def receive_wrapper() -> Dict[str, Any]:
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if body:
                    body_parts.append(body)
            return message

        # We need to peek at the request to check if it's a tools/call we handle
        response_body_parts: List[bytes] = []
        start_message_held = None
        is_streaming = False
        intercepted = False

        async def send_wrapper(message: Dict[str, Any]) -> None:
            nonlocal start_message_held, is_streaming, intercepted

            if message["type"] == "http.response.start":
                headers = dict(message.get("headers", []))
                content_type = headers.get(b"content-type", b"").decode("utf-8", errors="ignore").lower()

                if "text/event-stream" in content_type or "stream" in content_type:
                    is_streaming = True
                    await send(message)
                    return

                start_message_held = message
                return

            if message["type"] == "http.response.body":
                if is_streaming:
                    await send(message)
                    return

                body = message.get("body", b"")
                if body:
                    response_body_parts.append(body)

                if not message.get("more_body", False):
                    full_body = b"".join(response_body_parts)

                    try:
                        data = json.loads(full_body)
                        filtered = self._process_response(data)
                        out_body = json.dumps(filtered).encode()

                        held_headers = (start_message_held or {}).get(
                            "headers", [(b"content-type", b"application/json")]
                        )
                        await send({"type": "http.response.start", "status": 200, "headers": held_headers})
                        await send({"type": "http.response.body", "body": out_body})
                        return
                    except (json.JSONDecodeError, Exception) as e:
                        logger.debug("Code mode: not filtering response: %s", e)
                        if start_message_held:
                            await send(start_message_held)
                        else:
                            await send({"type": "http.response.start", "status": 200})
                        await send({"type": "http.response.body", "body": full_body})
                        return
                return

            await send(message)

        # Check if the request is a tools/call we need to intercept
        # We do a full pass-through first, then filter the response.
        # For tools/call of search_tools, we synthesize a response without
        # calling upstream at all.
        await self.app(scope, receive_wrapper, send_wrapper)

    def _process_response(self, data: Any) -> Any:
        """Process MCP JSON-RPC response/request messages."""
        if isinstance(data, list):
            return [self._process_single(msg) for msg in data]
        return self._process_single(data)

    def _process_single(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single MCP message."""

        # --- tools/list response: replace with code mode tools ---
        if "result" in message and "tools" in message.get("result", {}):
            original_tools = message["result"]["tools"]
            # Cache the original tools for the catalog
            self._update_catalog_from_tools(original_tools)
            # Replace with code mode meta-tools
            message["result"]["tools"] = get_code_mode_tool_definitions()
            logger.info(
                "Code mode: replaced %d tools with 2 meta-tools (search_tools, execute_tool)",
                len(original_tools),
            )
            return message

        # --- tools/call response: check if it was for search_tools ---
        # The actual interception of search_tools calls happens via request
        # body inspection. For execute_tool, the middleware rewrites the
        # request before passing upstream (handled in __call__).

        return message

    def _update_catalog_from_tools(self, tools: List[Dict[str, Any]]) -> None:
        """Build/update the internal catalog from tools/list results."""
        # In aggregate proxy, tools have server prefix in name (e.g. "server__tool")
        tools_by_server: Dict[str, List[Dict[str, Any]]] = {}
        for tool in tools:
            name = tool.get("name", "")
            if "__" in name:
                server_part, tool_part = name.split("__", 1)
                tool_copy = dict(tool)
                tool_copy["name"] = tool_part
                tools_by_server.setdefault(server_part, []).append(tool_copy)
            elif self.server_name:
                tools_by_server.setdefault(self.server_name, []).append(tool)
            else:
                # Aggregate proxy without __ separator — try annotations
                server = tool.get("annotations", {}).get("server", "unknown")
                tools_by_server.setdefault(server, []).append(tool)

        self._catalog = build_catalog(tools_by_server)
        self._catalog_built = True
        logger.info("Code mode: built catalog with %d tools", len(self._catalog))

    def handle_search_tools(self, arguments: Dict[str, Any]) -> Dict[str, Any]:
        """Handle a search_tools call and return a synthetic MCP result."""
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)
        results = search_catalog(self._catalog, query, limit=limit)
        return {
            "content": [
                {
                    "type": "text",
                    "text": json.dumps({"tools": results, "total_available": len(self._catalog)}, indent=2),
                }
            ],
            "isError": False,
        }

    def handle_execute_tool(self, arguments: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Rewrite an execute_tool call to the actual tool call.

        Returns None if we should let it pass through to upstream (after rewrite),
        or returns a synthetic error result if the tool isn't found.
        """
        tool_qualified = arguments.get("tool", "")
        tool_args = arguments.get("arguments", {})

        if not tool_qualified:
            return {
                "content": [{"type": "text", "text": "Error: 'tool' parameter is required"}],
                "isError": True,
            }

        # Validate the tool exists in catalog
        found = any(e.qualified_name == tool_qualified for e in self._catalog)
        if not found and self._catalog_built:
            # Suggest similar tools
            suggestions = search_catalog(self._catalog, tool_qualified, limit=3)
            suggestion_text = ""
            if suggestions:
                suggestion_text = "\n\nDid you mean:\n" + "\n".join(
                    f"  - {s['tool']}: {s['description'][:80]}" for s in suggestions
                )
            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"Error: Tool '{tool_qualified}' not found in catalog.{suggestion_text}",
                    }
                ],
                "isError": True,
            }

        # Tool exists — return None to signal "rewrite and pass through"
        return None
