"""
MCP Tool Filtering Middleware

Intercepts MCP protocol messages to filter tools based on mcpo_state.json.
This ensures tool enable/disable toggles work on the FastMCP proxy (port 8001).
"""

import json
import logging
from typing import Any, Dict, List

from starlette.types import ASGIApp, Receive, Scope, Send

from mcpo.services.state import get_state_manager

logger = logging.getLogger(__name__)


class MCPToolFilterMiddleware:
    """
    ASGI middleware that filters MCP tools based on StateManager.
    
    Intercepts:
    - tools/list responses → filters disabled tools
    - tools/call requests → blocks disabled tools with 403
    """
    
    def __init__(self, app: ASGIApp, server_name: str = None):
        self.app = app
        self.server_name = server_name  # None = aggregate proxy (multi-server)
        self.state_manager = get_state_manager()
    
    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """ASGI interface - intercept and filter MCP messages."""
        
        if scope["type"] != "http":
            # Only process HTTP requests
            await self.app(scope, receive, send)
            return
        
        # Capture request body to check for MCP method
        body_parts: List[bytes] = []
        
        async def receive_wrapper() -> Dict[str, Any]:
            """Capture request body for inspection."""
            message = await receive()
            if message["type"] == "http.request":
                body = message.get("body", b"")
                if body:
                    body_parts.append(body)
            return message
        
        # Track response state for filtering
        response_started = False
        response_body_parts: List[bytes] = []
        is_streaming = False
        start_message_held = None
        
        async def send_wrapper(message: Dict[str, Any]) -> None:
            """Intercept response to filter tools."""
            nonlocal response_started, is_streaming, start_message_held
            
            if message["type"] == "http.response.start":
                response_started = True
                
                # Check if this is a streaming response (SSE or chunked)
                headers = dict(message.get("headers", []))
                content_type = headers.get(b"content-type", b"").decode("utf-8", errors="ignore").lower()
                
                # Pass through SSE and streaming responses without modification
                if "text/event-stream" in content_type or "stream" in content_type:
                    is_streaming = True
                    await send(message)
                    return
                
                # Hold the start message for potential JSON filtering
                start_message_held = message
                return
            
            if message["type"] == "http.response.body":
                # If streaming, pass through immediately
                if is_streaming:
                    await send(message)
                    return
                
                body = message.get("body", b"")
                if body:
                    response_body_parts.append(body)
                
                # If this is the last chunk, process and send
                if not message.get("more_body", False):
                    full_body = b"".join(response_body_parts)
                    
                    # Try to parse as MCP JSON-RPC
                    try:
                        data = json.loads(full_body)
                        filtered_data = self._filter_mcp_message(data)
                        filtered_body = json.dumps(filtered_data).encode()
                        
                        # Send the start message we held, preserving original headers
                        held_headers = start_message_held.get("headers", [(b"content-type", b"application/json")])
                        await send({"type": "http.response.start", "status": start_message_held.get("status", 200), "headers": held_headers})
                        
                        # Send filtered body
                        await send({"type": "http.response.body", "body": filtered_body})
                        return
                    except (json.JSONDecodeError, Exception) as e:
                        logger.debug(f"Not filtering response: {e}")
                        # Not JSON or error - pass through as-is with original start message
                        if start_message_held:
                            await send(start_message_held)
                        else:
                            await send({"type": "http.response.start", "status": start_message_held.get("status", 200) if start_message_held else 200})
                        await send({"type": "http.response.body", "body": full_body})
                        return
                return
            
            # Pass through other messages
            await send(message)
        
        # Call the wrapped app
        await self.app(scope, receive_wrapper, send_wrapper)
    
    def _filter_mcp_message(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Filter MCP protocol messages based on tool enabled state."""
        
        # Handle JSON-RPC batch
        if isinstance(data, list):
            return [self._filter_single_message(msg) for msg in data]
        
        return self._filter_single_message(data)
    
    def _filter_single_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Filter a single MCP message."""
        
        # Check if this is a tools/list response
        if "result" in message and "tools" in message.get("result", {}):
            tools = message["result"]["tools"]
            
            if self.server_name:
                # Single server filtering
                filtered_tools = [
                    tool for tool in tools
                    if self.state_manager.is_tool_enabled(self.server_name, tool.get("name", ""))
                ]
                logger.info(
                    f"Filtered tools for '{self.server_name}': "
                    f"{len(tools)} → {len(filtered_tools)} tools"
                )
            else:
                # Aggregate proxy - infer server from tool name
                filtered_tools = []
                for tool in tools:
                    tool_name = tool.get("name", "")
                    # Try to infer server from tool name (format: "server__tool" or just "tool")
                    if "__" in tool_name:
                        server_part = tool_name.split("__")[0]
                    else:
                        # Check description or annotations for server hint
                        server_part = tool.get("annotations", {}).get("server")
                        if not server_part:
                            # Fallback: check all servers for this tool
                            server_part = self._find_server_for_tool(tool_name)
                    
                    if server_part and self.state_manager.is_tool_enabled(server_part, tool_name):
                        filtered_tools.append(tool)
                
                logger.info(
                    f"Filtered tools for aggregate proxy: "
                    f"{len(tools)} → {len(filtered_tools)} tools"
                )
            
            message["result"]["tools"] = filtered_tools
            return message
        
        # Check if this is a tools/call request (shouldn't happen in response, but defensive)
        if "method" in message and message.get("method") == "tools/call":
            params = message.get("params", {})
            tool_name = params.get("name", "")
            
            server_to_check = self.server_name
            if not server_to_check and "__" in tool_name:
                server_to_check = tool_name.split("__")[0]
            elif not server_to_check:
                server_to_check = self._find_server_for_tool(tool_name)
            
            if server_to_check and not self.state_manager.is_tool_enabled(server_to_check, tool_name):
                logger.warning(f"Blocked call to disabled tool: {server_to_check}/{tool_name}")
                return {
                    "jsonrpc": "2.0",
                    "id": message.get("id"),
                    "error": {
                        "code": 403,
                        "message": f"Tool '{tool_name}' is disabled",
                        "data": {"tool": tool_name, "server": server_to_check}
                    }
                }
        
        return message
    
    def _find_server_for_tool(self, tool_name: str) -> str:
        """Find which server a tool belongs to by checking state."""
        state = self.state_manager.get_all_states()
        for server_name, server_data in state.items():
            if tool_name in server_data.get("tools", {}):
                return server_name
        return ""
