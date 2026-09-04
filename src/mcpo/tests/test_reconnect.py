from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException
from mcp.shared.exceptions import McpError
from mcp.types import CallToolResult, ErrorData, INTERNAL_ERROR, INVALID_REQUEST, TextContent

# The bare positive code the python-sdk's StreamableHTTPTransport actually
# hardcodes for a synthesized session-terminated error — NOT the negative
# mcp.types.INVALID_REQUEST value (see _is_session_terminated in mcpo.utils.main).
SESSION_TERMINATED_CODE = 32600

from mcpo.utils.main import get_tool_handler


def _make_request(session_manager):
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(session_manager=session_manager))
    )


def _success_result(text="ok"):
    return CallToolResult(content=[TextContent(type="text", text=text)], isError=False)


@pytest.mark.asyncio
async def test_reconnects_on_synthesized_session_terminated_error():
    """The python-sdk synthesizes McpError(INVALID_REQUEST, "Session terminated")
    when the server 404s an existing session_id (see streamable_http.py). mcpo must
    treat this the same as a ClosedResourceError and retry once against a fresh session."""
    stale_session = AsyncMock()
    stale_session.call_tool.side_effect = McpError(
        ErrorData(code=SESSION_TERMINATED_CODE, message="Session terminated")
    )

    fresh_session = AsyncMock()
    fresh_session.call_tool.return_value = _success_result()

    session_manager = AsyncMock()
    session_manager.ensure_initialized.return_value = (stale_session, None)
    session_manager.reconnect.return_value = (fresh_session, None)

    tool = get_tool_handler("do_thing", form_model_fields=None)
    request = _make_request(session_manager)

    result = await tool(request)

    assert result == "ok"
    session_manager.reconnect.assert_awaited_once()
    fresh_session.call_tool.assert_awaited_once_with("do_thing", arguments={})


@pytest.mark.asyncio
async def test_does_not_reconnect_on_unrelated_mcp_error():
    """A real tool-level/server-originated McpError must propagate as-is, not
    trigger a reconnect+retry."""
    session = AsyncMock()
    session.call_tool.side_effect = McpError(
        ErrorData(code=INTERNAL_ERROR, message="tool blew up")
    )

    session_manager = AsyncMock()
    session_manager.ensure_initialized.return_value = (session, None)

    tool = get_tool_handler("do_thing", form_model_fields=None)
    request = _make_request(session_manager)

    with pytest.raises(HTTPException) as exc_info:
        await tool(request)

    assert exc_info.value.status_code == 500
    session_manager.reconnect.assert_not_awaited()


@pytest.mark.asyncio
async def test_does_not_reconnect_on_real_invalid_request_error():
    """A genuine JSON-RPC INVALID_REQUEST (-32600) with a different message is
    a real protocol error, not the synthesized session-terminated signal —
    must not trigger a reconnect."""
    session = AsyncMock()
    session.call_tool.side_effect = McpError(
        ErrorData(code=INVALID_REQUEST, message="Malformed request")
    )

    session_manager = AsyncMock()
    session_manager.ensure_initialized.return_value = (session, None)

    tool = get_tool_handler("do_thing", form_model_fields=None)
    request = _make_request(session_manager)

    with pytest.raises(HTTPException) as exc_info:
        await tool(request)

    assert exc_info.value.status_code == 400
    session_manager.reconnect.assert_not_awaited()
