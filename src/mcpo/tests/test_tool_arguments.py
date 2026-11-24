from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from mcp.types import CallToolResult, TextContent

from mcpo.utils.main import get_model_fields, get_tool_handler

SCHEMA = {
    "type": "object",
    "properties": {
        "text": {"type": "string"},
        "timezone": {
            "anyOf": [{"type": "string"}, {"type": "null"}],
            "default": "UTC",
        },
        "note": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "limit": {"type": "integer"},
        "repeat": {
            "type": "object",
            "properties": {
                "every": {"type": "string"},
                "until": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            },
        },
        "attendees": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "email": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
            },
        },
    },
    "required": ["text"],
}


async def _call_tool(body):
    session = AsyncMock()
    session.call_tool.return_value = CallToolResult(
        content=[TextContent(type="text", text="ok")], isError=False
    )
    request = SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(session=session))
    )
    fields = get_model_fields(
        "set_reminder_form_model", SCHEMA["properties"], SCHEMA["required"], SCHEMA
    )
    tool = get_tool_handler("set_reminder", fields)
    form_model = tool.__annotations__["form_data"]

    await tool(form_model.model_validate(body), request)

    session.call_tool.assert_awaited_once()
    return session.call_tool.await_args.kwargs["arguments"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        # Unset optional arguments are not sent as null, and defaults are filled in.
        ({"text": "stand-up"}, {"text": "stand-up", "timezone": "UTC"}),
        # An explicit null is passed on instead of being replaced by the default.
        (
            {"text": "stand-up", "timezone": None, "note": None},
            {"text": "stand-up", "timezone": None, "note": None},
        ),
        (
            {"text": "stand-up", "repeat": {"every": "day", "until": None}},
            {
                "text": "stand-up",
                "timezone": "UTC",
                "repeat": {"every": "day", "until": None},
            },
        ),
        (
            {"text": "stand-up", "repeat": {"every": "day"}},
            {"text": "stand-up", "timezone": "UTC", "repeat": {"every": "day"}},
        ),
        (
            {
                "text": "stand-up",
                "attendees": [{"name": "Asha"}, {"name": "Ravi", "email": None}],
            },
            {
                "text": "stand-up",
                "timezone": "UTC",
                "attendees": [{"name": "Asha"}, {"name": "Ravi", "email": None}],
            },
        ),
    ],
)
async def test_tool_arguments_keep_explicit_nulls(body, expected):
    assert await _call_tool(body) == expected
