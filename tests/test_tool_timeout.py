import asyncio
from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcpo.utils.main import get_tool_handler
from mcp import types

class SleepSession:
    def __init__(self, delay: float):
        self.delay = delay
    async def call_tool(self, endpoint_name, arguments):
        await asyncio.sleep(self.delay)
        return types.CallToolResult(content=[types.TextContent(type="text", text="done")], isError=False)


def build_app(default_timeout=1, max_timeout=5, delay=2, structured=False):
    app = FastAPI()
    app.state.tool_timeout = default_timeout
    app.state.tool_timeout_max = max_timeout
    app.state.structured_output = structured
    handler = get_tool_handler(SleepSession(delay), 'slow', form_model_fields=None, response_model_fields=None)
    app.post('/slow')(handler)
    return app


def test_timeout_triggers_504():
    app = build_app(default_timeout=0.05, max_timeout=5, delay=0.2)
    client = TestClient(app)
    resp = client.post('/slow')
    assert resp.status_code == 504
    body = resp.json()
    assert body['ok'] is False
    assert body['error']['code'] == 'timeout'
    assert body['error']['message'] == 'Tool timed out'


def test_override_header_allows_longer():
    app = build_app(default_timeout=0.05, max_timeout=5, delay=0.1)
    client = TestClient(app)
    # override to 0.2 which is > delay
    resp = client.post('/slow', headers={'X-Tool-Timeout': '1'})
    assert resp.status_code == 200
    body = resp.json()
    assert body['ok'] is True
    assert body['result'] == 'done'


def test_override_query_param():
    app = build_app(default_timeout=0.05, max_timeout=5, delay=0.05)
    client = TestClient(app)
    resp = client.post('/slow?timeout=1')
    assert resp.status_code == 200
    assert resp.json()['result'] == 'done'


def test_timeout_exceeds_max_rejected():
    app = build_app(default_timeout=0.05, max_timeout=1, delay=0.05)
    client = TestClient(app)
    resp = client.post('/slow?timeout=5')
    assert resp.status_code == 400
    body = resp.json()
    # Bare FastAPI returns {"detail": {...}}, full main_app returns {"error": {...}}
    detail = body.get('detail', body.get('error', {}))
    assert detail['code'] == 'invalid_timeout'
    assert detail['message'] == 'Timeout out of allowed range'


def test_invalid_timeout_value():
    app = build_app(default_timeout=0.05, max_timeout=1, delay=0.05)
    client = TestClient(app)
    resp = client.post('/slow?timeout=abc')
    assert resp.status_code == 400
    body = resp.json()
    # Bare FastAPI without custom exception handler returns {"detail": {...}}
    # while the full main_app would wrap it in {"error": {...}}.
    detail = body.get('detail', body.get('error', {}))
    assert detail.get('code') == 'invalid_timeout' or (
        isinstance(detail, dict) and detail.get('code') == 'invalid_timeout'
    )

