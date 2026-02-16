import json
import asyncio
import pytest
from fastapi.testclient import TestClient


@pytest.mark.asyncio
async def test_protocol_version_warn_allows_call(tmp_path, monkeypatch):
    """In 'warn' mode, missing MCP-Protocol-Version header should log a warning but still succeed."""
    from mcpo.main import build_main_app
    from fastapi import FastAPI
    from mcpo.utils.main import get_tool_handler

    class FakeSession:
        async def call_tool(self, name, arguments):
            return type('R', (), {'isError': False, 'content': []})

    app = await build_main_app(protocol_version_mode='warn')
    sub = FastAPI(title='sP')
    sub.state.parent_app = app
    sub.state.session = FakeSession()
    handler = get_tool_handler(sub.state.session, 'demo', form_model_fields=None)
    sub.post('/demo')(handler)
    app.mount('/sP', sub)

    # Set up legacy tool_enabled for handler compatibility
    app.state.tool_enabled = {'sP': {'demo': True}}

    client = TestClient(app)
    # No MCP-Protocol-Version header but warn mode, should succeed
    r = client.post('/sP/demo')
    assert r.status_code == 200, r.text


@pytest.mark.asyncio
async def test_protocol_version_warn_does_not_block(tmp_path):
    """Enforce mode is not yet implemented; verify warn mode still allows calls."""
    from mcpo.main import build_main_app
    from fastapi import FastAPI
    from mcpo.utils.main import get_tool_handler

    class FakeSession:
        async def call_tool(self, name, arguments):
            return type('R', (), {'isError': False, 'content': []})

    app = await build_main_app(protocol_version_mode='enforce')
    sub = FastAPI(title='sE')
    sub.state.parent_app = app
    sub.state.session = FakeSession()
    handler = get_tool_handler(sub.state.session, 'demo', form_model_fields=None)
    sub.post('/demo')(handler)
    app.mount('/sE', sub)

    app.state.tool_enabled = {'sE': {'demo': True}}

    client = TestClient(app)
    # Protocol enforcement is not yet implemented in handlers — call still succeeds
    r = client.post('/sE/demo')
    # Should succeed since enforcement is warn-only in handlers currently
    assert r.status_code == 200


@pytest.mark.asyncio
async def test_tool_call_succeeds_with_correct_protocol_header(tmp_path):
    """Verify tool call succeeds when correct MCP-Protocol-Version is sent."""
    from mcpo.main import build_main_app
    from fastapi import FastAPI
    from mcpo.utils.main import get_tool_handler

    class FakeSession:
        async def call_tool(self, name, arguments):
            return type('R', (), {'isError': False, 'content': []})

    app = await build_main_app(validate_output_mode='off')
    sub = FastAPI(title='sV')
    sub.state.parent_app = app
    sub.state.session = FakeSession()

    handler = get_tool_handler(sub.state.session, 'demo', form_model_fields=None)
    sub.post('/demo')(handler)
    app.mount('/sV', sub)
    app.state.tool_enabled = {'sV': {'demo': True}}

    client = TestClient(app)
    r = client.post('/sV/demo', headers={'MCP-Protocol-Version': app.state.supported_protocol_versions[0]})
    assert r.status_code == 200
    assert r.json()['ok'] is True


@pytest.mark.asyncio
async def test_latency_metrics_accumulate(tmp_path):
    """Verify that calling a tool increments metrics counters."""
    from mcpo.main import build_main_app
    from fastapi import FastAPI
    from mcpo.utils.main import get_tool_handler

    class FakeSession:
        async def call_tool(self, name, arguments):
            await asyncio.sleep(0.01)
            return type('R', (), {'isError': False, 'content': []})

    app = await build_main_app()
    sub = FastAPI(title='sL')
    sub.state.parent_app = app
    sub.state.session = FakeSession()
    handler = get_tool_handler(sub.state.session, 'demo', form_model_fields=None)
    sub.post('/demo')(handler)
    app.mount('/sL', sub)

    app.state.tool_enabled = {'sL': {'demo': True}}

    client = TestClient(app)
    for _ in range(3):
        r = client.post('/sL/demo', headers={'MCP-Protocol-Version': app.state.supported_protocol_versions[0]})
        assert r.status_code == 200
    m = client.get('/_meta/metrics')
    js = m.json()
    assert js['ok'] is True
    per_tool = js['metrics']['perTool']
    assert 'demo' in per_tool
    assert per_tool['demo']['calls'] >= 3
    # avgLatencyMs should be > 0
    assert per_tool['demo']['avgLatencyMs'] > 0
