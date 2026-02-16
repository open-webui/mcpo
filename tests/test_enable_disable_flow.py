from fastapi.testclient import TestClient
import json
import pytest
from mcpo.services.state import get_state_manager


@pytest.mark.asyncio
async def test_enable_disable_server_via_meta_endpoints(tmp_path):
    """Test server enable/disable via /_meta/servers/{name}/enable|disable endpoints."""
    from mcpo.main import build_main_app

    cfg_path = tmp_path / 'mcpo.json'
    cfg_path.write_text(json.dumps({"mcpServers": {"s1": {"command": "echo", "args": ["ok"]}}}))

    app = await build_main_app(config_path=str(cfg_path))
    client = TestClient(app)

    state_manager = get_state_manager()
    # Initially enabled via state manager
    assert state_manager.is_server_enabled('s1') is True

    # Disable server via endpoint
    r = client.post('/_meta/servers/s1/disable')
    assert r.status_code == 200 and r.json()['enabled'] is False

    # Verify via state manager
    assert state_manager.is_server_enabled('s1') is False

    # Re-enable server via endpoint
    r = client.post('/_meta/servers/s1/enable')
    assert r.status_code == 200 and r.json()['enabled'] is True

    # Verify via state manager
    assert state_manager.is_server_enabled('s1') is True


@pytest.mark.asyncio
async def test_disabled_tool_returns_disabled_code(monkeypatch, tmp_path):
    """Test that calling a disabled tool returns 403 with code 'disabled'."""
    from mcpo.main import build_main_app
    from mcpo.utils.main import get_tool_handler
    from fastapi import FastAPI

    class FakeResult:
        isError = False
        content = []

    class FakeSession:
        async def call_tool(self, name, arguments):
            return FakeResult()

    # Build app and sub-app manually to attach handler
    app = await build_main_app()
    sub = FastAPI(title='sX')
    sub.state.parent_app = app
    sub.state.session = FakeSession()

    # Register a tool handler named 'demo'
    handler = get_tool_handler(sub.state.session, 'demo', form_model_fields=None)
    sub.post('/demo')(handler)
    app.mount('/sX', sub)

    # Set up legacy tool_enabled dict on app.state for handler compatibility
    app.state.tool_enabled = {'sX': {'demo': False}}

    client = TestClient(app)
    resp = client.post('/sX/demo')
    assert resp.status_code == 403
    js = resp.json()
    # When the main_app exception handler is active, it wraps in error_envelope
    assert js['ok'] is False
    assert js['error']['message'] == 'Tool disabled'
