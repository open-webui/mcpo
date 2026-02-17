from fastapi.testclient import TestClient
import json
import pytest
import mcpo.services.state as _state_mod


@pytest.mark.asyncio
async def test_enable_disable_server_via_meta_endpoints(tmp_path, monkeypatch):
    """Test server enable/disable via /_meta/servers/{name}/enable|disable endpoints."""
    from mcpo.main import build_main_app

    # Reset the global singleton so a fresh StateManager is created with a
    # temporary state file (avoids leaking state from mcpo_state.json).
    monkeypatch.setattr(_state_mod, '_global_state_manager', None)

    state_file = tmp_path / 'state.json'
    cfg_path = tmp_path / 'mcpo.json'
    cfg_path.write_text(json.dumps({"mcpServers": {"s1": {"command": "echo", "args": ["ok"]}}}))

    app = await build_main_app(config_path=str(cfg_path))
    client = TestClient(app)

    state_manager = _state_mod.get_state_manager()

    # Disable server via endpoint
    r = client.post('/_meta/servers/s1/disable')
    assert r.status_code == 200 and r.json().get('enabled') is False

    # Verify via state manager
    assert state_manager.is_server_enabled('s1') is False

    # Re-enable server via endpoint
    r = client.post('/_meta/servers/s1/enable')
    assert r.status_code == 200 and r.json().get('enabled') is True

    # Verify via state manager
    assert state_manager.is_server_enabled('s1') is True


@pytest.mark.asyncio
async def test_disabled_tool_returns_403(monkeypatch, tmp_path):
    """Test that calling a disabled tool returns 403."""
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
    # The response may be wrapped in error_envelope ({"ok": False, ...}) if the
    # main_app exception handler intercepts, or as FastAPI's default
    # {"detail": {"message": ..., "code": ...}} if the sub-app handles it.
    # Either way, 403 confirms the tool is blocked.
    if 'ok' in js:
        assert js['ok'] is False
    elif 'detail' in js:
        detail = js['detail']
        if isinstance(detail, dict):
            assert detail.get('code') == 'disabled' or 'disabled' in detail.get('message', '').lower()

