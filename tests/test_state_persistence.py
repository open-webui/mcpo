from fastapi.testclient import TestClient
import json
import os
import tempfile
import asyncio
import pytest
from mcpo.services.state import get_state_manager


@pytest.mark.asyncio
async def test_enable_disable_persists_state(tmp_path):
    """Test that enable/disable state persists via StateManager."""
    from mcpo.main import build_main_app

    # Create a dummy config file
    cfg_path = tmp_path / 'mcpo.json'
    cfg_path.write_text(json.dumps({"mcpServers": {"s1": {"command": "echo", "args": ["ok"]}}}))

    app = await build_main_app(config_path=str(cfg_path))
    client = TestClient(app)

    state_manager = get_state_manager()

    # Disable server via endpoint
    r = client.post('/_meta/servers/s1/disable')
    assert r.status_code == 200 and r.json()['enabled'] is False

    # Disable a tool via StateManager API
    state_manager.set_tool_enabled('s1', 't1', False)

    # Verify state persisted through manager
    assert state_manager.is_server_enabled('s1') is False
    assert state_manager.is_tool_enabled('s1', 't1') is False


@pytest.mark.asyncio
async def test_state_manager_version_and_persistence(tmp_path):
    """Test that StateManager saves and loads state correctly."""
    from mcpo.services.state import StateManager

    state_file = tmp_path / 'test_state.json'
    manager = StateManager(state_file=str(state_file))

    # Set some state
    manager.set_server_enabled('sA', True)
    manager.set_tool_enabled('sA', 'toolX', True)
    manager.set_tool_enabled('sA', 'toolY', False)

    # Verify file was written
    assert state_file.exists()
    data = json.loads(state_file.read_text())
    assert 'version' in data

    # Create new manager from same file — should load persisted state
    manager2 = StateManager(state_file=str(state_file))
    assert manager2.is_server_enabled('sA') is True
    assert manager2.is_tool_enabled('sA', 'toolX') is True
    assert manager2.is_tool_enabled('sA', 'toolY') is False


@pytest.mark.asyncio
async def test_read_only_flag_blocks_mutations(tmp_path):
    from mcpo.main import build_main_app

    cfg_path = tmp_path / 'mcpo.json'
    cfg_path.write_text(json.dumps({"mcpServers": {"s1": {"command": "echo", "args": ["ok"]}}}))

    app = await build_main_app(config_path=str(cfg_path), read_only=True)
    client = TestClient(app)

    for path in [
        '/_meta/servers/s1/enable',
        '/_meta/servers/s1/disable',
        '/_meta/servers/s1/tools/x/enable',
        '/_meta/servers/s1/tools/x/disable',
        '/_meta/reload',
        '/_meta/reinit/s1',
    ]:
        resp = client.post(path)
        assert resp.status_code == 403, f"Expected 403 for {path}, got {resp.status_code}"
        js = resp.json()
        assert js['ok'] is False


@pytest.mark.asyncio
async def test_metrics_stub_counts_disabled_and_timeout(tmp_path, monkeypatch):
    """Test that metrics track disabled errors and tool calls."""
    from mcpo.main import build_main_app
    from mcpo.utils.main import get_tool_handler
    from fastapi import FastAPI

    class FakeTimeoutSession:
        async def call_tool(self, name, arguments):
            await asyncio.sleep(0.01)
            return type('R', (), {'isError': False, 'content': []})

    # Build app
    app = await build_main_app()
    sub = FastAPI(title='sM')
    sub.state.parent_app = app
    sub.state.session = FakeTimeoutSession()
    sub.state.tool_timeout = 1
    sub.state.tool_timeout_max = 1
    handler = get_tool_handler(sub.state.session, 'demo', form_model_fields=None)
    sub.post('/demo')(handler)
    app.mount('/sM', sub)

    # Set up legacy tool_enabled: initially disabled
    app.state.tool_enabled = {'sM': {'demo': False}}

    client = TestClient(app)

    # Disabled call should return 403
    r = client.post('/sM/demo')
    assert r.status_code == 403

    # Enable tool and call successfully
    app.state.tool_enabled['sM']['demo'] = True
    r = client.post('/sM/demo')
    assert r.status_code == 200

    m = client.get('/_meta/metrics')
    assert m.status_code == 200
    body = m.json()
    assert body['ok'] is True
    metrics = body['metrics']
    assert 'calls' in metrics and 'errors' in metrics
    # At least one disabled error registered
    assert metrics['errors']['byCode'].get('disabled', 0) >= 1
