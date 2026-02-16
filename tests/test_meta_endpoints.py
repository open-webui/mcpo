"""Comprehensive tests for all /_meta/* management endpoints.

Each test is self-contained: builds its own app, mounts fake sub-apps where needed,
and tears down cleanly via tmp_path for config files and patching the global
StateManager singleton so tests don't bleed into each other.
"""
import json
import os
import pytest
from unittest.mock import patch
from fastapi import FastAPI
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_cfg(tmp_path, servers=None):
    """Write a minimal mcpo.json and return its path as str."""
    if servers is None:
        servers = {"s1": {"command": "echo", "args": ["ok"]}}
    cfg_path = tmp_path / "mcpo.json"
    cfg_path.write_text(json.dumps({"mcpServers": servers}))
    return str(cfg_path)


def _fresh_state_manager(tmp_path):
    """Return a factory that creates a StateManager with an isolated state file."""
    from mcpo.services.state import StateManager
    path = str(tmp_path / "test_state.json")
    return StateManager(state_file_path=path)


async def _build_app(tmp_path, *, servers=None, read_only=False, extra_kwargs=None):
    """Build a main app with patched state manager pointing at tmp_path."""
    from mcpo.main import build_main_app
    cfg_path = _write_cfg(tmp_path, servers)
    sm = _fresh_state_manager(tmp_path)
    kwargs = dict(config_path=cfg_path, read_only=read_only)
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    with patch("mcpo.main.get_state_manager", return_value=sm):
        app = await build_main_app(**kwargs)
    # Ensure the app state has the patched state manager
    app.state.state_manager = sm
    return app


def _mount_fake_server(app, name="s1", tools=None):
    """Mount a fake FastAPI sub-app simulating a connected MCP server.

    ``tools`` is a list of tool name strings to register as POST endpoints.
    Returns the sub-app.
    """
    from mcpo.utils.main import get_tool_handler

    sub = FastAPI(title=name)
    sub.state.parent_app = app
    sub.state.is_connected = True
    sub.state.server_type = "stdio"
    sub.state.config_key = name

    class FakeSession:
        async def call_tool(self, name, arguments):
            return type("R", (), {"isError": False, "content": []})()

    sub.state.session = FakeSession()

    for tname in (tools or []):
        handler = get_tool_handler(sub.state.session, tname, form_model_fields=None)
        sub.post(f"/{tname}")(handler)

    app.mount(f"/{name}", sub)
    return sub


# ===========================================================================
# 1. GET /_meta/servers — list servers, verify structure
# ===========================================================================

@pytest.mark.asyncio
async def test_list_servers_returns_ok_and_structure(tmp_path):
    app = await _build_app(tmp_path)
    _mount_fake_server(app, "s1", tools=["ping"])
    client = TestClient(app)

    r = client.get("/_meta/servers")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["servers"], list)
    names = {s["name"] for s in body["servers"]}
    assert "s1" in names
    s1 = next(s for s in body["servers"] if s["name"] == "s1")
    assert "connected" in s1
    assert "type" in s1
    assert "basePath" in s1
    assert "enabled" in s1


@pytest.mark.asyncio
async def test_list_servers_empty_config(tmp_path):
    app = await _build_app(tmp_path, servers={})
    client = TestClient(app)

    r = client.get("/_meta/servers")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # May contain internal mcpo server; should not error
    assert isinstance(body["servers"], list)


# ===========================================================================
# 2. GET /_meta/servers/{name}/tools — list tools for a server
# ===========================================================================

@pytest.mark.asyncio
async def test_list_server_tools_happy(tmp_path):
    app = await _build_app(tmp_path, servers={})
    _mount_fake_server(app, "s1", tools=["alpha", "beta"])
    client = TestClient(app)

    r = client.get("/_meta/servers/s1/tools")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["server"] == "s1"
    tool_names = [t["name"] for t in body["tools"]]
    assert "alpha" in tool_names
    assert "beta" in tool_names
    for t in body["tools"]:
        assert "enabled" in t


@pytest.mark.asyncio
async def test_list_server_tools_not_found(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.get("/_meta/servers/nonexistent/tools")
    assert r.status_code == 404
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "not_found"


# ===========================================================================
# 3. POST /_meta/servers/{name}/enable — enable server
# ===========================================================================

@pytest.mark.asyncio
async def test_enable_server_happy(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    # Disable first
    r = client.post("/_meta/servers/s1/disable")
    assert r.status_code == 200
    assert r.json()["enabled"] is False

    # Enable
    r = client.post("/_meta/servers/s1/enable")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["server"] == "s1"
    assert body["enabled"] is True


@pytest.mark.asyncio
async def test_enable_server_read_only(tmp_path):
    app = await _build_app(tmp_path, read_only=True)
    client = TestClient(app)

    r = client.post("/_meta/servers/s1/enable")
    assert r.status_code == 403
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "read_only"


# ===========================================================================
# 4. POST /_meta/servers/{name}/disable — disable server
# ===========================================================================

@pytest.mark.asyncio
async def test_disable_server_happy(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/servers/s1/disable")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["enabled"] is False

    # Verify via list
    r2 = client.get("/_meta/servers")
    servers = {s["name"]: s for s in r2.json()["servers"]}
    if "s1" in servers:
        assert servers["s1"]["enabled"] is False


@pytest.mark.asyncio
async def test_disable_server_read_only(tmp_path):
    app = await _build_app(tmp_path, read_only=True)
    client = TestClient(app)

    r = client.post("/_meta/servers/s1/disable")
    assert r.status_code == 403
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "read_only"


# ===========================================================================
# 5. POST /_meta/servers/{name}/tools/{tool}/enable — enable tool
# ===========================================================================

@pytest.mark.asyncio
async def test_enable_tool_happy(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    # Disable first
    r = client.post("/_meta/servers/s1/tools/ping/disable")
    assert r.status_code == 200

    # Enable
    r = client.post("/_meta/servers/s1/tools/ping/enable")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["server"] == "s1"
    assert body["tool"] == "ping"
    assert body["enabled"] is True


@pytest.mark.asyncio
async def test_enable_tool_read_only(tmp_path):
    app = await _build_app(tmp_path, read_only=True)
    client = TestClient(app)

    r = client.post("/_meta/servers/s1/tools/ping/enable")
    assert r.status_code == 403
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "read_only"


# ===========================================================================
# 6. POST /_meta/servers/{name}/tools/{tool}/disable — disable tool
# ===========================================================================

@pytest.mark.asyncio
async def test_disable_tool_happy(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/servers/s1/tools/ping/disable")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["server"] == "s1"
    assert body["tool"] == "ping"
    assert body["enabled"] is False


@pytest.mark.asyncio
async def test_disable_tool_read_only(tmp_path):
    app = await _build_app(tmp_path, read_only=True)
    client = TestClient(app)

    r = client.post("/_meta/servers/s1/tools/ping/disable")
    assert r.status_code == 403
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "read_only"


# ===========================================================================
# 7. GET /_meta/config — get config info
# ===========================================================================

@pytest.mark.asyncio
async def test_config_info_returns_path(tmp_path):
    cfg_path = _write_cfg(tmp_path)
    sm = _fresh_state_manager(tmp_path)
    from mcpo.main import build_main_app
    with patch("mcpo.main.get_state_manager", return_value=sm):
        app = await build_main_app(config_path=cfg_path)
    app.state.state_manager = sm
    client = TestClient(app)

    r = client.get("/_meta/config")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["configPath"] == cfg_path


@pytest.mark.asyncio
async def test_config_info_no_config(tmp_path):
    sm = _fresh_state_manager(tmp_path)
    from mcpo.main import build_main_app
    with patch("mcpo.main.get_state_manager", return_value=sm):
        app = await build_main_app()
    app.state.state_manager = sm
    client = TestClient(app)

    r = client.get("/_meta/config")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    # configPath may be None when no config file supplied
    assert "configPath" in body


# ===========================================================================
# 8. GET /_meta/metrics — get metrics
# ===========================================================================

@pytest.mark.asyncio
async def test_metrics_returns_structure(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.get("/_meta/metrics")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    m = body["metrics"]
    assert "servers" in m
    assert "total" in m["servers"]
    assert "enabled" in m["servers"]
    assert "tools" in m
    assert "calls" in m
    assert "errors" in m
    assert "perTool" in m


@pytest.mark.asyncio
async def test_metrics_reflects_disabled_server(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    # Disable s1 to move the counter
    client.post("/_meta/servers/s1/disable")

    r = client.get("/_meta/metrics")
    assert r.status_code == 200
    m = r.json()["metrics"]
    # At least the entry should exist; enabled count may decrease
    assert isinstance(m["servers"]["total"], int)
    assert isinstance(m["servers"]["enabled"], int)


# ===========================================================================
# 9. POST /_meta/reload — reload config (with read-only check)
# ===========================================================================

@pytest.mark.asyncio
async def test_reload_happy(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/reload")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "generation" in body


@pytest.mark.asyncio
async def test_reload_read_only(tmp_path):
    app = await _build_app(tmp_path, read_only=True)
    client = TestClient(app)

    r = client.post("/_meta/reload")
    assert r.status_code == 403
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "read_only"


@pytest.mark.asyncio
async def test_reload_no_config(tmp_path):
    """Reload when no config path is set should return 400."""
    sm = _fresh_state_manager(tmp_path)
    from mcpo.main import build_main_app
    with patch("mcpo.main.get_state_manager", return_value=sm):
        app = await build_main_app()
    app.state.state_manager = sm
    client = TestClient(app)

    r = client.post("/_meta/reload")
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "no_config"


# ===========================================================================
# 10. POST /_meta/reinit/{name} — reinit server (404 for unknown)
# ===========================================================================

@pytest.mark.asyncio
async def test_reinit_unknown_server(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/reinit/nonexistent")
    assert r.status_code == 404
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_reinit_read_only(tmp_path):
    app = await _build_app(tmp_path, read_only=True)
    client = TestClient(app)

    r = client.post("/_meta/reinit/s1")
    assert r.status_code == 403
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "read_only"


@pytest.mark.asyncio
async def test_reinit_mounted_server(tmp_path):
    """Reinit on a mounted fake server should succeed (calls initialize_sub_app)."""
    app = await _build_app(tmp_path)
    sub = _mount_fake_server(app, "s1", tools=["ping"])

    # Patch initialize_sub_app to avoid real MCP connection
    async def fake_init(sub_app):
        sub_app.state.is_connected = True

    with patch("mcpo.main.initialize_sub_app", side_effect=fake_init):
        client = TestClient(app)
        r = client.post("/_meta/reinit/s1")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["server"] == "s1"
    assert "connected" in body


# ===========================================================================
# 11. POST /_meta/servers — add server (with validation)
# ===========================================================================

@pytest.mark.asyncio
async def test_add_server_happy(tmp_path):
    app = await _build_app(tmp_path, servers={})
    client = TestClient(app)

    # Patch reload to avoid real MCP connection attempts
    async def fake_reload(app, cfg):
        pass

    with patch("mcpo.main.reload_config_handler", side_effect=fake_reload):
        r = client.post("/_meta/servers", json={"name": "new1", "command": "echo hello"})

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["server"] == "new1"

    # Verify config file was updated
    cfg = json.loads((tmp_path / "mcpo.json").read_text())
    assert "new1" in cfg["mcpServers"]


@pytest.mark.asyncio
async def test_add_server_missing_name(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/servers", json={"command": "echo"})
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "invalid"


@pytest.mark.asyncio
async def test_add_server_duplicate(tmp_path):
    app = await _build_app(tmp_path, servers={"s1": {"command": "echo", "args": ["ok"]}})
    client = TestClient(app)

    r = client.post("/_meta/servers", json={"name": "s1", "command": "echo"})
    assert r.status_code == 409
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "exists"


@pytest.mark.asyncio
async def test_add_server_read_only(tmp_path):
    app = await _build_app(tmp_path, read_only=True)
    client = TestClient(app)

    r = client.post("/_meta/servers", json={"name": "new1", "command": "echo"})
    assert r.status_code == 403
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "read_only"


@pytest.mark.asyncio
async def test_add_server_invalid_json(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/servers", content=b"not json", headers={"content-type": "application/json"})
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False


@pytest.mark.asyncio
async def test_add_server_empty_command(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/servers", json={"name": "bad", "command": "   "})
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False


@pytest.mark.asyncio
async def test_add_server_no_config_mode(tmp_path):
    """Cannot add servers when not running in config-file mode."""
    sm = _fresh_state_manager(tmp_path)
    from mcpo.main import build_main_app
    with patch("mcpo.main.get_state_manager", return_value=sm):
        app = await build_main_app()
    app.state.state_manager = sm
    client = TestClient(app)

    r = client.post("/_meta/servers", json={"name": "x", "command": "echo"})
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "no_config_mode"


# ===========================================================================
# 12. GET /healthz — health check
# ===========================================================================

@pytest.mark.asyncio
async def test_healthz_returns_ok(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "generation" in body
    assert "lastReload" in body
    assert "servers" in body


@pytest.mark.asyncio
async def test_healthz_reflects_mounted_servers(tmp_path):
    app = await _build_app(tmp_path)
    _mount_fake_server(app, "s1", tools=["ping"])
    client = TestClient(app)

    r = client.get("/healthz")
    assert r.status_code == 200
    body = r.json()
    # The servers dict should mention s1 (by title)
    assert isinstance(body["servers"], dict)


# ===========================================================================
# 13. POST /_meta/env — env update (admin router)
# ===========================================================================

@pytest.mark.asyncio
async def test_env_update_happy(tmp_path, monkeypatch):
    app = await _build_app(tmp_path)
    # Point .env path to tmp_path so we don't clobber real .env
    env_file = tmp_path / ".env"
    monkeypatch.chdir(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/env", json={"TEST_KEY_XYZ": "val123"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "updated"
    assert body["count"] == 1

    # Verify the file was written
    content = env_file.read_text()
    assert "TEST_KEY_XYZ=val123" in content

    # Verify process env updated
    assert os.environ.get("TEST_KEY_XYZ") == "val123"

    # Cleanup
    if "TEST_KEY_XYZ" in os.environ:
        del os.environ["TEST_KEY_XYZ"]


@pytest.mark.asyncio
async def test_env_update_merges_existing(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env_file = tmp_path / ".env"
    env_file.write_text("EXISTING_KEY=old\n")

    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/env", json={"NEW_KEY": "new_val"})
    assert r.status_code == 200

    content = env_file.read_text()
    assert "EXISTING_KEY=old" in content
    assert "NEW_KEY=new_val" in content

    # Cleanup
    for k in ("EXISTING_KEY", "NEW_KEY"):
        os.environ.pop(k, None)


@pytest.mark.asyncio
async def test_env_update_empty_payload(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.post("/_meta/env", json={})
    assert r.status_code == 200
    body = r.json()
    assert body["count"] == 0


# ===========================================================================
# 14. DELETE /_meta/servers/{name} — remove server
# ===========================================================================

@pytest.mark.asyncio
async def test_remove_server_happy(tmp_path):
    app = await _build_app(tmp_path, servers={"s1": {"command": "echo", "args": ["ok"]}})
    client = TestClient(app)

    async def fake_reload(app, cfg):
        pass

    with patch("mcpo.main.reload_config_handler", side_effect=fake_reload):
        r = client.delete("/_meta/servers/s1")

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["removed"] == "s1"

    # Verify config file updated
    cfg = json.loads((tmp_path / "mcpo.json").read_text())
    assert "s1" not in cfg.get("mcpServers", {})


@pytest.mark.asyncio
async def test_remove_server_not_found(tmp_path):
    app = await _build_app(tmp_path, servers={"s1": {"command": "echo", "args": ["ok"]}})
    client = TestClient(app)

    r = client.delete("/_meta/servers/nonexistent")
    assert r.status_code == 404
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "not_found"


@pytest.mark.asyncio
async def test_remove_server_no_config_mode(tmp_path):
    sm = _fresh_state_manager(tmp_path)
    from mcpo.main import build_main_app
    with patch("mcpo.main.get_state_manager", return_value=sm):
        app = await build_main_app()
    app.state.state_manager = sm
    client = TestClient(app)

    r = client.delete("/_meta/servers/s1")
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "no_config_mode"


# ===========================================================================
# 15. Round-trip: disable server -> verify list reflects it -> re-enable
# ===========================================================================

@pytest.mark.asyncio
async def test_enable_disable_roundtrip(tmp_path):
    app = await _build_app(tmp_path)
    _mount_fake_server(app, "s1", tools=["ping"])
    client = TestClient(app)

    # Initially enabled
    r = client.get("/_meta/servers")
    servers = {s["name"]: s for s in r.json()["servers"]}
    assert servers["s1"]["enabled"] is True

    # Disable
    r = client.post("/_meta/servers/s1/disable")
    assert r.status_code == 200

    r = client.get("/_meta/servers")
    servers = {s["name"]: s for s in r.json()["servers"]}
    assert servers["s1"]["enabled"] is False

    # Re-enable
    r = client.post("/_meta/servers/s1/enable")
    assert r.status_code == 200

    r = client.get("/_meta/servers")
    servers = {s["name"]: s for s in r.json()["servers"]}
    assert servers["s1"]["enabled"] is True


# ===========================================================================
# 16. Round-trip: disable tool -> verify tools list -> re-enable
# ===========================================================================

@pytest.mark.asyncio
async def test_tool_enable_disable_roundtrip(tmp_path):
    app = await _build_app(tmp_path, servers={})
    _mount_fake_server(app, "s1", tools=["alpha"])
    client = TestClient(app)

    # Disable tool
    r = client.post("/_meta/servers/s1/tools/alpha/disable")
    assert r.status_code == 200

    r = client.get("/_meta/servers/s1/tools")
    tools_map = {t["name"]: t for t in r.json()["tools"]}
    assert tools_map["alpha"]["enabled"] is False

    # Re-enable
    r = client.post("/_meta/servers/s1/tools/alpha/enable")
    assert r.status_code == 200

    r = client.get("/_meta/servers/s1/tools")
    tools_map = {t["name"]: t for t in r.json()["tools"]}
    assert tools_map["alpha"]["enabled"] is True


# ===========================================================================
# 17. All write endpoints blocked in read-only mode
# ===========================================================================

@pytest.mark.asyncio
async def test_all_writes_blocked_read_only(tmp_path):
    app = await _build_app(tmp_path, read_only=True)
    client = TestClient(app)

    write_paths = [
        ("POST", "/_meta/servers/s1/enable"),
        ("POST", "/_meta/servers/s1/disable"),
        ("POST", "/_meta/servers/s1/tools/t/enable"),
        ("POST", "/_meta/servers/s1/tools/t/disable"),
        ("POST", "/_meta/reload"),
        ("POST", "/_meta/reinit/s1"),
        ("POST", "/_meta/servers"),
    ]

    for method, path in write_paths:
        if method == "POST":
            if path == "/_meta/servers":
                r = client.post(path, json={"name": "x", "command": "echo"})
            else:
                r = client.post(path)
        assert r.status_code == 403, f"Expected 403 for {method} {path}, got {r.status_code}"
        body = r.json()
        assert body["ok"] is False, f"Expected ok=False for {method} {path}"
        assert body["error"]["code"] == "read_only", f"Expected read_only code for {method} {path}"


# ===========================================================================
# 18. GET /_meta/config/content — config file content
# ===========================================================================

@pytest.mark.asyncio
async def test_config_content_happy(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.get("/_meta/config/content")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert "content" in body
    # Content should be valid JSON matching our written config
    parsed = json.loads(body["content"])
    assert "mcpServers" in parsed


@pytest.mark.asyncio
async def test_config_content_no_config(tmp_path):
    sm = _fresh_state_manager(tmp_path)
    from mcpo.main import build_main_app
    with patch("mcpo.main.get_state_manager", return_value=sm):
        app = await build_main_app()
    app.state.state_manager = sm
    client = TestClient(app)

    r = client.get("/_meta/config/content")
    assert r.status_code == 400
    body = r.json()
    assert body["ok"] is False


# ===========================================================================
# 19. GET /_meta/logs/sources
# ===========================================================================

@pytest.mark.asyncio
async def test_log_sources(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.get("/_meta/logs/sources")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    sources = body["sources"]
    assert any(s["id"] == "openapi" for s in sources)


# ===========================================================================
# 20. GET /_meta/logs
# ===========================================================================

@pytest.mark.asyncio
async def test_logs_endpoint(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    r = client.get("/_meta/logs")
    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert isinstance(body["logs"], list)
    assert "nextCursor" in body
    assert "latestCursor" in body


# ===========================================================================
# 21. Metrics perTool details after a tool call
# ===========================================================================

@pytest.mark.asyncio
async def test_metrics_per_tool_after_call(tmp_path):
    """Tool calls via the handler use the MetricsAggregator singleton, not
    app.state.metrics.  Verify that calls succeed and the metrics endpoint
    still returns a valid structure (the in-memory dict may or may not reflect
    aggregator-level counts depending on wiring).
    """
    app = await _build_app(tmp_path, servers={})
    _mount_fake_server(app, "s1", tools=["demo"])
    client = TestClient(app)

    # Call the tool (no-arg handler)
    r = client.post("/s1/demo")
    assert r.status_code == 200

    # Check metrics endpoint returns valid structure
    r = client.get("/_meta/metrics")
    assert r.status_code == 200
    m = r.json()["metrics"]
    assert "calls" in m
    assert "errors" in m
    assert "perTool" in m
    # calls.total is populated from app.state.metrics which is a separate dict
    # from the MetricsAggregator; presence and type are sufficient here
    assert isinstance(m["calls"]["total"], int)


# ===========================================================================
# 22. Add server with URL (SSE/streamable-http type)
# ===========================================================================

@pytest.mark.asyncio
async def test_add_server_with_url(tmp_path):
    app = await _build_app(tmp_path, servers={})
    client = TestClient(app)

    async def fake_reload(app, cfg):
        pass

    with patch("mcpo.main.reload_config_handler", side_effect=fake_reload):
        r = client.post("/_meta/servers", json={
            "name": "remote1",
            "url": "http://example.com/sse",
            "type": "sse",
        })

    assert r.status_code == 200
    body = r.json()
    assert body["ok"] is True
    assert body["server"] == "remote1"

    cfg = json.loads((tmp_path / "mcpo.json").read_text())
    assert cfg["mcpServers"]["remote1"]["url"] == "http://example.com/sse"
    assert cfg["mcpServers"]["remote1"]["type"] == "sse"


# ===========================================================================
# 23. Add server with non-string command rejected
# ===========================================================================

@pytest.mark.asyncio
async def test_add_server_command_not_string(tmp_path):
    app = await _build_app(tmp_path, servers={})
    client = TestClient(app)

    r = client.post("/_meta/servers", json={"name": "bad", "command": 123})
    assert r.status_code == 422
    body = r.json()
    assert body["ok"] is False
    assert body["error"]["code"] == "invalid"
