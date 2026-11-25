import os
import json
import sys
import tempfile
import asyncio
from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, Mock, patch
import pytest
from fastapi import FastAPI
from starlette.routing import Mount

from mcpo.main import (
    lifespan,
    load_config,
    mount_config_servers,
    reload_config_handler,
    validate_server_config,
)


def test_validate_server_config_stdio():
    """Test validation of stdio server configuration."""
    config = {"command": "echo", "args": ["hello", "world"]}
    # Should not raise
    validate_server_config("test_server", config)


def test_validate_server_config_sse():
    """Test validation of SSE server configuration."""
    config = {"type": "sse", "url": "http://example.com/sse"}
    # Should not raise
    validate_server_config("test_server", config)


def test_validate_server_config_invalid():
    """Test validation fails for invalid configuration."""
    config = {"invalid": "config"}
    with pytest.raises(
        ValueError, match="must have either 'command' for stdio or 'type' and 'url'"
    ):
        validate_server_config("test_server", config)


def test_validate_server_config_missing_url():
    """Test validation fails for SSE config missing URL."""
    config = {
        "type": "sse"
        # missing url
    }
    with pytest.raises(ValueError, match="requires a 'url' field"):
        validate_server_config("test_server", config)


def test_validate_server_config_disabled_tools_valid():
    """Test validation of server configuration with a valid disabledTools."""
    config = {"command": "echo", "args": ["hello"], "disabledTools": ["search-web"]}
    validate_server_config("test_server", config)


def test_validate_server_config_disabled_tools_invalid():
    """Test validation fails for an invalid disabledTools."""
    config = {"command": "echo", "args": ["hello"], "disabledTools": "not-a-list"}
    with pytest.raises(ValueError, match="'disabledTools' must be a list"):
        validate_server_config("test_server", config)


def test_load_config_valid():
    """Test loading a valid config file."""
    config_data = {
        "mcpServers": {"test_server": {"command": "echo", "args": ["hello"]}}
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        result = load_config(config_path)
        assert result == config_data
    finally:
        os.unlink(config_path)


def test_load_config_invalid_json():
    """Test loading invalid JSON fails."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write('{"invalid": json}')
        config_path = f.name

    try:
        with pytest.raises(json.JSONDecodeError):
            load_config(config_path)
    finally:
        os.unlink(config_path)


def test_load_config_missing_servers():
    """Test loading config without mcpServers fails."""
    config_data = {"other": "data"}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(config_data, f)
        config_path = f.name

    try:
        with pytest.raises(ValueError, match="No 'mcpServers' found"):
            load_config(config_path)
    finally:
        os.unlink(config_path)


@pytest.mark.asyncio
async def test_reload_config_handler():
    """Test the config reload handler."""
    # Create a mock FastAPI app
    app = FastAPI()
    app.state.config_data = {
        "mcpServers": {"old_server": {"command": "echo", "args": ["old"]}}
    }
    app.state.cors_allow_origins = ["*"]
    app.state.api_key = None
    app.state.strict_auth = False
    app.state.api_dependency = None
    app.state.connection_timeout = None
    app.state.lifespan = None
    app.state.path_prefix = "/"
    app.router.routes = []

    new_config = {"mcpServers": {"new_server": {"command": "echo", "args": ["new"]}}}

    # Mock create_sub_app to avoid actually creating apps
    with patch("mcpo.main.create_sub_app") as mock_create_sub_app:
        mock_sub_app = Mock()
        mock_create_sub_app.return_value = mock_sub_app

        # Mock app.mount to avoid actual mounting
        app.mount = Mock()

        await reload_config_handler(app, new_config)

        # Verify the config was updated
        assert app.state.config_data == new_config

        # Verify create_sub_app was called for the new server
        mock_create_sub_app.assert_called_once()

        # Verify mount was called
        app.mount.assert_called_once()


def test_config_watcher_initialization():
    """Test ConfigWatcher can be initialized."""
    from mcpo.utils.config_watcher import ConfigWatcher

    callback = Mock()

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"mcpServers": {}}, f)
        config_path = f.name

    try:
        watcher = ConfigWatcher(config_path, callback)
        assert watcher.config_path.name == os.path.basename(config_path)
        assert watcher.reload_callback == callback
    finally:
        os.unlink(config_path)


def _make_main_app(initial_config, sub_lifespan):
    main_app = FastAPI(lifespan=lifespan)
    main_app.state.config_data = initial_config
    main_app.state.cors_allow_origins = ["*"]
    main_app.state.api_key = None
    main_app.state.strict_auth = False
    main_app.state.api_dependency = None
    main_app.state.connection_timeout = None
    main_app.state.lifespan = sub_lifespan
    main_app.state.path_prefix = "/"
    main_app.state.active_lifespans = {}
    mount_config_servers(
        main_app,
        initial_config,
        ["*"],
        None,
        False,
        None,
        None,
        sub_lifespan,
        "/",
    )
    return main_app


@pytest.mark.asyncio
async def test_shutdown_closes_hot_reloaded_servers_without_double_closing_startup_servers():
    """Shutdown must close servers added or updated by hot reload without double-closing startup servers."""
    exited = []

    class TrackingLifespan:
        def __init__(self, sub_app):
            self.key = (sub_app.title, tuple(getattr(sub_app.state, "args", [])))
            self.sub_app = sub_app

        async def __aenter__(self):
            self.sub_app.state.is_connected = True
            return self

        async def __aexit__(self, exc_type, exc, tb):
            exited.append(self.key)

    cfg1 = {
        "mcpServers": {
            "unchanged": {"command": "echo", "args": ["v1"]},
            "updated": {"command": "echo", "args": ["v1"]},
        }
    }
    main_app = _make_main_app(cfg1, TrackingLifespan)

    async with lifespan(main_app):
        cfg2 = {
            "mcpServers": {
                "unchanged": {"command": "echo", "args": ["v1"]},
                "updated": {"command": "echo", "args": ["v2"]},
                "added": {"command": "echo", "args": ["v1"]},
            }
        }
        await reload_config_handler(main_app, cfg2)
        assert exited == [("updated", ("v1",))]

    assert exited.count(("unchanged", ("v1",))) == 1
    assert exited.count(("updated", ("v2",))) == 1
    assert exited.count(("added", ("v1",))) == 1
    assert main_app.state.active_lifespans == {}


@pytest.mark.asyncio
async def test_shutdown_closes_remaining_hot_reloaded_servers_when_one_fails_to_exit():
    """An error closing one hot-reloaded server on shutdown must not leak other servers."""
    exited = []

    @asynccontextmanager
    async def flaky_sub_lifespan(sub_app):
        name = sub_app.title
        sub_app.state.is_connected = True
        try:
            yield
        finally:
            exited.append(name)
            if name == "broken":
                raise RuntimeError("boom during shutdown")

    cfg1 = {"mcpServers": {"startup": {"command": "echo", "args": ["1"]}}}
    main_app = _make_main_app(cfg1, flaky_sub_lifespan)

    async with lifespan(main_app):
        cfg2 = {
            "mcpServers": {
                "startup": {"command": "echo", "args": ["1"]},
                "healthy": {"command": "echo", "args": ["2"]},
                "broken": {"command": "echo", "args": ["3"]},
            }
        }
        await reload_config_handler(main_app, cfg2)

    assert set(exited) == {"startup", "healthy", "broken"}


@pytest.mark.asyncio
async def test_shutdown_closes_real_stdio_server_added_by_hot_reload():
    """A real stdio MCP server added during hot reload is closed when the proxy shuts down."""
    server_code = """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("hot-reload-shutdown-test")
@mcp.tool()
def ping() -> str:
    return "pong"
mcp.run()
"""
    cfg1 = {
        "mcpServers": {
            "startup": {"command": sys.executable, "args": ["-c", server_code]}
        }
    }
    main_app = _make_main_app(cfg1, lifespan)

    async with lifespan(main_app):
        cfg2 = {
            "mcpServers": {
                "startup": {"command": sys.executable, "args": ["-c", server_code]},
                "reloaded": {"command": sys.executable, "args": ["-c", server_code]},
            }
        }
        await asyncio.wait_for(reload_config_handler(main_app, cfg2), timeout=20)
        reloaded_apps = [
            route.app
            for route in main_app.routes
            if isinstance(route, Mount) and route.path == "/reloaded"
        ]
        assert len(reloaded_apps) == 1
        reloaded_app = reloaded_apps[0]
        assert reloaded_app.state.is_connected is True
        assert reloaded_app.state.session_manager is not None
        assert reloaded_app.state.session_manager.current_session is not None

    assert reloaded_app.state.session_manager is None
    assert reloaded_app.state.session is None
    assert main_app.state.active_lifespans == {}
