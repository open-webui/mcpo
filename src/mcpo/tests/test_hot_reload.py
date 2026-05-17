import os
import json
import tempfile
import asyncio
from unittest.mock import AsyncMock, Mock, patch
import pytest
from fastapi import FastAPI

from mcpo.main import (
    load_config,
    register_health_endpoint,
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


@pytest.mark.asyncio
async def test_health_endpoint_all_healthy():
    """Test health endpoint reports healthy when all servers are ok."""
    from httpx import ASGITransport, AsyncClient

    app = FastAPI()
    register_health_endpoint(app)

    sub_app = FastAPI(title="test_server")
    sub_app.state.is_connected = True
    mock_session = AsyncMock()
    mock_session.send_ping = AsyncMock(return_value=None)
    mock_manager = AsyncMock()
    mock_manager.get_session = AsyncMock(return_value=mock_session)
    sub_app.state.session_manager = mock_manager

    app.mount("/test_server", sub_app)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["servers"]["test_server"]["status"] == "ok"


@pytest.mark.asyncio
async def test_health_endpoint_server_unhealthy():
    """Test health endpoint reports unhealthy when a server ping fails."""
    from httpx import ASGITransport, AsyncClient

    app = FastAPI()
    register_health_endpoint(app)

    sub_app = FastAPI(title="broken_server")
    sub_app.state.is_connected = True
    mock_session = AsyncMock()
    mock_session.send_ping = AsyncMock(side_effect=ConnectionError("pipe broken"))
    mock_manager = AsyncMock()
    mock_manager.get_session = AsyncMock(return_value=mock_session)
    sub_app.state.session_manager = mock_manager

    app.mount("/broken_server", sub_app)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["servers"]["broken_server"]["status"] == "error"
    assert "pipe broken" in data["servers"]["broken_server"]["error"]


@pytest.mark.asyncio
async def test_health_endpoint_disconnected_server():
    """Test health endpoint reports disconnected for servers that never connected."""
    from httpx import ASGITransport, AsyncClient

    app = FastAPI()
    register_health_endpoint(app)

    sub_app = FastAPI(title="offline_server")
    sub_app.state.is_connected = False
    sub_app.state.session_manager = None

    app.mount("/offline_server", sub_app)

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "unhealthy"
    assert data["servers"]["offline_server"]["status"] == "disconnected"


@pytest.mark.asyncio
async def test_health_endpoint_single_server_mode():
    """Test health endpoint works in single-server mode (no sub-apps)."""
    from httpx import ASGITransport, AsyncClient

    app = FastAPI(title="my_mcp_server")
    register_health_endpoint(app)

    mock_session = AsyncMock()
    mock_session.send_ping = AsyncMock(return_value=None)
    mock_manager = AsyncMock()
    mock_manager.get_session = AsyncMock(return_value=mock_session)
    app.state.session_manager = mock_manager

    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["servers"]["my_mcp_server"]["status"] == "ok"
