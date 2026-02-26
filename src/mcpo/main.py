import asyncio
import json
import logging
import os
import signal
import socket
import subprocess
import sys
import threading
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional, Dict, Any, List
from urllib.parse import urljoin
from datetime import datetime, timezone

import httpx
import uvicorn
from fastapi import Depends, FastAPI, Body, Query
from pydantic import BaseModel
from fastapi.responses import JSONResponse
from fastapi.requests import Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.routing import Mount

from mcp import ClientSession, StdioServerParameters
from mcp.client.sse import sse_client
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

from mcpo.utils.auth import APIKeyMiddleware, get_verify_api_key
from mcpo.utils.main import (
    get_model_fields,
    get_tool_handler,
    normalize_server_type,
)
from mcpo.utils.config import interpolate_env_placeholders_in_config
from mcpo.utils.config_watcher import ConfigWatcher
from mcpo.services.state import get_state_manager
from mcpo.services.logging import get_log_manager
from mcpo.services.logging_handlers import BufferedLogHandler

from mcpo.api.routers.admin import _mount_or_remount_fastmcp, router as admin_router
from mcpo.api.routers.chat import router as chat_router
from mcpo.api.routers.completions import router as completions_router

# MCP protocol version (used for outbound remote connections headers)
MCP_VERSION = "2025-06-18"


logger = logging.getLogger(__name__)

def error_envelope(message: str, code: str | None = None, data: Any | None = None):
    payload = {"ok": False, "error": {"message": message}}
    if code:
        payload["error"]["code"] = code
    if data is not None:
        payload["error"]["data"] = data
    return payload


# State management now uses StateManager singleton - see services/state.py

# Global reload lock to ensure atomic config reloads
_reload_lock = asyncio.Lock()

# Global health snapshot
_health_state: Dict[str, Any] = {
    "generation": 0,
    "last_reload": None,
    "servers": {},  # name -> {connected: bool, type: str}
}

# Global log buffer for UI display (mirrors centralized log manager)
_log_buffer: list[dict] = []
_log_buffer_lock = threading.Lock()
MAX_LOG_ENTRIES = 2000


def _update_health_snapshot(app: FastAPI):
    """Recompute health snapshot based on mounted sub apps."""
    servers = {}
    for route in app.router.routes:
        if isinstance(route, Mount) and isinstance(route.app, FastAPI):
            sub_app = route.app
            servers[sub_app.title] = {
                "connected": bool(getattr(sub_app.state, "is_connected", False)),
                "type": getattr(sub_app.state, "server_type", "unknown"),
            }
    _health_state["servers"] = servers


def _register_health_endpoint(app: FastAPI):
    @app.get("/healthz")
    async def healthz():  # noqa: D401
        """Basic health & connectivity info."""
        # Update on demand to reflect latest connection flags
        _update_health_snapshot(app)
        return {
            "status": "ok",
            "generation": _health_state["generation"],
            "lastReload": _health_state["last_reload"],
            "servers": _health_state["servers"],
        }


class GracefulShutdown:
    def __init__(self):
        self.shutdown_event = asyncio.Event()
        self.tasks = set()

    def handle_signal(self, sig, frame=None):
        """Handle shutdown signals gracefully"""
        logger.info(
            f"\nReceived {signal.Signals(sig).name}, initiating graceful shutdown..."
        )
        self.shutdown_event.set()

    def track_task(self, task):
        """Track tasks for cleanup"""
        self.tasks.add(task)
        task.add_done_callback(self.tasks.discard)


def validate_server_config(server_name: str, server_cfg: Dict[str, Any]) -> None:
    """Validate individual server configuration."""
    server_type = server_cfg.get("type")

    if normalize_server_type(server_type) in ("sse", "streamable-http"):
        if not server_cfg.get("url"):
            raise ValueError(f"Server '{server_name}' of type '{server_type}' requires a 'url' field")
    elif server_cfg.get("command"):
        # stdio server
        if not isinstance(server_cfg["command"], str):
            raise ValueError(f"Server '{server_name}' 'command' must be a string")
        if server_cfg.get("args") and not isinstance(server_cfg["args"], list):
            raise ValueError(f"Server '{server_name}' 'args' must be a list")
    elif server_cfg.get("url") and not server_type:
        # Fallback for old SSE config without explicit type
        pass
    else:
        raise ValueError(f"Server '{server_name}' must have either 'command' for stdio or 'type' and 'url' for remote servers")

    # Validate disabledTools
    disabled_tools = server_cfg.get("disabledTools")
    if disabled_tools is not None:
        if not isinstance(disabled_tools, list):
            raise ValueError(f"Server '{server_name}' 'disabledTools' must be a list")
        for tool_name in disabled_tools:
            if not isinstance(tool_name, str):
                raise ValueError(f"Server '{server_name}' 'disabledTools' must contain only strings")


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate config from file."""
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        # Expand any ${VAR} placeholders from the current environment before validation
        config_data = interpolate_env_placeholders_in_config(config_data)

        mcp_servers = config_data.get("mcpServers", {})
        if "mcpServers" not in config_data:
            logger.error(f"No 'mcpServers' found in config file: {config_path}")
            raise ValueError("No 'mcpServers' found in config file.")

        # Validate each server configuration
        for server_name, server_cfg in mcp_servers.items():
            validate_server_config(server_name, server_cfg)

        return config_data
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON in config file {config_path}: {e}")
        raise
    except FileNotFoundError:
        logger.error(f"Config file not found: {config_path}")
        raise
    except ValueError as e:
        logger.error(f"Invalid configuration: {e}")
        raise


def create_sub_app(server_name: str, server_cfg: Dict[str, Any], cors_allow_origins,
                   api_key: Optional[str], strict_auth: bool, api_dependency,
                   connection_timeout, lifespan) -> FastAPI:
    """Create a sub-application for an MCP server."""
    sub_app = FastAPI(
        title=f"{server_name}",
        description=f"{server_name} MCP Server\n\n- [back to tool list](/docs)",
        version="1.0",
        lifespan=lifespan,
    )

    sub_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_credentials=cors_allow_origins is not None and cors_allow_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Configure server type and connection parameters
    if server_cfg.get("command"):
        # stdio
        sub_app.state.server_type = "stdio"
        sub_app.state.command = server_cfg["command"]
        sub_app.state.args = server_cfg.get("args", [])
        sub_app.state.env = {**os.environ, **server_cfg.get("env", {})}

    server_config_type = server_cfg.get("type")
    if normalize_server_type(server_config_type) == "sse" and server_cfg.get("url"):
        sub_app.state.server_type = normalize_server_type("sse")
        sub_app.state.args = [server_cfg["url"]]
        headers = dict(server_cfg.get("headers") or {})
        headers["MCP-Protocol-Version"] = MCP_VERSION
        sub_app.state.headers = headers
    elif normalize_server_type(server_config_type) == "streamable-http" and server_cfg.get("url"):
        url = server_cfg["url"]
        sub_app.state.server_type = normalize_server_type("streamable-http")
        sub_app.state.args = [url]
        headers = dict(server_cfg.get("headers") or {})
        headers["MCP-Protocol-Version"] = MCP_VERSION
        sub_app.state.headers = headers
    elif not server_config_type and server_cfg.get("url"):
        # Fallback for old SSE config
        sub_app.state.server_type = normalize_server_type("sse")
        sub_app.state.args = [server_cfg["url"]]
        headers = dict(server_cfg.get("headers") or {})
        headers["MCP-Protocol-Version"] = MCP_VERSION
        sub_app.state.headers = headers

    if api_key and strict_auth:
        sub_app.add_middleware(APIKeyMiddleware, api_key=api_key)

    sub_app.state.api_dependency = api_dependency
    sub_app.state.connection_timeout = connection_timeout
    sub_app.state.disabled_tools = server_cfg.get("disabledTools", [])
    # Keep a stable config key for operation_id generation and discovery
    sub_app.state.config_key = server_name

    return sub_app


def mount_config_servers(main_app: FastAPI, config_data: Dict[str, Any],
                        cors_allow_origins, api_key: Optional[str], strict_auth: bool,
                        api_dependency, connection_timeout, lifespan, path_prefix: str):
    """Mount MCP servers from config data."""
    mcp_servers = config_data.get("mcpServers", {})

    logger.info("Configuring MCP Servers:")
    for server_name, server_cfg in mcp_servers.items():
        sub_app = create_sub_app(
            server_name, server_cfg, cors_allow_origins, api_key,
            strict_auth, api_dependency, connection_timeout, lifespan
        )
        # Link back to main app
        sub_app.state.parent_app = main_app
        main_app.mount(f"{path_prefix}{server_name}", sub_app)


def unmount_servers(main_app: FastAPI, path_prefix: str, server_names: list):
    """Unmount specific MCP servers."""
    active_lifespans = getattr(main_app.state, 'active_lifespans', {})
    for server_name in server_names:
        mount_path = f"{path_prefix}{server_name}"

        # Clean up lifespan context if it exists
        if server_name in active_lifespans:
            lifespan_context = active_lifespans[server_name]
            try:
                asyncio.create_task(lifespan_context.__aexit__(None, None, None))
            except Exception as e:
                logger.warning(f"Error cleaning up lifespan for {server_name}: {e}")
            finally:
                del active_lifespans[server_name]

        # Find and remove the mount
        routes_to_remove = []
        for route in main_app.router.routes:
            if hasattr(route, 'path') and route.path == mount_path:
                routes_to_remove.append(route)

        for route in routes_to_remove:
            main_app.router.routes.remove(route)
            logger.info(f"Unmounted server: {server_name}")


async def reload_config_handler(main_app: FastAPI, new_config_data: Dict[str, Any]):
    """Handle config reload by comparing and updating mounted servers."""
    async with _reload_lock:
        old_config_data = getattr(main_app.state, 'config_data', {})
        backup_routes = list(main_app.router.routes)  # Backup current routes for rollback

        try:
            old_servers = set(old_config_data.get("mcpServers", {}).keys())
            new_servers = set(new_config_data.get("mcpServers", {}).keys())

            servers_to_add = new_servers - old_servers
            servers_to_remove = old_servers - new_servers
            servers_to_check = old_servers & new_servers

            cors_allow_origins = getattr(main_app.state, 'cors_allow_origins', ["*"])
            api_key = getattr(main_app.state, 'api_key', None)
            strict_auth = getattr(main_app.state, 'strict_auth', False)
            api_dependency = getattr(main_app.state, 'api_dependency', None)
            connection_timeout = getattr(main_app.state, 'connection_timeout', None)
            lifespan = getattr(main_app.state, 'lifespan', None)
            path_prefix = getattr(main_app.state, 'path_prefix', "/")
            state_manager = get_state_manager()

            if servers_to_remove:
                logger.info(f"Removing servers: {list(servers_to_remove)}")
                unmount_servers(main_app, path_prefix, list(servers_to_remove))

            servers_to_update = []
            for server_name in servers_to_check:
                old_cfg = old_config_data["mcpServers"][server_name]
                new_cfg = new_config_data["mcpServers"][server_name]
                if old_cfg != new_cfg:
                    servers_to_update.append(server_name)

            if servers_to_update:
                logger.info(f"Updating servers: {servers_to_update}")
                unmount_servers(main_app, path_prefix, servers_to_update)
                servers_to_add.update(servers_to_update)

            if servers_to_add:
                logger.info(f"Adding servers: {list(servers_to_add)}")

                # Track lifespan contexts for dynamically added sub-apps
                if not hasattr(main_app.state, 'active_lifespans'):
                    main_app.state.active_lifespans = {}

                for server_name in servers_to_add:
                    server_cfg = new_config_data["mcpServers"][server_name]
                    try:
                        sub_app = create_sub_app(
                            server_name, server_cfg, cors_allow_origins, api_key,
                            strict_auth, api_dependency, connection_timeout, lifespan
                        )
                        sub_app.state.parent_app = main_app
                        main_app.mount(f"{path_prefix}{server_name}", sub_app)
                        # Ensure state manager tracks the server so UI reflects it immediately
                        try:
                            current_state = state_manager.get_server_state(server_name)
                            state_manager.set_server_enabled(server_name, current_state.get("enabled", True))
                        except Exception as state_err:  # pragma: no cover - defensive
                            logger.warning(f"Failed to persist state for server '{server_name}': {state_err}")

                        # Start lifespan for newly mounted sub-app and track for cleanup
                        try:
                            lifespan_context = sub_app.router.lifespan_context(sub_app)
                            await lifespan_context.__aenter__()
                            main_app.state.active_lifespans[server_name] = lifespan_context
                            is_connected = getattr(sub_app.state, "is_connected", False)
                            if is_connected:
                                logger.info(f"Successfully connected to new server: '{server_name}'")
                                sub_app.state.last_error = None
                            else:
                                logger.warning(f"Failed to connect to new server: '{server_name}'")
                        except Exception as init_err:  # pragma: no cover - defensive
                            sub_app.state.is_connected = False
                            sub_app.state.last_error = str(init_err)
                            logger.error(
                                f"Failed to initialize server '{server_name}': {init_err}. "
                                "Server remains mounted but marked disconnected."
                            )
                    except Exception as e:
                        logger.error(f"Failed to create server '{server_name}': {e}")
                        main_app.router.routes = backup_routes
                        raise

            main_app.state.config_data = new_config_data
            _health_state["generation"] += 1
            _health_state["last_reload"] = datetime.now(timezone.utc).isoformat()
            _update_health_snapshot(main_app)
            logger.info("Config reload completed successfully")
        except Exception as e:
            logger.error(f"Error during config reload, keeping previous configuration: {e}")
            main_app.router.routes = backup_routes
            raise


async def initialize_sub_app(sub_app: FastAPI):
    """Initialize a mounted sub-app by establishing an MCP session and creating tool endpoints.

    Mirrors the logic inside the lifespan branch for sub-apps but callable on-demand
    after dynamic (re)mounts.
    """
    if getattr(sub_app.state, 'is_connected', False):
        # Already initialized
        return
    server_type = normalize_server_type(getattr(sub_app.state, 'server_type', 'stdio'))
    command = getattr(sub_app.state, 'command', None)
    args = getattr(sub_app.state, 'args', [])
    args = args if isinstance(args, list) else [args]
    env = getattr(sub_app.state, 'env', {})
    connection_timeout = getattr(sub_app.state, 'connection_timeout', 10)
    api_dependency = getattr(sub_app.state, 'api_dependency', None)
    # Remove old tool endpoints if any (POST /toolName at root)
    retained = []
    for r in sub_app.router.routes:
        methods = getattr(r, 'methods', set())
        path = getattr(r, 'path', '')
        if 'POST' in methods and path.count('/') == 1 and path not in ('/openapi.json', '/docs'):
            # drop tool endpoint
            continue
        retained.append(r)
    sub_app.router.routes = retained
    try:
        if server_type == 'stdio':
            server_params = StdioServerParameters(
                command=command,
                args=args,
                env={**os.environ, **env},
            )
            client_context = stdio_client(server_params)
        elif server_type == 'sse':
            headers = getattr(sub_app.state, 'headers', None)
            client_context = sse_client(
                url=args[0],
                sse_read_timeout=connection_timeout or 900,
                headers=headers,
            )
        elif server_type == 'streamable-http':
            headers = getattr(sub_app.state, 'headers', None)
            client_context = streamablehttp_client(url=args[0], headers=headers)
        else:
            raise ValueError(f"Unsupported server type: {server_type}")
        async with client_context as (reader, writer, *_):
            async with ClientSession(reader, writer) as session:
                sub_app.state.session = session
                await create_dynamic_endpoints(sub_app, api_dependency=api_dependency)
                sub_app.state.is_connected = True
    except Exception:
        sub_app.state.is_connected = False
        raise


async def create_dynamic_endpoints(app: FastAPI, api_dependency=None):
    session: ClientSession = app.state.session
    if not session:
        raise ValueError("Session is not initialized in the app state.")

    result = await session.initialize()
    server_info = getattr(result, "serverInfo", None)
    if server_info:
        app.title = server_info.name or app.title
        app.description = (
            f"{server_info.name} MCP Server" if server_info.name else app.description
        )
        app.version = server_info.version or app.version

    instructions = getattr(result, "instructions", None)
    if instructions:
        app.description = instructions

    tools_result = await session.list_tools()
    tools = tools_result.tools

    # Filter out disabled tools configured for this server
    disabled_tools = getattr(app.state, "disabled_tools", [])
    if disabled_tools:
        original_count = len(tools)
        tools = [tool for tool in tools if tool.name not in disabled_tools]
        filtered_count = original_count - len(tools)
        if filtered_count > 0:
            logger.info(
                f"Filtered out {filtered_count} tool(s) for server '{app.title}': {disabled_tools}"
            )

    # Prefer config key from state to ensure uniqueness across servers
    server_key = getattr(app.state, "config_key", None) or (app.title or "server").replace(" ", "_")

    for tool in tools:
        endpoint_name = tool.name
        endpoint_description = tool.description

        inputSchema = tool.inputSchema
        outputSchema = getattr(tool, "outputSchema", None)

        form_model_fields = get_model_fields(
            f"{endpoint_name}_form_model",
            inputSchema.get("properties", {}),
            inputSchema.get("required", []),
            inputSchema.get("$defs", {}),
        )

        response_model_fields = None
        if outputSchema:
            response_model_fields = get_model_fields(
                f"{endpoint_name}_response_model",
                outputSchema.get("properties", {}),
                outputSchema.get("required", []),
                outputSchema.get("$defs", {}),
            )

        tool_handler = get_tool_handler(
            session,
            endpoint_name,
            form_model_fields,
            response_model_fields,
        )

        app.post(
            f"/{endpoint_name}",
            summary=endpoint_name.replace("_", " ").title(),
            description=endpoint_description,
            response_model_exclude_none=True,
            operation_id=f"{server_key}.{endpoint_name}",
            dependencies=[Depends(api_dependency)] if api_dependency else [],
        )(tool_handler)


@asynccontextmanager
async def lifespan(app: FastAPI):
    server_type = normalize_server_type(getattr(app.state, "server_type", "stdio"))
    command = getattr(app.state, "command", None)
    args = getattr(app.state, "args", [])
    args = args if isinstance(args, list) else [args]
    env = getattr(app.state, "env", {})
    connection_timeout = getattr(app.state, "connection_timeout", 10)
    api_dependency = getattr(app.state, "api_dependency", None)
    path_prefix = getattr(app.state, "path_prefix", "/")

    # Get shutdown handler from app state
    shutdown_handler = getattr(app.state, "shutdown_handler", None)

    is_main_app = not command and not (server_type in ["sse", "streamable-http"] and args)

    if is_main_app:
        async with AsyncExitStack() as stack:
            successful_servers = []
            failed_servers = []

            sub_lifespans = [
                (route.app, route.app.router.lifespan_context(route.app))
                for route in app.routes
                if isinstance(route, Mount) and isinstance(route.app, FastAPI)
            ]

            for sub_app, lifespan_context in sub_lifespans:
                server_name = sub_app.title
                
                # Skip internal MCPO management server - it doesn't need MCP connection
                if server_name == "MCPO Management Server":
                    logger.info(f"Skipping connection for internal server: '{server_name}' (already available)")
                    successful_servers.append(server_name)
                    continue
                
                logger.info(f"Initiating connection for server: '{server_name}'...")
                try:
                    await stack.enter_async_context(lifespan_context)
                    is_connected = getattr(sub_app.state, "is_connected", False)
                    if is_connected:
                        logger.info(f"Successfully connected to '{server_name}'.")
                        successful_servers.append(server_name)
                    else:
                        logger.warning(
                            f"Connection attempt for '{server_name}' finished, but status is not 'connected'."
                        )
                        failed_servers.append(server_name)
                except Exception as e:
                    error_class_name = type(e).__name__
                    if error_class_name == 'ExceptionGroup' or (hasattr(e, 'exceptions') and hasattr(e, 'message')):
                        logger.error(
                            f"Failed to establish connection for server: '{server_name}' - Multiple errors occurred:"
                        )
                        # Log each individual exception from the group
                        exceptions = getattr(e, 'exceptions', [])
                        for idx, exc in enumerate(exceptions):
                            logger.error(f"  Error {idx + 1}: {type(exc).__name__}: {exc}")
                            # Also log traceback for each exception
                            if hasattr(exc, '__traceback__'):
                                import traceback
                                tb_lines = traceback.format_exception(type(exc), exc, exc.__traceback__)
                                for line in tb_lines:
                                    logger.debug(f"    {line.rstrip()}")
                    else:
                        logger.error(
                            f"Failed to establish connection for server: '{server_name}' - {type(e).__name__}: {e}",
                            exc_info=True
                        )
                    failed_servers.append(server_name)

            logger.info("\n--- Server Startup Summary ---")
            if successful_servers:
                logger.info("Successfully connected to:")
                for name in successful_servers:
                    logger.info(f"  - {name}")
                app.description += "\n\n- **available tools**："
                for name in successful_servers:
                    docs_path = urljoin(path_prefix, f"{name}/docs")
                    app.description += f"\n    - [{name}]({docs_path})"
            if failed_servers:
                logger.warning("Failed to connect to:")
                for name in failed_servers:
                    logger.warning(f"  - {name}")
            logger.info("--------------------------\n")

            if not successful_servers:
                logger.error("No MCP servers could be reached.")

            yield
            # The AsyncExitStack will handle the graceful shutdown of all servers
            # when the 'with' block is exited.
    else:
        # This is a sub-app's lifespan
        app.state.is_connected = False
        try:
            if server_type == "stdio":
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env={**os.environ, **env},
                )
                client_context = stdio_client(server_params)
            elif server_type == "sse":
                headers = getattr(app.state, "headers", None)
                client_context = sse_client(
                    url=args[0],
                    sse_read_timeout=connection_timeout or 900,
                    headers=headers,
                )
            elif server_type == "streamable-http":
                headers = getattr(app.state, "headers", None)
                client_context = streamablehttp_client(url=args[0], headers=headers)
            else:
                raise ValueError(f"Unsupported server type: {server_type}")

            async with client_context as (reader, writer, *_):
                async with ClientSession(reader, writer) as session:
                    app.state.session = session
                    await create_dynamic_endpoints(app, api_dependency=api_dependency)
                    app.state.is_connected = True
                    yield
        except Exception as e:
            # Log the full exception with traceback for debugging
            logger.error(f"Failed to connect to MCP server '{app.title}': {type(e).__name__}: {e}", exc_info=True)
            app.state.is_connected = False
            # Re-raise the exception so it propagates to the main app's lifespan
            raise


async def create_internal_mcpo_server(main_app: FastAPI, api_dependency) -> FastAPI:
    """Create internal MCPO MCP server exposing management tools."""
    mcpo_app = FastAPI(
        title="MCPO Management Server",
        description="Internal MCP server exposing MCPO management capabilities",
        version="1.0",
        servers=[{"url": "/mcpo"}]
    )
    # Request models for clear OpenAPI requestBody schemas
    class PostConfigBody(BaseModel):
        config: Any

    class PostEnvBody(BaseModel):
        env_vars: Dict[str, Any]

    class PostRequirementsBody(BaseModel):
        content: str

    class InstallPythonPackageBody(BaseModel):
        package_name: str

    
    # Store reference to main app for accessing state
    mcpo_app.state.main_app = main_app

    @mcpo_app.post(
        "/install_python_package",
        summary="Install Python Package",
        description=(
            "Install exactly one named package. \n\n"
            "Rules:\n"
            "- Single package only; no bulk installs.\n"
            "- Do not add upgrade flags (e.g., -U/--upgrade) unless explicitly requested.\n"
            "- Do not install extras or transitive tools beyond what is requested.\n"
            "- If installation fails, report the error and stop. Do not retry with guesses."
        ),
        operation_id="mcpo.install_python_package",
        openapi_extra={
            "x-instructions": [
                "Install only the provided package_name.",
                "No upgrades or version changes unless explicitly specified by the user.",
                "No additional flags or extras.",
                "On error, return the error; do not attempt speculative fixes."
            ]
        },
    )
    async def install_python_package(request: Request, body: InstallPythonPackageBody):
        """Install a Python package using pip."""
        # Block in read-only mode
        main_app = request.app
        if getattr(main_app.state, 'read_only_mode', False):
            return JSONResponse(status_code=403, content={"ok": False, "error": {"message": "Read-only mode"}})
        try:
            package_name = body.package_name if body else None
            if not package_name or not isinstance(package_name, str):
                logger.warning("install_python_package: Missing package_name parameter")
                return JSONResponse(status_code=422, content={"ok": False, "error": {"message": "Missing package_name"}})

            # Validate package name: alphanumeric, hyphens, underscores, dots, and optional version spec
            import re as _re
            if not _re.match(r'^[A-Za-z0-9]([A-Za-z0-9._-]*[A-Za-z0-9])?(\[.*\])?(([<>=!~]=?|@)[A-Za-z0-9._*+-]*)*$', package_name):
                logger.warning(f"install_python_package: Invalid package name '{package_name}'")
                return JSONResponse(status_code=422, content={"ok": False, "error": {"message": f"Invalid package name: {package_name}"}})

            logger.info(f"install_python_package: Starting installation of package '{package_name}'")

            # Run pip install
            result = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                timeout=120,
            )

            if result.returncode == 0:
                logger.info(f"install_python_package: Successfully installed package '{package_name}'")
                return {"ok": True, "message": f"Successfully installed {package_name}", "output": result.stdout}
            else:
                logger.error(
                    f"install_python_package: Failed to install package '{package_name}' - {result.stderr}"
                )
                return JSONResponse(
                    status_code=500,
                    content={
                        "ok": False,
                        "error": {
                            "message": f"Failed to install {package_name}",
                            "details": result.stderr,
                        },
                    },
                )

        except subprocess.TimeoutExpired:
            logger.error(
                f"install_python_package: Installation of package '{package_name}' timed out after 120 seconds"
            )
            return JSONResponse(
                status_code=408, content={"ok": False, "error": {"message": "Installation timed out"}}
            )
        except Exception as e:
            logger.error(f"install_python_package: Installation failed with exception - {str(e)}")
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": {"message": f"Installation failed: {str(e)}"}},
            )

    @mcpo_app.get(
        "/get_config",
        summary="Get Configuration",
        description=(
            "Read-only retrieval of the current mcpo.json. Use this to fetch the source of truth "
            "before proposing any changes."
        ),
        operation_id="mcpo.get_config",
        openapi_extra={
            "x-instructions": [
                "This endpoint is read-only.",
                "Call this before any configuration change to obtain the exact current object.",
                "Do not infer, normalize, or fix values; if something seems wrong, ask instead."
            ]
        },
    )
    async def get_config_get(request: Request):
        try:
            main_app = mcpo_app.state.main_app
            config_path = getattr(main_app.state, 'config_path', None)
            if not config_path:
                return JSONResponse(
                    status_code=400,
                    content={"ok": False, "error": {"message": "No config file configured"}},
                )
            with open(config_path, 'r', encoding='utf-8') as f:
                config_content = f.read()
            config_data = json.loads(config_content)
            return {"config": config_data}
        except Exception as e:
            return JSONResponse(
                status_code=500,
                content={"ok": False, "error": {"message": f"Failed to get config: {str(e)}"}},
            )

    # legacy POST /get_config removed to enforce GET-only for reads
    
    @mcpo_app.post(
        "/post_config",
        summary="Update Configuration",
        description=(
            "Apply the minimal, explicitly requested change to the configuration and reload. \n\n"
            "Process:\n"
            "1) First GET /mcpo/get_config and start from the returned object.\n"
            "2) Modify only the fields explicitly requested.\n"
            "3) Preserve all other fields exactly; no reformatting, reordering, or inferred fixes.\n"
            "4) Post a valid JSON object (no comments). On validation error, stop and report."
        ),
        operation_id="mcpo.post_config",
        openapi_extra={
            "x-instructions": [
                "Always read current config first via GET /mcpo/get_config.",
                "Perform minimal diffs only; do not touch unrelated fields.",
                "Preserve structure and values verbatim for untouched areas.",
                "No speculative fixes or improvements; ask if unsure.",
                "Submit a valid JSON object; if an error occurs, stop and report."
            ]
        },
    )
    async def post_config(request: Request, body: PostConfigBody):
        """Update MCPO configuration."""
        try:
            main_app = mcpo_app.state.main_app
            config_path = getattr(main_app.state, 'config_path', None)
            if not config_path:
                return JSONResponse(status_code=400, content={"ok": False, "error": {"message": "No config file configured"}})
            
            config_data = body.config
            if not config_data:
                return JSONResponse(status_code=422, content={"ok": False, "error": {"message": "Missing config data"}})
            
            # Validate JSON
            if isinstance(config_data, str):
                config_data = json.loads(config_data)
            
            # Backup existing config
            backup_path = f"{config_path}.backup"
            if os.path.exists(config_path):
                import shutil
                shutil.copy2(config_path, backup_path)
            
            # Save new config
            with open(config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            # Trigger reload
            await reload_config_handler(main_app, config_data)
            
            return {"ok": True, "message": "Configuration updated and reloaded", "backup": backup_path}
            
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=422, content={"ok": False, "error": {"message": f"Invalid JSON: {str(e)}"}})
        except Exception as e:
            return JSONResponse(status_code=500, content={"ok": False, "error": {"message": f"Failed to update config: {str(e)}"}})
    
    @mcpo_app.get(
        "/get_logs",
        summary="Get Server Logs",
        description="Read-only access to recent server log entries.",
        operation_id="mcpo.get_logs",
        openapi_extra={
            "x-instructions": [
                "Use logs for observation only.",
                "Keep limits small (default 20).",
                "Do not trigger writes based on interpretations without explicit instruction."
            ]
        },
    )
    async def get_logs_get(request: Request, limit: int = Query(20, ge=1, le=100)):
        """Get recent server logs."""
        try:
            log_manager = get_log_manager(MAX_LOG_ENTRIES)
            recent_logs = log_manager.get_logs(source="openapi", limit=limit)

            return {"ok": True, "logs": recent_logs, "count": len(recent_logs)}
        except Exception as e:
            return JSONResponse(status_code=500, content={"ok": False, "error": {"message": f"Failed to get logs: {str(e)}"}})

    # legacy POST /get_logs removed to enforce GET-only for reads
    
    @mcpo_app.post(
        "/post_env",
        summary="Update Environment Variables",
        description=(
            "Write only the provided keys to .env. Secrets are opaque; do not echo values."
        ),
        operation_id="mcpo.post_env",
        openapi_extra={
            "x-instructions": [
                "Only set the keys provided by the user.",
                "Do not modify or delete other keys.",
                "Never echo or log secret values.",
                "Treat values as opaque strings."
            ]
        },
    )
    async def post_env(request: Request, body: PostEnvBody):
        """Update .env file with environment variables."""
        try:
            env_vars = body.env_vars
            if not env_vars or not isinstance(env_vars, dict):
                logger.warning("post_env: Missing or invalid env_vars parameter")
                return JSONResponse(status_code=422, content={"ok": False, "error": {"message": "Missing or invalid env_vars (must be dict)"}})
            
            logger.info(f"post_env: Updating {len(env_vars)} environment variables: {list(env_vars.keys())}")
            
            # Read existing .env file
            env_path = ".env"
            existing_vars = {}
            if os.path.exists(env_path):
                with open(env_path, 'r') as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            existing_vars[key] = value
            
            # Update with new variables
            existing_vars.update(env_vars)
            
            # Write back to .env file
            with open(env_path, 'w') as f:
                for key, value in existing_vars.items():
                    f.write(f"{key}={value}\n")
            
            # Set in current environment
            for key, value in env_vars.items():
                os.environ[key] = str(value)
            
            logger.info(f"post_env: Successfully updated environment variables: {list(env_vars.keys())}")
            return {"ok": True, "message": f"Updated {len(env_vars)} environment variables", "keys": list(env_vars.keys())}
            
        except Exception as e:
            logger.error(f"post_env: Failed to update environment variables: {str(e)}")
            return JSONResponse(status_code=500, content={"ok": False, "error": {"message": f"Failed to update env: {str(e)}"}})
    
    # setup_server endpoint removed per design: enforce granular operations only
    
    @mcpo_app.post(
        "/validate_and_install",
        summary="Validate Config and Install Dependencies",
        description=(
            "Validate current configuration and, if requirements.txt exists, install dependencies."
        ),
        operation_id="mcpo.validate_and_install",
        openapi_extra={
            "x-instructions": [
                "Use for validation and dependency setup only when explicitly requested.",
                "Do not alter configuration content beyond validation scope."
            ]
        },
    )
    async def validate_and_install(request: Request, form_data: Dict[str, Any]):
        """Validate configuration and install dependencies."""
        try:
            logger.info("validate_and_install: Starting validation and dependency installation")
            
            results = {
                "validation": {"status": "unknown", "errors": []},
                "installation": {"status": "unknown", "packages": []},
                "success": True
            }
            
            # Step 1: Validate current configuration
            try:
                main_app = mcpo_app.state.main_app
                config_path = getattr(main_app.state, 'config_path', None)
                
                if config_path and os.path.exists(config_path):
                    with open(config_path, 'r') as f:
                        config_data = json.load(f)  # This will raise JSONDecodeError if invalid
                    
                    # Basic validation
                    if "mcpServers" not in config_data:
                        results["validation"]["errors"].append("Missing 'mcpServers' section")
                        results["success"] = False
                    else:
                        server_count = len(config_data["mcpServers"])
                        results["validation"]["status"] = "valid"
                        results["validation"]["details"] = f"Configuration valid with {server_count} servers"
                        logger.info(f"validate_and_install: Configuration valid with {server_count} servers")
                else:
                    results["validation"]["errors"].append("No configuration file found")
                    results["success"] = False
                    
            except json.JSONDecodeError as e:
                results["validation"]["status"] = "invalid"
                results["validation"]["errors"].append(f"Invalid JSON: {str(e)}")
                results["success"] = False
                logger.error(f"validate_and_install: JSON validation failed - {str(e)}")
            except Exception as e:
                results["validation"]["status"] = "error"
                results["validation"]["errors"].append(f"Validation error: {str(e)}")
                results["success"] = False
                logger.error(f"validate_and_install: Validation error - {str(e)}")
            
            # Step 2: Install dependencies from requirements.txt if it exists
            if os.path.exists("requirements.txt"):
                try:
                    logger.info("validate_and_install: Installing dependencies from requirements.txt")
                    with open("requirements.txt", 'r') as f:
                        requirements = f.read().strip()
                    
                    if requirements:
                        result = subprocess.run(
                            [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                            capture_output=True, text=True, timeout=300
                        )
                        
                        if result.returncode == 0:
                            results["installation"]["status"] = "success"
                            results["installation"]["details"] = "Dependencies installed successfully"
                            logger.info("validate_and_install: Dependencies installed successfully")
                        else:
                            results["installation"]["status"] = "failed"
                            results["installation"]["error"] = result.stderr
                            results["success"] = False
                            logger.error(f"validate_and_install: Dependency installation failed - {result.stderr}")
                    else:
                        results["installation"]["status"] = "skipped"
                        results["installation"]["details"] = "No dependencies to install"
                        
                except Exception as e:
                    results["installation"]["status"] = "error"
                    results["installation"]["error"] = str(e)
                    results["success"] = False
                    logger.error(f"validate_and_install: Installation error - {str(e)}")
            else:
                results["installation"]["status"] = "skipped"
                results["installation"]["details"] = "No requirements.txt found"
            
            if results["success"]:
                logger.info("validate_and_install: Validation and installation completed successfully")
            else:
                logger.warning("validate_and_install: Validation and installation completed with errors")
            
            return results
            
        except Exception as e:
            logger.error(f"validate_and_install: Critical failure - {str(e)}")
            return JSONResponse(status_code=500, content={"ok": False, "error": {"message": f"Validation failed: {str(e)}"}})

    @mcpo_app.get(
        "/get_requirements",
        summary="Get requirements.txt contents",
        description="Read-only view of requirements.txt.",
        operation_id="mcpo.get_requirements",
        openapi_extra={
            "x-instructions": [
                "Do not make changes via this endpoint.",
                "Use POST /mcpo/post_requirements for edits."
            ]
        },
    )
    async def get_requirements_get(request: Request):
        try:
            if os.path.exists("requirements.txt"):
                with open("requirements.txt", "r", encoding="utf-8") as f:
                    content = f.read()
                return {"ok": True, "content": content}
            # create a sensible default template when missing
            return {"ok": True, "content": "# Python packages for MCP servers\n"}
        except Exception as e:
            return JSONResponse(status_code=500, content={"ok": False, "error": {"message": f"Failed to read requirements: {str(e)}"}})

    # legacy POST /get_requirements removed to enforce GET-only for reads

    @mcpo_app.post(
        "/post_requirements",
        summary="Write requirements.txt and install",
        description=(
            "Write the provided content verbatim to requirements.txt, install with pip -r, and attempt a hot reload. \n\n"
            "Rules:\n"
            "- Add or remove only what was explicitly requested.\n"
            "- Do not upgrade unrelated packages or add flags.\n"
            "- Keep other lines unchanged."
        ),
        operation_id="mcpo.post_requirements",
        openapi_extra={
            "x-instructions": [
                "Edit exactly as requested; preserve unrelated lines.",
                "Do not add upgrade flags or extras.",
                "On install error, report and stop."
            ]
        },
    )
    async def post_requirements(request: Request, body: PostRequirementsBody):
        try:
            content = body.content
            if content is None or not isinstance(content, str):
                return JSONResponse(status_code=422, content={"ok": False, "error": {"message": "Missing or invalid 'content'"}})

            # Write requirements.txt
            try:
                with open("requirements.txt", "w", encoding="utf-8") as f:
                    f.write(content)
            except Exception as e:
                return JSONResponse(status_code=500, content={"ok": False, "error": {"message": f"Failed to write requirements.txt: {str(e)}"}})

            # Parse packages for response
            packages = [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith('#')
            ]

            # Install with pip
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode != 0:
                    logger.error("pip install failed: %s", result.stderr)
                    return JSONResponse(status_code=500, content={"ok": False, "error": {"message": "Dependency installation failed", "details": result.stderr}})
            except subprocess.TimeoutExpired:
                return JSONResponse(status_code=408, content={"ok": False, "error": {"message": "Dependency installation timed out"}})

            # Attempt hot reload if config is present
            try:
                main = mcpo_app.state.main_app
                cfg_path = getattr(main.state, 'config_path', None)
                if cfg_path and os.path.exists(cfg_path):
                    new_config = load_config(cfg_path)
                    await reload_config_handler(main, new_config)
            except Exception as e:
                logger.warning("Reload after requirements install failed: %s", e)

            return {"ok": True, "message": "Requirements installed and servers reloaded", "packages": packages}
        except Exception as e:
            return JSONResponse(status_code=500, content={"ok": False, "error": {"message": f"Failed to process requirements: {str(e)}"}})
    
    # Add API key protection if main app has it
    if api_dependency:
        for route in mcpo_app.routes:
            if hasattr(route, 'dependencies'):
                route.dependencies.append(Depends(api_dependency))
    
    return mcpo_app


async def build_main_app(
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: Optional[str] = "",
    cors_allow_origins=["*"],
    **kwargs,
) -> FastAPI:
    """Build the main FastAPI application without running it."""
    hot_reload = kwargs.get("hot_reload", False)
    # Server API Key
    api_dependency = get_verify_api_key(api_key) if api_key else None
    connection_timeout = kwargs.get("connection_timeout", None)
    strict_auth = kwargs.get("strict_auth", False)
    tool_timeout = int(kwargs.get("tool_timeout", 30))
    tool_timeout_max = int(kwargs.get("tool_timeout_max", 600))
    structured_output = kwargs.get("structured_output", False)
    read_only_mode = kwargs.get("read_only", False)
    protocol_version_mode = kwargs.get("protocol_version_mode", "warn")  # off|warn|enforce
    validate_output_mode = kwargs.get("validate_output_mode", "off")  # off|warn|enforce
    supported_protocol_versions = kwargs.get("supported_protocol_versions") or [MCP_VERSION]

    # MCP Server
    server_type = normalize_server_type(kwargs.get("server_type"))
    server_command = kwargs.get("server_command")

    # MCP Config
    config_path = kwargs.get("config_path")

    # mcpo server
    name = kwargs.get("name") or "MCP OpenAPI Proxy"
    description = (
        kwargs.get("description") or "Automatically generated API from MCP Tool Schemas"
    )
    version = kwargs.get("version") or "1.0"

    ssl_certfile = kwargs.get("ssl_certfile")
    ssl_keyfile = kwargs.get("ssl_keyfile")
    path_prefix = kwargs.get("path_prefix") or "/"
    log_level = kwargs.get("log_level", "info").upper()

    # Configure basic logging
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Add log buffer handler for UI
    root_logger = logging.getLogger()
    root_logger.setLevel(getattr(logging, log_level, logging.INFO))
    if not any(isinstance(handler, BufferedLogHandler) for handler in root_logger.handlers):
        root_logger.addHandler(
            BufferedLogHandler(
                _log_buffer,
                _log_buffer_lock,
                default_source="openapi",
                max_entries=MAX_LOG_ENTRIES,
            )
        )

    # Ensure access logs are retained for troubleshooting; rely on logging config for verbosity
    logging.getLogger("uvicorn.access").setLevel(logging.INFO)
    logging.getLogger("httpx").setLevel(logging.INFO)
    logger.info("Starting MCPO Server...")
    logger.info(f"  Name: {name}")
    logger.info(f"  Version: {version}")
    logger.info(f"  Description: {description}")
    logger.info(f"  Hostname: {socket.gethostname()}")
    logger.info(f"  Port: {port}")
    logger.info(f"  API Key: {'Provided' if api_key else 'Not Provided'}")
    logger.info(f"  CORS Allowed Origins: {cors_allow_origins}")
    if ssl_certfile:
        logger.info(f"  SSL Certificate File: {ssl_certfile}")
    if ssl_keyfile:
        logger.info(f"  SSL Key File: {ssl_keyfile}")
    logger.info(f"  Path Prefix: {path_prefix}")

    # Create shutdown handler
    shutdown_handler = GracefulShutdown()

    main_app = FastAPI(
        title=name,
        description=description,
        version=version,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        lifespan=lifespan,
        openapi_version="3.1.0",
    )
    # Mount admin router for utility endpoints (e.g., /_meta/env)
    try:
        main_app.include_router(admin_router, prefix="/_meta")
    except Exception:
        # Router mount failures should not prevent core startup
        logger.exception("Failed to include admin router")
    try:
        main_app.include_router(chat_router, prefix="/chat")
    except Exception:
        logger.exception("Failed to include chat router")
    try:
        main_app.include_router(completions_router, prefix="")
    except Exception:
        logger.exception("Failed to include completions router")
    # Initialize shared StateManager
    state_manager = get_state_manager()
    main_app.state.state_manager = state_manager
    # Metrics counters (simple in-memory; reset on restart)
    main_app.state.metrics = {
        "tool_calls_total": 0,
        "tool_errors_total": 0,
        "tool_errors_by_code": {},  # code -> count
        "per_tool": {},  # tool_name -> stats dict
    }
    main_app.state.config_path = config_path  # Store for state persistence
    main_app.state.read_only_mode = read_only_mode
    main_app.state.protocol_version_mode = protocol_version_mode
    main_app.state.validate_output_mode = validate_output_mode
    main_app.state.supported_protocol_versions = supported_protocol_versions
    proxy_url = kwargs.get("mcp_proxy_url")
    if not proxy_url:
        proxy_host = host if host and host not in {"0.0.0.0", "::", "0"} else "127.0.0.1"
        proxy_port = kwargs.get("mcp_proxy_port") or (port + 1 if port else 8001)
        proxy_url = f"http://{proxy_host}:{proxy_port}"
    main_app.state.mcp_proxy_url = proxy_url

    # Standardized error handlers
    @main_app.exception_handler(Exception)
    async def generic_exception_handler(request: Request, exc: Exception):  # type: ignore
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(status_code=500, content=error_envelope("Internal Server Error"))

    # Convert FastAPI HTTPExceptions produced elsewhere into unified envelope
    from fastapi import HTTPException as _HTTPException  # type: ignore

    @main_app.exception_handler(_HTTPException)
    async def http_exception_handler(request: Request, exc: _HTTPException):  # type: ignore
        # exc.detail may be dict or str
        if isinstance(exc.detail, dict):
            message = exc.detail.get("message") or exc.detail.get("detail") or "HTTP Error"
            data = exc.detail.get("data") or {k: v for k, v in exc.detail.items() if k not in {"message", "detail"}}
        else:
            message = str(exc.detail)
            data = None
        return JSONResponse(status_code=exc.status_code, content=error_envelope(message, data=data))

    from fastapi.exceptions import RequestValidationError as _RVE  # local import

    @main_app.exception_handler(_RVE)
    async def validation_exception_handler(request: Request, exc: _RVE):  # type: ignore
        return JSONResponse(status_code=422, content=error_envelope("Validation Error", data=exc.errors()))

    # Pass shutdown handler to app state
    main_app.state.shutdown_handler = shutdown_handler
    main_app.state.path_prefix = path_prefix

    main_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_credentials=cors_allow_origins is not None and cors_allow_origins != ["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Store tool timeout in app state for handler usage
    main_app.state.tool_timeout = tool_timeout
    main_app.state.tool_timeout_max = tool_timeout_max
    main_app.state.structured_output = structured_output

    # Add middleware to protect also documentation and spec
    if api_key and strict_auth:
        main_app.add_middleware(APIKeyMiddleware, api_key=api_key)

    # Register health endpoint early
    _register_health_endpoint(main_app)

    # Startup security / hardening audit warnings
    if not api_key and not read_only_mode:
        logger.warning("Security: Server running without API key and not in read-only mode – management endpoints are writable.")
    if protocol_version_mode != "enforce":
        logger.info(f"Protocol negotiation not enforced (mode={protocol_version_mode}).")
    if validate_output_mode != "enforce":
        logger.info(f"Output schema validation not enforced (mode={validate_output_mode}).")

    # Lightweight meta endpoints (server + tool discovery) and static UI
    @main_app.get("/_meta/servers")
    async def list_servers():  # noqa: D401
        _update_health_snapshot(main_app)
        state_manager = main_app.state.state_manager
        servers = []
        for route in main_app.router.routes:
            if isinstance(route, Mount) and isinstance(route.app, FastAPI):
                sub = route.app
                # Extract config key from mount path (e.g., "/perplexity/" -> "perplexity")
                config_key = route.path.strip('/')
                
                # Special handling for internal MCPO management server
                if sub.title == "MCPO Management Server":
                    is_connected = True  # Internal server is always "connected"
                    server_type = "internal"
                else:
                    is_connected = bool(getattr(sub.state, "is_connected", False))
                    server_type = getattr(sub.state, "server_type", "unknown")
                
                servers.append({
                    "name": config_key,  # Use config key, not sub.title
                    "connected": is_connected,
                    "type": server_type,
                    "basePath": route.path.rstrip('/') + '/',
                    "enabled": state_manager.is_server_enabled(config_key),
                })
        return {"ok": True, "servers": servers}

    @main_app.get("/_meta/servers/{server_name}/tools")
    async def list_server_tools(server_name: str):  # noqa: D401
        state_manager = main_app.state.state_manager
        # Find mounted server
        for route in main_app.router.routes:
            if isinstance(route, Mount) and isinstance(route.app, FastAPI) and route.path.rstrip('/') == f"/{server_name}":
                sub = route.app
                tools = []
                for r in sub.router.routes:
                    if hasattr(r, 'methods') and 'POST' in getattr(r, 'methods', []) and getattr(r, 'path', '/').startswith('/'):
                        p = r.path
                        if p == '/docs' or p.startswith('/openapi'):
                            continue
                        if p.count('/') == 1:  # '/tool'
                            tname = p.lstrip('/')
                            # Special handling for internal MCPO management server
                            if sub.title == "MCPO Management Server":
                                is_enabled = True  # Internal tools are always enabled
                            else:
                                is_enabled = state_manager.is_tool_enabled(server_name, tname)
                            
                            tools.append({
                                "name": tname,
                                "enabled": is_enabled
                            })
                return {"ok": True, "server": server_name, "tools": sorted(tools, key=lambda x: x['name'])}
        return JSONResponse(status_code=404, content=error_envelope("Server not found", code="not_found"))

    @main_app.get("/chat/tools")
    async def list_chat_tools(server: Optional[str] = None):
        state_manager = main_app.state.state_manager
        if server:
            server_names = [server]
        else:
            server_names = [
                name
                for name, state in state_manager.get_all_states().items()
                if state.get("enabled", True)
            ]

        tools: Dict[str, List[str]] = {}
        for route in main_app.router.routes:
            if isinstance(route, Mount) and isinstance(route.app, FastAPI):
                mount_name = route.path.strip('/')
                if mount_name not in server_names:
                    continue
                sub_app = route.app
                collected: List[str] = []
                for r in sub_app.router.routes:
                    methods = getattr(r, 'methods', set())
                    path = getattr(r, 'path', '')
                    if 'POST' in methods and path.count('/') == 1 and path not in ('/docs', '/openapi.json'):
                        collected.append(path.lstrip('/'))
                tools[mount_name] = sorted(collected)

        return {"ok": True, "tools": tools}

    @main_app.get("/_meta/config")
    async def config_info():  # noqa: D401
        path = getattr(main_app.state, 'config_path', None)
        return {"ok": True, "configPath": path}

    @main_app.get("/_meta/metrics")
    async def metrics_snapshot():  # noqa: D401
        """Return a lightweight operational metrics snapshot.

        This is a stub (no persistence / no histograms) intended to be extended.
        Counts are derived from in-memory state; tool call/error counters are incremented
        inside the dynamic tool handlers.
        """
        state_manager = main_app.state.state_manager
        all_states = state_manager.get_all_states()
        
        servers_total = len(all_states)
        servers_enabled = sum(1 for state in all_states.values() if state.get('enabled', True))
        
        tools_total = 0
        tools_enabled = 0
        for state in all_states.values():
            server_tools = state.get('tools', {})
            tools_total += len(server_tools)
            tools_enabled += sum(1 for enabled in server_tools.values() if enabled)
        
        from mcpo.services.metrics import get_metrics_aggregator
        from mcpo.services.runner import get_runner_service

        aggregator = get_metrics_aggregator()
        runner = get_runner_service()
        runner_per_tool = runner.get_metrics()

        per_tool_metrics = {}
        for tname, stats in runner_per_tool.items():
            if not isinstance(stats, dict):
                continue
            calls = stats.get('calls', 0) or 0
            per_tool_metrics[tname] = {
                'calls': calls,
                'errors': stats.get('errors', 0) or 0,
                'avgLatencyMs': stats.get('avgLatencyMs', 0.0) or 0.0,
                'maxLatencyMs': stats.get('maxLatencyMs', 0.0) or 0.0,
            }

        agg = aggregator.build_metrics(per_tool_metrics)
        return {"ok": True, "metrics": {
            "servers": {"total": servers_total, "enabled": servers_enabled},
            "tools": {"total": tools_total, "enabled": tools_enabled},
            "calls": {"total": agg.get("calls", 0)},
            "errors": agg.get("errors", {}),
            "perTool": agg.get("perTool", {}),
        }}

    @main_app.post("/_meta/reload")
    async def reload_config():  # noqa: D401
        """Force a config reload (only valid when running from a config file).

        This compares config, mounts/unmounts servers, and initializes newly added ones
        so that their tools become available immediately.
        """
        if getattr(main_app.state, 'read_only_mode', False):
            return JSONResponse(status_code=403, content=error_envelope("Read-only mode enabled", code="read_only"))
        path = getattr(main_app.state, 'config_path', None)
        if not path:
            return JSONResponse(status_code=400, content=error_envelope("No config-driven servers active", code="no_config"))
        try:
            new_config = load_config(path)
            await reload_config_handler(main_app, new_config)
            return {"ok": True, "generation": _health_state["generation"], "lastReload": _health_state["last_reload"]}
        except Exception as e:  # pragma: no cover - defensive
            return JSONResponse(status_code=500, content=error_envelope("Reload failed", data=str(e), code="reload_failed"))

    @main_app.post("/_meta/reinit/{server_name}")
    async def reinit_server(server_name: str):  # noqa: D401
        """Tear down and reinitialize a single mounted server session (dynamic reconnect)."""
        if getattr(main_app.state, 'read_only_mode', False):
            return JSONResponse(status_code=403, content=error_envelope("Read-only mode enabled", code="read_only"))
        for route in main_app.router.routes:
            if isinstance(route, Mount) and isinstance(route.app, FastAPI) and route.path.rstrip('/') == f"/{server_name}":
                sub = route.app
                try:
                    if hasattr(sub.state, 'session'):
                        try:
                            sess = getattr(sub.state, 'session')
                            if hasattr(sess, 'close'):
                                await sess.close()  # type: ignore
                        except Exception:  # pragma: no cover
                            pass
                    sub.state.is_connected = False
                    await initialize_sub_app(sub)
                    return {"ok": True, "server": server_name, "connected": bool(getattr(sub.state, 'is_connected', False))}
                except Exception as e:  # pragma: no cover
                    return JSONResponse(status_code=500, content=error_envelope("Reinit failed", data=str(e), code="reinit_failed"))
        return JSONResponse(status_code=404, content=error_envelope("Server not found", code="not_found"))

    @main_app.post("/_meta/servers/{server_name}/enable")
    async def enable_server(server_name: str):
        if getattr(main_app.state, 'read_only_mode', False):
            return JSONResponse(status_code=403, content=error_envelope("Read-only mode enabled", code="read_only"))
        state_manager = main_app.state.state_manager
        state_manager.set_server_enabled(server_name, True)
        return {"ok": True, "server": server_name, "enabled": True}

    @main_app.post("/_meta/servers/{server_name}/disable")
    async def disable_server(server_name: str):
        if getattr(main_app.state, 'read_only_mode', False):
            return JSONResponse(status_code=403, content=error_envelope("Read-only mode enabled", code="read_only"))
        state_manager = main_app.state.state_manager
        state_manager.set_server_enabled(server_name, False)
        return {"ok": True, "server": server_name, "enabled": False}

    @main_app.post("/_meta/servers/{server_name}/tools/{tool_name}/enable")
    async def enable_tool(server_name: str, tool_name: str):
        if getattr(main_app.state, 'read_only_mode', False):
            return JSONResponse(status_code=403, content=error_envelope("Read-only mode enabled", code="read_only"))
        state_manager = main_app.state.state_manager
        state_manager.set_tool_enabled(server_name, tool_name, True)
        return {"ok": True, "server": server_name, "tool": tool_name, "enabled": True}

    @main_app.post("/_meta/servers/{server_name}/tools/{tool_name}/disable")
    async def disable_tool(server_name: str, tool_name: str):
        if getattr(main_app.state, 'read_only_mode', False):
            return JSONResponse(status_code=403, content=error_envelope("Read-only mode enabled", code="read_only"))
        state_manager = main_app.state.state_manager
        state_manager.set_tool_enabled(server_name, tool_name, False)
        return {"ok": True, "server": server_name, "tool": tool_name, "enabled": False}

    @main_app.post("/_meta/servers")
    async def add_server(request: Request):  # noqa: D401
        """Add a new server to config (only in config-driven mode) and reload."""
        if getattr(main_app.state, 'read_only_mode', False):
            return JSONResponse(status_code=403, content=error_envelope("Read-only mode enabled", code="read_only"))
        if not getattr(main_app.state, 'config_path', None):
            return JSONResponse(status_code=400, content=error_envelope("Not running with a config file", code="no_config_mode"))
        
        try:
            payload = await request.json()
        except Exception:
            return JSONResponse(status_code=422, content=error_envelope("Invalid JSON payload", code="invalid"))
            
        name = payload.get("name")
        if not name or not isinstance(name, str):
            return JSONResponse(status_code=422, content=error_envelope("Missing name", code="invalid"))
        # Normalize
        name = name.strip()
        config_data = getattr(main_app.state, 'config_data', {"mcpServers": {}})
        if name in config_data.get("mcpServers", {}):
            return JSONResponse(status_code=409, content=error_envelope("Server already exists", code="exists"))
        server_entry: Dict[str, Any] = {}
        # Accept stdio command
        command_str = payload.get("command")
        if command_str:
            if not isinstance(command_str, str):
                return JSONResponse(status_code=422, content=error_envelope("command must be string", code="invalid"))
            parts = command_str.strip().split()
            if not parts:
                return JSONResponse(status_code=422, content=error_envelope("Empty command", code="invalid"))
            server_entry["command"] = parts[0]
            if len(parts) > 1:
                server_entry["args"] = parts[1:]
        url = payload.get("url")
        stype = payload.get("type")
        if url:
            server_entry["url"] = url
            if stype:
                server_entry["type"] = stype
        env = payload.get("env")
        if env and isinstance(env, dict):
            server_entry["env"] = env
        # Basic validation reuse
        try:
            validate_server_config(name, server_entry)
        except Exception as e:  # pragma: no cover - defensive
            return JSONResponse(status_code=422, content=error_envelope(str(e), code="invalid"))
        # Insert and persist
        config_data.setdefault("mcpServers", {})[name] = server_entry
        main_app.state.config_data = config_data
        # Persist to file
        cfg_path = getattr(main_app.state, 'config_path')
        try:
            with open(cfg_path, 'w') as f:
                json.dump(config_data, f, indent=2, sort_keys=True)
        except Exception as e:  # pragma: no cover
            return JSONResponse(status_code=500, content=error_envelope("Failed to write config", data=str(e), code="io_error"))
        # Reload to mount
        try:
            await reload_config_handler(main_app, config_data)
        except Exception as e:  # pragma: no cover
            return JSONResponse(status_code=500, content=error_envelope("Reload failed", data=str(e), code="reload_failed"))
        return {"ok": True, "server": name}

    @main_app.delete("/_meta/servers/{server_name}")
    async def remove_server(server_name: str):  # noqa: D401
        if not getattr(main_app.state, 'config_path', None):
            return JSONResponse(status_code=400, content=error_envelope("Not running with a config file", code="no_config_mode"))
        config_data = getattr(main_app.state, 'config_data', {"mcpServers": {}})
        if server_name not in config_data.get("mcpServers", {}):
            return JSONResponse(status_code=404, content=error_envelope("Server not found", code="not_found"))
        del config_data["mcpServers"][server_name]
        main_app.state.config_data = config_data
        cfg_path = getattr(main_app.state, 'config_path')
        try:
            with open(cfg_path, 'w') as f:
                json.dump(config_data, f, indent=2, sort_keys=True)
        except Exception as e:  # pragma: no cover
            return JSONResponse(status_code=500, content=error_envelope("Failed to write config", data=str(e), code="io_error"))
        try:
            await reload_config_handler(main_app, config_data)
        except Exception as e:  # pragma: no cover
            return JSONResponse(status_code=500, content=error_envelope("Reload failed", data=str(e), code="reload_failed"))
        return {"ok": True, "removed": server_name}

    @main_app.get("/_meta/logs/sources")
    async def get_log_sources():
        """Get available log sources for UI."""
        proxy_url = getattr(main_app.state, "mcp_proxy_url", None)
        sources = [
            {
                "id": "openapi",
                "label": "OpenAPI Server",
                "port": 8000,
                "available": True
            }
        ]
        if proxy_url:
            sources.append({
                "id": "mcp",
                "label": "MCP Proxy",
                "port": 8001,
                "available": True
            })
        return {"ok": True, "sources": sources}

    @main_app.get("/_meta/logs")
    async def get_logs(
        cursor: Optional[int] = None,
        limit: int = 500,
        source: Optional[str] = None,
        category: Optional[str] = None,
    ):
        """Get recent log entries for UI display."""
        log_manager = get_log_manager(MAX_LOG_ENTRIES)
        log_source = (source or "openapi").lower()
        effective_limit = max(1, min(limit, MAX_LOG_ENTRIES))

        if log_source == "mcp":
            proxy_url = getattr(main_app.state, "mcp_proxy_url", None)
            if not proxy_url:
                return {
                    "ok": False,
                    "error": {"message": "MCP proxy URL not configured"},
                }
            try:
                target = urljoin(proxy_url.rstrip("/") + "/", "_meta/logs")
                async with httpx.AsyncClient(timeout=5.0) as client:
                    response = await client.get(
                        target,
                        params={
                            key: value
                            for key, value in {
                                "cursor": cursor,
                                "limit": effective_limit,
                                "category": category,
                                "source": "mcp",
                            }.items()
                            if value is not None
                        },
                    )
                response.raise_for_status()
                payload = response.json()
                if isinstance(payload, dict):
                    return payload
                return {
                    "ok": False,
                    "error": {"message": "Unexpected response from MCP proxy"},
                }
            except Exception as exc:  # pragma: no cover - network failure path
                logger.warning("Failed to fetch MCP proxy logs: %s", exc)
                return {
                    "ok": False,
                    "error": {"message": f"Failed to fetch MCP logs: {exc}"},
                }

        # Exclude httpx logs from openapi source to avoid showing proxy fetch logs
        exclude_logger = "httpx" if log_source == "openapi" else None
        entries = log_manager.get_logs(
            category=category,
            source=log_source,
            after=cursor,
            limit=effective_limit,
            exclude_logger=exclude_logger,
        )
        latest_cursor = log_manager.get_latest_sequence()
        next_cursor = entries[-1]["sequence"] if entries else latest_cursor
        return {
            "ok": True,
            "logs": entries,
            "nextCursor": next_cursor,
            "latestCursor": latest_cursor,
            "limit": effective_limit,
        }

    @main_app.get("/_meta/config/content")
    async def get_config_content():
        """Get the current config file contents."""
        config_path = getattr(main_app.state, 'config_path', None)
        if not config_path:
            return JSONResponse(status_code=400, content=error_envelope("No config file configured", code="no_config"))
        
        try:
            with open(config_path, 'r') as f:
                content = f.read()
            return {"ok": True, "content": content, "path": config_path}
        except FileNotFoundError:
            return JSONResponse(status_code=404, content=error_envelope("Config file not found", code="not_found"))
        except Exception as e:
            return JSONResponse(status_code=500, content=error_envelope("Failed to read config", data=str(e), code="io_error"))

    @main_app.get("/_meta/config/mcpServers")
    async def get_mcp_servers_content():
        """Return only the mcpServers section as a JSON string."""
        config_path = getattr(main_app.state, 'config_path', None)
        if not config_path:
            return JSONResponse(status_code=400, content=error_envelope("No config file configured", code="no_config"))
        try:
            with open(config_path, 'r') as f:
                cfg = json.load(f)
            mcp_servers = cfg.get("mcpServers", {})
            content = json.dumps(mcp_servers, indent=2)
            return {"ok": True, "content": content, "path": config_path}
        except FileNotFoundError:
            return JSONResponse(status_code=404, content=error_envelope("Config file not found", code="not_found"))
        except Exception as e:
            return JSONResponse(status_code=500, content=error_envelope("Failed to read mcpServers", data=str(e), code="io_error"))

    @main_app.post("/_meta/config/save")
    async def save_config_content(payload: Dict[str, Any]):
        """Save config file contents with validation."""
        config_path = getattr(main_app.state, 'config_path', None)
        if not config_path:
            return JSONResponse(status_code=400, content=error_envelope("No config file configured", code="no_config"))
        
        content = payload.get("content")
        if not content or not isinstance(content, str):
            return JSONResponse(status_code=422, content=error_envelope("Missing or invalid content", code="invalid"))
        
        try:
            # Validate JSON before saving
            config_data = json.loads(content)
            
            # Backup existing config
            backup_path = f"{config_path}.backup"
            if os.path.exists(config_path):
                import shutil
                shutil.copy2(config_path, backup_path)
            
            # Save new config
            with open(config_path, 'w') as f:
                f.write(content)
            
            # Reload configuration
            await reload_config_handler(main_app, config_data)
            
            return {"ok": True, "message": "Configuration saved and reloaded", "backup": backup_path}
            
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=422, content=error_envelope("Invalid JSON format", data=str(e), code="invalid"))
        except Exception as e:
            return JSONResponse(status_code=500, content=error_envelope("Failed to save config", data=str(e), code="io_error"))

    @main_app.post("/_meta/config/mcpServers/save")
    async def save_mcp_servers_content(payload: Dict[str, Any]):
        """Update only the mcpServers section of the config and reload.

        Accepts either:
        - { "content": "{ ... }" } where content is a JSON string of the mcpServers object
        - { "data": { ... } } where data is the parsed mcpServers object
        """
        config_path = getattr(main_app.state, 'config_path', None)
        if not config_path:
            return JSONResponse(status_code=400, content=error_envelope("No config file configured", code="no_config"))

        has_content = "content" in payload
        has_data = "data" in payload
        if not has_content and not has_data:
            return JSONResponse(status_code=422, content=error_envelope("Missing content or data", code="invalid"))

        try:
            if has_content:
                new_mcp = json.loads(payload["content"])
            else:
                new_mcp = payload["data"]

            if not isinstance(new_mcp, dict):
                return JSONResponse(status_code=422, content=error_envelope("mcpServers must be an object", code="invalid_type"))

            # Load full config
            try:
                with open(config_path, 'r') as f:
                    full_cfg = json.load(f)
            except FileNotFoundError:
                full_cfg = {}

            # Replace mcpServers section
            full_cfg["mcpServers"] = new_mcp

            # Persist
            with open(config_path, 'w') as f:
                json.dump(full_cfg, f, indent=2)

            # Reload runtime and remount proxy paths
            new_config = load_config(config_path)
            await reload_config_handler(main_app, new_config)
            _mount_or_remount_fastmcp(main_app, base_path="/mcp")
            main_app.state.aggregate_openapi_dirty = True

            return {"ok": True, "saved": True, "reloaded": True}
        except json.JSONDecodeError as e:
            return JSONResponse(status_code=422, content=error_envelope("Invalid JSON", data=str(e), code="invalid_json"))
        except Exception as e:
            return JSONResponse(status_code=500, content=error_envelope("Failed to save mcpServers", data=str(e), code="io_error"))

    @main_app.get("/_meta/requirements/content")
    async def get_requirements_content():
        """Get requirements.txt content."""
        try:
            if os.path.exists("requirements.txt"):
                with open("requirements.txt", 'r') as f:
                    content = f.read()
                return {"ok": True, "content": content}
            else:
                return {"ok": True, "content": "# Python packages for MCP servers\n"}
        except Exception as e:
            return JSONResponse(status_code=500, content=error_envelope("Failed to read requirements", data=str(e), code="io_error"))

    @main_app.post("/_meta/requirements/save")
    async def save_requirements_content(payload: Dict[str, Any]):
        """Save requirements.txt content."""
        content = payload.get("content")
        if content is None:
            return JSONResponse(status_code=422, content=error_envelope("Missing content", code="invalid"))
        
        try:
            with open("requirements.txt", 'w') as f:
                f.write(content)
            logger.info("Requirements.txt updated; installing packages...")

            # Parse package list for response
            packages = [
                line.strip()
                for line in content.splitlines()
                if line.strip() and not line.strip().startswith('#')
            ]

            # Install packages with pip
            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pip", "install", "-r", "requirements.txt"],
                    capture_output=True,
                    text=True,
                    timeout=600,
                )
                if result.returncode != 0:
                    logger.error("pip install failed: %s", result.stderr)
                    return JSONResponse(
                        status_code=500,
                        content=error_envelope("Dependency installation failed", data=result.stderr, code="pip_failed"),
                    )
            except subprocess.TimeoutExpired:
                return JSONResponse(
                    status_code=408,
                    content=error_envelope("Dependency installation timed out", code="pip_timeout"),
                )

            # Reload servers if running with a config file
            config_path = getattr(main_app.state, 'config_path', None)
            if config_path and os.path.exists(config_path):
                try:
                    new_config = load_config(config_path)
                    await reload_config_handler(main_app, new_config)
                except Exception as e:  # pragma: no cover
                    logger.warning("Reload after requirements install failed: %s", e)

            return {"ok": True, "message": "Requirements installed and servers reloaded", "packages": packages}
            
        except Exception as e:
            return JSONResponse(status_code=500, content=error_envelope("Failed to save requirements", data=str(e), code="io_error"))

    # Create internal MCPO MCP server for self-management tools (mounted under /mcpo)
    mcpo_app = await create_internal_mcpo_server(main_app, api_dependency)
    main_app.mount("/mcpo", mcpo_app, name="mcpo")

    # Mount static UI if available (served at /ui)
    try:
        main_app.mount("/ui", StaticFiles(directory="static/ui", html=True), name="ui")
    except Exception:
        # Ignore if directory missing; keeps zero-config behavior
        pass
    # Provide primary access path /mcp; prefer settings app if present
    try:
        # If a built dist exists (e.g., Vite), serve that, else raw source folder with index.html
        if os.path.isdir("mcp-server-settings/dist"):
            main_app.mount("/mcp", StaticFiles(directory="mcp-server-settings/dist", html=True), name="mcp")
        elif os.path.isfile("mcp-server-settings/index.html"):
            main_app.mount("/mcp", StaticFiles(directory="mcp-server-settings", html=True), name="mcp")
        else:
            # Fallback to minimal UI
            main_app.mount("/mcp", StaticFiles(directory="static/ui", html=True), name="mcp")
    except Exception:
        pass

    headers = kwargs.get("headers")
    if headers and isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON format for headers. Headers will be ignored.")
            headers = None

    protocol_version_header = {"MCP-Protocol-Version": MCP_VERSION}
    if server_type == "sse":
        logger.info(
            f"Configuring for a single SSE MCP Server with URL {server_command[0]}"
        )
        main_app.state.server_type = "sse"
        main_app.state.args = server_command[0]  # Expects URL as the first element
        main_app.state.api_dependency = api_dependency
        merged = dict(headers) if headers else {}
        merged["MCP-Protocol-Version"] = MCP_VERSION
        main_app.state.headers = merged
    elif server_type == "streamable-http":
        logger.info(
            f"Configuring for a single StreamableHTTP MCP Server with URL {server_command[0]}"
        )
        main_app.state.server_type = "streamable-http"
        main_app.state.args = server_command[0]  # Expects URL as the first element
        main_app.state.api_dependency = api_dependency
        merged = dict(headers) if headers else {}
        merged["MCP-Protocol-Version"] = MCP_VERSION
        main_app.state.headers = merged
    elif server_command:  # This handles stdio
        logger.info(
            f"Configuring for a single Stdio MCP Server with command: {' '.join(server_command)}"
        )
        main_app.state.server_type = "stdio"  # Explicitly set type
        main_app.state.command = server_command[0]
        main_app.state.args = server_command[1:]
        main_app.state.env = os.environ.copy()
        main_app.state.api_dependency = api_dependency
    elif config_path:
        logger.info(f"Loading MCP server configurations from: {config_path}")
        config_data = load_config(config_path)
        mount_config_servers(
            main_app, config_data, cors_allow_origins, api_key, strict_auth,
            api_dependency, connection_timeout, lifespan, path_prefix
        )

        # Store config info and app state for hot reload
        main_app.state.config_path = config_path
        main_app.state.config_data = config_data
        main_app.state.cors_allow_origins = cors_allow_origins
        main_app.state.api_key = api_key
        main_app.state.strict_auth = strict_auth
        main_app.state.api_dependency = api_dependency
        main_app.state.connection_timeout = connection_timeout
        main_app.state.lifespan = lifespan
        main_app.state.path_prefix = path_prefix
        main_app.state.tool_timeout = tool_timeout
        main_app.state.ssl_certfile = ssl_certfile
        main_app.state.ssl_keyfile = ssl_keyfile
    else:
        # Allow running without server configuration for testing/minimal mode
        logger.info("Running in minimal mode without MCP server configuration.")

    # Store SSL config in app state regardless of config source
    main_app.state.ssl_certfile = ssl_certfile
    main_app.state.ssl_keyfile = ssl_keyfile

    # Setup hot reload if enabled and config_path is provided
    config_watcher = None
    if hot_reload and config_path:
        logger.info(f"Enabling hot reload for config file: {config_path}")

        async def reload_callback(new_config):
            await reload_config_handler(main_app, new_config)

        config_watcher = ConfigWatcher(config_path, reload_callback)
        config_watcher.start()

    return main_app


async def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: Optional[str] = "",
    cors_allow_origins=["*"],
    **kwargs,
):
    """Build and run the FastAPI application."""
    main_app = await build_main_app(
        host=host,
        port=port,
        api_key=api_key,
        cors_allow_origins=cors_allow_origins,
        **kwargs,
    )

    # Get SSL config from kwargs and app state
    ssl_certfile = kwargs.get("ssl_certfile") or getattr(main_app.state, "ssl_certfile", None)
    ssl_keyfile = kwargs.get("ssl_keyfile") or getattr(main_app.state, "ssl_keyfile", None)
    
    # Get shutdown handler from app state
    shutdown_handler = getattr(main_app.state, "shutdown_handler", None)
    if not shutdown_handler:
        shutdown_handler = GracefulShutdown()

    # Get hot reload config
    hot_reload = kwargs.get("hot_reload", False)
    config_path = kwargs.get("config_path")
    config_watcher = None
    if hot_reload and config_path:
        logger.info(f"Enabling hot reload for config file: {config_path}")

        async def reload_callback(new_config):
            await reload_config_handler(main_app, new_config)

        config_watcher = ConfigWatcher(config_path, reload_callback)
        config_watcher.start()

    logger.info("Uvicorn server starting...")
    log_level = kwargs.get("log_level", "info").lower()
    config = uvicorn.Config(
        app=main_app,
        host=host,
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        log_level=log_level,
    )
    server = uvicorn.Server(config)

    # Setup signal handlers
    try:
        loop = asyncio.get_running_loop()
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig, lambda s=sig: shutdown_handler.handle_signal(s)
            )
    except NotImplementedError:
        logger.warning(
            "loop.add_signal_handler is not available on this platform. Using signal.signal()."
        )
        for sig in (signal.SIGINT, signal.SIGTERM):
            signal.signal(sig, lambda s, f: shutdown_handler.handle_signal(s))

    # Modified server startup
    try:
        # Create server task
        server_task = asyncio.create_task(server.serve())
        shutdown_handler.track_task(server_task)

        # Wait for either the server to fail or a shutdown signal
        shutdown_wait_task = asyncio.create_task(shutdown_handler.shutdown_event.wait())
        done, pending = await asyncio.wait(
            [server_task, shutdown_wait_task],
            return_when=asyncio.FIRST_COMPLETED,
        )

        if server_task in done:
            # Check if the server task raised an exception
            try:
                server_task.result()  # This will raise the exception if there was one
                logger.warning("Server task exited unexpectedly. Initiating shutdown.")
            except SystemExit as e:
                logger.error(f"Server failed to start: {e}")
                raise  # Re-raise SystemExit to maintain proper exit behavior
            except Exception as e:
                logger.error(f"Server task failed with exception: {e}")
                raise
            shutdown_handler.shutdown_event.set()

        # Cancel the other task
        for task in pending:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        # Graceful shutdown if server didn't fail with SystemExit
        logger.info("Initiating server shutdown...")
        server.should_exit = True

        # Cancel all tracked tasks
        for task in list(shutdown_handler.tasks):
            if not task.done():
                task.cancel()

        # Wait for all tasks to complete
        if shutdown_handler.tasks:
            await asyncio.gather(*shutdown_handler.tasks, return_exceptions=True)

    except SystemExit:
        # Re-raise SystemExit to allow proper program termination
        logger.info("Server startup failed, exiting...")
        raise
    except Exception as e:
        logger.error(f"Error during server execution: {e}")
        raise
    finally:
        # Stop config watcher if it was started
        if config_watcher:
            config_watcher.stop()
        logger.info("Server shutdown complete")
