import asyncio
import json
import logging
import os
import signal
import socket
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Optional, Dict, Any
from urllib.parse import urljoin

import uvicorn
from fastapi import Depends, FastAPI
from fastapi.middleware.cors import CORSMiddleware
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
from mcpo.utils.main import get_model_fields, get_tool_handler
from mcpo.utils.auth import get_verify_api_key, APIKeyMiddleware
from mcpo.utils.config_watcher import ConfigWatcher


logger = logging.getLogger(__name__)


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


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate config from file."""
    try:
        with open(config_path, "r") as f:
            config_data = json.load(f)

        mcp_servers = config_data.get("mcpServers", {})
        if not mcp_servers:
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
        allow_credentials=True,
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
    if server_config_type == "sse" and server_cfg.get("url"):
        sub_app.state.server_type = "sse"
        sub_app.state.args = [server_cfg["url"]]
        sub_app.state.headers = server_cfg.get("headers")
    elif normalize_server_type(server_config_type) == "streamable-http" and server_cfg.get("url"):
        url = server_cfg["url"]
        sub_app.state.server_type = "streamablehttp"
        sub_app.state.args = [url]
        sub_app.state.headers = server_cfg.get("headers")
    elif not server_config_type and server_cfg.get(
        "url"
    ):  # Fallback for old SSE config
        sub_app.state.server_type = "sse"
        sub_app.state.args = [server_cfg["url"]]
        sub_app.state.headers = server_cfg.get("headers")

    if api_key and strict_auth:
        sub_app.add_middleware(APIKeyMiddleware, api_key=api_key)

    sub_app.state.api_dependency = api_dependency
    sub_app.state.connection_timeout = connection_timeout

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
        main_app.mount(f"{path_prefix}{server_name}", sub_app)


def unmount_servers(main_app: FastAPI, path_prefix: str, server_names: list):
    """Unmount specific MCP servers."""
    for server_name in server_names:
        mount_path = f"{path_prefix}{server_name}"
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
    old_config_data = getattr(main_app.state, 'config_data', {})
    backup_routes = list(main_app.router.routes)  # Backup current routes for rollback

    try:
        old_servers = set(old_config_data.get("mcpServers", {}).keys())
        new_servers = set(new_config_data.get("mcpServers", {}).keys())

        # Find servers to add, remove, and potentially update
        servers_to_add = new_servers - old_servers
        servers_to_remove = old_servers - new_servers
        servers_to_check = old_servers & new_servers

        # Get app configuration from state
        cors_allow_origins = getattr(main_app.state, 'cors_allow_origins', ["*"])
        api_key = getattr(main_app.state, 'api_key', None)
        strict_auth = getattr(main_app.state, 'strict_auth', False)
        api_dependency = getattr(main_app.state, 'api_dependency', None)
        connection_timeout = getattr(main_app.state, 'connection_timeout', None)
        lifespan = getattr(main_app.state, 'lifespan', None)
        path_prefix = getattr(main_app.state, 'path_prefix', "/")

        # Remove servers that are no longer in config
        if servers_to_remove:
            logger.info(f"Removing servers: {list(servers_to_remove)}")
            unmount_servers(main_app, path_prefix, list(servers_to_remove))

        # Check for configuration changes in existing servers
        servers_to_update = []
        for server_name in servers_to_check:
            old_cfg = old_config_data["mcpServers"][server_name]
            new_cfg = new_config_data["mcpServers"][server_name]
            if old_cfg != new_cfg:
                servers_to_update.append(server_name)

        # Remove and re-add updated servers
        if servers_to_update:
            logger.info(f"Updating servers: {servers_to_update}")
            unmount_servers(main_app, path_prefix, servers_to_update)
            servers_to_add.update(servers_to_update)

        # Add new servers and updated servers
        if servers_to_add:
            logger.info(f"Adding servers: {list(servers_to_add)}")
            for server_name in servers_to_add:
                server_cfg = new_config_data["mcpServers"][server_name]
                try:
                    sub_app = create_sub_app(
                        server_name, server_cfg, cors_allow_origins, api_key,
                        strict_auth, api_dependency, connection_timeout, lifespan
                    )
                    main_app.mount(f"{path_prefix}{server_name}", sub_app)
                except Exception as e:
                    logger.error(f"Failed to create server '{server_name}': {e}")
                    # Rollback on failure
                    main_app.router.routes = backup_routes
                    raise

        # Update stored config data only after successful reload
        main_app.state.config_data = new_config_data
        logger.info("Config reload completed successfully")

    except Exception as e:
        logger.error(f"Error during config reload, keeping previous configuration: {e}")
        # Ensure we're back to the original state
        main_app.router.routes = backup_routes
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


async def run(
    host: str = "127.0.0.1",
    port: int = 8000,
    api_key: Optional[str] = "",
    cors_allow_origins=["*"],
    **kwargs,
):
    hot_reload = kwargs.get("hot_reload", False)
    # Server API Key
    api_dependency = get_verify_api_key(api_key) if api_key else None
    connection_timeout = kwargs.get("connection_timeout", None)
    strict_auth = kwargs.get("strict_auth", False)

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

    # Configure basic logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Suppress HTTP request logs
    class HTTPRequestFilter(logging.Filter):
        def filter(self, record):
            return not (
                record.levelname == "INFO" and "HTTP Request:" in record.getMessage()
            )

    # Apply filter to suppress HTTP request logs
    logging.getLogger("uvicorn.access").addFilter(HTTPRequestFilter())
    logging.getLogger("httpx.access").addFilter(HTTPRequestFilter())
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
    )

    # Pass shutdown handler to app state
    main_app.state.shutdown_handler = shutdown_handler
    main_app.state.path_prefix = path_prefix

    main_app.add_middleware(
        CORSMiddleware,
        allow_origins=cors_allow_origins or ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Add middleware to protect also documentation and spec
    if api_key and strict_auth:
        main_app.add_middleware(APIKeyMiddleware, api_key=api_key)

    headers = kwargs.get("headers")
    if headers and isinstance(headers, str):
        try:
            headers = json.loads(headers)
        except json.JSONDecodeError:
            logger.warning("Invalid JSON format for headers. Headers will be ignored.")
            headers = None

    if server_type == "sse":
        logger.info(
            f"Configuring for a single SSE MCP Server with URL {server_command[0]}"
        )
        main_app.state.server_type = "sse"
        main_app.state.args = server_command[0]  # Expects URL as the first element
        main_app.state.api_dependency = api_dependency
        main_app.state.headers = headers
    elif server_type == "streamable-http":
        logger.info(
            f"Configuring for a single StreamableHTTP MCP Server with URL {server_command[0]}"
        )
        main_app.state.server_type = "streamable-http"
        main_app.state.args = server_command[0]  # Expects URL as the first element
        main_app.state.api_dependency = api_dependency
        main_app.state.headers = headers
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
    else:
        logger.error("MCPO server_command or config_path must be provided.")
        raise ValueError("You must provide either server_command or config.")

    # Setup hot reload if enabled and config_path is provided
    config_watcher = None
    if hot_reload and config_path:
        logger.info(f"Enabling hot reload for config file: {config_path}")

        async def reload_callback(new_config):
            await reload_config_handler(main_app, new_config)

        config_watcher = ConfigWatcher(config_path, reload_callback)
        config_watcher.start()

    logger.info("Uvicorn server starting...")
    config = uvicorn.Config(
        app=main_app,
        host=host,
        port=port,
        ssl_certfile=ssl_certfile,
        ssl_keyfile=ssl_keyfile,
        log_level="info",
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
