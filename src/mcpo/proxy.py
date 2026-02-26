import argparse
import asyncio
import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any, Optional, AsyncGenerator

import uvicorn
from fastapi import APIRouter, FastAPI, Request
from fastapi.responses import RedirectResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

# Prefer vendored FastMCP in workspace if available
try:
    from fastmcp.server.server import FastMCP
    from fastmcp.mcp_config import MCPConfig
except Exception as e:
    raise

from mcpo.services.logging import get_log_manager
from mcpo.services.logging_handlers import BufferedLogHandler
from mcpo.services.state import get_state_manager
from mcpo.middleware.mcp_tool_filter import MCPToolFilterMiddleware
from mcpo.middleware.code_mode import CodeModeMCPMiddleware

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 8001
DEFAULT_PATH = "/global"
PROXY_MAX_LOG_ENTRIES = 2000

logger = logging.getLogger(__name__)

_proxy_log_buffer: list[Dict[str, Any]] = []
_proxy_log_lock = threading.Lock()


def _ensure_log_manager_capacity() -> None:
    """Initialize the shared log manager with an adequate buffer."""
    manager = get_log_manager(PROXY_MAX_LOG_ENTRIES)
    manager.max_entries = max(manager.max_entries, PROXY_MAX_LOG_ENTRIES)


def _build_proxy_log_router() -> APIRouter:
    router = APIRouter()

    @router.get("/_meta/logs")
    async def get_proxy_logs(
        category: Optional[str] = None,
        source: Optional[str] = None,
        cursor: Optional[int] = None,
        limit: int = 500,
    ):
        log_manager = get_log_manager(PROXY_MAX_LOG_ENTRIES)
        log_source = source or "mcp"
        effective_limit = max(1, min(limit, PROXY_MAX_LOG_ENTRIES))
        logs = log_manager.get_logs(
            category=category,
            source=log_source,
            after=cursor,
            limit=effective_limit,
        )
        latest_cursor = log_manager.get_latest_sequence()
        next_cursor = logs[-1]["sequence"] if logs else latest_cursor
        return {
            "ok": True,
            "logs": logs,
            "nextCursor": next_cursor,
            "latestCursor": latest_cursor,
            "limit": effective_limit,
        }

    @router.get("/_meta/logs/categorized")
    async def get_proxy_logs_categorized(source: Optional[str] = None):
        log_manager = get_log_manager(PROXY_MAX_LOG_ENTRIES)
        categorized = log_manager.get_logs_categorized(source=source or "mcp")
        cats: Dict[str, Dict[str, Any]] = {}
        for category, entries in categorized.items():
            cats[category] = {"count": len(entries), "logs": entries}
        return {"ok": True, "categories": cats}

    @router.post("/_meta/logs/clear/{category}")
    async def clear_proxy_logs(category: str):
        log_manager = get_log_manager(PROXY_MAX_LOG_ENTRIES)
        if category in ("all", "*"):
            log_manager.clear_logs(source="mcp")
        else:
            log_manager.clear_logs(category, source="mcp")
        return {"ok": True}

    @router.post("/_meta/logs/clear/all")
    async def clear_proxy_logs_all():
        log_manager = get_log_manager(PROXY_MAX_LOG_ENTRIES)
        log_manager.clear_logs(source="mcp")
        return {"ok": True}

    @router.get("/_meta/logs/sources")
    async def list_proxy_sources(request: Request):
        host = getattr(request.app.state, "bound_host", DEFAULT_HOST)
        port = getattr(request.app.state, "bound_port", DEFAULT_PORT)
        mounts = getattr(request.app.state, "proxy_mounts", [])
        sources: list[dict[str, Any]] = []
        for mount in mounts:
            mount_path = (mount.get("path") or "/")
            if mount_path != "/" and not mount_path.startswith("/"):
                mount_path = f"/{mount_path.lstrip('/')}"
            url_suffix = "" if mount_path in ("", "/") else mount_path
            label = mount.get("label") or mount.get("id") or (mount_path.lstrip("/") or "MCP Proxy")
            source_entry = {
                "id": mount.get("id") or mount_path.lstrip("/") or "root",
                "label": label,
                "host": host,
                "port": port,
                "url": f"http://{host}:{port}{url_suffix}",
                "available": True,
                "path": mount_path,
            }
            if "scope" in mount:
                source_entry["scope"] = mount["scope"]
            if "server" in mount:
                source_entry["server"] = mount["server"]
            sources.append(source_entry)
        if not sources:
            sources.append(
                {
                    "id": "mcp",
                    "label": "MCP Proxy",
                    "host": host,
                    "port": port,
                    "url": f"http://{host}:{port}",
                    "available": True,
                    "path": "/",
                }
            )
        return {"ok": True, "sources": sources}

    return router



def _create_fastmcp_lifespan(fastmcp_apps: list[tuple[FastAPI, str]]):
    """
    Create a lifespan context manager for FastMCP sub-app lifespans.
    
    Replaces the deprecated @app.on_event pattern with modern lifespan approach.
    """
    from contextlib import AsyncExitStack

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup: initialize all FastMCP sub-app lifespans
        stack = AsyncExitStack()
        async with stack:
            for sub_app, label in fastmcp_apps:
                try:
                    await stack.enter_async_context(sub_app.router.lifespan_context(sub_app))
                    logger.debug("Started FastMCP lifespan for %s", label)
                except Exception:
                    logger.exception("Failed to start FastMCP lifespan for %s", label)
                    raise
            
            yield  # Application runs here
            
            # Shutdown handled automatically by AsyncExitStack.__aexit__
            logger.debug("FastMCP lifespans shutting down")
    
    return lifespan


# OpenHubUI MCP server removed - now using OpenAPI endpoints only
def _interpolate_env_placeholders(config_obj: dict) -> dict:
    """Replace ${VAR} in server.env values with os.environ[VAR]."""
    import re
    pattern = re.compile(r"\$\{([A-Z0-9_]+)\}")

    servers = (config_obj or {}).get("mcpServers", {})
    for name, server in servers.items():
        env = server.get("env") if isinstance(server, dict) else None
        if isinstance(env, dict):
            for k, v in list(env.items()):
                if isinstance(v, str):
                    def repl(m):
                        var = m.group(1)
                        return os.environ.get(var, "")
                    new_v = pattern.sub(repl, v)
                    env[k] = new_v
    return config_obj


async def run_proxy(
    *,
    config_path: Path,
    host: str = DEFAULT_HOST,
    port: int = DEFAULT_PORT,
    path: str = DEFAULT_PATH,
    log_level: str = "info",
    stateless_http: Optional[bool] = None,
) -> None:
    """Start the FastMCP Streamable HTTP proxy with UI log capture enabled."""
    logging.basicConfig(level=getattr(logging, log_level.upper(), logging.INFO))

    _ensure_log_manager_capacity()
    root_logger = logging.getLogger()
    buffered_handler = BufferedLogHandler(
        _proxy_log_buffer,
        _proxy_log_lock,
        default_source="mcp",
        max_entries=PROXY_MAX_LOG_ENTRIES,
    )
    if not any(isinstance(h, BufferedLogHandler) for h in root_logger.handlers):
        root_logger.addHandler(buffered_handler)
    # Capture uvicorn-specific loggers that bypass the root logger so UI logs include proxy requests.
    for logger_name in ("uvicorn.access", "uvicorn.error"):
        logger_obj = logging.getLogger(logger_name)
        if not any(isinstance(h, BufferedLogHandler) for h in logger_obj.handlers):
            logger_obj.addHandler(buffered_handler)
    root_logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    with config_path.open("r", encoding="utf-8") as f:
        raw_cfg = json.load(f)
    raw_cfg = _interpolate_env_placeholders(raw_cfg)

    state_manager = get_state_manager()
    servers_cfg = raw_cfg.get("mcpServers", {}) if isinstance(raw_cfg, dict) else {}
    filtered_servers: Dict[str, Any] = {}
    for server_name, server_cfg in servers_cfg.items():
        if isinstance(server_cfg, dict) and not server_cfg.get("enabled", True):
            logger.info("Skipping FastMCP server '%s' (disabled in config)", server_name)
            continue
        if not state_manager.is_server_enabled(server_name):
            logger.info("Skipping FastMCP server '%s' (disabled in shared state)", server_name)
            continue
        filtered_servers[server_name] = server_cfg

    if isinstance(raw_cfg, dict):
        raw_cfg = dict(raw_cfg)
        raw_cfg["mcpServers"] = filtered_servers

    if servers_cfg and not filtered_servers:
        logger.warning("All MCP servers are disabled; FastMCP proxy will expose only metadata endpoints")

    cfg = MCPConfig.from_dict(raw_cfg)
    proxy = FastMCP.as_proxy(cfg)

    # Prepare shared list for FastMCP sub-apps (populated during mount loop)
    fastmcp_apps: list[tuple[FastAPI, str]] = []
    
    # Create lifespan context manager that references fastmcp_apps
    lifespan = _create_fastmcp_lifespan(fastmcp_apps)
    
    api_app = FastAPI(lifespan=lifespan)
    # Allow cross-origin requests for browser-based clients (e.g., OpenWebUI)
    api_app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    api_app.state.bound_host = host
    api_app.state.bound_port = port

    # Determine OpenAPI admin UI host:port from env or default
    _openapi_host = os.environ.get("MCPO_OPENAPI_HOST", "127.0.0.1")
    _openapi_port = os.environ.get("MCPO_OPENAPI_PORT", "8000")
    _openapi_base = f"http://{_openapi_host}:{_openapi_port}"

    # Friendly redirects so opening the proxy port in a browser isn't blank
    @api_app.get("/", response_class=HTMLResponse)
    async def _proxy_root_landing():
        try:
            # Redirect to the Admin UI if it's running on the common default port
            target = f"{_openapi_base}/ui/"
            return RedirectResponse(url=target, status_code=307)
        except Exception:
            # As a last resort, show a tiny hint page
            return HTMLResponse(
                content=(
                    "<html><body style='font-family:system-ui, sans-serif; padding:16px;'>"
                    "<h3>MCP Proxy</h3>"
                    "<p>This port serves the MCP Streamable HTTP proxy. The UI is available at "
                    f"<a href='{_openapi_base}/ui/'>{_openapi_base}/ui/</a>."
                    "</p>"
                    "</body></html>"
                ),
                status_code=200,
                media_type="text/html",
            )

    @api_app.get("/ui")
    async def _proxy_ui_redirect():
        return RedirectResponse(url=f"{_openapi_base}/ui/", status_code=307)

    proxy_mounts: list[dict[str, Any]] = []



    global_mount_path = path or "/global"
    if not global_mount_path.startswith("/"):
        global_mount_path = f"/{global_mount_path}"
    global_mount_path = global_mount_path.rstrip("/") or "/"

    global_app = proxy.http_app(path="/", transport="streamable-http", stateless_http=stateless_http)

    # Add tool filtering middleware for aggregate proxy
    # Pass None as server_name to enable multi-server filtering
    global_app.add_middleware(MCPToolFilterMiddleware, server_name=None)
    logger.info(f"Added tool filtering middleware for aggregate proxy covering {len(filtered_servers)} servers")

    # Add code mode middleware (wraps tool filter; active only when code mode is on)
    global_app.add_middleware(CodeModeMCPMiddleware, server_name=None)
    logger.info("Added code mode middleware for aggregate proxy")
    
    fastmcp_apps.append((global_app, f'global mount {global_mount_path}'))
    setattr(global_app.state, "is_fastmcp_proxy", True)
    setattr(global_app.state, "proxy_scope", "global")
    api_app.mount(global_mount_path, global_app)
    api_app.state.proxy_global_mount = global_mount_path
    api_app.state.proxy_stateless = stateless_http
    logger.info("Mounted aggregate FastMCP proxy at %s", global_mount_path)
    proxy_mounts.append(
        {
            "id": "global" if global_mount_path != "/" else "root",
            "label": "All MCPs",
            "path": global_mount_path,
            "scope": "global",
        }
    )

    # Back-compat alias: also mount at /mcp if different from chosen global path
    if global_mount_path != "/mcp":
        alias_path = "/mcp"
        try:
            api_app.mount(alias_path, global_app)
            logger.info("Mounted compatibility alias for FastMCP proxy at %s", alias_path)
            proxy_mounts.append(
                {
                    "id": "mcp",
                    "label": "All MCPs (/mcp)",
                    "path": alias_path,
                    "scope": "global",
                }
            )
        except Exception:
            logger.warning("Failed to mount /mcp alias; continuing without it", exc_info=True)

    servers = raw_cfg.get("mcpServers", {}) if isinstance(raw_cfg, dict) else {}
    for server_name, server_cfg in servers.items():
        mount_segment = str(server_name).strip().strip("/")
        if not mount_segment:
            logger.warning("Skipping MCP server with blank name in config")
            continue

        single_cfg_dict = {"mcpServers": {server_name: server_cfg}}
        try:
            single_cfg = MCPConfig.from_dict(single_cfg_dict)
            single_proxy = FastMCP.as_proxy(single_cfg)
        except Exception as exc:
            logger.error(
                "Failed to initialize FastMCP proxy for server '%s': %s",
                server_name,
                exc,
                exc_info=True,
            )
            continue

        sub_app = single_proxy.http_app(path="/", transport="streamable-http", stateless_http=stateless_http)

        # Wrap with tool filtering middleware to respect mcpo_state.json
        sub_app.add_middleware(MCPToolFilterMiddleware, server_name=server_name)
        logger.info(f"Added tool filtering middleware for server '{server_name}'")

        # Add code mode middleware for per-server proxy
        sub_app.add_middleware(CodeModeMCPMiddleware, server_name=server_name)
        logger.info(f"Added code mode middleware for server '{server_name}'")

        # Provide a single mount point for server-specific proxies at /{server}
        root_mount_path = f"/{mount_segment}".replace("//", "/").rstrip("/") or "/"
        fastmcp_apps.append((sub_app, f"server {server_name} ({root_mount_path})"))
        setattr(sub_app.state, "is_fastmcp_proxy", True)
        setattr(sub_app.state, "proxy_scope", "server")
        setattr(sub_app.state, "proxy_server_name", server_name)

        if not any(existing.get("path") == root_mount_path for existing in proxy_mounts):
            api_app.mount(root_mount_path, sub_app)
            logger.info("Mounted FastMCP proxy for server '%s' at %s", server_name, root_mount_path)
            proxy_mounts.append(
                {
                    "id": mount_segment,
                    "label": label if (label := (server_cfg.get("name") if isinstance(server_cfg, dict) else None) or server_name) else server_name,
                    "path": root_mount_path,
                    "scope": "server",
                    "server": server_name,
                }
            )
        else:
            logger.warning("Mount path %s already in use; skipping for '%s'", root_mount_path, server_name)

    api_app.state.proxy_mounts = proxy_mounts
    # Note: FastMCP lifespan handlers are now managed via the lifespan context manager
    # passed to FastAPI() at construction time - no separate registration needed
    api_app.include_router(_build_proxy_log_router())

    config = uvicorn.Config(
        api_app,
        host=host,
        port=port,
        log_level=log_level,
        timeout_graceful_shutdown=0,
        lifespan="on",
    )
    server = uvicorn.Server(config)
    await server.serve()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Start FastMCP Streamable HTTP proxy (8001) from mcpo.json")
    p.add_argument("--config", "-c", type=str, default="mcpo.json", help="Path to MCP config (mcpo.json)")
    p.add_argument("--host", "-H", type=str, default=DEFAULT_HOST, help="Bind host")
    p.add_argument("--port", "-p", type=int, default=DEFAULT_PORT, help="Bind port")
    p.add_argument("--path", type=str, default=DEFAULT_PATH, help="Mount path for the aggregate Streamable HTTP endpoint")
    p.add_argument("--log-level", "-l", type=str, default="info", help="Log level (debug, info, warning, error)")
    p.add_argument("--stateless-http", action="store_true", help="Force stateless HTTP mode")
    p.add_argument("--env", "-e", action="append", help="Extra env KEY=VALUE to set before start")
    p.add_argument("--env-path", type=str, help=".env file to load before start")
    return p.parse_args()


def apply_env(env_items: Optional[list[str]], env_path: Optional[str]) -> None:
    if env_path and Path(env_path).exists():
        try:
            from dotenv import load_dotenv
            load_dotenv(env_path)
        except Exception:
            pass
    if env_items:
        for item in env_items:
            if "=" in item:
                k, v = item.split("=", 1)
                os.environ[k] = v


def main() -> None:
    args = parse_args()
    apply_env(args.env, args.env_path)

    config_path = Path(args.config).expanduser()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    asyncio.run(
        run_proxy(
            config_path=config_path,
            host=args.host,
            port=args.port,
            path=args.path,
            log_level=args.log_level,
            stateless_http=args.stateless_http or None,
        )
    )


if __name__ == "__main__":
    main()


