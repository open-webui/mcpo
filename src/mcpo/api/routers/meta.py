import asyncio
import json
import logging
import os
import sys
import threading
import copy
import hashlib
from datetime import datetime
from typing import Dict, Any, Optional
from urllib.parse import urlparse

import httpx
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from starlette.routing import Mount

# Router for all /_meta endpoints
router = APIRouter()

# Import services for cleaner architecture
from mcpo.services.state import get_state_manager
from mcpo.services.logging import get_log_manager
from mcpo.services.runner import get_runner_service
from mcpo.services.metrics import get_metrics_aggregator

# Import remaining dependencies from main (to be phased out)
from mcpo.main import (
    load_config,
    reload_config_handler,
    _mount_or_remount_fastmcp,
    unmount_servers,
    create_sub_app,
    error_envelope,
    MCP_VERSION,
)

logger = logging.getLogger(__name__)

PROXY_REQUEST_TIMEOUT = 2.5


async def _get_proxy_servers(request: Request) -> list[dict[str, Any]]:
    """Fetch server list from MCP proxy and format it."""
    proxy_servers = []
    proxy_data = await _proxy_meta_request(request, "/_meta/logs/sources")
    if not proxy_data or not proxy_data.get("ok"):
        return proxy_servers

    state_manager = get_state_manager()
    sources = proxy_data.get("sources", [])
    for source in sources:
        # We only care about per-server mounts, not the global/aggregate ones
        if source.get("scope") != "server":
            continue

        server_name = source.get("server") or source.get("id")
        if not server_name:
            continue

        is_enabled = state_manager.is_server_enabled(server_name)
        proxy_servers.append({
            "name": server_name,
            "connected": False,
            "type": "internal" if server_name == "openhubui" else "proxy",
            "basePath": (source.get("path", f"/{server_name}/") or "/").rstrip("/") + "/",
            "enabled": is_enabled,
        })
    return proxy_servers


async def _proxy_meta_request(
    request: Request,
    path: str,
    *,
    method: str = "GET",
    params: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Forward a log-related request to the MCP proxy if configured."""
    base_url = getattr(request.app.state, "mcp_proxy_url", None)
    if not base_url:
        return None

    url = f"{base_url.rstrip('/')}{path}"
    try:
        timeout = httpx.Timeout(PROXY_REQUEST_TIMEOUT, connect=1.0)
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method.upper() == "POST":
                response = await client.post(url, params=params)
            else:
                response = await client.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            return data
    except Exception as exc:
        logger.debug(f"Proxy meta request failed for {url}: {exc}")
    return None


@router.get("/servers")
async def list_servers(request: Request):
    main_app = request.app
    state_manager = get_state_manager()
    servers = []
    server_names = set()
    for route in main_app.router.routes:
        if isinstance(route, Mount) and hasattr(route.app, 'state'):
            sub = route.app
            config_key = route.path.strip('/')
            
            # Include OpenHubUI admin tools, but skip other FastMCP proxy apps
            if getattr(sub.state, "is_fastmcp_proxy", False) and config_key != "openhubui":
                continue
                
            is_connected = bool(getattr(sub.state, "is_connected", False))
            server_type = getattr(sub.state, "server_type", "internal" if config_key == "openhubui" else "unknown")
            
            # Get server enabled state from state manager
            is_enabled = state_manager.is_server_enabled(config_key)
            
            servers.append({
                "name": config_key,
                "connected": is_connected,
                "type": server_type,
                "basePath": route.path.rstrip('/') + '/',
                "enabled": is_enabled,
            })
            server_names.add(config_key)

    # Fetch and merge servers from the proxy
    try:
        proxy_servers = await _get_proxy_servers(request)
        for proxy_server in proxy_servers:
            if proxy_server["name"] not in server_names:
                servers.append(proxy_server)
                server_names.add(proxy_server["name"])
    except Exception as e:
        logger.warning(f"Could not retrieve servers from MCP proxy: {e}")

    # Sort servers alphabetically
    servers.sort(key=lambda x: x["name"])

    return {"ok": True, "servers": servers}


@router.get("/servers/{server_name}/tools")
async def list_server_tools(server_name: str, request: Request):
    main_app = request.app
    state_manager = get_state_manager()

    # Standard handling for mounted MCP servers
    for route in main_app.router.routes:
        if isinstance(route, Mount) and hasattr(route.app, 'router') and route.path.rstrip('/') == f"/{server_name}":
            sub = route.app
            tools = []
            
            # Get server state for tool enabled status from state manager
            server_state = state_manager.get_server_state(server_name)
            tool_states = server_state["tools"]
            
            for r in sub.router.routes:
                methods = getattr(r, 'methods', None) or []
                if hasattr(r, 'methods') and 'POST' in methods and getattr(r, 'path', '/').startswith('/'):
                    p = r.path
                    if p == '/docs' or p.startswith('/openapi'):
                        continue
                    if p.count('/') == 1:  # '/tool'
                        tname = p.lstrip('/')
                        is_enabled = tool_states.get(tname, True)  # Default to enabled
                        tools.append({
                            "name": tname,
                            "enabled": is_enabled
                        })
            return {"ok": True, "server": server_name, "tools": sorted(tools, key=lambda x: x['name'])}
    return JSONResponse(status_code=404, content=error_envelope("Server not found", code="not_found"))


@router.get("/config")
async def config_info(request: Request):
    path = getattr(request.app.state, 'config_path', None)
    return {"ok": True, "configPath": path}


@router.get("/config/content")
async def get_config_content(request: Request):
    path = getattr(request.app.state, 'config_path', None)
    if not path:
        return JSONResponse(status_code=400, content=error_envelope("No config file", code="no_config"))
    try:
        with open(path, 'r') as f:
            content = f.read()
        return {"ok": True, "content": content}
    except Exception as e:
        return JSONResponse(status_code=500, content=error_envelope("Failed to read config", data=str(e)))


@router.post("/config/save")
async def save_config_content(payload: Dict[str, Any], request: Request):
    main_app = request.app
    path = getattr(main_app.state, 'config_path', None)
    if not path:
        return JSONResponse(status_code=400, content=error_envelope("No config-driven servers active", code="no_config"))
    content = payload.get("content")
    if content is None:
        return JSONResponse(status_code=422, content=error_envelope("Missing content", code="invalid"))
    try:
        # Validate JSON
        parsed = json.loads(content)
        # Write file
        with open(path, 'w') as f:
            f.write(json.dumps(parsed, indent=2))
        # Reload in-place
        new_config = load_config(path)
        await reload_config_handler(main_app, new_config)
        _mount_or_remount_fastmcp(main_app, base_path="/mcp")
        
        # Invalidate aggregate OpenAPI cache
        main_app.state.aggregate_openapi_dirty = True
        
        return {"ok": True, "saved": True, "reloaded": True}
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=422, content=error_envelope("Invalid JSON", data=str(e), code="invalid_json"))
    except Exception as e:
        return JSONResponse(status_code=500, content=error_envelope("Failed to save config", data=str(e)))


@router.get("/config/mcpServers")
async def get_mcp_servers_content(request: Request):
    """Return only the mcpServers section as a JSON string.

    Response shape:
    { ok: true, content: "{...}" }
    """
    path = getattr(request.app.state, 'config_path', None)
    if not path:
        return JSONResponse(status_code=400, content=error_envelope("No config file", code="no_config"))
    try:
        with open(path, 'r') as f:
            cfg = json.load(f)
        mcp_servers = cfg.get("mcpServers", {})
        content = json.dumps(mcp_servers, indent=2)
        return {"ok": True, "content": content}
    except Exception as e:
        return JSONResponse(status_code=500, content=error_envelope("Failed to read mcpServers", data=str(e)))


@router.post("/config/mcpServers/save")
async def save_mcp_servers_content(payload: Dict[str, Any], request: Request):
    """Update only the mcpServers section of the config and reload.

    Accepts either:
    - { "content": "{ ... }" } where content is a JSON string of the mcpServers object
    - { "data": { ... } } where data is the parsed mcpServers object
    """
    main_app = request.app
    path = getattr(main_app.state, 'config_path', None)
    if not path:
        return JSONResponse(status_code=400, content=error_envelope("No config-driven servers active", code="no_config"))

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
            with open(path, 'r') as f:
                full_cfg = json.load(f)
        except FileNotFoundError:
            full_cfg = {}

        # Replace mcpServers section
        full_cfg["mcpServers"] = new_mcp

        # Persist
        with open(path, 'w') as f:
            json.dump(full_cfg, f, indent=2)

        # Reload runtime and remount proxy paths
        new_config = load_config(path)
        await reload_config_handler(main_app, new_config)
        _mount_or_remount_fastmcp(main_app, base_path="/mcp")
        main_app.state.aggregate_openapi_dirty = True

        return {"ok": True, "saved": True, "reloaded": True}
    except json.JSONDecodeError as e:
        return JSONResponse(status_code=422, content=error_envelope("Invalid JSON", data=str(e), code="invalid_json"))
    except Exception as e:
        return JSONResponse(status_code=500, content=error_envelope("Failed to save mcpServers", data=str(e)))


@router.get("/logs")
async def get_logs(
    request: Request,
    source: Optional[str] = None,
    category: Optional[str] = None,
    cursor: Optional[int] = None,
    limit: int = 500,
):
    """Get recent log entries for UI display."""
    if source == "mcp":
        proxy_data = await _proxy_meta_request(
            request,
            "/_meta/logs",
            params={
                key: value
                for key, value in {
                    "category": category,
                    "cursor": cursor,
                    "limit": limit,
                }.items()
                if value is not None
            }
        )
        if proxy_data:
            return proxy_data

    log_manager = get_log_manager()
    entries = log_manager.get_logs(
        category=category,
        source=source,
        after=cursor,
        limit=limit,
    )
    latest_cursor = log_manager.get_latest_sequence()
    next_cursor = entries[-1]["sequence"] if entries else latest_cursor
    return {
        "ok": True,
        "logs": entries,
        "nextCursor": next_cursor,
        "latestCursor": latest_cursor,
        "limit": limit,
    }


@router.get("/logs/categorized")
async def get_logs_categorized(request: Request, source: Optional[str] = None):
    log_manager = get_log_manager()

    if source == "mcp":
        proxy_data = await _proxy_meta_request(request, "/_meta/logs/categorized")
        if proxy_data:
            return proxy_data

    categorized_logs = log_manager.get_logs_categorized(source=source)

    cats: Dict[str, Dict[str, Any]] = {}
    for category, logs in categorized_logs.items():
        cats[category] = {"count": len(logs), "logs": logs}
    return {"ok": True, "categories": cats}


@router.get("/logs/sources")
async def get_log_sources(request: Request):
    """Describe available log sources for the UI."""
    log_manager = get_log_manager()
    tracked_sources = set(log_manager.get_sources())

    bound_host = getattr(request.app.state, "bound_host", None)
    bound_port = getattr(request.app.state, "bound_port", None)

    sources: list[Dict[str, Any]] = [
        {
            "id": "openapi",
            "label": "OpenAPI Server",
            "host": bound_host,
            "port": bound_port,
            "url": f"http://{bound_host}:{bound_port}" if bound_host and bound_port else None,
            "available": True,
        }
    ]

    proxy_url = getattr(request.app.state, "mcp_proxy_url", None)
    if proxy_url:
        parsed = urlparse(proxy_url)
        proxy_port = parsed.port or (443 if parsed.scheme == "https" else 80)
        sources.append(
            {
                "id": "mcp",
                "label": "MCP Proxy",
                "host": parsed.hostname,
                "port": proxy_port,
                "url": proxy_url,
                "available": True,
            }
        )
    else:
        sources.append(
            {
                "id": "mcp",
                "label": "MCP Proxy",
                "available": False,
            }
        )

    for entry in sources:
        if entry["id"] in tracked_sources:
            entry["available"] = True

    return {"ok": True, "sources": sources}


@router.post("/logs/clear/{category}")
async def clear_logs_category(category: str, request: Request, source: Optional[str] = None):
    if source == "mcp":
        proxy_data = await _proxy_meta_request(
            request, f"/_meta/logs/clear/{category}", method="POST"
        )
        if proxy_data:
            return proxy_data

    log_manager = get_log_manager()
    if category in ("all", "*"):
        log_manager.clear_logs(source=source)
    else:
        log_manager.clear_logs(category, source=source)
    return {"ok": True}


@router.post("/logs/clear/all")
async def clear_logs_all(request: Request, source: Optional[str] = None):
    if source == "mcp":
        proxy_data = await _proxy_meta_request(request, "/_meta/logs/clear/all", method="POST")
        if proxy_data:
            return proxy_data

    log_manager = get_log_manager()
    log_manager.clear_logs(source=source)
    return {"ok": True}


@router.post("/reload")
async def reload_config(request: Request):
    main_app = request.app
    if getattr(main_app.state, "read_only_mode", False):
        return JSONResponse(status_code=403, content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}})
    path = getattr(main_app.state, 'config_path', None)
    if not path:
        return JSONResponse(status_code=400, content=error_envelope("No config-driven servers active", code="no_config"))
    try:
        new_config = load_config(path)
        await reload_config_handler(main_app, new_config)
        # Remount FastMCP proxy to reflect changes
        _mount_or_remount_fastmcp(main_app, base_path="/mcp")
        
        # Invalidate aggregate OpenAPI cache
        main_app.state.aggregate_openapi_dirty = True
        
        return {"ok": True, "reloaded": True}
    except Exception as e:
        return JSONResponse(status_code=500, content=error_envelope("Reload failed", data=str(e), code="reload_failed"))


@router.post("/reinit/{server_name}")
async def reinit_server(server_name: str, request: Request):
    main_app = request.app
    if getattr(main_app.state, "read_only_mode", False):
        return JSONResponse(status_code=403, content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}})
    path = getattr(main_app.state, 'config_path', None)
    if not path:
        return JSONResponse(status_code=400, content=error_envelope("Reinit is only supported for config-driven servers", code="no_config"))
    try:
        config_data = getattr(main_app.state, 'config_data', {})
        servers = config_data.get("mcpServers", {})
        if server_name not in servers:
            return JSONResponse(status_code=404, content=error_envelope("Server not found in config", code="not_found"))
        # Unmount and remount just this server
        unmount_servers(main_app, getattr(main_app.state, 'path_prefix', '/'), [server_name])
        server_cfg = servers[server_name]
        sub_app = create_sub_app(
            server_name,
            server_cfg,
            getattr(main_app.state, 'cors_allow_origins', ["*"]),
            getattr(main_app.state, 'api_key', None),
            getattr(main_app.state, 'strict_auth', False),
            getattr(main_app.state, 'api_dependency', None),
            getattr(main_app.state, 'connection_timeout', None),
            getattr(main_app.state, 'lifespan', None),
        )
        main_app.mount(f"{getattr(main_app.state, 'path_prefix', '/')}{server_name}", sub_app)
        # Remount FastMCP proxy as well in case config impacts it
        _mount_or_remount_fastmcp(main_app, base_path="/mcp")
        return {"ok": True, "reinitialized": True}
    except Exception as e:
        logger.error(f"Failed to reinit server {server_name}: {e}", exc_info=True)
        return JSONResponse(status_code=500, content=error_envelope("Reinit failed", data=str(e)))


@router.post("/install-dependencies")
async def install_dependencies():
    if not os.path.exists("requirements.txt"):
        return JSONResponse(status_code=400, content=error_envelope("requirements.txt not found", code="no_requirements"))

    async def _run_install():
        try:
            proc = await asyncio.create_subprocess_exec(
                sys.executable,
                "-m",
                "pip",
                "install",
                "-r",
                "requirements.txt",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert proc.stdout is not None
            async for line in proc.stdout:
                text = line.decode(errors="replace").rstrip()
                record = logging.LogRecord(
                    name="pip",
                    level=logging.INFO,
                    pathname=__file__,
                    lineno=0,
                    msg=text,
                    args=(),
                    exc_info=None,
                )
                logging.getLogger().handle(record)
            await proc.wait()
            logging.info("Dependency installation finished with code %s", proc.returncode)
        except Exception as e:
            logging.error(f"Dependency installation failed: {e}")

    # fire-and-forget background task
    asyncio.create_task(_run_install())
    return {"ok": True, "started": True}


@router.post("/servers/{server_name}/enable")
async def enable_server(server_name: str, request: Request):
    """Enable a server."""
    try:
        if getattr(request.app.state, "read_only_mode", False):
            return JSONResponse(status_code=403, content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}})
        state_manager = get_state_manager()
        state_manager.set_server_enabled(server_name, True)
        
        # Invalidate aggregate OpenAPI cache
        request.app.state.aggregate_openapi_dirty = True
        
        logger.info(f"Server '{server_name}' enabled")
        return JSONResponse(content={"ok": True, "enabled": True})
    except Exception as e:
        logger.error(f"Error enabling server {server_name}: {e}")
        return JSONResponse(
            status_code=500,
            content=error_envelope(f"Failed to enable server: {str(e)}")
        )


@router.post("/servers/{server_name}/disable")
async def disable_server(server_name: str, request: Request):
    """Disable a server."""
    try:
        if getattr(request.app.state, "read_only_mode", False):
            return JSONResponse(status_code=403, content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}})
        state_manager = get_state_manager()
        state_manager.set_server_enabled(server_name, False)
        
        # Invalidate aggregate OpenAPI cache
        request.app.state.aggregate_openapi_dirty = True
        
        logger.info(f"Server '{server_name}' disabled")
        return JSONResponse(content={"ok": True, "enabled": False})
    except Exception as e:
        logger.error(f"Error disabling server {server_name}: {e}")
        return JSONResponse(
            status_code=500,
            content=error_envelope(f"Failed to disable server: {str(e)}")
        )


@router.post("/servers/{server_name}/tools/{tool_name:path}/enable")
async def enable_tool(server_name: str, tool_name: str, request: Request):
    """Enable a specific tool."""
    try:
        if getattr(request.app.state, "read_only_mode", False):
            return JSONResponse(status_code=403, content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}})
        state_manager = get_state_manager()
        state_manager.set_tool_enabled(server_name, tool_name, True)
        
        # Invalidate aggregate OpenAPI cache
        request.app.state.aggregate_openapi_dirty = True
        
        logger.info(f"Tool '{tool_name}' on server '{server_name}' enabled")
        return JSONResponse(content={"ok": True, "enabled": True})
    except Exception as e:
        logger.error(f"Error enabling tool {tool_name}: {e}")
        return JSONResponse(
            status_code=500,
            content=error_envelope(f"Failed to enable tool: {str(e)}")
        )


@router.post("/servers/{server_name}/tools/{tool_name:path}/disable")
async def disable_tool(server_name: str, tool_name: str, request: Request):
    """Disable a specific tool."""
    try:
        if getattr(request.app.state, "read_only_mode", False):
            return JSONResponse(status_code=403, content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}})
        state_manager = get_state_manager()
        state_manager.set_tool_enabled(server_name, tool_name, False)
        
        # Invalidate aggregate OpenAPI cache
        request.app.state.aggregate_openapi_dirty = True
        
        logger.info(f"Tool '{tool_name}' on server '{server_name}' disabled")
        return JSONResponse(content={"ok": True, "enabled": False})
    except Exception as e:
        logger.error(f"Error disabling tool {tool_name}: {e}")
        return JSONResponse(
            status_code=500,
            content=error_envelope(f"Failed to disable tool: {str(e)}")
        )


@router.get("/status")
async def get_status(request: Request):
    """Get overall server status."""
    try:
        main_app = request.app
        servers_count = 0
        connected_count = 0
        
        for route in main_app.router.routes:
            if isinstance(route, Mount) and hasattr(route.app, 'title'):
                servers_count += 1
                if getattr(route.app.state, 'is_connected', False):
                    connected_count += 1
        
        return JSONResponse(content={
            "status": "running",
            "servers": {
                "total": servers_count,
                "connected": connected_count,
                "disconnected": servers_count - connected_count
            }
        })
    except Exception as e:
        logger.error(f"Error getting status: {e}")
        return JSONResponse(
            status_code=500,
            content=error_envelope(f"Failed to get status: {str(e)}")
        )


@router.get("/stats")
async def get_stats(request: Request):
    """Basic server stats for UI (uptime, version)."""
    try:
        main_app = request.app
        # Uptime since process start via lifespan timestamp
        start_ts = getattr(main_app.state, "start_time", None)
        if start_ts is None:
            # Initialize on first call if not set
            import time
            start_ts = time.time()
            setattr(main_app.state, "start_time", start_ts)
        import time
        uptime_s = time.time() - start_ts
        version = getattr(main_app, "version", None) or MCP_VERSION
        return JSONResponse(content={"ok": True, "uptimeSeconds": uptime_s, "version": version})
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        return JSONResponse(status_code=500, content=error_envelope("Failed to get stats", data=str(e)))

@router.get("/metrics")
async def get_metrics():
    """Expose execution metrics collected by RunnerService."""
    try:
        runner = get_runner_service()
        per_tool = runner.get_metrics()
        aggregator = get_metrics_aggregator()
        consolidated = aggregator.build_metrics(per_tool)
        return JSONResponse(content={"ok": True, "metrics": consolidated})
    except Exception as e:
        logger.error(f"Error getting metrics: {e}")
        return JSONResponse(status_code=500, content=error_envelope("Failed to get metrics", data=str(e)))

@router.get("/requirements/content")
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


@router.post("/requirements/save")
async def save_requirements_content(payload: Dict[str, Any]):
    """Save requirements.txt content."""
    content = payload.get("content")
    if content is None:
        return JSONResponse(status_code=422, content=error_envelope("Missing content", code="invalid"))
    
    try:
        with open("requirements.txt", 'w') as f:
            f.write(content)
        
        # Optionally trigger pip install here
        logger.info("Requirements.txt updated")
        return {"ok": True, "message": "Requirements saved successfully"}
        
    except Exception as e:
        return JSONResponse(status_code=500, content=error_envelope("Failed to save requirements", data=str(e), code="io_error"))


@router.get("/aggregate_openapi")
async def aggregate_openapi(request: Request, force_refresh: bool = False):
    """
    Provide a unified OpenAPI 3.1 specification combining all enabled MCP servers.
    
    This endpoint aggregates OpenAPI specifications from all connected and enabled
    MCP servers into a single comprehensive API definition. Features include:
    - Smart caching with automatic invalidation
    - Schema collision detection and resolution
    - Reference path rewriting for proper component linking
    - Real-time filtering based on server and tool enable/disable states
    """
    main_app = request.app
    state_manager = get_state_manager()
    
    # Check cache unless force refresh requested
    cache_key = "aggregate_openapi_cache"
    if not force_refresh and hasattr(main_app.state, cache_key):
        cache = getattr(main_app.state, cache_key)
        if cache and not getattr(main_app.state, "aggregate_openapi_dirty", False):
            logger.debug("Returning cached aggregate OpenAPI specification")
            return JSONResponse(content=cache['schema'])
    
    logger.info("Building aggregate OpenAPI specification...")
    
    # Base OpenAPI 3.1 structure
    base_spec = {
        "openapi": "3.1.0",
        "info": {
            "title": "MCPO Aggregated API",
            "version": "1.0.0",
            "description": "Unified API combining all enabled MCP servers"
        },
        "paths": {},
        "components": {
            "schemas": {},
            "securitySchemes": {},
            "responses": {},
            "parameters": {},
            "examples": {},
            "requestBodies": {},
            "headers": {},
            "links": {},
            "callbacks": {}
        },
        "tags": []
    }
    
    component_mapping = {}  # Track renamed components: key f"{server}#{name}" -> new_name
    server_tags = set()
    
    # Process each mounted server
    for route in main_app.router.routes:
        if not isinstance(route, Mount) or not hasattr(route.app, 'state'):
            continue
            
        sub_app = route.app
        server_name = route.path.strip('/')
        
        # Skip FastMCP proxy apps and disabled servers
        if getattr(sub_app.state, "is_fastmcp_proxy", False):
            continue
            
        if not state_manager.is_server_enabled(server_name):
            continue
            
        # Skip if not connected
        if not getattr(sub_app.state, "is_connected", False):
            continue
            
        try:
            # Get server's OpenAPI spec
            server_spec = sub_app.openapi()
            if not server_spec or "paths" not in server_spec:
                continue
            
            mount_path = route.path.rstrip('/')
            server_title = server_spec.get("info", {}).get("title", server_name)
            server_tags.add(server_title)
            
            # First merge components with collision handling to build mapping
            server_components = server_spec.get("components", {})
            for component_type, components in server_components.items():
                if component_type not in base_spec["components"]:
                    base_spec["components"][component_type] = {}
                    
                for comp_name, comp_spec in components.items():
                    # Check for naming collision
                    if comp_name in base_spec["components"][component_type]:
                        existing_spec = base_spec["components"][component_type][comp_name]
                        
                        # Compare specs (simple content hash comparison)
                        existing_hash = hashlib.md5(json.dumps(existing_spec, sort_keys=True).encode()).hexdigest()
                        new_hash = hashlib.md5(json.dumps(comp_spec, sort_keys=True).encode()).hexdigest()
                        
                        if existing_hash != new_hash:
                            # Collision detected - rename with server prefix
                            new_name = f"{comp_name}_{server_name}"
                            base_spec["components"][component_type][new_name] = comp_spec
                            component_mapping[f"{server_name}#{comp_name}"] = new_name
                            logger.debug(f"Component collision resolved: {comp_name} -> {new_name}")
                        else:
                            # Same content, can reuse existing component
                            component_mapping[f"{server_name}#{comp_name}"] = comp_name
                    else:
                        # No collision, add directly
                        base_spec["components"][component_type][comp_name] = comp_spec
                        component_mapping[f"{server_name}#{comp_name}"] = comp_name
            
            # Helper to rewrite $ref values within an object tree for this server
            def _rewrite_refs(obj: Any) -> Any:
                if isinstance(obj, dict):
                    if "$ref" in obj and isinstance(obj["$ref"], str):
                        ref = obj["$ref"]
                        if ref.startswith("#/components/"):
                            parts = ref.split('/')
                            # Expect: ["#", "components", type, name, ...]
                            if len(parts) >= 4:
                                comp_name = parts[3]
                                key = f"{server_name}#{comp_name}"
                                if key in component_mapping:
                                    parts[3] = component_mapping[key]
                                    obj["$ref"] = '/'.join(parts)
                    for k, v in list(obj.items()):
                        obj[k] = _rewrite_refs(v)
                    return obj
                elif isinstance(obj, list):
                    return [_rewrite_refs(v) for v in obj]
                else:
                    return obj
            
            # Rewrite refs inside newly added components for this server
            for component_type, components in server_components.items():
                for comp_name in components.keys():
                    new_name = component_mapping.get(f"{server_name}#{comp_name}", comp_name)
                    comp_obj = base_spec["components"][component_type].get(new_name)
                    if comp_obj:
                        base_spec["components"][component_type][new_name] = _rewrite_refs(comp_obj)
            
            # Now filter and merge paths with $ref rewriting
            server_state = state_manager.get_server_state(server_name)
            tool_states = server_state["tools"]
            
            for path, path_spec in server_spec.get("paths", {}).items():
                # Skip docs and openapi paths
                if path in ['/docs', '/openapi.json'] or path.startswith('/openapi'):
                    continue
                
                # Extract tool name from path (e.g., '/tool_name' -> 'tool_name')
                if path.startswith('/') and path.count('/') == 1:
                    tool_name = path.lstrip('/')
                    if not tool_states.get(tool_name, True):  # Skip disabled tools
                        continue
                
                # Merge path with server prefix
                merged_path = mount_path + path if path != '/' else mount_path
                if merged_path in base_spec["paths"]:
                    logger.warning(f"Path collision detected: {merged_path}")
                    merged_path = f"{mount_path}_{server_name}{path}"
                
                # Add server tag to all operations and rewrite $refs
                if isinstance(path_spec, dict):
                    path_spec = _rewrite_refs(path_spec)
                    for method, operation in path_spec.items():
                        if isinstance(operation, dict):
                            if "tags" not in operation:
                                operation["tags"] = []
                            if server_title not in operation["tags"]:
                                operation["tags"].append(server_title)
                
                base_spec["paths"][merged_path] = path_spec
                        
        except Exception as e:
            logger.error(f"Failed to process server {server_name} for aggregation: {e}")
            continue
    
    # Add server tags
    for tag_name in sorted(server_tags):
        base_spec["tags"].append({
            "name": tag_name,
            "description": f"Tools from {tag_name} server"
        })
    
    # All $refs for each server's segments were rewritten during merge
    
    # Cache the result
    cache_data = {
        'schema': base_spec,
        'built_at': datetime.now().isoformat(),
        'hash': hashlib.md5(json.dumps(base_spec, sort_keys=True).encode()).hexdigest()
    }
    setattr(main_app.state, cache_key, cache_data)
    setattr(main_app.state, "aggregate_openapi_dirty", False)
    
    logger.info(f"Aggregate OpenAPI built with {len(base_spec['paths'])} paths from {len(server_tags)} servers")
    return JSONResponse(content=base_spec)
