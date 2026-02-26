"""
Admin router for server management and configuration operations.
"""
import logging
import re
from typing import Dict, Any
from fastapi import FastAPI, APIRouter, Depends, Request, HTTPException
from fastapi.responses import JSONResponse
import os
from pydantic import BaseModel, Field

from mcpo.services.state import get_state_manager
from mcpo.services.skills import (
    compile_skills_system_prompt,
    delete_skill_file,
    get_skill,
    list_skills,
    upsert_skill_file,
)

logger = logging.getLogger(__name__)

router = APIRouter()


def _admin_error_envelope(message: str, code: str = "error") -> dict:
    return {"ok": False, "error": {"message": message, "code": code}}


class SkillUpsertRequest(BaseModel):
    id: str = Field(..., description="Stable skill ID")
    title: str = Field(..., description="Display name")
    description: str = Field("", description="Short description")
    content: str = Field(..., description="Skill markdown body")

# Initialize global variable first
_FASTMCP_AVAILABLE = None

def _mount_or_remount_fastmcp(main_app: FastAPI, base_path: str = "/mcp") -> None:
    """Mount or remount FastMCP proxy http_app(s) at base_path if FastMCP is available."""
    global _FASTMCP_AVAILABLE
    try:
        # Remove any existing FastMCP proxy mounts first
        removed_count = 0
        for route in list(main_app.router.routes):
            route_app = getattr(route, "app", None)
            if route_app and hasattr(route_app, "state") and getattr(route_app.state, "is_fastmcp_proxy", False):
                main_app.router.routes.remove(route)
                removed_count += 1
                logger.info("Removed FastMCP proxy mount at %s", getattr(route, "path", "<unknown>"))

        # Determine availability lazily
        if _FASTMCP_AVAILABLE is None:
            try:
                from fastmcp.server.server import FastMCP  # type: ignore
                from fastmcp.mcp_config import MCPConfig  # type: ignore
                _FASTMCP_AVAILABLE = True
            except Exception:
                _FASTMCP_AVAILABLE = False

        if not _FASTMCP_AVAILABLE:
            logger.info("FastMCP not available; skipping proxy mount")
            main_app.state.fastmcp_proxy_mounts = []
            main_app.state.fastmcp_proxy_global_mount = None
            return

        # Retrieve config
        config_data = getattr(main_app.state, "config_data", None)
        if not isinstance(config_data, dict):
            logger.info("No config_data present; skipping FastMCP proxy mount")
            main_app.state.fastmcp_proxy_mounts = []
            main_app.state.fastmcp_proxy_global_mount = None
            return

        from mcpo.utils.config import interpolate_env_placeholders_in_config
        raw_cfg = interpolate_env_placeholders_in_config(config_data)

        from fastmcp.server.server import FastMCP  # type: ignore
        from fastmcp.mcp_config import MCPConfig  # type: ignore

        cfg = MCPConfig.from_dict(raw_cfg)
        proxy = FastMCP.as_proxy(cfg)

        def _compose_path(*segments: str) -> str:
            parts = []
            for segment in segments:
                if not segment:
                    continue
                stripped = segment.strip("/")
                if stripped:
                    parts.append(stripped)
            if not parts:
                return "/"
            return "/" + "/".join(parts)

        path_prefix = getattr(main_app.state, "path_prefix", "/") or "/"
        api_key = getattr(main_app.state, "api_key", None)
        strict_auth = getattr(main_app.state, "strict_auth", False)
        api_key_middleware_cls = None
        if api_key and strict_auth:
            from mcpo.utils.auth import APIKeyMiddleware  # local import to avoid circular deps

            api_key_middleware_cls = APIKeyMiddleware

        def _secure_app(app_to_secure):
            if api_key_middleware_cls is None:
                return app_to_secure
            secure_wrapper = FastAPI()
            secure_wrapper.add_middleware(api_key_middleware_cls, api_key=api_key)
            secure_wrapper.mount("/", app_to_secure)
            setattr(secure_wrapper.state, "is_fastmcp_proxy", True)
            setattr(secure_wrapper.state, "proxy_scope", getattr(app_to_secure.state, "proxy_scope", None))
            if hasattr(app_to_secure.state, "proxy_server_name"):
                setattr(
                    secure_wrapper.state,
                    "proxy_server_name",
                    getattr(app_to_secure.state, "proxy_server_name"),
                )
            return secure_wrapper

        proxy_mounts: list[dict[str, Any]] = []
        used_paths: set[str] = set()

        from mcpo.middleware.code_mode import CodeModeMCPMiddleware

        # Aggregate (global) proxy mount
        global_app = proxy.http_app(path="/", transport="streamable-http", stateless_http=None)
        global_app.add_middleware(CodeModeMCPMiddleware, server_name=None)
        setattr(global_app.state, "is_fastmcp_proxy", True)
        setattr(global_app.state, "proxy_scope", "global")
        setattr(global_app.state, "proxy_mount_base", base_path)

        global_mount_path = _compose_path(path_prefix, base_path)
        used_paths.add(global_mount_path)
        wrapped_global = _secure_app(global_app)
        main_app.mount(global_mount_path, wrapped_global)
        logger.info("Mounted aggregate FastMCP proxy at %s", global_mount_path)

        proxy_mounts.append(
            {
                "id": "global" if global_mount_path != "/" else "root",
                "label": "All MCPs",
                "path": global_mount_path,
                "scope": "global",
            }
        )

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

            sub_app = single_proxy.http_app(path="/", transport="streamable-http", stateless_http=None)
            sub_app.add_middleware(CodeModeMCPMiddleware, server_name=server_name)
            setattr(sub_app.state, "is_fastmcp_proxy", True)
            setattr(sub_app.state, "proxy_scope", "server")
            setattr(sub_app.state, "proxy_server_name", server_name)

            mount_path = _compose_path(path_prefix, base_path, mount_segment)
            if mount_path in used_paths:
                logger.warning(
                    "Skipping MCP server '%s' because mount path %s is already in use",
                    server_name,
                    mount_path,
                )
                continue

            used_paths.add(mount_path)
            wrapped_sub_app = _secure_app(sub_app)
            main_app.mount(mount_path, wrapped_sub_app)
            logger.info("Mounted FastMCP proxy for server '%s' at %s", server_name, mount_path)

            label = server_name
            if isinstance(server_cfg, dict):
                label = server_cfg.get("name") or server_cfg.get("displayName") or server_name

            proxy_mounts.append(
                {
                    "id": mount_segment,
                    "label": label,
                    "path": mount_path,
                    "scope": "server",
                    "server": server_name,
                }
            )

        main_app.state.fastmcp_proxy_mounts = proxy_mounts
        main_app.state.fastmcp_proxy_global_mount = global_mount_path
    except Exception as e:
        logger.error(f"Failed to mount FastMCP proxy: {e}", exc_info=True)



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


# Global variable already initialized above

@router.post("/env")
def update_env_vars(data: Dict[str, str], request: Request):
    """Update environment variables in local .env file. Minimal implementation.
    This endpoint is intended for local/dev use; secure or disable in production.
    """
    import re as _re
    # Block in read-only mode
    if getattr(request.app.state, 'read_only_mode', False):
        raise HTTPException(status_code=403, detail="Read-only mode")
    
    # Deny dangerous keys
    DENIED_KEYS = {"PATH", "HOME", "USER", "SHELL", "PYTHONPATH", "LD_PRELOAD", "LD_LIBRARY_PATH"}
    
    try:
        env_path = ".env"
        existing: Dict[str, str] = {}
        if os.path.exists(env_path):
            with open(env_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#') or '=' not in line:
                        continue
                    k, v = line.split('=', 1)
                    existing[k] = v
        # update
        for k, v in data.items():
            if not isinstance(k, str):
                continue
            # Validate key format
            if not _re.match(r'^[A-Za-z_][A-Za-z0-9_]*$', k):
                raise HTTPException(status_code=422, detail=f"Invalid env var key: {k}")
            if k.upper() in DENIED_KEYS:
                raise HTTPException(status_code=403, detail=f"Cannot modify protected key: {k}")
            existing[k] = str(v)
            # also update process env for this process
            os.environ[k] = str(v)
        # write back
        with open(env_path, 'w', encoding='utf-8') as f:
            for k, v in existing.items():
                f.write(f"{k}={v}\n")
        return {"status": "updated", "count": len(data)}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/skills")
async def list_agent_skills():
    skills = list_skills()
    return {
        "ok": True,
        "skills": [
            {
                "id": item.id,
                "title": item.title,
                "description": item.description,
                "content": item.content,
                "enabled": item.enabled,
                "priority": item.priority,
                "scopes": item.scopes or [],
                "providers": item.providers or [],
                "models": item.models or [],
                "tags": item.tags or [],
                "sourcePath": item.source_path,
            }
            for item in skills
        ],
    }


@router.get("/skills/{skill_id}")
async def get_agent_skill(skill_id: str):
    skill = get_skill(skill_id)
    if not skill:
        return JSONResponse(
            status_code=404,
            content=_admin_error_envelope("Skill not found", code="not_found"),
        )
    return {
        "ok": True,
        "skill": {
            "id": skill.id,
            "title": skill.title,
            "description": skill.description,
            "content": skill.content,
            "enabled": skill.enabled,
            "priority": skill.priority,
            "scopes": skill.scopes or [],
            "providers": skill.providers or [],
            "models": skill.models or [],
            "tags": skill.tags or [],
            "sourcePath": skill.source_path,
        },
    }


@router.post("/skills")
async def upsert_agent_skill(payload: SkillUpsertRequest, request: Request):
    if getattr(request.app.state, "read_only_mode", False):
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}},
        )
    try:
        skill = upsert_skill_file(
            skill_id=payload.id,
            title=payload.title,
            description=payload.description,
            content=payload.content,
        )
        return {
            "ok": True,
            "skill": {
                "id": skill.id,
                "title": skill.title,
                "description": skill.description,
                "enabled": skill.enabled,
                "priority": skill.priority,
            },
        }
    except Exception as exc:
        return JSONResponse(
            status_code=500,
            content=_admin_error_envelope(f"Failed to save skill: {exc}", code="save_failed"),
        )


@router.delete("/skills/{skill_id}")
async def delete_agent_skill(skill_id: str, request: Request):
    if getattr(request.app.state, "read_only_mode", False):
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}},
        )
    deleted = delete_skill_file(skill_id)
    if not deleted:
        return JSONResponse(
            status_code=404,
            content=_admin_error_envelope("Skill not found", code="not_found"),
        )
    return {"ok": True, "deleted": True}


@router.post("/skills/{skill_id}/enable")
async def enable_agent_skill(skill_id: str, request: Request):
    if getattr(request.app.state, "read_only_mode", False):
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}},
        )
    state = get_state_manager()
    state.set_skill_enabled(skill_id, True)
    return {"ok": True, "enabled": True}


@router.post("/skills/{skill_id}/disable")
async def disable_agent_skill(skill_id: str, request: Request):
    if getattr(request.app.state, "read_only_mode", False):
        return JSONResponse(
            status_code=403,
            content={"ok": False, "error": {"message": "Read-only mode", "code": "read_only"}},
        )
    state = get_state_manager()
    state.set_skill_enabled(skill_id, False)
    return {"ok": True, "enabled": False}


@router.post("/skills/preview-prompt")
async def preview_skill_prompt(payload: Dict[str, Any]):
    scope = str(payload.get("scope") or "chat")
    model = payload.get("model")
    provider = payload.get("provider")
    skill_ids = payload.get("skill_ids")
    prompt = compile_skills_system_prompt(
        scope=scope,
        model=model,
        provider=provider,
        requested_skill_ids=skill_ids if isinstance(skill_ids, list) else None,
    )
    return {"ok": True, "prompt": prompt}


# --- Code Mode ---

@router.get("/code-mode")
async def get_code_mode():
    """Get current code mode state."""
    state = get_state_manager()
    return {"ok": True, "enabled": state.is_code_mode_enabled()}


class CodeModeRequest(BaseModel):
    enabled: bool = Field(..., description="Enable or disable code mode")


@router.post("/code-mode")
async def set_code_mode(payload: CodeModeRequest):
    """Enable or disable code mode.

    When enabled, the MCP proxy exposes only two meta-tools (search_tools and
    execute_tool) instead of all individual tools. This reduces context usage
    for LLMs while still allowing access to the full tool catalog via search.
    """
    state = get_state_manager()
    state.set_code_mode_enabled(payload.enabled)
    return {"ok": True, "enabled": state.is_code_mode_enabled()}
