---
name: architecture
description: Background knowledge about OpenHubUI codebase architecture, patterns, and conventions. Auto-loaded when working on this project.
user-invocable: false
---

# OpenHubUI (MCPO) Architecture Guide

## What This Project Is

OpenHubUI aggregates any MCP (Model Context Protocol) server — stdio, SSE, or streamable-http — into a single unified streamable-http endpoint. It bridges the gap between diverse MCP server transports and clients like OpenWebUI that expect streamable HTTP.

## Dual Port Architecture

- **Port 8000** (REST): OpenAPI proxy endpoints `/{server}/{tool}`, Admin UI `/ui`, management APIs `/_meta/*`, chat/completions
- **Port 8001** (MCP): Aggregated streamable-http at `/mcp`

## Key Source Files

| File | Purpose |
|------|---------|
| `src/mcpo/main.py` | Core app builder (`build_main_app`), all management endpoints (~2000 lines) |
| `src/mcpo/proxy.py` | FastMCP-based streamable-http proxy (port 8001) |
| `src/mcpo/cli/commands.py` | ConfigManager, ServerRunner — CLI orchestration |
| `src/mcpo/services/state.py` | StateManager — thread-safe state persistence (RLock) |
| `src/mcpo/services/mcp_tools.py` | MCP session & tool discovery |
| `src/mcpo/services/hot_reload.py` | Config file watcher |
| `src/mcpo/utils/auth.py` | APIKeyMiddleware, JWT verification |
| `src/mcpo/models/config.py` | ServerConfig, AppConfig (Pydantic) |
| `src/mcpo/middleware/mcp_tool_filter.py` | Filters enabled/disabled tools for MCP proxy |

## Patterns & Conventions

### Singleton Services
Access via factory functions: `get_state_manager()`, `get_log_manager()`, `get_metrics_aggregator()`. Thread-safe with RLock.

### Sub-App Mounting
Each MCP server is mounted as a FastAPI sub-app via Starlette `Mount`. Sub-app state stores `ClientSession`, connection flags, and enabled state.

### Error Handling
Use `error_envelope()` for standardized error responses. Custom exception handlers in `build_main_app()`.

### Configuration
- `mcpo.json` — server definitions
- `mcpo_state.json` — runtime state (enable/disable)
- Environment variables for API keys and secrets

### Testing
- pytest + pytest-asyncio
- Tests in `tests/` (integration) and `src/mcpo/tests/` (unit)
- Run with `uv run pytest -v`

## Known Issues (from memory.md)

1. `middleware/mcp_tool_filter.py:200` — calls `get_state()` which doesn't exist, should use `get_server_state()`
2. `app/factory.py` — imports non-existent `build_app`, should be `build_main_app` (file unused)
3. `health.py` and `meta.py` routers are defined but not mounted

## Development

```bash
uv sync --dev          # Install with dev deps
uv run mcpo serve --config mcpo.json --hot-reload  # Start server
uv run pytest -v       # Run tests
```
