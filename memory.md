# MCPO Codebase Memory File
> Generated: 2026-02-16 — Full audit session

## Architecture Overview

MCPO is a Python proxy that exposes MCP (Model Context Protocol) servers as HTTP/OpenAPI endpoints. Two server processes:

- **Port 8000 (OpenAPI Proxy):** FastAPI app (`main.py`) connects to MCP servers (stdio/SSE/streamable-http), discovers tools, dynamically generates `POST /toolName` endpoints per server.
- **Port 8001 (MCP Proxy):** FastMCP-based streamable-http proxy (`proxy.py`) re-exposes MCP servers via native MCP protocol.

### Startup Flow
1. CLI: `mcpo serve` → `cli/__init__.py` → `ServerRunner.start_server()` → `asyncio.run(mcpo.main.run(...))`
2. Legacy: `mcpo` → `__init__.py` Typer → `asyncio.run(mcpo.main.run(...))`
3. `run()` → `build_main_app()` → FastAPI + sub-apps per server from config → uvicorn

### Package Structure
```
src/mcpo/
├── __init__.py          # Legacy Typer CLI
├── __main__.py          # python -m mcpo → cli
├── main.py              # GOD FILE (~2057 lines) - app builder, all mgmt endpoints
├── proxy.py             # FastMCP proxy (port 8001)
├── api/routers/         # admin.py, chat.py, completions.py, health.py, meta.py, tools.py
├── app/factory.py       # BROKEN - imports non-existent build_app
├── cli/                 # Typer CLI + ConfigManager/ServerRunner
├── middleware/           # mcp_tool_filter.py
├── models/              # config.py, domain.py, responses.py
├── providers/           # openai.py, anthropic.py, google.py, minimax.py, openrouter.py
├── services/            # state.py, chat_sessions.py, hot_reload.py, logging.py, etc.
└── utils/               # auth.py, config.py, config_watcher.py, main.py
```

## Complete Endpoint Inventory

### Mounted Routers (active)
| Method | Path | Router | Handler |
|--------|------|--------|---------|
| POST | `/_meta/env` | admin.py | `update_env_vars` |
| GET | `/chat/sessions/models` | chat.py | `list_models` |
| GET | `/chat/sessions/favorites` | chat.py | `get_favorite_models` |
| POST | `/chat/sessions/favorites` | chat.py | `set_favorite_models` |
| POST | `/chat/sessions` | chat.py | `create_session` |
| GET | `/chat/sessions/{id}` | chat.py | `get_session` |
| DELETE | `/chat/sessions/{id}` | chat.py | `delete_session` |
| POST | `/chat/sessions/{id}/reset` | chat.py | `reset_session` |
| POST | `/chat/sessions/{id}/messages` | chat.py | `post_message` |
| POST | `/v1/chat/completions` | completions.py | `chat_completions` |
| POST | `/v1/completions` | completions.py | `legacy_completions` |
| GET | `/v1/chat/completions/models` | completions.py | `list_completion_models` |
| GET | `/v1/completions/models` | completions.py | `legacy_list_completion_models` |
| GET | `/v1/models` | completions.py | `list_models_root` |

### Inline endpoints in main.py (active)
~20 `/_meta/*` endpoints for server/tool management, config CRUD, logging, metrics.
Internal MCPO server at `/mcpo/*` with pip install, config, env, logs, requirements.

### Dead Code Routers (NOT mounted)
- health.py — `/healthz` never included
- meta.py — `/_meta/*` duplicated inline in main.py, never included

### Dynamic Endpoints
- `POST /{tool_name}` per MCP tool — created by tools.py during lifespan

## Singleton State Map

| Singleton | Lock-protected init? | Thread-safe methods? |
|-----------|---------------------|---------------------|
| StateManager | NO | Yes (RLock) |
| ChatSessionManager | Yes (asyncio.Lock) | Partial (dict only) |
| HotReloadService | Yes (threading.Lock) | Yes |
| LogManager | NO | Yes (threading.Lock) |
| MetricsAggregator | Yes (threading.Lock) | Yes |
| ServerRegistry | NO | NO |
| RunnerService | Yes (threading.Lock) | Yes |

## Provider Capabilities

| Provider | Streaming | Retry | Tool Calling | Thinking | Timeout Config |
|----------|-----------|-------|-------------|----------|----------------|
| OpenAI | Yes | Yes (exp backoff) | Yes | Yes | Env vars |
| Anthropic | Yes | Yes (exp backoff) | Yes | Yes | Env vars (import-time!) |
| Google | Yes | Yes (exp backoff) | Yes | Yes | Env vars (import-time!) |
| MiniMax | Yes (delegated) | Yes (delegated) | Yes | Yes | Hardcoded 120s |
| OpenRouter | Yes | NO | Yes | Yes | Hardcoded 60s |

---

## CRITICAL BUGS (Must Fix Before Production)

### BUG-001: MCPToolFilterMiddleware crashes — `get_state()` doesn't exist
- **File:** middleware/mcp_tool_filter.py:200
- **Impact:** AttributeError at runtime for aggregate proxy tool lookups without `__` separator
- **Fix:** Use `get_server_state()` or `get_all_states()` instead

### BUG-002: app/factory.py imports non-existent `build_app`
- **File:** app/factory.py:16
- **Impact:** ImportError on any import of this module. Function is `build_main_app`
- **Fix:** Change import to `build_main_app`

### BUG-003: Unauthenticated RCE via pip install
- **File:** main.py:670-700 (`install_python_package`)
- **Impact:** No validation, no auth → arbitrary package install = code execution
- **Fix:** Require auth, add allowlist

### BUG-004: Unauthenticated env injection
- **File:** admin.py:152-177 (`POST /_meta/env`) and main.py:843-868 (`post_env`)
- **Impact:** Write arbitrary env vars (PATH, LD_PRELOAD, API keys)
- **Fix:** Require auth, validate key format, deny dangerous keys

### BUG-005: OpenAI Responses API discards conversation history
- **File:** openai.py:447
- **Impact:** Only last message sent to o3/o4/gpt-5/codex models
- **Fix:** Include full message history in `input` field

## HIGH BUGS

### BUG-006: Read-only mode bypass — wrong attribute name
- **File:** meta.py uses `request.app.state.read_only`, main.py sets `main_app.state.read_only_mode`
- **Impact:** All meta router read-only checks silently pass

### BUG-007: OpenRouter streaming timeout=None → hangs forever
- **File:** openrouter.py:120
- **Fix:** Add bounded timeout

### BUG-008: OpenRouter has no retry logic
- **File:** openrouter.py (entire file)
- **Fix:** Add retry with backoff like other providers

### BUG-009: Chat sessions — unbounded memory leak
- **File:** services/chat_sessions.py
- **Impact:** No TTL, no eviction, no max count
- **Fix:** Add TTL or LRU eviction

### BUG-010: Middleware drops response headers on filter
- **File:** middleware/mcp_tool_filter.py:99-101
- **Impact:** CORS, cache-control headers lost → browser clients break

### BUG-011: ServerRegistry has zero thread safety
- **File:** services/registry.py
- **Impact:** Concurrent hot-reload + API = data corruption

### BUG-012: Anthropic/Google env vars evaluated at import time
- **File:** anthropic.py:20-24, google.py:20-24
- **Impact:** `--env-path` loaded after import → stale defaults

### BUG-013: error_envelope shadowed in meta.py
- **File:** meta.py:23-36
- **Impact:** Local def immediately overwritten by import; different signatures

### BUG-014: Pydantic v1 validators in v2 codebase
- **File:** models/domain.py:73-79
- **Impact:** Deprecation warnings, will break in strict mode or Pydantic v3

## MEDIUM BUGS

### BUG-015: API key comparison timing attack
- **File:** utils/auth.py:69,83
- **Fix:** Use hmac.compare_digest()

### BUG-016: Non-atomic state/config file writes
- **File:** services/state.py, services/registry.py
- **Fix:** Write to temp file + os.replace()

### BUG-017: Singleton factory races (no locks)
- **File:** state.py:160, logging.py:150, registry.py:127
- **Fix:** Add threading.Lock to factory functions

### BUG-018: Proxy hardcoded redirect to 127.0.0.1:8000
- **File:** proxy.py:279

### BUG-019: Duplicate config watcher implementations
- **File:** utils/config_watcher.py AND services/hot_reload.py

### BUG-020: hot_reload stop_watching doesn't unschedule observer
- **File:** services/hot_reload.py:107-111

### BUG-021: Config env interpolation only matches uppercase
- **File:** utils/config.py and proxy.py — regex `[A-Z0-9_]+`

### BUG-022: mcpo.json.backup structure mismatch
- **File:** mcpo.json.backup — wrapper `config` key not expected by load_config

### BUG-023: Google API key in URL query parameter
- **File:** completions.py:182, chat.py:150

### BUG-024: No rate limiting on any endpoint

### BUG-025: CORS maximally permissive with credentials
- **File:** main.py — allow_origins=["*"], allow_credentials=True

### BUG-026: Stale session references after server reinit
- **File:** chat.py — tool_index stores direct session references

### BUG-027: Double-JSON-encoded timestamp in error envelope
- **File:** utils/main.py:680

### BUG-028: MiniMax/OpenRouter timeouts not env-configurable

### BUG-029: OpenAI uses non-standard OPEN_AI_API_KEY env var
- **File:** openai.py:106

### BUG-030: Gemini v1beta API in production
- **File:** google.py:17

## LOW BUGS

- BUG-031: pyproject.toml version not PEP 440 compliant
- BUG-032: Chinese colon in main.py description string
- BUG-033: Duplicate 200-line handler factory in utils/main.py
- BUG-034: LogManager add_log return type annotated -> None but returns LogEntry
- BUG-035: Dual log storage (2x memory)
- BUG-036: log_buffer.pop(0) is O(n)
- BUG-037: Duplicate import in runner.py
- BUG-038: Unused imports in auth.py
- BUG-039: Variable shadowing in utils/main.py (status)
- BUG-040: Dead OpenAI provider — not imported by any router
- BUG-041: SessionMetrics.latencies grows unbounded
- BUG-042: New httpx.AsyncClient per request (no connection pooling)

## Test Coverage Gaps

Existing tests cover: config models, enable/disable flow, health/timeout, hot reload, MCP tools integration, meta logs, model/tool simulation, OpenAPI schema, protocol validation, schema processing, state persistence, structured output, tool errors, tool timeout.

Missing coverage:
- All chat session endpoints (CRUD, messaging, streaming)
- All completions endpoints (/v1/chat/completions, /v1/models)
- Admin endpoints (/_meta/env, /_meta/config, /_meta/servers CRUD)
- Provider dispatch and error handling
- Middleware tool filtering
- Auth middleware (API key, Bearer, Basic)
- Dynamic tool endpoint execution
- Config interpolation edge cases
- Singleton initialization concurrency
- File atomicity under failure

## New Test Files Created (This Audit)

| File | Tests | Coverage |
|------|-------|----------|
| tests/test_meta_endpoints.py | 47 | All /_meta/* management endpoints |
| tests/test_completions_endpoints.py | 35 | /v1/chat/completions, /v1/models, provider inference |
| tests/test_services.py | 48 | StateManager, ChatSessionManager, LogManager, MetricsAggregator, ServerRegistry, RunnerService |
| tests/test_chat_endpoints.py | 21 | /chat/sessions CRUD, messaging, streaming, favorites |
| tests/test_utils_and_models.py | 62 | Config interpolation, auth middleware, schema processing, all Pydantic models |
| **Total** | **213** | **All API layers** |

**Result: 213 passed, 0 failed in 1.92s** (3 Pydantic deprecation warnings from production code)

### Pre-existing Test Failures (12 tests, NOT caused by this audit)
| Test File | Failures | Root Cause |
|-----------|----------|------------|
| test_enable_disable_flow.py | 2 | Uses `app.state.server_enabled` dict (migrated to StateManager) |
| test_state_persistence.py | 3 | Same stale `app.state.server_enabled` pattern |
| test_protocol_and_output_validation.py | 4 | Same stale attribute access |
| test_openapi_no_empty_schemas.py | 1 | Schema generation issue |
| test_tool_timeout.py | 1 | `body['error']['code']` KeyError |
| test_time_server_integration.py | 1 | Missing `trio` module |

## Files Modified (Uncommitted on mcppo branch)
.env.example, admin.py, chat.py, completions.py, anthropic.py, google.py, minimax.py, openai.py, openrouter.py, chat_sessions.py, state.py, chat.css, config.css, index.html, navigation.js, api.js, chat.js

## Backup
Branch: `pre-audit-backup-2026-02-16` — full snapshot of all working changes
