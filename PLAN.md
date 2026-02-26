# Code Mode — Implementation Plan

## Concept

Inspired by [Cloudflare's Code Mode MCP](https://blog.cloudflare.com/code-mode-mcp/), Code Mode
collapses all MCP tools across all connected servers into exactly **two tools** exposed to the LLM:

| Tool | Purpose |
|------|---------|
| `search_tools` | Query the tool catalog by keyword. Returns matching tool schemas. |
| `execute_tool` | Execute a discovered tool by its fully-qualified name with arguments. |

This keeps the LLM's context window fixed at ~800 tokens of tool definitions regardless of how many
MCP servers or tools are connected. The agent discovers what it needs on-demand, then calls it.

### User-facing toggle

A single boolean in settings: **Code Mode** (on/off). When on, every chat session and completions
request receives only the two meta-tools instead of the full catalog. When off, behaviour is
unchanged (current flow).

---

## Architecture Overview

```
Client (WebSocket or HTTP)
  │
  ├─ ws://host/v1/responses          ← NEW WebSocket endpoint
  │     │
  │     ├─ Client sends:  { "type": "response.create", "input": [...], "tools": auto }
  │     ├─ Server sends:  { "type": "response.output_text.delta", "delta": "..." }
  │     ├─ Server sends:  { "type": "response.function_call_arguments.delta", ... }
  │     ├─ Server sends:  { "type": "response.function_call_arguments.done", ... }
  │     │     ↓  (server-side: auto-execute tool, feed result back to LLM)
  │     ├─ Server sends:  { "type": "response.tool_result", "call_id": "...", "output": "..." }
  │     ├─ Server sends:  { "type": "response.output_text.delta", "delta": "..." }
  │     └─ Server sends:  { "type": "response.completed", ... }
  │
  └─ POST /v1/chat/completions       ← EXISTING (enhanced with code_mode param)
        └─ SSE stream (unchanged, but tools[] is now search+execute when code_mode=true)
```

---

## Detailed File-by-File Plan

### Phase 1 — Tool Catalog Index & Code Mode Service

#### 1.1  `src/mcpo/services/code_mode.py` (NEW — ~200 lines)

The brain of Code Mode. Maintains an in-memory searchable index of all tools across all servers.

```python
class ToolEntry:
    server_name: str          # "perplexity"
    tool_name: str            # "deep_research"
    fqn: str                  # "perplexity.deep_research"
    description: str          # from MCP tool schema
    input_schema: dict        # full JSON Schema
    keywords: list[str]       # [server_name, tool_name, ...words from description]

class CodeModeService:
    _index: dict[str, ToolEntry]          # fqn → entry
    _keyword_map: dict[str, list[str]]    # keyword → [fqn, ...]

    async def rebuild_index(app: FastAPI) -> None
        """Walk all mounted sub-apps, call session.list_tools(), populate index.
        Called on startup and on config reload."""

    def search(query: str, limit: int = 10) -> list[ToolEntry]
        """Case-insensitive keyword search across server names, tool names, descriptions.
        Returns matching ToolEntry objects with their full schemas.
        Ranking: exact server name match > exact tool name match > description word match."""

    def get_tool(fqn: str) -> ToolEntry | None
        """Lookup a single tool by fully-qualified name."""

    def get_search_context() -> str
        """Returns a compact string for the LLM system prompt:
        - List of server names with 1-line descriptions
        - Search tips (e.g. 'search by server name to see all its tools')
        - Total tool count
        ~200-400 tokens."""
```

**Search algorithm** — simple but effective:

1. Tokenize query into lowercase words
2. For each word, lookup `_keyword_map[word]` → set of FQNs
3. Score each FQN by number of keyword hits + bonus for exact server/tool name match
4. Return top `limit` results sorted by score descending
5. Each result includes the full `inputSchema` so the LLM can immediately call `execute_tool`

**Index rebuild triggers:**
- App startup (after all sub-apps mounted)
- Config hot-reload
- Server add/remove via admin API
- Server/tool enable/disable toggle

#### 1.2  `src/mcpo/services/state.py` (MODIFY — ~10 lines)

Add `code_mode` to persisted state:

```python
# In StateManager.__init__:
self._code_mode_enabled: bool = False

# In load_state:
self._code_mode_enabled = data.get('code_mode_enabled', False)

# In save_state:
state_data["code_mode_enabled"] = self._code_mode_enabled

# New methods:
def is_code_mode_enabled(self) -> bool
def set_code_mode_enabled(self, enabled: bool) -> None
```

#### 1.3  `src/mcpo/models/config.py` (MODIFY — ~5 lines)

No changes needed — code_mode is runtime state, not server config.

---

### Phase 2 — Meta-Tool Definitions

#### 2.1  `src/mcpo/services/code_mode.py` (continued)

Add two static methods that return OpenAI-format tool definitions:

```python
@staticmethod
def get_meta_tools() -> list[dict]:
    """Return the two Code Mode tool definitions in OpenAI function-calling format."""
    return [
        {
            "type": "function",
            "function": {
                "name": "search_tools",
                "description": (
                    "Search the tool catalog by keyword. Use server names, tool names, "
                    "or descriptive words. Returns matching tool schemas with their "
                    "fully-qualified names and input parameters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search keywords (e.g. server name, action verb, topic)"
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Max results to return (default 5)",
                            "default": 5
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "execute_tool",
                "description": (
                    "Execute a tool by its fully-qualified name (server.tool_name) "
                    "with the specified arguments. Use search_tools first to discover "
                    "available tools and their required parameters."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "tool_fqn": {
                            "type": "string",
                            "description": "Fully-qualified tool name: server_name.tool_name"
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments matching the tool's input schema",
                            "additionalProperties": True
                        }
                    },
                    "required": ["tool_fqn", "arguments"]
                }
            }
        }
    ]
```

---

### Phase 3 — Integrate Code Mode into Chat Router

#### 3.1  `src/mcpo/api/routers/chat.py` (MODIFY — ~60 lines)

**In `_gather_tool_catalog()`** — branch on code mode:

```python
async def _gather_tool_catalog(app, allowlist=None):
    state_manager = get_state_manager()

    if state_manager.is_code_mode_enabled():
        # Code Mode: return only the two meta-tools
        code_mode_svc = get_code_mode_service()
        await code_mode_svc.rebuild_index(app)  # ensure fresh
        tools = code_mode_svc.get_meta_tools()
        # tool_map not needed — execution handled by code_mode_svc
        return tools, {}  # empty tool_map; code mode handles dispatch

    # ... existing tool gathering logic unchanged ...
```

**In `_execute_tool()`** — handle meta-tool dispatch:

```python
async def _execute_tool(app, tool_call, tool_map, ...):
    func_name = tool_call["function"]["name"]
    args = json.loads(tool_call["function"]["arguments"])

    if func_name == "search_tools":
        # Direct dispatch to CodeModeService
        code_mode_svc = get_code_mode_service()
        results = code_mode_svc.search(args["query"], args.get("limit", 5))
        return json.dumps([{
            "name": r.fqn,
            "description": r.description,
            "input_schema": r.input_schema
        } for r in results])

    elif func_name == "execute_tool":
        # Resolve FQN → server + tool, then execute via runner
        code_mode_svc = get_code_mode_service()
        entry = code_mode_svc.get_tool(args["tool_fqn"])
        if not entry:
            return json.dumps({"error": f"Tool '{args['tool_fqn']}' not found"})

        # Get the MCP session for this server
        session = _get_session_for_server(app, entry.server_name)
        if not session:
            return json.dumps({"error": f"Server '{entry.server_name}' not connected"})

        runner = get_runner_service()
        result = await runner.execute_tool(
            session, entry.tool_name, args.get("arguments", {}), timeout=30.0
        )
        return json.dumps(result) if not isinstance(result, str) else result

    # ... existing tool dispatch for non-code-mode ...
```

**System prompt injection** — when code mode is active, prepend context to the first user message
or inject a system message:

```python
if state_manager.is_code_mode_enabled():
    code_mode_svc = get_code_mode_service()
    context = code_mode_svc.get_search_context()
    messages.insert(0, {
        "role": "system",
        "content": (
            "You are in Code Mode. You have two tools: search_tools and execute_tool.\n"
            f"{context}\n"
            "Workflow: 1) search_tools to find relevant tools, "
            "2) execute_tool with the exact fqn and arguments from the schema."
        )
    })
```

---

### Phase 4 — WebSocket Responses Endpoint

#### 4.1  `src/mcpo/api/routers/websocket_responses.py` (NEW — ~350 lines)

A WebSocket endpoint that mirrors OpenAI's Responses API event protocol. Runs the full
search→execute loop server-side within a single stream.

**Connection:** `ws://host/v1/responses`

**Event Protocol (mirroring OpenAI):**

Client → Server events:
```json
{
  "type": "response.create",
  "model": "openrouter/anthropic/claude-sonnet-4-20250514",
  "input": [
    {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "..."}]}
  ],
  "tools": [],                              // optional override; empty = use code mode tools
  "previous_response_id": "resp_abc123"     // optional continuation
}
```

Server → Client events:
```
response.created           → { response_id, status: "in_progress" }
response.in_progress       → { response_id }
response.output_item.added → { output_index, item: { type: "message"|"function_call", ... } }
response.output_text.delta → { output_index, delta }
response.function_call_arguments.delta → { output_index, call_id, delta }
response.function_call_arguments.done  → { output_index, call_id, name, arguments }
response.tool_result       → { call_id, output }    // our extension
response.output_item.done  → { output_index, item }
response.completed         → { response_id, usage }
error                      → { message, code }
```

**Implementation:**

```python
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json, uuid, asyncio

router = APIRouter(tags=["websocket-responses"])

class ResponseSession:
    """Manages state for a single WebSocket connection."""
    def __init__(self, ws: WebSocket, app):
        self.ws = ws
        self.app = app
        self.responses: dict[str, dict] = {}  # response_id → state

    async def send_event(self, event_type: str, **data):
        await self.ws.send_json({"type": event_type, **data})

    async def handle_response_create(self, payload: dict):
        response_id = f"resp_{uuid.uuid4().hex[:12]}"
        model = payload.get("model", "openrouter/anthropic/claude-sonnet-4-20250514")
        input_items = payload.get("input", [])
        tools_override = payload.get("tools")
        prev_id = payload.get("previous_response_id")

        # Build message history
        messages = self._build_messages(input_items, prev_id)

        # Determine tools
        state_manager = get_state_manager()
        if state_manager.is_code_mode_enabled() and not tools_override:
            code_mode_svc = get_code_mode_service()
            await code_mode_svc.rebuild_index(self.app)
            tools = code_mode_svc.get_meta_tools()
            context = code_mode_svc.get_search_context()
            messages.insert(0, {"role": "system", "content": f"Code Mode active.\n{context}"})
        else:
            tools = tools_override or await self._gather_all_tools()

        await self.send_event("response.created",
            response_id=response_id, status="in_progress")

        # Agentic loop: call LLM, handle tool calls, repeat
        output_index = 0
        max_iterations = 20  # safety limit

        for iteration in range(max_iterations):
            # Call provider with streaming
            client = _get_client_for_model(model)
            stream = client.chat_completion_stream(
                model=model, messages=messages, tools=tools
            )

            # Accumulate tool calls and text from stream
            current_text = ""
            current_tool_calls = []
            async for chunk in stream:
                delta = chunk.get("choices", [{}])[0].get("delta", {})

                # Stream text deltas
                if delta.get("content"):
                    text_delta = delta["content"]
                    current_text += text_delta
                    await self.send_event("response.output_text.delta",
                        response_id=response_id,
                        output_index=output_index,
                        delta=text_delta)

                # Stream function call argument deltas
                if delta.get("tool_calls"):
                    for tc_delta in delta["tool_calls"]:
                        idx = tc_delta.get("index", 0)
                        while len(current_tool_calls) <= idx:
                            current_tool_calls.append({
                                "id": "", "name": "", "arguments": ""
                            })
                        tc = current_tool_calls[idx]
                        if tc_delta.get("id"):
                            tc["id"] = tc_delta["id"]
                        func = tc_delta.get("function", {})
                        if func.get("name"):
                            tc["name"] = func["name"]
                        if func.get("arguments"):
                            tc["arguments"] += func["arguments"]
                            await self.send_event(
                                "response.function_call_arguments.delta",
                                response_id=response_id,
                                output_index=output_index,
                                call_id=tc["id"],
                                delta=func["arguments"])

            # If no tool calls, we're done
            if not current_tool_calls:
                if current_text:
                    await self.send_event("response.output_item.done",
                        response_id=response_id,
                        output_index=output_index,
                        item={"type": "message", "content": current_text})
                break

            # Process each tool call
            for tc in current_tool_calls:
                await self.send_event(
                    "response.function_call_arguments.done",
                    response_id=response_id,
                    call_id=tc["id"],
                    name=tc["name"],
                    arguments=tc["arguments"])

                # Execute the tool server-side
                result = await self._execute_meta_tool(tc["name"], tc["arguments"])

                await self.send_event("response.tool_result",
                    response_id=response_id,
                    call_id=tc["id"],
                    output=result)

                # Append assistant + tool result to messages for next iteration
                messages.append({
                    "role": "assistant",
                    "tool_calls": [{"id": tc["id"], "type": "function",
                                    "function": {"name": tc["name"],
                                                 "arguments": tc["arguments"]}}]
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc["id"],
                    "content": result
                })

            output_index += 1

        # Done
        await self.send_event("response.completed", response_id=response_id)
        self.responses[response_id] = {"messages": messages}

    async def _execute_meta_tool(self, name: str, arguments_json: str) -> str:
        """Execute search_tools or execute_tool, return string result."""
        args = json.loads(arguments_json)
        code_mode_svc = get_code_mode_service()

        if name == "search_tools":
            results = code_mode_svc.search(args["query"], args.get("limit", 5))
            return json.dumps([{
                "name": r.fqn,
                "description": r.description,
                "input_schema": r.input_schema
            } for r in results])

        elif name == "execute_tool":
            entry = code_mode_svc.get_tool(args["tool_fqn"])
            if not entry:
                return json.dumps({"error": f"Tool '{args['tool_fqn']}' not found"})
            session = _get_session_for_server(self.app, entry.server_name)
            if not session:
                return json.dumps({"error": f"Server '{entry.server_name}' not connected"})
            runner = get_runner_service()
            result = await runner.execute_tool(session, entry.tool_name, args.get("arguments", {}))
            return json.dumps(result) if not isinstance(result, str) else result

        return json.dumps({"error": f"Unknown tool: {name}"})


@router.websocket("/v1/responses")
async def websocket_responses(websocket: WebSocket):
    await websocket.accept()
    app = websocket.app
    session = ResponseSession(websocket, app)

    try:
        while True:
            raw = await websocket.receive_text()
            event = json.loads(raw)

            if event.get("type") == "response.create":
                await session.handle_response_create(event)
            else:
                await session.send_event("error",
                    message=f"Unknown event type: {event.get('type')}",
                    code="unknown_event")
    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await session.send_event("error", message=str(e), code="internal_error")
        except:
            pass
```

---

### Phase 5 — Admin API & UI Toggle

#### 5.1  `src/mcpo/api/routers/admin.py` (MODIFY — ~20 lines)

Add endpoints for toggling code mode:

```python
@router.get("/_meta/code_mode")
async def get_code_mode():
    return {"enabled": get_state_manager().is_code_mode_enabled()}

@router.post("/_meta/code_mode/enable")
async def enable_code_mode():
    get_state_manager().set_code_mode_enabled(True)
    return {"ok": True, "enabled": True}

@router.post("/_meta/code_mode/disable")
async def disable_code_mode():
    get_state_manager().set_code_mode_enabled(False)
    return {"ok": True, "enabled": False}
```

#### 5.2  `static/ui/index.html` (MODIFY — ~30 lines)

Add a toggle switch in the settings panel:

```html
<!-- In settings section -->
<div class="setting-row">
  <label>Code Mode</label>
  <p class="setting-desc">
    Collapse all tools into search + execute. Reduces token usage by ~99%.
  </p>
  <input type="checkbox" id="code-mode-toggle" onchange="toggleCodeMode(this.checked)">
</div>
```

```javascript
async function toggleCodeMode(enabled) {
  await fetch(`/_meta/code_mode/${enabled ? 'enable' : 'disable'}`, {method: 'POST'});
}

async function loadCodeModeState() {
  const res = await fetch('/_meta/code_mode');
  const data = await res.json();
  document.getElementById('code-mode-toggle').checked = data.enabled;
}
```

---

### Phase 6 — Mount & Wiring

#### 6.1  `src/mcpo/main.py` (MODIFY — ~15 lines)

```python
# At import section:
from mcpo.api.routers.websocket_responses import router as ws_responses_router

# In build_main_app():
app.include_router(ws_responses_router)

# After mount_config_servers(), rebuild code mode index:
from mcpo.services.code_mode import get_code_mode_service
code_mode_svc = get_code_mode_service()
await code_mode_svc.rebuild_index(app)

# In reload_config_handler(), after server remount:
await code_mode_svc.rebuild_index(app)
```

---

## Event Flow — Complete Example

User asks: "Search for recent AI news using Perplexity"

```
1. Client → WS:  { "type": "response.create", "model": "...",
                    "input": [{"type":"message","role":"user",
                    "content":[{"type":"input_text","text":"Search for recent AI news using Perplexity"}]}]}

2. Server → WS:  { "type": "response.created", "response_id": "resp_a1b2c3", "status": "in_progress" }

3. LLM decides to call search_tools("perplexity")
   Server → WS:  { "type": "response.function_call_arguments.delta",
                    "call_id": "call_001", "delta": "{\"qu" }
   Server → WS:  { "type": "response.function_call_arguments.delta",
                    "call_id": "call_001", "delta": "ery\":\"perplexity\"}" }
   Server → WS:  { "type": "response.function_call_arguments.done",
                    "call_id": "call_001", "name": "search_tools",
                    "arguments": "{\"query\":\"perplexity\"}" }

4. Server executes search_tools internally:
   → Returns 3 tools: perplexity.search, perplexity.deep_research, perplexity.citations

   Server → WS:  { "type": "response.tool_result", "call_id": "call_001",
                    "output": "[{name: 'perplexity.deep_research', description: '...', input_schema: {...}}, ...]" }

5. LLM sees schemas, decides to call execute_tool("perplexity.deep_research", {query: "recent AI news"})
   Server → WS:  { "type": "response.function_call_arguments.delta", ... }
   Server → WS:  { "type": "response.function_call_arguments.done",
                    "call_id": "call_002", "name": "execute_tool",
                    "arguments": "{\"tool_fqn\":\"perplexity.deep_research\",\"arguments\":{\"query\":\"recent AI news\"}}" }

6. Server executes the actual MCP tool via runner service:
   Server → WS:  { "type": "response.tool_result", "call_id": "call_002",
                    "output": "{ ... deep research results ... }" }

7. LLM generates final answer, streamed as deltas:
   Server → WS:  { "type": "response.output_text.delta", "delta": "Here are the latest..." }
   Server → WS:  { "type": "response.output_text.delta", "delta": " developments in AI..." }
   ...
   Server → WS:  { "type": "response.output_item.done", "item": { "type": "message", "content": "..." } }
   Server → WS:  { "type": "response.completed", "response_id": "resp_a1b2c3" }
```

All of this happens over **a single WebSocket stream**. The client never sees intermediate
HTTP requests. The agentic loop (search → inspect → execute → respond) runs entirely server-side.

---

## File Summary

| File | Action | Lines Changed |
|------|--------|---------------|
| `src/mcpo/services/code_mode.py` | **NEW** | ~200 |
| `src/mcpo/services/state.py` | MODIFY | ~15 |
| `src/mcpo/api/routers/websocket_responses.py` | **NEW** | ~350 |
| `src/mcpo/api/routers/chat.py` | MODIFY | ~60 |
| `src/mcpo/api/routers/admin.py` | MODIFY | ~20 |
| `src/mcpo/main.py` | MODIFY | ~15 |
| `static/ui/index.html` | MODIFY | ~30 |
| `tests/test_code_mode.py` | **NEW** | ~150 |
| `tests/test_websocket_responses.py` | **NEW** | ~200 |

**Total: ~1,040 lines across 9 files**

---

## Key Design Decisions

1. **No code execution sandbox** — Unlike Cloudflare, we don't run user-generated code. The LLM
   calls `search_tools` and `execute_tool` as standard function calls. This is simpler, more
   secure, and works with every LLM provider.

2. **Server-side agentic loop** — The WebSocket endpoint handles the full search→execute→respond
   cycle internally. The client receives a single stream of events. This matches OpenAI's
   Responses API pattern where the server drives the conversation.

3. **Keyword search, not code** — The search function uses keyword matching against server names,
   tool names, and descriptions. The server name IS the primary search keyword (e.g., searching
   "perplexity" returns all Perplexity tools). Simple, fast, no V8 sandbox needed.

4. **FQN format: `server.tool`** — Unambiguous addressing. The dot separator is chosen because
   underscores and dashes are valid in tool names. The CodeModeService resolves FQNs to
   (server, tool) pairs and dispatches to the correct MCP session.

5. **WebSocket protocol mirrors OpenAI** — Same event types (`response.created`,
   `response.output_text.delta`, `response.function_call_arguments.delta`, etc.) so clients
   built for OpenAI's Responses API work with minimal adaptation. We add one extension:
   `response.tool_result` for tool execution results.

6. **Code mode is global state, not per-session** — Toggled in settings, persisted in
   `mcpo_state.json`. This keeps the implementation simple and matches the existing pattern
   for server/tool enable/disable.

7. **Graceful degradation** — If code mode is enabled but no tools are indexed (e.g., all servers
   disconnected), `search_tools` returns an empty array and `execute_tool` returns a clear error.
   The LLM can still respond conversationally.

---

## Testing Strategy

### Unit Tests (`tests/test_code_mode.py`)

- `test_index_builds_from_mock_sessions` — Verify tool entries are created correctly
- `test_search_by_server_name` — "perplexity" → returns perplexity.* tools
- `test_search_by_tool_name` — "deep_research" → returns matching tools
- `test_search_by_description_word` — "weather" → returns weather-related tools
- `test_search_ranking` — Exact name match ranks higher than description match
- `test_execute_tool_resolves_fqn` — "perplexity.search" → correct server + tool
- `test_get_meta_tools_schema` — Verify the two tool definitions are valid OpenAI format
- `test_get_search_context` — Verify context string is compact and informative
- `test_state_toggle` — code_mode_enabled persists across save/load
- `test_code_mode_tools_in_chat` — Chat endpoint returns only 2 tools when enabled

### Integration Tests (`tests/test_websocket_responses.py`)

- `test_ws_connect_and_create` — Basic WebSocket connect, send response.create, receive events
- `test_ws_search_then_execute` — Full agentic loop with mocked LLM
- `test_ws_streaming_deltas` — Verify text and function_call deltas stream correctly
- `test_ws_error_handling` — Unknown tool, disconnected server, invalid JSON
- `test_ws_previous_response_id` — Continuation chaining works
- `test_ws_max_iterations` — Safety limit prevents infinite loops
