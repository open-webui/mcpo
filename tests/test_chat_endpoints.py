"""
Tests for the /chat/sessions/* endpoints defined in mcpo.api.routers.chat.

Covers:
  - Session CRUD (create, get, delete, reset)
  - Model catalog listing
  - Favorite models get/set
  - Message posting (non-stream, with mocked provider)
  - Error paths (404 for missing sessions, 422 for bad payloads)
"""
from __future__ import annotations

import asyncio
import json
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _build_app(tmp_path) -> Any:
    """Build an MCPO app with a minimal config and return it."""
    from mcpo.main import build_main_app

    cfg = {"mcpServers": {"stub": {"command": "echo", "args": ["ok"]}}}
    cfg_path = tmp_path / "mcpo.json"
    cfg_path.write_text(json.dumps(cfg))
    app = await build_main_app(config_path=str(cfg_path))
    return app


def _reset_session_manager_singleton() -> None:
    """Reset the global ChatSessionManager singleton so tests are isolated."""
    import mcpo.services.chat_sessions as mod
    mod._session_manager = None


def _reset_state_manager_singleton() -> None:
    """Reset the global StateManager singleton so tests are isolated."""
    import mcpo.services.state as mod
    mod._global_state_manager = None


def _create_session_via_client(
    client: TestClient,
    *,
    model: str = "test/fake-model",
    system_prompt: str | None = None,
) -> Dict[str, Any]:
    """POST a new session and return the parsed response body."""
    body: Dict[str, Any] = {"model": model}
    if system_prompt is not None:
        body["system_prompt"] = system_prompt
    resp = client.post("/chat/sessions", json=body)
    assert resp.status_code == 200, f"Unexpected status {resp.status_code}: {resp.text}"
    return resp.json()


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _isolate_singletons(tmp_path):
    """Ensure every test starts with fresh singletons."""
    _reset_session_manager_singleton()
    _reset_state_manager_singleton()
    # Point the state manager at a temp file so it doesn't read the real one
    import mcpo.services.state as state_mod
    state_mod._global_state_manager = state_mod.StateManager(
        state_file_path=str(tmp_path / "test_state.json")
    )
    yield
    _reset_session_manager_singleton()
    _reset_state_manager_singleton()


# We patch _gather_tool_catalog globally for most tests because it calls into
# real MCP sessions which are not available in unit tests.
@pytest.fixture()
def patched_tool_catalog():
    """Patch _gather_tool_catalog to return an empty catalog."""
    with patch(
        "mcpo.api.routers.chat._gather_tool_catalog",
        new_callable=AsyncMock,
        return_value=([], {}),
    ) as mock_cat:
        yield mock_cat


# Patch model catalog so we don't need real API keys or network.
@pytest.fixture()
def patched_model_catalog():
    """Patch _load_model_catalog to return a deterministic model list."""
    models = [
        {"id": "test/fake-model", "label": "Fake Model"},
        {"id": "test/another-model", "label": "Another Model"},
    ]
    with patch(
        "mcpo.api.routers.chat._load_model_catalog",
        new_callable=AsyncMock,
        return_value=models,
    ) as mock_cat:
        yield mock_cat


# ---------------------------------------------------------------------------
# 1. POST /chat/sessions — create session (valid)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_valid(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    body = {"model": "test/fake-model", "system_prompt": "You are helpful."}
    resp = client.post("/chat/sessions", json=body)

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    session = data["session"]
    assert "id" in session
    assert session["model"] == "test/fake-model"
    assert session["systemPrompt"] == "You are helpful."
    # System prompt should appear as the first message
    assert len(session["messages"]) == 1
    assert session["messages"][0]["role"] == "system"


# ---------------------------------------------------------------------------
# 2. POST /chat/sessions — missing model falls back to first model
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_no_model_uses_default(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.post("/chat/sessions", json={})
    assert resp.status_code == 200
    data = resp.json()
    # When model is None, the router picks the first from the catalog
    assert data["session"]["model"] == "test/fake-model"


# ---------------------------------------------------------------------------
# 3. GET /chat/sessions/{id} — valid session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_valid(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    created = _create_session_via_client(client, model="test/fake-model")
    session_id = created["session"]["id"]

    resp = client.get(f"/chat/sessions/{session_id}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["session"]["id"] == session_id
    assert data["session"]["model"] == "test/fake-model"


# ---------------------------------------------------------------------------
# 4. GET /chat/sessions/{id} — invalid session → 404
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_session_not_found(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.get("/chat/sessions/nonexistent-id")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 5. DELETE /chat/sessions/{id} — valid session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_session_valid(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    created = _create_session_via_client(client, model="test/fake-model")
    session_id = created["session"]["id"]

    resp = client.delete(f"/chat/sessions/{session_id}")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True

    # Confirm the session is gone
    resp2 = client.get(f"/chat/sessions/{session_id}")
    assert resp2.status_code == 404


# ---------------------------------------------------------------------------
# 6. DELETE /chat/sessions/{id} — non-existent id (silently succeeds)
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_delete_session_nonexistent(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    # delete_session uses dict.pop(key, None) so it returns 200 even if missing
    resp = client.delete("/chat/sessions/does-not-exist")
    assert resp.status_code == 200
    assert resp.json()["ok"] is True


# ---------------------------------------------------------------------------
# 7. POST /chat/sessions/{id}/reset — valid session
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_session_valid(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    created = _create_session_via_client(
        client, model="test/fake-model", system_prompt="Keep it short."
    )
    session_id = created["session"]["id"]

    resp = client.post(f"/chat/sessions/{session_id}/reset")
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    session = data["session"]
    # After reset, only the system prompt message should remain
    assert len(session["messages"]) == 1
    assert session["messages"][0]["role"] == "system"
    assert session["steps"] == []
    assert session["tools"] == []


# ---------------------------------------------------------------------------
# 7b. POST /chat/sessions/{id}/reset — non-existent → 404
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_session_not_found(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.post("/chat/sessions/nonexistent-id/reset")
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 8. GET /chat/sessions/models — returns model list
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_list_models(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.get("/chat/sessions/models")
    assert resp.status_code == 200
    data = resp.json()
    assert "models" in data
    ids = [m["id"] for m in data["models"]]
    assert "test/fake-model" in ids
    assert "test/another-model" in ids


# ---------------------------------------------------------------------------
# 9. GET /chat/sessions/favorites — empty initially
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_favorites_initially_empty(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.get("/chat/sessions/favorites")
    assert resp.status_code == 200
    data = resp.json()
    assert "favorites" in data
    assert data["favorites"] == []


# ---------------------------------------------------------------------------
# 10. POST /chat/sessions/favorites — set favorites
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_and_get_favorites(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    fav_ids = ["test/fake-model", "test/another-model"]
    resp = client.post("/chat/sessions/favorites", json={"models": fav_ids})
    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert set(data["favorites"]) == set(fav_ids)

    # Verify via GET
    resp2 = client.get("/chat/sessions/favorites")
    assert resp2.status_code == 200
    assert set(resp2.json()["favorites"]) == set(fav_ids)


# ---------------------------------------------------------------------------
# 10b. POST /chat/sessions/favorites — missing body → 422
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_set_favorites_missing_body(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.post("/chat/sessions/favorites", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 11. POST /chat/sessions/{id}/messages — non-stream, mocked provider
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_message_non_stream(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    created = _create_session_via_client(client, model="test/fake-model")
    session_id = created["session"]["id"]

    mock_client = AsyncMock()
    mock_client.chat_completion.return_value = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Hello there!"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 10, "completion_tokens": 5},
    }

    with patch(
        "mcpo.api.routers.chat._get_client_for_model", return_value=mock_client
    ), patch(
        "mcpo.api.routers.chat._ensure_tools", new_callable=AsyncMock
    ):
        resp = client.post(
            f"/chat/sessions/{session_id}/messages",
            json={"message": "Hi!", "stream": False},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert "message" in data
    assert data["message"]["role"] == "assistant"
    assert "Hello there!" in data["message"]["content"]

    # Session should now contain both user and assistant messages
    session_resp = client.get(f"/chat/sessions/{session_id}")
    messages = session_resp.json()["session"]["messages"]
    roles = [m["role"] for m in messages]
    assert "user" in roles
    assert "assistant" in roles


# ---------------------------------------------------------------------------
# 11b. POST /chat/sessions/{id}/messages — with tool_calls loop
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_message_with_tool_calls(tmp_path, patched_tool_catalog, patched_model_catalog):
    """Provider returns a tool_call, then after tool execution returns text."""
    app = await _build_app(tmp_path)
    client = TestClient(app)

    created = _create_session_via_client(client, model="test/fake-model")
    session_id = created["session"]["id"]

    # First call: provider returns a tool_call
    tool_call_response = {
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "call_abc123",
                            "type": "function",
                            "function": {
                                "name": "stub__some_tool",
                                "arguments": '{"x": 1}',
                            },
                        }
                    ],
                },
                "finish_reason": "tool_calls",
            }
        ],
    }

    # Second call: provider returns final text
    final_response = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "Done!"},
                "finish_reason": "stop",
            }
        ],
    }

    mock_client = AsyncMock()
    mock_client.chat_completion.side_effect = [tool_call_response, final_response]

    mock_runner = MagicMock()
    mock_runner.execute_tool = AsyncMock(return_value={"result": "42"})

    # Inject a tool_index entry so _execute_tool can find the mapping
    fake_tool = MagicMock()
    fake_tool.name = "some_tool"
    fake_tool.inputSchema = {"type": "object", "properties": {"x": {"type": "integer"}}}

    fake_session = MagicMock()

    async def fake_ensure_tools(session, request):
        session.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "stub__some_tool",
                    "description": "A test tool",
                    "parameters": fake_tool.inputSchema,
                },
            }
        ]
        session.tool_index = {
            "stub__some_tool": {
                "server": "stub",
                "session": fake_session,
                "tool": fake_tool,
                "originalName": "stub.some_tool",
            }
        }

    with patch(
        "mcpo.api.routers.chat._get_client_for_model", return_value=mock_client
    ), patch(
        "mcpo.api.routers.chat._ensure_tools",
        side_effect=fake_ensure_tools,
    ), patch(
        "mcpo.api.routers.chat.get_runner_service", return_value=mock_runner
    ):
        resp = client.post(
            f"/chat/sessions/{session_id}/messages",
            json={"message": "Run tool", "stream": False},
        )

    assert resp.status_code == 200
    data = resp.json()
    assert data["ok"] is True
    assert data["message"]["content"] == "Done!"

    # Provider was called twice (tool_call + final)
    assert mock_client.chat_completion.call_count == 2

    # Runner executed the tool once
    mock_runner.execute_tool.assert_called_once()
    call_args = mock_runner.execute_tool.call_args
    assert call_args[0][1] == "some_tool"  # tool name


# ---------------------------------------------------------------------------
# 12. POST /chat/sessions/{id}/messages — non-existent session → 404
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_message_session_not_found(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/chat/sessions/nonexistent-id/messages",
        json={"message": "Hello", "stream": False},
    )
    assert resp.status_code == 404


# ---------------------------------------------------------------------------
# 13. POST /chat/sessions/{id}/messages — missing message field → 422
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_message_missing_field(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    created = _create_session_via_client(client, model="test/fake-model")
    session_id = created["session"]["id"]

    resp = client.post(f"/chat/sessions/{session_id}/messages", json={})
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 14. Create session with server_allowlist
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_with_server_allowlist(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    body = {"model": "test/fake-model", "server_allowlist": ["stub"]}
    resp = client.post("/chat/sessions", json=body)
    assert resp.status_code == 200
    session = resp.json()["session"]
    assert session["serverAllowlist"] == ["stub"]


# ---------------------------------------------------------------------------
# 15. Create session — system prompt not set → no system message
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_create_session_no_system_prompt(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.post("/chat/sessions", json={"model": "test/fake-model"})
    assert resp.status_code == 200
    session = resp.json()["session"]
    assert session["systemPrompt"] is None
    assert session["messages"] == []


# ---------------------------------------------------------------------------
# 16. POST /chat/sessions/{id}/messages — streaming returns SSE
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_post_message_stream_returns_sse(tmp_path, patched_tool_catalog, patched_model_catalog):
    """Streaming endpoint returns text/event-stream content type."""
    app = await _build_app(tmp_path)
    client = TestClient(app)

    created = _create_session_via_client(client, model="test/fake-model")
    session_id = created["session"]["id"]

    # Build an async iterator that yields SSE-formatted lines
    async def fake_stream(*args, **kwargs):
        chunks = [
            'data: {"choices":[{"delta":{"role":"assistant","content":"Hi"},"finish_reason":null}]}\n',
            'data: {"choices":[{"delta":{"content":"!"},"finish_reason":"stop"}]}\n',
            "data: [DONE]\n",
        ]
        for c in chunks:
            yield c

    mock_client = AsyncMock()
    mock_client.chat_completion_stream = AsyncMock(return_value=fake_stream())

    with patch(
        "mcpo.api.routers.chat._get_client_for_model", return_value=mock_client
    ), patch(
        "mcpo.api.routers.chat._ensure_tools", new_callable=AsyncMock
    ):
        resp = client.post(
            f"/chat/sessions/{session_id}/messages",
            json={"message": "Hello", "stream": True},
        )

    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")
    # Body should contain SSE data lines
    body = resp.text
    assert "data:" in body


# ---------------------------------------------------------------------------
# 17. Multiple sessions are independent
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_multiple_sessions_independent(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    s1 = _create_session_via_client(client, model="test/fake-model", system_prompt="A")
    s2 = _create_session_via_client(client, model="test/another-model", system_prompt="B")

    id1 = s1["session"]["id"]
    id2 = s2["session"]["id"]
    assert id1 != id2

    r1 = client.get(f"/chat/sessions/{id1}")
    r2 = client.get(f"/chat/sessions/{id2}")
    assert r1.json()["session"]["model"] == "test/fake-model"
    assert r2.json()["session"]["model"] == "test/another-model"

    # Deleting one does not affect the other
    client.delete(f"/chat/sessions/{id1}")
    assert client.get(f"/chat/sessions/{id1}").status_code == 404
    assert client.get(f"/chat/sessions/{id2}").status_code == 200


# ---------------------------------------------------------------------------
# 18. Reset preserves model but clears messages and tools
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_preserves_model(tmp_path, patched_tool_catalog, patched_model_catalog):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    created = _create_session_via_client(
        client, model="test/fake-model", system_prompt="sys"
    )
    session_id = created["session"]["id"]

    resp = client.post(f"/chat/sessions/{session_id}/reset")
    assert resp.status_code == 200
    session = resp.json()["session"]
    assert session["model"] == "test/fake-model"
    assert session["systemPrompt"] == "sys"
    assert session["tools"] == []
