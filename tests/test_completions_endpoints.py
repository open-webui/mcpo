"""
Tests for /v1/* completions endpoints (chat completions, models, legacy aliases).

All provider HTTP calls are mocked — no real API calls are made.
"""

import json
import pytest
from unittest.mock import patch, AsyncMock, MagicMock

from fastapi.testclient import TestClient


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _build_app(tmp_path):
    """Build a minimal MCPO app with an empty config for testing."""
    from mcpo.main import build_main_app

    cfg_path = tmp_path / "mcpo.json"
    cfg_path.write_text(json.dumps({"mcpServers": {}}))
    app = await build_main_app(config_path=str(cfg_path))
    return app


FAKE_MODELS_OPENROUTER = [
    {"id": "openrouter/auto", "label": "Auto", "provider": "openrouter"},
]
FAKE_MODELS_OPENAI = [
    {"id": "gpt-4o", "label": "OpenAI: gpt-4o", "provider": "openai"},
]
FAKE_MODELS_GOOGLE = [
    {"id": "gemini-2.0-flash", "label": "Google: Gemini 2.0 Flash", "provider": "gemini"},
]
FAKE_MODELS_ANTHROPIC = [
    {"id": "claude-sonnet-4-20250514", "label": "Anthropic: Claude Sonnet 4", "provider": "anthropic"},
]
FAKE_MODELS_MINIMAX = [
    {"id": "minimax-m2", "label": "MiniMax: minimax-m2", "provider": "minimax"},
]


def _all_fake_models():
    return (
        FAKE_MODELS_OPENROUTER
        + FAKE_MODELS_OPENAI
        + FAKE_MODELS_GOOGLE
        + FAKE_MODELS_ANTHROPIC
        + FAKE_MODELS_MINIMAX
    )


FAKE_COMPLETION_RESPONSE = {
    "id": "chatcmpl-test-123",
    "object": "chat.completion",
    "created": 1700000000,
    "model": "gpt-4o",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "Hello from mock!"},
            "finish_reason": "stop",
        }
    ],
}


# ---------------------------------------------------------------------------
# 1. GET /v1/models — returns model list structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._list_models")
@patch("mcpo.api.routers.completions._filter_by_favorites", side_effect=lambda m: m)
async def test_get_v1_models(mock_filter, mock_list, tmp_path):
    mock_list.return_value = _all_fake_models()
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.get("/v1/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert isinstance(body["data"], list)
    assert len(body["data"]) == len(_all_fake_models())
    # Each entry has minimal OpenAI-compatible fields
    for entry in body["data"]:
        assert "id" in entry
        assert entry["object"] == "model"
    # Raw models list also returned
    assert "models" in body


# ---------------------------------------------------------------------------
# 2. GET /v1/chat/completions/models — same structure
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._list_models")
@patch("mcpo.api.routers.completions._filter_by_favorites", side_effect=lambda m: m)
async def test_get_chat_completions_models(mock_filter, mock_list, tmp_path):
    mock_list.return_value = _all_fake_models()
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.get("/v1/chat/completions/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == len(_all_fake_models())
    # This endpoint adds a "label" field
    for entry in body["data"]:
        assert "id" in entry
        assert "label" in entry


# ---------------------------------------------------------------------------
# 3. GET /v1/completions/models — legacy alias works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._list_models")
@patch("mcpo.api.routers.completions._filter_by_favorites", side_effect=lambda m: m)
async def test_get_completions_models_legacy(mock_filter, mock_list, tmp_path):
    mock_list.return_value = FAKE_MODELS_OPENAI
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.get("/v1/completions/models")
    assert resp.status_code == 200
    body = resp.json()
    assert body["object"] == "list"
    assert len(body["data"]) == 1
    assert body["data"][0]["id"] == "gpt-4o"


# ---------------------------------------------------------------------------
# 4. POST /v1/chat/completions — valid request, mocked provider
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._resolve_provider")
async def test_post_chat_completions_valid(mock_resolve, tmp_path):
    mock_provider = MagicMock()
    mock_provider.name = "openai"
    mock_provider.complete = AsyncMock(return_value=FAKE_COMPLETION_RESPONSE)
    mock_resolve.return_value = mock_provider

    app = await _build_app(tmp_path)
    client = TestClient(app)

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Say hello"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "chatcmpl-test-123"
    assert body["choices"][0]["message"]["content"] == "Hello from mock!"
    mock_provider.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# 5. POST /v1/chat/completions — missing model field → 422
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_chat_completions_missing_model(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    payload = {
        "messages": [{"role": "user", "content": "hello"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 422


# ---------------------------------------------------------------------------
# 6. POST /v1/chat/completions — empty messages → validation error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_post_chat_completions_empty_messages(tmp_path):
    app = await _build_app(tmp_path)
    client = TestClient(app)

    payload = {
        "model": "gpt-4o",
        "messages": [],
    }
    # Empty list is technically valid for Pydantic (List[ChatMessage] = []),
    # but the provider will probably error. At the HTTP level, 422 if
    # min_length is enforced, or 200/5xx once it hits the provider.
    # Since the schema has no min_length, this should pass validation.
    # We just verify it doesn't crash the router.
    with patch("mcpo.api.routers.completions._resolve_provider") as mock_resolve:
        mock_provider = MagicMock()
        mock_provider.name = "openai"
        mock_provider.complete = AsyncMock(return_value=FAKE_COMPLETION_RESPONSE)
        mock_resolve.return_value = mock_provider

        resp = client.post("/v1/chat/completions", json=payload)
        assert resp.status_code == 200


# ---------------------------------------------------------------------------
# 7. POST /v1/chat/completions — provider inference for different models
# ---------------------------------------------------------------------------


class TestProviderInference:
    """Test _infer_provider() for various model name patterns."""

    def test_claude_model(self):
        from mcpo.api.routers.completions import _infer_provider

        assert _infer_provider("claude-sonnet-4-20250514") == "anthropic"
        assert _infer_provider("claude-3-haiku-20240307") == "anthropic"

    def test_gemini_model(self):
        from mcpo.api.routers.completions import _infer_provider

        assert _infer_provider("gemini-2.0-flash") == "gemini"
        assert _infer_provider("gemini-pro") == "gemini"

    def test_openrouter_prefixed(self):
        from mcpo.api.routers.completions import _infer_provider

        assert _infer_provider("openrouter/auto") == "openrouter"
        assert _infer_provider("router/test") == "openrouter"

    def test_minimax_model(self):
        from mcpo.api.routers.completions import _infer_provider

        assert _infer_provider("minimax-chat") == "minimax"

    def test_gpt_falls_back_to_openai(self):
        from mcpo.api.routers.completions import _infer_provider

        assert _infer_provider("gpt-4o") == "openai"
        assert _infer_provider("gpt-3.5-turbo") == "openai"

    def test_unknown_model_defaults_openai(self):
        from mcpo.api.routers.completions import _infer_provider

        # When no OpenRouter key is set, slash-models with no key fall to openai
        with patch.dict("os.environ", {}, clear=False):
            # Remove OPENROUTER_API_KEY if present
            import os
            old = os.environ.pop("OPENROUTER_API_KEY", None)
            try:
                assert _infer_provider("unknown-model-xyz") == "openai"
            finally:
                if old is not None:
                    os.environ["OPENROUTER_API_KEY"] = old

    @patch.dict("os.environ", {"OPENROUTER_API_KEY": "sk-test"})
    def test_slash_model_with_openrouter_key(self):
        from mcpo.api.routers.completions import _infer_provider

        assert _infer_provider("moonshotai/kimi-k2:free") == "openrouter"


# ---------------------------------------------------------------------------
# 8. POST /v1/chat/completions — streaming response (mock SSE)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._resolve_provider")
async def test_post_chat_completions_stream(mock_resolve, tmp_path):
    mock_provider = MagicMock()
    mock_provider.name = "openai"

    # Simulate an async generator yielding SSE chunks
    async def fake_stream(payload):
        yield 'data: {"id":"chunk1","choices":[{"delta":{"content":"Hi"}}]}\n\n'
        yield 'data: {"id":"chunk2","choices":[{"delta":{},"finish_reason":"stop"}]}\n\n'
        yield "data: [DONE]\n\n"

    mock_provider.stream = fake_stream
    mock_resolve.return_value = mock_provider

    app = await _build_app(tmp_path)
    client = TestClient(app)

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "Hi"}],
        "stream": True,
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    assert "text/event-stream" in resp.headers.get("content-type", "")

    # Collect all SSE lines
    body = resp.text
    assert "data:" in body
    assert "[DONE]" in body


# ---------------------------------------------------------------------------
# 9. POST /v1/completions — legacy alias works
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._resolve_provider")
async def test_post_completions_legacy(mock_resolve, tmp_path):
    mock_provider = MagicMock()
    mock_provider.name = "openai"
    mock_provider.complete = AsyncMock(return_value=FAKE_COMPLETION_RESPONSE)
    mock_resolve.return_value = mock_provider

    app = await _build_app(tmp_path)
    client = TestClient(app)

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "test"}],
    }
    resp = client.post("/v1/completions", json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == "chatcmpl-test-123"
    mock_provider.complete.assert_awaited_once()


# ---------------------------------------------------------------------------
# 10. CompletionRequest model validation
# ---------------------------------------------------------------------------


class TestCompletionRequestValidation:
    """Unit tests for CompletionRequest pydantic model."""

    def test_stop_string_normalized_to_list(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stop="STOP",
        )
        assert req.stop == ["STOP"]

    def test_stop_list_preserved(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            stop=["END", "STOP"],
        )
        assert req.stop == ["END", "STOP"]

    def test_stop_none_ok(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.stop is None

    def test_temperature_valid_range(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=1.5,
        )
        assert req.temperature == 1.5

    def test_temperature_too_high(self):
        from mcpo.api.routers.completions import CompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                temperature=2.5,
            )

    def test_temperature_negative(self):
        from mcpo.api.routers.completions import CompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                temperature=-0.1,
            )

    def test_temperature_boundary_zero(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
        )
        assert req.temperature == 0.0

    def test_temperature_boundary_two(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            temperature=2.0,
        )
        assert req.temperature == 2.0

    def test_top_p_valid(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            top_p=0.9,
        )
        assert req.top_p == 0.9

    def test_top_p_out_of_range(self):
        from mcpo.api.routers.completions import CompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                top_p=1.5,
            )

    def test_max_tokens_positive(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            max_tokens=1024,
        )
        assert req.max_tokens == 1024

    def test_max_tokens_zero_invalid(self):
        from mcpo.api.routers.completions import CompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompletionRequest(
                model="test",
                messages=[{"role": "user", "content": "hi"}],
                max_tokens=0,
            )

    def test_model_required(self):
        from mcpo.api.routers.completions import CompletionRequest
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            CompletionRequest(
                messages=[{"role": "user", "content": "hi"}],
            )

    def test_stream_default_false(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
        )
        assert req.stream is False

    def test_extra_body_passthrough(self):
        from mcpo.api.routers.completions import CompletionRequest

        req = CompletionRequest(
            model="test",
            messages=[{"role": "user", "content": "hi"}],
            extra_body={"reasoning_split": 0.5},
        )
        assert req.extra_body == {"reasoning_split": 0.5}


# ---------------------------------------------------------------------------
# Additional edge-case tests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._resolve_provider")
async def test_provider_error_returns_502(mock_resolve, tmp_path):
    """When the provider raises CompletionProviderError, router returns 502."""
    from mcpo.api.routers.completions import CompletionProviderError

    mock_provider = MagicMock()
    mock_provider.name = "openai"
    mock_provider.complete = AsyncMock(side_effect=CompletionProviderError("upstream failed"))
    mock_resolve.return_value = mock_provider

    app = await _build_app(tmp_path)
    client = TestClient(app)

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "test"}],
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 502
    body = resp.json()
    # FastAPI HTTPException puts error in "detail" key
    detail = body.get("detail") or json.dumps(body)
    assert "upstream failed" in detail


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._list_models")
@patch("mcpo.api.routers.completions._filter_by_favorites", side_effect=lambda m: m)
async def test_models_empty_when_no_providers(mock_filter, mock_list, tmp_path):
    """When no provider keys are set, models list is empty."""
    mock_list.return_value = []
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.get("/v1/models")
    assert resp.status_code == 200
    assert resp.json()["data"] == []


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._list_models")
async def test_favorites_filter_applied(mock_list, tmp_path):
    """When favorites are set, only those models appear."""
    mock_list.return_value = _all_fake_models()

    app = await _build_app(tmp_path)
    client = TestClient(app)

    # Patch the state manager to return a favorites list
    with patch("mcpo.api.routers.completions._filter_by_favorites") as mock_filter:
        mock_filter.return_value = [{"id": "gpt-4o", "label": "OpenAI: gpt-4o", "provider": "openai"}]
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        assert len(resp.json()["data"]) == 1
        assert resp.json()["data"][0]["id"] == "gpt-4o"


@pytest.mark.asyncio
async def test_post_chat_completions_invalid_json(tmp_path):
    """Sending non-JSON body returns 422."""
    app = await _build_app(tmp_path)
    client = TestClient(app)

    resp = client.post(
        "/v1/chat/completions",
        content="not json",
        headers={"Content-Type": "application/json"},
    )
    assert resp.status_code == 422


@pytest.mark.asyncio
@patch("mcpo.api.routers.completions._resolve_provider")
async def test_post_chat_completions_with_tools(mock_resolve, tmp_path):
    """Request with tool definitions passes through to provider."""
    mock_provider = MagicMock()
    mock_provider.name = "openai"
    mock_provider.complete = AsyncMock(return_value=FAKE_COMPLETION_RESPONSE)
    mock_resolve.return_value = mock_provider

    app = await _build_app(tmp_path)
    client = TestClient(app)

    payload = {
        "model": "gpt-4o",
        "messages": [{"role": "user", "content": "What time is it?"}],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "get_time",
                    "description": "Get current time",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        "tool_choice": "auto",
    }
    resp = client.post("/v1/chat/completions", json=payload)
    assert resp.status_code == 200
    # Verify the provider was called with the payload including tools
    call_args = mock_provider.complete.call_args
    request_payload = call_args[0][0]
    assert request_payload.tools is not None
    assert len(request_payload.tools) == 1
    assert request_payload.tools[0].function.name == "get_time"
