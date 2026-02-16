"""
Tests for utility functions, middleware, and model layers.

Covers:
  - utils/config.py   – env var interpolation
  - utils/auth.py     – API key dependency & middleware
  - utils/main.py     – schema processing, tool response, envelope helpers
  - models/config.py  – ServerConfig / AppConfig validation
  - models/domain.py  – ServerInfo, ToolInfo, ServerRuntimeState, SessionMetrics
  - models/responses.py – ErrorDetail, SuccessResponse, HealthStatus, envelope compat
"""

import base64
import os
import pytest

from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import ValidationError

# ---------------------------------------------------------------------------
# Imports under test
# ---------------------------------------------------------------------------
from mcpo.utils.config import (
    interpolate_env_placeholders,
    interpolate_env_placeholders_in_config,
)
from mcpo.utils.auth import APIKeyMiddleware, get_verify_api_key
from mcpo.utils.main import (
    create_error_envelope,
    create_success_envelope,
    get_model_fields,
    process_tool_response,
)
from mcpo.models.config import AppConfig, ServerConfig
from mcpo.models.domain import (
    ServerConfiguration,
    ServerInfo,
    ServerRuntimeState,
    ServerType,
    SessionMetrics,
    ToolInfo,
    ToolState,
)
from mcpo.models.responses import (
    ErrorDetail,
    HealthStatus,
    SuccessResponse,
)

from mcp.types import CallToolResult, TextContent


# ===================================================================
# Config interpolation (utils/config.py)
# ===================================================================

class TestInterpolateEnvPlaceholders:
    """Tests for interpolate_env_placeholders."""

    def test_matching_env_var_replaced(self, monkeypatch):
        """${VARNAME} is replaced with the matching env var value."""
        monkeypatch.setenv("TEST_INTERP_VAR", "hello-world")
        result = interpolate_env_placeholders({"key": "${TEST_INTERP_VAR}"})
        assert result["key"] == "hello-world"

    def test_missing_env_var_becomes_empty(self):
        """${MISSING_VAR} with no env entry resolves to empty string."""
        os.environ.pop("MISSING_INTERP_VAR_XYZ", None)
        result = interpolate_env_placeholders({"key": "${MISSING_INTERP_VAR_XYZ}"})
        assert result["key"] == ""

    def test_no_placeholders_unchanged(self):
        """Strings without ${…} are returned as-is."""
        result = interpolate_env_placeholders({"key": "plain-value"})
        assert result["key"] == "plain-value"

    def test_non_string_values_preserved(self):
        """Non-string values (int, bool, list) pass through."""
        result = interpolate_env_placeholders({"a": 42, "b": True, "c": [1, 2]})
        assert result == {"a": 42, "b": True, "c": [1, 2]}

    def test_none_input_returns_empty_dict(self):
        """None instead of a dict returns {}."""
        assert interpolate_env_placeholders(None) == {}

    def test_only_uppercase_digits_underscore_matched(self, monkeypatch):
        """Lowercase letters in the placeholder name are now matched (BUG-021 fix)."""
        monkeypatch.setenv("myvar", "should-appear")
        result = interpolate_env_placeholders({"key": "${myvar}"})
        # regex [A-Za-z0-9_]+ now matches lowercase → value is interpolated
        assert result["key"] == "should-appear"


class TestInterpolateEnvPlaceholdersInConfig:
    """Tests for interpolate_env_placeholders_in_config."""

    def test_processes_nested_env_and_headers(self, monkeypatch):
        """env and headers blocks inside mcpServers are interpolated."""
        monkeypatch.setenv("DB_HOST", "localhost")
        monkeypatch.setenv("API_TOKEN", "tok-123")
        cfg = {
            "mcpServers": {
                "db": {
                    "env": {"HOST": "${DB_HOST}"},
                    "headers": {"Authorization": "Bearer ${API_TOKEN}"},
                }
            }
        }
        out = interpolate_env_placeholders_in_config(cfg)
        assert out["mcpServers"]["db"]["env"]["HOST"] == "localhost"
        assert out["mcpServers"]["db"]["headers"]["Authorization"] == "Bearer tok-123"

    def test_original_config_not_mutated(self, monkeypatch):
        """Deep-copies so the caller's dict is untouched."""
        monkeypatch.setenv("X_VAL", "replaced")
        cfg = {"mcpServers": {"s": {"env": {"K": "${X_VAL}"}}}}
        interpolate_env_placeholders_in_config(cfg)
        assert cfg["mcpServers"]["s"]["env"]["K"] == "${X_VAL}"

    def test_non_dict_passthrough(self):
        """Non-dict input is returned unchanged."""
        assert interpolate_env_placeholders_in_config("not-a-dict") == "not-a-dict"

    def test_servers_without_env_or_headers(self):
        """Server entries without env/headers are left alone."""
        cfg = {"mcpServers": {"s": {"command": "python"}}}
        out = interpolate_env_placeholders_in_config(cfg)
        assert out["mcpServers"]["s"]["command"] == "python"


# ===================================================================
# Auth (utils/auth.py)
# ===================================================================

class TestGetVerifyApiKey:
    """Tests for get_verify_api_key dependency factory."""

    def test_returns_callable_for_valid_key(self):
        """A non-empty key produces a callable dependency."""
        dep = get_verify_api_key("secret-key-42")
        assert callable(dep)

    def test_returns_callable_for_empty_string(self):
        """Even an empty string produces a callable (no None guard in code)."""
        dep = get_verify_api_key("")
        assert callable(dep)


class TestAPIKeyMiddleware:
    """Integration tests for APIKeyMiddleware via TestClient."""

    @staticmethod
    def _make_app(api_key: str) -> FastAPI:
        app = FastAPI()
        app.add_middleware(APIKeyMiddleware, api_key=api_key)

        @app.get("/test")
        def endpoint():
            return {"ok": True}

        return app

    def test_blocks_request_without_auth(self):
        """No Authorization header → 401."""
        client = TestClient(self._make_app("test-secret"))
        r = client.get("/test")
        assert r.status_code in (401, 403)

    def test_allows_correct_bearer_token(self):
        """Correct Bearer token → 200."""
        client = TestClient(self._make_app("test-secret"))
        r = client.get("/test", headers={"Authorization": "Bearer test-secret"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_rejects_wrong_bearer_token(self):
        """Wrong Bearer token → 403."""
        client = TestClient(self._make_app("test-secret"))
        r = client.get("/test", headers={"Authorization": "Bearer wrong-key"})
        assert r.status_code == 403

    def test_allows_correct_basic_auth(self):
        """Correct Basic auth (password matches api_key) → 200."""
        client = TestClient(self._make_app("test-secret"))
        creds = base64.b64encode(b"user:test-secret").decode()
        r = client.get("/test", headers={"Authorization": f"Basic {creds}"})
        assert r.status_code == 200
        assert r.json() == {"ok": True}

    def test_rejects_wrong_basic_auth(self):
        """Wrong Basic auth password → 403."""
        client = TestClient(self._make_app("test-secret"))
        creds = base64.b64encode(b"user:wrong-password").decode()
        r = client.get("/test", headers={"Authorization": f"Basic {creds}"})
        assert r.status_code == 403

    def test_options_request_passes_through(self):
        """OPTIONS (CORS preflight) is never blocked."""
        client = TestClient(self._make_app("test-secret"))
        r = client.options("/test")
        # Should not be 401/403
        assert r.status_code not in (401, 403)


# ===================================================================
# Schema processing (utils/main.py)
# ===================================================================

class TestGetModelFields:
    """Tests for get_model_fields."""

    def test_simple_string_property(self):
        """A string property yields a str field."""
        props = {"name": {"type": "string", "description": "The name"}}
        fields = get_model_fields("TestModel", props, required_fields=[])
        assert "name" in fields
        type_hint, field_info = fields["name"]
        assert type_hint is str

    def test_required_field_is_required(self):
        """Required fields are marked as required (no usable default)."""
        from pydantic_core import PydanticUndefined

        props = {"age": {"type": "integer", "description": "Age"}}
        fields = get_model_fields("TestModel", props, required_fields=["age"])
        _, field_info = fields["age"]
        assert field_info.default is PydanticUndefined

    def test_optional_field_has_none_default(self):
        """Non-required fields default to None."""
        props = {"tag": {"type": "string", "description": "Optional tag"}}
        fields = get_model_fields("TestModel", props, required_fields=[])
        _, field_info = fields["tag"]
        assert field_info.default is None

    def test_nested_object_creates_sub_model(self):
        """An 'object' property with nested properties creates a sub-model."""
        props = {
            "address": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "zip": {"type": "string"},
                },
                "required": ["city"],
            }
        }
        fields = get_model_fields("TestModel", props, required_fields=[])
        type_hint, _ = fields["address"]
        # Nested object should produce a Pydantic model (class), not a plain dict
        assert hasattr(type_hint, "__fields__") or hasattr(type_hint, "model_fields")

    def test_integer_and_boolean_types(self):
        """Integer and boolean are mapped correctly."""
        props = {
            "count": {"type": "integer"},
            "active": {"type": "boolean"},
        }
        fields = get_model_fields("TestModel", props, required_fields=[])
        assert fields["count"][0] is int
        assert fields["active"][0] is bool

    def test_array_type(self):
        """Array property produces List type."""
        props = {
            "tags": {
                "type": "array",
                "items": {"type": "string"},
            }
        }
        fields = get_model_fields("TestModel", props, required_fields=[])
        type_hint, _ = fields["tags"]
        # Should be List[str] — check origin
        import typing
        origin = getattr(type_hint, "__origin__", None)
        assert origin is list


class TestProcessToolResponse:
    """Tests for process_tool_response."""

    def test_text_content_returns_text(self):
        """TextContent is returned as plain string."""
        result = CallToolResult(content=[TextContent(type="text", text="hello")])
        out = process_tool_response(result)
        assert out == ["hello"]

    def test_text_content_json_parsed(self):
        """TextContent with valid JSON string is parsed to dict."""
        result = CallToolResult(
            content=[TextContent(type="text", text='{"key": "value"}')]
        )
        out = process_tool_response(result)
        assert out == [{"key": "value"}]

    def test_multiple_text_contents(self):
        """Multiple content items are all returned."""
        result = CallToolResult(
            content=[
                TextContent(type="text", text="first"),
                TextContent(type="text", text="second"),
            ]
        )
        out = process_tool_response(result)
        assert len(out) == 2
        assert out[0] == "first"
        assert out[1] == "second"

    def test_empty_content_returns_empty_list(self):
        """No content items → empty list."""
        result = CallToolResult(content=[])
        out = process_tool_response(result)
        assert out == []


class TestEnvelopeHelpers:
    """Tests for create_error_envelope / create_success_envelope in utils/main.py."""

    def test_error_envelope_structure(self):
        """Error envelope has ok=False, error.message, and timestamp."""
        env = create_error_envelope("something broke", code="ERR_42", data={"ctx": 1})
        assert env["ok"] is False
        assert env["error"]["message"] == "something broke"
        assert env["error"]["code"] == "ERR_42"
        assert env["error"]["data"] == {"ctx": 1}
        assert "timestamp" in env["error"]

    def test_error_envelope_without_optional_fields(self):
        """Code and data are omitted when not supplied."""
        env = create_error_envelope("fail")
        assert env["ok"] is False
        assert "code" not in env["error"]
        assert "data" not in env["error"]

    def test_success_envelope_structure(self):
        """Success envelope has ok=True and optional data/message."""
        env = create_success_envelope(data={"result": "ok"}, message="done")
        assert env["ok"] is True
        assert env["data"] == {"result": "ok"}
        assert env["message"] == "done"

    def test_success_envelope_minimal(self):
        """Bare success envelope only has ok=True."""
        env = create_success_envelope()
        assert env["ok"] is True
        assert "data" not in env
        assert "message" not in env


# ===================================================================
# Config models (models/config.py)
# ===================================================================

class TestServerConfig:
    """Tests for ServerConfig validation."""

    def test_stdio_valid(self):
        cfg = ServerConfig(name="local", type="stdio", command="python")
        assert cfg.name == "local"
        assert cfg.type == "stdio"
        assert cfg.command == "python"

    def test_stdio_missing_command_raises(self):
        with pytest.raises(ValidationError, match="stdio requires 'command'"):
            ServerConfig(name="local", type="stdio")

    def test_remote_sse_valid(self):
        cfg = ServerConfig(name="remote", type="sse", url="http://localhost:8080")
        assert cfg.type == "sse"
        assert str(cfg.url) == "http://localhost:8080/"

    def test_remote_streamable_http_valid(self):
        cfg = ServerConfig(
            name="remote2", type="streamable-http", url="http://localhost:9090/mcp"
        )
        assert cfg.type == "streamable-http"

    def test_remote_missing_url_raises(self):
        with pytest.raises(ValidationError, match="remote server requires 'url'"):
            ServerConfig(name="remote", type="sse")

    def test_args_optional(self):
        cfg = ServerConfig(name="s", type="stdio", command="node", args=["--flag"])
        assert cfg.args == ["--flag"]


class TestAppConfig:
    """Tests for AppConfig validation."""

    def test_unique_names_pass(self):
        cfg = AppConfig(
            servers=[
                ServerConfig(name="a", type="stdio", command="python"),
                ServerConfig(name="b", type="stdio", command="node"),
            ]
        )
        assert len(cfg.servers) == 2

    def test_duplicate_names_raise(self):
        with pytest.raises(ValidationError, match="unique"):
            AppConfig(
                servers=[
                    ServerConfig(name="dup", type="stdio", command="python"),
                    ServerConfig(name="dup", type="stdio", command="node"),
                ]
            )


# ===================================================================
# Domain models (models/domain.py)
# ===================================================================

class TestServerInfo:
    """Tests for ServerInfo."""

    def test_basic_creation(self):
        info = ServerInfo(name="my-server", version="1.0.0", description="A test server")
        assert info.name == "my-server"
        assert info.version == "1.0.0"
        assert info.description == "A test server"

    def test_optional_fields_default_none(self):
        info = ServerInfo(name="minimal")
        assert info.version is None
        assert info.description is None
        assert info.instructions is None


class TestToolInfo:
    """Tests for ToolInfo."""

    def test_creation(self):
        tool = ToolInfo(name="search", description="Search the web")
        assert tool.name == "search"
        assert tool.description == "Search the web"
        assert tool.input_schema == {}

    def test_with_schemas(self):
        tool = ToolInfo(
            name="calc",
            input_schema={"type": "object", "properties": {"x": {"type": "number"}}},
            output_schema={"type": "number"},
        )
        assert "properties" in tool.input_schema
        assert tool.output_schema == {"type": "number"}


class TestServerRuntimeState:
    """Tests for ServerRuntimeState enable/disable tool."""

    @staticmethod
    def _make_state() -> ServerRuntimeState:
        config = ServerConfiguration(
            name="test", server_type=ServerType.STDIO, command="python"
        )
        return ServerRuntimeState(name="test", config=config)

    def test_enable_tool(self):
        state = self._make_state()
        state.enable_tool("tool_a")
        assert state.tools["tool_a"] == ToolState.ENABLED

    def test_disable_tool(self):
        state = self._make_state()
        state.enable_tool("tool_a")
        state.disable_tool("tool_a")
        assert state.tools["tool_a"] == ToolState.DISABLED

    def test_get_enabled_tools(self):
        state = self._make_state()
        state.enable_tool("t1")
        state.enable_tool("t2")
        state.disable_tool("t2")
        assert state.get_enabled_tools() == ["t1"]

    def test_get_tool_count(self):
        state = self._make_state()
        state.enable_tool("t1")
        state.disable_tool("t2")
        counts = state.get_tool_count()
        assert counts["total"] == 2
        assert counts["enabled"] == 1
        assert counts["disabled"] == 1

    def test_set_connected(self):
        state = self._make_state()
        info = ServerInfo(name="test", version="2.0")
        state.set_connected(info)
        assert state.state.value == "connected"
        assert state.server_info.version == "2.0"
        assert state.connected_at is not None

    def test_set_error(self):
        state = self._make_state()
        state.set_error("connection refused")
        assert state.state.value == "error"
        assert state.last_error == "connection refused"


class TestSessionMetrics:
    """Tests for SessionMetrics.record_call and aggregation."""

    def test_record_successful_call(self):
        m = SessionMetrics()
        m.record_call("tool1", 150.0, success=True)
        assert m.tool_calls["tool1"] == 1
        assert m.latencies["tool1"] == [150.0]
        assert "tool1" not in m.errors

    def test_record_failed_call(self):
        m = SessionMetrics()
        m.record_call("tool1", 200.0, success=False)
        assert m.tool_calls["tool1"] == 1
        assert m.errors["tool1"] == 1

    def test_multiple_calls_accumulate(self):
        m = SessionMetrics()
        m.record_call("t", 100.0)
        m.record_call("t", 200.0)
        m.record_call("t", 300.0, success=False)
        assert m.tool_calls["t"] == 3
        assert len(m.latencies["t"]) == 3
        assert m.errors["t"] == 1

    def test_get_avg_latency(self):
        m = SessionMetrics()
        m.record_call("t", 100.0)
        m.record_call("t", 200.0)
        assert m.get_avg_latency("t") == 150.0

    def test_get_avg_latency_missing_tool(self):
        m = SessionMetrics()
        assert m.get_avg_latency("nonexistent") == 0.0

    def test_get_summary(self):
        m = SessionMetrics()
        m.record_call("a", 50.0)
        m.record_call("b", 100.0, success=False)
        summary = m.get_summary()
        assert summary["totalCalls"] == 2
        assert summary["totalErrors"] == 1
        assert summary["successRate"] == 0.5
        assert "a" in summary["perTool"]
        assert "b" in summary["perTool"]
        assert summary["uptimeMs"] >= 0


# ===================================================================
# Response models (models/responses.py)
# ===================================================================

class TestErrorDetail:
    """Tests for ErrorDetail.create."""

    def test_create_with_all_fields(self):
        err = ErrorDetail.create("bad request", code="BAD_REQ", data={"field": "x"})
        assert err.message == "bad request"
        assert err.code == "BAD_REQ"
        assert err.data == {"field": "x"}
        assert isinstance(err.timestamp, int)

    def test_create_minimal(self):
        err = ErrorDetail.create("oops")
        assert err.message == "oops"
        assert err.code is None
        assert err.data is None
        assert err.timestamp is not None


class TestSuccessResponse:
    """Tests for SuccessResponse.create."""

    def test_create_with_result_and_message(self):
        resp = SuccessResponse.create(result={"items": [1, 2]}, message="found")
        assert resp.ok is True
        assert resp.result == {"items": [1, 2]}
        assert resp.message == "found"

    def test_create_minimal(self):
        resp = SuccessResponse.create()
        assert resp.ok is True
        assert resp.result is None
        assert resp.data is None


class TestHealthStatus:
    """Tests for HealthStatus."""

    def test_creation(self):
        hs = HealthStatus(status="healthy", version="1.2.3")
        assert hs.status == "healthy"
        assert hs.version == "1.2.3"
        assert isinstance(hs.timestamp, int)

    def test_defaults(self):
        hs = HealthStatus()
        assert hs.status == "healthy"
        assert hs.version is None
        assert hs.uptime is None
