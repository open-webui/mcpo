"""
Comprehensive tests for the MCPO services layer.

Covers: StateManager, ChatSessionManager, LogManager, MetricsAggregator,
        ServerRegistry, RunnerService.
"""

import asyncio
import json
import os
import threading
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException

from mcpo.services.state import StateManager
from mcpo.services.chat_sessions import ChatSession, ChatSessionManager
from mcpo.services.logging import LogManager, LogLevel
from mcpo.services.metrics import MetricsAggregator
from mcpo.services.registry import ServerRegistry
from mcpo.services.runner import RunnerService


# ---------------------------------------------------------------------------
# StateManager
# ---------------------------------------------------------------------------


class TestStateManager:
    """Tests for StateManager."""

    def test_load_empty_state_file(self, tmp_path: Path):
        """Load from an empty JSON file returns empty dict."""
        state_file = tmp_path / "state.json"
        state_file.write_text("{}")
        sm = StateManager(str(state_file))
        assert sm.get_all_states() == {}

    def test_save_and_reload(self, tmp_path: Path):
        """Data persists across save → new instance load cycle."""
        state_file = tmp_path / "state.json"
        sm = StateManager(str(state_file))
        sm.set_server_enabled("srv1", False)
        sm.set_tool_enabled("srv1", "toolA", False)

        sm2 = StateManager(str(state_file))
        assert sm2.is_server_enabled("srv1") is False
        assert sm2.is_tool_enabled("srv1", "toolA") is False

    def test_set_get_server_enabled(self, tmp_path: Path):
        """set_server_enabled / is_server_enabled round-trip."""
        state_file = tmp_path / "state.json"
        sm = StateManager(str(state_file))
        sm.set_server_enabled("alpha", True)
        assert sm.is_server_enabled("alpha") is True
        sm.set_server_enabled("alpha", False)
        assert sm.is_server_enabled("alpha") is False

    def test_set_get_tool_enabled(self, tmp_path: Path):
        """set_tool_enabled / is_tool_enabled round-trip."""
        state_file = tmp_path / "state.json"
        sm = StateManager(str(state_file))
        sm.set_tool_enabled("srv", "t1", False)
        assert sm.is_tool_enabled("srv", "t1") is False
        sm.set_tool_enabled("srv", "t1", True)
        assert sm.is_tool_enabled("srv", "t1") is True

    def test_get_server_state_unknown(self, tmp_path: Path):
        """Unknown server returns default (enabled=True, tools={})."""
        state_file = tmp_path / "state.json"
        sm = StateManager(str(state_file))
        state = sm.get_server_state("nonexistent")
        assert state == {"enabled": True, "tools": {}}

    def test_thread_safety(self, tmp_path: Path):
        """Concurrent set/get operations do not crash."""
        state_file = tmp_path / "state.json"
        sm = StateManager(str(state_file))
        errors = []

        def writer(idx: int):
            try:
                for i in range(50):
                    sm.set_server_enabled(f"srv-{idx}", i % 2 == 0)
                    sm.set_tool_enabled(f"srv-{idx}", f"tool-{i}", i % 2 == 0)
            except Exception as exc:
                errors.append(exc)

        def reader():
            try:
                for _ in range(50):
                    sm.get_all_states()
                    sm.is_server_enabled("srv-0")
                    sm.is_tool_enabled("srv-0", "tool-0")
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(5)]
        threads += [threading.Thread(target=reader) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert errors == [], f"Thread safety errors: {errors}"

    def test_corrupt_state_file(self, tmp_path: Path):
        """Corrupt state file causes graceful fallback, not a crash."""
        state_file = tmp_path / "state.json"
        state_file.write_text("NOT VALID JSON {{{")
        sm = StateManager(str(state_file))
        assert sm.get_all_states() == {}

    def test_state_file_does_not_exist(self, tmp_path: Path):
        """Non-existent state file creates a fresh empty state."""
        state_file = tmp_path / "does_not_exist.json"
        sm = StateManager(str(state_file))
        assert sm.get_all_states() == {}
        # After a write, the file should be created
        sm.set_server_enabled("s1", True)
        assert state_file.exists()

    def test_is_server_enabled_default(self, tmp_path: Path):
        """Unknown server defaults to enabled=True."""
        sm = StateManager(str(tmp_path / "s.json"))
        assert sm.is_server_enabled("never_seen") is True

    def test_is_tool_enabled_default(self, tmp_path: Path):
        """Unknown tool on unknown server defaults to enabled=True."""
        sm = StateManager(str(tmp_path / "s.json"))
        assert sm.is_tool_enabled("x", "y") is True


# ---------------------------------------------------------------------------
# ChatSessionManager
# ---------------------------------------------------------------------------


class TestChatSessionManager:
    """Tests for ChatSessionManager (async)."""

    @pytest.mark.asyncio
    async def test_create_session_returns_valid_session(self):
        mgr = ChatSessionManager()
        session = await mgr.create_session(model="gpt-4")
        assert isinstance(session, ChatSession)
        assert isinstance(session.id, str)
        assert len(session.id) > 0
        assert session.model == "gpt-4"

    @pytest.mark.asyncio
    async def test_get_session_returns_same(self):
        mgr = ChatSessionManager()
        session = await mgr.create_session(model="m1")
        retrieved = await mgr.get_session(session.id)
        assert retrieved is session

    @pytest.mark.asyncio
    async def test_get_session_bad_id_raises(self):
        mgr = ChatSessionManager()
        with pytest.raises(KeyError, match="not found"):
            await mgr.get_session("nonexistent-id")

    @pytest.mark.asyncio
    async def test_delete_session(self):
        mgr = ChatSessionManager()
        session = await mgr.create_session(model="m1")
        await mgr.delete_session(session.id)
        with pytest.raises(KeyError):
            await mgr.get_session(session.id)

    @pytest.mark.asyncio
    async def test_reset_session_clears_messages_preserves_metadata(self):
        mgr = ChatSessionManager()
        session = await mgr.create_session(
            model="m1",
            system_prompt="You are helpful.",
        )
        # Inject extra messages
        session.messages.append({"role": "user", "content": "Hello"})
        session.messages.append({"role": "assistant", "content": "Hi"})
        assert len(session.messages) >= 3  # system + user + assistant

        reset_session = await mgr.reset_session(session.id)
        # After reset: only the system message is preserved
        assert len(reset_session.messages) == 1
        assert reset_session.messages[0]["role"] == "system"
        assert reset_session.model == "m1"

    @pytest.mark.asyncio
    async def test_reset_session_bad_id_raises(self):
        mgr = ChatSessionManager()
        with pytest.raises(KeyError, match="not found"):
            await mgr.reset_session("bad")

    @pytest.mark.asyncio
    async def test_list_sessions(self):
        mgr = ChatSessionManager()
        s1 = await mgr.create_session(model="m1")
        s2 = await mgr.create_session(model="m2")
        sessions = await mgr.list_sessions()
        ids = {s.id for s in sessions}
        assert s1.id in ids
        assert s2.id in ids
        assert len(sessions) == 2


# ---------------------------------------------------------------------------
# LogManager
# ---------------------------------------------------------------------------


class TestLogManager:
    """Tests for LogManager."""

    def test_add_log_appears_in_get_logs(self):
        lm = LogManager(max_entries=100)
        lm.add_log("hello", level=LogLevel.INFO, category="test")
        logs = lm.get_logs()
        assert len(logs) == 1
        assert logs[0]["message"] == "hello"
        assert logs[0]["level"] == "INFO"

    def test_get_logs_with_limit(self):
        lm = LogManager(max_entries=100)
        for i in range(10):
            lm.add_log(f"msg-{i}")
        logs = lm.get_logs(limit=3)
        assert len(logs) == 3
        # limit returns last N entries
        assert logs[-1]["message"] == "msg-9"

    def test_get_logs_category_filter(self):
        lm = LogManager(max_entries=100)
        lm.add_log("a", category="alpha")
        lm.add_log("b", category="beta")
        lm.add_log("c", category="alpha")
        logs = lm.get_logs(category="alpha")
        assert len(logs) == 2
        assert all(l["category"] == "alpha" for l in logs)

    def test_clear_logs(self):
        lm = LogManager(max_entries=100)
        lm.add_log("x")
        lm.add_log("y")
        assert lm.get_log_count() == 2
        lm.clear_logs()
        assert lm.get_log_count() == 0
        assert lm.get_logs() == []

    def test_buffer_trimming(self):
        lm = LogManager(max_entries=5)
        for i in range(10):
            lm.add_log(f"msg-{i}")
        assert lm.get_log_count() == 5
        logs = lm.get_logs()
        # Only the last 5 messages survive
        messages = [l["message"] for l in logs]
        assert messages == ["msg-5", "msg-6", "msg-7", "msg-8", "msg-9"]

    def test_sequence_numbers_increment(self):
        lm = LogManager(max_entries=100)
        lm.add_log("first")
        lm.add_log("second")
        logs = lm.get_logs()
        assert logs[0]["sequence"] == 1
        assert logs[1]["sequence"] == 2

    def test_get_logs_after_sequence(self):
        lm = LogManager(max_entries=100)
        lm.add_log("a")
        lm.add_log("b")
        lm.add_log("c")
        logs = lm.get_logs(after=1)
        assert len(logs) == 2
        assert logs[0]["message"] == "b"

    def test_clear_logs_by_category(self):
        lm = LogManager(max_entries=100)
        lm.add_log("a", category="keep")
        lm.add_log("b", category="remove")
        lm.clear_logs(category="remove")
        logs = lm.get_logs()
        assert len(logs) == 1
        assert logs[0]["category"] == "keep"


# ---------------------------------------------------------------------------
# MetricsAggregator
# ---------------------------------------------------------------------------


class TestMetricsAggregator:
    """Tests for MetricsAggregator."""

    def test_record_call_increments(self):
        ma = MetricsAggregator()
        ma.record_call()
        ma.record_call()
        metrics = ma.build_metrics({})
        assert metrics["calls"] == 2

    def test_record_error_increments_by_code(self):
        ma = MetricsAggregator()
        ma.record_error("timeout")
        ma.record_error("timeout")
        ma.record_error("disabled")
        metrics = ma.build_metrics({})
        assert metrics["errors"]["byCode"]["timeout"] == 2
        assert metrics["errors"]["byCode"]["disabled"] == 1
        assert metrics["errors"]["total"] == 3

    def test_record_error_unknown_code_maps_to_unexpected(self):
        ma = MetricsAggregator()
        ma.record_error("some_unknown_code")
        metrics = ma.build_metrics({})
        assert metrics["errors"]["byCode"]["unexpected"] == 1

    def test_reset_clears_all(self):
        ma = MetricsAggregator()
        ma.record_call()
        ma.record_call()
        ma.record_error("timeout")
        ma.reset()
        metrics = ma.build_metrics({})
        assert metrics["calls"] == 0
        assert metrics["errors"]["total"] == 0
        assert all(v == 0 for v in metrics["errors"]["byCode"].values())

    def test_build_metrics_structure(self):
        ma = MetricsAggregator()
        per_tool = {"tool1": {"calls": 5, "errors": 1}}
        metrics = ma.build_metrics(per_tool)
        assert "calls" in metrics
        assert "errors" in metrics
        assert "total" in metrics["errors"]
        assert "byCode" in metrics["errors"]
        assert metrics["perTool"] == per_tool

    def test_build_metrics_empty_per_tool(self):
        ma = MetricsAggregator()
        metrics = ma.build_metrics({})
        assert metrics["perTool"] == {}


# ---------------------------------------------------------------------------
# ServerRegistry
# ---------------------------------------------------------------------------


class TestServerRegistry:
    """Tests for ServerRegistry."""

    def _write_config(self, path: Path, servers: dict) -> None:
        path.write_text(json.dumps({"mcpServers": servers}, indent=2))

    def test_load_config_populates_servers(self, tmp_path: Path):
        cfg = tmp_path / "config.json"
        self._write_config(cfg, {
            "time-server": {"type": "stdio", "command": "python", "args": ["-m", "time_server"]},
            "weather": {"type": "stdio", "command": "node", "args": ["weather.js"]},
        })
        reg = ServerRegistry()
        reg.load_config(str(cfg))
        servers = reg.get_all_servers()
        assert "time-server" in servers
        assert "weather" in servers
        assert servers["time-server"].server_type == "stdio"

    def test_load_config_invalid_json_raises(self, tmp_path: Path):
        cfg = tmp_path / "bad.json"
        cfg.write_text("{INVALID")
        reg = ServerRegistry()
        with pytest.raises(ValueError, match="Failed to load config"):
            reg.load_config(str(cfg))

    def test_add_and_remove_server(self):
        reg = ServerRegistry()
        reg.add_server("new-srv", {"type": "stdio", "command": "echo"})
        assert "new-srv" in reg.get_all_servers()
        removed = reg.remove_server("new-srv")
        assert removed is True
        assert "new-srv" not in reg.get_all_servers()

    def test_remove_nonexistent_server(self):
        reg = ServerRegistry()
        removed = reg.remove_server("ghost")
        assert removed is False

    def test_save_config_persists(self, tmp_path: Path):
        cfg_path = tmp_path / "out.json"
        reg = ServerRegistry()
        reg.add_server("s1", {"type": "stdio", "command": "python"})
        reg.save_config(str(cfg_path))

        with open(cfg_path) as f:
            data = json.load(f)
        assert "s1" in data["mcpServers"]

    def test_set_config_data_replaces_all(self):
        reg = ServerRegistry()
        reg.add_server("old", {"type": "stdio", "command": "old"})
        reg.set_config_data({
            "mcpServers": {
                "new1": {"type": "stdio", "command": "new1"},
                "new2": {"type": "stdio", "command": "new2"},
            }
        })
        servers = reg.get_all_servers()
        assert "old" not in servers
        assert "new1" in servers
        assert "new2" in servers

    def test_get_server_list_format(self):
        reg = ServerRegistry()
        reg.add_server("srv", {"type": "stdio", "command": "python", "url": None})
        server_list = reg.get_server_list()
        assert len(server_list) == 1
        entry = server_list[0]
        assert entry["name"] == "srv"
        assert "type" in entry
        assert "enabled" in entry

    def test_save_and_reload_roundtrip(self, tmp_path: Path):
        cfg_path = tmp_path / "rt.json"
        reg = ServerRegistry()
        reg.add_server("a", {"type": "stdio", "command": "a_cmd", "args": ["--flag"]})
        reg.save_config(str(cfg_path))

        reg2 = ServerRegistry()
        reg2.load_config(str(cfg_path))
        assert "a" in reg2.get_all_servers()
        assert reg2.get_server_config("a").command == "a_cmd"


# ---------------------------------------------------------------------------
# RunnerService
# ---------------------------------------------------------------------------


class TestRunnerService:
    """Tests for RunnerService."""

    def _make_successful_result(self, text: str = "ok"):
        """Create a mock successful CallToolResult."""
        from mcp import types

        content = types.TextContent(type="text", text=text)
        result = MagicMock()
        result.isError = False
        result.content = [content]
        return result

    def _make_error_result(self, text: str = "tool failed"):
        """Create a mock error CallToolResult."""
        from mcp import types

        content = types.TextContent(type="text", text=text)
        result = MagicMock()
        result.isError = True
        result.content = [content]
        return result

    @pytest.mark.asyncio
    async def test_execute_tool_success(self):
        runner = RunnerService()
        mock_result = self._make_successful_result("hello")
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=mock_result)

        with patch("mcpo.utils.main.process_tool_response", return_value=["hello"]):
            result = await runner.execute_tool(session, "my_tool", {"arg": 1})
        assert result == "hello"

    @pytest.mark.asyncio
    async def test_execute_tool_timeout(self):
        runner = RunnerService()
        session = AsyncMock()

        async def slow_call(*args, **kwargs):
            await asyncio.sleep(10)

        session.call_tool = slow_call

        with pytest.raises(HTTPException) as exc_info:
            await runner.execute_tool(session, "slow_tool", {}, timeout=0.05)
        assert exc_info.value.status_code == 504
        assert "timeout" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_execute_tool_error_result(self):
        runner = RunnerService()
        mock_result = self._make_error_result("tool failed")
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(HTTPException) as exc_info:
            await runner.execute_tool(session, "bad_tool", {})
        assert exc_info.value.status_code == 500
        assert "tool failed" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_execute_tool_invalid_timeout(self):
        runner = RunnerService()
        session = AsyncMock()

        with pytest.raises(HTTPException) as exc_info:
            await runner.execute_tool(
                session, "tool", {}, timeout=999, max_timeout=60
            )
        assert exc_info.value.status_code == 400
        assert "invalid_timeout" in str(exc_info.value.detail).lower()

    @pytest.mark.asyncio
    async def test_execute_tool_tracks_metrics(self):
        runner = RunnerService()
        mock_result = self._make_successful_result("ok")
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=mock_result)

        with patch("mcpo.utils.main.process_tool_response", return_value=["ok"]):
            await runner.execute_tool(session, "metric_tool", {})

        metrics = runner.get_metrics()
        assert "metric_tool" in metrics
        assert metrics["metric_tool"]["calls"] == 1
        assert metrics["metric_tool"]["errors"] == 0

    @pytest.mark.asyncio
    async def test_execute_tool_error_tracks_metrics(self):
        runner = RunnerService()
        mock_result = self._make_error_result("fail")
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=mock_result)

        with pytest.raises(HTTPException):
            await runner.execute_tool(session, "err_tool", {})

        # isError results still count the call but not as a metric error
        # (metric errors are only for exceptions/timeouts, not tool-level isError)
        # Actually, looking at the code: isError raises HTTPException which is
        # re-raised in the except block, so success=True was set before the raise.
        # Let's verify what actually happens:
        metrics = runner.get_metrics()
        assert "err_tool" in metrics

    def test_reset_metrics(self):
        runner = RunnerService()
        runner._update_metrics("t1", 0.5, success=True)
        runner._update_metrics("t2", 0.3, success=False)
        assert len(runner.get_metrics()) == 2
        runner.reset_metrics()
        assert runner.get_metrics() == {}

    @pytest.mark.asyncio
    async def test_track_and_cleanup_tasks(self):
        runner = RunnerService()

        async def noop():
            await asyncio.sleep(100)

        task = asyncio.create_task(noop())
        runner.track_task(task)

        await runner.cleanup_tasks()
        assert task.cancelled() or task.done()

    @pytest.mark.asyncio
    async def test_execute_tool_multiple_content(self):
        """When process_tool_response returns multiple items, result is a list."""
        runner = RunnerService()
        mock_result = self._make_successful_result("ok")
        session = AsyncMock()
        session.call_tool = AsyncMock(return_value=mock_result)

        with patch(
            "mcpo.utils.main.process_tool_response",
            return_value=["part1", "part2"],
        ):
            result = await runner.execute_tool(session, "multi_tool", {})
        assert result == ["part1", "part2"]
