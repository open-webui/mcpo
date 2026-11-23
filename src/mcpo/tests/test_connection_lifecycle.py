import asyncio
from contextlib import asynccontextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
import socket
import subprocess
import sys
import threading
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pytest

from mcpo.main import MCPConnectionManager


@pytest.mark.asyncio
async def test_stdio_session_can_reconnect_and_close_from_another_task():
    server = """
from mcp.server.fastmcp import FastMCP
mcp = FastMCP("lifecycle-test")
@mcp.tool()
def status() -> str:
    return "ok"
mcp.run()
"""
    manager = MCPConnectionManager(
        server_type="stdio",
        command=sys.executable,
        args=["-c", server],
        env={},
        headers=None,
        connection_timeout=None,
    )
    try:
        session, _ = await asyncio.wait_for(manager.ensure_initialized(), timeout=20)
        result = await asyncio.wait_for(
            manager.run_session_operation(session.call_tool("status", arguments={})),
            timeout=5,
        )
        assert result.content[0].text == "ok"
        replacement, _ = await asyncio.wait_for(manager.reconnect(), timeout=20)
        assert replacement is not session
        result = await asyncio.wait_for(
            manager.run_session_operation(
                replacement.call_tool("status", arguments={})
            ),
            timeout=5,
        )
        assert result.content[0].text == "ok"
    finally:
        await asyncio.wait_for(manager.close(), timeout=10)


@pytest.mark.asyncio
async def test_connection_is_opened_and_closed_in_the_same_task(monkeypatch):
    owners = []

    @asynccontextmanager
    async def transport():
        owners.append(asyncio.current_task())
        try:
            yield None, None
        finally:
            owners.append(asyncio.current_task())

    @asynccontextmanager
    async def session(*args):
        yield object()

    monkeypatch.setattr("mcpo.main.ClientSession", session)
    manager = MCPConnectionManager(
        server_type="streamable-http",
        command=None,
        args=[],
        env={},
        headers=None,
        connection_timeout=None,
    )
    monkeypatch.setattr(manager, "_create_client_context", transport)

    await asyncio.create_task(manager.get_session())
    await manager.close()

    assert len(owners) == 2
    assert owners[0] is owners[1]


@pytest.mark.asyncio
async def test_cancelling_startup_closes_the_transport(monkeypatch):
    opening = asyncio.Event()
    closed = asyncio.Event()

    @asynccontextmanager
    async def transport():
        try:
            opening.set()
            await asyncio.Event().wait()
            yield None, None
        finally:
            closed.set()

    manager = MCPConnectionManager(
        server_type="streamable-http",
        command=None,
        args=[],
        env={},
        headers=None,
        connection_timeout=None,
    )
    monkeypatch.setattr(manager, "_create_client_context", transport)
    startup = asyncio.create_task(manager.get_session())
    await opening.wait()
    startup.cancel()
    with pytest.raises(asyncio.CancelledError):
        await startup
    assert closed.is_set()
    assert manager.current_session is None


@pytest.mark.asyncio
async def test_cancelling_a_request_keeps_the_connection_open(monkeypatch):
    @asynccontextmanager
    async def transport():
        yield None, None

    @asynccontextmanager
    async def session(*args):
        yield object()

    monkeypatch.setattr("mcpo.main.ClientSession", session)
    manager = MCPConnectionManager(
        server_type="streamable-http",
        command=None,
        args=[],
        env={},
        headers=None,
        connection_timeout=None,
    )
    monkeypatch.setattr(manager, "_create_client_context", transport)
    original_session = await manager.get_session()
    started = asyncio.Event()
    finished = asyncio.Event()

    async def operation():
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            finished.set()

    try:
        request = asyncio.create_task(manager.run_session_operation(operation()))
        await started.wait()
        request.cancel()
        with pytest.raises(asyncio.CancelledError):
            await request
        assert finished.is_set()
        assert await manager.get_session() is original_session
    finally:
        await manager.close()


@pytest.mark.parametrize("fail_initialize", [False, True])
def test_http_failure_does_not_cancel_proxy_lifespan(tmp_path, fail_initialize):
    class Upstream(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        fail_calls = True

        def log_message(self, *args):
            pass

        def reply(self, code, payload=None):
            body = json.dumps(payload).encode() if payload is not None else b""
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(405)

        def do_POST(self):
            request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            method = request["method"]
            if method == "initialize":
                if fail_initialize:
                    self.reply(502, {"error": "upstream unavailable"})
                    return
                result = {
                    "protocolVersion": "2025-06-18",
                    "capabilities": {"tools": {}},
                    "serverInfo": {"name": "fault-injection", "version": "1.0"},
                }
            elif method.startswith("notifications/"):
                self.reply(202)
                return
            elif method == "tools/list":
                result = {
                    "tools": [
                        {
                            "name": "status",
                            "inputSchema": {"type": "object", "properties": {}},
                        }
                    ]
                }
            elif method == "tools/call":
                if self.fail_calls:
                    self.reply(502, {"error": "upstream unavailable"})
                    return
                result = {"content": [{"type": "text", "text": "recovered"}]}
            else:
                self.reply(400)
                return
            self.reply(200, {"jsonrpc": "2.0", "id": request["id"], "result": result})

    upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=upstream.serve_forever, daemon=True)
    thread.start()
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        proxy_port = sock.getsockname()[1]
    config = tmp_path / "config.json"
    config.write_text(
        json.dumps(
            {
                "mcpServers": {
                    "upstream": {
                        "type": "streamable-http",
                        "url": f"http://127.0.0.1:{upstream.server_port}/mcp",
                    }
                }
            }
        )
    )
    base_url = f"http://127.0.0.1:{proxy_port}/upstream"
    log_path = tmp_path / "proxy.log"
    try:
        with log_path.open("w", encoding="utf-8") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    "from mcpo import app; app()",
                    "--host",
                    "127.0.0.1",
                    "--config",
                    str(config),
                    "--port",
                    str(proxy_port),
                ],
                stdout=log,
                stderr=log,
                env={**os.environ, "PYTHONIOENCODING": "utf-8"},
            )
            try:
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    assert process.poll() is None, log_path.read_text(encoding="utf-8")
                    try:
                        with urlopen(
                            base_url + "/openapi.json", timeout=0.5
                        ) as response:
                            schema = json.load(response)
                        break
                    except (URLError, OSError):
                        time.sleep(0.05)
                else:
                    pytest.fail(log_path.read_text(encoding="utf-8"))

                if fail_initialize:
                    assert "/status" not in schema["paths"]
                else:
                    assert "/status" in schema["paths"]
                    request = Request(
                        base_url + "/status",
                        data=b"{}",
                        headers={"Content-Type": "application/json"},
                    )
                    with pytest.raises(HTTPError) as error:
                        urlopen(request, timeout=5)
                    assert error.value.code == 500
                    error.value.close()
                    Upstream.fail_calls = False
                    with urlopen(request, timeout=5) as response:
                        assert response.status == 200
                        assert "recovered" in response.read().decode()
                with urlopen(base_url + "/docs", timeout=2) as response:
                    assert response.status == 200
                assert "Attempted to exit cancel scope" not in log_path.read_text(
                    encoding="utf-8"
                )
            finally:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=10)
    finally:
        upstream.shutdown()
        upstream.server_close()
        thread.join(timeout=5)
