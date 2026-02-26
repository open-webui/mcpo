---
name: debug-connection
description: Diagnose MCP server connection issues in OpenHubUI
argument-hint: "[server-name]"
allowed-tools: Read, Bash, Grep, Glob
---

# Debug MCP Server Connection

Diagnose why an MCP server is not connecting or responding.

## Diagnostic Steps

1. **Check configuration** — Read `mcpo.json` and verify the server entry:
   - Is the server `"enabled": true`?
   - Is the `type` correct (stdio/sse/streamable-http)?
   - For stdio: does the command exist? (`which <command>`)
   - For remote: is the URL reachable? (`curl -s <url>`)

2. **Check state** — Read `mcpo_state.json` if it exists:
   - Is the server disabled in state?
   - Are any tools disabled?

3. **Check logs** — Query the log endpoint or check stderr output:
   ```bash
   curl -s http://localhost:8000/_meta/logs | python -m json.tool
   ```

4. **Check health** — Hit the health/meta endpoints:
   ```bash
   curl -s http://localhost:8000/_meta/servers
   curl -s http://localhost:8000/_meta/servers/{server-name}
   ```

5. **Check process** — For stdio servers, verify the subprocess:
   - Is the command installed? (`which npx`, `which uvx`, etc.)
   - Can it run standalone? (test the command directly)
   - Are required env vars set?

6. **Check ports** — Verify the server is running:
   ```bash
   curl -s http://localhost:8000/health
   lsof -i :8000 -i :8001
   ```

## Common Issues

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| "Connection refused" | Server not running | Start with `uv run mcpo serve` |
| "Tool not found" | Server disabled in state | Check `mcpo_state.json`, enable via UI |
| "Subprocess failed" | Missing command/package | Install the MCP server package |
| "Timeout" | Slow server startup | Increase timeout or check server health |
| "Auth error" | Missing/wrong API key | Check `MCPO_API_KEY` env var |

## Report

After diagnosis, provide:
- Root cause identified
- Specific fix steps
- Whether the issue is config, network, or server-side
