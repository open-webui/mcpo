---
name: serve
description: Start the OpenHubUI development server with common configurations
argument-hint: "[--hot-reload] [--port PORT]"
disable-model-invocation: true
allowed-tools: Bash, Read
---

# Start OpenHubUI Dev Server

Start the OpenHubUI (MCPO) development server.

## Default Command

```bash
uv run mcpo serve --config mcpo.json --hot-reload
```

## Options

Apply any of these based on user arguments ($ARGUMENTS):

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `0.0.0.0` | Host address |
| `--port` / `-p` | `8000` | REST/Admin UI port |
| `--config` / `-c` | `mcpo.json` | Config file path |
| `--hot-reload` | off | Watch config for changes |
| `--api-key` / `-k` | none | Require API key auth |
| `--strict-auth` | off | Protect all endpoints |
| `--ssl-certfile` | none | SSL certificate |
| `--ssl-keyfile` | none | SSL private key |

## Ports

- **8000** — REST/OpenAPI endpoints + Admin UI at `/ui`
- **8001** — MCP streamable-http endpoint at `/mcp`

## Steps

1. Check that `mcpo.json` (or the specified config) exists
2. Ensure dependencies are installed (`uv sync` if needed)
3. Run the serve command with the requested options
4. Inform the user about the available endpoints
