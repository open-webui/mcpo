<h2>mcpo</h2>

<p>mcpo is an MCP-to-OpenAPI proxy. It takes any MCP server and serves its tools as a standard OpenAPI HTTP server, with interactive docs and API key auth, so they work with <a href="https://docs.openwebui.com/features/extensibility/plugin/tools/openapi-servers/open-webui">Open WebUI</a> and any other app that speaks OpenAPI.</p>

![GitHub stars](https://img.shields.io/github/stars/open-webui/mcpo?style=social)
![GitHub forks](https://img.shields.io/github/forks/open-webui/mcpo?style=social)
[![PyPI](https://img.shields.io/pypi/v/mcpo)](https://pypi.org/project/mcpo/)
![GitHub last commit](https://img.shields.io/github/last-commit/open-webui/mcpo?color=red)
[![Discord](https://img.shields.io/badge/Discord-Open_WebUI-blue?logo=discord&logoColor=white)](https://discord.gg/5rJgQTnV4s)

[Read the docs →](https://docs.openwebui.com/features/extensibility/plugin/tools/openapi-servers/mcp)

### Why mcpo

Most MCP servers talk over stdio and run as a local process. An app in the browser or on another machine cannot reach them. mcpo runs the MCP server for you and puts plain HTTP in front of it. Every tool gets its own endpoint, its own OpenAPI schema and live docs at `/docs`, with no glue code.

Open WebUI connects to Streamable HTTP MCP servers [natively](https://docs.openwebui.com/features/extensibility/mcp). Use mcpo for stdio and SSE servers, or when a client only speaks OpenAPI.

### Installation

#### Python

```bash
# With uv installed
uvx mcpo --port 8000 --api-key "top-secret" -- your_mcp_server_command

# Or install and run with pip (Python 3.11+)
pip install mcpo
mcpo --port 8000 --api-key "top-secret" -- your_mcp_server_command
```

For example, with the MCP time server:

```bash
uvx mcpo --port 8000 --api-key "top-secret" -- uvx mcp-server-time --local-timezone=America/New_York
```

Then open [http://localhost:8000/docs](http://localhost:8000/docs) to try the tools live.

mcpo listens on `127.0.0.1` by default. Add `--host 0.0.0.0` when Open WebUI or another client runs on a different machine or in a container.

#### Docker

```bash
docker run -p 8000:8000 ghcr.io/open-webui/mcpo:main --api-key "top-secret" -- your_mcp_server_command
```

The image listens on all interfaces and runs as a non-root user.

#### Connect to Open WebUI

Add the mcpo URL as a tool server in Open WebUI. See the [Open WebUI integration guide](https://docs.openwebui.com/features/extensibility/plugin/tools/openapi-servers/open-webui) for the steps.

### Remote MCP Servers

mcpo also proxies MCP servers that already run over the network.

```bash
# SSE
mcpo --port 8000 --api-key "top-secret" --server-type "sse" -- http://127.0.0.1:8001/sse

# Streamable HTTP
mcpo --port 8000 --api-key "top-secret" --server-type "streamable-http" -- http://127.0.0.1:8002/mcp

# With headers for the remote server
mcpo --port 8000 --api-key "top-secret" --server-type "sse" --header '{"Authorization": "Bearer token"}' -- http://127.0.0.1:8001/sse
```

### Config File

Serve several MCP servers from one mcpo instance with a config file in the [Claude Desktop](https://modelcontextprotocol.io/quickstart/user) format.

```bash
mcpo --config /path/to/config.json

# Reload servers automatically when the file changes
mcpo --config /path/to/config.json --hot-reload
```

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=America/New_York"],
      "disabledTools": ["convert_time"] // Disable specific tools if needed
    },
    "mcp_sse": {
      "type": "sse", // Explicitly define type
      "url": "http://127.0.0.1:8001/sse",
      "headers": {
        "Authorization": "Bearer token",
        "X-Custom-Header": "value"
      }
    },
    "mcp_streamable_http": {
      "type": "streamable-http",
      "url": "http://127.0.0.1:8002/mcp"
    } // Streamable HTTP MCP Server
  }
}
```

Each server gets its own route, such as `http://localhost:8000/memory` and `http://localhost:8000/time`, with its own docs at `/memory/docs` and `/time/docs`.

| Key | Purpose |
| --- | --- |
| `command`, `args` | Command that starts a local (stdio) server |
| `env` | Extra environment variables for a local server |
| `cwd` | Folder a local server starts in |
| `type` | `sse` or `streamable-http` for a remote server |
| `url`, `headers` | Address of a remote server and the headers sent to it |
| `disabledTools` | Tools to hide from the generated API |
| `oauth` | OAuth settings, see [OAuth](#oauth) |

### Options

| Option | Purpose |
| --- | --- |
| `--host`, `--port` | Address to listen on (default `127.0.0.1:8000`) |
| `--api-key` | API key that clients send as a bearer token. Also read from `MCPO_API_KEY`, which keeps it out of the process list |
| `--strict-auth` | Also protect the docs and OpenAPI schema with the API key |
| `--server-type` | `stdio` (default), `sse` or `streamable-http` |
| `--header` | Headers in JSON for a remote server |
| `--config`, `--hot-reload` | Config file, and reload it when it changes |
| `--env`, `--env-path` | Environment variables for the MCP server |
| `--cors-allow-origins` | Allowed CORS origins (default `*`) |
| `--root-path` | Serve under a subpath behind a reverse proxy, such as `/api/mcpo` |
| `--path-prefix` | URL prefix for the server routes |
| `--ssl-certfile`, `--ssl-keyfile` | Serve over HTTPS |
| `--name`, `--description`, `--version` | Title, description and version shown in the docs |
| `--log-level` | `DEBUG`, `INFO`, `WARNING`, `ERROR` or `CRITICAL` |

Run `mcpo --help` for the full list. Setting `MCPO_API_KEY` through `--env` or `--env-path` has no effect.

### OAuth

mcpo supports OAuth 2.1 for Streamable HTTP servers that require it. It uses dynamic client registration by default, so most servers need only the server URL.

```json
{
  "mcpServers": {
    "oauth-protected-server": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp",
      "oauth": {
        "server_url": "http://localhost:8000"
      }
    }
  }
}
```

| Option | Purpose |
| --- | --- |
| `server_url` | OAuth server base URL (required) |
| `storage_type` | `file` (default, persistent) or `memory` (session only) |
| `callback_port` | Local port for the OAuth callback (default `3030`) |
| `use_loopback` | Open the browser for sign-in automatically (default `true`) |

On first connection mcpo registers a client if needed, opens the browser for sign-in, captures the callback and stores the tokens in `~/.mcpo/tokens/`. Later requests reuse them.

For a provider without dynamic client registration, register a client with the provider and put its client ID and secret into the client metadata.

```json
{
  "mcpServers": {
    "legacy-oauth-server": {
      "type": "streamable-http",
      "url": "http://api.example.com/mcp",
      "oauth": {
        "server_url": "http://api.example.com",
        "client_metadata": {
          "client_id": "your-client-id",
          "client_secret": "your-client-secret",
          "client_name": "My MCPO Client",
          "redirect_uris": ["http://localhost:3030/callback"]
        }
      }
    }
  }
}
```

Leave out `scope`, `authorization_endpoint` and `token_endpoint`. mcpo discovers them from the server's OAuth metadata. See the [OAuth guide](OAUTH_GUIDE.md) for details.

### Development

```bash
git clone https://github.com/open-webui/mcpo.git
cd mcpo
uv sync --dev

# Run the tests
uv run pytest

# Run mcpo from your checkout
uv run mcpo --port 8000 -- uvx mcp-server-time --local-timezone=America/New_York
```

### Support

[Open an issue](https://github.com/open-webui/mcpo/issues) or ask on [Discord](https://discord.gg/5rJgQTnV4s).

### Security

Report vulnerabilities privately through [GitHub security advisories](https://github.com/open-webui/mcpo/security/advisories/new). See [SECURITY.md](SECURITY.md) for the policy.

### Star History

<a href="https://star-history.com/#open-webui/mcpo&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open-webui/mcpo&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=open-webui/mcpo&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=open-webui/mcpo&type=Date" />
  </picture>
</a>

#### License

MIT. See [LICENSE](./LICENSE).

Created by [Timothy Jaeryang Baek](https://github.com/tjbck).
