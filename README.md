# ⚡️ mcpo

Expose any MCP tool as an OpenAPI-compatible HTTP server—instantly.

mcpo is a dead-simple proxy that takes an MCP server command and makes it accessible via standard RESTful OpenAPI, so your tools "just work" with LLM agents and apps expecting OpenAPI servers.

No custom protocol. No glue code. No hassle.

## 🤔 Why Use mcpo Instead of Native MCP?

MCP servers usually speak over raw stdio, which is:

- 🔓 Inherently insecure
- ❌ Incompatible with most tools
- 🧩 Missing standard features like docs, auth, error handling, etc.

mcpo solves all of that—without extra effort:

- ✅ Works instantly with OpenAPI tools, SDKs, and UIs
- 🛡 Adds security, stability, and scalability using trusted web standards
- 🧠 Auto-generates interactive docs for every tool, no config needed
- 🔌 Uses pure HTTP—no sockets, no glue code, no surprises

What feels like "one more step" is really fewer steps with better outcomes.

mcpo makes your AI tools usable, secure, and interoperable—right now, with zero hassle.

## 🚀 Quick Usage

We recommend using uv for lightning-fast startup and zero config.

```bash
uvx mcpo --port 8000 --api-key "top-secret" -- your_mcp_server_command
```

Or, if you’re using Python:

```bash
pip install mcpo
mcpo --port 8000 --api-key "top-secret" -- your_mcp_server_command
```

To use an SSE-compatible MCP server, simply specify the server type and endpoint:

```bash
mcpo --port 8000 --api-key "top-secret" --server-type "sse" -- http://127.0.0.1:8001/sse
```

You can also provide headers for the SSE connection:

```bash
mcpo --port 8000 --api-key "top-secret" --server-type "sse" --header '{"Authorization": "Bearer token", "X-Custom-Header": "value"}' -- http://127.0.0.1:8001/sse
```

To use a Streamable HTTP-compatible MCP server, specify the server type and endpoint:

```bash
mcpo --port 8000 --api-key "top-secret" --server-type "streamable-http" -- http://127.0.0.1:8002/mcp
```

You can also run mcpo via Docker with no installation:

```bash
docker run -p 8000:8000 ghcr.io/open-webui/mcpo:main --api-key "top-secret" -- your_mcp_server_command
```

Example:

```bash
uvx mcpo --port 8000 --api-key "top-secret" -- uvx mcp-server-time --local-timezone=America/New_York
```

That’s it. Your MCP tool is now available at http://localhost:8000 with a generated OpenAPI schema — test it live at [http://localhost:8000/docs](http://localhost:8000/docs).

🤝 **To integrate with Open WebUI after launching the server, check our [docs](https://docs.openwebui.com/openapi-servers/open-webui/).**

### 🔄 Using a Config File

You can serve multiple MCP tools via a single config file that follows the [Claude Desktop](https://modelcontextprotocol.io/quickstart/user) format.

Enable hot-reload mode with `--hot-reload` to automatically watch your config file for changes and reload servers without downtime:

Start via:

```bash
mcpo --config /path/to/config.json
```

Or with hot-reload enabled:

```bash
mcpo --config /path/to/config.json --hot-reload
```

Example config.json:

```json
{
  "mcpServers": {
    "memory": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-memory"]
    },
    "time": {
      "command": "uvx",
      "args": ["mcp-server-time", "--local-timezone=America/New_York"]
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

Each tool will be accessible under its own unique route, e.g.:
- http://localhost:8000/memory
- http://localhost:8000/time

Each with a dedicated OpenAPI schema and proxy handler. Access full schema UI at: `http://localhost:8000/<tool>/docs`  (e.g. /memory/docs, /time/docs)

### 🔐 OAuth 2.1 Authentication

mcpo supports OAuth 2.1 authentication for MCP servers that require it. The implementation defaults to **dynamic client registration**, so most servers only need minimal configuration. mcpo supports both **single-user** and **multi-user** OAuth modes.

#### Single-User OAuth (Traditional)

For single-user scenarios where one user authenticates per mcpo instance:

```json
{
  "mcpServers": {
    "oauth-protected-server": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp",
      "oauth": {
        "server_url": "http://localhost:8000",
        "multi_user": false
      }
    }
  }
}
```

#### Multi-User OAuth (Web-Based)

For multi-user scenarios where multiple users can authenticate through a web interface:

```json
{
  "mcpServers": {
    "multi-user-server": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp",
      "oauth": {
        "server_url": "http://localhost:8000",
        "multi_user": true,
        "session_timeout_minutes": 30
      }
    }
  }
}
```

#### OpenWebUI Integration with JWT Validation

For integration with OpenWebUI, add JWT validation to verify session tokens:

```json
{
  "mcpServers": {
    "openwebui-server": {
      "type": "streamable-http", 
      "url": "http://localhost:8000/mcp",
      "oauth": {
        "server_url": "http://localhost:8000",
        "multi_user": true,
        "webui_secret_key": "your-openwebui-secret-key",
        "session_timeout_minutes": 60
      }
    }
  }
}
```

When `webui_secret_key` is configured, MCPO validates JWT tokens using:
- **HS256 signature verification** with the shared secret
- **Token expiration checking** to reject expired tokens  
- **Graceful fallback** to no verification if secret not provided

With multi-user OAuth enabled:
- Each server gets dedicated OAuth endpoints: `/oauth/{server_name}/authorize`, `/oauth/{server_name}/callback`, `/oauth/{server_name}/status`
- Users authenticate via web browser at the server's root URL
- Secure session management with UUID-based user identification
- Per-user token isolation and automatic session cleanup
- Dynamic OpenAPI schema generation based on authentication status

#### OAuth Configuration Options

**Basic Options:**
- `server_url` (required): OAuth server base URL
- `multi_user`: Enable multi-user mode (default: true)
- `webui_secret_key`: Shared secret for JWT validation (for OpenWebUI integration)
- `storage_type`: "file" (persistent) or "memory" (session-only, default: "file")
- `session_timeout_minutes`: Session timeout for multi-user mode (default: 30)
- `callback_port`: Local port for OAuth callback in single-user mode (default: 3030)
- `use_loopback`: Auto-open browser for auth in single-user mode (default: true)

**Advanced Options (rarely needed):**
For servers that don't support dynamic client registration, you can specify static client metadata:

```json
{
  "mcpServers": {
    "legacy-oauth-server": {
      "type": "streamable-http", 
      "url": "http://api.example.com/mcp",
      "oauth": {
        "server_url": "http://api.example.com",
        "multi_user": false,
        "client_metadata": {
          "client_name": "My MCPO Client",
          "redirect_uris": ["http://localhost:3030/callback"]
        }
      }
    }
  }
}
```

> **Note**: Avoid setting `scope`, `authorization_endpoint`, or `token_endpoint` in the config. These are automatically discovered from the server's OAuth metadata during the dynamic registration flow.

#### Authentication Flow

**Single-User Mode:**
1. Perform dynamic client registration (if supported)
2. Open your browser for authorization
3. Capture the OAuth callback automatically  
4. Store tokens securely (in `~/.mcpo/tokens/` for file storage)
5. Use tokens for all subsequent requests

**Multi-User Mode:**
1. Users visit the server URL in their browser
2. Automatic redirect to OAuth authorization endpoint
3. After authorization, users can access MCP tools through the web interface
4. Secure session cookies maintain authentication state
5. Per-user token storage with automatic cleanup of expired sessions

OAuth is supported for `streamable-http` server types. The multi-user implementation follows RFC 6749 (OAuth 2.0), RFC 7636 (PKCE), RFC 7591 (Dynamic Client Registration), and RFC 9728 (OAuth Protected Resource Discovery). See [OAUTH_GUIDE.md](OAUTH_GUIDE.md) for detailed documentation.

## 🔧 Requirements

- Python 3.8+
- uv (optional, but highly recommended for performance + packaging)

## 🛠️ Development & Testing

To contribute or run tests locally:

1.  **Set up the environment:**
    ```bash
    # Clone the repository
    git clone https://github.com/open-webui/mcpo.git
    cd mcpo

    # Install dependencies (including dev dependencies)
    uv sync --dev
    ```

2.  **Run tests:**
    ```bash
    uv run pytest
    ```

3.  **Running Locally with Active Changes:**

    To run `mcpo` with your local modifications from a specific branch (e.g., `my-feature-branch`):

    ```bash
    # Ensure you are on your development branch
    git checkout my-feature-branch

    # Make your code changes in the src/mcpo directory or elsewhere

    # Run mcpo using uv, which will use your local, modified code
    # This command starts mcpo on port 8000 and proxies your_mcp_server_command
    uv run mcpo --port 8000 -- your_mcp_server_command

    # Example with a test MCP server (like mcp-server-time):
    # uv run mcpo --port 8000 -- uvx mcp-server-time --local-timezone=America/New_York
    ```
    This allows you to test your changes interactively before committing or creating a pull request. Access your locally running `mcpo` instance at `http://localhost:8000` and the auto-generated docs at `http://localhost:8000/docs`.


## 🪪 License

MIT

## 🤝 Contributing

We welcome and strongly encourage contributions from the community!

Whether you're fixing a bug, adding features, improving documentation, or just sharing ideas—your input is incredibly valuable and helps make mcpo better for everyone.

Getting started is easy:

- Fork the repo
- Create a new branch
- Make your changes
- Open a pull request

Not sure where to start? Feel free to open an issue or ask a question—we’re happy to help you find a good first task.

## ✨ Star History

<a href="https://star-history.com/#open-webui/mcpo&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=open-webui/mcpo&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=open-webui/mcpo&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=open-webui/mcpo&type=Date" />
  </picture>
</a>

---

✨ Let's build the future of interoperable AI tooling together!
