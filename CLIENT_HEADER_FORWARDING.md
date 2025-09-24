# Client Header Forwarding in MCPO

MCPO supports forwarding HTTP headers from incoming client requests to MCP servers. This enables passing user context, authentication tokens, and other request-specific information to your MCP tools.

## Configuration

Add client header forwarding configuration to your MCP server config:

```json
{
  "mcpServers": {
    "some-mcp": {
      "command": "uvx",
      "args": ["some-mcp"],
      "client_header_forwarding": {
        "enabled": true,
        "whitelist": ["Authorization", "X-User-*", "X-Request-ID"],
        "blacklist": ["Host", "Content-Length"],
        "debug_headers": false
      }
    }
  }
}
```

## Configuration Options

- `enabled`: Enable/disable client header forwarding for this server (default: false)
- `whitelist`: List of header patterns to forward (supports wildcards with `*`)
- `blacklist`: List of header patterns to block (takes precedence over whitelist)
- `debug_headers`: Enable debug logging for header processing (default: false)

## Header Pattern Matching

- **Exact match**: `"Authorization"` matches only the `Authorization` header
- **Wildcard match**: `"X-User-*"` matches `X-User-ID`, `X-User-Email`, etc.
- **Global wildcard**: `"*"` matches all headers (use with caution)

## How It Works

1. **Client Request**: A client makes an HTTP request to mcpo with headers like `Authorization: Bearer <token>`
2. **Header Filtering**: Headers are filtered based on whitelist/blacklist rules
3. **MCP Forwarding**: Filtered headers are passed to the MCP server via the `_meta.headers` field in tool calls

## Transport Support

Client header forwarding works with all MCP transport types:
- **stdio**: Headers are passed via `_meta` field in JSON-RPC calls
- **SSE**: Headers are passed via `_meta` field in JSON-RPC calls  
- **HTTP**: Headers are passed via `_meta` field in JSON-RPC calls

## Complementary Features

Client header forwarding works alongside mcpo's connection-level headers:

```json
{
  "mcpServers": {
    "protected-server": {
      "type": "sse",
      "url": "https://api.example.com/mcp",
      "headers": {
        "Authorization": "Bearer server-token-123"
      },
      "client_header_forwarding": {
        "enabled": true,
        "whitelist": ["Authorization", "X-User-*"]
      }
    }
  }
}
```

- **`headers`**: Static headers for mcpo ↔ MCP server authentication
- **`client_header_forwarding`**: Dynamic headers from client ↔ MCP server

## Security Considerations

- **Whitelist Headers**: Only forward necessary headers to minimize attack surface
- **Blacklist Sensitive Headers**: Block headers like `Host`, `Content-Length`, etc.
- **Debug Mode**: Only enable `debug_headers` in development environments

## MCP Server Integration

Your MCP server can access forwarded headers through the `_meta` field in tool calls:

```python
from mcp.server.fastmcp import FastMCP, Context
from mcp.server.session import ServerSession

mcp = FastMCP(name="Example Server")

@mcp.tool()
async def protected_tool(data: str, ctx: Context[ServerSession, None]) -> str:
    # Access forwarded headers
    headers = getattr(ctx.request_meta, 'headers', {}) if hasattr(ctx, 'request_meta') else {}
    
    # Check authorization
    auth_header = headers.get('Authorization', '')
    if not auth_header.startswith('Bearer '):
        raise ValueError("Missing or invalid authorization")
    
    # Extract user context
    user_id = headers.get('X-User-ID', 'unknown')
    request_id = headers.get('X-Request-ID', 'unknown')
    
    return f"Protected data for user {user_id} (request: {request_id}): {data}"
```

## Example Use Cases

### 1. User Authentication
```json
{
  "client_header_forwarding": {
    "enabled": true,
    "whitelist": ["Authorization"]
  }
}
```

Forward JWT tokens or API keys for user authentication.

### 2. Request Tracing
```json
{
  "client_header_forwarding": {
    "enabled": true,
    "whitelist": ["X-Request-ID", "X-Trace-ID"]
  }
}
```

Forward tracing headers for request correlation across services.

### 3. User Context
```json
{
  "client_header_forwarding": {
    "enabled": true,
    "whitelist": ["X-User-*"],
    "blacklist": ["X-User-Secret"]
  }
}
```

Forward user information while blocking sensitive headers.

## Hot Reload Support

Client header forwarding configurations are automatically reloaded when using mcpo's `--hot-reload` feature. Changes to the configuration file will be applied without restarting the server.
