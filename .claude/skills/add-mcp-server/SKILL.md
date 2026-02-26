---
name: add-mcp-server
description: Add a new MCP server to the OpenHubUI configuration file
argument-hint: "[server-name]"
allowed-tools: Read, Edit, Bash, Grep, Glob
---

# Add MCP Server

Help the user add a new MCP server to their `mcpo.json` configuration.

## Configuration Format

MCP servers are defined in `mcpo.json` under the `mcpServers` key:

```json
{
  "mcpServers": {
    "server-name": {
      ...server config...
    }
  }
}
```

## Server Types

### stdio (local command)
```json
{
  "type": "stdio",
  "command": "npx",
  "args": ["@modelcontextprotocol/server-memory"],
  "env": {
    "OPTIONAL_VAR": "value"
  },
  "enabled": true
}
```

### SSE (remote, Server-Sent Events)
```json
{
  "type": "sse",
  "url": "https://example.com/sse",
  "headers": {
    "Authorization": "Bearer token"
  },
  "enabled": true
}
```

### streamable-http (remote, HTTP streaming)
```json
{
  "type": "streamable-http",
  "url": "https://example.com/mcp",
  "headers": {},
  "enabled": true
}
```

## Steps

1. Read the current `mcpo.json` to see existing servers
2. Ask the user what type of server they want to add (stdio/sse/streamable-http) if not clear from $ARGUMENTS
3. Gather required info:
   - **stdio**: command, args, optional env vars
   - **sse/streamable-http**: URL, optional headers/auth
4. Choose a descriptive server name (lowercase, hyphenated)
5. Add the server entry to `mcpo.json`
6. Validate the JSON is well-formed after editing
7. If `--hot-reload` is active, the server will auto-connect; otherwise inform the user to restart
