# OAuth 2.1 Support in MCPO

MCPO now supports OAuth 2.1 client-to-server authentication for MCP servers that require it. This enables secure authentication with servers using the Authorization Code flow with refresh tokens.

## Features

- **OAuth 2.1 Authorization Code Flow**: Full support for the modern OAuth 2.1 standard
- **Multi-User Support**: Session-based authentication for multiple concurrent users
- **JWT Validation**: Signature verification for OpenWebUI integration
- **Automatic Token Refresh**: Tokens are automatically refreshed when they expire
- **Persistent Token Storage**: Tokens can be stored in memory or on disk
- **Browser-based Authorization**: Automatically opens browser for user authorization
- **Dynamic Tool Discovery**: Session-aware OpenAPI generation

## Configuration

### Basic OAuth Configuration (Recommended)

MCPO defaults to **dynamic client registration** for maximum compatibility. Most OAuth servers only need minimal configuration:

```json
{
  "mcpServers": {
    "my-oauth-server": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp",
      "oauth": {
        "server_url": "http://localhost:8000"
      }
    }
  }
}
```

This minimal configuration allows MCPO to:
- Automatically discover OAuth endpoints from `/.well-known/oauth-authorization-server`
- Perform dynamic client registration with the server
- Use server-provided scopes and configuration

### Static Client Configuration (Legacy Servers)

For servers that don't support dynamic client registration, you can specify static client metadata:

```json
{
  "mcpServers": {
    "legacy-oauth-server": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp", 
      "oauth": {
        "server_url": "http://localhost:8000",
        "storage_type": "file",
        "client_metadata": {
          "client_name": "My MCPO Client",
          "redirect_uris": ["http://localhost:3030/callback"],
          "grant_types": ["authorization_code", "refresh_token"],
          "response_types": ["code"]
        }
      }
    }
  }
}
```

> ⚠️ **Important**: Avoid hardcoding `scope`, `authorization_endpoint`, or `token_endpoint` in your configuration. These should be automatically discovered from the server's OAuth metadata during the flow.

## OAuth Configuration Options

### Required Fields

- `server_url`: The base URL of the OAuth server (without `/mcp` endpoint)

### Optional Fields

- `callback_port`: Port for the local callback server (default: 3030)
- `use_loopback`: Whether to use automatic browser flow (default: true)
- `storage_type`: Where to store tokens - "memory" or "file" (default: "file")
- `multi_user`: Enable multi-user mode (default: true)
- `webui_secret_key`: Shared secret for JWT validation (for OpenWebUI integration)
- `session_timeout_minutes`: Session timeout for multi-user mode (default: 30)
- `client_metadata`: OAuth client registration metadata

### Client Metadata Fields (for static registration only)

- `client_name`: Display name for your client
- `redirect_uris`: List of allowed redirect URIs
- `grant_types`: OAuth grant types to request (typically `["authorization_code", "refresh_token"]`)
- `response_types`: OAuth response types to accept (typically `["code"]`)
- `token_endpoint_auth_method`: How to authenticate with the token endpoint

**Fields handled automatically (don't set these):**
- `scope`: Automatically provided by the server during dynamic registration
- `authorization_endpoint`: Discovered from server metadata
- `token_endpoint`: Discovered from server metadata

## How It Works

1. **Metadata Discovery**: MCPO automatically discovers OAuth endpoints from the server's `/.well-known/oauth-authorization-server` metadata
2. **Dynamic Registration**: If supported, MCPO registers itself as a client with the OAuth server
3. **Token Check**: MCPO checks for existing valid tokens for this server
4. **Authorization Flow**: If no valid tokens exist, MCPO:
   - Opens your browser to the authorization URL
   - Starts a local HTTP server to receive the callback
   - Exchanges the authorization code for tokens
5. **Token Storage**: Tokens are stored securely:
   - **File storage**: In `~/.mcpo/tokens/` directory
   - **Memory storage**: Only for the duration of the session
6. **Token Refresh**: When tokens expire, MCPO automatically refreshes them using the refresh token
7. **Authentication**: All requests to the MCP server include the access token

## OpenWebUI Integration

MCPO provides seamless integration with OpenWebUI through JWT validation and multi-user session management.

### Configuration for OpenWebUI

To enable OpenWebUI integration, configure MCPO with JWT validation:

```json
{
  "mcpServers": {
    "openwebui-tools": {
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

### JWT Validation

When `webui_secret_key` is configured, MCPO validates incoming JWT tokens from OpenWebUI:

- **Signature Verification**: Uses HS256 algorithm with the shared secret
- **Expiration Checking**: Automatically rejects expired tokens
- **User Context Extraction**: Extracts user ID and claims from valid tokens
- **Backward Compatibility**: Falls back to no verification if secret not provided

### Integration Benefits

✅ **Session-based Tool Discovery**: Dynamic OpenAPI generation based on user authentication  
✅ **Per-user Tool Isolation**: Each user gets their own MCP session and tool access  
✅ **Token Security**: Proper JWT signature verification prevents token forgery  
✅ **Automatic Cleanup**: Expired sessions and tokens are automatically cleaned up  
✅ **Zero OpenWebUI Changes**: Works with existing OpenWebUI session forwarding  

### Usage with OpenWebUI

1. **Configure Shared Secret**: Use the same `WEBUI_SECRET_KEY` in both OpenWebUI and MCPO
2. **Tool Registration**: Add MCPO server URL to OpenWebUI's tool server configuration
3. **Dynamic Discovery**: OpenWebUI will automatically discover user-appropriate tools
4. **Session Management**: User sessions are maintained across requests

## Storage Types

### File Storage (Recommended)
- Tokens persist between MCPO restarts
- Stored in `~/.mcpo/tokens/` with per-server isolation
- Each server gets its own token file based on a hash of the server name

### Memory Storage
- Tokens only exist for the current session
- Lost when MCPO restarts
- Useful for testing or temporary sessions

## Authorization Flows

### Automatic Browser Flow (Default)
When `use_loopback` is `true`:
1. MCPO opens your default browser to the authorization page
2. After you authorize, you're redirected to `http://localhost:PORT/callback`
3. MCPO's built-in server captures the authorization code
4. The browser shows a success message

### Manual Copy/Paste Flow
When `use_loopback` is `false`:
1. MCPO prints the authorization URL
2. You manually open the URL in a browser
3. After authorization, you copy the full callback URL
4. You paste it back into MCPO when prompted

## Server Support

OAuth authentication is supported for:
- ✅ `streamable-http` servers (single-user and multi-user modes)
- ❌ `sse` servers (not currently supported)
- ❌ `stdio` servers (OAuth not applicable)

### Multi-User Mode Features

When `multi_user: true` is configured:
- ✅ Per-user session isolation
- ✅ JWT token validation (when `webui_secret_key` provided)
- ✅ Dynamic OpenAPI schema generation
- ✅ Automatic session cleanup
- ✅ Web-based authentication flows

## Security Considerations

1. **Token Storage**: File-based tokens are stored in plaintext. For production use, consider implementing encrypted storage
2. **Redirect URIs**: Always use `localhost` for development. For production, use HTTPS URLs
3. **Scopes**: Only request the minimum scopes necessary for your use case
4. **Token Expiry**: Tokens are automatically refreshed, but expired tokens are not automatically deleted

## Troubleshooting

### "OAuth server_url required"
Ensure your configuration includes the `server_url` field in the `oauth` section.

### Browser doesn't open
Check that `use_loopback` is set to `true` and that your system has a default browser configured.

### "No authorization code found"
If using manual flow, ensure you're copying the complete callback URL including all query parameters.

### Port already in use
Change the `callback_port` to an available port number.

### Tokens not persisting
Ensure `storage_type` is set to `"file"` and that MCPO has write permissions to `~/.mcpo/tokens/`.

### JWT validation failures
- Check that `webui_secret_key` matches the `WEBUI_SECRET_KEY` used by OpenWebUI
- Verify JWT tokens are not expired
- Ensure tokens are properly formatted and include required claims (`id` field)

## Example: Testing OAuth

1. Create a config file with OAuth settings:
```json
{
  "mcpServers": {
    "test-oauth": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp",
      "oauth": {
        "server_url": "http://localhost:8000",
        "storage_type": "file"
      }
    }
  }
}
```

2. Start MCPO:
```bash
mcpo --config config_oauth.json
```

3. MCPO will:
   - Open your browser for authorization
   - Capture the callback
   - Store tokens
   - Connect to the MCP server

4. Subsequent connections will reuse the stored tokens

## Example: OpenWebUI Integration

1. Create a config file with OpenWebUI integration:
```json
{
  "mcpServers": {
    "openwebui-tools": {
      "type": "streamable-http",
      "url": "http://localhost:8000/mcp",
      "oauth": {
        "server_url": "http://localhost:8000",
        "multi_user": true,
        "webui_secret_key": "sk-your-secret-key-here",
        "session_timeout_minutes": 60
      }
    }
  }
}
```

2. Start MCPO:
```bash
mcpo --config config_openwebui.json
```

3. Configure OpenWebUI:
   - Set the same `WEBUI_SECRET_KEY` in OpenWebUI environment
   - Add MCPO server URL to OpenWebUI's tool server configuration
   - OpenWebUI will automatically handle JWT token forwarding

4. Usage:
   - Users authenticate through OpenWebUI's normal login flow
   - JWT tokens are automatically validated by MCPO
   - Tools are dynamically discovered based on user authentication
   - Each user gets isolated sessions and tool access

## Implementation Details

The OAuth implementation uses the Python MCP SDK's built-in `OAuthClientProvider` which handles:
- PKCE (Proof Key for Code Exchange) when required
- Token refresh logic
- Authorization header injection
- Dynamic client registration (if supported by server)

Tokens are bound to individual server instances and isolated from each other, ensuring that each OAuth-enabled server maintains its own authentication session.