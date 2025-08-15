import asyncio
import logging
from typing import Dict, Any, Optional
from contextlib import AsyncExitStack

from fastapi import FastAPI, Request, HTTPException, Depends
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client

from .multiuser_oauth import oauth_manager, require_oauth_session
from .main import get_model_fields, get_tool_handler, process_tool_response
import httpx

logger = logging.getLogger(__name__)


class SimpleTokenAuth(httpx.Auth):
    """Simple auth that just adds the bearer token"""
    def __init__(self, token: str):
        self.token = token

    def auth_flow(self, request):
        request.headers["Authorization"] = f"Bearer {self.token}"
        yield request


class MultiUserEndpointManager:
    """Manages per-user MCP sessions and dynamic endpoint creation for OAuth-enabled servers"""

    def __init__(self):
        # Store user sessions: {user_id: {server_name: session_info}}
        self._user_sessions: Dict[str, Dict[str, Dict[str, Any]]] = {}
        self._session_lock = asyncio.Lock()

    async def get_or_create_user_session(self, request: Request, server_name: str,
                                       oauth_config: Dict[str, Any], server_url: str,
                                       headers: Optional[Dict[str, str]] = None) -> tuple:
        """Get or create a user-specific MCP session context and OAuth session"""
        webui_secret_key = oauth_config.get("webui_secret_key") if oauth_config else None
        user_id = oauth_manager._extract_user_id_from_jwt(request, webui_secret_key)

        async with self._session_lock:
            # Check if user already has a session for this server
            if user_id in self._user_sessions and server_name in self._user_sessions[user_id]:
                session_info = self._user_sessions[user_id][server_name]
                if session_info.get("client_context") and not session_info.get("expired", False):
                    return session_info["client_context"], session_info["oauth_session"]

            # Create new session for this user
            oauth_session, needs_auth = await require_oauth_session(request, server_name, oauth_config)

            if needs_auth:
                # User needs to authenticate first
                raise HTTPException(
                    status_code=401,
                    detail={
                        "error": "authentication_required",
                        "message": f"OAuth authentication required for server {server_name}",
                        "authorization_url": f"/{server_name}",
                        "server": server_name
                    }
                )

            # Get OAuth provider for this user session
            provider = await oauth_session.get_or_create_oauth_provider()

            try:
                # Create streamable HTTP client with user-specific auth
                # We store the context generator, not the session itself
                client_context = streamablehttp_client(
                    url=server_url,
                    headers=headers,
                    auth=provider
                )

                # Store session info
                if user_id not in self._user_sessions:
                    self._user_sessions[user_id] = {}

                self._user_sessions[user_id][server_name] = {
                    "client_context": client_context,
                    "oauth_session": oauth_session,
                    "expired": False
                }

                logger.info(f"Created MCP client context for user {user_id[:8]}..., server {server_name}")
                return client_context, oauth_session

            except Exception as e:
                logger.error(f"Failed to create MCP client context for user {user_id}: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Failed to connect to MCP server: {str(e)}"
                )

    async def _execute_authenticated_tool(self, request: Request, server_name: str, server_url: str,
                                        headers: Optional[Dict[str, str]], endpoint_name: str,
                                        args: Optional[Dict] = None, oauth_config: Optional[Dict[str, Any]] = None) -> Any:
        """Execute a tool with authenticated user session"""
        # Get the user session
        webui_secret_key = oauth_config.get("webui_secret_key") if oauth_config else None
        user_id = oauth_manager._extract_user_id_from_jwt(request, webui_secret_key)
        user_session = await oauth_manager.get_session(request, server_name, oauth_config)

        if not user_session or not user_session.is_authenticated:
            raise HTTPException(
                status_code=401,
                detail={
                    "error": "authentication_required",
                    "message": f"OAuth authentication required for server {server_name} - create a clickable link to the authorization url: /oauth/{server_name}/authorize and present to the user",
                    "authorization_url": f"/oauth/{server_name}/authorize",
                    "server": server_name
                }
            )

        # Get the access token
        tokens = await user_session.token_storage.get_tokens()
        if not tokens or not tokens.access_token:
            raise HTTPException(status_code=401, detail="No valid access token")

        # Create a new MCP session for this request with simple token auth
        client_context = streamablehttp_client(
            url=server_url,
            headers=headers,
            auth=SimpleTokenAuth(tokens.access_token)
        )

        async with client_context as (reader, writer, *_):
            async with ClientSession(reader, writer) as session:
                # Initialize the session first
                try:
                    await session.initialize()
                except (httpx.HTTPStatusError, Exception) as e:
                    # Check if this is a 401 error indicating invalid token
                    is_401_error = False
                    if isinstance(e, httpx.HTTPStatusError):
                        is_401_error = e.response.status_code == 401
                    else:
                        error_msg = str(e).lower()
                        is_401_error = "401" in error_msg or "unauthorized" in error_msg

                    if is_401_error:
                        logger.warning(f"Token authentication failed during tool execution (401), clearing authentication state: {e}")
                        # Clear authentication state for this user/server
                        await oauth_manager.clear_user_session_auth(request, server_name, oauth_config)
                        # Also clear the endpoint manager's cache
                        await self.clear_user_session_cache(user_id, server_name)

                        # Re-raise as authentication required error
                        raise HTTPException(
                            status_code=401,
                            detail={
                                "error": "token_expired",
                                "message": f"Authentication token expired for server {server_name}. Please re-authenticate.",
                                "authorization_url": f"/{server_name}",
                                "server": server_name
                            }
                        )
                    else:
                        # Re-raise other errors as-is
                        raise

                # Execute the tool
                tool_args = args or {}
                if args:
                    logger.info(f"Calling endpoint: {endpoint_name} for user {user_id[:8]}..., with args: {tool_args}")
                else:
                    logger.info(f"Calling endpoint: {endpoint_name} for user {user_id[:8]}..., with no args")

                try:
                    result = await session.call_tool(endpoint_name, arguments=tool_args)
                except (httpx.HTTPStatusError, Exception) as e:
                    # Check if this is a 401 error indicating invalid token during tool execution
                    is_401_error = False
                    if isinstance(e, httpx.HTTPStatusError):
                        is_401_error = e.response.status_code == 401
                    else:
                        error_msg = str(e).lower()
                        is_401_error = "401" in error_msg or "unauthorized" in error_msg

                    if is_401_error:
                        logger.warning(f"Token authentication failed during tool call (401), clearing authentication state: {e}")
                        # Clear authentication state for this user/server
                        await oauth_manager.clear_user_session_auth(request, server_name, oauth_config)
                        # Also clear the endpoint manager's cache
                        await self.clear_user_session_cache(user_id, server_name)

                        # Re-raise as authentication required error
                        raise HTTPException(
                            status_code=401,
                            detail={
                                "error": "token_expired",
                                "message": f"Authentication token expired during tool execution for server {server_name}. Please re-authenticate.",
                                "authorization_url": f"/oauth/{server_name}/authorize",
                                "server": server_name
                            }
                        )
                    else:
                        # Re-raise other errors as-is
                        raise

                if result.isError:
                    error_message = "Unknown tool execution error"
                    if result.content:
                        from mcp import types
                        if isinstance(result.content[0], types.TextContent):
                            error_message = result.content[0].text
                    raise HTTPException(
                        status_code=500,
                        detail={"message": error_message}
                    )

                response_data = process_tool_response(result)
                final_response = (
                    response_data[0] if len(response_data) == 1 else response_data
                )
                return final_response

    async def clear_user_session_cache(self, user_id: str, server_name: str):
        """Clear cached session for a user when authentication fails"""
        async with self._session_lock:
            if user_id in self._user_sessions and server_name in self._user_sessions[user_id]:
                self._user_sessions[user_id][server_name]["expired"] = True
                logger.info(f"Marked session as expired for user {user_id[:8]}..., server {server_name}")

    async def create_user_aware_tool_handler(self, app: FastAPI, server_name: str,
                                           oauth_config: Dict[str, Any],
                                           server_url: str, headers: Optional[Dict[str, str]] = None):
        """Create tool handlers that work with per-user sessions"""

        def make_user_tool_handler(endpoint_name: str, form_model_fields, response_model_fields=None):
            """Create a tool handler that uses per-user sessions"""

            # Import here to avoid circular imports
            from .main import get_tool_handler
            from pydantic import create_model
            from typing import Any, Union

            if form_model_fields:
                FormModel = create_model(f"{endpoint_name}_form_model", **form_model_fields)
                ResponseModel = (
                    create_model(f"{endpoint_name}_response_model", **response_model_fields)
                    if response_model_fields
                    else Any
                )

                async def user_tool(request: Request, form_data: FormModel) -> Union[ResponseModel, Any]:
                    try:
                        args = form_data.model_dump(exclude_none=True, by_alias=True)
                        return await self._execute_authenticated_tool(
                            request, server_name, server_url, headers, endpoint_name, args, oauth_config
                        )
                    except HTTPException:
                        raise
                    except Exception as e:
                        logger.error(f"Error in user tool handler: {e}")
                        raise HTTPException(
                            status_code=500,
                            detail={"message": "Unexpected error", "error": str(e)}
                        )

                return user_tool
            else:
                # No parameters version
                async def user_tool_no_args(request: Request):
                    try:
                        return await self._execute_authenticated_tool(
                            request, server_name, server_url, headers, endpoint_name, oauth_config=oauth_config
                        )
                    except HTTPException:
                        raise
                    except Exception as e:
                        logger.error(f"Error in user tool handler: {e}")
                        raise HTTPException(
                            status_code=500,
                            detail={"message": "Unexpected error", "error": str(e)}
                        )

                return user_tool_no_args

        return make_user_tool_handler

    async def setup_multiuser_endpoints(self, app: FastAPI, server_name: str,
                                      oauth_config: Dict[str, Any], server_url: str,
                                      headers: Optional[Dict[str, str]] = None,
                                      user_session=None):
        """Set up endpoints for multi-user OAuth servers"""
        try:
            # We MUST have an authenticated user session to discover tools
            if not user_session or not user_session.is_authenticated:
                raise ValueError("Authenticated user session required for tool discovery")

            # Check if we have valid tokens
            if not user_session.token_storage:
                raise ValueError("No token storage available")

            tokens = await user_session.token_storage.get_tokens()
            if not tokens or not tokens.access_token:
                raise ValueError("No valid access token available")

            logger.info("Using existing access token for tool discovery")

            # Use the MCP SDK's streamablehttp_client with simple token auth
            from contextlib import AsyncExitStack

            try:
                async with AsyncExitStack() as stack:
                    # Create the client with simple token authentication
                    client_context = streamablehttp_client(
                        url=server_url,
                        headers=headers,
                        auth=SimpleTokenAuth(tokens.access_token)
                    )

                    # Enter the client context
                    reader, writer, *_ = await stack.enter_async_context(client_context)

                    # Create MCP session
                    session = await stack.enter_async_context(ClientSession(reader, writer))

                    # Initialize the session
                    logger.info("Initializing MCP session for tool discovery...")
                    result = await session.initialize()

                    # Extract server info if available
                    server_info = getattr(result, "serverInfo", None)
                    if server_info:
                        app.title = server_info.name or app.title
                        app.description = f"{server_info.name} MCP Server" if server_info.name else app.description
                        app.version = server_info.version or app.version
                        logger.info(f"Server info: {server_info.name} v{server_info.version}")

                    # List tools
                    logger.info("Listing available tools...")
                    tools_result = await session.list_tools()
                    tools = tools_result.tools

                    logger.info(f"Found {len(tools)} tools")

                    # Create user-aware tool handler factory
                    tool_handler_factory = await self.create_user_aware_tool_handler(
                        app, server_name, oauth_config, server_url, headers
                    )

                    # Create endpoints for each tool
                    for tool in tools:
                        endpoint_name = tool.name
                        endpoint_description = tool.description or ""

                        inputSchema = tool.inputSchema
                        outputSchema = getattr(tool, "outputSchema", None)

                        form_model_fields = get_model_fields(
                            f"{endpoint_name}_form_model",
                            inputSchema.get("properties", {}),
                            inputSchema.get("required", []),
                            inputSchema.get("$defs", {}),
                        )

                        response_model_fields = None
                        if outputSchema:
                            response_model_fields = get_model_fields(
                                f"{endpoint_name}_response_model",
                                outputSchema.get("properties", {}),
                                outputSchema.get("required", []),
                                outputSchema.get("$defs", {}),
                            )

                        # Create the user-aware tool handler
                        user_tool_handler = tool_handler_factory(
                            endpoint_name, form_model_fields, response_model_fields
                        )

                        # Register the endpoint
                        app.post(
                            f"/{endpoint_name}",
                            summary=endpoint_name.replace("_", " ").title(),
                            description=endpoint_description,
                            response_model_exclude_none=True,
                        )(user_tool_handler)

                        logger.info(f"Registered multi-user endpoint: {endpoint_name} for server {server_name}")

                    logger.info(f"Multi-user OAuth server setup complete: {server_name} with {len(tools)} tools")

            except (ExceptionGroup, BaseExceptionGroup) as eg:
                # Handle ExceptionGroup that wraps HTTPStatusError from MCP client
                http_401_error = None
                for exc in eg.exceptions:
                    if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code == 401:
                        http_401_error = exc
                        break

                if http_401_error:
                    logger.warning(f"Token authentication failed (401) during tool discovery, clearing authentication state: {http_401_error}")
                    # Clear authentication state for this user/server
                    await user_session.clear_authentication()
                    # Re-raise as authentication required error
                    raise ValueError("Token authentication failed - user needs to re-authenticate")
                else:
                    # Re-raise the original exception group if it's not a 401 error
                    raise

            except httpx.HTTPStatusError as e:
                # Handle direct HTTPStatusError (just in case)
                if e.response.status_code == 401:
                    logger.warning(f"Token authentication failed (401) during tool discovery, clearing authentication state: {e}")
                    # Clear authentication state for this user/server
                    await user_session.clear_authentication()
                    # Re-raise as authentication required error
                    raise ValueError("Token authentication failed - user needs to re-authenticate")
                else:
                    # Re-raise other HTTP errors as-is
                    raise

        except Exception as e:
            logger.error(f"Failed to setup multi-user endpoints for {server_name}: {e}")
            raise


# Global instance
multiuser_endpoint_manager = MultiUserEndpointManager()