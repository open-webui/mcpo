import asyncio
import base64
import hashlib
import httpx
import logging
import secrets
import time
import traceback
from typing import Dict, Any, Optional
from urllib.parse import urlencode, parse_qs, urlparse

from fastapi import Request, Response, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse, JSONResponse
from mcp.client.auth import OAuthClientProvider
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken
from pydantic import AnyUrl

from .multiuser_oauth import oauth_manager, require_oauth_session, UserSession
from .oauth import _load_callback_html

logger = logging.getLogger(__name__)


class OAuthEndpoints:
    """OAuth endpoints for multi-user authentication"""

    def __init__(self):
        # Store ongoing OAuth flows: {state: {session_id, server_name, user_id, created_at}}
        self._oauth_flows: Dict[str, Dict[str, Any]] = {}

    async def get_oauth_status(self, request: Request, server_name: str) -> JSONResponse:
        """Get OAuth authentication status for the current user"""
        try:
            status = await oauth_manager.get_session_status(request, server_name)
            response = JSONResponse(content=status)
            return response
        except Exception as e:
            logger.error(f"Error getting OAuth status for {server_name}: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    async def initiate_oauth(self, request: Request, server_name: str, oauth_config: Dict[str, Any]) -> Response:
        """Initiate OAuth flow for a user using MCP SDK"""
        try:
            # Clean up expired flows on each OAuth initiation
            self._cleanup_expired_flows()

            session, needs_auth = await require_oauth_session(request, server_name, oauth_config)

            if not needs_auth:
                # User is already authenticated
                response = JSONResponse(content={
                    "message": "User is already authenticated. Tell them to hard refresh (cmd+shift+r on Mac) to see updated tools and try their call again!",
                    "authenticated": True,
                    "server": server_name
                })
                return response

            logger.info(f"OAuth initiation requested for user {session.user_id[:8]}, server {server_name}")

            # Ensure the OAuth provider is created for this user session
            await session.get_or_create_oauth_provider()

            # Generate state for CSRF protection
            state = secrets.token_urlsafe(32)

            # Generate PKCE parameters if required
            code_verifier = base64.urlsafe_b64encode(secrets.token_bytes(32)).decode('utf-8').rstrip('=')
            code_challenge = base64.urlsafe_b64encode(
                hashlib.sha256(code_verifier.encode('utf-8')).digest()
            ).decode('utf-8').rstrip('=')

            # Store the OAuth flow information including PKCE verifier
            self._oauth_flows[state] = {
                "session_id": session.user_id,
                "server_name": server_name,
                "user_id": session.user_id,
                "code_verifier": code_verifier,  # Store for token exchange
                "created_at": time.time()  # TTL tracking
            }

            # Clean up expired flows (older than 10 minutes)
            self._cleanup_expired_flows()

            # Build redirect URI for our callback
            base_url = f"{request.url.scheme}://{request.url.netloc}"
            callback_url = f"{base_url}/oauth/{server_name}/callback"

            # Update session's client metadata with our callback URL
            if session.client_metadata:
                session.client_metadata.redirect_uris = [AnyUrl(callback_url)]

            # Let the MCP SDK handle discovery and get the authorization URL
            # The SDK will perform discovery and dynamic client registration
            server_url = oauth_config.get("server_url", "http://localhost:8000")

            try:
                # The provider will handle discovery and dynamic client registration
                # Perform discovery using the MCP SDK pattern
                async with httpx.AsyncClient() as client:
                    # Discover OAuth configuration using RFC 9728
                    discovery_url = f"{server_url}/.well-known/oauth-authorization-server"
                    logger.info(f"Attempting OAuth discovery at: {discovery_url}")

                    # Try GET request directly (OPTIONS often returns empty)
                    discovery_response = await client.get(discovery_url, follow_redirects=True)

                    if discovery_response.status_code == 200:
                        auth_config = discovery_response.json()
                        logger.info(f"OAuth discovery successful: {auth_config}")

                        # Check if we have valid OAuth configuration
                        if not auth_config or "authorization_endpoint" not in auth_config:
                            logger.error("OAuth discovery returned incomplete configuration, no authorization_endpoint found")
                            raise HTTPException(status_code=500, detail="OAuth authorization endpoint not found")
                        # Perform dynamic client registration if needed
                        elif "registration_endpoint" in auth_config:
                            reg_endpoint = auth_config["registration_endpoint"]

                            # Register client dynamically
                            reg_data = {
                                "client_name": f"MCPO Client for {server_name}",
                                "redirect_uris": [callback_url],
                                "grant_types": ["authorization_code", "refresh_token"],
                                "response_types": ["code"],
                                "token_endpoint_auth_method": "client_secret_post"
                            }

                            reg_response = await client.post(reg_endpoint, json=reg_data)
                            if reg_response.status_code in (200, 201):
                                client_info = reg_response.json()
                                logger.info(f"Dynamic client registration successful: client_id={client_info.get('client_id')}")
                                logger.debug(f"Registration response: {client_info}")

                                # Store client info in session's token storage
                                # Note: The registration response may not include redirect_uris, so we preserve our original metadata
                                if session.client_metadata:
                                    # Update the client metadata with the registered redirect URIs if provided
                                    if "redirect_uris" in client_info:
                                        session.client_metadata.redirect_uris = [AnyUrl(uri) for uri in client_info["redirect_uris"]]

                                oauth_client_info = OAuthClientInformationFull(
                                    client_id=client_info["client_id"],
                                    client_secret=client_info.get("client_secret", ""),  # Some servers don't return secret
                                    client_metadata=session.client_metadata if session.client_metadata else None,
                                    redirect_uris=session.client_metadata.redirect_uris if session.client_metadata else [callback_url]
                                )
                                if session.token_storage:
                                    await session.token_storage.set_client_info(oauth_client_info)

                                # Build authorization URL with PKCE if required
                                auth_endpoint = auth_config["authorization_endpoint"]
                                auth_params = {
                                    "client_id": client_info["client_id"],
                                    "redirect_uri": callback_url,
                                    "response_type": "code",
                                    "state": state,
                                    "scope": oauth_config.get("scope", "openid profile email")  # Use configured or default scopes
                                }

                                # Add PKCE parameters if required
                                if auth_config.get("pkce_required", False):
                                    auth_params["code_challenge"] = code_challenge
                                    auth_params["code_challenge_method"] = "S256"
                                    logger.info("Adding PKCE parameters to authorization request")

                                auth_url = f"{auth_endpoint}?{urlencode(auth_params)}"
                            else:
                                logger.error(f"Dynamic client registration failed: {reg_response.status_code}")
                                raise HTTPException(status_code=500, detail="Dynamic client registration failed")
                        else:
                            # No dynamic registration, use static client ID
                            auth_endpoint = auth_config["authorization_endpoint"]
                            auth_params = {
                                "client_id": "mcpo-client",
                                "redirect_uri": callback_url,
                                "response_type": "code",
                                "state": state,
                                "scope": oauth_config.get("scope", "openid profile email")
                            }

                            # Add PKCE parameters if required
                            if auth_config.get("pkce_required", False):
                                auth_params["code_challenge"] = code_challenge
                                auth_params["code_challenge_method"] = "S256"
                                logger.info("Adding PKCE parameters to authorization request")

                            auth_url = f"{auth_endpoint}?{urlencode(auth_params)}"
                    else:
                        logger.error(f"OAuth discovery failed: {discovery_response.status_code}")
                        # Fallback to basic OAuth URL construction
                        auth_url = f"{server_url}/oauth/authorize?client_id=mcpo-client&redirect_uri={callback_url}&state={state}&response_type=code"

            except Exception as discovery_error:
                logger.error(f"OAuth discovery/registration error: {discovery_error}")
                # Fallback URL
                auth_url = f"{server_url}/oauth/authorize?client_id=mcpo-client&redirect_uri={callback_url}&state={state}&response_type=code"

            # Check if this is a browser request (HTML Accept header)
            accept_header = request.headers.get("accept", "")
            if "text/html" in accept_header:
                # Browser request - redirect directly
                response = RedirectResponse(url=auth_url, status_code=302)
            else:
                # API request - return JSON with auth URL
                response = JSONResponse(content={
                    "error": "authentication_required",
                    "message": f"OAuth authentication required for server {server_name}",
                    "authorization_url": auth_url,
                    "server": server_name
                }, status_code=401)
            return response

        except Exception as e:
            logger.error(f"Error in OAuth initiation: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=f"OAuth initiation failed: {str(e)}")

    async def handle_oauth_callback(self, request: Request, server_name: str, oauth_config: Dict[str, Any]) -> Response:
        """Handle OAuth callback and exchange code for tokens"""
        try:
            # Parse callback parameters
            query_params = dict(request.query_params)
            code = query_params.get("code")
            state = query_params.get("state")
            error = query_params.get("error")

            if error:
                error_desc = query_params.get("error_description", error)
                logger.error(f"OAuth error for {server_name}: {error_desc}")
                html = _load_callback_html(
                    status="error",
                    title="Authorization Failed - MCPO",
                    heading="Authorization Failed",
                    message=f"The OAuth authorization process encountered an error: {error_desc}",
                    action_text="Please close this tab and check the application logs for more details."
                )
                return HTMLResponse(content=html, status_code=400)

            if not code or not state:
                html = _load_callback_html(
                    status="error",
                    title="Invalid Callback - MCPO",
                    heading="Invalid Callback",
                    message="Missing authorization code or state parameter.",
                    action_text="Please close this tab and try the authorization process again."
                )
                return HTMLResponse(content=html, status_code=400)

            # Verify state and get OAuth flow info
            flow_info = self._oauth_flows.get(state)
            if not flow_info:
                html = _load_callback_html(
                    status="error",
                    title="Invalid State - MCPO",
                    heading="Invalid State",
                    message="The OAuth state parameter is invalid or expired.",
                    action_text="Please close this tab and try the authorization process again."
                )
                return HTMLResponse(content=html, status_code=400)

            # Clean up the flow and any expired ones
            del self._oauth_flows[state]
            self._cleanup_expired_flows()

            # Get the user session
            session = None
            async with oauth_manager._lock:
                user_sessions = oauth_manager._user_sessions.get(flow_info["user_id"], {})
                session = user_sessions.get(server_name)

            if not session:
                html = _load_callback_html(
                    status="error",
                    title="Session Not Found - MCPO",
                    heading="Session Not Found",
                    message="The user session has expired or is invalid.",
                    action_text="Please close this tab and start a new authorization process."
                )
                return HTMLResponse(content=html, status_code=400)

            # Ensure provider is created and get server URL
            await session.get_or_create_oauth_provider()
            server_url = oauth_config.get("server_url", "http://localhost:8000")

            try:

                # Get stored client info from session's token storage
                client_info = None
                if session.token_storage:
                    client_info = await session.token_storage.get_client_info()

                # Build callback URL
                base_url = f"{request.url.scheme}://{request.url.netloc}"
                callback_url = f"{base_url}/oauth/{server_name}/callback"

                # Exchange authorization code for tokens
                async with httpx.AsyncClient() as client:
                    # First discover token endpoint
                    discovery_url = f"{server_url}/.well-known/oauth-authorization-server"
                    discovery_response = await client.get(discovery_url)

                    if discovery_response.status_code == 200:
                        auth_config = discovery_response.json()
                        token_endpoint = auth_config["token_endpoint"]
                    else:
                        # Fallback to standard endpoint
                        token_endpoint = f"{server_url}/oauth/token"

                    # Exchange code for tokens
                    token_data = {
                        "grant_type": "authorization_code",
                        "code": code,
                        "redirect_uri": callback_url,
                        "client_id": client_info.client_id if client_info else "mcpo-client",
                        "client_secret": client_info.client_secret if client_info else ""
                    }

                    # Add PKCE code_verifier if it was used
                    if flow_info.get("code_verifier"):
                        token_data["code_verifier"] = flow_info["code_verifier"]
                        logger.info("Including PKCE code_verifier in token exchange")

                    token_response = await client.post(token_endpoint, data=token_data)

                    if token_response.status_code == 200:
                        token_json = token_response.json()

                        # Store tokens using provider's storage
                        oauth_token = OAuthToken(
                            access_token=token_json["access_token"],
                            token_type=token_json.get("token_type", "Bearer"),
                            refresh_token=token_json.get("refresh_token"),
                            expires_in=token_json.get("expires_in"),
                            scope=token_json.get("scope", "mcp")
                        )

                        # Store tokens using session's token storage
                        if session.token_storage:
                            await session.token_storage.set_tokens(oauth_token)

                        # Mark session as authenticated
                        session.is_authenticated = True
                        session.update_activity()

                        logger.info(f"OAuth completed for user {session.user_id}, server {server_name}")

                        html = _load_callback_html(
                            status="success",
                            title="Authorization Successful - MCPO",
                            heading="Authorization Successful!",
                            message="Your OAuth authorization was completed successfully.",
                            action_text="You can safely close this browser tab and return to your application."
                        )
                        response = HTMLResponse(content=html)
                        return response
                    else:
                        error_detail = token_response.text
                        logger.error(f"Token exchange failed: {token_response.status_code} - {error_detail}")
                        html = _load_callback_html(
                            status="error",
                            title="Token Exchange Failed - MCPO",
                            heading="Token Exchange Failed",
                            message=f"Failed to exchange authorization code for tokens. Error: {error_detail}",
                            action_text="Please close this tab and check the application logs for more details."
                        )
                        return HTMLResponse(content=html, status_code=500)

            except Exception as e:
                logger.error(f"Failed to complete OAuth for {server_name}: {e}")
                logger.error(f"Traceback: {traceback.format_exc()}")
                html = _load_callback_html(
                    status="error",
                    title="Authorization Failed - MCPO",
                    heading="Authorization Failed",
                    message=f"Failed to complete the authorization process: {str(e)}",
                    action_text="Please close this tab and check the application logs for more details."
                )
                return HTMLResponse(content=html, status_code=500)

        except Exception as e:
            logger.error(f"Error in OAuth callback: {e}")
            html = _load_callback_html(
                status="error",
                title="Callback Error - MCPO",
                heading="Callback Error",
                message=f"An error occurred processing the OAuth callback: {str(e)}",
                action_text="Please close this tab and check the application logs for more details."
            )
            return HTMLResponse(content=html, status_code=500)

    async def _discover_oauth_config(self, server_url: str) -> Dict[str, Any]:
        """Discover OAuth configuration using RFC 9728"""

        # Try to discover OAuth authorization server
        try:
            async with httpx.AsyncClient() as client:
                # First try RFC 9728 protected resource discovery
                response = await client.get(f"{server_url}/.well-known/oauth-protected-resource")
                if response.status_code == 200:
                    resource_config = response.json()
                    auth_servers = resource_config.get("authorization_servers", [])
                    if auth_servers:
                        auth_server_url = auth_servers[0]

                        # Now get the authorization server metadata
                        auth_response = await client.get(f"{auth_server_url}/.well-known/oauth-authorization-server")
                        if auth_response.status_code == 200:
                            return auth_response.json()

                # Fallback: try direct authorization server discovery
                response = await client.get(f"{server_url}/.well-known/oauth-authorization-server")
                if response.status_code == 200:
                    return response.json()

            raise HTTPException(status_code=500, detail="Unable to discover OAuth configuration")

        except Exception as e:
            logger.error(f"OAuth discovery failed: {e}")
            raise HTTPException(status_code=500, detail=f"OAuth discovery failed: {str(e)}")


    def _cleanup_expired_flows(self) -> None:
        """Remove OAuth flows older than 10 minutes to prevent memory leaks"""
        cutoff = time.time() - 600  # 10 minutes
        expired_states = [
            state for state, flow in self._oauth_flows.items()
            if flow.get('created_at', 0) < cutoff
        ]

        for state in expired_states:
            del self._oauth_flows[state]

        if expired_states:
            logger.info(f"Cleaned up {len(expired_states)} expired OAuth flows")


# Global instance
oauth_endpoints = OAuthEndpoints()