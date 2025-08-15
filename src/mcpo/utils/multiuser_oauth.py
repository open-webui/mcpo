import asyncio
import logging
import time
import jwt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, Tuple
from urllib.parse import urlencode

from fastapi import Request, HTTPException
from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.client.session import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.shared.auth import OAuthClientInformationFull, OAuthToken, OAuthClientMetadata

from pydantic import AnyUrl

from .oauth import InMemoryTokenStorage, FileTokenStorage

logger = logging.getLogger(__name__)


class UserSession:
    """Represents a single user's OAuth session for a specific server"""

    def __init__(self, user_id: str, server_name: str, oauth_config: Dict[str, Any]):
        self.user_id = user_id
        self.server_name = server_name
        self.oauth_config = oauth_config
        self.created_at = datetime.utcnow()
        self.last_activity = datetime.utcnow()
        self.oauth_provider: Optional[OAuthClientProvider] = None
        self.token_storage: Optional[TokenStorage] = None  # Store reference to token storage
        self.client_metadata: Optional[OAuthClientMetadata] = None  # Store reference to client metadata
        self.client_session: Optional[ClientSession] = None
        self.is_authenticated = False
        self.session_timeout_minutes = oauth_config.get("session_timeout_minutes", 30)
        self._lock = asyncio.Lock()

    def is_expired(self) -> bool:
        """Check if the session has expired"""
        timeout = timedelta(minutes=self.session_timeout_minutes)
        return datetime.utcnow() - self.last_activity > timeout

    def update_activity(self):
        """Update last activity timestamp"""
        self.last_activity = datetime.utcnow()

    async def get_or_create_oauth_provider(self) -> OAuthClientProvider:
        """Get or create OAuth provider for this user session"""
        async with self._lock:
            if not self.oauth_provider:
                storage_type = self.oauth_config.get("storage_type", "file")
                # For multi-user, we need a special provider that doesn't open browsers
                self.oauth_provider = await self._create_multiuser_oauth_provider(
                    storage_type=storage_type
                )
                logger.info(f"Created OAuth provider for user {self.user_id}, server {self.server_name}")
            return self.oauth_provider

    async def _create_multiuser_oauth_provider(self, storage_type: str = "file") -> OAuthClientProvider:
        """Create an OAuth provider suitable for multi-user scenarios (no browser opening)"""

        # Extract OAuth configuration
        server_url = self.oauth_config.get("server_url")
        if not server_url:
            raise ValueError("OAuth server_url is required")

        # Create client metadata using proper constructor with AnyUrl
        client_metadata = OAuthClientMetadata(
            client_name=f"MCPO {self.server_name}",
            redirect_uris=[AnyUrl("http://localhost:8001/oauth/callback")],  # Will be updated dynamically
            grant_types=["authorization_code", "refresh_token"],
            response_types=["code"],
            token_endpoint_auth_method="client_secret_post",
        )
        
        # Store the client metadata reference so we can access/update it later
        self.client_metadata = client_metadata

        # Choose storage backend with user isolation
        if storage_type == "memory":
            storage = InMemoryTokenStorage(self.server_name, self.user_id)
        else:
            storage = FileTokenStorage(self.server_name, self.user_id)

        # Store the storage reference so we can access it later
        self.token_storage = storage

        # For multi-user, we need handlers that prevent browser opening but don't error
        async def multiuser_redirect_handler(url: str) -> None:
            """Redirect handler for multi-user mode - logs but doesn't open browser"""
            logger.info(f"OAuth redirect would go to: {url}")
            # In multi-user mode, this shouldn't be called if we have valid tokens
            # If it is called, it means tokens are invalid/expired

        async def multiuser_callback_handler() -> tuple[str, str | None]:
            """Callback handler for multi-user mode"""
            # This might be called if tokens are invalid
            # Return None to indicate no auth code available
            logger.warning("OAuth callback handler called in multi-user mode - tokens may be invalid")
            return None, None  # No auth code available

        # Create provider with multi-user aware handlers
        return OAuthClientProvider(
            server_url=server_url,
            client_metadata=client_metadata,
            storage=storage,
            redirect_handler=multiuser_redirect_handler,
            callback_handler=multiuser_callback_handler,
        )

    async def authenticate_if_needed(self) -> bool:
        """Authenticate the user if not already authenticated and tokens are available"""
        if self.is_authenticated:
            return True

        # Ensure provider and storage are created
        provider = await self.get_or_create_oauth_provider()

        # Use our stored reference to the token storage
        if not self.token_storage:
            logger.error(f"Token storage not initialized for user {self.user_id}")
            return False

        tokens = await self.token_storage.get_tokens()

        if tokens:
            # TODO: Verify token validity/expiration
            self.is_authenticated = True
            logger.info(f"User {self.user_id} already has valid tokens for {self.server_name}")
            return True

        return False

    async def clear_authentication(self):
        """Clear authentication state and tokens when they become invalid"""
        logger.info(f"Clearing authentication for user {self.user_id}, server {self.server_name}")
        self.is_authenticated = False
        
        # Clear stored tokens
        if self.token_storage:
            try:
                await self.token_storage.clear_tokens()
                logger.info(f"Cleared tokens for user {self.user_id}")
            except Exception as e:
                logger.error(f"Failed to clear tokens for user {self.user_id}: {e}")
        
        # Close existing client session if any
        if self.client_session:
            try:
                # Note: ClientSession doesn't have explicit close, it's managed by context managers
                self.client_session = None
                logger.info(f"Cleared client session for user {self.user_id}")
            except Exception as e:
                logger.error(f"Failed to clear client session for user {self.user_id}: {e}")

    async def create_client_session(self, server_url: str, headers: Optional[Dict[str, str]] = None) -> Optional[ClientSession]:
        """Create an MCP client session with OAuth authentication"""
        async with self._lock:
            if self.client_session:
                return self.client_session

            if not self.is_authenticated:
                # Try to authenticate with existing tokens
                if not await self.authenticate_if_needed():
                    return None

            provider = await self.get_or_create_oauth_provider()

            try:
                client_context = streamablehttp_client(
                    url=server_url,
                    headers=headers,
                    auth=provider
                )

                # We can't actually create the session here as we need the context manager
                # The caller will need to handle this
                return client_context

            except Exception as e:
                logger.error(f"Failed to create client session for user {self.user_id}: {e}")
                return None


class MultiUserOAuthManager:
    """Manages OAuth sessions for multiple users across multiple servers"""

    def __init__(self):
        self._user_sessions: Dict[str, Dict[str, UserSession]] = {}  # {user_id: {server_name: session}}
        self._cleanup_task: Optional[asyncio.Task] = None
        self._lock = asyncio.Lock()

    def _extract_user_id_from_jwt(self, request: Request) -> Optional[str]:
        """Extract user ID from JWT token in Authorization header"""
        auth_header = request.headers.get("authorization")
        if not auth_header:
            return None
            
        # Check if it's a Bearer token
        if not auth_header.startswith("Bearer "):
            return None
            
        token = auth_header[7:]  # Remove "Bearer " prefix
        
        try:
            # Decode JWT without verification for now (Open-WebUI should handle verification)
            # In production, you might want to verify with a shared secret
            decoded = jwt.decode(token, options={"verify_signature": False})
            user_id = decoded.get("id")
            
            if user_id:
                logger.info(f"Extracted user ID from JWT: {user_id[:8]}...")
                return user_id
            else:
                logger.warning("JWT token does not contain 'id' field")
                return None
                
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
        except Exception as e:
            logger.error(f"Error decoding JWT token: {e}")
            return None
    

    async def get_or_create_session(self, request: Request, server_name: str, oauth_config: Dict[str, Any]) -> UserSession:
        """Get or create a user session for the specified server"""
        user_id = self._extract_user_id_from_jwt(request)
        
        if not user_id:
            raise HTTPException(status_code=401, detail="No valid JWT token found in Authorization header")

        async with self._lock:
            if user_id not in self._user_sessions:
                self._user_sessions[user_id] = {}
                logger.info(f"Created session container for user {user_id}")

            if server_name not in self._user_sessions[user_id]:
                session = UserSession(user_id, server_name, oauth_config)
                self._user_sessions[user_id][server_name] = session
                logger.info(f"Created session {user_id[:8]}... for user {user_id}, server {server_name}")

            session = self._user_sessions[user_id][server_name]
            session.update_activity()
            return session

    async def get_session(self, request: Request, server_name: str) -> Optional[UserSession]:
        """Get existing user session"""
        user_id = self._extract_user_id_from_jwt(request)
        
        if not user_id:
            return None

        async with self._lock:
            user_sessions = self._user_sessions.get(user_id, {})
            session = user_sessions.get(server_name)

            if session and not session.is_expired():
                session.update_activity()
                return session

            return None

    async def cleanup_expired_sessions(self):
        """Clean up expired sessions"""
        async with self._lock:
            users_to_remove = []

            for user_id, user_sessions in self._user_sessions.items():
                servers_to_remove = []

                for server_name, session in user_sessions.items():
                    if session.is_expired():
                        servers_to_remove.append(server_name)
                        logger.info(f"Cleaning up expired session for user {user_id}, server {server_name}")

                for server_name in servers_to_remove:
                    del user_sessions[server_name]

                if not user_sessions:
                    users_to_remove.append(user_id)

            for user_id in users_to_remove:
                del self._user_sessions[user_id]
                logger.info(f"Removed all sessions for user {user_id}")

    async def start_cleanup_task(self):
        """Start the background cleanup task"""
        if self._cleanup_task is not None:
            return

        async def cleanup_loop():
            while True:
                try:
                    await asyncio.sleep(300)  # Clean up every 5 minutes
                    await self.cleanup_expired_sessions()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Error in cleanup task: {e}")

        self._cleanup_task = asyncio.create_task(cleanup_loop())
        logger.info("Started OAuth session cleanup task")

    async def stop_cleanup_task(self):
        """Stop the background cleanup task"""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
            logger.info("Stopped OAuth session cleanup task")

    async def get_session_status(self, request: Request, server_name: str) -> Dict[str, Any]:
        """Get the authentication status for a user session"""
        session = await self.get_session(request, server_name)
        user_id = self._extract_user_id_from_jwt(request)

        if not session or not user_id:
            return {
                "authenticated": False,
                "server": server_name,
                "user_id": user_id[:8] + "..." if user_id else "unknown",  # Partial user ID for privacy
                "session_created": None,
                "last_activity": None
            }

        return {
            "authenticated": session.is_authenticated,
            "server": server_name,
            "user_id": session.user_id[:8] + "...",  # Partial user ID for privacy
            "session_created": session.created_at.isoformat() + "Z",
            "last_activity": session.last_activity.isoformat() + "Z"
        }

    async def clear_user_session_auth(self, request: Request, server_name: str) -> None:
        """Clear authentication state for a user session when tokens become invalid"""
        user_id = self._extract_user_id_from_jwt(request)
        
        if not user_id:
            return
        
        async with self._lock:
            user_sessions = self._user_sessions.get(user_id, {})
            session = user_sessions.get(server_name)
            
            if session:
                await session.clear_authentication()
                logger.info(f"Cleared authentication for user {user_id[:8]}..., server {server_name}")


# Global instance
oauth_manager = MultiUserOAuthManager()


async def require_oauth_session(request: Request, server_name: str, oauth_config: Dict[str, Any]) -> Tuple[UserSession, bool]:
    """
    Require OAuth authentication for a request
    Returns (session, needs_auth) tuple
    """
    session = await oauth_manager.get_or_create_session(request, server_name, oauth_config)

    # Check if user is already authenticated
    if await session.authenticate_if_needed():
        return session, False

    # User needs to authenticate
    return session, True


def create_oauth_authorization_url(request: Request, server_name: str, session_id: str, state: str) -> str:
    """Create OAuth authorization URL for the user"""
    # This would typically include the callback URL with session info
    base_url = f"{request.url.scheme}://{request.url.netloc}"
    callback_url = f"{base_url}/oauth/{server_name}/callback"

    params = {
        "session": session_id,
        "state": state,
        "redirect_uri": callback_url
    }

    auth_url = f"/oauth/{server_name}/authorize?" + urlencode(params)
    return auth_url