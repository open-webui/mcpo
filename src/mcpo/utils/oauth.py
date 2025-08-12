import asyncio
import json
import logging
import os
import hashlib
from typing import Optional, Dict, Any, Tuple
from pathlib import Path
import webbrowser
import time
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

from mcp.client.auth import OAuthClientProvider, TokenStorage
from mcp.shared.auth import OAuthClientInformationFull, OAuthClientMetadata, OAuthToken
from pydantic import AnyUrl

logger = logging.getLogger(__name__)

def _load_callback_html(status: str, title: str, heading: str, message: str, action_text: str) -> str:
    """Load and render the OAuth callback HTML template"""
    template_path = Path(__file__).parent / "oauth_callback.html"
    
    try:
        logger.debug(f"Loading OAuth template from: {template_path}")
        logger.debug(f"Template exists: {template_path.exists()}")
        
        with open(template_path, 'r', encoding='utf-8') as f:
            template = f.read()
        
        logger.debug(f"Template loaded, length: {len(template)}")
        
        # Define icons and status classes
        icon = "✓" if status == "success" else "✕"
        status_class = status
        
        # Replace template variables
        html = template.format(
            title=title,
            status_class=status_class,
            icon=icon,
            heading=heading,
            message=message,
            action_text=action_text
        )
        
        logger.debug("Template rendered successfully")
        return html
    except Exception as e:
        logger.error(f"Failed to load OAuth callback template from {template_path}: {e}")
        logger.error(f"Template path exists: {template_path.exists()}")
        # Fallback to simple HTML
        return f"<html><body><h2>{heading}</h2><p>{message}</p></body></html>"

class InMemoryTokenStorage(TokenStorage):
    """Simple in-memory token storage per server instance"""
    def __init__(self, server_name: str):
        self.server_name = server_name
        self.tokens: Optional[OAuthToken] = None
        self.client_info: Optional[OAuthClientInformationFull] = None
        
    async def get_tokens(self) -> Optional[OAuthToken]:
        return self.tokens
        
    async def set_tokens(self, tokens: OAuthToken) -> None:
        self.tokens = tokens
        logger.info(f"OAuth tokens stored for server: {self.server_name}")
        
    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        return self.client_info
        
    async def set_client_info(self, info: OAuthClientInformationFull) -> None:
        self.client_info = info


class FileTokenStorage(TokenStorage):
    """File-based token storage with per-server isolation"""
    def __init__(self, server_name: str, storage_dir: str = None):
        self.server_name = server_name
        # Use hash of server name to avoid filesystem issues
        safe_name = hashlib.md5(server_name.encode()).hexdigest()[:8]
        if storage_dir is None:
            storage_dir = os.path.expanduser("~/.mcpo/tokens")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.token_file = self.storage_dir / f"{safe_name}_tokens.json"
        self.client_file = self.storage_dir / f"{safe_name}_client.json"
        
    async def get_tokens(self) -> Optional[OAuthToken]:
        if self.token_file.exists():
            try:
                with open(self.token_file, 'r') as f:
                    data = json.load(f)
                    return OAuthToken.model_validate(data)
            except Exception as e:
                logger.error(f"Failed to load tokens for {self.server_name}: {e}")
        return None
        
    async def set_tokens(self, tokens: OAuthToken) -> None:
        try:
            with open(self.token_file, 'w') as f:
                json.dump(tokens.model_dump(mode='json'), f)
            logger.info(f"OAuth tokens persisted for server: {self.server_name}")
        except Exception as e:
            logger.error(f"Failed to save tokens for {self.server_name}: {e}")
            
    async def get_client_info(self) -> Optional[OAuthClientInformationFull]:
        if self.client_file.exists():
            try:
                with open(self.client_file, 'r') as f:
                    data = json.load(f)
                    return OAuthClientInformationFull.model_validate(data)
            except Exception as e:
                logger.error(f"Failed to load client info for {self.server_name}: {e}")
        return None
        
    async def set_client_info(self, info: OAuthClientInformationFull) -> None:
        try:
            with open(self.client_file, 'w') as f:
                json.dump(info.model_dump(mode='json'), f)
        except Exception as e:
            logger.error(f"Failed to save client info for {self.server_name}: {e}")


class CallbackHandler(BaseHTTPRequestHandler):
    """HTTP handler for OAuth callbacks"""
    def __init__(self, request, client_address, server, data):
        self.data = data
        super().__init__(request, client_address, server)
        
    def do_GET(self):
        q = parse_qs(urlparse(self.path).query)
        if "code" in q:
            self.data["authorization_code"] = q["code"][0]
            self.data["state"] = q.get("state", [None])[0]
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = _load_callback_html(
                status="success",
                title="Authorization Successful - MCPO",
                heading="Authorization Successful!",
                message="Your OAuth authorization was completed successfully. The application can now access the requested resources.",
                action_text="You can safely close this browser tab and return to your application."
            )
            self.wfile.write(html.encode('utf-8'))
            
        elif "error" in q:
            error_desc = q.get("error_description", [q["error"][0]])[0]
            self.data["error"] = q["error"][0]
            self.send_response(400)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            
            html = _load_callback_html(
                status="error",
                title="Authorization Failed - MCPO",
                heading="Authorization Failed",
                message=f"The OAuth authorization process encountered an error: {error_desc}",
                action_text="Please close this tab and check the application logs for more details."
            )
            self.wfile.write(html.encode('utf-8'))
            
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.end_headers()
            
    def log_message(self, *_):
        pass  # Suppress request logs


class CallbackServer:
    """Local HTTP server for OAuth callbacks"""
    def __init__(self, port: int = 3030):
        self.port = port
        self.server = None
        self.thread = None
        self.data = {"authorization_code": None, "state": None, "error": None}
        
    def _handler(self):
        data = self.data
        class H(CallbackHandler):
            def __init__(self, req, addr, srv):
                super().__init__(req, addr, srv, data)
        return H
        
    def start(self):
        self.server = HTTPServer(("localhost", self.port), self._handler())
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        logger.info(f"OAuth callback server listening on http://localhost:{self.port}/callback")
        
    def stop(self):
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(timeout=1)
            
    def wait_code(self, timeout: int = 300) -> str:
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.data["authorization_code"]:
                return self.data["authorization_code"]
            if self.data["error"]:
                raise RuntimeError(f"OAuth error: {self.data['error']}")
            time.sleep(0.05)
        raise TimeoutError("No OAuth callback received within timeout")
        
    def state(self) -> Optional[str]:
        return self.data["state"]


async def create_oauth_provider(
    server_name: str,
    oauth_config: Dict[str, Any],
    storage_type: str = "file"
) -> OAuthClientProvider:
    """Create an OAuth provider for a server"""
    
    # Extract OAuth configuration
    server_url = oauth_config.get("server_url")
    if not server_url:
        raise ValueError(f"OAuth server_url required for {server_name}")
        
    # Build client metadata - only set defaults for required fields
    metadata_dict = oauth_config.get("client_metadata", {})
    if not metadata_dict.get("client_name"):
        metadata_dict["client_name"] = f"MCPO Client for {server_name}"
    if not metadata_dict.get("redirect_uris"):
        callback_port = oauth_config.get("callback_port", 3030)
        metadata_dict["redirect_uris"] = [f"http://localhost:{callback_port}/callback"]
    if not metadata_dict.get("grant_types"):
        metadata_dict["grant_types"] = ["authorization_code", "refresh_token"]
    if not metadata_dict.get("response_types"):
        metadata_dict["response_types"] = ["code"]
        
    # Don't set scope by default - let dynamic registration handle it
    # Don't set authorization_endpoint or token_endpoint - these are discovered
        
    # Convert redirect_uris to AnyUrl
    redirect_uris = [AnyUrl(uri) for uri in metadata_dict["redirect_uris"]]
    metadata_dict["redirect_uris"] = redirect_uris
    
    client_metadata = OAuthClientMetadata.model_validate(metadata_dict)
    
    # Choose storage backend
    if storage_type == "memory":
        storage = InMemoryTokenStorage(server_name)
    else:
        storage = FileTokenStorage(server_name)
    
    # Setup callback handling
    use_loopback = oauth_config.get("use_loopback", True)
    callback_port = oauth_config.get("callback_port", 3030)
    
    if use_loopback:
        # Loopback server for automatic callback handling
        callback_server = CallbackServer(callback_port)
        
        async def redirect_handler(url: str) -> None:
            logger.info(f"Opening browser for OAuth: {url}")
            webbrowser.open(url)
            
        async def callback_handler() -> Tuple[str, Optional[str]]:
            callback_server.start()
            try:
                code = callback_server.wait_code()
                return code, callback_server.state()
            finally:
                callback_server.stop()
    else:
        # Manual copy/paste flow
        async def redirect_handler(url: str) -> None:
            print(f"\n\nPlease visit this URL to authorize:\n{url}\n")
            
        async def callback_handler() -> Tuple[str, Optional[str]]:
            callback_url = input("Paste the callback URL here: ")
            q = parse_qs(urlparse(callback_url).query)
            code = q.get("code", [None])[0]
            state = q.get("state", [None])[0]
            if not code:
                raise ValueError("No authorization code found in callback URL")
            return code, state
    
    return OAuthClientProvider(
        server_url=server_url,
        client_metadata=client_metadata,
        storage=storage,
        redirect_handler=redirect_handler,
        callback_handler=callback_handler,
    )