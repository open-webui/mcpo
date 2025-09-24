import logging
import re
from typing import Dict, List, Optional, Any
from fastapi import Request

logger = logging.getLogger(__name__)


def validate_client_header_forwarding_config(server_name: str, config: Dict[str, Any]) -> None:
    """Validate client header forwarding configuration for a server."""
    if not isinstance(config, dict):
        raise ValueError(f"Server '{server_name}' client_header_forwarding must be a dictionary")
    
    enabled = config.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"Server '{server_name}' client_header_forwarding.enabled must be a boolean")
    
    if not enabled:
        return  # No further validation needed if disabled
    
    whitelist = config.get("whitelist", [])
    blacklist = config.get("blacklist", [])
    
    if whitelist and not isinstance(whitelist, list):
        raise ValueError(f"Server '{server_name}' client_header_forwarding.whitelist must be a list")
    
    if blacklist and not isinstance(blacklist, list):
        raise ValueError(f"Server '{server_name}' client_header_forwarding.blacklist must be a list")
    
    debug_headers = config.get("debug_headers", False)
    if not isinstance(debug_headers, bool):
        raise ValueError(f"Server '{server_name}' client_header_forwarding.debug_headers must be a boolean")


def match_header_pattern(header_name: str, patterns: List[str]) -> bool:
    """Check if header name matches any of the given patterns."""
    for pattern in patterns:
        if pattern == "*":
            return True
        if pattern.endswith("*"):
            # Wildcard pattern like "X-User-*"
            prefix = pattern[:-1]
            if header_name.startswith(prefix):
                return True
        elif pattern == header_name:
            return True
    return False


def filter_headers(
    request_headers: Dict[str, str], 
    whitelist: List[str],
    blacklist: List[str],
    debug_headers: bool = False
) -> Dict[str, str]:
    """Filter request headers based on whitelist and blacklist."""
    filtered_headers = {}
    
    for header_name, header_value in request_headers.items():
        # Skip if in blacklist
        if blacklist and match_header_pattern(header_name, blacklist):
            if debug_headers:
                logger.debug(f"Header '{header_name}' blocked by blacklist")
            continue
        
        # Include if in whitelist (or no whitelist specified)
        if not whitelist or match_header_pattern(header_name, whitelist):
            filtered_headers[header_name] = header_value
            if debug_headers:
                logger.debug(f"Header '{header_name}' forwarded")
        elif debug_headers:
            logger.debug(f"Header '{header_name}' not in whitelist")
    
    return filtered_headers


def process_headers_for_server(
    request: Request,
    header_config: Dict[str, Any]
) -> Dict[str, str]:
    """Process and filter headers for a specific MCP server."""
    if not header_config.get("enabled", False):
        return {}
    
    # Convert FastAPI headers to dict
    request_headers = dict(request.headers)
    
    # Get configuration values
    whitelist = header_config.get("whitelist", [])
    blacklist = header_config.get("blacklist", [])
    debug_headers = header_config.get("debug_headers", False)
    
    # Filter headers based on whitelist/blacklist
    filtered_headers = filter_headers(request_headers, whitelist, blacklist, debug_headers)
    
    if debug_headers:
        logger.debug(f"Final forwarded headers: {list(filtered_headers.keys())}")
    
    return filtered_headers
