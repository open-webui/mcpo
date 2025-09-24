def normalize_server_type(server_type: str) -> str:
    """Normalize server_type to a standard value."""
    if server_type in ["streamable_http", "streamablehttp", "streamable-http"]:
        return "streamable-http"
    return server_type