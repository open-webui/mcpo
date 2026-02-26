"""
Configuration utilities for environment variable interpolation.
"""
import json
import os
import re
from typing import Dict, Any, Optional


def interpolate_env_placeholders(config_env: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Replace ${VARNAME} in string values with os.environ values."""
    if not isinstance(config_env, dict):
        return {}

    pattern = re.compile(r"\$\{([A-Za-z0-9_]+)\}")

    result = dict(config_env)
    for k, v in list(result.items()):
        if isinstance(v, str):
            def repl(m):
                var = m.group(1)
                return os.environ.get(var, "")
            result[k] = pattern.sub(repl, v)
    return result


def interpolate_env_placeholders_in_config(config_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Walk mcpServers[..].env/headers and expand ${VAR} placeholders in-place copy."""
    if not isinstance(config_obj, dict):
        return config_obj
    cfg = json.loads(json.dumps(config_obj))  # deep copy
    servers = cfg.get("mcpServers", {})
    for name, server in list(servers.items()):
        if not isinstance(server, dict):
            continue
        env = server.get("env")
        if isinstance(env, dict):
            servers[name]["env"] = interpolate_env_placeholders(env)
        headers = server.get("headers")
        if isinstance(headers, dict):
            servers[name]["headers"] = interpolate_env_placeholders(headers)
    return cfg


def normalize_config_shape(config_obj: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize accepted config shapes to include top-level mcpServers.

    Supports both:
    - {"mcpServers": {...}}
    - {"config": {"mcpServers": {...}}, ...}
    """
    if not isinstance(config_obj, dict):
        return config_obj

    if "mcpServers" in config_obj:
        return config_obj

    nested = config_obj.get("config")
    if isinstance(nested, dict) and isinstance(nested.get("mcpServers"), dict):
        normalized = dict(config_obj)
        normalized["mcpServers"] = nested["mcpServers"]
        return normalized

    return config_obj
