"""
Code Mode service for OpenHubUI/MCPO.

When code mode is enabled, instead of exposing all MCP tools directly
(which can overwhelm the LLM's context with 50+ tool schemas), the proxy
exposes only two meta-tools:

  1. search_tools - Search the tool catalog by keyword/tag
  2. execute_tool - Execute a specific tool by its fully qualified name

The LLM uses search_tools to discover relevant tools, then execute_tool
to invoke them. This keeps the tool count constant regardless of how many
MCP servers are configured.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tool catalog entry
# ---------------------------------------------------------------------------

@dataclass
class CatalogEntry:
    """A single tool in the code mode catalog."""
    server_name: str
    tool_name: str
    qualified_name: str  # e.g. "brave-search.web_search"
    description: str
    input_schema: Dict[str, Any]
    tags: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Catalog builder
# ---------------------------------------------------------------------------

def _extract_tags(name: str, description: str) -> List[str]:
    """Extract searchable tags from tool name and description."""
    tags: List[str] = []
    # Split name on common separators
    parts = re.split(r"[._\-/]", name.lower())
    tags.extend(p for p in parts if len(p) > 1)
    # Extract meaningful words from description
    if description:
        words = re.findall(r"[a-zA-Z]{3,}", description.lower())
        tags.extend(words[:20])  # cap to avoid bloat
    return list(set(tags))


def build_catalog(tools_by_server: Dict[str, List[Dict[str, Any]]]) -> List[CatalogEntry]:
    """
    Build a searchable catalog from MCP tool listings.

    Args:
        tools_by_server: {server_name: [tool_dict, ...]} where tool_dict
            has 'name', 'description', 'inputSchema' keys (MCP tool format).
    """
    catalog: List[CatalogEntry] = []
    for server_name, tools in tools_by_server.items():
        for tool in tools:
            tool_name = tool.get("name", "")
            qualified = f"{server_name}.{tool_name}"
            desc = tool.get("description", "")
            schema = tool.get("inputSchema") or {"type": "object", "properties": {}}
            tags = _extract_tags(qualified, desc)
            catalog.append(CatalogEntry(
                server_name=server_name,
                tool_name=tool_name,
                qualified_name=qualified,
                description=desc,
                input_schema=schema,
                tags=tags,
            ))
    return catalog


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------

def search_catalog(
    catalog: List[CatalogEntry],
    query: str,
    limit: int = 10,
) -> List[Dict[str, Any]]:
    """
    Search the tool catalog by keyword matching against names, descriptions, and tags.

    Returns a list of result dicts suitable for returning to the LLM.
    """
    if not query or not query.strip():
        # Return all (up to limit)
        results = catalog[:limit]
    else:
        q = query.strip().lower()
        terms = q.split()

        scored: List[Tuple[float, CatalogEntry]] = []
        for entry in catalog:
            score = 0.0
            text = f"{entry.qualified_name} {entry.description} {' '.join(entry.tags)}".lower()

            for term in terms:
                # Exact name match is highest signal
                if term in entry.qualified_name.lower():
                    score += 10.0
                # Server name match
                if term in entry.server_name.lower():
                    score += 5.0
                # Tag match
                if any(term in tag for tag in entry.tags):
                    score += 3.0
                # Description match
                if term in entry.description.lower():
                    score += 1.0

            if score > 0:
                scored.append((score, entry))

        scored.sort(key=lambda x: -x[0])
        results = [entry for _, entry in scored[:limit]]

    return [
        {
            "tool": entry.qualified_name,
            "server": entry.server_name,
            "description": entry.description,
            "parameters": entry.input_schema,
        }
        for entry in results
    ]


# ---------------------------------------------------------------------------
# Meta-tool definitions (MCP tool schema format)
# ---------------------------------------------------------------------------

SEARCH_TOOLS_SCHEMA: Dict[str, Any] = {
    "name": "search_tools",
    "description": (
        "Search the available tool catalog by keyword. "
        "Use this to discover which tools are available before calling execute_tool. "
        "Returns tool names, descriptions, and parameter schemas."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": (
                    "Search query - use keywords like server name, tool purpose, "
                    "or action type. Examples: 'perplexity', 'search', 'github issues', "
                    "'brave web'. Leave empty to list all tools."
                ),
            },
            "limit": {
                "type": "integer",
                "description": "Maximum number of results to return (default 10).",
                "default": 10,
            },
        },
        "required": [],
    },
}

EXECUTE_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "execute_tool",
    "description": (
        "Execute a tool by its fully qualified name (server.tool_name). "
        "Use search_tools first to discover available tools and their parameter schemas, "
        "then call this with the tool name and arguments."
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "tool": {
                "type": "string",
                "description": (
                    "Fully qualified tool name in 'server_name.tool_name' format, "
                    "as returned by search_tools."
                ),
            },
            "arguments": {
                "type": "object",
                "description": "Arguments to pass to the tool, matching its parameter schema.",
                "additionalProperties": True,
            },
        },
        "required": ["tool"],
    },
}


def get_code_mode_tool_definitions() -> List[Dict[str, Any]]:
    """Return the two meta-tool definitions for code mode."""
    return [SEARCH_TOOLS_SCHEMA, EXECUTE_TOOL_SCHEMA]
