#!/usr/bin/env python3
"""
MCP Server for Mem0 integration with ElevenLabs
Implements the Model Context Protocol for tool communication
"""

import os
import json
import asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse
from mem0 import MemoryClient
import uuid
from pydantic import BaseModel

# Initialize FastAPI
app = FastAPI(title="Mem0 MCP Server")

# Initialize Mem0 client
mem0_client = MemoryClient(
    api_key=os.environ.get('MEM0_API_KEY', 'm0-IQGqsMWB42QhWG77RuzpSdNcyEppgRHeBhz0KcNu')
)

# Configuration
USER_ID = os.environ.get('USER_ID', 'quinn_may')
CLIENT = 'ElevenLabs'
PROJECT_TYPE = 'voice_agent'
DEVICE = 'mcp_server'

class MCPRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = {}
    id: Optional[str] = None

class MCPResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[Dict[str, Any]] = None
    id: Optional[str] = None

async def handle_initialize(request_id: str) -> dict:
    """Handle MCP initialize request"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "protocolVersion": "0.1.0",
            "capabilities": {
                "tools": {
                    "listChanged": True
                },
                "prompts": {
                    "listChanged": False
                }
            },
            "serverInfo": {
                "name": "mem0-mcp-server",
                "version": "1.0.0"
            }
        }
    }

async def handle_tools_list(request_id: str) -> dict:
    """Return available tools"""
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "tools": [
                {
                    "name": "store_memory",
                    "description": "Store important information from conversations in long-term memory",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "message": {
                                "type": "string",
                                "description": "The information to remember"
                            },
                            "category": {
                                "type": "string",
                                "description": "Category of memory (optional)",
                                "enum": ["personal", "work", "preference", "context", "general"]
                            }
                        },
                        "required": ["message"]
                    }
                },
                {
                    "name": "search_memory",
                    "description": "Search for previously stored memories",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Number of results (default 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"]
                    }
                },
                {
                    "name": "get_recent_memories",
                    "description": "Get the most recent memories",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "limit": {
                                "type": "integer",
                                "description": "Number of memories to retrieve",
                                "default": 10
                            }
                        }
                    }
                }
            ]
        }
    }

async def handle_tool_call(params: dict, request_id: str) -> dict:
    """Handle tool execution"""
    tool_name = params.get("name")
    arguments = params.get("arguments", {})
    
    try:
        if tool_name == "store_memory":
            message = arguments.get("message")
            category = arguments.get("category", "general")
            
            if not message:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: message is required"
                    }
                }
            
            # Store in Mem0
            now = datetime.now()
            metadata = {
                "category": category,
                "day": now.strftime("%Y-%m-%d"),
                "month": now.strftime("%Y-%m"),
                "year": now.strftime("%Y"),
                "client": CLIENT,
                "project_type": PROJECT_TYPE,
                "device": DEVICE,
                "timestamp": now.isoformat(),
                "source": "mcp_server"
            }
            
            result = mem0_client.add(
                messages=[{"role": "user", "content": message}],
                user_id=USER_ID,
                metadata=metadata
            )
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"✅ Memory stored successfully with ID: {result.get('id', 'unknown')}"
                        }
                    ],
                    "isError": False
                }
            }
            
        elif tool_name == "search_memory":
            query = arguments.get("query")
            limit = arguments.get("limit", 5)
            
            if not query:
                return {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "error": {
                        "code": -32602,
                        "message": "Invalid params: query is required"
                    }
                }
            
            # Search memories
            results = mem0_client.search(
                query=query,
                user_id=USER_ID,
                limit=limit
            )
            
            memories_text = "\n".join([
                f"• {mem.get('memory', '')}" 
                for mem in results.get('results', [])
            ])
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Found {len(results.get('results', []))} memories:\n{memories_text}"
                        }
                    ],
                    "isError": False
                }
            }
            
        elif tool_name == "get_recent_memories":
            limit = arguments.get("limit", 10)
            
            # Get all memories
            all_memories = mem0_client.get_all(
                user_id=USER_ID
            )
            
            # Sort by created_at and get recent ones
            sorted_memories = sorted(
                all_memories,
                key=lambda x: x.get('created_at', ''),
                reverse=True
            )[:limit]
            
            memories_text = "\n".join([
                f"• {mem.get('memory', '')}" 
                for mem in sorted_memories
            ])
            
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": f"Recent memories:\n{memories_text}"
                        }
                    ],
                    "isError": False
                }
            }
            
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {tool_name}"
                }
            }
            
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32603,
                "message": f"Internal error: {str(e)}"
            }
        }

async def sse_generator(request: Request):
    """Generate SSE events for MCP protocol"""
    # Send initial connection event
    yield f"data: {json.dumps({'type': 'connection', 'message': 'MCP Server Connected'})}\n\n"
    
    # Keep connection alive
    while True:
        # Wait for incoming data
        if await request.is_disconnected():
            break
        
        # Send heartbeat
        await asyncio.sleep(30)
        yield f"data: {json.dumps({'type': 'heartbeat'})}\n\n"

@app.post("/mcp")
async def mcp_endpoint(request: Request):
    """Main MCP endpoint for ElevenLabs"""
    try:
        body = await request.json()
        
        # Parse MCP request
        method = body.get("method")
        params = body.get("params", {})
        request_id = body.get("id", str(uuid.uuid4()))
        
        # Route to appropriate handler
        if method == "initialize":
            response = await handle_initialize(request_id)
        elif method == "tools/list":
            response = await handle_tools_list(request_id)
        elif method == "tools/call":
            response = await handle_tool_call(params, request_id)
        else:
            response = {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}"
                }
            }
        
        return response
        
    except Exception as e:
        return {
            "jsonrpc": "2.0",
            "id": body.get("id") if "body" in locals() else None,
            "error": {
                "code": -32700,
                "message": f"Parse error: {str(e)}"
            }
        }

@app.get("/mcp")
async def mcp_sse_endpoint(request: Request):
    """SSE endpoint for MCP protocol"""
    return StreamingResponse(
        sse_generator(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "mem0-mcp-server"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)