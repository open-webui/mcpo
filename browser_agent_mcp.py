#!/usr/bin/env python3
"""
Browser Agent MCP Server
Integrates the NAS Operator with MCP architecture for use with Claude Code
"""

import asyncio
import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from browser_use import Agent
    from browser_use.llm import ChatOpenAI
    from dotenv import load_dotenv
    from mcp.server import Server
    from mcp.types import (
        Resource,
        Tool,
        TextContent,
        ImageContent,
        EmbeddedResource,
        LoggingLevel
    )
    import mcp.server.stdio
    
except ImportError as e:
    logger.error(f"Missing dependencies: {e}")
    logger.info("Install: pip install browser-use python-dotenv mcp")
    exit(1)

class BrowserAgentMCP:
    """Browser Agent with MCP server integration"""
    
    def __init__(self):
        load_dotenv()
        
        # Validate API key
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY required")
        
        # Initialize LLM
        self.llm = ChatOpenAI(
            model="gpt-4o",
            api_key=api_key,
            temperature=0.1
        )
        
        # Setup directories
        self.base_dir = Path("./browser_agent_data")
        self.screenshots_dir = self.base_dir / "screenshots"
        self.recordings_dir = self.base_dir / "recordings"
        self.logs_dir = self.base_dir / "logs"
        
        for directory in [self.screenshots_dir, self.recordings_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Task tracking
        self.active_tasks = {}
        self.completed_tasks = []
        
        logger.info("🤖 Browser Agent MCP initialized")
    
    async def browse_web(self, task: str, max_steps: int = 15, save_gif: bool = True) -> Dict[str, Any]:
        """Execute web browsing task"""
        
        task_id = f"task_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        try:
            logger.info(f"🚀 Browser task: {task}")
            
            # Track active task
            self.active_tasks[task_id] = {
                "task": task,
                "start_time": datetime.now(),
                "status": "running"
            }
            
            # Create agent
            agent = Agent(
                task=task,
                llm=self.llm,
                use_vision=True,
                save_conversation_path=self.logs_dir / f"{task_id}_conversation.json",
                max_actions_per_step=max_steps,
                generate_gif=save_gif
            )
            
            # Execute task
            result = await agent.run()
            
            # Update task tracking
            task_info = self.active_tasks.pop(task_id)
            task_info.update({
                "end_time": datetime.now(),
                "status": "completed",
                "result": str(result) if result else "Completed"
            })
            self.completed_tasks.append(task_info)
            
            logger.info(f"✅ Browser task completed: {task_id}")
            
            return {
                "success": True,
                "task_id": task_id,
                "result": str(result) if result else "Task completed successfully",
                "conversation_log": str(self.logs_dir / f"{task_id}_conversation.json")
            }
            
        except Exception as e:
            # Update task tracking
            if task_id in self.active_tasks:
                task_info = self.active_tasks.pop(task_id)
                task_info.update({
                    "end_time": datetime.now(),
                    "status": "failed",
                    "error": str(e)
                })
                self.completed_tasks.append(task_info)
            
            logger.error(f"❌ Browser task failed: {e}")
            
            return {
                "success": False,
                "task_id": task_id,
                "error": str(e)
            }
    
    async def get_task_status(self, task_id: str = None) -> Dict[str, Any]:
        """Get status of tasks"""
        if task_id:
            # Check active tasks
            if task_id in self.active_tasks:
                return {"status": "active", "task": self.active_tasks[task_id]}
            
            # Check completed tasks
            for task in self.completed_tasks:
                if task_id in str(task):
                    return {"status": "completed", "task": task}
            
            return {"status": "not_found", "task_id": task_id}
        
        # Return overall status
        return {
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.completed_tasks),
            "active": list(self.active_tasks.keys()),
            "recent_completed": [str(t.get("task", ""))[:50] for t in self.completed_tasks[-5:]]
        }
    
    async def list_recordings(self) -> List[str]:
        """List available automation recordings"""
        recordings = []
        for file in self.recordings_dir.glob("*.gif"):
            recordings.append(str(file))
        return recordings
    
    async def list_screenshots(self) -> List[str]:
        """List available screenshots"""
        screenshots = []
        for file in self.screenshots_dir.glob("*.png"):
            screenshots.append(str(file))
        return screenshots

# Initialize browser agent
browser_agent = BrowserAgentMCP()

# Create MCP server
server = Server("browser-agent")

@server.list_tools()
async def handle_list_tools() -> List[Tool]:
    """List available browser automation tools"""
    return [
        Tool(
            name="browse_web",
            description="Execute web browsing and automation tasks using AI",
            inputSchema={
                "type": "object",
                "properties": {
                    "task": {
                        "type": "string",
                        "description": "Natural language description of the web task to perform"
                    },
                    "max_steps": {
                        "type": "integer",
                        "description": "Maximum number of automation steps (default: 15)",
                        "default": 15
                    },
                    "save_gif": {
                        "type": "boolean",
                        "description": "Whether to save automation as GIF (default: true)",
                        "default": True
                    }
                },
                "required": ["task"]
            }
        ),
        Tool(
            name="get_task_status",
            description="Get status of browser automation tasks",
            inputSchema={
                "type": "object",
                "properties": {
                    "task_id": {
                        "type": "string",
                        "description": "Optional task ID to check specific task status"
                    }
                }
            }
        ),
        Tool(
            name="list_recordings",
            description="List available automation recording files",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        ),
        Tool(
            name="list_screenshots",
            description="List available screenshot files",
            inputSchema={
                "type": "object",
                "properties": {}
            }
        )
    ]

@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> List[TextContent]:
    """Handle tool calls"""
    
    try:
        if name == "browse_web":
            result = await browser_agent.browse_web(
                task=arguments["task"],
                max_steps=arguments.get("max_steps", 15),
                save_gif=arguments.get("save_gif", True)
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "get_task_status":
            result = await browser_agent.get_task_status(
                task_id=arguments.get("task_id")
            )
            return [TextContent(type="text", text=json.dumps(result, indent=2, default=str))]
        
        elif name == "list_recordings":
            recordings = await browser_agent.list_recordings()
            return [TextContent(type="text", text=json.dumps(recordings, indent=2))]
        
        elif name == "list_screenshots":
            screenshots = await browser_agent.list_screenshots()
            return [TextContent(type="text", text=json.dumps(screenshots, indent=2))]
        
        else:
            return [TextContent(type="text", text=f"Unknown tool: {name}")]
    
    except Exception as e:
        logger.error(f"Tool call error: {e}")
        return [TextContent(type="text", text=f"Error: {str(e)}")]

@server.list_resources()
async def handle_list_resources() -> List[Resource]:
    """List available resources"""
    resources = []
    
    # Add conversation logs
    for log_file in browser_agent.logs_dir.glob("*.json"):
        resources.append(Resource(
            uri=f"file://{log_file}",
            name=f"Conversation Log: {log_file.stem}",
            description=f"Browser automation conversation log",
            mimeType="application/json"
        ))
    
    # Add screenshots
    for screenshot in browser_agent.screenshots_dir.glob("*.png"):
        resources.append(Resource(
            uri=f"file://{screenshot}",
            name=f"Screenshot: {screenshot.stem}",
            description=f"Browser automation screenshot",
            mimeType="image/png"
        ))
    
    return resources

@server.read_resource()
async def handle_read_resource(uri: str) -> str:
    """Read resource content"""
    try:
        if uri.startswith("file://"):
            file_path = Path(uri[7:])  # Remove file:// prefix
            
            if file_path.suffix == ".json":
                with open(file_path, 'r') as f:
                    return f.read()
            elif file_path.suffix == ".png":
                # Return base64 encoded image
                import base64
                with open(file_path, 'rb') as f:
                    encoded = base64.b64encode(f.read()).decode('utf-8')
                    return f"data:image/png;base64,{encoded}"
            else:
                with open(file_path, 'r') as f:
                    return f.read()
        
        return f"Unknown resource: {uri}"
    
    except Exception as e:
        return f"Error reading resource: {str(e)}"

async def main():
    """Run the MCP server"""
    # For testing, run a simple task
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "test":
        logger.info("🧪 Running test mode")
        result = await browser_agent.browse_web(
            "Go to google.com and search for 'browser automation'",
            max_steps=5
        )
        logger.info(f"Test result: {result}")
        return
    
    # Run MCP server
    async with mcp.server.stdio.stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream,
            write_stream,
            server.create_initialization_options()
        )

if __name__ == "__main__":
    logger.info("🚀 Starting Browser Agent MCP Server")
    asyncio.run(main())