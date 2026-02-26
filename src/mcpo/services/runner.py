"""
Runner service for managing tool execution, background tasks, and timeouts.
Centralizes all tool execution logic with proper error handling and state management.
"""

import asyncio
import json
import logging
import time
import threading
from typing import Dict, Any, Optional, Set
from contextlib import asynccontextmanager

from mcp.client.session import ClientSession
from mcp import types
from fastapi import HTTPException

logger = logging.getLogger(__name__)


class RunnerService:
    """Service for managing tool execution and background tasks."""
    
    def __init__(self):
        self._background_tasks: Set[asyncio.Task] = set()
        self._task_lock = threading.Lock()
        self._metrics: Dict[str, Dict[str, Any]] = {}
        self._metrics_lock = threading.Lock()
    
    def track_task(self, task: asyncio.Task) -> None:
        """Track a background task for cleanup."""
        with self._task_lock:
            self._background_tasks.add(task)
            # Add callback to remove completed tasks
            task.add_done_callback(self._remove_completed_task)
    
    def _remove_completed_task(self, task: asyncio.Task) -> None:
        """Remove completed task from tracking."""
        with self._task_lock:
            self._background_tasks.discard(task)
    
    async def cleanup_tasks(self) -> None:
        """Cancel all tracked background tasks."""
        with self._task_lock:
            tasks = list(self._background_tasks)
        
        if tasks:
            logger.info(f"Cancelling {len(tasks)} background tasks")
            for task in tasks:
                task.cancel()
            
            # Wait for tasks to cancel
            await asyncio.gather(*tasks, return_exceptions=True)
    
    async def execute_tool(
        self, 
        session: ClientSession,
        endpoint_name: str,
        arguments: Dict[str, Any],
        timeout: Optional[float] = None,
        max_timeout: Optional[float] = None
    ) -> Any:
        """
        Execute a tool with optional timeout and metrics tracking.
        
        Args:
            session: MCP client session
            endpoint_name: Name of the tool to execute
            arguments: Tool arguments
            timeout: Execution timeout in seconds
            max_timeout: Maximum allowed timeout
            
        Returns:
            Tool execution result
            
        Raises:
            HTTPException: On timeout or execution error
        """
        # Validate timeout (leave error standardization to handler)
        if timeout is not None and max_timeout is not None:
            if timeout > max_timeout:
                raise HTTPException(
                    status_code=400,
                    detail={
                        "message": "Timeout out of allowed range",
                        "code": "invalid_timeout"
                    }
                )
        
        start_time = time.time()
        
        try:
            if timeout:
                # Execute with timeout
                result = await asyncio.wait_for(
                    session.call_tool(endpoint_name, arguments=arguments),
                    timeout=timeout
                )
            else:
                # Execute without timeout
                result = await session.call_tool(endpoint_name, arguments=arguments)
            
            # Track metrics
            execution_time = time.time() - start_time
            self._update_metrics(endpoint_name, execution_time, success=True)
            
            # Process error results
            if result.isError:
                error_message = "Unknown tool execution error"
                error_data = None
                
                if result.content and isinstance(result.content[0], types.TextContent):
                    error_message = result.content[0].text
                    try:
                        error_data = json.loads(error_message)
                    except (json.JSONDecodeError, TypeError):
                        pass
                
                detail = {"message": error_message}
                if error_data is not None:
                    detail["data"] = error_data
                    
                raise HTTPException(status_code=500, detail=detail)
            
            # Process successful results
            # Local import to avoid circular import with mcpo.utils.main
            from mcpo.utils.main import process_tool_response
            response_data = process_tool_response(result)
            # Return primitives only; envelope building is done in handler
            return response_data[0] if len(response_data) == 1 else response_data
            
        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            self._update_metrics(endpoint_name, execution_time, success=False)
            logger.warning(f"Tool {endpoint_name} timed out after {timeout}s")
            
            raise HTTPException(
                status_code=504,
                detail={
                    "message": "Tool timed out",
                    "code": "timeout"
                }
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            self._update_metrics(endpoint_name, execution_time, success=False)
            
            # Re-raise HTTP exceptions as-is
            if isinstance(e, HTTPException):
                raise
            
            logger.error(f"Unexpected error executing {endpoint_name}: {e}", exc_info=True)
            raise HTTPException(
                status_code=500,
                detail={"message": "Unexpected error", "error": str(e)}
            )
    
    def _update_metrics(self, endpoint_name: str, execution_time: float, success: bool) -> None:
        """Update execution metrics for a tool."""
        with self._metrics_lock:
            if endpoint_name not in self._metrics:
                self._metrics[endpoint_name] = {
                    'calls': 0,
                    'totalLatency': 0.0,
                    'avgLatencyMs': 0.0,
                    'errors': 0
                }
            
            metrics = self._metrics[endpoint_name]
            metrics['calls'] += 1
            metrics['totalLatency'] += execution_time
            metrics['avgLatencyMs'] = (metrics['totalLatency'] / metrics['calls']) * 1000
            
            if not success:
                metrics['errors'] += 1
    
    def get_metrics(self) -> Dict[str, Dict[str, Any]]:
        """Get execution metrics for all tools."""
        with self._metrics_lock:
            return dict(self._metrics)
    
    def reset_metrics(self) -> None:
        """Reset all execution metrics."""
        with self._metrics_lock:
            self._metrics.clear()


# Global runner service instance
_runner_service: Optional[RunnerService] = None
_runner_lock = threading.Lock()


def get_runner_service() -> RunnerService:
    """Get or create the global runner service instance."""
    global _runner_service
    with _runner_lock:
        if _runner_service is None:
            _runner_service = RunnerService()
        return _runner_service


@asynccontextmanager
async def runner_lifespan():
    """Lifespan context manager for runner service cleanup."""
    runner = get_runner_service()
    try:
        yield runner
    finally:
        await runner.cleanup_tasks()
