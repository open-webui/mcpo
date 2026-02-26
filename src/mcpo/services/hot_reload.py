"""
Hot reload service for watching configuration file changes and triggering reloads.
Manages file system monitoring and reload callbacks in a centralized manner.
"""

import asyncio
import logging
import threading
import time
from pathlib import Path
from typing import Optional, Callable, Dict, Any, TYPE_CHECKING
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler, FileModifiedEvent

if TYPE_CHECKING:
    from watchdog.observers import Observer

logger = logging.getLogger(__name__)


class ConfigFileHandler(FileSystemEventHandler):
    """File system event handler for configuration file changes."""
    
    def __init__(self, config_path: str, reload_callback: Callable):
        self.config_path = Path(config_path).resolve()
        self.reload_callback = reload_callback
        self.last_reload_time = 0
        self.min_reload_interval = 1.0  # Minimum seconds between reloads
    
    def on_modified(self, event):
        """Handle file modification events."""
        if isinstance(event, FileModifiedEvent):
            event_path = Path(event.src_path).resolve()
            
            # Check if this is our config file
            if event_path == self.config_path:
                current_time = time.time()
                
                # Debounce rapid file changes
                if current_time - self.last_reload_time >= self.min_reload_interval:
                    self.last_reload_time = current_time
                    logger.info(f"Config file modified: {self.config_path}")
                    
                    try:
                        # Schedule callback in event loop if available
                        try:
                            loop = asyncio.get_running_loop()
                            loop.call_soon_threadsafe(
                                lambda: asyncio.create_task(self._safe_reload())
                            )
                        except RuntimeError:
                            # No event loop, run in a new loop for this thread
                            loop = asyncio.new_event_loop()
                            try:
                                loop.run_until_complete(self._safe_reload())
                            finally:
                                loop.close()
                    except Exception as e:
                        logger.error(f"Error scheduling config reload: {e}")
    
    async def _safe_reload(self):
        """Safely execute the reload callback with error handling."""
        try:
            if asyncio.iscoroutinefunction(self.reload_callback):
                await self.reload_callback()
            else:
                self.reload_callback()
        except Exception as e:
            logger.error(f"Error during config reload: {e}")


class HotReloadService:
    """Service for managing hot reload functionality."""
    
    def __init__(self):
        self._observer = None  # Type: Optional[Observer]
        self._observer_lock = threading.Lock()
        self._watchers: Dict[str, ConfigFileHandler] = {}
    
    def start_watching(self, config_path: str, reload_callback: Callable) -> bool:
        """
        Start watching a configuration file for changes.
        
        Args:
            config_path: Path to configuration file to watch
            reload_callback: Function to call when file changes
            
        Returns:
            True if watching started successfully, False otherwise
        """
        try:
            config_file = Path(config_path)
            if not config_file.exists():
                logger.warning(f"Config file does not exist: {config_path}")
                return False
            
            # Create parent directory watcher
            watch_dir = config_file.parent
            
            with self._observer_lock:
                # Initialize observer if needed
                if self._observer is None:
                    self._observer = Observer()
                    self._observer.start()
                
                # Create file handler
                handler = ConfigFileHandler(config_path, reload_callback)
                self._watchers[config_path] = handler
                
                # Start watching directory
                self._observer.schedule(handler, str(watch_dir), recursive=False)
                
                logger.info(f"Started watching config file: {config_path}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to start watching {config_path}: {e}")
            return False
    
    def stop_watching(self, config_path: Optional[str] = None) -> None:
        """
        Stop watching configuration file(s).
        
        Args:
            config_path: Specific file to stop watching, or None to stop all
        """
        with self._observer_lock:
            if config_path:
                # Stop watching specific file
                if config_path in self._watchers:
                    del self._watchers[config_path]
                    logger.info(f"Stopped watching config file: {config_path}")
            else:
                # Stop all watching
                if self._observer:
                    self._observer.stop()
                    self._observer.join()
                    self._observer = None
                    self._watchers.clear()
                    logger.info("Stopped all config file watching")
    
    def is_watching(self, config_path: str) -> bool:
        """Check if a specific config file is being watched."""
        return config_path in self._watchers
    
    def get_watched_files(self) -> list[str]:
        """Get list of currently watched config files."""
        return list(self._watchers.keys())


# Global hot reload service instance
_hot_reload_service: Optional[HotReloadService] = None
_hot_reload_lock = threading.Lock()


def get_hot_reload_service() -> HotReloadService:
    """Get or create the global hot reload service instance."""
    global _hot_reload_service
    with _hot_reload_lock:
        if _hot_reload_service is None:
            _hot_reload_service = HotReloadService()
        return _hot_reload_service


def start_config_watching(config_path: str, reload_callback: Callable) -> bool:
    """Convenience function to start watching a config file."""
    service = get_hot_reload_service()
    return service.start_watching(config_path, reload_callback)


def stop_config_watching(config_path: Optional[str] = None) -> None:
    """Convenience function to stop watching config file(s)."""
    service = get_hot_reload_service()
    service.stop_watching(config_path)
