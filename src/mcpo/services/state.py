"""
State management service for server and tool state persistence.
"""
import json
import os
import tempfile
import threading
from typing import Dict, Any, List
from pathlib import Path


class StateManager:
    """Manages server and tool states with thread-safe operations."""
    
    def __init__(self, state_file_path: str = "mcpo_state.json"):
        self.state_file_path = state_file_path
        self._server_states = {}  # {server_name: {"enabled": bool, "tools": {tool_name: bool}}}
        self._provider_states = {}  # {provider_id: {"enabled": bool}}
        self._model_states = {}  # {model_id: {"enabled": bool}}
        self._favorite_models: List[str] = []  # List of starred model IDs
        # Use a re-entrant lock because save_state is called from within other
        # lock-protected methods, and a standard Lock would deadlock.
        self._lock = threading.RLock()
        self.load_state()
    
    def load_state(self) -> Dict[str, Any]:
        """Load server and tool states from state file."""
        with self._lock:
            try:
                if Path(self.state_file_path).exists():
                    with open(self.state_file_path, 'r') as f:
                        data = json.load(f)
                        self._server_states = data.get('server_states', {})
                        self._provider_states = data.get('provider_states', {})
                        self._model_states = data.get('model_states', {})
                        self._favorite_models = data.get('favorite_models', [])
                else:
                    self._server_states = {}
                    self._provider_states = {}
                    self._model_states = {}
                    self._favorite_models = []
            except Exception:
                # If loading fails, start with empty state
                self._server_states = {}
                self._provider_states = {}
                self._model_states = {}
                self._favorite_models = []
            return self._server_states
    
    def save_state(self) -> None:
        """Save server and tool states to state file."""
        with self._lock:
            try:
                state_data = {
                    "server_states": self._server_states,
                    "provider_states": self._provider_states,
                    "model_states": self._model_states,
                    "favorite_models": self._favorite_models,
                }
                dir_path = os.path.dirname(os.path.abspath(self.state_file_path))
                fd, tmp_path = tempfile.mkstemp(dir=dir_path, suffix='.tmp')
                try:
                    with os.fdopen(fd, 'w') as f:
                        json.dump(state_data, f, indent=2)
                    os.replace(tmp_path, self.state_file_path)
                except:
                    os.unlink(tmp_path)
                    raise
            except Exception as e:
                # Log error but don't crash
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Failed to save state: {e}")
    
    def get_server_state(self, server_name: str) -> Dict[str, Any]:
        """Get state for a specific server."""
        with self._lock:
            return self._server_states.get(server_name, {
                "enabled": True,
                "tools": {}
            })
    
    def set_server_enabled(self, server_name: str, enabled: bool) -> None:
        """Set server enabled state."""
        with self._lock:
            if server_name not in self._server_states:
                self._server_states[server_name] = {"enabled": enabled, "tools": {}}
            else:
                self._server_states[server_name]["enabled"] = enabled
            self.save_state()
    
    def set_tool_enabled(self, server_name: str, tool_name: str, enabled: bool) -> None:
        """Set tool enabled state for a specific server."""
        with self._lock:
            if server_name not in self._server_states:
                self._server_states[server_name] = {"enabled": True, "tools": {}}
            self._server_states[server_name]["tools"][tool_name] = enabled
            self.save_state()
    
    def is_server_enabled(self, server_name: str) -> bool:
        """Check if server is enabled."""
        with self._lock:
            return self._server_states.get(server_name, {}).get("enabled", True)
    
    def is_tool_enabled(self, server_name: str, tool_name: str) -> bool:
        """Check if tool is enabled."""
        with self._lock:
            server_state = self._server_states.get(server_name, {})
            return server_state.get("tools", {}).get(tool_name, True)
    
    def get_all_states(self) -> Dict[str, Any]:
        """Get all server states."""
        with self._lock:
            return dict(self._server_states)

    # --- Providers ---
    def get_provider_state(self, provider_id: str) -> Dict[str, Any]:
        with self._lock:
            return self._provider_states.get(provider_id, {"enabled": True})

    def set_provider_enabled(self, provider_id: str, enabled: bool) -> None:
        with self._lock:
            self._provider_states[provider_id] = {"enabled": bool(enabled)}
            self.save_state()

    def is_provider_enabled(self, provider_id: str) -> bool:
        with self._lock:
            return self._provider_states.get(provider_id, {}).get("enabled", True)

    def get_all_provider_states(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._provider_states)

    # --- Models ---
    def is_model_enabled(self, model_id: str) -> bool:
        with self._lock:
            return self._model_states.get(model_id, {}).get("enabled", True)

    def set_model_enabled(self, model_id: str, enabled: bool) -> None:
        with self._lock:
            self._model_states[model_id] = {"enabled": bool(enabled)}
            self.save_state()

    def get_all_model_states(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._model_states)

    # --- Favorite Models ---
    def get_favorite_models(self) -> List[str]:
        """Get list of favorite model IDs."""
        with self._lock:
            return list(self._favorite_models)

    def set_favorite_models(self, model_ids: List[str]) -> None:
        """Set the entire list of favorite model IDs."""
        with self._lock:
            self._favorite_models = list(model_ids)
            self.save_state()

    def add_favorite_model(self, model_id: str) -> None:
        """Add a model to favorites."""
        with self._lock:
            if model_id not in self._favorite_models:
                self._favorite_models.append(model_id)
                self.save_state()

    def remove_favorite_model(self, model_id: str) -> None:
        """Remove a model from favorites."""
        with self._lock:
            if model_id in self._favorite_models:
                self._favorite_models.remove(model_id)
                self.save_state()

    def is_model_favorite(self, model_id: str) -> bool:
        """Check if a model is in favorites."""
        with self._lock:
            return model_id in self._favorite_models


# Global instance for backward compatibility
_global_state_manager = None
_global_state_lock = threading.Lock()

def get_state_manager(state_file_path: str = "mcpo_state.json") -> StateManager:
    """Get or create global state manager instance."""
    global _global_state_manager
    if _global_state_manager is None:
        with _global_state_lock:
            if _global_state_manager is None:
                _global_state_manager = StateManager(state_file_path)
    return _global_state_manager
