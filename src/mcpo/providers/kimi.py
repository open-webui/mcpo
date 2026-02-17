"""
Kimi (Moonshot) provider with selectable endpoint mode.

Modes:
1) Coding plan / Anthropic-compatible
2) Regular OpenAI-compatible

Model IDs are env-driven only (KIMI_MODELS / KIMI_MODEL).
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from mcpo.providers.anthropic import AnthropicClient, AnthropicError
from mcpo.providers.openrouter import OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)

KIMI_ANTHROPIC_BASE_URL = "https://api.moonshot.ai/anthropic"
KIMI_OPENAI_BASE_URL = "https://api.moonshot.ai/v1"


class KimiError(RuntimeError):
    """Raised when Kimi provider configuration or request handling fails."""


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _kimi_api_key_from_env() -> Optional[str]:
    return os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")


def get_kimi_models() -> List[Dict[str, str]]:
    models: List[Dict[str, str]] = []
    raw_list = os.getenv("KIMI_MODELS")
    if raw_list:
        for item in raw_list.split(","):
            model_id = item.strip()
            if model_id:
                if not (model_id.startswith("kimi/") or model_id.startswith("moonshot/")):
                    model_id = f"kimi/{model_id}"
                models.append({"id": model_id, "label": f"Kimi: {model_id}"})
    preferred = os.getenv("KIMI_MODEL")
    if preferred and not (preferred.startswith("kimi/") or preferred.startswith("moonshot/")):
        preferred = f"kimi/{preferred}"
    if preferred and not any(m["id"] == preferred for m in models):
        models.insert(0, {"id": preferred, "label": f"Kimi: {preferred}"})
    return models


def is_kimi_model(model_id: str) -> bool:
    if not model_id:
        return False
    m = model_id.lower()
    return m.startswith("kimi/") or m.startswith("moonshot/")


def get_kimi_model_name(model_id: str) -> str:
    if model_id.startswith("kimi/"):
        return model_id.split("/", 1)[1]
    if model_id.startswith("moonshot/"):
        return model_id.split("/", 1)[1]
    return model_id


class KimiClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._api_key = api_key or _kimi_api_key_from_env()
        if not self._api_key:
            raise KimiError("KIMI_API_KEY or MOONSHOT_API_KEY environment variable is required")

        self._use_coding_plan = _env_flag("KIMI_USE_CODING_PLAN_ENDPOINT", False)
        regular_base = os.getenv("KIMI_REGULAR_BASE_URL", KIMI_OPENAI_BASE_URL)
        coding_base = os.getenv("KIMI_CODING_PLAN_BASE_URL", KIMI_ANTHROPIC_BASE_URL)
        mode_default = coding_base if self._use_coding_plan else regular_base
        self._base_url = (base_url or os.getenv("KIMI_BASE_URL") or mode_default).rstrip("/")
        self._timeout = timeout or 120.0

        self._anthropic: Optional[AnthropicClient] = None
        self._openai_compat: Optional[OpenRouterClient] = None
        try:
            if self._use_coding_plan:
                self._anthropic = AnthropicClient(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                    enable_prompt_caching=False,
                )
            else:
                self._openai_compat = OpenRouterClient(
                    api_key=self._api_key,
                    base_url=self._base_url,
                    timeout=self._timeout,
                )
        except (AnthropicError, OpenRouterError) as exc:
            raise KimiError(str(exc)) from exc

    async def chat_completion(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        **_: Any,
    ) -> Dict[str, Any]:
        actual_model = get_kimi_model_name(model)
        max_tokens = max_output_tokens or 4096
        try:
            if self._anthropic is not None:
                return await self._anthropic.chat_completion(
                    messages=messages,
                    model=actual_model,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_budget=thinking_budget,
                )
            if self._openai_compat is None:
                raise KimiError("Kimi client is not initialized")
            return await self._openai_compat.chat_completion(
                messages=messages,
                model=actual_model,
                tools=tools,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        except (AnthropicError, OpenRouterError) as exc:
            raise KimiError(str(exc)) from exc

    async def chat_completion_stream(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        thinking_budget: Optional[int] = None,
        **_: Any,
    ) -> AsyncIterator[str]:
        actual_model = get_kimi_model_name(model)
        max_tokens = max_output_tokens or 4096
        try:
            if self._anthropic is not None:
                return await self._anthropic.chat_completion_stream(
                    messages=messages,
                    model=actual_model,
                    tools=tools,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    thinking_budget=thinking_budget,
                )
            if self._openai_compat is None:
                raise KimiError("Kimi client is not initialized")
            return await self._openai_compat.chat_completion_stream(
                messages=messages,
                model=actual_model,
                tools=tools,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        except (AnthropicError, OpenRouterError) as exc:
            raise KimiError(str(exc)) from exc
