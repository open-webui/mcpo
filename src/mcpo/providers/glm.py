"""
GLM (Zhipu) provider supporting two endpoint modes:

1) Coding plan endpoint (Anthropic-compatible)
   https://open.bigmodel.cn/api/anthropic
2) Regular API endpoint (OpenAI-compatible)
   https://open.bigmodel.cn/api/paas/v4

Mode selection:
- GLM_USE_CODING_PLAN_ENDPOINT=true  -> Anthropic-compatible
- GLM_USE_CODING_PLAN_ENDPOINT=false -> OpenAI-compatible
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

from mcpo.providers.anthropic import AnthropicClient, AnthropicError
from mcpo.providers.openrouter import OpenRouterClient, OpenRouterError

logger = logging.getLogger(__name__)

GLM_ANTHROPIC_BASE_URL = "https://open.bigmodel.cn/api/anthropic"
GLM_OPENAI_BASE_URL = "https://open.bigmodel.cn/api/paas/v4"


class GLMError(RuntimeError):
    """Raised when GLM provider configuration or request handling fails."""


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _glm_api_key_from_env() -> Optional[str]:
    return os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY")


def get_glm_models() -> List[Dict[str, str]]:
    """
    Return GLM models from env only.
    Env:
    - GLM_MODELS=glm/model-a,glm/model-b
    - GLM_MODEL=glm/model-a
    """
    configured: List[Dict[str, str]] = []
    raw_list = os.getenv("GLM_MODELS")
    if raw_list:
        for item in raw_list.split(","):
            model_id = item.strip()
            if not model_id:
                continue
            if not (model_id.startswith("glm/") or model_id.startswith("zhipu/")):
                model_id = f"glm/{model_id}"
            configured.append({"id": model_id, "label": f"GLM: {model_id}"})

    preferred = os.getenv("GLM_MODEL")
    if preferred and not (preferred.startswith("glm/") or preferred.startswith("zhipu/")):
        preferred = f"glm/{preferred}"
    if preferred and not any(m["id"] == preferred for m in configured):
        configured.insert(0, {"id": preferred, "label": f"GLM: {preferred}"})

    return configured


def is_glm_model(model_id: str) -> bool:
    if not model_id:
        return False
    m = model_id.lower()
    return m.startswith("glm/") or m.startswith("glm-") or m.startswith("zhipu/")


def get_glm_model_name(model_id: str) -> str:
    if model_id.startswith("glm/"):
        return model_id.split("/", 1)[1]
    if model_id.startswith("zhipu/"):
        return model_id.split("/", 1)[1]
    return model_id


class GLMClient:
    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._api_key = api_key or _glm_api_key_from_env()
        if not self._api_key:
            raise GLMError("GLM_API_KEY or ZHIPU_API_KEY environment variable is required")

        self._use_coding_plan = _env_flag("GLM_USE_CODING_PLAN_ENDPOINT", True)
        regular_base = os.getenv("GLM_REGULAR_BASE_URL", GLM_OPENAI_BASE_URL)
        coding_base = os.getenv("GLM_CODING_PLAN_BASE_URL", GLM_ANTHROPIC_BASE_URL)
        mode_default = coding_base if self._use_coding_plan else regular_base
        self._base_url = (base_url or os.getenv("GLM_BASE_URL") or mode_default).rstrip("/")
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
            raise GLMError(str(exc)) from exc

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
        actual_model = get_glm_model_name(model)
        max_tokens = max_output_tokens or 4096
        mode = "coding-plan/anthropic" if self._use_coding_plan else "regular/openai-compatible"
        logger.info("[GLM] Non-stream via %s: model=%s, base_url=%s", mode, actual_model, self._base_url)
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
                raise GLMError("GLM client is not initialized")
            return await self._openai_compat.chat_completion(
                messages=messages,
                model=actual_model,
                tools=tools,
                temperature=temperature,
                max_output_tokens=max_tokens,
                include_reasoning=True,
            )
        except (AnthropicError, OpenRouterError) as exc:
            raise GLMError(str(exc)) from exc

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
        actual_model = get_glm_model_name(model)
        max_tokens = max_output_tokens or 4096
        mode = "coding-plan/anthropic" if self._use_coding_plan else "regular/openai-compatible"
        logger.info("[GLM] Stream via %s: model=%s, base_url=%s", mode, actual_model, self._base_url)
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
                raise GLMError("GLM client is not initialized")
            return await self._openai_compat.chat_completion_stream(
                messages=messages,
                model=actual_model,
                tools=tools,
                temperature=temperature,
                max_output_tokens=max_tokens,
                include_reasoning=True,
            )
        except (AnthropicError, OpenRouterError) as exc:
            raise GLMError(str(exc)) from exc
