from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional

import httpx

logger = logging.getLogger(__name__)


class OpenRouterError(RuntimeError):
    """Raised when the OpenRouter API returns an error payload."""


class OpenRouterClient:
    """
    Thin async client for the OpenRouter chat completions API.
    
    Supports interleaved thinking/reasoning for models like:
    - Claude 3.5+ (reasoning in content)
    - MiniMax M2 (reasoning_details field)
    - OpenAI o1/o3 (reasoning_content)
    - DeepSeek R1 (thinking in content)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        site_url: Optional[str] = None,
        app_name: Optional[str] = None,
        timeout: Optional[float] = None,
    ) -> None:
        self._api_key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not self._api_key:
            raise OpenRouterError("OPENROUTER_API_KEY environment variable is required")

        self._base_url = base_url or os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        self._site_url = site_url or os.getenv("OPENROUTER_SITE_URL")
        self._app_name = app_name or os.getenv("OPENROUTER_APP_NAME")
        self._timeout = timeout or float(os.getenv("OPENROUTER_TIMEOUT_SECONDS", "60"))
        self._stream_timeout = float(os.getenv("OPENROUTER_STREAM_TIMEOUT_SECONDS", "300"))
        self._max_retries = int(os.getenv("OPENROUTER_MAX_RETRIES", "2"))

    def _default_headers(self) -> Dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self._site_url:
            headers["HTTP-Referer"] = self._site_url
        if self._app_name:
            headers["X-Title"] = self._app_name
        return headers

    async def chat_completion(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        response_format: Optional[Dict[str, Any]] = None,
        include_reasoning: bool = True,
        reasoning_effort: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            # Request reasoning tokens for thinking models (DeepSeek-R1, Gemini Thinking, etc.)
            "include_reasoning": include_reasoning,
        }
        if tools:
            payload["tools"] = list(tools)
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if response_format is not None:
            payload["response_format"] = response_format
        # OpenRouter reasoning effort control (high/medium/low) for models like GPT-4
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        async with httpx.AsyncClient(timeout=self._timeout) as client:
            url = f"{self._base_url}/chat/completions"
            logger.info(f"[OPENROUTER] Non-stream request: model={model}, url={url}")
            logger.info(f"[OPENROUTER] Payload messages: {len(payload.get('messages', []))}, tools: {len(payload.get('tools', []) or [])}")
            logger.debug(f"[OPENROUTER] Full payload: {json.dumps(payload, default=str)[:3000]}")
            
            last_exc: Optional[Exception] = None
            for attempt in range(self._max_retries + 1):
                try:
                    response = await client.post(
                        url,
                        headers=self._default_headers(),
                        content=json.dumps(payload),
                    )
                    logger.info(f"[OPENROUTER] Response status: {response.status_code}")
                    if response.status_code >= 500 and attempt < self._max_retries:
                        logger.warning(f"[OPENROUTER] Server error {response.status_code}, retry {attempt + 1}/{self._max_retries}")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    if response.status_code >= 400:
                        logger.error(f"[OPENROUTER] Error response: {response.text[:2000]}")
                    await _raise_for_response(response)
                    data = response.json()
                    logger.info(f"[OPENROUTER] Success: id={data.get('id')}, model={data.get('model')}")
                    return data
                except (httpx.ConnectError, httpx.ReadTimeout) as exc:
                    last_exc = exc
                    if attempt < self._max_retries:
                        logger.warning(f"[OPENROUTER] {type(exc).__name__}, retry {attempt + 1}/{self._max_retries}")
                        await asyncio.sleep(2 ** attempt)
                    else:
                        raise
            # Should not reach here, but if it does:
            raise last_exc  # type: ignore[misc]

    async def chat_completion_stream(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_output_tokens: Optional[int] = None,
        include_reasoning: bool = True,
        reasoning_effort: Optional[str] = None,
    ) -> AsyncIterator[str]:
        payload: Dict[str, Any] = {
            "model": model,
            "messages": list(messages),
            "stream": True,
            "include_reasoning": include_reasoning,
        }
        if tools:
            payload["tools"] = list(tools)
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if reasoning_effort is not None:
            payload["reasoning_effort"] = reasoning_effort

        async def _iter() -> AsyncIterator[str]:
            async with httpx.AsyncClient(timeout=self._stream_timeout) as client:
                url = f"{self._base_url}/chat/completions"
                logger.info(f"[OPENROUTER] Stream request: model={model}, url={url}")
                logger.info(f"[OPENROUTER] Stream payload messages: {len(payload.get('messages', []))}, tools: {len(payload.get('tools', []) or [])}")
                logger.debug(f"[OPENROUTER] Stream payload: {json.dumps(payload, default=str)[:3000]}")
                
                async with client.stream(
                    "POST",
                    url,
                    headers=self._default_headers(),
                    content=json.dumps(payload),
                ) as response:
                    logger.info(f"[OPENROUTER] Stream response status: {response.status_code}")
                    if response.status_code >= 400:
                        error_body = await response.aread()
                        logger.error(f"[OPENROUTER] Stream error: {error_body.decode()[:2000]}")
                    await _raise_for_stream_response(response)
                    chunk_count = 0
                    async for line in response.aiter_lines():
                        if line:
                            chunk_count += 1
                            if chunk_count <= 5:
                                logger.debug(f"[OPENROUTER] Stream chunk {chunk_count}: {line[:300]}")
                            yield line
                    logger.info(f"[OPENROUTER] Stream complete: {chunk_count} chunks")

        return _iter()


# =============================================================================
# REASONING EXTRACTION UTILITIES
# =============================================================================

# Pattern for <think>...</think> blocks (DeepSeek, some Claude models)
THINK_TAG_PATTERN = re.compile(r"<think>(.*?)</think>", re.DOTALL)


def extract_reasoning_from_content(content: str) -> tuple[str, str]:
    """
    Extract reasoning/thinking from content that uses <think> tags.
    
    Returns:
        (clean_content, reasoning) - content with tags removed, extracted reasoning
    """
    if not content:
        return "", ""
    
    reasoning_parts: List[str] = []
    
    def capture_thinking(match: re.Match) -> str:
        reasoning_parts.append(match.group(1).strip())
        return ""
    
    clean_content = THINK_TAG_PATTERN.sub(capture_thinking, content).strip()
    reasoning = "\n\n".join(reasoning_parts) if reasoning_parts else ""
    
    return clean_content, reasoning


def merge_reasoning_details(
    existing: Optional[List[Dict[str, Any]]],
    new_details: Optional[List[Dict[str, Any]]]
) -> List[Dict[str, Any]]:
    """
    Merge reasoning_details arrays for interleaved thinking continuity.
    
    Per MiniMax spec: the entire response including reasoning_details must be
    preserved and passed back to maintain the reasoning chain.
    """
    result: List[Dict[str, Any]] = []
    
    if existing:
        result.extend(existing)
    
    if new_details:
        for detail in new_details:
            # Find existing entry with same id/index
            existing_idx = next(
                (i for i, d in enumerate(result) 
                 if d.get("id") == detail.get("id") or d.get("index") == detail.get("index")),
                None
            )
            if existing_idx is not None:
                # Append text to existing entry
                result[existing_idx]["text"] = (
                    result[existing_idx].get("text", "") + detail.get("text", "")
                )
            else:
                result.append(detail)
    
    return result


async def _raise_for_response(response: httpx.Response) -> None:
    try:
        response.raise_for_status()
    except httpx.HTTPStatusError as exc:
        detail: str
        try:
            payload = response.json()
            if isinstance(payload, dict) and payload.get("error"):
                detail = json.dumps(payload["error"])
            else:
                detail = response.text
        except json.JSONDecodeError:
            detail = response.text
        raise OpenRouterError(detail) from exc


async def _raise_for_stream_response(response: httpx.Response) -> None:
    if response.status_code < 400:
        return
    detail = f"OpenRouter streaming request failed with status {response.status_code}"
    content_bytes: bytes = b""
    try:
        content_bytes = await response.aread()
    except RuntimeError:
        raise OpenRouterError(detail)
    except Exception as exc:  # pragma: no cover - defensive fallback
        raise OpenRouterError(f"{detail}: {exc}") from exc

    if content_bytes:
        try:
            payload = json.loads(content_bytes.decode() or "{}")
            if isinstance(payload, dict) and payload.get("error"):
                detail = json.dumps(payload["error"])
            elif isinstance(payload, str):
                detail = payload
            else:
                detail = content_bytes.decode()
        except (json.JSONDecodeError, UnicodeDecodeError):
            decoded = content_bytes.decode(errors="replace").strip()
            if decoded:
                detail = decoded

    raise OpenRouterError(detail)
