from __future__ import annotations

import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, List, Optional, Union

import httpx
from fastapi import APIRouter, HTTPException, status
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator

from mcpo.services.skills import compile_skills_system_prompt

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["completions"])


class ChatMessage(BaseModel):
    role: str
    content: Any
    name: Optional[str] = None
    tool_call_id: Optional[str] = Field(None, alias="tool_call_id")


class ToolFunction(BaseModel):
    name: str
    description: Optional[str] = None
    parameters: Optional[Dict[str, Any]] = None


class ToolDefinition(BaseModel):
    type: str = "function"
    function: ToolFunction


class CompletionRequest(BaseModel):
    model: str = Field(..., description="Model identifier for the target provider")
    messages: List[ChatMessage]
    provider: Optional[str] = Field(
        None,
        description="Provider hint: openai|openrouter|anthropic|google/gemini. "
        "If omitted, the provider is inferred from the model name.",
    )
    base_url: Optional[str] = Field(None, description="Override provider base URL")
    api_key: Optional[str] = Field(None, description="Override provider API key")
    stream: bool = Field(
        False, description="When true, returns an event-stream compatible response"
    )
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    top_p: Optional[float] = Field(None, ge=0.0, le=1.0)
    max_tokens: Optional[int] = Field(None, gt=0)
    stop: Optional[Union[str, List[str]]] = None
    tools: Optional[List[ToolDefinition]] = None
    tool_choice: Optional[Union[str, Dict[str, Any]]] = Field(
        None, description="Tool choice hint (auto|none|function selection)"
    )
    metadata: Optional[Dict[str, Any]] = None
    response_format: Optional[Dict[str, Any]] = None
    # Extra body passthrough for provider-specific switches (e.g., reasoning_split)
    extra_body: Optional[Dict[str, Any]] = Field(
        None, description="Additional provider-specific payload fields"
    )
    skill_ids: Optional[List[str]] = Field(
        None, description="Skill IDs to inject as system prompt instructions"
    )

    @field_validator("stop")
    @classmethod
    def normalize_stop(cls, value: Any) -> Any:
        if value is None:
            return value
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [str(v) for v in value]
        raise TypeError("stop must be string or list of strings")


class CompletionChoice(BaseModel):
    index: int
    message: Dict[str, Any]
    finish_reason: Optional[str] = None


class CompletionResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[CompletionChoice]


# -----------------------------------------------------------------------------
# Model discovery helpers - fetch available models from each provider dynamically
# -----------------------------------------------------------------------------


async def _fetch_openrouter_models() -> List[Dict[str, str]]:
    """Fetch OpenRouter model catalog when OPENROUTER_API_KEY is present."""
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        return []
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or []
            models: List[Dict[str, str]] = []
            for entry in data:
                model_id = entry.get("id")
                if not model_id:
                    continue
                label = entry.get("name") or model_id
                models.append({"id": model_id, "label": label, "provider": "openrouter"})
            return models
    except Exception:
        return []


async def _fetch_openai_models() -> List[Dict[str, str]]:
    """Fetch OpenAI model catalog when OPEN_AI_API_KEY is present."""
    api_key = os.getenv("OPEN_AI_API_KEY")
    if not api_key:
        return []
    base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").rstrip("/")
    url = f"{base_url}/models"
    headers = {"Authorization": f"Bearer {api_key}"}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or []
            models: List[Dict[str, str]] = []
            for entry in data:
                model_id = entry.get("id")
                if not model_id:
                    continue
                # Filter to chat-capable models (gpt-*, o1-*, chatgpt-*, etc.)
                if any(model_id.startswith(p) for p in ("gpt-", "o1", "o3", "o4", "chatgpt-")):
                    models.append({"id": model_id, "label": f"OpenAI: {model_id}", "provider": "openai"})
            return models
    except Exception:
        return []


async def _fetch_google_models() -> List[Dict[str, str]]:
    """Fetch Google/Gemini model catalog when GOOGLE_API_KEY is present."""
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return []
    base_url = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base_url}/v1beta/models"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers={"x-goog-api-key": api_key})
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("models") or []
            models: List[Dict[str, str]] = []
            for entry in data:
                # Format: "models/gemini-2.0-flash" -> "gemini-2.0-flash"
                full_name = entry.get("name", "")
                model_id = full_name.replace("models/", "") if full_name.startswith("models/") else full_name
                if not model_id:
                    continue
                # Filter to generative models that support generateContent
                supported = entry.get("supportedGenerationMethods") or []
                if "generateContent" not in supported:
                    continue
                display = entry.get("displayName") or model_id
                models.append({"id": model_id, "label": f"Google: {display}", "provider": "gemini"})
            return models
    except Exception:
        return []


async def _fetch_anthropic_models() -> List[Dict[str, str]]:
    """Fetch Anthropic model catalog when ANTHROPIC_API_KEY is present."""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return []
    base_url = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com").rstrip("/")
    url = f"{base_url}/v1/models"
    version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")
    headers = {"x-api-key": api_key, "anthropic-version": version}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, headers=headers)
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("data") or []
            models: List[Dict[str, str]] = []
            for entry in data:
                model_id = entry.get("id")
                if not model_id:
                    continue
                display = entry.get("display_name") or model_id
                models.append({"id": model_id, "label": f"Anthropic: {display}", "provider": "anthropic"})
            return models
    except Exception:
        return []


async def _fetch_minimax_models() -> List[Dict[str, str]]:
    """Return MiniMax models from env config (no public listing API)."""
    api_key = os.getenv("MINIMAX_API_KEY")
    if not api_key:
        return []
    models: List[Dict[str, str]] = []
    raw_list = os.getenv("MINIMAX_MODELS")
    if raw_list:
        for item in raw_list.split(","):
            model_id = item.strip()
            if model_id:
                models.append({"id": model_id, "label": f"MiniMax: {model_id}", "provider": "minimax"})
    mm_default = os.getenv("MINIMAX_MODEL")
    if mm_default and not any(m["id"] == mm_default for m in models):
        models.insert(0, {"id": mm_default, "label": f"MiniMax: {mm_default}", "provider": "minimax"})
    return models


async def _list_models() -> List[Dict[str, str]]:
    """Aggregate models from all configured providers."""
    import asyncio
    
    # Fetch from all providers in parallel
    # Note: Perplexity excluded - their API key is for MCP server only, not completions
    results = await asyncio.gather(
        _fetch_openrouter_models(),
        _fetch_openai_models(),
        _fetch_google_models(),
        _fetch_anthropic_models(),
        _fetch_minimax_models(),
        return_exceptions=True,
    )
    
    all_models: List[Dict[str, str]] = []
    seen: set = set()
    
    for result in results:
        if isinstance(result, Exception):
            continue
        for model in result:
            model_id = model.get("id")
            if model_id and model_id not in seen:
                all_models.append(model)
                seen.add(model_id)
    
    return all_models


def _json(obj: Any) -> str:
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)


def _now_ts() -> int:
    return int(time.time())


def _as_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)


def _ensure_data_prefix(chunk: str) -> str:
    body = chunk if chunk.startswith("data:") else f"data: {chunk}"
    return body if body.endswith("\n\n") else body + "\n\n"


def _map_finish(reason: Optional[str]) -> Optional[str]:
    if not reason:
        return None
    mapping = {
        "end_turn": "stop",
        "stop_sequence": "stop",
        "stop": "stop",
        "max_tokens": "length",
        "length": "length",
        "tool_use": "tool_calls",
        "recitation": "content_filter",
    }
    return mapping.get(reason, reason)


class CompletionProviderError(RuntimeError):
    pass


class BaseCompletionProvider:
    name: str

    async def complete(self, payload: CompletionRequest) -> Dict[str, Any]:
        raise NotImplementedError

    async def stream(self, payload: CompletionRequest) -> AsyncIterator[str]:
        raise NotImplementedError


class OpenAICompatibleProvider(BaseCompletionProvider):
    def __init__(
        self,
        name: str,
        *,
        base_url: Optional[str],
        api_key: Optional[str],
        extra_headers: Optional[Dict[str, str]] = None,
    ) -> None:
        self.name = name
        default_base = "https://api.openai.com/v1"
        if name == "openrouter":
            default_base = "https://openrouter.ai/api/v1"
        self.base_url = (base_url or default_base).rstrip("/")
        self.api_key = api_key
        self.extra_headers = extra_headers or {}
        self.timeout = 60.0

    def _headers(self) -> Dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        headers.update(self.extra_headers)
        return headers

    def _build_payload(self, payload: CompletionRequest, *, stream: bool, extra_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {
            "model": payload.model,
            "messages": json.loads(payload.model_dump_json(include={"messages"})),
            "stream": stream,
        }
        if payload.temperature is not None:
            body["temperature"] = payload.temperature
        if payload.top_p is not None:
            body["top_p"] = payload.top_p
        if payload.max_tokens is not None:
            body["max_tokens"] = payload.max_tokens
        if payload.stop:
            body["stop"] = payload.stop
        if payload.tools:
            body["tools"] = json.loads(payload.model_dump_json(include={"tools"}))
        if payload.tool_choice is not None:
            body["tool_choice"] = payload.tool_choice
        if payload.metadata:
            body["metadata"] = payload.metadata
        if payload.response_format:
            body["response_format"] = payload.response_format
        if payload.extra_body:
            body.update(payload.extra_body)
        if extra_body:
            body.update(extra_body)
        return body

    async def _raise_for_response(self, response: httpx.Response) -> None:
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as exc:
            detail = response.text
            try:
                body = response.json()
                if isinstance(body, dict):
                    detail = _json(body)
            except Exception:
                pass
            raise CompletionProviderError(detail) from exc

    async def complete(self, payload: CompletionRequest, *, extra_body: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        body = self._build_payload(payload, stream=False, extra_body=extra_body)
        url = f"{self.base_url}/chat/completions"
        logger.info(f"[COMPLETIONS] {self.name} non-stream request: model={payload.model}, url={url}")
        logger.debug(f"[COMPLETIONS] Request body: {json.dumps(body, default=str)[:2000]}")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                headers=self._headers(),
                json=body,
            )
            logger.info(f"[COMPLETIONS] {self.name} response: status={response.status_code}")
            if response.status_code >= 400:
                logger.error(f"[COMPLETIONS] {self.name} error response: {response.text[:1000]}")
            await self._raise_for_response(response)
            data = response.json()
            logger.info(f"[COMPLETIONS] {self.name} success: id={data.get('id')}, model={data.get('model')}")
            return data

    async def stream(self, payload: CompletionRequest, *, extra_body: Optional[Dict[str, Any]] = None) -> AsyncIterator[str]:
        body = self._build_payload(payload, stream=True, extra_body=extra_body)
        url = f"{self.base_url}/chat/completions"
        logger.info(f"[COMPLETIONS] {self.name} stream request: model={payload.model}, url={url}")
        logger.debug(f"[COMPLETIONS] Stream request body: {json.dumps(body, default=str)[:2000]}")
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                url,
                headers=self._headers(),
                json=body,
            ) as response:
                logger.info(f"[COMPLETIONS] {self.name} stream response: status={response.status_code}")
                if response.status_code >= 400:
                    error_body = await response.aread()
                    logger.error(f"[COMPLETIONS] {self.name} stream error: {error_body.decode()[:1000]}")
                await self._raise_for_response(response)
                chunk_count = 0
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    chunk_count += 1
                    if chunk_count <= 3:
                        logger.debug(f"[COMPLETIONS] Stream chunk {chunk_count}: {line[:200]}")
                    yield _ensure_data_prefix(line)
                logger.info(f"[COMPLETIONS] {self.name} stream complete: {chunk_count} chunks")


class AnthropicProvider(BaseCompletionProvider):
    def __init__(self, *, base_url: Optional[str], api_key: Optional[str]) -> None:
        self.name = "anthropic"
        self.base_url = (base_url or "https://api.anthropic.com").rstrip("/")
        self.api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise CompletionProviderError("Anthropic API key is required")
        self.timeout = 60.0
        self.version = os.getenv("ANTHROPIC_VERSION", "2023-06-01")

    def _headers(self) -> Dict[str, str]:
        return {
            "x-api-key": self.api_key,
            "anthropic-version": self.version,
            "content-type": "application/json",
        }

    def _map_tools(self, tools: Optional[List[ToolDefinition]]) -> Optional[List[Dict[str, Any]]]:
        if not tools:
            return None
        mapped: List[Dict[str, Any]] = []
        for tool in tools:
            fn = tool.function
            mapped.append(
                {
                    "name": fn.name,
                    "description": fn.description or "",
                    "input_schema": fn.parameters or {"type": "object", "properties": {}},
                }
            )
        return mapped

    def _map_tool_choice(self, tool_choice: Any) -> Any:
        if tool_choice is None:
            return None
        if isinstance(tool_choice, str):
            if tool_choice in {"auto", "any"}:
                return tool_choice
            if tool_choice == "none":
                return {"type": "none"}
            return {"type": "tool", "name": tool_choice}
        if isinstance(tool_choice, dict):
            if tool_choice.get("type") == "function":
                name = (tool_choice.get("function") or {}).get("name")
                if name:
                    return {"type": "tool", "name": name}
            if tool_choice.get("type") == "tool" and tool_choice.get("name"):
                return {"type": "tool", "name": tool_choice["name"]}
        return tool_choice

    def _map_messages(self, payload: CompletionRequest) -> Dict[str, Any]:
        system_msgs: List[str] = []
        messages: List[Dict[str, Any]] = []
        for msg in payload.messages:
            role = msg.role.lower()
            if role == "system":
                system_msgs.append(_as_text(msg.content))
                continue
            mapped_role = "assistant" if role == "assistant" else "user"
            messages.append(
                {
                    "role": mapped_role,
                    "content": [{"type": "text", "text": _as_text(msg.content)}],
                }
            )
        return {"system": "\n\n".join(system_msgs) or None, "messages": messages}

    def _as_openai_response(self, payload: CompletionRequest, response: Dict[str, Any]) -> Dict[str, Any]:
        text_parts: List[str] = []
        tool_calls: List[Dict[str, Any]] = []
        for block in response.get("content", []):
            if block.get("type") == "text":
                text_parts.append(block.get("text", ""))
            elif block.get("type") == "tool_use":
                tool_calls.append(
                    {
                        "id": block.get("id") or uuid.uuid4().hex,
                        "type": "function",
                        "function": {
                            "name": block.get("name", ""),
                            "arguments": _json(block.get("input", {})),
                        },
                    }
                )
        message: Dict[str, Any] = {"role": "assistant", "content": "".join(text_parts)}
        if tool_calls:
            message["tool_calls"] = tool_calls
        finish_reason = _map_finish(response.get("stop_reason"))
        return {
            "id": response.get("id") or f"anthropic-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": _now_ts(),
            "model": response.get("model") or payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _request_body(self, payload: CompletionRequest, *, stream: bool) -> Dict[str, Any]:
        mapped = self._map_messages(payload)
        tools = self._map_tools(payload.tools)
        body: Dict[str, Any] = {
            "model": payload.model,
            "messages": mapped["messages"],
            "max_tokens": payload.max_tokens or 1024,
            "stream": stream,
        }
        if mapped["system"]:
            body["system"] = mapped["system"]
        if payload.temperature is not None:
            body["temperature"] = payload.temperature
        if payload.top_p is not None:
            body["top_p"] = payload.top_p
        if payload.stop:
            body["stop_sequences"] = payload.stop
        if tools:
            body["tools"] = tools
        tool_choice = self._map_tool_choice(payload.tool_choice)
        if tool_choice:
            body["tool_choice"] = tool_choice
        if payload.metadata:
            body["metadata"] = payload.metadata
        return body

    async def complete(self, payload: CompletionRequest) -> Dict[str, Any]:
        body = self._request_body(payload, stream=False)
        url = f"{self.base_url}/v1/messages"
        logger.info(f"[COMPLETIONS] Anthropic non-stream request: model={payload.model}, url={url}")
        logger.debug(f"[COMPLETIONS] Anthropic request body: {json.dumps(body, default=str)[:2000]}")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(
                url,
                headers=self._headers(),
                json=body,
            )
            logger.info(f"[COMPLETIONS] Anthropic response: status={response.status_code}")
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text
                logger.error(f"[COMPLETIONS] Anthropic error: {detail[:1000]}")
                try:
                    detail = _json(response.json())
                except Exception:
                    pass
                raise CompletionProviderError(detail) from exc
            data = response.json()
            logger.info(f"[COMPLETIONS] Anthropic success: id={data.get('id')}, model={data.get('model')}")
            return self._as_openai_response(payload, data)

    def _text_chunk(self, model: str, message_id: str, delta: str) -> Dict[str, Any]:
        return {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": _now_ts(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": delta},
                    "finish_reason": None,
                }
            ],
        }

    def _tool_chunk(self, model: str, message_id: str, tool_call: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": _now_ts(),
            "model": model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"tool_calls": [tool_call]},
                    "finish_reason": None,
                }
            ],
        }

    def _finish_chunk(self, model: str, message_id: str, finish_reason: Optional[str]) -> Dict[str, Any]:
        return {
            "id": message_id,
            "object": "chat.completion.chunk",
            "created": _now_ts(),
            "model": model,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }

    async def stream(self, payload: CompletionRequest) -> AsyncIterator[str]:
        body = self._request_body(payload, stream=True)
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/v1/messages",
                headers=self._headers(),
                json=body,
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = response.text
                    try:
                        detail = _json(response.json())
                    except Exception:
                        pass
                    raise CompletionProviderError(detail) from exc

                message_id = f"anthropic-{uuid.uuid4().hex}"
                model = payload.model
                tool_buffers: Dict[int, Dict[str, Any]] = {}
                finish_seen = False

                # Emit role upfront for better client compatibility
                initial_chunk = {
                    "id": message_id,
                    "object": "chat.completion.chunk",
                    "created": _now_ts(),
                    "model": model,
                    "choices": [
                        {"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}
                    ],
                }
                yield _ensure_data_prefix(_json(initial_chunk))

                async for raw_line in response.aiter_lines():
                    if not raw_line or raw_line.startswith("event:"):
                        continue
                    line = raw_line.strip()
                    if not line.startswith("data:"):
                        continue
                    data = line[len("data:") :].strip()
                    if not data or data == "[DONE]":
                        continue
                    try:
                        event = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    event_type = event.get("type") or event.get("event")
                    if event_type == "message_start":
                        msg = event.get("message") or {}
                        message_id = msg.get("id", message_id)
                        model = msg.get("model", model)
                    elif event_type == "content_block_start":
                        content_block = event.get("content_block") or {}
                        if content_block.get("type") == "tool_use":
                            idx = event.get("index", len(tool_buffers))
                            tool_buffers[idx] = {
                                "id": content_block.get("id") or uuid.uuid4().hex,
                                "type": "function",
                                "function": {
                                    "name": content_block.get("name", ""),
                                    "arguments": "",
                                },
                            }
                            yield _ensure_data_prefix(_json(self._tool_chunk(model, message_id, tool_buffers[idx])))
                    elif event_type == "content_block_delta":
                        delta = event.get("delta") or {}
                        idx = event.get("index")
                        if delta.get("type") == "text_delta":
                            text_delta = delta.get("text") or ""
                            if text_delta:
                                yield _ensure_data_prefix(_json(self._text_chunk(model, message_id, text_delta)))
                        elif delta.get("type") == "input_json_delta" and idx is not None:
                            buf = tool_buffers.get(idx)
                            if buf:
                                buf["function"]["arguments"] += delta.get("partial_json", "")
                                yield _ensure_data_prefix(_json(self._tool_chunk(model, message_id, buf)))
                    elif event_type == "message_delta":
                        delta = event.get("delta") or {}
                        finish_reason = _map_finish(delta.get("stop_reason"))
                        if finish_reason:
                            finish_seen = True
                            yield _ensure_data_prefix(_json(self._finish_chunk(model, message_id, finish_reason)))

                if not finish_seen:
                    yield _ensure_data_prefix(_json(self._finish_chunk(model, message_id, "stop")))
                yield _ensure_data_prefix("[DONE]")


class GeminiProvider(BaseCompletionProvider):
    def __init__(self, *, base_url: Optional[str], api_key: Optional[str]) -> None:
        self.name = "gemini"
        self.base_url = (base_url or "https://generativelanguage.googleapis.com").rstrip(
            "/"
        )
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise CompletionProviderError("Google API key is required")
        self.timeout = 60.0

    def _contents(self, payload: CompletionRequest) -> Dict[str, Any]:
        contents: List[Dict[str, Any]] = []
        system_parts: List[str] = []
        for msg in payload.messages:
            role = msg.role.lower()
            if role == "system":
                system_parts.append(_as_text(msg.content))
                continue
            parts = [{"text": _as_text(msg.content)}]
            contents.append(
                {
                    "role": "user" if role == "user" else "model",
                    "parts": parts,
                }
            )
        generation: Dict[str, Any] = {}
        if payload.temperature is not None:
            generation["temperature"] = payload.temperature
        if payload.top_p is not None:
            generation["topP"] = payload.top_p
        if payload.max_tokens is not None:
            generation["maxOutputTokens"] = payload.max_tokens
        if payload.stop:
            generation["stopSequences"] = payload.stop
        body: Dict[str, Any] = {"contents": contents}
        if generation:
            body["generationConfig"] = generation
        if system_parts:
            body["systemInstruction"] = {"parts": [{"text": "\n\n".join(system_parts)}]}
        return body

    def _extract_text(self, data: Dict[str, Any]) -> str:
        candidates = data.get("candidates") or []
        if not candidates:
            return ""
        parts = candidates[0].get("content", {}).get("parts", []) or []
        texts: List[str] = []
        for part in parts:
            if isinstance(part, dict) and part.get("text"):
                texts.append(part["text"])
        return "".join(texts)

    def _finish_reason(self, data: Dict[str, Any]) -> Optional[str]:
        candidates = data.get("candidates") or []
        if not candidates:
            return None
        reason = candidates[0].get("finishReason")
        if reason == "STOP":
            return "stop"
        if reason == "MAX_TOKENS":
            return "length"
        if reason == "SAFETY":
            return "content_filter"
        return reason.lower() if isinstance(reason, str) else None

    def _response(self, payload: CompletionRequest, data: Dict[str, Any]) -> Dict[str, Any]:
        content = self._extract_text(data)
        finish_reason = self._finish_reason(data)
        return {
            "id": f"gemini-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": _now_ts(),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": finish_reason or "stop",
                }
            ],
        }

    def _chunk(self, payload: CompletionRequest, text: str, finish: Optional[str]) -> Dict[str, Any]:
        return {
            "id": f"gemini-{uuid.uuid4().hex}",
            "object": "chat.completion.chunk",
            "created": _now_ts(),
            "model": payload.model,
            "choices": [
                {
                    "index": 0,
                    "delta": {"content": text},
                    "finish_reason": finish,
                }
            ],
        }

    async def complete(self, payload: CompletionRequest) -> Dict[str, Any]:
        body = self._contents(payload)
        url = f"{self.base_url}/v1beta/models/{payload.model}:generateContent"
        logger.info(f"[COMPLETIONS] Gemini non-stream request: model={payload.model}, url={url}")
        logger.debug(f"[COMPLETIONS] Gemini request body: {json.dumps(body, default=str)[:2000]}")
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.post(url, headers={"x-goog-api-key": self.api_key}, json=body)
            logger.info(f"[COMPLETIONS] Gemini response: status={response.status_code}")
            try:
                response.raise_for_status()
            except httpx.HTTPStatusError as exc:
                detail = response.text
                logger.error(f"[COMPLETIONS] Gemini error: {detail[:1000]}")
                try:
                    detail = _json(response.json())
                except Exception:
                    pass
                raise CompletionProviderError(detail) from exc
            data = response.json()
            logger.info(f"[COMPLETIONS] Gemini success")
            return self._response(payload, data)

    async def stream(self, payload: CompletionRequest) -> AsyncIterator[str]:
        body = self._contents(payload)
        url = f"{self.base_url}/v1beta/models/{payload.model}:streamGenerateContent"
        async with httpx.AsyncClient(timeout=None) as client:
            async with client.stream(
                "POST", url, headers={"x-goog-api-key": self.api_key}, json=body
            ) as response:
                try:
                    response.raise_for_status()
                except httpx.HTTPStatusError as exc:
                    detail = response.text
                    try:
                        detail = _json(response.json())
                    except Exception:
                        pass
                    raise CompletionProviderError(detail) from exc

                finish_emitted = False
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    text = self._extract_text(data)
                    finish = self._finish_reason(data)
                    if text:
                        yield _ensure_data_prefix(_json(self._chunk(payload, text, None)))
                    if finish and not finish_emitted:
                        finish_emitted = True
                        yield _ensure_data_prefix(_json(self._chunk(payload, "", finish)))
                if not finish_emitted:
                    yield _ensure_data_prefix(_json(self._chunk(payload, "", "stop")))
                yield _ensure_data_prefix("[DONE]")


class MiniMaxProvider(BaseCompletionProvider):
    """
    MiniMax provider via the Anthropic Coding Plan endpoint.

    Delegates to the inline AnthropicProvider with MiniMax's
    Anthropic-compatible base URL (https://api.minimax.io/anthropic).
    """

    def __init__(self, *, base_url: Optional[str], api_key: Optional[str]) -> None:
        self.name = "minimax"
        self.base_url = (
            base_url
            or os.getenv("MINIMAX_BASE_URL")
            or "https://api.minimax.io/anthropic"
        ).rstrip("/")
        self.api_key = api_key or os.getenv("MINIMAX_API_KEY")
        if not self.api_key:
            raise CompletionProviderError("MiniMax API key is required")

    async def complete(self, payload: CompletionRequest) -> Dict[str, Any]:
        delegate = AnthropicProvider(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        return await delegate.complete(payload)

    async def stream(self, payload: CompletionRequest) -> AsyncIterator[str]:
        delegate = AnthropicProvider(
            base_url=self.base_url,
            api_key=self.api_key,
        )
        return await delegate.stream(payload)


def _infer_provider(model: str) -> str:
    m = model.lower()
    if m.startswith("claude") or "anthropic" in m:
        return "anthropic"
    if m.startswith("gemini") or "google" in m:
        return "gemini"
    if m.startswith("openrouter") or m.startswith("router"):
        return "openrouter"
    if "minimax" in m or m.startswith("m2") or "m2" in m:
        return "minimax"
    # Many OpenRouter models are vendor-prefixed (e.g., moonshotai/kimi-k2:free).
    # If the model contains a slash (and optionally colon variants) and an OpenRouter key is set, treat it as OpenRouter.
    if ("/" in m or (":" in m and os.getenv("OPENROUTER_API_KEY"))) and os.getenv("OPENROUTER_API_KEY"):
        return "openrouter"
    return "openai"


def _resolve_provider(payload: CompletionRequest) -> BaseCompletionProvider:
    inferred = _infer_provider(payload.model)
    provider = (payload.provider or inferred).lower()
    logger.info(f"[COMPLETIONS] _resolve_provider: model={payload.model}, inferred={inferred}, final_provider={provider}")

    if provider in {"openai", "openrouter", "azure"}:
        api_key = payload.api_key or os.getenv("OPEN_AI_API_KEY")
        extra_headers: Dict[str, str] = {}
        if provider == "openrouter":
            api_key = payload.api_key or os.getenv("OPENROUTER_API_KEY")
            site = os.getenv("OPENROUTER_SITE_URL")
            app = os.getenv("OPENROUTER_APP_NAME")
            if site:
                extra_headers["HTTP-Referer"] = site
            if app:
                extra_headers["X-Title"] = app
        if not api_key and provider != "azure":
            logger.error(f"[COMPLETIONS] {provider} API key missing")
            raise CompletionProviderError(f"{provider} API key is required")
        base_url = payload.base_url or os.getenv("OPENAI_BASE_URL")
        logger.info(f"[COMPLETIONS] Creating OpenAICompatibleProvider: name={provider}, base_url={base_url}")
        return OpenAICompatibleProvider(
            provider,
            base_url=base_url,
            api_key=api_key,
            extra_headers=extra_headers,
        )

    if provider in {"anthropic", "claude"}:
        logger.info(f"[COMPLETIONS] Creating AnthropicProvider")
        return AnthropicProvider(base_url=payload.base_url, api_key=payload.api_key)

    if provider in {"google", "gemini"}:
        logger.info(f"[COMPLETIONS] Creating GeminiProvider")
        return GeminiProvider(base_url=payload.base_url, api_key=payload.api_key)

    if provider == "minimax":
        logger.info(f"[COMPLETIONS] Creating MiniMaxProvider")
        return MiniMaxProvider(
            base_url=payload.base_url,
            api_key=payload.api_key,
        )



async def _sse(stream: AsyncIterator[str]) -> AsyncIterator[str]:
    saw_done = False
    async for chunk in stream:
        if chunk.strip().endswith("[DONE]"):
            saw_done = True
        yield _ensure_data_prefix(chunk)
    if not saw_done:
        yield _ensure_data_prefix("[DONE]")


@router.post(
    "/chat/completions",
    response_model=CompletionResponse,
    responses={206: {"description": "Streaming response"}},
)
async def chat_completions(payload: CompletionRequest):
    logger.info(f"[COMPLETIONS] Incoming request: model={payload.model}, provider_hint={payload.provider}, stream={payload.stream}")
    logger.info(f"[COMPLETIONS] Messages count={len(payload.messages)}, tools={len(payload.tools) if payload.tools else 0}")
    if payload.messages:
        for i, msg in enumerate(payload.messages[-3:]):  # Log last 3 messages
            content_preview = str(msg.content)[:100] if msg.content else "(empty)"
            logger.debug(f"[COMPLETIONS] Message[-{len(payload.messages)-i}]: role={msg.role}, content={content_preview}")

    # Inject skills into the messages as a system prompt
    skills_prompt = compile_skills_system_prompt(
        scope="completions",
        model=payload.model,
        provider=payload.provider,
        requested_skill_ids=payload.skill_ids if payload.skill_ids else None,
    )
    if skills_prompt:
        # Prepend or merge into existing system message
        has_system = payload.messages and payload.messages[0].role == "system"
        if has_system:
            existing = payload.messages[0].content or ""
            payload.messages[0].content = f"{existing}\n\n{skills_prompt}" if existing else skills_prompt
        else:
            payload.messages.insert(0, ChatMessage(role="system", content=skills_prompt))
        logger.info(f"[COMPLETIONS] Injected skills system prompt ({len(skills_prompt)} chars)")

    try:
        provider = _resolve_provider(payload)
        logger.info(f"[COMPLETIONS] Resolved provider: {provider.name}")
    except CompletionProviderError as exc:
        logger.error(f"[COMPLETIONS] Provider resolution failed: {exc}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)
        ) from exc

    if payload.stream:
        logger.info(f"[COMPLETIONS] Starting stream response")
        stream = provider.stream(payload)
        return StreamingResponse(_sse(stream), media_type="text/event-stream")

    try:
        result = await provider.complete(payload)
        logger.info(f"[COMPLETIONS] Non-stream response complete")
        return result
    except CompletionProviderError as exc:
        logger.error(f"[COMPLETIONS] Provider error: {exc}")
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY, detail=str(exc)
        ) from exc


# Compatibility alias for simple completions payloads
@router.post(
    "/completions",
    response_model=CompletionResponse,
    responses={206: {"description": "Streaming response"}},
)
async def legacy_completions(payload: CompletionRequest):
    return await chat_completions(payload)


def _filter_by_favorites(models: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Filter models list to only include starred/favorite models.
    
    If no favorites are set, returns all models (no filtering).
    """
    from mcpo.services.state import get_state_manager
    state = get_state_manager()
    favorites = state.get_favorite_models()
    
    if not favorites:
        # No favorites set - return all models
        return models
    
    favorites_set = set(favorites)
    return [m for m in models if m["id"] in favorites_set]


@router.get("/chat/completions/models")
async def list_completion_models():
    """
    Return model catalog for completions endpoint.

    Compatible with OpenAI client expectations and OpenWebUI's external tool UI.
    Only returns starred/favorite models if any are set.
    """
    models = await _list_models()
    models = _filter_by_favorites(models)
    data = [{"id": m["id"], "object": "model", "label": m.get("label", m["id"])} for m in models]
    return {"object": "list", "data": data, "models": models}


@router.get("/completions/models")
async def legacy_list_completion_models():
    return await list_completion_models()


@router.get("/models")
async def list_models_root():
    """OpenAI-compatible models list at the root of /v1 for OpenWebUI/OpenAI clients.
    
    Only returns starred/favorite models if any are set.
    """
    models = await _list_models()
    models = _filter_by_favorites(models)
    data = [{"id": m["id"], "object": "model"} for m in models]
    return {"object": "list", "data": data, "models": models}
