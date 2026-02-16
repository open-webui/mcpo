from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, TypedDict

import httpx

# --- Configuration & Constants ---
logger = logging.getLogger("gemini_client")

GEMINI_DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com"
GEMINI_DEFAULT_API_VERSION = "v1beta"

# Environment-configurable defaults
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("GEMINI_TIMEOUT_SECONDS", "120"))
DEFAULT_STREAM_TIMEOUT_SECONDS = float(os.getenv("GEMINI_STREAM_TIMEOUT_SECONDS", "300"))
DEFAULT_MAX_RETRIES = int(os.getenv("GEMINI_MAX_RETRIES", "2"))
DEFAULT_ENABLE_CONTEXT_CACHING = os.getenv("GEMINI_ENABLE_CONTEXT_CACHING", "true").lower() in ("true", "1", "yes")

class GeminiError(RuntimeError):
    """Base exception for Gemini API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

# Backwards-compatible aliases for imports expecting "Google" naming
GoogleError = GeminiError

# --- Type Definitions ---
class ProviderSpecific(TypedDict, total=False):
    thought_signature: Optional[str]
    is_redacted: bool
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]

# --- Helper Functions ---

def _as_json_str(content: Any) -> str:
    """Safely serialize content to JSON string."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)

def _json(obj: Any) -> str:
    """Compact JSON serialization."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

def _is_flash_model(model: str) -> bool:
    """Check if model is a Flash variant."""
    return "flash" in model.lower()

def _is_pro_model(model: str) -> bool:
    """Check if model is a Pro variant."""
    return "pro" in model.lower()

def _get_thinking_defaults(model: str) -> Dict[str, Any]:
    """
    Get appropriate thinking defaults based on model type.
    
    Flash models:
    - Can turn thinking ON/OFF completely
    - Budget range: 0 to 24576 tokens
    - 0 = thinking OFF (fastest, cheapest)
    - -1 = auto (model decides)
    - 1-24576 = explicit token budget
    
    Pro models:
    - Always thinking ON (cannot disable)
    - Budget range: -1 (auto) or explicit token count
    - -1 = auto (recommended, model decides)
    """
    if _is_flash_model(model):
        return {
            "can_disable": True,
            "min_budget": 0,
            "max_budget": 24576,
            "default_budget": -1,  # Auto
            "off_value": 0
        }
    elif _is_pro_model(model):
        return {
            "can_disable": False,
            "min_budget": -1,  # Auto only, or explicit positive value
            "max_budget": None,  # No documented limit
            "default_budget": -1,  # Auto (always on)
            "off_value": None  # Cannot turn off
        }
    else:
        # Unknown model - assume Flash-like behavior
        return {
            "can_disable": True,
            "min_budget": 0,
            "max_budget": 24576,
            "default_budget": -1
        }

class GeminiClient:
    """
    Advanced Async Client for Google Gemini 3 & 2.5.
    
    Key Features:
    1. Model-Aware Thinking: Automatically handles Flash vs Pro thinking differences
    2. Thought Signatures: Preserves reasoning context across multi-turn conversations
    3. Thinking Budget Control: Fine-grained control (Flash: 0-24576, Pro: auto/-1)
    4. Context Caching: Automatic caching for large contexts (2048+ tokens)
    5. Tool Calling: Full function calling with signature handling
    6. Streaming: Real-time response streaming with thinking blocks
    7. State Persistence: Provider-specific metadata for agent resumption
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        enable_context_caching: Optional[bool] = None,
        stream_timeout: Optional[float] = None
    ) -> None:
        self._api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self._api_key:
            raise GeminiError("GEMINI_API_KEY or GOOGLE_API_KEY environment variable is required")
            
        self._base_url = (base_url or os.getenv("GEMINI_BASE_URL") or GEMINI_DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout if timeout is not None else float(os.getenv("GEMINI_TIMEOUT_SECONDS", "120"))
        self._max_retries = max_retries if max_retries is not None else int(os.getenv("GEMINI_MAX_RETRIES", "2"))
        self._enable_caching = enable_context_caching if enable_context_caching is not None else os.getenv("GEMINI_ENABLE_CONTEXT_CACHING", "true").lower() in ("true", "1", "yes")
        self._stream_timeout = stream_timeout if stream_timeout is not None else float(os.getenv("GEMINI_STREAM_TIMEOUT_SECONDS", "300"))
        self._api_version = os.getenv("GEMINI_API_VERSION", GEMINI_DEFAULT_API_VERSION)

    def _get_headers(self) -> Dict[str, str]:
        return {
            "x-goog-api-key": self._api_key,
            "content-type": "application/json",
        }

    def _reconstruct_model_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critical for agent state resumption.
        Reconstructs Gemini's content parts including thought signatures.
        """
        parts = []
        
        # 1. Re-inject Thought Signature
        if "provider_specific" in msg:
            ps: ProviderSpecific = msg["provider_specific"]
            sig = ps.get("thought_signature")
            
            if sig:
                is_redacted = ps.get("is_redacted", False)
                
                if is_redacted:
                    parts.append({
                        "thoughtSignature": sig
                    })
                else:
                    parts.append({
                        "thought": msg.get("reasoning_content", ""),
                        "thoughtSignature": sig
                    })

        # 2. Re-inject Text
        if msg.get("content"):
            parts.append({
                "text": _as_json_str(msg["content"])
            })

        # 3. Re-inject Function Calls
        if "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                parts.append({
                    "functionCall": {
                        "name": fn.get("name"),
                        "args": json.loads(fn.get("arguments", "{}"))
                    }
                })
        
        return {"role": "model", "parts": parts}

    def _map_messages(self, messages: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transforms generic chat history into Gemini's strict format.
        Handles system instructions, thought signatures, and tool results.
        """
        formatted_contents = []
        system_instruction = None
        cached_content = None

        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content")

            if role == "system":
                system_instruction = _as_json_str(content)
                
                if self._enable_caching and len(str(content)) > 2048:
                    cached_content = {
                        "contents": [],
                        "systemInstruction": {"parts": [{"text": system_instruction}]},
                        "ttl": "3600s"
                    }
                continue

            if role == "assistant":
                formatted_contents.append(self._reconstruct_model_message(msg))
                continue

            if role == "tool":
                function_response = {
                    "functionResponse": {
                        "name": msg.get("name") or msg.get("tool_name") or "function",
                        "response": {
                            "result": _as_json_str(content)
                        }
                    }
                }
                
                if formatted_contents and formatted_contents[-1]["role"] == "user":
                    formatted_contents[-1]["parts"].append(function_response)
                else:
                    formatted_contents.append({
                        "role": "user",
                        "parts": [function_response]
                    })
                continue

            if role == "user":
                parts = []
                
                if isinstance(content, str):
                    parts.append({"text": content})
                elif isinstance(content, list):
                    for item in content:
                        if isinstance(item, dict):
                            if item.get("type") == "text":
                                parts.append({"text": item.get("text", "")})
                            elif item.get("type") == "image_url":
                                image_url = item.get("image_url", {})
                                url = image_url.get("url", "")
                                if url.startswith("data:"):
                                    parts.append({
                                        "inlineData": {
                                            "mimeType": "image/jpeg",
                                            "data": url.split(",")[1] if "," in url else url
                                        }
                                    })
                
                if self._enable_caching and len(formatted_contents) > 10:
                     if parts:
                         parts[-1]["cacheControl"] = {"type": "ephemeral"}

                formatted_contents.append({"role": "user", "parts": parts})

        return {
            "system_instruction": system_instruction,
            "contents": formatted_contents,
            "cached_content": cached_content
        }

    def _extract_response(self, data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """Parses Gemini response, extracting text, reasoning, and signatures."""
        candidates = data.get("candidates", [])
        if not candidates:
            raise GeminiError("No candidates in response")
        
        candidate = candidates[0]
        content = candidate.get("content", {})
        parts = content.get("parts", [])
        
        final_text = ""
        reasoning_text = ""
        tool_calls = []
        
        signature = None
        is_redacted = False

        for part in parts:
            if "text" in part:
                final_text += part["text"]
            elif "thought" in part:
                reasoning_text += part.get("thought", "")
                signature = part.get("thoughtSignature")
            elif "thoughtSignature" in part and "thought" not in part:
                signature = part["thoughtSignature"]
                reasoning_text = "[Redacted Thinking]"
                is_redacted = True
            elif "functionCall" in part:
                fc = part["functionCall"]
                tool_calls.append({
                    "id": f"call_{uuid.uuid4().hex[:24]}",
                    "type": "function",
                    "function": {
                        "name": fc.get("name", ""),
                        "arguments": json.dumps(fc.get("args", {}))
                    }
                })

        message = {
            "role": "assistant",
            "content": final_text
        }
        
        if tool_calls:
            message["tool_calls"] = tool_calls
        if reasoning_text:
            message["reasoning_content"] = reasoning_text

        usage_metadata = data.get("usageMetadata", {})
        usage = {
            "prompt_tokens": usage_metadata.get("promptTokenCount", 0),
            "completion_tokens": usage_metadata.get("candidatesTokenCount", 0),
            "total_tokens": usage_metadata.get("totalTokenCount", 0)
        }

        # Gemini returns cachedContentTokenCount for reads; creation is not surfaced separately
        cached_tokens = usage_metadata.get("cachedContentTokenCount")
        provider_specific: ProviderSpecific = {
            "thought_signature": signature,
            "is_redacted": is_redacted,
            "cache_creation_input_tokens": None,  # Gemini does not expose creation vs read split
            "cache_read_input_tokens": cached_tokens if cached_tokens else None
        }
        
        if signature or provider_specific.get("cache_read_input_tokens"):
            message["provider_specific"] = provider_specific

        return {
            "id": f"gemini-{uuid.uuid4().hex}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "usage": usage,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": candidate.get("finishReason", "STOP").lower()
            }]
        }

    def _prepare_thinking_config(
        self,
        model: str,
        thinking_budget: Optional[int] = None,
        include_thoughts: bool = True
    ) -> Optional[Dict[str, Any]]:
        """
        Prepare thinking configuration based on model type.
        
        Args:
            model: Model name (flash/pro)
            thinking_budget: Token budget for thinking
                - Flash: 0 (off), -1 (auto), 1-24576 (explicit)
                - Pro: -1 (auto, recommended), or explicit positive value
            include_thoughts: Whether to include thought summaries in response
        """
        defaults = _get_thinking_defaults(model)
        
        # If no budget specified, use model defaults
        if thinking_budget is None:
            thinking_budget = defaults["default_budget"]
        
        # Validate based on model type
        if _is_flash_model(model):
            # Flash: 0-24576 or -1 (auto)
            if thinking_budget == 0:
                logger.info(f"[{model}] Thinking DISABLED (budget=0)")
            elif thinking_budget == -1:
                logger.info(f"[{model}] Thinking AUTO (model decides)")
            elif thinking_budget < 0 or thinking_budget > 24576:
                logger.warning(
                    f"[{model}] Invalid budget {thinking_budget}. "
                    f"Flash supports 0 (off), -1 (auto), or 1-24576. Using auto."
                )
                thinking_budget = -1
        
        elif _is_pro_model(model):
            # Pro: Cannot disable, always thinking
            if thinking_budget == 0:
                logger.warning(
                    f"[{model}] Pro models CANNOT disable thinking. "
                    f"Ignoring budget=0 and using auto (-1)."
                )
                thinking_budget = -1
            elif thinking_budget is not None and thinking_budget < -1:
                logger.warning(
                    f"[{model}] Invalid budget {thinking_budget}. Using auto (-1)."
                )
                thinking_budget = -1
        
        # Build config
        config = {
            "includeThoughts": include_thoughts
        }
        
        # Only set thinkingBudget if not using default auto
        if thinking_budget != -1:
            config["thinkingBudget"] = thinking_budget
        
        return config

    def _prepare_request_body(
        self,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]],
        temperature: Optional[float],
        max_tokens: int,
        thinking_budget: Optional[int],
        include_thoughts: bool,
        top_p: Optional[float],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepare request body for Gemini API."""
        mapped_payload = self._map_messages(messages)
        
        generation_config: Dict[str, Any] = {
            "maxOutputTokens": max_tokens,
        }
        
        # Gemini 2.5+ recommendation: keep temperature at 1.0 for best reasoning
        if temperature is not None:
            generation_config["temperature"] = temperature
        else:
            generation_config["temperature"] = 1.0
            
        if top_p is not None:
            generation_config["topP"] = top_p

        # Thinking configuration (model-aware)
        thinking_config = self._prepare_thinking_config(
            model, thinking_budget, include_thoughts
        )
        if thinking_config:
            generation_config["thinkingConfig"] = thinking_config

        body: Dict[str, Any] = {
            "contents": mapped_payload["contents"],
            "generationConfig": generation_config
        }
        
        # Wire cached content if present (large system prompts)
        if mapped_payload.get("cached_content"):
            body["cachedContent"] = mapped_payload["cached_content"]
        
        if mapped_payload["system_instruction"]:
            body["systemInstruction"] = {
                "parts": [{"text": mapped_payload["system_instruction"]}]
            }

        if tools:
            tool_declarations = []
            for tool in tools:
                fn = tool.get("function") or tool
                tool_declarations.append({
                    "functionDeclarations": [{
                        "name": fn.get("name"),
                        "description": fn.get("description", ""),
                        "parameters": fn.get("parameters", {})
                    }]
                })
            body["tools"] = tool_declarations

        return body

    async def chat_completion(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
        include_thoughts: bool = True,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a non-streaming chat completion.
        
        Args:
            thinking_budget (int): 
                - Flash: 0 (disable), -1 (auto), 1-24576 (explicit)
                - Pro: -1 (auto, always on), or explicit positive value
            include_thoughts (bool): Whether to include thought summaries
        """
        body = self._prepare_request_body(
            messages, model, tools, temperature, max_tokens,
            thinking_budget, include_thoughts, top_p, **kwargs
        )
        
        url = f"{self._base_url}/{self._api_version}/models/{model}:generateContent"
        headers = self._get_headers()
        
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, headers=headers, json=body)
                    
                    if resp.status_code == 429 or resp.status_code >= 500:
                        if attempt < self._max_retries:
                            wait_time = 2 ** attempt
                            logger.warning(f"Gemini API {resp.status_code}. Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    if resp.status_code >= 400:
                        error_data = resp.json() if resp.text else {}
                        error_msg = error_data.get("error", {}).get("message", resp.text)
                        raise GeminiError(
                            f"Gemini API Error {resp.status_code}: {error_msg}",
                            status_code=resp.status_code,
                            body=resp.text
                        )
                        
                    return self._extract_response(resp.json(), model)

            except httpx.RequestError as e:
                if attempt < self._max_retries:
                    logger.warning(f"Network error: {e}. Retrying...")
                    await asyncio.sleep(1)
                    continue
                raise GeminiError(f"Connection failed after {self._max_retries} retries: {str(e)}")

        raise GeminiError("Max retries exceeded")

    async def chat_completion_stream(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
        include_thoughts: bool = True,
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Execute a streaming chat completion with full support for thinking blocks.
        
        Yields:
            Server-Sent Event (SSE) formatted strings: "data: {json}\n\n"
        """
        body = self._prepare_request_body(
            messages, model, tools, temperature, max_tokens,
            thinking_budget, include_thoughts, top_p, **kwargs
        )
        
        url = f"{self._base_url}/{self._api_version}/models/{model}:streamGenerateContent?alt=sse"
        headers = self._get_headers()

        async def _stream_generator() -> AsyncIterator[str]:
            # Bounded timeout: connect quickly, allow long reads for SSE
            stream_timeout = httpx.Timeout(
                connect=30.0,
                read=self._stream_timeout,
                write=30.0,
                pool=30.0
            )
            for attempt in range(self._max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=stream_timeout) as client:
                        async with client.stream("POST", url, headers=headers, json=body) as response:
                            if response.status_code >= 400:
                                error_body = await response.aread()
                                raise GeminiError(
                                    f"Stream error {response.status_code}: {error_body.decode()}",
                                    status_code=response.status_code,
                                    body=error_body.decode()
                                )

                            message_id = f"gemini-{uuid.uuid4().hex}"
                            model_name = model
                            thinking_signature = None
                            finish_seen = False

                            initial = {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model_name,
                                "choices": [{
                                    "index": 0,
                                    "delta": {"role": "assistant"},
                                    "finish_reason": None
                                }]
                            }
                            yield f"data: {_json(initial)}\n\n"

                            async for line in response.aiter_lines():
                                if not line or not line.startswith("data:"):
                                    continue
                                
                                payload = line[5:].strip()
                                if not payload:
                                    continue

                                try:
                                    chunk_data = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue

                                candidates = chunk_data.get("candidates", [])
                                if not candidates:
                                    continue

                                candidate = candidates[0]
                                content = candidate.get("content", {})
                                parts = content.get("parts", [])

                                for part in parts:
                                    if "text" in part:
                                        text_delta = part["text"]
                                        
                                        chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": text_delta},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"
                                    
                                    elif "thought" in part:
                                        thinking_delta = part.get("thought", "")
                                        thinking_signature = part.get("thoughtSignature")
                                        
                                        chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"reasoning_content": thinking_delta},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"
                                    
                                    elif "thoughtSignature" in part and "thought" not in part:
                                        thinking_signature = part["thoughtSignature"]
                                        
                                        chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "reasoning_content": "[Redacted Thinking]",
                                                    "provider_specific": {
                                                        "thought_signature": thinking_signature,
                                                        "is_redacted": True
                                                    }
                                                },
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"
                                    
                                    elif "functionCall" in part:
                                        fc = part["functionCall"]
                                        tool_call = {
                                            "id": f"call_{uuid.uuid4().hex[:24]}",
                                            "type": "function",
                                            "function": {
                                                "name": fc.get("name", ""),
                                                "arguments": json.dumps(fc.get("args", {}))
                                            }
                                        }
                                        
                                        chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"tool_calls": [tool_call]},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"

                                finish_reason = candidate.get("finishReason")
                                if finish_reason:
                                    finish_seen = True
                                    
                                    if thinking_signature:
                                        sig_chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "provider_specific": {
                                                        "thought_signature": thinking_signature
                                                    }
                                                },
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(sig_chunk)}\n\n"
                                    
                                    finish_chunk = {
                                        "id": message_id,
                                        "object": "chat.completion.chunk",
                                        "created": int(time.time()),
                                        "model": model_name,
                                        "choices": [{
                                            "index": 0,
                                            "delta": {},
                                            "finish_reason": finish_reason.lower()
                                        }]
                                    }
                                    yield f"data: {_json(finish_chunk)}\n\n"

                            if not finish_seen:
                                finish_chunk = {
                                    "id": message_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model_name,
                                    "choices": [{
                                        "index": 0,
                                        "delta": {},
                                        "finish_reason": "stop"
                                    }]
                                }
                                yield f"data: {_json(finish_chunk)}\n\n"
                            
                            yield "data: [DONE]\n\n"
                            return

                except httpx.RequestError as e:
                    if attempt < self._max_retries:
                        logger.warning(f"Stream connection error: {e}. Retrying...")
                        await asyncio.sleep(2 ** attempt)
                        continue
                    raise GeminiError(f"Stream failed after {self._max_retries} retries: {str(e)}")

            raise GeminiError("Max retries exceeded for streaming")

        return _stream_generator()


# Backwards-compatible alias for imports expecting "Google" naming
GoogleClient = GeminiClient


def is_google_model(model: str) -> bool:
    """Check if a model string indicates a Google/Gemini model."""
    if not model:
        return False
    model_lower = model.lower()
    return model_lower.startswith("gemini") or model_lower.startswith("google/")


# --- Usage Examples ---
if __name__ == "__main__":
    async def example_flash_thinking_off():
        """Flash: Thinking OFF (fastest, cheapest)"""
        client = GeminiClient()
        
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "What's 2+2?"}
            ],
            model="gemini-2.5-flash",
            thinking_budget=0,  # OFF for Flash
            max_tokens=100
        )
        
        msg = response["choices"][0]["message"]
        print(f"💬 Response: {msg['content']}")
        print(f"💭 Thinking: {msg.get('reasoning_content', 'DISABLED')}")

    async def example_flash_thinking_auto():
        """Flash: Thinking AUTO (model decides)"""
        client = GeminiClient()
        
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "Solve x^2 + 4x + 4 = 0"}
            ],
            model="gemini-2.5-flash",
            thinking_budget=-1,  # AUTO (default)
            max_tokens=1024
        )
        
        msg = response["choices"][0]["message"]
        print(f"💭 Thinking: {msg.get('reasoning_content', 'N/A')}")
        print(f"💬 Response: {msg['content']}")

    async def example_flash_thinking_explicit():
        """Flash: Explicit budget (1-24576 tokens)"""
        client = GeminiClient()
        
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "Explain quantum entanglement"}
            ],
            model="gemini-2.5-flash",
            thinking_budget=2048,  # Explicit budget
            max_tokens=1024
        )
        
        msg = response["choices"][0]["message"]
        print(f"💭 Thinking: {msg.get('reasoning_content', 'N/A')[:200]}...")
        print(f"💬 Response: {msg['content'][:200]}...")

    async def example_pro_always_thinking():
        """Pro: Always thinking (cannot disable)"""
        client = GeminiClient()
        
        # Note: budget=0 will be ignored for Pro
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "Write a Python binary search"}
            ],
            model="gemini-2.5-pro",
            thinking_budget=-1,  # AUTO (recommended for Pro)
            max_tokens=2048
        )
        
        msg = response["choices"][0]["message"]
        print(f"💭 Thinking (Pro always on): {msg.get('reasoning_content', 'N/A')[:200]}...")
        print(f"💬 Code: {msg['content'][:400]}...")

    # Run examples
    # asyncio.run(example_flash_thinking_off())
    # asyncio.run(example_flash_thinking_auto())
    # asyncio.run(example_flash_thinking_explicit())
    # asyncio.run(example_pro_always_thinking())