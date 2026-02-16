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
logger = logging.getLogger("anthropic_client")

ANTHROPIC_DEFAULT_BASE_URL = "https://api.anthropic.com"
ANTHROPIC_API_VERSION = "2023-06-01"

# Environment-configurable defaults
DEFAULT_TIMEOUT_SECONDS = float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "120"))
DEFAULT_STREAM_TIMEOUT_SECONDS = float(os.getenv("ANTHROPIC_STREAM_TIMEOUT_SECONDS", "300"))
DEFAULT_MAX_RETRIES = int(os.getenv("ANTHROPIC_MAX_RETRIES", "2"))
DEFAULT_ENABLE_PROMPT_CACHING = os.getenv("ANTHROPIC_ENABLE_PROMPT_CACHING", "true").lower() in ("true", "1", "yes")

# Critical Beta Headers for Agentic Workflows (Dec 2025 Standards)
DEFAULT_BETA_HEADERS = [
    "prompt-caching-2024-07-31",
    "output-128k-2025-02-19" 
]

class AnthropicError(RuntimeError):
    """Base exception for Anthropic API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

# --- Type Definitions ---
class ProviderSpecific(TypedDict, total=False):
    thought_signature: Optional[str]
    is_redacted: bool
    cache_creation_input_tokens: Optional[int]
    cache_read_input_tokens: Optional[int]

# --- Helper Functions ---

def _as_json_str(content: Any) -> str:
    """Safely serialize tool inputs or object content to JSON string."""
    if isinstance(content, str):
        return content
    try:
        return json.dumps(content, ensure_ascii=False)
    except Exception:
        return str(content)

def _json(obj: Any) -> str:
    """Compact JSON serialization for streaming."""
    return json.dumps(obj, separators=(",", ":"), ensure_ascii=False)

class AnthropicClient:
    """
    Advanced Async Client for Anthropic Claude.
    
    Optimized for:
    1. Interleaved Thinking: Preserves 'thinking' blocks and opaque signatures.
    2. Long-Running Agents: 'provider_specific' field allows DB persistence/resumption.
    3. Tool Use: Handles 'tool_use' and 'tool_result' mapping strictly.
    4. Prompt Caching: Support for ephemeral cache headers.
    5. Streaming: Full Server-Sent Events (SSE) support with thinking blocks.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        stream_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        enable_prompt_caching: Optional[bool] = None
    ) -> None:
        self._api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        if not self._api_key:
            raise AnthropicError("ANTHROPIC_API_KEY environment variable is required")
            
        self._base_url = (base_url or os.getenv("ANTHROPIC_BASE_URL") or ANTHROPIC_DEFAULT_BASE_URL).rstrip("/")
        self._timeout = timeout if timeout is not None else float(os.getenv("ANTHROPIC_TIMEOUT_SECONDS", "120"))
        self._stream_timeout = stream_timeout if stream_timeout is not None else float(os.getenv("ANTHROPIC_STREAM_TIMEOUT_SECONDS", "300"))
        self._max_retries = max_retries if max_retries is not None else int(os.getenv("ANTHROPIC_MAX_RETRIES", "2"))
        self._enable_prompt_caching = enable_prompt_caching if enable_prompt_caching is not None else os.getenv("ANTHROPIC_ENABLE_PROMPT_CACHING", "true").lower() in ("true", "1", "yes")

    def _get_headers(self) -> Dict[str, str]:
        headers = {
            "x-api-key": self._api_key,
            "anthropic-version": ANTHROPIC_API_VERSION,
            "content-type": "application/json",
            "anthropic-beta": ",".join(DEFAULT_BETA_HEADERS)
        }
        return headers

    def _reconstruct_assistant_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critical logic for Resuming Agents.
        Reconstructs the 'thinking' block using the saved opaque signature.
        """
        content_blocks = []
        
        # 1. Re-inject Thinking / Signature
        if "provider_specific" in msg:
            ps: ProviderSpecific = msg["provider_specific"]
            sig = ps.get("thought_signature")
            
            if sig:
                is_redacted = ps.get("is_redacted", False)
                reasoning_text = msg.get("reasoning_content", "")

                if is_redacted or reasoning_text == "[Redacted Thinking]":
                    content_blocks.append({
                        "type": "redacted_thinking",
                        "data": sig
                    })
                else:
                    content_blocks.append({
                        "type": "thinking",
                        "thinking": reasoning_text,
                        "signature": sig
                    })

        # 2. Re-inject Text
        if msg.get("content"):
            content_blocks.append({
                "type": "text", 
                "text": _as_json_str(msg["content"])
            })

        # 3. Re-inject Tool Calls
        if "tool_calls" in msg:
            for tc in msg["tool_calls"]:
                fn = tc.get("function", {})
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc.get("id"),
                    "name": fn.get("name"),
                    "input": json.loads(fn.get("arguments", "{}"))
                })
        
        return {"role": "assistant", "content": content_blocks}

    def _map_messages(self, messages: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Transforms generic chat history into Anthropic's strict block format.
        Handles prompt caching markers and tool result ordering.
        """
        formatted_messages = []
        system_prompt = []

        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content")

            # --- System Prompt ---
            if role == "system":
                block = {"type": "text", "text": _as_json_str(content)}
                if self._enable_prompt_caching and len(str(content)) > 1024:
                     block["cache_control"] = {"type": "ephemeral"}
                system_prompt.append(block)
                continue

            # --- Assistant (Reconstruction) ---
            if role == "assistant":
                formatted_messages.append(self._reconstruct_assistant_message(msg))
                continue

            # --- Tool Results (User) ---
            if role == "tool":
                tool_result_block = {
                    "type": "tool_result",
                    "tool_use_id": msg.get("tool_call_id") or msg.get("id"),
                    "content": _as_json_str(content),
                    "is_error": msg.get("is_error", False)
                }

                if formatted_messages and formatted_messages[-1]["role"] == "user":
                    formatted_messages[-1]["content"].append(tool_result_block)
                else:
                    formatted_messages.append({"role": "user", "content": [tool_result_block]})
                continue

            # --- User ---
            if role == "user":
                content_blocks = []
                if isinstance(content, str):
                    content_blocks.append({"type": "text", "text": content})
                elif isinstance(content, list):
                    content_blocks.extend(content)
                
                if self._enable_prompt_caching and len(formatted_messages) > 10:
                     if content_blocks:
                         content_blocks[-1]["cache_control"] = {"type": "ephemeral"}

                formatted_messages.append({"role": "user", "content": content_blocks})

        return {
            "system": system_prompt if system_prompt else None,
            "messages": formatted_messages
        }

    def _extract_response(self, data: Dict[str, Any], model: str) -> Dict[str, Any]:
        """
        Parses Anthropic response, separating Text, Reasoning, and Signatures.
        """
        content_blocks = data.get("content", [])
        
        final_text = ""
        reasoning_text = ""
        tool_calls = []
        
        signature = None
        is_redacted = False

        for block in content_blocks:
            b_type = block.get("type")
            
            if b_type == "text":
                final_text += block.get("text", "")
            elif b_type == "thinking":
                reasoning_text += block.get("thinking", "")
                signature = block.get("signature")
            elif b_type == "redacted_thinking":
                reasoning_text = "[Redacted Thinking]"
                signature = block.get("data")
                is_redacted = True
            elif b_type == "tool_use":
                tool_calls.append({
                    "id": block.get("id"),
                    "type": "function",
                    "function": {
                        "name": block.get("name"),
                        "arguments": json.dumps(block.get("input", {}))
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

        usage = data.get("usage", {})
        provider_specific: ProviderSpecific = {
            "thought_signature": signature,
            "is_redacted": is_redacted,
            "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
            "cache_read_input_tokens": usage.get("cache_read_input_tokens")
        }
        
        if signature or provider_specific.get("cache_read_input_tokens"):
            message["provider_specific"] = provider_specific

        return {
            "id": data.get("id"),
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "usage": usage,
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": data.get("stop_reason")
            }]
        }

    def _prepare_request_body(
        self,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]],
        temperature: Optional[float],
        max_tokens: int,
        thinking_budget: Optional[int],
        top_p: Optional[float],
        stream: bool,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepare request body for both streaming and non-streaming requests."""
        mapped_payload = self._map_messages(messages)
        
        body = {
            "model": model,
            "messages": mapped_payload["messages"],
            "max_tokens": max_tokens,
            "stream": stream,
            **kwargs
        }
        
        if mapped_payload["system"]:
            body["system"] = mapped_payload["system"]

        if thinking_budget:
            if thinking_budget < 1024:
                logger.warning("Thinking budget should be >= 1024. Adjusting automatically.")
                thinking_budget = 1024
            body["thinking"] = {
                "type": "enabled",
                "budget_tokens": thinking_budget
            }
            if temperature is not None:
                body["temperature"] = temperature
        else:
            if temperature is not None:
                body["temperature"] = temperature
            if top_p is not None:
                body["top_p"] = top_p

        if tools:
            body["tools"] = [{
                "name": (t.get("function") or t).get("name"),
                "description": (t.get("function") or t).get("description"),
                "input_schema": (t.get("function") or t).get("parameters")
            } for t in tools]

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
        top_p: Optional[float] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a non-streaming chat completion.
        
        Args:
            thinking_budget (int): Minimum 1024. Enables 'extended thinking'.
        """
        body = self._prepare_request_body(
            messages, model, tools, temperature, max_tokens,
            thinking_budget, top_p, stream=False, **kwargs
        )
        
        url = f"{self._base_url}/v1/messages"
        headers = self._get_headers()
        
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, headers=headers, json=body)
                    
                    if resp.status_code == 429 or resp.status_code >= 500:
                        if attempt < self._max_retries:
                            wait_time = 2 ** attempt
                            logger.warning(f"Anthropic API {resp.status_code}. Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    if resp.status_code >= 400:
                        raise AnthropicError(
                            f"Anthropic API Error {resp.status_code}: {resp.text}",
                            status_code=resp.status_code,
                            body=resp.text
                        )
                        
                    return self._extract_response(resp.json(), model)

            except httpx.RequestError as e:
                if attempt < self._max_retries:
                    logger.warning(f"Network error: {e}. Retrying...")
                    await asyncio.sleep(1)
                    continue
                raise AnthropicError(f"Connection failed after {self._max_retries} retries: {str(e)}")

        raise AnthropicError("Max retries exceeded")

    async def chat_completion_stream(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        thinking_budget: Optional[int] = None,
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
            thinking_budget, top_p, stream=True, **kwargs
        )
        
        url = f"{self._base_url}/v1/messages"
        headers = self._get_headers()

        async def _stream_generator() -> AsyncIterator[str]:
            for attempt in range(self._max_retries + 1):
                try:
                    async with httpx.AsyncClient(timeout=self._stream_timeout) as client:
                        async with client.stream("POST", url, headers=headers, json=body) as response:
                            if response.status_code >= 400:
                                error_body = await response.aread()
                                raise AnthropicError(
                                    f"Stream error {response.status_code}: {error_body.decode()}",
                                    status_code=response.status_code,
                                    body=error_body.decode()
                                )

                            # Stream state tracking
                            message_id = f"anthropic-{uuid.uuid4().hex}"
                            model_name = model
                            tool_buffers: Dict[int, Dict[str, Any]] = {}
                            
                            # Thinking state
                            thinking_text = ""
                            thinking_signature = None
                            is_redacted_thinking = False
                            
                            finish_seen = False

                            # Initial role chunk
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

                            async for raw_line in response.aiter_lines():
                                if not raw_line or raw_line.startswith("event:"):
                                    continue
                                    
                                line = raw_line.strip()
                                if not line.startswith("data:"):
                                    continue
                                    
                                payload = line[len("data:"):].strip()
                                if not payload or payload == "[DONE]":
                                    continue

                                try:
                                    event = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue

                                event_type = event.get("type")

                                # Message start: capture ID and model
                                if event_type == "message_start":
                                    msg = event.get("message", {})
                                    message_id = msg.get("id", message_id)
                                    model_name = msg.get("model", model_name)

                                # Content block start
                                elif event_type == "content_block_start":
                                    block = event.get("content_block", {})
                                    block_type = block.get("type")
                                    idx = event.get("index", 0)

                                    if block_type == "thinking":
                                        # Start collecting thinking content
                                        thinking_signature = block.get("signature")
                                        thinking_text = ""
                                        
                                    elif block_type == "redacted_thinking":
                                        # Redacted thinking (signature only)
                                        thinking_signature = block.get("data")
                                        thinking_text = "[Redacted Thinking]"
                                        is_redacted_thinking = True
                                        
                                        # Emit thinking chunk
                                        chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "reasoning_content": thinking_text,
                                                    "provider_specific": {
                                                        "thought_signature": thinking_signature,
                                                        "is_redacted": True
                                                    }
                                                },
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"
                                        
                                    elif block_type == "tool_use":
                                        # Tool use start
                                        tool_buffers[idx] = {
                                            "id": block.get("id", f"toolu_{uuid.uuid4().hex}"),
                                            "type": "function",
                                            "function": {
                                                "name": block.get("name", ""),
                                                "arguments": ""
                                            }
                                        }
                                        
                                        chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"tool_calls": [tool_buffers[idx]]},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"

                                # Content block delta
                                elif event_type == "content_block_delta":
                                    delta = event.get("delta", {})
                                    delta_type = delta.get("type")
                                    idx = event.get("index", 0)

                                    if delta_type == "thinking_delta":
                                        # Thinking content streaming
                                        thinking_delta = delta.get("thinking", "")
                                        thinking_text += thinking_delta
                                        
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
                                        
                                    elif delta_type == "text_delta":
                                        # Regular text streaming
                                        text = delta.get("text", "")
                                        if text:
                                            chunk = {
                                                "id": message_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": model_name,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"content": text},
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {_json(chunk)}\n\n"
                                            
                                    elif delta_type == "input_json_delta":
                                        # Tool arguments streaming
                                        if idx in tool_buffers:
                                            partial_json = delta.get("partial_json", "")
                                            tool_buffers[idx]["function"]["arguments"] += partial_json
                                            
                                            chunk = {
                                                "id": message_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": model_name,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {"tool_calls": [tool_buffers[idx]]},
                                                    "finish_reason": None
                                                }]
                                            }
                                            yield f"data: {_json(chunk)}\n\n"

                                # Content block stop
                                elif event_type == "content_block_stop":
                                    # If we collected thinking, emit final signature
                                    if thinking_signature and not is_redacted_thinking:
                                        chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {
                                                    "provider_specific": {
                                                        "thought_signature": thinking_signature,
                                                        "is_redacted": False
                                                    }
                                                },
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"

                                # Message completion
                                elif event_type == "message_delta":
                                    delta = event.get("delta", {})
                                    stop_reason = delta.get("stop_reason")
                                    
                                    if stop_reason:
                                        finish_seen = True
                                        finish_chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model_name,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": stop_reason
                                            }]
                                        }
                                        yield f"data: {_json(finish_chunk)}\n\n"

                            # Ensure finish chunk is sent
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
                    raise AnthropicError(f"Stream failed after {self._max_retries} retries: {str(e)}")

            raise AnthropicError("Max retries exceeded for streaming")

        return _stream_generator()


# --- Usage Examples ---
if __name__ == "__main__":
    async def example_non_streaming():
        """Example: Non-streaming with thinking."""
        client = AnthropicClient()
        
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "Explain quantum entanglement in simple terms."}
            ],
            model="claude-3-5-sonnet-20241022",
            thinking_budget=2048,
            max_tokens=1024
        )
        
        msg = response["choices"][0]["message"]
        print(f"💭 Thinking: {msg.get('reasoning_content', 'N/A')}")
        print(f"💬 Response: {msg['content']}")
        print(f"🔑 Signature: {msg.get('provider_specific', {}).get('thought_signature', 'N/A')}")

    async def example_streaming():
        """Example: Streaming with real-time output."""
        client = AnthropicClient()
        
        print("Starting stream...\n")
        
        stream = await client.chat_completion_stream(
            messages=[
                {"role": "user", "content": "Write a haiku about coding."}
            ],
            model="claude-3-5-sonnet-20241022",
            max_tokens=512
        )
        
        async for chunk in stream:
            if chunk.startswith("data: "):
                data_str = chunk[6:].strip()
                if data_str and data_str != "[DONE]":
                    try:
                        data = json.loads(data_str)
                        delta = data["choices"][0]["delta"]
                        
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)
                        elif "reasoning_content" in delta:
                            print(f"\n[Thinking: {delta['reasoning_content']}]", end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        
        print("\n\nStream complete!")

    # Run examples
    # asyncio.run(example_non_streaming())
    # asyncio.run(example_streaming())