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
logger = logging.getLogger("openai_client")

OPENAI_DEFAULT_BASE_URL = "https://api.openai.com"
OPENAI_API_VERSION = "v1"

class OpenAIError(RuntimeError):
    """Base exception for OpenAI API errors."""
    def __init__(self, message: str, status_code: Optional[int] = None, body: Any = None):
        super().__init__(message)
        self.status_code = status_code
        self.body = body

# --- Type Definitions ---
class ProviderSpecific(TypedDict, total=False):
    reasoning_item_id: Optional[str]  # For Responses API
    reasoning_tokens: Optional[int]
    cached_tokens: Optional[int]  # OpenAI doesn't distinguish creation vs read

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

def _is_reasoning_model(model: str) -> bool:
    """
    Check if model is a reasoning model that uses max_completion_tokens 
    and reasoning_effort instead of temperature.
    
    Includes: o1, o1-pro, o3, o4, gpt-5.x, codex-mini
    """
    m = model.lower()
    return (
        m.startswith("o1") or 
        m.startswith("o3") or 
        m.startswith("o4") or
        m.startswith("gpt-5") or
        "gpt-5" in m or  # Catches gpt-5.1, gpt-5.5, etc.
        m.startswith("codex")  # codex-mini is a reasoning model
    )

def _supports_responses_api(model: str) -> bool:
    """
    Check if model supports the Responses API (stateful, reasoning summaries).
    
    Responses API supported: o1-pro, o3, o4, gpt-5.x, codex-mini
    Note: o1-mini/o1-preview use Chat Completions, o1-pro uses Responses API
    """
    m = model.lower()
    return (
        "o1-pro" in m or  # o1-pro, o1-pro-preview
        "o3" in m or 
        "o4" in m or
        "gpt-5" in m or  # gpt-5, gpt-5.1, gpt-5.5, gpt-5-mini, etc.
        m.startswith("codex")  # codex-mini
    )

class OpenAIClient:
    """
    Advanced Async Client for OpenAI GPT-4, GPT-5, and o-series models.
    
    Feature parity with Anthropic/Gemini clients:
    1. Reasoning Control: Support for reasoning_effort (low/medium/high/minimal)
    2. Reasoning Summaries: Access to reasoning_item_id for state persistence (Responses API)
    3. Context Caching: Automatic prompt caching for long contexts
    4. Tool Calling: Full function calling with parallel execution support
    5. Streaming: Real-time SSE streaming for both Chat Completions and Responses API
    6. State Persistence: Provider-specific metadata for agent resumption
    7. Dual API Support: Chat Completions (legacy) + Responses API (stateful)
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        stream_timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        reasoning_effort: Optional[str] = None,
        use_responses_api: Optional[bool] = None
    ) -> None:
        # Support both OPEN_AI_API_KEY and OPENAI_API_KEY env vars
        self._api_key = api_key or os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
        if not self._api_key:
            raise OpenAIError(
                "OPEN_AI_API_KEY or OPENAI_API_KEY environment variable is required"
            )
        
        # Base URL    
        self._base_url = (
            base_url or 
            os.getenv("OPEN_AI_BASE_URL") or
            os.getenv("OPENAI_BASE_URL") or 
            OPENAI_DEFAULT_BASE_URL
        ).rstrip("/")
        
        # Request timeout
        self._timeout = (
            timeout if timeout is not None else
            float(os.getenv("OPEN_AI_TIMEOUT_SECONDS", "120"))
        )
        
        # Stream timeout (longer for SSE)
        self._stream_timeout = (
            stream_timeout if stream_timeout is not None else
            float(os.getenv("OPEN_AI_STREAM_TIMEOUT_SECONDS", "300"))
        )
        
        # Retry count
        self._max_retries = (
            max_retries if max_retries is not None else
            int(os.getenv("OPEN_AI_MAX_RETRIES", "2"))
        )
        
        # Default reasoning effort for reasoning models
        self._default_reasoning_effort = (
            reasoning_effort or
            os.getenv("OPEN_AI_REASONING_EFFORT")  # None if not set
        )
        
        # Prefer Responses API for reasoning models
        self._use_responses_api = (
            use_responses_api if use_responses_api is not None else
            os.getenv("OPEN_AI_USE_RESPONSES_API", "true").lower() in ("true", "1", "yes")
        )

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def _should_use_responses_api(self, model: str) -> bool:
        """Determine if we should use Responses API for this model."""
        return self._use_responses_api and _supports_responses_api(model)

    def _reconstruct_assistant_message(self, msg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Critical for agent state resumption.
        Reconstructs message with reasoning_item_id for Responses API.
        """
        content = msg.get("content", "")
        role = msg.get("role", "assistant")
        
        message = {"role": role, "content": content}
        
        # Re-inject tool calls
        if "tool_calls" in msg:
            message["tool_calls"] = msg["tool_calls"]
        
        # Note: reasoning_item_id is preserved in provider_specific
        # but NOT sent back to the API - it's for tracking only
        
        return message

    def _map_messages(
        self, 
        messages: Iterable[Dict[str, Any]],
        model: str
    ) -> List[Dict[str, Any]]:
        """
        Transforms generic chat history into OpenAI format.
        Handles system messages, developer messages (for reasoning models), and tool results.
        """
        formatted_messages = []
        
        for msg in messages:
            role = msg.get("role", "").lower()
            content = msg.get("content")

            # System message handling
            if role == "system":
                # For reasoning models (o3, o4, gpt-5), convert to developer message
                if _is_reasoning_model(model):
                    formatted_messages.append({
                        "role": "developer",
                        "content": _as_json_str(content)
                    })
                else:
                    formatted_messages.append({
                        "role": "system",
                        "content": _as_json_str(content)
                    })
                continue

            # Assistant message
            if role == "assistant":
                formatted_messages.append(self._reconstruct_assistant_message(msg))
                continue

            # Tool result
            if role == "tool":
                formatted_messages.append({
                    "role": "tool",
                    "tool_call_id": msg.get("tool_call_id") or msg.get("id"),
                    "content": _as_json_str(content)
                })
                continue

            # User message
            if role == "user":
                if isinstance(content, str):
                    formatted_messages.append({"role": "user", "content": content})
                elif isinstance(content, list):
                    # Multimodal content (images, etc.)
                    formatted_messages.append({"role": "user", "content": content})
                continue

        return formatted_messages

    def _extract_chat_completion_response(
        self, 
        data: Dict[str, Any], 
        model: str
    ) -> Dict[str, Any]:
        """
        Parses Chat Completions API response.
        """
        choices = data.get("choices", [])
        if not choices:
            raise OpenAIError("No choices in response")
        
        choice = choices[0]
        message = choice.get("message", {})
        
        # Build message
        result_message = {
            "role": message.get("role", "assistant"),
            "content": message.get("content", "")
        }
        
        if message.get("tool_calls"):
            result_message["tool_calls"] = message["tool_calls"]

        # Usage stats
        usage = data.get("usage", {})
        
        # Provider-specific metadata
        # FIXED: OpenAI only returns cached_tokens (not separate creation/read)
        cached_tokens = usage.get("prompt_tokens_details", {}).get("cached_tokens")
        reasoning_tokens = usage.get("completion_tokens_details", {}).get("reasoning_tokens")
        
        provider_specific: ProviderSpecific = {}
        
        if reasoning_tokens is not None:
            provider_specific["reasoning_tokens"] = reasoning_tokens
        if cached_tokens is not None:
            provider_specific["cached_tokens"] = cached_tokens
        
        # Only add if we have relevant data
        if provider_specific:
            result_message["provider_specific"] = provider_specific

        return {
            "id": data.get("id"),
            "object": "chat.completion",
            "created": data.get("created", int(time.time())),
            "model": model,
            "usage": usage,
            "choices": [{
                "index": 0,
                "message": result_message,
                "finish_reason": choice.get("finish_reason", "stop")
            }]
        }

    def _extract_responses_api_response(
        self,
        data: Dict[str, Any],
        model: str
    ) -> Dict[str, Any]:
        """
        Parses Responses API response.
        Extracts reasoning summaries and reasoning_item_id for state persistence.
        """
        outputs = data.get("output", [])
        
        final_text = ""
        reasoning_summary = ""
        tool_calls = []
        reasoning_item_id = None
        reasoning_tokens = 0
        
        for output in outputs:
            output_type = output.get("type")
            
            if output_type == "text":
                final_text += output.get("content", "")
            
            elif output_type == "reasoning":
                # Reasoning summary (only available with Responses API)
                reasoning_summary += output.get("summary", "")
                reasoning_item_id = output.get("id")
            
            elif output_type == "tool_call":
                fn = output.get("function", {})
                tool_calls.append({
                    "id": output.get("id"),
                    "type": "function",
                    "function": {
                        "name": fn.get("name", ""),
                        "arguments": fn.get("arguments", "{}")
                    }
                })

        # Build message
        message = {
            "role": "assistant",
            "content": final_text
        }
        
        if tool_calls:
            message["tool_calls"] = tool_calls
        
        if reasoning_summary:
            message["reasoning_content"] = reasoning_summary

        # Usage stats
        usage = data.get("usage", {})
        reasoning_tokens = usage.get("output_tokens_details", {}).get("reasoning_tokens", 0)
        cached_tokens = usage.get("input_tokens_details", {}).get("cached_tokens")
        
        # Provider-specific metadata for state persistence
        provider_specific: ProviderSpecific = {}
        
        if reasoning_item_id:
            provider_specific["reasoning_item_id"] = reasoning_item_id
        if reasoning_tokens:
            provider_specific["reasoning_tokens"] = reasoning_tokens
        if cached_tokens is not None:
            provider_specific["cached_tokens"] = cached_tokens
        
        if provider_specific:
            message["provider_specific"] = provider_specific

        return {
            "id": data.get("id"),
            "object": "chat.completion",
            "created": data.get("created_at", int(time.time())),
            "model": model,
            "usage": {
                "prompt_tokens": usage.get("input_tokens", 0),
                "completion_tokens": usage.get("output_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            },
            "choices": [{
                "index": 0,
                "message": message,
                "finish_reason": data.get("status", "completed")
            }]
        }

    def _prepare_chat_completion_body(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]],
        temperature: Optional[float],
        max_tokens: int,
        reasoning_effort: Optional[str],
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepare body for Chat Completions API."""
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "stream": False
        }
        
        # Reasoning models use max_completion_tokens
        if _is_reasoning_model(model):
            body["max_completion_tokens"] = max_tokens
            
            # Reasoning effort (o3-mini, o1, o4-mini)
            if reasoning_effort:
                valid_efforts = ["low", "medium", "high"]
                # GPT-5 also supports "minimal"
                if model.lower().startswith("gpt-5"):
                    valid_efforts.append("minimal")
                
                if reasoning_effort.lower() not in valid_efforts:
                    logger.warning(f"Invalid reasoning_effort '{reasoning_effort}'. Using 'medium'.")
                    reasoning_effort = "medium"
                
                body["reasoning_effort"] = reasoning_effort.lower()
            
            # Reasoning models don't support temperature
            if temperature is not None:
                logger.warning(f"Reasoning model '{model}' does not support temperature. Ignoring.")
        else:
            # Standard models
            body["max_tokens"] = max_tokens
            if temperature is not None:
                body["temperature"] = temperature

        # Tools
        if tools:
            body["tools"] = list(tools)

        return body

    def _prepare_responses_api_body(
        self,
        messages: List[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]],
        reasoning_effort: Optional[str],
        reasoning_summary: Optional[str],
        max_tokens: int,
        **kwargs: Any
    ) -> Dict[str, Any]:
        """Prepare body for Responses API."""
        # Build input as a list of message objects for multi-turn context
        input_messages = []
        instructions = None
        for msg in messages:
            role = msg.get("role", "")
            if role in ["system", "developer"]:
                # Extract instructions from system/developer messages
                if instructions is None:
                    instructions = msg.get("content", "")
                continue
            input_messages.append({"role": role, "content": msg.get("content", "")})
        
        # Fallback: if no non-system messages, use empty string
        input_content = input_messages if input_messages else ""

        body: Dict[str, Any] = {
            "model": model,
            "input": input_content,
            "max_output_tokens": max_tokens
        }

        if instructions:
            body["instructions"] = instructions

        # Reasoning configuration
        if reasoning_effort or reasoning_summary:
            reasoning_config: Dict[str, Any] = {}
            
            if reasoning_effort:
                valid_efforts = ["low", "medium", "high"]
                if model.lower().startswith("gpt-5"):
                    valid_efforts.append("minimal")
                
                if reasoning_effort.lower() not in valid_efforts:
                    logger.warning(f"Invalid reasoning_effort. Using 'medium'.")
                    reasoning_effort = "medium"
                
                reasoning_config["effort"] = reasoning_effort.lower()
            
            if reasoning_summary:
                # "auto", "detailed", "none"
                if reasoning_summary not in ["auto", "detailed", "none"]:
                    reasoning_summary = "auto"
                reasoning_config["summary"] = reasoning_summary
            
            body["reasoning"] = reasoning_config

        # Tools
        if tools:
            body["tools"] = list(tools)

        return body

    async def chat_completion(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = "auto",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        Execute a non-streaming chat completion.
        
        Args:
            reasoning_effort (str): "low", "medium", "high" (GPT-5: also "minimal")
            reasoning_summary (str): "auto", "detailed", "none" (Responses API only)
        """
        # Use env default if not specified
        effective_reasoning_effort = reasoning_effort or self._default_reasoning_effort
        
        use_responses_api = self._should_use_responses_api(model)
        formatted_messages = self._map_messages(messages, model)
        
        if use_responses_api:
            # Responses API (stateful, reasoning summaries)
            body = self._prepare_responses_api_body(
                formatted_messages, model, tools, effective_reasoning_effort,
                reasoning_summary, max_tokens, **kwargs
            )
            url = f"{self._base_url}/{OPENAI_API_VERSION}/responses"
        else:
            # Chat Completions API (standard)
            body = self._prepare_chat_completion_body(
                formatted_messages, model, tools, temperature,
                max_tokens, effective_reasoning_effort, **kwargs
            )
            url = f"{self._base_url}/{OPENAI_API_VERSION}/chat/completions"
        
        headers = self._get_headers()
        
        for attempt in range(self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    resp = await client.post(url, headers=headers, json=body)
                    
                    if resp.status_code == 429 or resp.status_code >= 500:
                        if attempt < self._max_retries:
                            wait_time = 2 ** attempt
                            logger.warning(f"OpenAI API {resp.status_code}. Retrying in {wait_time}s...")
                            await asyncio.sleep(wait_time)
                            continue
                    
                    if resp.status_code >= 400:
                        error_data = resp.json() if resp.text else {}
                        error_msg = error_data.get("error", {}).get("message", resp.text)
                        raise OpenAIError(
                            f"OpenAI API Error {resp.status_code}: {error_msg}",
                            status_code=resp.status_code,
                            body=resp.text
                        )
                    
                    data = resp.json()
                    
                    if use_responses_api:
                        return self._extract_responses_api_response(data, model)
                    else:
                        return self._extract_chat_completion_response(data, model)

            except httpx.RequestError as e:
                if attempt < self._max_retries:
                    logger.warning(f"Network error: {e}. Retrying...")
                    await asyncio.sleep(1)
                    continue
                raise OpenAIError(f"Connection failed after {self._max_retries} retries: {str(e)}")

        raise OpenAIError("Max retries exceeded")

    async def chat_completion_stream(
        self,
        *,
        messages: Iterable[Dict[str, Any]],
        model: str,
        tools: Optional[Iterable[Dict[str, Any]]] = None,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        reasoning_effort: Optional[str] = None,
        reasoning_summary: Optional[str] = "auto",
        **kwargs: Any,
    ) -> AsyncIterator[str]:
        """
        Execute a streaming chat completion.
        
        Yields:
            Server-Sent Event (SSE) formatted strings: "data: {json}\n\n"
        """
        # Use env default if not specified
        effective_reasoning_effort = reasoning_effort or self._default_reasoning_effort
        
        use_responses_api = self._should_use_responses_api(model)
        formatted_messages = self._map_messages(messages, model)
        
        if use_responses_api:
            body = self._prepare_responses_api_body(
                formatted_messages, model, tools, effective_reasoning_effort,
                reasoning_summary, max_tokens, **kwargs
            )
            body["stream"] = True
            url = f"{self._base_url}/{OPENAI_API_VERSION}/responses"
        else:
            body = self._prepare_chat_completion_body(
                formatted_messages, model, tools, temperature,
                max_tokens, effective_reasoning_effort, **kwargs
            )
            body["stream"] = True
            url = f"{self._base_url}/{OPENAI_API_VERSION}/chat/completions"
        
        headers = self._get_headers()

        async def _stream_generator() -> AsyncIterator[str]:
            for attempt in range(self._max_retries + 1):
                try:
                    # FIXED: Use bounded timeout instead of None
                    async with httpx.AsyncClient(timeout=self._stream_timeout) as client:
                        async with client.stream("POST", url, headers=headers, json=body) as response:
                            if response.status_code >= 400:
                                error_body = await response.aread()
                                raise OpenAIError(
                                    f"Stream error {response.status_code}: {error_body.decode()}",
                                    status_code=response.status_code,
                                    body=error_body.decode()
                                )

                            message_id = f"openai-{uuid.uuid4().hex}"
                            finish_seen = False
                            reasoning_item_id = None

                            # Initial role chunk
                            initial = {
                                "id": message_id,
                                "object": "chat.completion.chunk",
                                "created": int(time.time()),
                                "model": model,
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
                                if not payload or payload == "[DONE]":
                                    continue

                                try:
                                    chunk_data = json.loads(payload)
                                except json.JSONDecodeError:
                                    continue

                                if use_responses_api:
                                    # Responses API streaming
                                    delta = chunk_data.get("delta", {})
                                    
                                    if delta.get("type") == "text":
                                        chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"content": delta.get("content", "")},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"
                                    
                                    elif delta.get("type") == "reasoning":
                                        reasoning_summary = delta.get("summary", "")
                                        reasoning_item_id = delta.get("id")
                                        
                                        chunk = {
                                            "id": message_id,
                                            "object": "chat.completion.chunk",
                                            "created": int(time.time()),
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {"reasoning_content": reasoning_summary},
                                                "finish_reason": None
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"
                                    
                                    elif chunk_data.get("status") == "completed":
                                        finish_seen = True
                                        
                                        if reasoning_item_id:
                                            sig_chunk = {
                                                "id": message_id,
                                                "object": "chat.completion.chunk",
                                                "created": int(time.time()),
                                                "model": model,
                                                "choices": [{
                                                    "index": 0,
                                                    "delta": {
                                                        "provider_specific": {
                                                            "reasoning_item_id": reasoning_item_id
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
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": {},
                                                "finish_reason": "stop"
                                            }]
                                        }
                                        yield f"data: {_json(finish_chunk)}\n\n"
                                else:
                                    # Chat Completions API streaming
                                    choices = chunk_data.get("choices", [])
                                    if not choices:
                                        continue
                                    
                                    choice = choices[0]
                                    delta = choice.get("delta", {})
                                    
                                    # Stream delta
                                    if delta:
                                        chunk = {
                                            "id": chunk_data.get("id", message_id),
                                            "object": "chat.completion.chunk",
                                            "created": chunk_data.get("created", int(time.time())),
                                            "model": model,
                                            "choices": [{
                                                "index": 0,
                                                "delta": delta,
                                                "finish_reason": choice.get("finish_reason")
                                            }]
                                        }
                                        yield f"data: {_json(chunk)}\n\n"
                                    
                                    # Check finish
                                    if choice.get("finish_reason"):
                                        finish_seen = True

                            if not finish_seen:
                                finish_chunk = {
                                    "id": message_id,
                                    "object": "chat.completion.chunk",
                                    "created": int(time.time()),
                                    "model": model,
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
                    raise OpenAIError(f"Stream failed after {self._max_retries} retries: {str(e)}")

            raise OpenAIError("Max retries exceeded for streaming")

        return _stream_generator()


# --- Usage Examples ---
if __name__ == "__main__":
    async def example_gpt4_standard():
        """GPT-4: Standard chat completion."""
        client = OpenAIClient()
        
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "Write a Python function to reverse a string."}
            ],
            model="gpt-4o",
            temperature=0.7,
            max_tokens=512
        )
        
        msg = response["choices"][0]["message"]
        print(f"💬 Response: {msg['content'][:200]}...")

    async def example_o3_reasoning():
        """o3: Advanced reasoning with Responses API."""
        client = OpenAIClient()
        
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "Solve: x^2 + 5x + 6 = 0"}
            ],
            model="o3-mini",
            reasoning_effort="high",
            reasoning_summary="detailed",
            max_tokens=2048
        )
        
        msg = response["choices"][0]["message"]
        print(f"💭 Reasoning: {msg.get('reasoning_content', 'N/A')}")
        print(f"💬 Answer: {msg['content']}")
        print(f"🔑 Item ID: {msg.get('provider_specific', {}).get('reasoning_item_id', 'N/A')}")

    async def example_gpt5_minimal_effort():
        """GPT-5: Minimal reasoning effort (fastest)."""
        client = OpenAIClient()
        
        response = await client.chat_completion(
            messages=[
                {"role": "user", "content": "What's 2+2?"}
            ],
            model="gpt-5-mini",
            reasoning_effort="minimal",  # GPT-5 only
            max_tokens=100
        )
        
        msg = response["choices"][0]["message"]
        print(f"💬 Fast answer: {msg['content']}")

    async def example_streaming():
        """Streaming with o4-mini."""
        client = OpenAIClient()
        
        print("Starting stream...\n")
        
        stream = await client.chat_completion_stream(
            messages=[
                {"role": "user", "content": "Explain TCP/IP in one sentence."}
            ],
            model="o4-mini",
            reasoning_effort="low",
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
                            print(f"\n[Reasoning: {delta['reasoning_content']}]", end="", flush=True)
                    except json.JSONDecodeError:
                        pass
        
        print("\n\nStream complete!")

    # Run examples
    # asyncio.run(example_gpt4_standard())
    # asyncio.run(example_o3_reasoning())
    # asyncio.run(example_gpt5_minimal_effort())
    # asyncio.run(example_streaming())