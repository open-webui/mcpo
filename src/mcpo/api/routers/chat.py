from __future__ import annotations

import asyncio
import json
import logging
import os
import uuid
from typing import Any, AsyncIterator, Dict, Iterable, List, Optional, Tuple, Union

import httpx
from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.routing import APIRoute
from pydantic import BaseModel, Field
from starlette.routing import Mount

from mcpo.providers.google import GoogleClient, GoogleError, is_google_model
from mcpo.providers.openrouter import OpenRouterClient, OpenRouterError
from mcpo.providers.minimax import MiniMaxClient, MiniMaxError, get_minimax_models, is_minimax_model
from mcpo.services.chat_sessions import ChatSession, ChatSessionManager, ChatStep, get_chat_session_manager
from mcpo.services.mcp_tools import (
    collect_enabled_mcp_sessions_with_names,
    sanitize_tool_name,
)
from mcpo.services.runner import get_runner_service
from mcpo.services.skills import compile_skills_system_prompt, select_skills

logger = logging.getLogger(__name__)


def _normalize_tool_arguments(raw: Any) -> str:
    """Ensure tool_call arguments are valid JSON strings before reuse."""
    try:
        if isinstance(raw, str):
            json.loads(raw)  # validate it parses
            return raw
        return json.dumps(raw)
    except Exception as e:
        safe = json.dumps({"raw": str(raw)})
        logger.warning(f"[CHAT] Invalid tool_call arguments (type={type(raw).__name__}, err={e}); wrapped for safety")
        return safe


def _sanitize_tool_calls_in_messages(messages: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize tool_call argument strings across message history before provider call."""
    sanitized: List[Dict[str, Any]] = []
    for msg in messages:
        if not isinstance(msg, dict):
            sanitized.append(msg)
            continue
        clone = dict(msg)
        tool_calls = clone.get("tool_calls") or []
        if isinstance(tool_calls, list) and tool_calls:
            logger.debug(f"[CHAT] Sanitizing message with {len(tool_calls)} tool_calls")
            cleaned_calls = []
            for tc in tool_calls:
                if not isinstance(tc, dict):
                    cleaned_calls.append(tc)
                    continue
                tc_copy = dict(tc)
                func = tc_copy.get("function") or {}
                if isinstance(func, dict):
                    func_copy = dict(func)
                    original_args = func_copy.get("arguments", "{}")
                    func_copy["arguments"] = _normalize_tool_arguments(original_args)
                    logger.debug(f"[CHAT] tool_call id={tc.get('id')}, args_type={type(original_args).__name__}, len={len(str(original_args))}")
                    tc_copy["function"] = func_copy
                cleaned_calls.append(tc_copy)
            clone["tool_calls"] = cleaned_calls
        sanitized.append(clone)
    return sanitized

router = APIRouter(tags=["chat"], prefix="/sessions")

# Type alias for provider clients
ChatClient = Union[OpenRouterClient, MiniMaxClient, GoogleClient]
ProviderError = Union[OpenRouterError, MiniMaxError, GoogleError]


def _get_client_for_model(model: str) -> ChatClient:
    """
    Return the appropriate provider client based on model ID prefix.
    
    Model ID format:
    - minimax/MiniMax-M2 -> MiniMaxClient
    - openai/gpt-4 -> OpenRouterClient (via OpenRouter)
    - anthropic/claude-3 -> OpenRouterClient (via OpenRouter)
    """
    if is_minimax_model(model):
        return MiniMaxClient()
    if is_google_model(model):
        return GoogleClient()
    return OpenRouterClient()


def _format_model_label(model_id: str) -> str:
    tail = model_id.split("/")[-1]
    return tail.replace("-", " ").replace("_", " ").title()


async def _fetch_openrouter_models() -> List[Dict[str, str]]:
    """Fetch full model list from OpenRouter API when API key is present."""
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
                label = entry.get("name") or _format_model_label(model_id)
                models.append({"id": model_id, "label": label})
            return models
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("OpenRouter model discovery failed: %s", exc)
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
                    models.append({"id": model_id, "label": f"OpenAI: {model_id}"})
            return models
    except Exception as exc:
        logger.warning("OpenAI model discovery failed: %s", exc)
        return []


async def _fetch_google_models() -> List[Dict[str, str]]:
    """Fetch Google/Gemini model catalog when GOOGLE_API_KEY is present."""
    api_key = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
    if not api_key:
        return []
    base_url = os.getenv("GOOGLE_BASE_URL", "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base_url}/v1beta/models"
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url, params={"key": api_key})
            resp.raise_for_status()
            payload = resp.json()
            data = payload.get("models") or []
            models: List[Dict[str, str]] = []
            for entry in data:
                full_name = entry.get("name", "")
                model_id = full_name.replace("models/", "") if full_name.startswith("models/") else full_name
                if not model_id:
                    continue
                supported = entry.get("supportedGenerationMethods") or []
                if "generateContent" not in supported:
                    continue
                display = entry.get("displayName") or model_id
                models.append({"id": model_id, "label": f"Google: {display}"})
            return models
    except Exception as exc:  # pragma: no cover - network errors
        logger.warning("Google model discovery failed: %s", exc)
        return []


async def _load_model_catalog() -> List[Dict[str, str]]:
    """
    Build model catalog from all available providers:
    - MiniMax (if MINIMAX_API_KEY set)
    - OpenRouter (live list or configured)
    - OpenAI (if OPEN_AI_API_KEY set)
    """
    combined: List[Dict[str, str]] = []
    existing: set = set()

    # Add MiniMax models if API key present
    if os.getenv("MINIMAX_API_KEY"):
        for model in get_minimax_models():
            if model["id"] not in existing:
                combined.append(model)
                existing.add(model["id"])

    # OpenRouter configured models
    configured: List[Dict[str, str]] = []
    raw_list = os.getenv("OPENROUTER_MODELS")
    if raw_list:
        for item in raw_list.split(","):
            model_id = item.strip()
            if not model_id:
                continue
            configured.append({
                "id": model_id,
                "label": _format_model_label(model_id),
            })

    env_model = os.getenv("OPENROUTER_MODEL")
    if env_model:
        label = os.getenv("OPENROUTER_MODEL_LABEL") or _format_model_label(env_model)
        configured.insert(0, {"id": env_model, "label": label})

    # Prefer live OpenRouter list if we can fetch it
    live_models = await _fetch_openrouter_models()

    for model in (live_models or []):
        if model["id"] not in existing:
            combined.append(model)
            existing.add(model["id"])

    for entry in configured:
        if entry["id"] not in existing:
            combined.append(entry)
            existing.add(entry["id"])

    # Add OpenAI models if API key present
    openai_models = await _fetch_openai_models()
    for model in openai_models:
        if model["id"] not in existing:
            combined.append(model)
            existing.add(model["id"])

    # Add Google/Gemini models if API key present
    google_models = await _fetch_google_models()
    for model in google_models:
        if model["id"] not in existing:
            combined.append(model)
            existing.add(model["id"])

    if not combined:
        combined = [{"id": "openrouter/auto", "label": "OpenRouter Auto"}]

    return combined


class CreateSessionRequest(BaseModel):
    model: Optional[str] = Field(None, description="OpenRouter model identifier")
    system_prompt: Optional[str] = Field(None, description="Optional system prompt")
    server_allowlist: Optional[List[str]] = Field(
        None, description="Restrict available MCP servers to this allowlist"
    )
    skill_ids: Optional[List[str]] = Field(
        None, description="Skill IDs to activate for this session"
    )


class CreateSessionResponse(BaseModel):
    ok: bool = True
    session: Dict[str, Any]


class ChatMessageRequest(BaseModel):
    message: str = Field(..., description="User message to send")
    stream: bool = Field(True, description="Whether to stream the assistant response")
    temperature: Optional[float] = Field(None, ge=0.0, le=2.0)
    max_output_tokens: Optional[int] = Field(None, gt=0)
    model: Optional[str] = Field(None, description="Override the session model")
    include_reasoning: Optional[bool] = Field(True, description="Request provider reasoning tokens when supported")
    reasoning_effort: Optional[str] = Field(None, description="Provider-specific reasoning effort hint (e.g., low/medium/high)")
    skill_ids: Optional[List[str]] = Field(
        None, description="Skill IDs to inject for this message"
    )


class ResetSessionResponse(BaseModel):
    ok: bool = True
    session: Dict[str, Any]


async def get_session_manager() -> ChatSessionManager:
    return await get_chat_session_manager()


def _json_default(value: Any) -> Any:
    if isinstance(value, (ChatSession,)):
        return value.to_dict()
    if isinstance(value, ChatStep):
        return value.to_dict()
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:  # pragma: no cover - defensive
            pass
    return value


def _extract_mcpo_tools(main_app) -> List[Tuple[str, Dict[str, Any]]]:
    """
    Extract MCPO management tools from the /mcpo mounted FastAPI app.
    Returns list of (function_name, tool_definition) tuples.
    """
    tools: List[Tuple[str, Dict[str, Any]]] = []
    
    # Find the /mcpo mounted app
    mcpo_app = None
    for route in main_app.router.routes:
        if isinstance(route, Mount) and route.path == "/mcpo":
            mcpo_app = getattr(route, "app", None)
            break
    
    if not mcpo_app:
        logger.debug("MCPO app not found at /mcpo mount")
        return tools
    
    # Extract tool definitions from MCPO app routes
    for route in mcpo_app.routes:
        if not isinstance(route, APIRoute):
            continue
        
        # Get operation details
        path = route.path  # e.g., "/get_config"
        methods = route.methods or set()
        
        # Determine primary method (prefer POST, then GET)
        method = "POST" if "POST" in methods else ("GET" if "GET" in methods else None)
        if not method:
            continue
        
        # Build function name from operation_id or path
        operation_id = getattr(route, "operation_id", None)
        if operation_id:
            function_name = sanitize_tool_name(operation_id)
        else:
            # Fallback: mcpo_<path_without_slash>
            function_name = sanitize_tool_name(f"mcpo{path.replace('/', '_')}")
        
        # Extract description and parameters from route
        description = route.description or route.summary or f"MCPO management tool: {path}"
        
        # Build parameter schema from route's request body model
        parameters: Dict[str, Any] = {"type": "object", "properties": {}, "required": []}
        
        if route.body_field and route.body_field.type_:
            model = route.body_field.type_
            if hasattr(model, "model_json_schema"):
                schema = model.model_json_schema()
                parameters["properties"] = schema.get("properties", {})
                parameters["required"] = schema.get("required", [])
        
        tools.append((function_name, {
            "method": method,
            "path": f"/mcpo{path}",
            "description": description,
            "parameters": parameters,
        }))
    
    return tools


async def _gather_tool_catalog(
    app: Request, allowlist: Optional[List[str]] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    tool_defs: List[Dict[str, Any]] = []
    tool_index: Dict[str, Dict[str, Any]] = {}

    sessions = collect_enabled_mcp_sessions_with_names(app.app, allowlist=allowlist)
    used_names: Dict[str, int] = {}

    # Collect all tools from MCP sessions
    tools_by_server: Dict[str, List[Dict[str, Any]]] = {}
    for server_name, session in sessions:
        try:
            result = await session.list_tools()
        except Exception:
            continue

        for tool in getattr(result, "tools", []) or []:
            original_name = f"{server_name}.{tool.name}"
            sanitized_name = sanitize_tool_name(original_name)
            if sanitized_name in tool_index:
                counter = used_names.get(sanitized_name, 1)
                while f"{sanitized_name}_{counter}" in tool_index:
                    counter += 1
                used_names[sanitized_name] = counter + 1
                function_name = f"{sanitized_name}_{counter}"
            else:
                used_names[sanitized_name] = 1
                function_name = sanitized_name
            tool_schema = tool.inputSchema or {"type": "object", "properties": {}}
            description = tool.description or f"Tool '{tool.name}' on '{server_name}'"

            tool_defs.append(
                {
                    "type": "function",
                    "function": {
                        "name": function_name,
                        "description": description,
                        "parameters": tool_schema,
                    },
                }
            )

            tool_index[function_name] = {
                "server": server_name,
                "session": session,
                "tool": tool,
                "originalName": original_name,
            }

            # Track for code mode catalog
            tools_by_server.setdefault(server_name, []).append({
                "name": tool.name,
                "description": description,
                "inputSchema": tool_schema,
            })

    # Check if code mode is active
    from mcpo.services.state import get_state_manager as _get_sm
    state_mgr = _get_sm()
    if state_mgr.is_code_mode_enabled():
        # Replace all individual tools with the two code mode meta-tools
        from mcpo.services.code_mode import (
            build_catalog,
            get_code_mode_tool_definitions,
        )
        code_catalog = build_catalog(tools_by_server)

        # Keep the original tool_index for execute_tool routing
        original_tool_index = dict(tool_index)

        tool_defs = []
        tool_index = {}

        for meta_tool in get_code_mode_tool_definitions():
            tool_defs.append({
                "type": "function",
                "function": {
                    "name": meta_tool["name"],
                    "description": meta_tool["description"],
                    "parameters": meta_tool["inputSchema"],
                },
            })
            tool_index[meta_tool["name"]] = {
                "server": "__code_mode__",
                "is_code_mode_tool": True,
                "code_catalog": code_catalog,
                "original_tool_index": original_tool_index,
            }

        logger.info(
            "Code mode: replaced %d tools with 2 meta-tools (search_tools, execute_tool)",
            len(original_tool_index),
        )
        return tool_defs, tool_index

    # Add MCPO management tools
    mcpo_tools = _extract_mcpo_tools(app.app)
    for function_name, tool_info in mcpo_tools:
        # Skip if name collision (unlikely but defensive)
        if function_name in tool_index:
            logger.warning(f"MCPO tool name collision: {function_name}")
            continue

        tool_defs.append({
            "type": "function",
            "function": {
                "name": function_name,
                "description": tool_info["description"],
                "parameters": tool_info["parameters"],
            },
        })

        tool_index[function_name] = {
            "server": "mcpo",
            "is_mcpo_tool": True,
            "method": tool_info["method"],
            "path": tool_info["path"],
            "originalName": function_name,
            "main_app": app.app,  # Store reference for ASGI transport
        }

    logger.info(f"Tool catalog: {len(tool_defs)} tools ({len(mcpo_tools)} MCPO management tools)")

    return tool_defs, tool_index


async def _ensure_tools(session: ChatSession, request: Request) -> None:
    if session.tool_definitions:
        return
    tool_defs, tool_index = await _gather_tool_catalog(
        request,
        allowlist=session.server_allowlist,
    )
    session.tool_definitions = tool_defs
    session.tool_index = tool_index


@router.get("/models")
async def list_models() -> Dict[str, Any]:
    models = await _load_model_catalog()
    return {"models": models}


# --- Favorite Models Endpoints ---
from mcpo.services.state import get_state_manager


class FavoriteModelsRequest(BaseModel):
    models: List[str] = Field(..., description="List of favorite model IDs")


@router.get("/favorites")
async def get_favorite_models() -> Dict[str, Any]:
    """Get the list of favorite model IDs (persisted server-side)."""
    state = get_state_manager()
    return {"favorites": state.get_favorite_models()}


@router.post("/favorites")
async def set_favorite_models(payload: FavoriteModelsRequest) -> Dict[str, Any]:
    """Set the list of favorite model IDs (persisted server-side)."""
    state = get_state_manager()
    state.set_favorite_models(payload.models)
    return {"ok": True, "favorites": state.get_favorite_models()}


@router.post("", response_model=CreateSessionResponse)
async def create_session(
    request: Request,
    payload: CreateSessionRequest,
    manager: ChatSessionManager = Depends(get_session_manager),
) -> CreateSessionResponse:
    tool_defs, tool_index = await _gather_tool_catalog(request, allowlist=payload.server_allowlist)
    available_models = await _load_model_catalog()
    if payload.model is None:
        payload.model = available_models[0]["id"]
    else:
        ids = {entry["id"] for entry in available_models}
        if payload.model not in ids:
            available_models.insert(0, {"id": payload.model, "label": _format_model_label(payload.model)})

    # Compile skills into the system prompt
    system_prompt = payload.system_prompt
    skill_ids = payload.skill_ids or []
    skills_prompt = compile_skills_system_prompt(
        scope="chat",
        model=payload.model,
        provider=None,
        requested_skill_ids=skill_ids if skill_ids else None,
    )
    if skills_prompt:
        system_prompt = f"{system_prompt}\n\n{skills_prompt}" if system_prompt else skills_prompt

    session = await manager.create_session(
        model=payload.model,
        system_prompt=system_prompt or None,
        tool_definitions=tool_defs,
        tool_index=tool_index,
        server_allowlist=payload.server_allowlist,
        skill_ids=skill_ids,
    )
    return CreateSessionResponse(session=session.to_dict())


@router.get("/{session_id}")
async def get_session(session_id: str, manager: ChatSessionManager = Depends(get_session_manager)):
    try:
        session = await manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return {"ok": True, "session": session.to_dict()}


@router.delete("/{session_id}")
async def delete_session(session_id: str, manager: ChatSessionManager = Depends(get_session_manager)):
    await manager.delete_session(session_id)
    return JSONResponse({"ok": True})


@router.post("/{session_id}/reset", response_model=ResetSessionResponse)
async def reset_session(
    session_id: str,
    manager: ChatSessionManager = Depends(get_session_manager),
):
    try:
        session = await manager.reset_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")
    return ResetSessionResponse(session=session.to_dict())


@router.post("/{session_id}/messages")
async def post_message(
    request: Request,
    session_id: str,
    payload: ChatMessageRequest,
    manager: ChatSessionManager = Depends(get_session_manager),
):
    try:
        session = await manager.get_session(session_id)
    except KeyError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Session not found")

    if payload.model and payload.model != session.model:
        session.model = payload.model

    await _ensure_tools(session, request)

    runner = get_runner_service()
    client = _get_client_for_model(session.model)

    tool_timeout = getattr(request.app.state, "tool_timeout", 30)
    tool_timeout_max = getattr(request.app.state, "tool_timeout_max", 600)

    if payload.stream:
        return StreamingResponse(
            _stream_exchange(
                session, client, runner, payload, tool_timeout, tool_timeout_max
            ),
            media_type="text/event-stream",
        )

    result = await _perform_exchange(
        session,
        client,
        runner,
        payload,
        tool_timeout,
        tool_timeout_max,
        emitter=None,
    )
    return {"ok": True, "session": session.to_dict(), "message": result}


async def _stream_exchange(
    session: ChatSession,
    client: ChatClient,
    runner,
    payload: ChatMessageRequest,
    tool_timeout: Optional[float],
    tool_timeout_max: Optional[float],
) -> AsyncIterator[str]:
    queue: asyncio.Queue[str] = asyncio.Queue()

    async def emit(event_type: str, **data: Any) -> None:
        event = {"type": event_type, **data}
        blob = json.dumps(event, default=_json_default)
        await queue.put(f"data: {blob}\n\n")

    async def worker() -> None:
        try:
            await _perform_exchange(
                session,
                client,
                runner,
                payload,
                tool_timeout,
                tool_timeout_max,
                emitter=emit,
            )
        except (OpenRouterError, MiniMaxError) as exc:
            await emit("error", message=str(exc))
        except HTTPException as exc:
            await emit("error", message=str(exc.detail))
        except Exception as exc:  # pragma: no cover - unexpected
            await emit("error", message=str(exc))
        finally:
            await emit("done")
            await queue.put("__CLOSE__")

    asyncio.create_task(worker())

    while True:
        chunk = await queue.get()
        if chunk == "__CLOSE__":
            break
        yield chunk


def _inject_skills_system_message(session: ChatSession, skills_prompt: str) -> None:
    """Insert or update a skills system message at the start of the conversation.

    If the session already has a system message containing skills, update it.
    Otherwise prepend a new system message with the skills prompt.
    """
    skills_marker = "Agent Skills (system-managed instructions):"
    for i, msg in enumerate(session.messages):
        if msg.get("role") == "system" and skills_marker in (msg.get("content") or ""):
            session.messages[i]["content"] = msg["content"].split(skills_marker)[0].rstrip() + "\n\n" + skills_prompt
            return
    # No existing skills system message — check if there's any system message to append to
    for i, msg in enumerate(session.messages):
        if msg.get("role") == "system":
            session.messages[i]["content"] = msg["content"] + "\n\n" + skills_prompt
            return
    # No system message at all — insert one
    session.messages.insert(0, {"role": "system", "content": skills_prompt})


async def _perform_exchange(
    session: ChatSession,
    client: ChatClient,
    runner,
    payload: ChatMessageRequest,
    tool_timeout: Optional[float],
    tool_timeout_max: Optional[float],
    emitter: Optional[Any],
) -> Dict[str, Any]:
    # Inject skills into the conversation if skill_ids are provided on this message
    request_skill_ids = payload.skill_ids or session.skill_ids or []
    if request_skill_ids:
        session.skill_ids = list(request_skill_ids)
    loaded_skills = select_skills(
        scope="chat",
        model=session.model,
        provider=None,
        requested_skill_ids=request_skill_ids if request_skill_ids else None,
    )
    if loaded_skills:
        skills_prompt = compile_skills_system_prompt(
            scope="chat",
            model=session.model,
            provider=None,
            requested_skill_ids=request_skill_ids if request_skill_ids else None,
        )
        # Update the system message in the session if skills content changed
        if skills_prompt:
            _inject_skills_system_message(session, skills_prompt)
        if emitter:
            await emitter(
                "skills.loaded",
                skills=[
                    {"id": s.id, "title": s.title}
                    for s in loaded_skills
                ],
            )

    user_message = {"role": "user", "content": payload.message}
    session.messages.append(user_message)
    if emitter:
        await emitter("session.updated", session=session.to_dict())

    assistant_message: Optional[Dict[str, Any]] = None
    iteration = 0

    while True:
        iteration += 1
        step = ChatStep(
            id=uuid.uuid4().hex,
            type="agent_step",
            title=f"Step {iteration}: Generating response",
            detail={"phase": "generation"},
        )
        session.steps.append(step)
        if emitter:
            await emitter("step.started", step=step.to_dict())

        result = await _call_provider(
            session,
            client,
            payload,
            stream=payload.stream and emitter is not None,
            emitter=emitter,
        )

        assistant_message = result["message"]
        tool_calls: List[Dict[str, Any]] = result["tool_calls"]
        finish_reason = result["finish_reason"]
        
        logger.info(f"[CHAT] Provider returned: finish_reason={finish_reason}, tool_calls_count={len(tool_calls)}")
        if tool_calls:
            logger.info(f"[CHAT] Tool calls to execute: {[tc.get('function',{}).get('name') for tc in tool_calls]}")

        if tool_calls:
            # Append the assistant message with tool_calls first (required by OpenAI API)
            session.messages.append(assistant_message)
            logger.info(f"[CHAT] Appended assistant message with {len(tool_calls)} tool_calls to session")
            
            step.detail.setdefault("toolCalls", [])
            for tool_call in tool_calls:
                tc_func = tool_call.get('function', {})
                tc_id = tool_call.get('id')
                tc_name = tc_func.get('name')
                logger.info(f"[TOOL_EVENT] Processing tool_call: id={tc_id}, name={tc_name}")
                step.detail["toolCalls"].append(tool_call)
                if emitter:
                    logger.info(f"[TOOL_EVENT] Emitting tool.call.started: id={tc_id}")
                    await emitter("tool.call.started", toolCall=tool_call)
                output = await _execute_tool(
                    session,
                    runner,
                    tool_call,
                    tool_timeout,
                    tool_timeout_max,
                )
                logger.info(f"[TOOL_EVENT] Tool execution complete: id={tc_id}, output_type={type(output).__name__}")
                tool_func = tool_call.get("function", {})
                tool_name = tool_func.get("name") or tool_call.get("name")
                tool_message = {
                    "role": "tool",
                    "tool_call_id": tool_call["id"],  # Required by OpenAI API
                    "name": tool_name,
                    "content": json.dumps(output, default=_json_default),
                }
                session.messages.append(tool_message)
                if emitter:
                    result_payload = {**tool_call, "result": output}
                    logger.info(f"[TOOL_EVENT] Emitting tool.call.result: id={tc_id}, has_result={output is not None}")
                    await emitter("tool.call.result", toolCall=result_payload)
            if emitter:
                logger.info(f"[TOOL_EVENT] Emitting step.completed with status=tools_executed")
                await emitter("step.completed", step=step.to_dict(), status="tools_executed")
            continue

        session.messages.append(assistant_message)
        
        # For UI display, create message with clean content (no <think> tags)
        # but session.messages retains full content for interleaved thinking
        ui_message = dict(assistant_message)
        clean_content = result.get("clean_content")
        if clean_content is not None:
            ui_message["content"] = clean_content
        
        step.detail.update(
            {
                "finishReason": finish_reason,
                "summary": _summarize(clean_content or assistant_message.get("content", "")),
            }
        )
        session.steps[-1] = step
        if emitter:
            await emitter(
                "message.completed",
                message=ui_message,
                finishReason=finish_reason,
            )
            await emitter("step.completed", step=step.to_dict(), status="completed")
            await emitter("session.updated", session=session.to_dict())
        break

    return assistant_message or {"role": "assistant", "content": ""}


async def _call_provider(
    session: ChatSession,
    client: ChatClient,
    payload: ChatMessageRequest,
    *,
    stream: bool,
    emitter: Optional[Any],
) -> Dict[str, Any]:
    """Call the appropriate provider (OpenRouter or MiniMax) for chat completion."""
    messages = list(session.messages)
    logger.info(f"[CHAT] _call_provider: model={session.model}, stream={stream}, messages_count={len(messages)}")
    
    # Log message roles/types before sanitization
    for i, msg in enumerate(messages):
        role = msg.get("role", "?")
        has_tool_calls = bool(msg.get("tool_calls"))
        has_tool_call_id = bool(msg.get("tool_call_id"))
        content_len = len(str(msg.get("content", "")))
        logger.debug(f"[CHAT] Message[{i}]: role={role}, has_tool_calls={has_tool_calls}, has_tool_call_id={has_tool_call_id}, content_len={content_len}")
    
    messages = _sanitize_tool_calls_in_messages(messages)
    tools = session.tool_definitions or None
    
    logger.info(f"[CHAT] After sanitize: messages_count={len(messages)}, tools_count={len(tools) if tools else 0}")

    if stream and emitter:
        return await _call_provider_stream(
            messages,
            tools,
            client,
            payload,
            session.model,
            emitter,
        )

    extra_kwargs: Dict[str, Any] = {}
    # Only OpenRouter supports reasoning flags; avoid breaking MiniMax
    if isinstance(client, OpenRouterClient):
        extra_kwargs["include_reasoning"] = payload.include_reasoning
        extra_kwargs["reasoning_effort"] = payload.reasoning_effort

    logger.info(f"[COMPLETION] Non-streaming call to {session.model}")
    response = await client.chat_completion(
        messages=messages,
        model=session.model,
        tools=tools,
        temperature=payload.temperature,
        max_output_tokens=payload.max_output_tokens,
        **extra_kwargs,
    )
    logger.info(f"[COMPLETION] Response received, choices={len(response.get('choices', []))}")
    return _interpret_completion(response)


async def _call_provider_stream(
    messages: Iterable[Dict[str, Any]],
    tools: Optional[List[Dict[str, Any]]],
    client: ChatClient,
    payload: ChatMessageRequest,
    model: str,
    emitter: Any,
) -> Dict[str, Any]:
    from mcpo.providers.openrouter import extract_reasoning_from_content
    
    stream_kwargs: Dict[str, Any] = {}
    # Only OpenRouter supports reasoning flags; avoid breaking MiniMax
    if isinstance(client, OpenRouterClient):
        stream_kwargs["include_reasoning"] = payload.include_reasoning
        stream_kwargs["reasoning_effort"] = payload.reasoning_effort

    iterator = await client.chat_completion_stream(
        messages=messages,
        model=model,
        tools=tools,
        temperature=payload.temperature,
        max_output_tokens=payload.max_output_tokens,
        **stream_kwargs,
    )
    
    logger.info(f"[STREAM] Starting stream for model={model}")

    content_parts: List[str] = []
    reasoning_parts: List[str] = []
    reasoning_details: List[Dict[str, Any]] = []
    tool_calls: Dict[str, Dict[str, Any]] = {}
    finish_reason: Optional[str] = None
    role: Optional[str] = None
    in_think_tag = False
    think_buffer = ""
    chunk_count = 0

    async for line in iterator:
        line = line.strip()
        if not line:
            continue
        if line.startswith("data:"):
            data = line[len("data:") :].strip()
        else:
            data = line
        if data == "[DONE]":
            logger.info(f"[STREAM] Received [DONE] after {chunk_count} chunks")
            break
        try:
            payload_obj = json.loads(data)
            chunk_count += 1
            # Log first few chunks and periodically for visibility
            if chunk_count <= 3 or chunk_count % 25 == 0:
                logger.info(f"[STREAM] Chunk #{chunk_count}: {json.dumps(payload_obj)[:300]}")
        except json.JSONDecodeError:
            logger.warning(f"[STREAM] Failed to parse chunk: {data[:200]}")
            continue
        choices = payload_obj.get("choices", [])
        for choice in choices:
            delta = choice.get("delta", {})
            if delta.get("role"):
                role = delta["role"]
            
            # Handle reasoning_details (MiniMax-style interleaved thinking)
            if delta.get("reasoning_details"):
                for rd in delta["reasoning_details"]:
                    reasoning_text = rd.get("text", "")
                    if reasoning_text:
                        reasoning_parts.append(reasoning_text)
                        await emitter("reasoning.delta", text=reasoning_text)
                    # Accumulate full reasoning_details for message history
                    # CRITICAL: Preserve ALL fields including format for MiniMax spec
                    existing_idx = next(
                        (i for i, d in enumerate(reasoning_details) 
                         if d.get("id") == rd.get("id") or d.get("index") == rd.get("index")),
                        None
                    )
                    if existing_idx is not None:
                        reasoning_details[existing_idx]["text"] = (
                            reasoning_details[existing_idx].get("text", "") + reasoning_text
                        )
                    else:
                        # Preserve all MiniMax reasoning_details fields
                        reasoning_details.append({
                            "type": rd.get("type", "reasoning.text"),
                            "id": rd.get("id"),
                            "format": rd.get("format"),  # MiniMax-response-v1
                            "index": rd.get("index", len(reasoning_details)),
                            "text": reasoning_text,
                        })

            # Handle OpenAI/OpenRouter reasoning_content deltas
            if delta.get("reasoning_content"):
                raw_reason = delta["reasoning_content"]
                reason_list = raw_reason if isinstance(raw_reason, list) else [raw_reason]
                for entry in reason_list:
                    if isinstance(entry, dict):
                        text = entry.get("text", "")
                    else:
                        text = str(entry)
                    if text:
                        reasoning_parts.append(text)
                        await emitter("reasoning.delta", text=text)
            
            if delta.get("content"):
                chunk = delta["content"]
                
                # Handle <think> tags inline (DeepSeek R1 style)
                # Stream reasoning separately from content
                i = 0
                while i < len(chunk):
                    if not in_think_tag:
                        # Look for <think> start
                        think_start = chunk.find("<think>", i)
                        if think_start != -1:
                            # Emit content before the tag
                            before = chunk[i:think_start]
                            if before:
                                content_parts.append(before)
                                await emitter("message.delta", text=before)
                            in_think_tag = True
                            i = think_start + 7  # len("<think>")
                        else:
                            # No tag, emit rest as content
                            rest = chunk[i:]
                            content_parts.append(rest)
                            await emitter("message.delta", text=rest)
                            break
                    else:
                        # Inside think tag, look for </think>
                        think_end = chunk.find("</think>", i)
                        if think_end != -1:
                            # Capture thinking content
                            thinking = chunk[i:think_end]
                            think_buffer += thinking
                            reasoning_parts.append(think_buffer)
                            await emitter("reasoning.delta", text=think_buffer)
                            think_buffer = ""
                            in_think_tag = False
                            i = think_end + 8  # len("</think>")
                        else:
                            # Still inside, buffer it
                            think_buffer += chunk[i:]
                            break
            
            if delta.get("tool_calls"):
                for call in delta["tool_calls"]:
                    # Use index for accumulation - OpenAI sends id only once, then uses index
                    idx = call.get("index", 0)
                    call_id = call.get("id")
                    entry = tool_calls.setdefault(
                        idx,
                        {
                            "id": "",
                            "name": "",
                            "arguments": "",
                            "type": "function",
                        },
                    )
                    # Update id if provided (first delta has the id)
                    if call_id:
                        entry["id"] = call_id
                    if call.get("type"):
                        entry["type"] = call["type"]
                    function = call.get("function") or {}
                    if function.get("name"):
                        entry["name"] = function["name"]
                    if function.get("arguments"):
                        entry["arguments"] += function["arguments"]
                        await emitter(
                            "tool.call.delta",
                            toolCall={"id": entry["id"], "arguments": entry["arguments"]},
                        )
            finish_reason = choice.get("finish_reason") or finish_reason
    
    # Build assistant message
    # CRITICAL: Per MiniMax spec, for <think> tag format, preserve full original
    # content (with tags) in message history for interleaved thinking continuity.
    # For reasoning_details format, we already have separate fields.
    
    full_reasoning = "".join(reasoning_parts)
    
    # If using <think> tags, reconstruct original content with tags for message history
    if full_reasoning and not reasoning_details:
        # Reconstruct content with <think> tags for message history preservation
        # This is required for MiniMax/DeepSeek interleaved thinking to work
        original_content = f"<think>\n{full_reasoning}\n</think>\n\n" + "".join(content_parts)
        clean_content = "".join(content_parts)  # For UI display
    else:
        original_content = "".join(content_parts)
        clean_content = original_content
    
    assistant_message: Dict[str, Any] = {
        "role": role or "assistant",
        "content": original_content,  # Full content for message history
    }
    if full_reasoning:
        assistant_message["reasoning_content"] = full_reasoning
    
    # Preserve reasoning_details for interleaved thinking continuity (MiniMax format)
    if reasoning_details:
        assistant_message["reasoning_details"] = reasoning_details
    if full_reasoning:
        assistant_message["reasoning"] = full_reasoning
    
    # Convert tool_calls to OpenAI API format with nested function object
    tool_call_list = []
    for value in tool_calls.values():
        if value.get("name"):
            arguments = _normalize_tool_arguments(value.get("arguments", "{}"))
            tool_call_list.append({
                "id": value["id"],
                "type": value.get("type", "function"),
                "function": {
                    "name": value["name"],
                    "arguments": arguments,
                }
            })
    
    if tool_call_list:
        logger.info(f"[STREAM] Built {len(tool_call_list)} tool_calls from stream")
        for tc in tool_call_list:
            logger.info(f"[STREAM] tool_call: id={tc['id']}, name={tc['function']['name']}, args={tc['function']['arguments'][:100]}...")
        assistant_message["tool_calls"] = tool_call_list
    
    logger.info(f"[STREAM] Complete: {chunk_count} chunks, content_len={len(clean_content)}, reasoning_len={len(full_reasoning)}, tool_calls={len(tool_call_list)}, finish_reason={finish_reason}")
    
    return {
        "message": assistant_message,
        "tool_calls": tool_call_list,
        "finish_reason": finish_reason,
        "reasoning": full_reasoning if full_reasoning else None,
        "clean_content": clean_content,  # For UI display without <think> tags
    }


def _interpret_completion(response: Dict[str, Any]) -> Dict[str, Any]:
    from mcpo.providers.openrouter import extract_reasoning_from_content
    
    choices = response.get("choices", [])
    if not choices:
        return {
            "message": {"role": "assistant", "content": ""},
            "tool_calls": [],
            "finish_reason": "stop",
        }
    choice = choices[0]
    message = choice.get("message", {}) or {}
    tool_calls = message.get("tool_calls") or []
    for tc in tool_calls:
        func = tc.get("function") or {}
        func["arguments"] = _normalize_tool_arguments(func.get("arguments", "{}"))
    
    # CRITICAL: Per MiniMax spec, preserve FULL content including <think> tags
    # in message history for interleaved thinking continuity.
    # Extract reasoning for UI display only, but keep original content.
    content = message.get("content", "")
    clean_content, reasoning = extract_reasoning_from_content(content)
    reasoning_content = message.get("reasoning_content")

    # Prefer explicit reasoning_content if provided by provider
    if reasoning_content and not reasoning:
        if isinstance(reasoning_content, list):
            reasoning = "".join(str(part.get("text", part)) for part in reasoning_content)
        else:
            reasoning = str(reasoning_content)
    
    # Build result message - keep original content with <think> tags intact
    result_message = dict(message)
    
    # Preserve reasoning_details if present (MiniMax interleaved thinking format)
    if message.get("reasoning_details"):
        result_message["reasoning_details"] = message["reasoning_details"]
        # Extract reasoning text from reasoning_details for UI
        if not reasoning:
            reasoning = "".join(
                rd.get("text", "") for rd in message["reasoning_details"]
            )
    
    if reasoning_content is not None:
        result_message["reasoning_content"] = reasoning_content

    if reasoning:
        result_message["reasoning"] = reasoning
    
    return {
        "message": result_message,
        "tool_calls": tool_calls,
        "finish_reason": choice.get("finish_reason", "stop"),
        "reasoning": reasoning if reasoning else None,
        "clean_content": clean_content if clean_content != content else None,  # For UI display
    }


async def _execute_mcpo_tool(
    mapping: Dict[str, Any],
    arguments: Dict[str, Any],
    timeout: Optional[float],
) -> Dict[str, Any]:
    """Execute an MCPO management tool via internal ASGI call."""
    method = mapping.get("method", "POST")
    path = mapping["path"]
    main_app = mapping.get("main_app")
    
    if not main_app:
        return {
            "ok": False,
            "error": "MCPO tool execution failed: main_app reference not available",
            "server": "mcpo",
            "tool": mapping["originalName"],
        }
    
    # Use ASGI transport to call the endpoint directly without network I/O
    transport = httpx.ASGITransport(app=main_app)
    async with httpx.AsyncClient(transport=transport, base_url="http://testserver", timeout=timeout or 30.0) as client:
        try:
            if method == "GET":
                response = await client.get(path, params=arguments if arguments else None)
            else:
                response = await client.post(path, json=arguments if arguments else {})
            
            response.raise_for_status()
            result = response.json()
            return {
                "ok": True,
                "output": result,
                "server": "mcpo",
                "tool": mapping["originalName"],
            }
        except httpx.HTTPStatusError as e:
            error_detail = e.response.text
            try:
                error_detail = e.response.json()
            except Exception:
                pass
            return {
                "ok": False,
                "error": f"MCPO tool failed: {e.response.status_code}",
                "detail": error_detail,
                "server": "mcpo",
                "tool": mapping["originalName"],
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"MCPO tool execution failed: {str(e)}",
                "server": "mcpo",
                "tool": mapping["originalName"],
            }


async def _execute_code_mode_tool(
    session: ChatSession,
    runner,
    tool_name: str,
    arguments: Dict[str, Any],
    mapping: Dict[str, Any],
    tool_timeout: Optional[float],
    tool_timeout_max: Optional[float],
) -> Dict[str, Any]:
    """Handle search_tools and execute_tool meta-tool calls for code mode."""
    from mcpo.services.code_mode import search_catalog

    code_catalog = mapping.get("code_catalog", [])
    original_tool_index = mapping.get("original_tool_index", {})

    if tool_name == "search_tools":
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)
        results = search_catalog(code_catalog, query, limit=limit)
        return {
            "ok": True,
            "output": {"tools": results, "total_available": len(code_catalog)},
            "server": "__code_mode__",
            "tool": "search_tools",
        }

    if tool_name == "execute_tool":
        qualified_name = arguments.get("tool", "")
        tool_args = arguments.get("arguments", {})

        if not qualified_name:
            return {
                "ok": False,
                "error": "Missing 'tool' parameter — use search_tools first to find tools",
                "server": "__code_mode__",
                "tool": "execute_tool",
            }

        # Find the tool in the original index by matching qualified name
        # The original index uses sanitized names (e.g. "server-web_search")
        target_mapping = None
        for idx_name, idx_mapping in original_tool_index.items():
            original = idx_mapping.get("originalName", "")
            if original == qualified_name or idx_name == sanitize_tool_name(qualified_name):
                target_mapping = idx_mapping
                break

        if not target_mapping:
            # Try fuzzy match: search catalog for suggestions
            results = search_catalog(code_catalog, qualified_name, limit=3)
            suggestion_text = ""
            if results:
                suggestion_text = "\nDid you mean:\n" + "\n".join(
                    f"  - {r['tool']}: {r['description'][:80]}" for r in results
                )
            return {
                "ok": False,
                "error": f"Tool '{qualified_name}' not found.{suggestion_text}",
                "server": "__code_mode__",
                "tool": "execute_tool",
            }

        # Execute the actual tool via the runner
        try:
            mcp_session = target_mapping["session"]
            mcp_tool = target_mapping["tool"]
            logger.info(
                "[CODE_MODE] Routing execute_tool(%s) → server=%s tool=%s",
                qualified_name, target_mapping["server"], mcp_tool.name,
            )
            result = await runner.execute_tool(
                mcp_session,
                mcp_tool.name,
                tool_args,
                timeout=tool_timeout,
                max_timeout=tool_timeout_max,
            )
            return {
                "ok": True,
                "output": result,
                "server": target_mapping["server"],
                "tool": mcp_tool.name,
            }
        except HTTPException as e:
            error_detail = e.detail if isinstance(e.detail, str) else str(e.detail)
            return {
                "ok": False,
                "error": error_detail,
                "server": target_mapping.get("server", "unknown"),
                "tool": qualified_name,
            }
        except Exception as e:
            return {
                "ok": False,
                "error": f"{type(e).__name__}: {str(e)}",
                "server": target_mapping.get("server", "unknown"),
                "tool": qualified_name,
            }

    return {
        "ok": False,
        "error": f"Unknown code mode tool: {tool_name}",
        "server": "__code_mode__",
        "tool": tool_name,
    }


async def _execute_tool(
    session: ChatSession,
    runner,
    tool_call: Dict[str, Any],
    tool_timeout: Optional[float],
    tool_timeout_max: Optional[float],
) -> Dict[str, Any]:
    # Handle both flat format (name, arguments at top level) 
    # and OpenAI format (nested under function)
    function = tool_call.get("function", {})
    tool_name = function.get("name") or tool_call.get("name")
    raw_arguments = function.get("arguments") or tool_call.get("arguments") or {}
    
    logger.info(f"[TOOL] _execute_tool called: tool_name={tool_name}")
    logger.info(f"[TOOL] raw_arguments type={type(raw_arguments).__name__}, value={raw_arguments}")
    
    if not tool_name:
        return {
            "ok": False,
            "error": "Tool call missing name",
            "server": "unknown",
            "tool": "unknown",
        }

    mapping = session.tool_index.get(tool_name)
    if not mapping:
        return {
            "ok": False,
            "error": f"Tool '{tool_name}' is unavailable",
            "server": "unknown",
            "tool": tool_name,
        }

    arguments: Any
    if isinstance(raw_arguments, str):
        try:
            arguments = json.loads(raw_arguments) if raw_arguments else {}
            logger.info(f"[TOOL] Parsed JSON arguments: {arguments}")
        except json.JSONDecodeError as e:
            logger.warning(f"[TOOL] JSON decode failed: {e}, using raw string")
            arguments = {"_": raw_arguments}
    elif isinstance(raw_arguments, dict):
        arguments = raw_arguments
        logger.info(f"[TOOL] Using dict arguments directly: {arguments}")
    else:
        arguments = {}
        logger.warning(f"[TOOL] Unknown arguments type, defaulting to empty dict")

    # Check if this is a code mode meta-tool
    if mapping.get("is_code_mode_tool"):
        logger.info(f"[TOOL] Executing code mode tool: {tool_name}")
        return await _execute_code_mode_tool(
            session, runner, tool_name, arguments, mapping,
            tool_timeout, tool_timeout_max,
        )

    # Check if this is an MCPO management tool (HTTP-based) vs MCP tool (session-based)
    if mapping.get("is_mcpo_tool"):
        logger.info(f"[TOOL] Executing MCPO internal tool: {tool_name}")
        return await _execute_mcpo_tool(mapping, arguments, tool_timeout)

    # Standard MCP tool execution via runner
    try:
        logger.info(f"[TOOL] Executing MCP tool: server={mapping['server']}, tool={mapping['tool'].name}")
        logger.info(f"[TOOL] Arguments being sent: {json.dumps(arguments, default=str)}")
        result = await runner.execute_tool(
            mapping["session"],
            mapping["tool"].name,
            arguments,
            timeout=tool_timeout,
            max_timeout=tool_timeout_max,
        )
        logger.info(f"Tool {mapping['tool'].name} executed successfully")
        return {
            "ok": True,
            "output": result,
            "server": mapping["server"],
            "tool": mapping["tool"].name,
        }
    except HTTPException as e:
        # Return error as tool result so LLM can see it and retry
        logger.error(f"HTTPException executing tool {mapping['tool'].name}: {e.detail}")
        if isinstance(e.detail, str):
            error_detail = e.detail
        else:
            # Include both message and underlying error if present
            msg = e.detail.get("message", "")
            underlying = e.detail.get("error", "")
            error_detail = f"{msg}: {underlying}" if underlying else msg or str(e.detail)
        return {
            "ok": False,
            "error": error_detail,
            "server": mapping["server"],
            "tool": mapping["tool"].name,
        }
    except Exception as e:
        logger.error(f"Exception executing tool {mapping['tool'].name}: {type(e).__name__}: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        return {
            "ok": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "server": mapping["server"],
            "tool": mapping["tool"].name,
        }


def _summarize(content: str, *, max_length: int = 200) -> str:
    summary = (content or "").strip()
    if len(summary) <= max_length:
        return summary
    return summary[: max_length - 1].rstrip() + "…"
