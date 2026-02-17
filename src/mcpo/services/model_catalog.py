from __future__ import annotations

import asyncio
import os
from typing import Dict, List

import httpx

from mcpo.providers.glm import get_glm_models
from mcpo.providers.kimi import get_kimi_models
from mcpo.providers.minimax import get_minimax_models


def _format_model_label(model_id: str) -> str:
    tail = model_id.split("/")[-1]
    return tail.replace("-", " ").replace("_", " ").title()


async def fetch_openrouter_models() -> List[Dict[str, str]]:
    models: List[Dict[str, str]] = []
    seen: set[str] = set()

    # Live catalog (if key exists)
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        url = f"{base_url}/models"
        headers = {"Authorization": f"Bearer {api_key}"}
        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                resp = await client.get(url, headers=headers)
                resp.raise_for_status()
                payload = resp.json()
                data = payload.get("data") or []
                for entry in data:
                    model_id = entry.get("id")
                    if not model_id or model_id in seen:
                        continue
                    seen.add(model_id)
                    models.append(
                        {
                            "id": model_id,
                            "label": entry.get("name") or _format_model_label(model_id),
                            "provider": "openrouter",
                        }
                    )
        except Exception:
            pass

    # Configured catalog (env-driven, key optional)
    raw_list = os.getenv("OPENROUTER_MODELS")
    if raw_list:
        for item in raw_list.split(","):
            model_id = item.strip()
            if not model_id or model_id in seen:
                continue
            seen.add(model_id)
            models.append({"id": model_id, "label": _format_model_label(model_id), "provider": "openrouter"})

    env_model = os.getenv("OPENROUTER_MODEL")
    if env_model and env_model not in seen:
        models.insert(
            0,
            {
                "id": env_model,
                "label": os.getenv("OPENROUTER_MODEL_LABEL") or _format_model_label(env_model),
                "provider": "openrouter",
            },
        )

    return models


async def fetch_openai_models() -> List[Dict[str, str]]:
    api_key = os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
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
            out: List[Dict[str, str]] = []
            for entry in data:
                model_id = entry.get("id")
                if not model_id:
                    continue
                if any(model_id.startswith(p) for p in ("gpt-", "o1", "o3", "o4", "chatgpt-", "codex")):
                    out.append({"id": model_id, "label": f"OpenAI: {model_id}", "provider": "openai"})
            return out
    except Exception:
        return []


async def fetch_google_models() -> List[Dict[str, str]]:
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
            out: List[Dict[str, str]] = []
            for entry in data:
                full_name = entry.get("name", "")
                model_id = full_name.replace("models/", "") if full_name.startswith("models/") else full_name
                if not model_id:
                    continue
                supported = entry.get("supportedGenerationMethods") or []
                if "generateContent" not in supported:
                    continue
                display = entry.get("displayName") or model_id
                out.append({"id": model_id, "label": f"Google: {display}", "provider": "gemini"})
            return out
    except Exception:
        return []


async def fetch_anthropic_models() -> List[Dict[str, str]]:
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
            out: List[Dict[str, str]] = []
            for entry in data:
                model_id = entry.get("id")
                if not model_id:
                    continue
                display = entry.get("display_name") or model_id
                out.append({"id": model_id, "label": f"Anthropic: {display}", "provider": "anthropic"})
            return out
    except Exception:
        return []


async def fetch_minimax_models() -> List[Dict[str, str]]:
    if not os.getenv("MINIMAX_API_KEY"):
        return []
    return [{"id": m["id"], "label": m.get("label", m["id"]), "provider": "minimax"} for m in get_minimax_models()]


async def fetch_glm_models() -> List[Dict[str, str]]:
    if not (os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY")):
        return []
    return [{"id": m["id"], "label": m.get("label", m["id"]), "provider": "glm"} for m in get_glm_models()]


async def fetch_kimi_models() -> List[Dict[str, str]]:
    if not (os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")):
        return []
    return [{"id": m["id"], "label": m.get("label", m["id"]), "provider": "kimi"} for m in get_kimi_models()]


async def list_all_models() -> List[Dict[str, str]]:
    results = await asyncio.gather(
        fetch_minimax_models(),
        fetch_glm_models(),
        fetch_kimi_models(),
        fetch_openrouter_models(),
        fetch_openai_models(),
        fetch_google_models(),
        fetch_anthropic_models(),
        return_exceptions=True,
    )
    all_models: List[Dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for result in results:
        if isinstance(result, Exception):
            continue
        for model in result:
            model_id = model.get("id")
            provider = model.get("provider")
            if model_id and provider and (provider, model_id) not in seen:
                seen.add((provider, model_id))
                all_models.append(model)
    return all_models
