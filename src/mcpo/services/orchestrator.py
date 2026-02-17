from __future__ import annotations

import os
from typing import Any, Dict, Literal, TypedDict

from mcpo.providers.anthropic import is_anthropic_model
from mcpo.providers.anthropic import AnthropicClient
from mcpo.providers.glm import GLMClient
from mcpo.providers.glm import is_glm_model
from mcpo.providers.google import GoogleClient, is_google_model
from mcpo.providers.kimi import KimiClient
from mcpo.providers.minimax import is_minimax_model
from mcpo.providers.minimax import MiniMaxClient
from mcpo.providers.openai import OpenAIClient, is_openai_model
from mcpo.providers.openrouter import OpenRouterClient

ProviderName = Literal["openai", "openrouter", "anthropic", "gemini", "minimax", "glm", "kimi", "azure"]

# Transport boundary by provider family.
# This is intentionally explicit to avoid endpoint confusion:
# - openai/openrouter: OpenAI-compatible /chat/completions
# - anthropic: Anthropic /v1/messages
# - gemini: Google generateContent/streamGenerateContent
# - minimax/glm/kimi: selectable anthropic-compatible or openai-compatible via env flags
PROVIDER_ENDPOINT_PROTOCOL: Dict[ProviderName, str] = {
    "openai": "openai-compatible",
    "openrouter": "openai-compatible",
    "azure": "openai-compatible",
    "anthropic": "anthropic-messages",
    "gemini": "google-generate-content",
    "minimax": "mode-selectable",
    "glm": "mode-selectable",
    "kimi": "mode-selectable",
}


def has_openrouter_key() -> bool:
    return bool(os.getenv("OPENROUTER_API_KEY"))


def has_openai_key() -> bool:
    return bool(os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY"))


def has_anthropic_key() -> bool:
    return bool(os.getenv("ANTHROPIC_API_KEY"))


def has_google_key() -> bool:
    return bool(os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"))


def has_minimax_key() -> bool:
    return bool(os.getenv("MINIMAX_API_KEY"))


def has_glm_key() -> bool:
    return bool(os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY"))


def has_kimi_key() -> bool:
    return bool(os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY"))


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _canonical_provider_name(provider: str) -> ProviderName:
    p = (provider or "").lower()
    aliases: Dict[str, ProviderName] = {
        "claude": "anthropic",
        "google": "gemini",
        "zhipu": "glm",
        "moonshot": "kimi",
    }
    canonical = aliases.get(p, p)
    allowed: set[str] = {"openai", "openrouter", "azure", "anthropic", "gemini", "minimax", "glm", "kimi"}
    if canonical not in allowed:
        raise ValueError(
            f"Unsupported provider '{provider}'. Use one of: openai, openrouter, anthropic, gemini, minimax, glm, kimi."
        )
    return canonical  # type: ignore[return-value]


def infer_provider_from_model(model: str) -> ProviderName:
    m = (model or "").lower()

    if m.startswith("openai/"):
        return "openai"
    if is_anthropic_model(m):
        return "anthropic"
    if is_minimax_model(m):
        return "minimax"
    if is_glm_model(m):
        return "glm"
    if m.startswith("kimi/") or m.startswith("moonshot/"):
        return "kimi"
    if is_google_model(m):
        return "gemini"
    if m.startswith("openrouter/"):
        return "openrouter"
    raise ValueError(
        "Unable to infer provider from model. Use explicit provider-prefixed ids."
    )


def choose_chat_provider(model: str) -> ProviderName:
    """
    Resolve chat routing with explicit direct-vs-OpenRouter boundaries.
    """
    if is_minimax_model(model) and has_minimax_key():
        return "minimax"
    if is_glm_model(model) and has_glm_key():
        return "glm"
    if (model.lower().startswith("kimi/") or model.lower().startswith("moonshot/")) and has_kimi_key():
        return "kimi"
    if is_google_model(model) and has_google_key():
        return "gemini"
    if is_openai_model(model) and has_openai_key():
        return "openai"
    if is_anthropic_model(model) and has_anthropic_key():
        return "anthropic"
    if model.lower().startswith("openrouter/") and has_openrouter_key():
        return "openrouter"
    raise ValueError(
        "Unable to resolve provider for model. Use explicit provider prefixes and configured keys."
    )


def create_chat_client(model: str) -> Any:
    provider = choose_chat_provider(model)
    if provider == "minimax":
        return MiniMaxClient()
    if provider == "glm":
        return GLMClient()
    if provider == "kimi":
        return KimiClient()
    if provider == "gemini":
        return GoogleClient()
    if provider == "openai":
        return OpenAIClient()
    if provider == "anthropic":
        return AnthropicClient()
    if provider == "openrouter":
        return OpenRouterClient()
    raise ValueError(f"Unsupported provider route: {provider}")


def provider_name_from_chat_client(client: Any) -> ProviderName:
    if isinstance(client, MiniMaxClient):
        return "minimax"
    if isinstance(client, GLMClient):
        return "glm"
    if isinstance(client, KimiClient):
        return "kimi"
    if isinstance(client, GoogleClient):
        return "gemini"
    if isinstance(client, OpenAIClient):
        return "openai"
    if isinstance(client, AnthropicClient):
        return "anthropic"
    if isinstance(client, OpenRouterClient):
        return "openrouter"
    raise ValueError(f"Unsupported chat client type: {type(client).__name__}")


def normalize_model_for_provider(model: str, provider: ProviderName) -> str:
    m = model or ""
    ml = m.lower()
    if provider == "openai" and ml.startswith("openai/"):
        return m.split("/", 1)[1]
    if provider == "anthropic" and ml.startswith("anthropic/"):
        return m.split("/", 1)[1]
    if provider == "gemini" and ml.startswith("google/"):
        return m.split("/", 1)[1]
    if provider == "minimax" and ml.startswith("minimax/"):
        return m.split("/", 1)[1]
    if provider == "glm" and (ml.startswith("glm/") or ml.startswith("zhipu/")):
        return m.split("/", 1)[1]
    if provider == "kimi" and (ml.startswith("kimi/") or ml.startswith("moonshot/")):
        return m.split("/", 1)[1]
    return m


class CompletionRoute(TypedDict):
    provider: ProviderName
    api_key: str | None
    base_url: str | None
    extra_headers: Dict[str, str]


def resolve_completion_route(
    *,
    model: str,
    provider_hint: str | None,
    api_key: str | None,
    base_url: str | None,
) -> CompletionRoute:
    inferred = infer_provider_from_model(model)
    provider = _canonical_provider_name(provider_hint or inferred)
    headers: Dict[str, str] = {}
    resolved_key = api_key
    resolved_base = base_url

    if provider == "openrouter":
        resolved_key = api_key or os.getenv("OPENROUTER_API_KEY")
        resolved_base = base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
        site = os.getenv("OPENROUTER_SITE_URL")
        app = os.getenv("OPENROUTER_APP_NAME")
        if site:
            headers["HTTP-Referer"] = site
        if app:
            headers["X-Title"] = app
    elif provider in {"openai", "azure"}:
        resolved_key = api_key or os.getenv("OPEN_AI_API_KEY") or os.getenv("OPENAI_API_KEY")
        resolved_base = base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPEN_AI_BASE_URL")
    elif provider == "anthropic":
        resolved_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        resolved_base = base_url or os.getenv("ANTHROPIC_BASE_URL")
    elif provider == "gemini":
        resolved_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        resolved_base = base_url or os.getenv("GEMINI_BASE_URL") or os.getenv("GOOGLE_BASE_URL")
    elif provider == "minimax":
        resolved_key = api_key or os.getenv("MINIMAX_API_KEY")
        use_coding = _env_bool("MINIMAX_USE_CODING_PLAN_ENDPOINT", True)
        mode_base = os.getenv("MINIMAX_CODING_PLAN_BASE_URL") if use_coding else os.getenv("MINIMAX_REGULAR_BASE_URL")
        resolved_base = base_url or os.getenv("MINIMAX_BASE_URL") or mode_base
    elif provider == "glm":
        resolved_key = api_key or os.getenv("GLM_API_KEY") or os.getenv("ZHIPU_API_KEY")
        use_coding = _env_bool("GLM_USE_CODING_PLAN_ENDPOINT", True)
        mode_base = os.getenv("GLM_CODING_PLAN_BASE_URL") if use_coding else os.getenv("GLM_REGULAR_BASE_URL")
        resolved_base = base_url or os.getenv("GLM_BASE_URL") or mode_base
    elif provider == "kimi":
        resolved_key = api_key or os.getenv("KIMI_API_KEY") or os.getenv("MOONSHOT_API_KEY")
        use_coding = _env_bool("KIMI_USE_CODING_PLAN_ENDPOINT", False)
        mode_base = os.getenv("KIMI_CODING_PLAN_BASE_URL") if use_coding else os.getenv("KIMI_REGULAR_BASE_URL")
        resolved_base = base_url or os.getenv("KIMI_BASE_URL") or mode_base

    if provider != "azure" and not resolved_key:
        raise ValueError(f"{provider} API key is required")

    return {
        "provider": provider,
        "api_key": resolved_key,
        "base_url": resolved_base,
        "extra_headers": headers,
    }
