"""
Provider adapter for DEC-203 AI Explanation calls.

The prompt builder and cache live in `ai_explanation.py`; this module owns only
the vendor-specific API call. It intentionally uses raw REST for non-OpenAI
providers so sharing the project does not require extra SDK dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import requests

from app.config.settings import get_settings
from app.infrastructure.llm import get_openai_client


OPENAI_DEFAULT_MODEL = "gpt-5.5"
ANTHROPIC_DEFAULT_MODEL = "claude-sonnet-4-20250514"
GEMINI_DEFAULT_MODEL = "gemini-2.5-flash"

AI_PROVIDER_CATALOG = {
    "openai": {
        "label": "OpenAI",
        "requires_key": "OPENAI_API_KEY",
        "default_model": OPENAI_DEFAULT_MODEL,
        "models": [
            {"id": "gpt-5.5", "label": "GPT-5.5"},
            {"id": "gpt-5", "label": "GPT-5"},
        ],
    },
    "anthropic": {
        "label": "Anthropic",
        "requires_key": "ANTHROPIC_API_KEY",
        "default_model": ANTHROPIC_DEFAULT_MODEL,
        "models": [
            {"id": "claude-sonnet-4-20250514", "label": "Claude Sonnet 4"},
            {"id": "claude-opus-4-1-20250805", "label": "Claude Opus 4.1"},
        ],
    },
    "gemini": {
        "label": "Gemini",
        "requires_key": "GEMINI_API_KEY",
        "default_model": GEMINI_DEFAULT_MODEL,
        "models": [
            {"id": "gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
            {"id": "gemini-2.5-pro", "label": "Gemini 2.5 Pro"},
            {"id": "gemini-3-flash-preview", "label": "Gemini 3 Flash Preview"},
        ],
    },
}

_ANTHROPIC_MESSAGES_URL = "https://api.anthropic.com/v1/messages"
_GEMINI_GENERATE_URL = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

# OpenAI GPT-5.5 cost estimate used only for the UI's rough recompute warning.
_OPENAI_PRICE_INPUT_PER_1K = 5.0 / 1000.0
_OPENAI_PRICE_OUTPUT_PER_1K = 30.0 / 1000.0
_OPENAI_PRICE_PER_WEB_SEARCH = 0.01


@dataclass(frozen=True)
class AiProviderResult:
    provider_name: str
    model_name: str
    output_text: str
    raw_response: Any
    cost_estimate_usd: float | None = None


def resolve_ai_provider_config(
    provider_name: str | None = None,
    model_name: str | None = None,
) -> tuple[str, str]:
    """Resolve configured provider/model from explicit args or env settings."""
    settings = get_settings()
    provider = (provider_name or settings.ai_provider or "openai").strip().lower()
    if provider not in {"openai", "anthropic", "gemini"}:
        raise ValueError(
            f"Unsupported AI_PROVIDER={provider!r}; expected openai, anthropic, or gemini"
        )

    model = (model_name or settings.ai_model or "").strip()
    if not model:
        model = str(AI_PROVIDER_CATALOG[provider]["default_model"])
    return provider, model


def run_ai_provider(
    prompt: str,
    *,
    provider_name: str | None = None,
    model_name: str | None = None,
) -> AiProviderResult:
    provider, model = resolve_ai_provider_config(provider_name, model_name)
    if provider == "openai":
        return _run_openai(prompt, model)
    if provider == "anthropic":
        return _run_anthropic(prompt, model)
    if provider == "gemini":
        return _run_gemini(prompt, model)
    raise AssertionError(f"unhandled provider {provider!r}")


def _run_openai(prompt: str, model_name: str) -> AiProviderResult:
    client = get_openai_client()
    response = client.responses.create(
        model=model_name,
        input=prompt,
        tools=[{"type": "web_search"}],
    )
    return AiProviderResult(
        provider_name="openai",
        model_name=model_name,
        output_text=(getattr(response, "output_text", None) or "").strip(),
        raw_response=response,
        cost_estimate_usd=_estimate_openai_cost(response),
    )


def _run_anthropic(prompt: str, model_name: str) -> AiProviderResult:
    settings = get_settings()
    if not settings.anthropic_api_key:
        raise RuntimeError("ANTHROPIC_API_KEY is required when AI_PROVIDER=anthropic")

    body = {
        "model": model_name,
        "max_tokens": 4096,
        "messages": [{"role": "user", "content": prompt}],
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": 5,
            }
        ],
    }
    response = requests.post(
        _ANTHROPIC_MESSAGES_URL,
        headers={
            "x-api-key": settings.anthropic_api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        json=body,
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return AiProviderResult(
        provider_name="anthropic",
        model_name=model_name,
        output_text=_extract_anthropic_text(payload),
        raw_response=payload,
        cost_estimate_usd=None,
    )


def _run_gemini(prompt: str, model_name: str) -> AiProviderResult:
    settings = get_settings()
    if not settings.gemini_api_key:
        raise RuntimeError("GEMINI_API_KEY is required when AI_PROVIDER=gemini")

    body = {
        "contents": [{"parts": [{"text": prompt}]}],
        "tools": [{"google_search": {}}],
    }
    response = requests.post(
        _GEMINI_GENERATE_URL.format(model=model_name),
        headers={
            "x-goog-api-key": settings.gemini_api_key,
            "content-type": "application/json",
        },
        json=body,
        timeout=180,
    )
    response.raise_for_status()
    payload = response.json()
    return AiProviderResult(
        provider_name="gemini",
        model_name=model_name,
        output_text=_extract_gemini_text(payload),
        raw_response=payload,
        cost_estimate_usd=None,
    )


def _extract_anthropic_text(payload: dict[str, Any]) -> str:
    pieces: list[str] = []
    for block in payload.get("content") or []:
        if not isinstance(block, dict):
            continue
        if block.get("type") == "text":
            text = block.get("text")
            if text:
                pieces.append(str(text))
    return "\n\n".join(pieces).strip()


def _extract_gemini_text(payload: dict[str, Any]) -> str:
    candidates = payload.get("candidates") or []
    if not candidates or not isinstance(candidates[0], dict):
        return ""
    content = candidates[0].get("content") or {}
    pieces: list[str] = []
    for part in content.get("parts") or []:
        if isinstance(part, dict) and part.get("text"):
            pieces.append(str(part["text"]))
    return "\n\n".join(pieces).strip()


def _estimate_openai_cost(response: Any) -> float:
    usage = getattr(response, "usage", None)
    input_tokens = 0
    output_tokens = 0
    if usage is not None:
        input_tokens = int(getattr(usage, "input_tokens", 0) or 0)
        output_tokens = int(getattr(usage, "output_tokens", 0) or 0)

    return (
        (input_tokens / 1000.0) * _OPENAI_PRICE_INPUT_PER_1K
        + (output_tokens / 1000.0) * _OPENAI_PRICE_OUTPUT_PER_1K
        + _count_openai_web_search_calls(response) * _OPENAI_PRICE_PER_WEB_SEARCH
    )


def _count_openai_web_search_calls(response: Any) -> int:
    count = 0
    output = getattr(response, "output", None) or []
    for item in output:
        item_type = getattr(item, "type", None) or (
            item.get("type") if isinstance(item, dict) else None
        )
        if item_type and "web_search" in str(item_type):
            count += 1
    return count
