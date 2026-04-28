from __future__ import annotations

from functools import lru_cache

from app.config.settings import get_settings


@lru_cache(maxsize=1)
def get_openai_client():
    from openai import OpenAI

    settings = get_settings()
    if settings.openai_api_key:
        return OpenAI(api_key=settings.openai_api_key)
    return OpenAI()
