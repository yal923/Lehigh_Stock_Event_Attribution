from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os


_ENV_FILE = ".env"


def _load_dotenv(path: Path) -> None:
    if not path.exists() or not path.is_file():
        return

    try:
        for raw_line in path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
    except Exception as exc:  # noqa: BLE001
        print(f"[config] Failed to load .env: {exc}")


def _default_env_path() -> Path:
    return Path(__file__).resolve().parents[2] / _ENV_FILE


@dataclass(frozen=True)
class Settings:
    database_url: str
    finnhub_api_key: str | None
    stocknews_api_key: str | None
    stocknews_base_url: str
    enable_google_rss: bool
    news_budget_ratio: float
    stocknews_remaining_quota: int | None
    finnhub_remaining_quota: int | None
    openai_api_key: str | None
    anthropic_api_key: str | None
    gemini_api_key: str | None
    ai_provider: str
    ai_model: str | None
    secret_key: str | None
    enable_signal_episode_v2: bool
    default_episode_horizon_n: int
    default_episode_threshold: float
    default_episode_merge_gap_days: int
    default_episode_lookback_days: int


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except Exception:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_optional_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv(_default_env_path())

    return Settings(
        database_url=os.getenv("DATABASE_URL", "sqlite:///sa41.db"),
        finnhub_api_key=os.getenv("FINNHUB_API_KEY"),
        stocknews_api_key=os.getenv("STOCKNEWS_API_KEY"),
        stocknews_base_url=os.getenv("STOCKNEWS_BASE_URL", "https://stocknewsapi.com/api/v1"),
        enable_google_rss=_env_bool("ENABLE_GOOGLE_RSS", True),
        news_budget_ratio=_env_float("NEWS_BUDGET_RATIO", 0.5),
        stocknews_remaining_quota=_env_optional_int("STOCKNEWS_REMAINING_QUOTA"),
        finnhub_remaining_quota=_env_optional_int("FINNHUB_REMAINING_QUOTA"),
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY"),
        gemini_api_key=os.getenv("GEMINI_API_KEY"),
        ai_provider=os.getenv("AI_PROVIDER", "openai").strip().lower(),
        ai_model=(os.getenv("AI_MODEL") or "").strip() or None,
        secret_key=os.getenv("SECRET_KEY"),
        enable_signal_episode_v2=_env_bool("ENABLE_SIGNAL_EPISODE_V2", True),
        default_episode_horizon_n=_env_int("DEFAULT_EPISODE_HORIZON_N", 3),
        default_episode_threshold=_env_float("DEFAULT_EPISODE_THRESHOLD", 0.05),
        default_episode_merge_gap_days=_env_int("DEFAULT_EPISODE_MERGE_GAP_DAYS", 2),
        default_episode_lookback_days=_env_int("DEFAULT_EPISODE_LOOKBACK_DAYS", 365),
    )
