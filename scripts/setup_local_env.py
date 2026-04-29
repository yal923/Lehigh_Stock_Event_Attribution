"""Interactive local .env setup for first-time users."""
from __future__ import annotations

from getpass import getpass
from pathlib import Path
import secrets


REPO_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = REPO_ROOT / ".env"

AI_PROVIDERS = {
    "1": ("none", None),
    "2": ("openai", "OPENAI_API_KEY"),
    "3": ("anthropic", "ANTHROPIC_API_KEY"),
    "4": ("gemini", "GEMINI_API_KEY"),
}

DEFAULT_ENV = {
    "SECRET_KEY": "",
    "DATABASE_URL": "sqlite:///sa41.db",
    "ENABLE_GOOGLE_RSS": "true",
    "STOCKNEWS_API_KEY": "",
    "FINNHUB_API_KEY": "",
    "AI_PROVIDER": "openai",
    "AI_MODEL": "",
    "OPENAI_API_KEY": "",
    "ANTHROPIC_API_KEY": "",
    "GEMINI_API_KEY": "",
    "ENABLE_SIGNAL_EPISODE_V2": "true",
    "DEFAULT_EPISODE_HORIZON_N": "3",
    "DEFAULT_EPISODE_THRESHOLD": "0.05",
    "DEFAULT_EPISODE_MERGE_GAP_DAYS": "2",
    "DEFAULT_EPISODE_LOOKBACK_DAYS": "365",
}


def _read_existing_env() -> dict[str, str]:
    if not ENV_PATH.exists():
        return {}

    values: dict[str, str] = {}
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip().strip('"').strip("'")
    return values


def _required_secret(prompt: str) -> str:
    while True:
        value = getpass(prompt).strip()
        if value:
            return value
        print("This value is required. Please paste the key and press Enter.")


def _choose_ai_provider() -> tuple[str, str | None, str]:
    print("")
    print("AI Explanation is optional. You can skip it now and add a key later.")
    print("1. Skip AI setup for now")
    print("2. OpenAI")
    print("3. Anthropic")
    print("4. Gemini")

    while True:
        choice = input("Choose an option [1-4]: ").strip() or "1"
        if choice in AI_PROVIDERS:
            provider, key_name = AI_PROVIDERS[choice]
            break
        print("Please type 1, 2, 3, or 4.")

    if provider == "none" or key_name is None:
        return "openai", None, ""

    key_value = _required_secret(f"Paste your {key_name}: ")
    return provider, key_name, key_value


def _write_env(values: dict[str, str]) -> None:
    lines = [
        "# Flask / app",
        f"SECRET_KEY={values['SECRET_KEY']}",
        f"DATABASE_URL={values['DATABASE_URL']}",
        "",
        "# Homepage market cards require Finnhub.",
        f"FINNHUB_API_KEY={values['FINNHUB_API_KEY']}",
        "",
        "# Optional news provider for pipeline runs.",
        f"ENABLE_GOOGLE_RSS={values['ENABLE_GOOGLE_RSS']}",
        f"STOCKNEWS_API_KEY={values['STOCKNEWS_API_KEY']}",
        "",
        "# AI Explanation provider. Optional until you use AI Explanation.",
        "# Supported providers: openai, anthropic, gemini.",
        f"AI_PROVIDER={values['AI_PROVIDER']}",
        f"AI_MODEL={values['AI_MODEL']}",
        f"OPENAI_API_KEY={values['OPENAI_API_KEY']}",
        f"ANTHROPIC_API_KEY={values['ANTHROPIC_API_KEY']}",
        f"GEMINI_API_KEY={values['GEMINI_API_KEY']}",
        "",
        "# Episode defaults",
        f"ENABLE_SIGNAL_EPISODE_V2={values['ENABLE_SIGNAL_EPISODE_V2']}",
        f"DEFAULT_EPISODE_HORIZON_N={values['DEFAULT_EPISODE_HORIZON_N']}",
        f"DEFAULT_EPISODE_THRESHOLD={values['DEFAULT_EPISODE_THRESHOLD']}",
        f"DEFAULT_EPISODE_MERGE_GAP_DAYS={values['DEFAULT_EPISODE_MERGE_GAP_DAYS']}",
        f"DEFAULT_EPISODE_LOOKBACK_DAYS={values['DEFAULT_EPISODE_LOOKBACK_DAYS']}",
        "",
    ]
    ENV_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("Stock Event Attribution local setup")
    print("This creates a private .env file for this computer.")
    print("Do not commit .env to GitHub.")
    print("")

    existing = _read_existing_env()
    if existing:
        answer = input("A .env file already exists. Update it? [y/N]: ").strip().lower()
        if answer not in {"y", "yes"}:
            print("No changes made.")
            return

    values = dict(DEFAULT_ENV)
    values.update(existing)
    if not values.get("SECRET_KEY") or values["SECRET_KEY"] == "change-me":
        values["SECRET_KEY"] = secrets.token_urlsafe(32)

    print("Finnhub is required for the homepage market cards.")
    values["FINNHUB_API_KEY"] = _required_secret("Paste your FINNHUB_API_KEY: ")

    provider, key_name, key_value = _choose_ai_provider()
    values["AI_PROVIDER"] = provider
    values["OPENAI_API_KEY"] = ""
    values["ANTHROPIC_API_KEY"] = ""
    values["GEMINI_API_KEY"] = ""
    if key_name:
        values[key_name] = key_value

    _write_env(values)
    print("")
    print(f"Done. Wrote {ENV_PATH}")
    print("Next step: ./.venv/bin/python run.py")


if __name__ == "__main__":
    main()
