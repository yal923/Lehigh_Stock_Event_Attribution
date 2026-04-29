import os

from flask import Flask

from app.config.settings import get_settings
from app.models import db


def _ensure_startup_company_data() -> None:
    if os.environ.get("SA41_SKIP_AUTO_SP500_SEED") == "1":
        return

    try:
        from app.services.sp500_ingestion import ensure_sp500_companies

        result = ensure_sp500_companies(verbose=True)
        if not result.skipped:
            print(
                "[startup] S&P 500 company data ready "
                f"(inserted={result.inserted}, updated={result.updated})."
            )
    except Exception as exc:  # noqa: BLE001
        print(f"[startup] S&P 500 auto-ingest failed; server will still start: {exc}")


def create_app():
    app = Flask(__name__)
    from app.web.routes import bp
    app.register_blueprint(bp)

    settings = get_settings()
    app.config["SQLALCHEMY_DATABASE_URI"] = settings.database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = settings.secret_key
    if not settings.finnhub_api_key:
        print("[config] FINNHUB_API_KEY is required for homepage market cards.")

    db.init_app(app)

    # DEC-203: ensure new tables (e.g. episode_ai_explanation_cache) are
    # auto-created on first boot. Idempotent for existing tables.
    with app.app_context():
        db.create_all()
        try:
            from app.demo_seed import install_demo_seed

            install_demo_seed(app, verbose=True)
        except Exception as exc:  # noqa: BLE001
            print(f"[startup] Demo seed install failed; server will still start: {exc}")
        _ensure_startup_company_data()

    return app
