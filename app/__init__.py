from flask import Flask

from app.config.settings import get_settings
from app.models import db


def create_app():
    app = Flask(__name__)
    from app.web.routes import bp
    app.register_blueprint(bp)

    settings = get_settings()
    app.config["SQLALCHEMY_DATABASE_URI"] = settings.database_url
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["SECRET_KEY"] = settings.secret_key

    db.init_app(app)

    # DEC-203: ensure new tables (e.g. episode_ai_explanation_cache) are
    # auto-created on first boot. Idempotent for existing tables.
    with app.app_context():
        db.create_all()

    return app
