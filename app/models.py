from __future__ import annotations

from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Index, UniqueConstraint
from sqlalchemy.sql import func

db = SQLAlchemy()


class Company(db.Model):
    """
    SP500 company master data. Populated by scripts/ingest_sp500.py.
    """
    __tablename__ = "companies"

    id = db.Column(db.Integer, primary_key=True)
    ticker = db.Column(db.String(16), unique=True, index=True, nullable=False)
    name = db.Column(db.String(128), nullable=False)
    gics_sector = db.Column(db.String(128), nullable=True)
    gics_sub_industry = db.Column(db.String(256), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
    updated_at = db.Column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )

    episodes = db.relationship("Episode", back_populates="company", lazy="dynamic")

    def __repr__(self) -> str:
        return f"<Company {self.ticker}>"


class Episode(db.Model):
    """
    A significant price-movement episode detected from yfinance data via
    the signal-v2 detection path. One company can have many episodes.
    """
    __tablename__ = "episodes"

    id = db.Column(db.Integer, primary_key=True)

    company_id = db.Column(
        db.Integer,
        db.ForeignKey("companies.id"),
        nullable=False,
        index=True,
    )

    # Detection parameters (encoded in mode string, stored explicitly for filtering)
    mode = db.Column(db.String(16), nullable=False)          # e.g. "v2h3g2"
    threshold = db.Column(db.Float, nullable=False)           # e.g. 0.05
    lookback_days = db.Column(db.Integer, nullable=False)
    horizon_n = db.Column(db.Integer, nullable=False)
    merge_gap_days = db.Column(db.Integer, nullable=False)

    # Episode window
    peak_date = db.Column(db.Date, nullable=False, index=True)
    window_start = db.Column(db.Date, nullable=False)
    window_end = db.Column(db.Date, nullable=False)

    # Move info
    pct_move = db.Column(db.Float, nullable=False)
    direction = db.Column(db.String(8), nullable=False)       # "up" / "down"
    signal_count = db.Column(db.Integer, nullable=False, default=1)

    created_at = db.Column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    company = db.relationship("Company", back_populates="episodes")

    __table_args__ = (
        UniqueConstraint(
            "company_id",
            "mode",
            "threshold",
            "lookback_days",
            "peak_date",
            name="uq_episode_company_mode_threshold_lookback_peak",
        ),
        Index("ix_episode_company_peak", "company_id", "peak_date"),
    )

    def __repr__(self) -> str:
        return (
            f"<Episode id={self.id} company={self.company_id} "
            f"{self.peak_date} {self.direction} {self.pct_move:.1f}%>"
        )


class EpisodeAiExplanationCache(db.Model):
    """
    DEC-203: cache for the parallel AI Explanation path (GPT-5.5 + web_search).
    One row per episode_id. Indefinite TTL — invalidated only by user-triggered
    recompute, which overwrites the row in place.
    """
    __tablename__ = "episode_ai_explanation_cache"

    id = db.Column(db.Integer, primary_key=True)

    episode_id = db.Column(
        db.Integer,
        db.ForeignKey("episodes.id"),
        nullable=False,
        unique=True,
        index=True,
    )

    model_name = db.Column(db.String(64), nullable=False)
    payload_json = db.Column(db.Text, nullable=False)
    cost_estimate_usd = db.Column(db.Float, nullable=True)
    raw_response_path = db.Column(db.String(512), nullable=True)
    created_at = db.Column(
        db.DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    episode = db.relationship("Episode")
