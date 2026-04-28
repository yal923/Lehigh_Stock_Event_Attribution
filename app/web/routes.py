import json
from pathlib import Path

from flask import Blueprint, render_template, request, jsonify, abort, url_for, current_app
from sqlalchemy import case, func

from app.config.settings import get_settings
from app.domain.attribution import episode_is_attributable
from app.domain.attribution_router import (
    build_router_input_from_attribution_payload,
    route_attribution,
)
from app.domain.branch_analysis import build_multi_branch_analysis_plan

from app.services.company_service import (
    PRICE_HISTORY_CACHE_TTL_SECONDS,
    get_home_companies,
    get_company_by_ticker,
    get_company_prices,
    get_episode_for_company,
    parse_chart_lookback,
)
from app.services.episode_service import (
    build_episode_decision_summary,
    build_episode_metrics,
    ensure_and_get_company_episodes,
    parse_episode_params,
)
from app.services.home_market_service import QUOTE_TTL_SECONDS, get_home_market_cards
from app.services.pipeline_service import run_episode_pipeline, run_batch_episode_pipeline
from app.services.pipeline_service import run_episode_secondary_branch_analysis
from app.services.ai_explanation import (
    cache_age_seconds,
    get_cached_ai_explanation,
    get_or_compute_ai_explanation,
)
from app.services.ai_provider import AI_PROVIDER_CATALOG
from app.models import db, Company


bp = Blueprint("routes", __name__)


def _safe_episode_metrics(company_id: int, episode) -> dict:
    """Build episode metrics; fall back to stored episode fields if yfinance fails."""
    try:
        return build_episode_metrics(company_id, episode)
    except Exception:
        return {
            "start": episode.window_start.isoformat() if episode.window_start else None,
            "end": episode.window_end.isoformat() if episode.window_end else None,
            "peak": episode.peak_date.isoformat() if episode.peak_date else None,
            "direction": episode.direction,
            "peak_pct": episode.pct_move,
            "cumulative_pct": None,
            "signal_count": episode.signal_count or 0,
            "signal_anchors": [],
            "daily_pct": [],
        }


def _repo_root() -> Path:
    return Path(current_app.root_path).parent


@bp.route("/")
def home():
    companies = get_home_companies()
    return render_template("home.html", companies=companies)


def _load_release_verification_file(path: Path, *, artifact_source: str) -> dict:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # noqa: BLE001
        return {
            "available": False,
            "artifact_name": path.name,
            "artifact_path": str(path),
            "artifact_source": artifact_source,
            "load_error": str(exc),
            "results": [],
        }

    payload["available"] = True
    payload["artifact_name"] = path.name
    payload["artifact_path"] = str(path)
    payload["artifact_source"] = artifact_source
    payload["results"] = payload.get("results") or []
    return payload


def _load_latest_release_verification() -> dict:
    verification_dir = Path(current_app.instance_path) / "verification"
    files = sorted(
        verification_dir.glob("release_verification_*.json"),
        key=lambda p: p.stat().st_mtime,
    )
    if files:
        return _load_release_verification_file(files[-1], artifact_source="instance")

    demo_path = _repo_root() / "docs" / "demo_cache" / "2026_release" / "release_verification.json"
    if demo_path.exists():
        return _load_release_verification_file(demo_path, artifact_source="demo_cache")

    return {
        "available": False,
        "artifact_name": None,
        "artifact_path": None,
        "artifact_source": None,
        "results": [],
    }


def _load_event_match_reviews() -> dict[str, dict]:
    path = _repo_root() / "docs" / "verification_event_match_2026.json"
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    reviews: dict[str, dict] = {}
    for item in payload.get("results") or []:
        key = item.get("key") or f"{item.get('ticker')}#{item.get('episode_id')}"
        if key:
            reviews[key] = item
    return reviews


@bp.route("/verification")
def verification_review():
    summary = _load_latest_release_verification()
    reviews = _load_event_match_reviews()
    for item in summary.get("results") or []:
        key = f"{item.get('ticker')}#{item.get('episode_id')}"
        item["event_review"] = reviews.get(key, {})
    return render_template(
        "verification.html",
        summary=summary,
        results=summary.get("results") or [],
    )


@bp.route("/api/config/capabilities")
def api_config_capabilities():
    """Expose configured provider availability without exposing secrets."""
    settings = get_settings()

    ai_key_available = {
        "openai": bool(settings.openai_api_key),
        "anthropic": bool(settings.anthropic_api_key),
        "gemini": bool(settings.gemini_api_key),
    }
    ai_providers = []
    for provider_id, meta in AI_PROVIDER_CATALOG.items():
        available = bool(ai_key_available.get(provider_id))
        ai_providers.append({
            "id": provider_id,
            "label": meta["label"],
            "available": available,
            "requires_key": meta["requires_key"],
            "missing_key": None if available else meta["requires_key"],
            "default_model": meta["default_model"],
            "models": meta["models"],
        })

    return jsonify({
        "news_providers": [
            {
                "id": "stocknewsapi",
                "label": "StockNewsAPI",
                "available": bool(settings.stocknews_api_key),
                "requires_key": "STOCKNEWS_API_KEY",
                "missing_key": None if settings.stocknews_api_key else "STOCKNEWS_API_KEY",
            },
            {
                "id": "finnhub",
                "label": "Finnhub",
                "available": bool(settings.finnhub_api_key),
                "requires_key": "FINNHUB_API_KEY",
                "missing_key": None if settings.finnhub_api_key else "FINNHUB_API_KEY",
            },
            {
                "id": "google_rss",
                "label": "Google RSS",
                "available": bool(settings.enable_google_rss),
                "requires_key": None,
                "missing_key": None,
            },
        ],
        "home_enrichment": {
            "quotes": {
                "available": bool(settings.finnhub_api_key),
                "source": "Finnhub",
                "requires_key": "FINNHUB_API_KEY",
                "ttl_seconds": QUOTE_TTL_SECONDS,
            },
            "logos": {
                "available": bool(settings.finnhub_api_key),
                "source": "Finnhub company_profile2",
                "requires_key": "FINNHUB_API_KEY",
                "cache": "instance/company_assets/company_assets.json",
            },
        },
        "ai_providers": ai_providers,
        "defaults": {
            "ai_provider": settings.ai_provider,
            "ai_model": settings.ai_model,
        },
    })


@bp.route("/company/<ticker>")
def company_detail(ticker: str):
    company = get_company_by_ticker(ticker)
    if company is None:
        abort(404)

    try:
        params = parse_episode_params(request.args)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    _mode, episodes = ensure_and_get_company_episodes(company, params)

    episodes_json = [
        {
            "id": e.id,
            "start": e.window_start.isoformat() if e.window_start else None,
            "end": e.window_end.isoformat() if e.window_end else None,
            "peak_date": e.peak_date.isoformat() if e.peak_date else None,
            "direction": e.direction,
            "pct_move": e.pct_move,
        }
        for e in episodes
    ]

    episode_metrics_map = {e.id: _safe_episode_metrics(company.id, e) for e in episodes}

    return render_template(
        "company_detail.html",
        company=company,
        episodes=episodes,
        episodes_json=episodes_json,
        episode_metrics_map=episode_metrics_map,
        episode_params={
            "horizon_n": params.horizon_n,
            "threshold": params.threshold,
            "merge_gap_days": params.merge_gap_days,
            "lookback_days": params.lookback_days,
        },
    )


@bp.route("/api/company/<ticker>/prices")
def api_company_prices(ticker: str):
    company = get_company_by_ticker(ticker)
    if company is None:
        return jsonify({"error": "unknown ticker"}), 404

    chart_range = request.args.get("range") or request.args.get("lookback_days")
    lookback_days = parse_chart_lookback(chart_range)
    rows = get_company_prices(company, lookback_days=lookback_days)
    payload = {
        "lookback_days": lookback_days,
        "cache_ttl_seconds": PRICE_HISTORY_CACHE_TTL_SECONDS,
        "dates":  [r.date.isoformat() for r in rows],
        "opens":  [r.open for r in rows],
        "highs":  [r.high for r in rows],
        "lows":   [r.low for r in rows],
        "closes": [r.close for r in rows],
    }
    return jsonify(payload)


@bp.route("/api/company/<ticker>/episodes")
def api_company_episodes(ticker: str):
    company = get_company_by_ticker(ticker)
    if company is None:
        return jsonify([])

    try:
        params = parse_episode_params(request.args)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    _mode, episodes = ensure_and_get_company_episodes(company, params)

    payload = []
    for e in episodes or []:
        payload.append({
            "id": e.id,
            "start": e.window_start.isoformat() if e.window_start else None,
            "end": e.window_end.isoformat() if e.window_end else None,
            "direction": e.direction,
            "pct_move": float(e.pct_move) if e.pct_move is not None else None,
            "decision": build_episode_decision_summary(e),
        })

    return jsonify(payload)


@bp.route("/api/companies/autocomplete")
def api_company_autocomplete():
    query = (request.args.get("q") or "").strip()
    if len(query) < 1:
        return jsonify({"query": query, "results": []})

    q = query.lower()
    score = case(
        (func.lower(Company.ticker).like(f"{q}%"), 0),
        (func.lower(Company.name).like(f"{q}%"), 1),
        (func.lower(Company.name).like(f"%{q}%"), 2),
        else_=3,
    )

    matches = (
        Company.query
        .filter(
            (func.lower(Company.ticker).like(f"%{q}%")) |
            (func.lower(Company.name).like(f"%{q}%"))
        )
        .order_by(score, Company.ticker.asc())
        .limit(12)
        .all()
    )

    results = [
        {
            "ticker": c.ticker,
            "name": c.name,
            "url": url_for("routes.company_detail", ticker=c.ticker),
        }
        for c in matches
    ]

    return jsonify({"query": query, "results": results})


@bp.route("/api/home/market_cards")
def api_home_market_cards():
    companies = get_home_companies()
    return jsonify(get_home_market_cards(companies))


# DEC-203: route topology
#   /company/<ticker>/episode/<id>           → AI Explanation view (default)
#   /company/<ticker>/episode/<id>/pipeline  → V2 Pipeline Analysis view
# Both render the same `episode_detail.html` template with `explanation_source`
# switching the §3 Qual + §4 branch-card data binding.

@bp.route("/company/<ticker>/episode/<int:episode_id>")
def episode_analysis(ticker, episode_id):
    """AI Explanation view (DEC-203 default)."""
    company = get_company_by_ticker(ticker)
    if company is None:
        abort(404)

    episode = get_episode_for_company(company, episode_id)
    if episode is None:
        abort(404)
    if not episode_is_attributable(episode):
        abort(404)

    # Always run the V2 pipeline first (cached) to populate the shared
    # quantitative half (§1 Hero, §2 Snapshot, §3 LEFT, §4 card framework).
    pipeline_result = run_episode_pipeline(episode, force_recompute=False)
    episode_metrics = _safe_episode_metrics(company.id, episode)
    branch_plan = _build_branch_plan_from_result(pipeline_result)

    # AI Explanation: cache-only on page render (do not auto-trigger paid
    # API calls during route handling). If absent, the template surfaces
    # an empty state + a "Generate AI Explanation" CTA → /loading.
    ai_payload = get_cached_ai_explanation(episode_id)

    return render_template(
        "episode_detail.html",
        company=company,
        episode=episode,
        episode_metrics=episode_metrics,
        analysis=pipeline_result.get("analysis"),
        branch_plan=branch_plan,
        raw=pipeline_result,
        explanation_source="ai",
        ai_explanation=ai_payload,
        ai_cache_age_seconds=cache_age_seconds(episode_id),
    )


@bp.route("/company/<ticker>/episode/<int:episode_id>/pipeline")
def episode_analysis_pipeline(ticker, episode_id):
    """V2 Pipeline Analysis view (DEC-203 — was the previous default)."""
    company = get_company_by_ticker(ticker)
    if company is None:
        abort(404)

    episode = get_episode_for_company(company, episode_id)
    if episode is None:
        abort(404)
    if not episode_is_attributable(episode):
        abort(404)

    result = run_episode_pipeline(episode, force_recompute=False)
    episode_metrics = _safe_episode_metrics(company.id, episode)
    branch_plan = _build_branch_plan_from_result(result)
    return render_template(
        "episode_detail.html",
        company=company,
        episode=episode,
        episode_metrics=episode_metrics,
        analysis=result.get("analysis"),
        branch_plan=branch_plan,
        raw=result,
        explanation_source="pipeline",
    )


@bp.route("/company/<ticker>/episode/<int:episode_id>/loading")
def episode_analysis_loading(ticker, episode_id):
    """Loading page for either path. `source=ai|pipeline` query arg picks the
    backend run + the post-completion redirect target."""
    company = get_company_by_ticker(ticker)
    if company is None:
        abort(404)

    episode = get_episode_for_company(company, episode_id)
    if episode is None:
        abort(404)
    if not episode_is_attributable(episode):
        abort(404)

    force_recompute = request.args.get("force_recompute", "").strip().lower() in {
        "1", "true", "yes",
    }
    # `source` = which backend to run. `redirect_to` = which view to land
    # on after the run completes. They differ on first-visit: clicking an
    # episode from company_detail runs V2 (free, builds the quant skeleton)
    # but lands on the AI view (default landing per DEC-203).
    source = (request.args.get("source") or "pipeline").strip().lower()
    if source not in {"ai", "pipeline"}:
        source = "pipeline"
    ai_provider = (request.args.get("provider") or "").strip()
    ai_model = (request.args.get("model") or "").strip()
    redirect_to = (request.args.get("redirect_to") or source).strip().lower()
    if redirect_to not in {"ai", "pipeline"}:
        redirect_to = source

    if source == "ai":
        run_url = url_for("routes.run_ai_explanation_api", ticker=ticker, episode_id=episode_id)
    else:
        run_url = url_for("routes.run_episode_pipeline_api", ticker=ticker, episode_id=episode_id)

    if redirect_to == "ai":
        redirect_url = url_for("routes.episode_analysis", ticker=ticker, episode_id=episode_id)
    else:
        redirect_url = url_for("routes.episode_analysis_pipeline", ticker=ticker, episode_id=episode_id)

    return render_template(
        "episode_loading.html",
        company=company,
        episode=episode,
        run_url=run_url,
        redirect_url=redirect_url,
        force_recompute=force_recompute,
        explanation_source=source,
        ai_provider=ai_provider,
        ai_model=ai_model,
    )


@bp.route("/api/company/<ticker>/episode/<int:episode_id>/run", methods=["POST"])
def run_episode_pipeline_api(ticker, episode_id):
    company = get_company_by_ticker(ticker)
    if company is None:
        return jsonify({"ok": False, "error": "unknown ticker"}), 404

    episode = get_episode_for_company(company, episode_id)
    if episode is None:
        return jsonify({"ok": False, "error": "episode not found"}), 404
    if not episode_is_attributable(episode):
        return jsonify({"ok": False, "error": "episode company metadata missing or unsupported"}), 422

    payload = request.get_json(silent=True) or {}
    force_recompute = bool(payload.get("force_recompute", False))

    result = run_episode_pipeline(episode, force_recompute=force_recompute)
    analysis = result.get("analysis") or {}
    attribution = analysis.get("attribution_summary") or result.get("attribution") or {}
    decision = result.get("decision") or {}

    return jsonify({
        "ok": True,
        "episode_id": episode_id,
        "force_recompute": force_recompute,
        "analysis": analysis,
        "attribution": attribution,
        "decision": decision,
    })


@bp.route("/api/company/<ticker>/episode/<int:episode_id>/ai_explanation/run", methods=["POST"])
def run_ai_explanation_api(ticker, episode_id):
    """Trigger the configured AI provider + web-search call for this episode.

    Always reads attribution from the V2 pipeline cache first (free) so the
    AI prompt has factor shares to anchor on. The AI call itself is paid;
    `force_recompute=True` skips the cache."""
    company = get_company_by_ticker(ticker)
    if company is None:
        return jsonify({"ok": False, "error": "unknown ticker"}), 404

    episode = get_episode_for_company(company, episode_id)
    if episode is None:
        return jsonify({"ok": False, "error": "episode not found"}), 404
    if not episode_is_attributable(episode):
        return jsonify({"ok": False, "error": "episode company metadata missing or unsupported"}), 422

    payload = request.get_json(silent=True) or {}
    force_recompute = bool(payload.get("force_recompute", False))

    pipeline_result = run_episode_pipeline(episode, force_recompute=False)
    attribution = (
        (pipeline_result.get("analysis") or {}).get("attribution_summary")
        or pipeline_result.get("attribution")
        or {}
    )
    episode_metrics = _safe_episode_metrics(company.id, episode)

    try:
        ai_payload = get_or_compute_ai_explanation(
            episode,
            attribution_payload=attribution,
            episode_metrics=episode_metrics,
            force_recompute=force_recompute,
            provider_name=payload.get("provider_name") or payload.get("provider"),
            model_name=payload.get("model_name") or payload.get("model"),
        )
    except Exception as exc:  # noqa: BLE001
        return jsonify({"ok": False, "error": str(exc)}), 500

    return jsonify({
        "ok": True,
        "episode_id": episode_id,
        "force_recompute": force_recompute,
        "ai_explanation": ai_payload,
    })


@bp.route("/api/pipeline/run_batch", methods=["POST"])
def run_batch_pipeline_api():
    payload = request.get_json(silent=True) or {}

    try:
        limit = int(payload.get("limit", 20))
    except Exception:
        return jsonify({"ok": False, "error": "invalid limit"}), 400

    if limit <= 0:
        return jsonify({"ok": False, "error": "limit must be positive"}), 400

    force_recompute = bool(payload.get("force_recompute", False))
    result = run_batch_episode_pipeline(
        limit=min(limit, 200),
        force_recompute=force_recompute,
    )
    return jsonify({"ok": True, **result})


@bp.route("/api/company/<ticker>/episode/<int:episode_id>/analysis")
def api_episode_analysis(ticker, episode_id):
    company = get_company_by_ticker(ticker)
    if company is None:
        return jsonify({"error": "unknown ticker"}), 404

    episode = get_episode_for_company(company, episode_id)
    if episode is None:
        return jsonify({"error": "episode not found"}), 404
    if not episode_is_attributable(episode):
        return jsonify({"error": "episode company metadata missing or unsupported"}), 422

    result = run_episode_pipeline(episode, force_recompute=False)
    analysis = result.get("analysis") or {}
    return jsonify({
        "episode_id": episode_id,
        "analysis": analysis,
        "attribution": result.get("attribution") or analysis.get("attribution_summary") or {},
        "decision": result.get("decision") or {},
    })


@bp.route("/api/company/<ticker>/episode/<int:episode_id>/analysis_branch")
def api_episode_analysis_branch(ticker, episode_id):
    company = get_company_by_ticker(ticker)
    if company is None:
        return jsonify({"error": "unknown ticker"}), 404

    episode = get_episode_for_company(company, episode_id)
    if episode is None:
        return jsonify({"error": "episode not found"}), 404
    if not episode_is_attributable(episode):
        return jsonify({"error": "episode company metadata missing or unsupported"}), 422

    branch_mode = (request.args.get("mode") or "").strip().lower()
    if branch_mode not in {"market", "industry", "firm"}:
        return jsonify({"error": "invalid branch mode"}), 400

    try:
        result = run_episode_secondary_branch_analysis(
            episode,
            branch_mode=branch_mode,
            force_recompute=False,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    analysis = result.get("analysis") or {}
    return jsonify({
        "episode_id": episode_id,
        "branch_mode": branch_mode,
        "analysis": analysis,
        "attribution": result.get("attribution") or analysis.get("attribution_summary") or {},
        "decision": result.get("decision") or {},
        "multi_branch_plan": result.get("multi_branch_plan") or {},
    })


@bp.app_errorhandler(404)
def handle_404(error):
    return render_template("404.html"), 404


@bp.app_errorhandler(500)
def handle_500(error):
    return render_template("500.html"), 500


def _build_branch_plan_from_result(result: dict) -> dict:
    if result.get("branch_plan"):
        return result["branch_plan"]
    attribution = result.get("attribution") or result.get("analysis", {}).get("attribution_summary") or {}
    router_result = route_attribution(
        build_router_input_from_attribution_payload(attribution)
    )
    plan = build_multi_branch_analysis_plan(router_result)
    return {
        "routing_mode": plan.routing_mode,
        "primary_mode": plan.primary_mode,
        "default_modes": list(plan.default_modes),
        "lazy_modes": list(plan.lazy_modes),
    }
