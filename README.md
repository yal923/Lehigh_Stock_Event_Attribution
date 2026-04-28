# Stock Event Attribution

SA41 is a research terminal for decomposing notable stock-price episodes into
firm, industry, and market drivers. The current deliverable exposes two
parallel explanation views for the same episode:

- **AI Explanation**: web-grounded LLM explanation, cached after generation.
- **Pipeline Analysis**: the structured V2 attribution/news pipeline baseline.

This project is an interpretability and review tool. It is not a trading signal
or investment recommendation system.

## Current Deliverable

The fastest way to inspect the release demo is:

```bash
./.venv/bin/python run.py
```

Then open:

```text
http://127.0.0.1:5000/verification
```

That page shows the 10 test episodes used for presentation and paper review.
It works from the tracked demo cache bundle under:

```text
docs/demo_cache/2026_release/
```

The bundle contains the paid AI Explanation artifacts for the 10 release
episodes, including raw provider responses. The local runtime cache directory
`instance/` remains ignored and should not be pushed.

## Setup

Create a local virtual environment:

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt
```

Optional but recommended for the structured pipeline:

```bash
./.venv/bin/python -m spacy download en_core_web_sm
```

Create local config:

```bash
cp .env.example .env
```

Never commit `.env`. It is ignored because it may contain API keys.

Run the app:

```bash
./.venv/bin/python run.py
```

## API Keys And Providers

RSS retrieval is enabled by default and does not require an API key.

Optional news providers:

```text
STOCKNEWS_API_KEY=
FINNHUB_API_KEY=
```

AI Explanation requires at least one configured AI provider key:

```text
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
```

Supported AI providers:

```text
AI_PROVIDER=openai | anthropic | gemini
AI_MODEL=
```

The browser UI lets users select the configured AI provider/model on episode
detail pages. The app also exposes `/api/config/capabilities`, which reports
which providers are available without exposing secrets.

## Data And Runtime State

Ignored local runtime files:

```text
.env
.venv/
instance/
app/_episode_cache/
app/_profile_cache/
```

The app creates database tables on startup. To populate company master data for
full app use:

```bash
./.venv/bin/python scripts/ingest_sp500.py
```

Company pages generate signal-v2 episodes on request using yfinance. Full
pipeline and AI recompute flows may require network access and configured API
keys.

## Demo / Verification Artifacts

Tracked release evidence:

```text
docs/verification_event_match_2026.json
docs/demo_cache/2026_release/release_verification.json
docs/demo_cache/2026_release/release_verification.jsonl
docs/demo_cache/2026_release/ai_raw/
```

The `/verification` route prefers local `instance/verification/` artifacts. If
those are absent, it falls back to `docs/demo_cache/2026_release/`, so a fresh
clone can still inspect the 10 test episodes without rerunning paid AI calls.

To rebuild the tracked demo bundle from local paid cache:

```bash
./.venv/bin/python scripts/verification/build_demo_cache_bundle.py
./.venv/bin/python scripts/verification/evaluate_event_match.py \
  --source docs/demo_cache/2026_release/release_verification.json \
  --write
```

## Key Routes

```text
/                                      Home / company search
/verification                          10 test episode shortcut table
/company/<ticker>                      Company page and price chart
/company/<ticker>/episode/<id>         AI Explanation view
/company/<ticker>/episode/<id>/pipeline V2 Pipeline Analysis view
```

## Project Notes

The public repo keeps only the documentation needed for setup and demo review:
this README plus the tracked 10-episode demo cache and event-match artifact.
Local runtime state and API keys stay outside Git.

For cleanup/delivery context, see:

```text
docs/demo_cache/2026_release/
docs/verification_event_match_2026.json
```
