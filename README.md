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

Follow these steps from the project folder after cloning the repo.

### 1. Create the local Python environment

```bash
python3.11 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
./.venv/bin/python -m pip install -r requirements.txt
```

Optional but recommended for the structured pipeline:

```bash
./.venv/bin/python -m spacy download en_core_web_sm
```

### 2. Create your private local settings file

Run the guided setup:

```bash
./.venv/bin/python scripts/setup_local_env.py
```

The setup script asks for:

1. **Finnhub API key**: required for the homepage market cards.
2. **AI provider**: optional. Skip this unless you want to use AI Explanation.
3. **AI API key**: required only if you chose an AI provider.

The script also creates the app `SECRET_KEY` automatically. It writes everything
to `.env`, which is private to your computer and ignored by Git. Never commit
or share `.env`.

### 3. Start the app

Run:

```bash
./.venv/bin/python run.py
```

Then open:

```text
http://127.0.0.1:5000/
```

On first startup, the app checks whether company master data exists. If the
local database is empty, it automatically downloads the S&P 500 company list
and stores it in the local SQLite database. You do not need to run
`scripts/ingest_sp500.py` manually for normal use.

## API Keys And Providers

### Required key

Finnhub is required for the homepage market cards:

```text
FINNHUB_API_KEY=
```

To get a Finnhub key:

1. Go to [Finnhub](https://finnhub.io/).
2. Create an account or sign in.
3. Open the Finnhub dashboard.
4. Copy your API key.
5. Paste it into the setup script when it asks for `FINNHUB_API_KEY`.

### Optional AI keys

AI Explanation requires one configured AI provider key. The rest of the app can
run without an AI key.

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

Use one provider at a time:

1. **OpenAI**: create or view an API key at
   [OpenAI API keys](https://platform.openai.com/api-keys).
2. **Anthropic**: create an API key from
   [Anthropic Console](https://console.anthropic.com/).
3. **Gemini**: create or view a Gemini API key at
   [Google AI Studio API keys](https://aistudio.google.com/apikey).

The browser UI lets users select the configured AI provider/model on episode
detail pages. The app also exposes `/api/config/capabilities`, which reports
which providers are available without exposing secrets.

RSS retrieval is enabled by default and does not require an API key.
`STOCKNEWS_API_KEY` is optional and only used for extra pipeline news retrieval.

## Data And Runtime State

Ignored local runtime files:

```text
.env
.venv/
instance/
app/_episode_cache/
app/_profile_cache/
```

The app creates database tables on startup. It also auto-populates company
master data from the S&P 500 list if the local database has no companies.

Manual refresh is still available for developers:

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
