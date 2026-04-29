# Stock Event Attribution

Stock Event Attribution is a research review tool for understanding notable
stock-price episodes. It helps compare two explanation paths for the same move:

- **AI Explanation**: a cached, web-grounded AI narrative with sources.
- **Pipeline Analysis**: a structured attribution and news-evidence baseline.

The app is for interpretation and review. It is not a trading signal or
investment recommendation system.

## What You Can See

The repo includes a cached 10-episode demo set used for presentation review.
Those episodes can be opened from:

```text
http://127.0.0.1:5000/verification
```

The 10 demo pages include paid AI Explanation artifacts that are already cached
in the repo. You do not need an AI API key to view those cached demo pages. You
only need an AI API key if you want to generate or recompute AI explanations.

## Quick Start

Follow these steps from Terminal.

### 1. Clone The Repo

Using SSH:

```bash
git clone git@github.com:yal923/Lehigh_Stock_Event_Attribution.git
cd Lehigh_Stock_Event_Attribution
```

If SSH is not set up on your computer, use HTTPS instead:

```bash
git clone https://github.com/yal923/Lehigh_Stock_Event_Attribution.git
cd Lehigh_Stock_Event_Attribution
```

### 2. Create The Python Environment

```bash
python3.11 -m venv .venv
```

```bash
./.venv/bin/python -m pip install --upgrade pip setuptools wheel
```

```bash
./.venv/bin/python -m pip install -r requirements.txt
```

Optional, for fuller structured-pipeline NLP support:

```bash
./.venv/bin/python -m spacy download en_core_web_sm
```

### 3. Create Your Private Settings File

Run the guided setup script:

```bash
./.venv/bin/python scripts/setup_local_env.py
```

The script will ask for:

1. **Finnhub API key**: required for homepage logos and live market cards.
2. **AI provider**: optional. Choose skip unless you plan to generate or
   recompute AI explanations.
3. **AI API key**: required only if you chose an AI provider.

The script automatically creates a Flask `SECRET_KEY` and writes everything to
`.env`. Do not commit or share `.env`; it is private to your computer.

### 4. Run The Server

```bash
./.venv/bin/python run.py
```

Then open:

```text
http://127.0.0.1:5000/
```

To inspect the cached 10-episode demo set, open:

```text
http://127.0.0.1:5000/verification
```

On first startup, the app installs the tracked demo seed data into local runtime
state. If no company data exists after that, it falls back to ingesting the
S&P 500 company list automatically. You do not need to run
`scripts/ingest_sp500.py` for normal use.

## API Keys

### Required

Finnhub is required for homepage market cards:

```text
FINNHUB_API_KEY=
```

To get a Finnhub key:

1. Go to [Finnhub](https://finnhub.io/).
2. Create an account or sign in.
3. Open the Finnhub dashboard.
4. Copy your API key.
5. Paste it into the setup script when asked.

### Optional

AI provider keys are optional. They are only needed for generating or
recomputing AI Explanation results:

```text
OPENAI_API_KEY=
ANTHROPIC_API_KEY=
GEMINI_API_KEY=
```

Supported providers:

```text
AI_PROVIDER=openai | anthropic | gemini
AI_MODEL=
```

Provider key pages:

1. [OpenAI API keys](https://platform.openai.com/api-keys)
2. [Anthropic Console](https://console.anthropic.com/)
3. [Google AI Studio API keys](https://aistudio.google.com/apikey)

## Demo Data

Tracked demo artifacts live under:

```text
docs/demo_cache/2026_release/
```

This folder includes:

```text
demo_seed.sqlite                         Local seed database for demo episodes
pipeline_cache/episode_*.json            Cached Pipeline Analysis artifacts
ai_raw/episode_*_openai.json             Raw cached AI provider responses
release_verification.json                10-episode verification summary
release_verification.jsonl               Line-by-line verification log
```

At runtime, the app copies or installs these artifacts into ignored local
runtime locations. The tracked demo bundle stays unchanged.

## Runtime Files

These files are local-only and should not be committed:

```text
.env
.venv/
instance/
app/_episode_cache/
app/_profile_cache/
```

`instance/` stores the local SQLite database and cached homepage logo data.
`app/_episode_cache/` stores runtime pipeline cache files.

## Key Routes

```text
/                                       Homepage and company search
/verification                           Cached 10-episode demo table
/company/<ticker>                       Company page and price chart
/company/<ticker>/episode/<id>          AI Explanation view
/company/<ticker>/episode/<id>/pipeline Pipeline Analysis view
```

## Developer Notes

Manual S&P 500 refresh is still available:

```bash
./.venv/bin/python scripts/ingest_sp500.py
```

The public repo keeps setup instructions, demo artifacts, and review evidence.
Secrets and local runtime state stay outside Git.
