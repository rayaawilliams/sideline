# Sideline 🏟️

**In college it's hard to keep up.** Sideline is a college sports cheat sheet: ask about your teams and favourite players in plain English and get a short answer, with the stats and stories it came from.

> "Who leads college football in rushing touchdowns?"
> "How did Northwestern do last weekend?"
> "Who was the top goal scorer in women's lacrosse last season?"
> "How are my players doing?"

It covers college football and men's and women's lacrosse (lacrosse runs February to May, so out of season it shows last season's numbers).

## How it works

Sports questions come in two kinds, and each needs a different tool:

| Question | Tool | Why |
|---|---|---|
| Numbers: leaders, stats, scores | **SQL** on a stats database | Exact numbers need exact lookups. Searching text for "the most goals" often picks the wrong row |
| Stories: what happened, previews | **RAG**: vector search over news articles | Finds passages by meaning, not exact words |

For each question the language model makes two calls:

```
question
  1. PLAN    model sees the table schema, what data is loaded and your favourites,
             and writes a read-only SQL query and/or a news search phrase
  2. LOOK UP SQL runs on SQLite (read-only, SELECT only); news search runs on Chroma
             (if the SQL fails, the error goes back to the model once to fix it)
  3. ANSWER  model answers from those results only, citing [S] for stats and [1] for articles,
             or says it couldn't find it instead of guessing
```

Data comes from the free, open-source [NCAA API](https://github.com/henrygd/ncaa-api), which returns the data behind ncaa.com as JSON. A daily update pulls stat leaderboards, scores and news into the app. The API is unofficial and rate-limited (5 requests per second), so the app caches everything locally, spaces out requests, and keeps working from saved data if the API is down.

| Folder / file | What it does |
|---|---|
| `sources.json` | Which sports, stat leaderboards and feeds to pull |
| `sports/ncaa.py` | API client (rate limiting, retries) and parsers for stats, scoreboards and news |
| `sports/update.py` | Daily update: stats and games into SQLite, news into `data/docs/news/` |
| `sports/db.py` | Database schema and the read-only query guard |
| `sports/assistant.py` | Plan, look up, answer |
| `rag/` | Chunking, embeddings (`all-MiniLM-L6-v2`, runs locally), Chroma vector store, and one interface for Gemini, Claude and Ollama models |
| `eval/` | Generates ground-truth questions from the data and scores each model |
| `app.py` | Streamlit web app with a favourites list |

## Setup

Requires Python 3.11 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env               # paste in your API key(s)
set -a; source .env; set +a        # load the keys into your terminal
```

Models (use any or all):
- **Gemini:** API key from Google AI Studio (free tier available).
- **Claude:** API key from the Anthropic Console.
- **Ollama (free, runs on your laptop):** install from ollama.com, then `ollama pull llama3.2`.

Model names change. If a default in `rag/config.py` stops working, set `RAG_MODEL` to a current one.

## Choose your data

Edit `sources.json`. Football rushing touchdowns are filled in as an example. To add a leaderboard:
1. Go to `https://www.ncaa.com/stats/lacrosse-women/d1` (or `lacrosse-men/d1`, `football/fbs`).
2. Pick a stat from the dropdown, for example goals per game.
3. Copy the number at the end of the URL (`.../current/individual/NNN`) into `stats`, with a category name.

## Run it

```bash
python -m sports.update              # pull stats, scores and news (run daily)
python -m rag.ingest --reset         # index the news for search

python -m rag.ask "Who leads college football in rushing touchdowns?" --show-work
streamlit run app.py                 # the web app
```

`--show-work` prints the SQL query the model wrote and the news it searched, which is the best way to see why an answer came out the way it did. Every API response is also saved in `data/raw/` for debugging.

## Evaluate

```bash
python -m eval.make_questions                      # ground-truth questions from your database
python -m eval.evaluate --providers gemini claude ollama
```

`make_questions` writes questions whose answers come straight from the database ("Which player leads ... in rushing touchdowns?", "Who won when X played Y?"), plus questions the data can't answer. Add your own news and favourites questions by hand (see `eval/questions.example.jsonl`).

| Metric | Meaning |
|---|---|
| answer acc | Answerable questions answered with all the expected facts |
| refusal acc | Unanswerable questions where the model correctly said it couldn't find it |
| SQL ok | Stats queries that ran without errors (after the one allowed retry) |
| news hit | News questions where the right article was retrieved |
| p50 / p95 latency | Typical and slow-case time per question |
| $ / 1k q | Cost per 1,000 questions, once you add prices to `rag/config.py` |

## Deploy to Google Cloud Run

```bash
gcloud auth login
gcloud config set project YOUR_PROJECT_ID
gcloud services enable run.googleapis.com cloudbuild.googleapis.com secretmanager.googleapis.com artifactregistry.googleapis.com
printf "%s" "$GEMINI_API_KEY" | gcloud secrets create gemini-api-key --data-file=-

gcloud run deploy sideline --source . --region us-central1 --allow-unauthenticated \
  --memory 2Gi --set-secrets GEMINI_API_KEY=gemini-api-key:latest
```

On each start the container pulls fresh data and indexes the news, so restarting the service refreshes it. The Cloud Run service account needs the **Secret Manager Secret Accessor** role. Favourites saved in the deployed app reset when the service restarts.

## Tests

```bash
python -m pytest
```

Tests run offline with made-up sample API responses, a keyword-hashing embedder and a scripted fake model. They cover the parsers, the update, the SQL safety guard (including that the database is opened read-only), the retry on a failed query, news search, refusals, and the evaluation maths.
