"""All settings in one place. Environment variables override the defaults."""
import os
from dataclasses import dataclass, field, replace

# Default model per provider. Model names change often: check each provider's docs
# and override with RAG_MODEL if one of these has been retired.
DEFAULT_MODELS = {
    "gemini": "gemini-3.8-flash",
    "claude": "claude-haiku-5-5",
    "ollama": "llama3.2",
}


@dataclass(frozen=True)
class Settings:
    # Sports data
    sources_file: str = "sources.json"  # which sports, stat categories and feeds to pull
    sports_db: str = "data/sports.db"  # SQLite database of stat leaders and games
    follows_file: str = "follows.json"  # the user's favourite players and teams
    ncaa_api_base: str = "https://ncaa-api.henrygd.me"
    raw_dir: str = "data/raw"  # copies of every API response, for debugging

    # Documents for search (news is saved here by the updater)
    docs_dir: str = "data/docs"
    db_dir: str = ".chroma"
    collection: str = "sideline"

    # Chunking: documents are split into pieces of about this many characters,
    # overlapping so a sentence split across a boundary still appears whole in one chunk.
    chunk_chars: int = 1200
    chunk_overlap: int = 200

    # Retrieval: how many news chunks are given to the model as context.
    top_k: int = 4
    # Most stat rows shown to the model from one query.
    max_sql_rows: int = 25

    # "sentence-transformers" runs locally and needs no API key.
    # "hashing" is a crude keyword embedder for tests and offline smoke runs.
    embedder: str = "sentence-transformers"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    provider: str = "gemini"
    model: str = ""  # blank means DEFAULT_MODELS[provider]
    max_output_tokens: int = 500

    # Optional prices in US dollars per million tokens, as {"provider": (input, output)}.
    # Left empty on purpose: fill in current prices from each provider's pricing page.
    prices_per_million: dict = field(default_factory=dict)

    @property
    def model_name(self) -> str:
        return self.model or DEFAULT_MODELS[self.provider]


def load_settings(**overrides) -> Settings:
    env = {
        "docs_dir": os.getenv("RAG_DOCS_DIR"),
        "db_dir": os.getenv("RAG_DB_DIR"),
        "sports_db": os.getenv("SPORTS_DB"),
        "ncaa_api_base": os.getenv("NCAA_API_BASE"),
        "embedder": os.getenv("RAG_EMBEDDER"),
        "provider": os.getenv("RAG_PROVIDER"),
        "model": os.getenv("RAG_MODEL"),
        "top_k": int(os.environ["RAG_TOP_K"]) if os.getenv("RAG_TOP_K") else None,
    }
    values = {k: v for k, v in env.items() if v is not None}
    values.update({k: v for k, v in overrides.items() if v is not None})
    return replace(Settings(), **values)
