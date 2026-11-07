"""The assistant: plan how to answer, look up stats and news, then answer with citations.

Two model calls per question:
  1. Plan: given the question, the stats tables and what's loaded, write a read-only SQL query
     for stat questions and/or a search phrase for news questions.
  2. Answer: given the query results and the news passages, answer using only those, citing them.
If the first query fails, the error goes back to the model once so it can fix the query.
"""
import json
import re
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from . import db

NOT_FOUND = "I couldn't find that in my sports data."

PLAN_PROMPT = """You plan how to answer college sports questions for Sideline, a college sports cheat sheet.
You can use two tools:
  sql    - one read-only SQLite SELECT query on the stats database (for numbers: leaders, stats, scores, records)
  search - a short search phrase for recent news articles (for what happened, storylines, injuries, previews)

{schema}

Data currently loaded:
{catalog}

The user follows: {follows}
Today is {today}.

Reply with JSON only, no other text: {{"sql": "<query or null>", "search": "<phrase or null>"}}
Use null for a tool you don't need. Use only sports, divisions and categories listed above.
If the question is about "my players" or "my teams", filter on the ones the user follows."""

ANSWER_PROMPT = f"""You are Sideline, a friendly college sports cheat sheet for busy students.
Answer using ONLY the stats table rows and news passages provided. Do not use outside knowledge.
Cite stats as [S] and news passages by number, like [1].
Mention when the stats were last updated if it matters.
If the provided data does not answer the question, reply exactly: "{NOT_FOUND}"
Keep it short and casual: one to three sentences."""


@dataclass
class Answer:
    question: str
    text: str
    sql: str | None = None
    sql_error: str | None = None
    sql_rows: list = field(default_factory=list)
    search: str | None = None
    sources: list = field(default_factory=list)  # news passages: [{"source", "page", "score"}]
    provider: str = ""
    model: str = ""
    seconds: float = 0.0
    input_tokens: int | None = None
    output_tokens: int | None = None

    @property
    def latency_s(self) -> float:
        return self.seconds

    @property
    def is_refusal(self) -> bool:
        return NOT_FOUND.lower().rstrip(".") in self.text.lower()


def parse_plan(text: str) -> dict:
    """Pull {"sql": ..., "search": ...} out of the model's reply, tolerating code fences."""
    match = re.search(r"\{.*\}", text, re.S)
    if not match:
        return {"sql": None, "search": None}
    try:
        plan = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"sql": None, "search": None}
    clean = lambda v: v.strip() if isinstance(v, str) and v.strip() and v.strip().lower() != "null" else None
    return {"sql": clean(plan.get("sql")), "search": clean(plan.get("search"))}


def format_rows(columns: list, rows: list) -> str:
    if not rows:
        return "(query returned no rows)"
    lines = [" | ".join(columns)]
    lines += [" | ".join("" if v is None else str(v) for v in row) for row in rows]
    return "\n".join(lines)


def load_follows(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {"players": [], "teams": []}


class Assistant:
    def __init__(self, settings, llm, store=None, embedder=None):
        self.settings = settings
        self.llm = llm
        self.store = store  # vector store of news; optional
        self.embedder = embedder

    def _complete(self, system, prompt, usage):
        completion = self.llm.complete(system, prompt)
        for key, value in (("in", completion.input_tokens), ("out", completion.output_tokens)):
            if value is not None:
                usage[key] = usage.get(key, 0) + value
        return completion.text

    def plan(self, question: str, usage: dict, error_note: str = "") -> dict:
        follows = load_follows(self.settings.follows_file)
        system = PLAN_PROMPT.format(
            schema=db.SCHEMA_GUIDE,
            catalog=db.catalog(self.settings.sports_db),
            follows=json.dumps(follows),
            today=date.today().isoformat(),
        )
        return parse_plan(self._complete(system, f"Question: {question}{error_note}", usage))

    def search_news(self, phrase: str) -> list:
        if not (phrase and self.store and self.embedder):
            return []
        return self.store.query(self.embedder.embed([phrase])[0], self.settings.top_k)

    def answer(self, question: str) -> Answer:
        started, usage = time.perf_counter(), {}
        plan = self.plan(question, usage)
        result = Answer(question, "", sql=plan["sql"], search=plan["search"])

        columns, rows = [], []
        if plan["sql"]:
            try:
                columns, rows = db.run_query(self.settings.sports_db, plan["sql"], self.settings.max_sql_rows)
            except Exception as error:  # bad SQL: give the model one chance to fix it
                retry = self.plan(question, usage, f"\n\nYour previous query failed with: {error}\nQuery: {plan['sql']}")
                result.sql, result.search = retry["sql"], retry["search"] or plan["search"]
                try:
                    if result.sql:
                        columns, rows = db.run_query(self.settings.sports_db, result.sql, self.settings.max_sql_rows)
                except Exception as second_error:
                    result.sql_error = str(second_error)
        result.sql_rows = [dict(zip(columns, row)) for row in rows]

        hits = self.search_news(result.search)
        result.sources = [
            {"source": h.chunk.source, "page": h.chunk.page, "score": round(h.score, 3)} for h in hits
        ]

        updated = db.get_meta(self.settings.sports_db, "last_update") or "unknown"
        passages = "\n\n".join(f"[{i}] ({h.chunk.source})\n{h.chunk.text}" for i, h in enumerate(hits, 1))
        prompt = (
            f"Stats last updated: {updated}\n\n"
            f"[S] Stats query: {result.sql or 'none'}\n{format_rows(columns, rows) if result.sql else '(no stats query)'}\n\n"
            f"News passages:\n{passages or '(none)'}\n\n"
            f"Question: {question}"
        )
        result.text = self._complete(ANSWER_PROMPT, prompt, usage).strip()
        result.provider, result.model = self.llm.provider, self.llm.model
        result.seconds = time.perf_counter() - started
        result.input_tokens, result.output_tokens = usage.get("in"), usage.get("out")
        return result


def build_assistant(settings, llm=None, embedder=None):
    """Wire up the model, and the news search if the news index exists."""
    from rag.embeddings import make_embedder
    from rag.llm import make_llm
    from rag.store import VectorStore

    store = None
    if Path(settings.db_dir).exists():
        embedder = embedder or make_embedder(settings)
        store = VectorStore(settings.db_dir, settings.collection, embedder.name)
    return Assistant(settings, llm or make_llm(settings), store, embedder)
