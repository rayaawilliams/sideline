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

Examples of good plans:
Q: How did Northwestern do last weekend?
{{"sql": "SELECT date, away, away_score, home, home_score FROM games WHERE (home LIKE '%Northwestern%' OR away LIKE '%Northwestern%') AND state = 'final' ORDER BY date DESC LIMIT 1", "search": null}}
Q: Who leads college football in rushing touchdowns?
{{"sql": "SELECT rank, name, team, value FROM stat_leaders WHERE sport = 'football' AND category = 'Rushing touchdowns' ORDER BY rank LIMIT 3", "search": null}}
Q: How are my players doing? / How are my teams doing? (user follows team Northwestern, no players)
{{"sql": "SELECT date, away, away_score, home, home_score FROM games WHERE (home LIKE '%Northwestern%' OR away LIKE '%Northwestern%') AND state = 'final' ORDER BY date DESC LIMIT 3", "search": "Northwestern"}}
Q: Who is the quarterback on Ohio St.?
{{"sql": "SELECT name, team, position, category, rank, value FROM stat_leaders WHERE team = 'Ohio St.' AND position = 'QB' ORDER BY rank LIMIT 3", "search": null}}
Q: Did Northwestern win the women's lacrosse championship?
{{"sql": null, "search": "Northwestern women's lacrosse championship"}}

Reply with JSON only, no other text: {{"sql": "<query or null>", "search": "<phrase or null>"}}
Use null for a tool you don't need. Use only sports, divisions and categories listed above.
If the question is about "my players" or "my teams", filter on the ones the user follows.
"My players" with no players followed means the teams the user follows: plan the same as "my teams"."""

ANSWER_PROMPT = f"""You are Sideline, a friendly college sports cheat sheet for busy students.
Answer using ONLY the stats table rows and news passages provided. Do not use outside knowledge.
Cite stats as [S] and news passages by number, like [1].
Only mention when the stats were updated if the user asks how current they are.
For "my players" or "my teams", cover every followed team and player, and say plainly when there is no data for one.
If the provided data does not answer the question, reply exactly: "{NOT_FOUND}"
Keep it short and friendly: one to three sentences. Don't start with a greeting like "Hey".
Write whole numbers without decimals (20, not 20.0)."""


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


def _cell(value) -> str:
    """Show 20.0 as 20 so answers read naturally."""
    if value is None:
        return ""
    if isinstance(value, float) and value.is_integer():
        return str(int(value))
    return str(value)


def format_rows(columns: list, rows: list) -> str:
    if not rows:
        return "(query returned no rows)"
    lines = [" | ".join(columns)]
    lines += [" | ".join(_cell(v) for v in row) for row in rows]
    return "\n".join(lines)


def load_follows(path: str) -> dict:
    try:
        return json.loads(Path(path).read_text())
    except (OSError, json.JSONDecodeError):
        return {"players": [], "teams": []}


MY_FOLLOWS = re.compile(r"\bmy\s+(?:favou?rite\s+)?(teams?|players?)\b", re.I)
STATUS_WORDS = re.compile(
    r"\b(how|doing|going|update|latest|recap|summary|next|when|schedule|upcoming|last|score|result|"
    r"win|won|lose|lost|play|playing|played)\b", re.I)


def is_status_question(question: str) -> bool:
    """'How are my teams doing?' style questions: answered with the fixed summary."""
    return bool(MY_FOLLOWS.search(question) and STATUS_WORDS.search(question))


def fill_in_follows(question: str, follows: dict) -> str:
    """'Who is the quarterback on my team?' -> 'Who is the quarterback on Northwestern?'"""
    teams, players = follows.get("teams", []), follows.get("players", [])

    def swap(match):
        kind = match.group(1).lower()
        names = teams if kind.startswith("team") else players
        if not names:
            return match.group(0)
        return " and ".join(names) if kind.endswith("s") or len(names) == 1 else names[0]

    return MY_FOLLOWS.sub(swap, question)


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

    def direct_lookup(self, question: str) -> list:
        """[(sql, columns, rows)] for teams and players named in the question, or the user's follows."""
        try:
            teams_known, players_known = db.known_names(self.settings.sports_db)
        except Exception:
            return []
        teams = db.find_names(question, teams_known)
        players = db.find_names(question, players_known)
        if not (teams or players) and re.search(r"\bmy\b", question.lower()):
            follows = load_follows(self.settings.follows_file)
            wanted = " ".join(follows.get("teams", []) + follows.get("players", []))
            teams = db.find_names(wanted, teams_known)
            players = db.find_names(wanted, players_known)
        results = []
        for sql in db.lookup_sql(teams, players):
            try:
                columns, rows = db.run_query(self.settings.sports_db, sql, self.settings.max_sql_rows)
            except Exception:
                continue
            if rows:
                results.append((sql, columns, rows))
        return results

    def my_follows_answer(self, question: str, started: float) -> Answer:
        """'How are my teams/players doing?': always the same lookups, no model-written SQL."""
        usage = {}
        follows = load_follows(self.settings.follows_file)
        teams, players = follows.get("teams", []), follows.get("players", [])
        result = Answer(question, "")
        queries, blocks, all_rows = [], [], []
        for team in teams:
            sql = db.team_games_sql(team)
            columns, rows = db.run_query(self.settings.sports_db, sql, self.settings.max_sql_rows)
            queries.append(sql)
            blocks.append(f"Games for {team}:\n" + (format_rows(columns, rows) if rows else "(no games found)"))
            all_rows += [dict(zip(columns, row)) for row in rows]
        missing = []
        if players:
            sql = db.player_stats_sql(players)
            columns, rows = db.run_query(self.settings.sports_db, sql, self.settings.max_sql_rows)
            queries.append(sql)
            found = {row[0] for row in rows}
            missing = [p for p in players if p not in found]
            if rows:
                blocks.append("Player stats:\n" + format_rows(columns, rows))
            all_rows += [dict(zip(columns, row)) for row in rows]
        result.sql = "; ".join(queries) or None
        result.sql_rows = all_rows
        if not teams and not players:
            result.text = ("You haven't picked any favourite teams or players yet. "
                           "Use **Edit favourites** in the sidebar to add some.")
        else:
            prompt = (
                "The user asked about their followed teams and players.\n"
                f"Followed teams: {', '.join(teams) or 'none'}. Followed players: {', '.join(players) or 'none'}.\n"
                "In games, state 'final' = played (with scores) and 'pre' = upcoming. "
                "Answer what the user asked: e.g. only the next game if they asked about the next game, "
                "only the last result if they asked about last weekend. If they asked generally, give each "
                "team's latest result and next game, and each player's stats.\n"
                + (f"No stats were found for these players, so say 'There's no recent data on {', '.join(missing)} yet.': "
                   f"{', '.join(missing)}\n" if missing else "")
                + "\n[S] Stats query: lookups below\n" + "\n\n".join(blocks)
                + f"\n\nNews passages:\n(none)\n\nQuestion: {question}"
            )
            result.text = self._complete(ANSWER_PROMPT, prompt, usage).strip()
        result.provider, result.model = self.llm.provider, self.llm.model
        result.seconds = time.perf_counter() - started
        result.input_tokens, result.output_tokens = usage.get("in"), usage.get("out")
        return result

    def answer(self, question: str) -> Answer:
        started, usage = time.perf_counter(), {}
        if is_status_question(question):
            return self.my_follows_answer(question, started)
        asked = question
        question = fill_in_follows(question, load_follows(self.settings.follows_file))
        plan = self.plan(question, usage)
        result = Answer(asked, "", sql=plan["sql"], search=plan["search"])

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
        stats_block = format_rows(columns, rows) if result.sql else "(no stats query)"

        hits = self.search_news(result.search)
        planned_stats = bool(plan["sql"])
        if not rows and (planned_stats or not hits):
            # The model's query found nothing: look up the teams and players directly.
            direct = self.direct_lookup(question)
            if direct:
                result.sql = "; ".join(sql for sql, _, _ in direct)
                result.sql_error = None
                result.sql_rows = [dict(zip(cols, row)) for _, cols, found in direct for row in found]
                stats_block = "\n\n".join(format_rows(cols, found) for _, cols, found in direct)
        if not hits and not result.sql_rows:
            result.search = result.search or question
            hits = self.search_news(question)
        result.sources = [
            {"source": h.chunk.source, "page": h.chunk.page, "score": round(h.score, 3)} for h in hits
        ]

        updated = db.get_meta(self.settings.sports_db, "last_update") or "unknown"
        passages = "\n\n".join(f"[{i}] ({h.chunk.source})\n{h.chunk.text}" for i, h in enumerate(hits, 1))
        prompt = (
            f"Stats last updated: {updated}\n\n"
            f"[S] Stats query: {result.sql or 'none'}\n{stats_block}\n\n"
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
