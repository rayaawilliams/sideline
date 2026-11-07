"""The stats database (SQLite) and the safe, read-only queries the assistant runs on it."""
import re
import sqlite3
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS stat_leaders (
    sport TEXT,        -- 'football', 'lacrosse-women', 'lacrosse-men'
    division TEXT,     -- 'fbs', 'd1', ...
    category TEXT,     -- leaderboard name, e.g. 'Rushing touchdowns', 'Goals per game'
    scope TEXT,        -- 'individual' or 'team'
    rank INTEGER,      -- 1 = leader
    name TEXT,         -- player name (individual) or team name (team)
    team TEXT,
    class TEXT,        -- Fr., So., Jr., Sr.
    position TEXT,
    games INTEGER,
    stat TEXT,         -- column the leaderboard is ranked by, e.g. 'Rush TD'
    value REAL,        -- that stat's value
    details TEXT,      -- every column from the source, as JSON
    updated TEXT,      -- when the source last updated this leaderboard
    PRIMARY KEY (sport, division, category, name, team)
);
CREATE TABLE IF NOT EXISTS games (
    game_id TEXT PRIMARY KEY,
    sport TEXT, division TEXT,
    date TEXT,         -- as given by the source, e.g. '11-01-2026'
    away TEXT, home TEXT,
    away_score INTEGER, home_score INTEGER,
    state TEXT,        -- 'final', 'live', 'pre'
    title TEXT
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""

# What the planning model is told about the tables. Keep in step with SCHEMA.
SCHEMA_GUIDE = """Table stat_leaders: one row per player (or team) per leaderboard.
  sport, division, category, scope ('individual'|'team'), rank (1 = leader), name, team,
  class, position, games, stat (ranked column), value (its number), updated
Table games: game_id, sport, division, date, away, home, away_score, home_score, state, title
Use LIKE with % for names and teams (e.g. team LIKE '%Northwestern%'). Order leaderboards by rank."""


def connect(path: str) -> sqlite3.Connection:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path)
    connection.executescript(SCHEMA)
    return connection


def replace_leaderboard(connection, sport, division, category, scope, rows, updated) -> None:
    """Swap in a fresh copy of one leaderboard."""
    connection.execute(
        "DELETE FROM stat_leaders WHERE sport=? AND division=? AND category=?", (sport, division, category),
    )
    connection.executemany(
        """INSERT OR REPLACE INTO stat_leaders
           (sport, division, category, scope, rank, name, team, class, position, games, stat, value, details, updated)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        [
            (sport, division, category, scope, r["rank"], r["name"], r["team"], r["class"], r["position"],
             r["games"], r["stat"], r["value"], r["details"], updated)
            for r in rows
        ],
    )


def upsert_games(connection, sport, division, games) -> None:
    connection.executemany(
        """INSERT OR REPLACE INTO games
           (game_id, sport, division, date, away, home, away_score, home_score, state, title)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        [
            (g["game_id"], sport, division, g["date"], g["away"], g["home"], g["away_score"],
             g["home_score"], g["state"], g["title"])
            for g in games
        ],
    )


def set_meta(connection, key: str, value: str) -> None:
    connection.execute("INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)", (key, value))


def get_meta(path: str, key: str) -> str | None:
    if not Path(path).exists():
        return None
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        try:
            row = connection.execute("SELECT value FROM meta WHERE key=?", (key,)).fetchone()
        except sqlite3.OperationalError:
            return None
    return row[0] if row else None


def catalog(path: str) -> str:
    """Which leaderboards exist, so the planner uses real category names."""
    if not Path(path).exists():
        return "(no stats loaded yet)"
    with sqlite3.connect(f"file:{path}?mode=ro", uri=True) as connection:
        try:
            rows = connection.execute(
                "SELECT sport, division, scope, category, COUNT(*) FROM stat_leaders "
                "GROUP BY sport, division, scope, category ORDER BY sport, category"
            ).fetchall()
            game_rows = connection.execute(
                "SELECT sport, division, COUNT(*), MIN(date), MAX(date) FROM games GROUP BY sport, division"
            ).fetchall()
        except sqlite3.OperationalError:
            return "(no stats loaded yet)"
    lines = [f"- stat_leaders: sport='{s}', division='{d}', scope='{sc}', category='{c}' ({n} rows)" for s, d, sc, c, n in rows]
    lines += [f"- games: sport='{s}', division='{d}' ({n} games, {lo} to {hi})" for s, d, n, lo, hi in game_rows]
    return "\n".join(lines) or "(no stats loaded yet)"


class UnsafeQuery(ValueError):
    pass


FORBIDDEN = re.compile(
    r"\b(insert|update|delete|drop|alter|create|attach|detach|pragma|vacuum|reindex)\b", re.I,
)


def check_query(sql: str) -> str:
    """Allow a single SELECT (or WITH ... SELECT) statement only."""
    query = sql.strip().rstrip(";").strip()
    if ";" in query:
        raise UnsafeQuery("Only one statement is allowed.")
    if not re.match(r"^(select|with)\b", query, re.I):
        raise UnsafeQuery("Only SELECT queries are allowed.")
    if FORBIDDEN.search(query):
        raise UnsafeQuery("Query contains a forbidden keyword.")
    return query


def run_query(path: str, sql: str, max_rows: int) -> tuple:
    """Run a checked query on a read-only connection. Returns (column names, rows)."""
    query = check_query(sql)
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        cursor = connection.execute(query)
        columns = [c[0] for c in cursor.description or []]
        rows = cursor.fetchmany(max_rows)
    finally:
        connection.close()
    return columns, rows
