import json
from pathlib import Path

import pytest

from rag.config import load_settings
from sports import db
from sports.ncaa import parse_news, parse_scoreboard, parse_stat_rows, to_number
from sports.update import update

FIX = Path("tests/fixtures")


def fixture(name):
    return json.loads((FIX / name).read_text())


def test_to_number_handles_formats():
    assert to_number("1,234") == 1234.0
    assert to_number(".512") == 0.512
    assert to_number("45.5%") == 45.5
    assert to_number("-") is None and to_number("") is None and to_number(None) is None


def test_stat_rows_use_last_numeric_column_as_value():
    rows = parse_stat_rows(fixture("stats_football_750.json"), "individual")
    assert [r["name"] for r in rows][:2] == ["Jordan Testplayer", "Avery Sample"]
    assert rows[0]["stat"] == "Rush TD" and rows[0]["value"] == 17 and rows[0]["games"] == 9
    lax = parse_stat_rows(fixture("stats_lax_goals.json"), "individual")
    assert lax[0]["stat"] == "Per Game" and lax[0]["value"] == 4.4
    assert json.loads(lax[0]["details"])["Goals"] == "88"


def test_scoreboard_parses_games_and_skips_broken_entries():
    games = parse_scoreboard(fixture("scoreboard_football.json"))
    assert len(games) == 2
    assert games[0] == {
        "game_id": "900001", "date": "10-31-2026", "away": "Riverside", "home": "Northwestern",
        "away_score": 17, "home_score": 24, "state": "final", "title": "Riverside Northwestern",
    }
    assert games[1]["away_score"] is None


def test_news_strips_html_and_skips_untitled():
    items = parse_news(fixture("news_football.json"))
    assert len(items) == 2
    assert "<p>" not in items[0]["text"] and "two touchdowns" in items[0]["text"]
    assert parse_news({"items": fixture("news_football.json")}) == items


def test_query_guard_allows_select_only():
    assert db.check_query("SELECT * FROM games;") == "SELECT * FROM games"
    assert db.check_query("with t as (select 1) select * from t")
    for bad in ["DROP TABLE games", "SELECT 1; DELETE FROM games", "UPDATE games SET home='x'",
                "select * from games where 1=1; attach 'x' as y", "PRAGMA table_info(games)"]:
        with pytest.raises(db.UnsafeQuery):
            db.check_query(bad)


class FakeClient:
    """Serves fixture files instead of calling the NCAA API."""

    def __init__(self):
        self.paths = []

    def get(self, path):
        self.paths.append(path)
        if path.startswith("/stats/football/fbs/current/individual/750"):
            return fixture("stats_football_750.json")
        if path.startswith("/stats/lacrosse-women/d1/current/individual/1"):
            return fixture("stats_lax_goals.json")
        if path.startswith("/scoreboard/football/fbs/2026/10/"):
            return fixture("scoreboard_football.json")
        if path == "/news/football/fbs":
            return fixture("news_football.json")
        return None


@pytest.fixture
def loaded(tmp_path):
    sources = {
        "season_year": 2026,
        "feeds": [
            {"sport": "football", "division": "fbs", "label": "College football",
             "stats": [{"id": 750, "scope": "individual", "category": "Rushing touchdowns"}],
             "scoreboard": {"type": "weekly", "weeks": [9, 10]}, "news": True},
            {"sport": "lacrosse-women", "division": "d1", "label": "Women's lacrosse",
             "stats": [{"id": 1, "scope": "individual", "category": "Goals per game"}],
             "scoreboard": None, "news": True},
        ],
    }
    (tmp_path / "sources.json").write_text(json.dumps(sources))
    settings = load_settings(
        sources_file=str(tmp_path / "sources.json"), sports_db=str(tmp_path / "sports.db"),
        docs_dir=str(tmp_path / "docs"), raw_dir=None,
    )
    client = FakeClient()
    counts = update(settings, client=client)
    return settings, counts, client


def test_update_fills_database_and_news_docs(loaded):
    settings, counts, client = loaded
    assert counts == {"stat_rows": 6, "games": 2, "news": 2}
    assert "/scoreboard/football/fbs/2026/09/all-conf" in client.paths  # weeks are zero-padded
    columns, rows = db.run_query(
        settings.sports_db, "SELECT name, value FROM stat_leaders WHERE category='Rushing touchdowns' AND rank=1", 5,
    )
    assert rows == [("Jordan Testplayer", 17.0)]
    news = sorted(p.name for p in Path(settings.docs_dir, "news").iterdir())
    assert news[0].startswith("football-northwestern-holds-off")
    assert db.get_meta(settings.sports_db, "last_update")


def test_update_is_repeatable_without_duplicates(loaded):
    settings, _, _ = loaded
    update(settings, client=FakeClient())
    _, rows = db.run_query(settings.sports_db, "SELECT COUNT(*) FROM stat_leaders", 1)
    assert rows == [(6,)]


def test_catalog_lists_loaded_data(loaded):
    text = db.catalog(loaded[0].sports_db)
    assert "category='Rushing touchdowns'" in text and "sport='lacrosse-women'" in text
    assert "games: sport='football'" in text


def test_read_only_connection_blocks_writes_even_if_guard_missed(loaded):
    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        conn = sqlite3.connect(f"file:{loaded[0].sports_db}?mode=ro", uri=True)
        conn.execute("DELETE FROM games")
