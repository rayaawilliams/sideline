"""Pull the latest stats, scores and news from the NCAA API.

Usage: python -m sports.update        (then: python -m rag.ingest --reset, to index the news)

Run it once a day. Every response is also saved to data/raw/ for debugging.
"""
import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from rag.config import load_settings

from . import db
from .ncaa import NcaaClient, parse_news, parse_scoreboard, parse_stat_rows


def load_sources(path: str) -> dict:
    return json.loads(Path(path).read_text())


def update_stats(client, connection, feed) -> int:
    total = 0
    for stat in feed.get("stats", []):
        rows, updated, page, pages = [], "", 1, 1
        while page <= pages and page <= 10:
            path = f"/stats/{feed['sport']}/{feed['division']}/current/{stat['scope']}/{stat['id']}"
            payload = client.get(path + (f"?page={page}" if page > 1 else ""))
            if not payload:
                print(f"  ! no data for {feed['sport']} {stat['category']} ({path})")
                break
            rows += parse_stat_rows(payload, stat["scope"])
            updated = payload.get("updated", updated)
            pages = int(payload.get("pages") or 1)
            page += 1
        if rows:
            db.replace_leaderboard(
                connection, feed["sport"], feed["division"], stat["category"], stat["scope"], rows, updated,
            )
            print(f"  {feed['sport']}: {stat['category']}, {len(rows)} rows")
            total += len(rows)
    return total


def update_scores(client, connection, feed, season_year: int) -> int:
    board = feed.get("scoreboard")
    if not board:
        return 0
    games = []
    if board.get("type") == "weekly":
        for week in board.get("weeks", []):
            payload = client.get(f"/scoreboard/{feed['sport']}/{feed['division']}/{season_year}/{week:02d}/all-conf")
            if payload:
                games += parse_scoreboard(payload)
    for date in board.get("dates", []):  # "YYYY/MM/DD", for sports scheduled by day
        payload = client.get(f"/scoreboard/{feed['sport']}/{feed['division']}/{date}/all-conf")
        if payload:
            games += parse_scoreboard(payload)
    db.upsert_games(connection, feed["sport"], feed["division"], games)
    if games:
        print(f"  {feed['sport']}: {len(games)} games")
    return len(games)


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:60]


def update_news(client, feed, docs_dir: str) -> int:
    if not feed.get("news"):
        return 0
    payload = client.get(f"/news/{feed['sport']}/{feed['division']}")
    items = parse_news(payload) if payload else []
    out = Path(docs_dir) / "news"
    out.mkdir(parents=True, exist_ok=True)
    for item in items:
        body = f"# {item['title']}\n\nSport: {feed.get('label', feed['sport'])}\nDate: {item['date']}\nLink: {item['link']}\n\n{item['text']}\n"
        (out / f"{feed['sport']}-{slug(item['title'])}.md").write_text(body)
    if items:
        print(f"  {feed['sport']}: {len(items)} news articles")
    return len(items)


def update(settings, client=None) -> dict:
    sources = load_sources(settings.sources_file)
    client = client or NcaaClient(settings.ncaa_api_base, settings.raw_dir)
    connection = db.connect(settings.sports_db)
    counts = {"stat_rows": 0, "games": 0, "news": 0}
    try:
        for feed in sources["feeds"]:
            print(feed.get("label", feed["sport"]))
            counts["stat_rows"] += update_stats(client, connection, feed)
            counts["games"] += update_scores(client, connection, feed, sources["season_year"])
            counts["news"] += update_news(client, feed, settings.docs_dir)
        db.set_meta(connection, "last_update", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))
        connection.commit()
    finally:
        connection.close()
    return counts


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    counts = update(load_settings())
    print(f"Done: {counts['stat_rows']} stat rows, {counts['games']} games, {counts['news']} news articles.")
    print("Now run: python -m rag.ingest --reset")


if __name__ == "__main__":
    main()
