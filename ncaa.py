"""Client for the free, unofficial NCAA API (https://github.com/henrygd/ncaa-api).

It mirrors ncaa.com: request the same path as the website and get JSON back.
The public instance allows 5 requests per second per IP, so requests are spaced out.
Because the API is unofficial and can change, the parsers below read fields defensively
and skip anything they don't recognise instead of crashing.
"""
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

USER_AGENT = "sideline-college-sports-cheat-sheet/1.0"


class NcaaClient:
    def __init__(self, base_url: str, raw_dir: str | None = None, min_interval_s: float = 0.3):
        self.base_url = base_url.rstrip("/")
        self.raw_dir = Path(raw_dir) if raw_dir else None
        self.min_interval_s = min_interval_s
        self._last_request = 0.0

    def get(self, path: str, retries: int = 2):
        """GET a path like /stats/football/fbs/current/individual/750. Returns parsed JSON or None on 404."""
        url = f"{self.base_url}/{path.lstrip('/')}"
        for attempt in range(retries + 1):
            wait = self.min_interval_s - (time.monotonic() - self._last_request)
            if wait > 0:
                time.sleep(wait)
            self._last_request = time.monotonic()
            request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    data = json.load(response)
            except urllib.error.HTTPError as error:
                if error.code == 404:
                    return None
                if error.code == 429 or error.code >= 500:
                    time.sleep(2 * (attempt + 1))
                    continue
                raise
            except urllib.error.URLError:
                if attempt == retries:
                    raise
                time.sleep(2 * (attempt + 1))
                continue
            self._save_raw(path, data)
            return data
        return None

    def _save_raw(self, path: str, data) -> None:
        if not self.raw_dir:
            return
        name = re.sub(r"[^a-zA-Z0-9]+", "_", path.strip("/")) + ".json"
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        (self.raw_dir / name).write_text(json.dumps(data, indent=1))


# ---------- parsers: API JSON -> plain dicts ready for the database ----------

NAME_KEYS = ("Name", "Player")
KNOWN_KEYS = {"Rank", "Name", "Player", "Team", "Cl", "Class", "Position", "Pos", "G", "GP", "Ht", "Height"}


def to_number(value):
    """'1,234' -> 1234.0, '.512' -> 0.512, '-' or '' -> None."""
    if value is None:
        return None
    text = str(value).replace(",", "").replace("%", "").strip()
    try:
        return float(text)
    except ValueError:
        return None


def parse_stat_rows(payload: dict, scope: str) -> list:
    """Rows of one stat leaderboard. The ranked stat is the last numeric column."""
    rows = []
    for item in payload.get("data") or []:
        if not isinstance(item, dict):
            continue
        name = next((item[k] for k in NAME_KEYS if item.get(k)), None)
        team = item.get("Team")
        if scope == "team":
            name = name or team
        if not name:
            continue
        stat_key, value = None, None
        for key, raw in item.items():
            if key not in KNOWN_KEYS and to_number(raw) is not None:
                stat_key, value = key, to_number(raw)
        rank = to_number(item.get("Rank"))
        games = to_number(item.get("G") or item.get("GP"))
        rows.append({
            "rank": int(rank) if rank is not None else None,
            "name": str(name).strip(),
            "team": (team or name or "").strip(),
            "class": item.get("Cl") or item.get("Class"),
            "position": item.get("Position") or item.get("Pos"),
            "games": int(games) if games is not None else None,
            "stat": stat_key,
            "value": value,
            "details": json.dumps(item),
        })
    return rows


def _team_name(side: dict) -> str:
    names = side.get("names") or {}
    return (names.get("short") or names.get("full") or names.get("char6") or "").strip()


def parse_scoreboard(payload: dict) -> list:
    """Games from a scoreboard response: {"games": [{"game": {...}}, ...]}."""
    games = []
    for entry in payload.get("games") or []:
        game = entry.get("game", entry) if isinstance(entry, dict) else None
        if not isinstance(game, dict):
            continue
        home, away = game.get("home") or {}, game.get("away") or {}
        if not (_team_name(home) and _team_name(away)):
            continue
        games.append({
            "game_id": str(game.get("gameID") or game.get("url") or f"{_team_name(away)}@{_team_name(home)}:{game.get('startDate')}"),
            "date": game.get("startDate"),
            "away": _team_name(away),
            "home": _team_name(home),
            "away_score": int(to_number(away.get("score"))) if to_number(away.get("score")) is not None else None,
            "home_score": int(to_number(home.get("score"))) if to_number(home.get("score")) is not None else None,
            "state": game.get("gameState") or game.get("currentPeriod"),
            "title": game.get("title"),
        })
    return games


def parse_news(payload) -> list:
    """News items from a parsed RSS feed. Accepts a list or a dict holding one."""
    items = payload
    if isinstance(payload, dict):
        items = next(
            (payload[k] for k in ("items", "articles", "news", "data", "entries") if isinstance(payload.get(k), list)),
            [],
        )
    news = []
    for item in items or []:
        if not isinstance(item, dict) or not item.get("title"):
            continue
        description = item.get("description") or item.get("contentSnippet") or item.get("summary") or ""
        news.append({
            "title": str(item["title"]).strip(),
            "date": item.get("pubDate") or item.get("isoDate") or item.get("date") or "",
            "link": item.get("link") or item.get("url") or "",
            "text": re.sub(r"<[^>]+>", " ", str(description)).strip(),
        })
    return news
