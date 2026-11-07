"""Build an evaluation question set from the stats database, so every answer is known exactly.

Usage: python -m eval.make_questions [--per-category 3] [--out eval/questions.jsonl]

Because the questions come from the same data the assistant uses, the expected answers are
ground truth for that snapshot. Re-run it after each update so the answers stay current.
Add your own questions (news, favourites) by hand to eval/questions.jsonl afterwards.
"""
import argparse
import json
import random
import sqlite3

from rag.config import load_settings

SPORT_NAMES = {
    "football": "college football",
    "lacrosse-women": "women's college lacrosse",
    "lacrosse-men": "men's college lacrosse",
}

# Questions the data can't answer: the assistant should refuse rather than guess.
UNANSWERABLE = [
    "Who will win the national championship next year?",
    "What is the starting quarterback's GPA?",
    "How many goals did the leading scorer have in high school?",
    "Which team has the best mascot?",
]


def fmt_value(value) -> str:
    return str(int(value)) if value is not None and float(value).is_integer() else f"{value}"


def make_questions(sports_db: str, per_category: int = 3, seed: int = 7) -> list:
    rng = random.Random(seed)
    connection = sqlite3.connect(f"file:{sports_db}?mode=ro", uri=True)
    questions = []
    boards = connection.execute(
        "SELECT DISTINCT sport, division, category, scope FROM stat_leaders ORDER BY sport, category"
    ).fetchall()
    for sport, division, category, scope in boards:
        sport_name = SPORT_NAMES.get(sport, sport)
        rows = connection.execute(
            "SELECT rank, name, team, value FROM stat_leaders WHERE sport=? AND division=? AND category=? "
            "AND rank IS NOT NULL ORDER BY rank",
            (sport, division, category),
        ).fetchall()
        if not rows:
            continue
        leaders = [r for r in rows if r[0] == rows[0][0]]
        if len(leaders) == 1:  # skip ties: there's no single right answer
            rank, name, team, value = leaders[0]
            who = "team" if scope == "team" else "player"
            questions.append({
                "kind": "leader",
                "question": f"Which {who} leads {sport_name} ({division.upper()}) in {category.lower()}?",
                "answer_keywords": [name.split()[-1]],
            })
        for rank, name, team, value in rng.sample(rows[1:], min(per_category - 1, len(rows) - 1)):
            if scope == "individual" and value is not None:
                questions.append({
                    "kind": "player_stat",
                    "question": f"What is {name}'s {category.lower()} number in {sport_name} this season?",
                    "answer_keywords": [fmt_value(value)],
                })
    games = connection.execute(
        "SELECT away, home, away_score, home_score FROM games WHERE state LIKE 'final%' "
        "AND away_score IS NOT NULL AND home_score IS NOT NULL AND away_score != home_score"
    ).fetchall()
    for away, home, away_score, home_score in rng.sample(games, min(5, len(games))):
        winner = away if away_score > home_score else home
        questions.append({
            "kind": "game",
            "question": f"Who won when {away} played {home}?",
            "answer_keywords": [winner],
        })
    connection.close()
    questions += [{"kind": "unanswerable", "question": q, "answerable": False} for q in UNANSWERABLE]
    for number, item in enumerate(questions, start=1):
        item["id"] = f"q{number:02d}"
        item.setdefault("answerable", True)
    return questions


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--per-category", type=int, default=3)
    parser.add_argument("--out", default="eval/questions.jsonl")
    args = parser.parse_args()
    questions = make_questions(load_settings().sports_db, args.per_category)
    with open(args.out, "w") as file:
        for item in questions:
            file.write(json.dumps(item) + "\n")
    print(f"Wrote {len(questions)} questions to {args.out}")


if __name__ == "__main__":
    main()
