"""The assistant end to end, with a fake model that plays both the planner and the answerer."""
import json
import re

import pytest

from rag.embeddings import HashingEmbedder
from rag.ingest import ingest
from rag.llm import FakeLLM
from rag.store import VectorStore
from sports.assistant import NOT_FOUND, Assistant, parse_plan
from tests.test_sports_data import loaded  # noqa: F401  (pytest fixture)


def scripted_llm(plans):
    """plans: question -> list of plan dicts, returned in order (to test retries)."""
    calls = {}

    def respond(system, prompt):
        if system.startswith("You plan"):
            question = prompt.split("Question: ", 1)[1].split("\n")[0]
            index = calls.get(question, 0)
            calls[question] = index + 1
            return "```json\n" + json.dumps(plans[question][min(index, len(plans[question]) - 1)]) + "\n```"
        # Answerer: echo the first stats row, else the first news passage, else refuse.
        rows = prompt.split("[S] Stats query:", 1)[1].split("News passages:", 1)[0].strip().split("\n")
        if len(rows) >= 3 and "(" not in rows[2][:1]:
            return f"{rows[2]} [S]"
        passage = re.search(r"\[1\] \([^)]*\)\n(.+?)(\n\n|\nQuestion:)", prompt, re.S)
        return f"{passage.group(1)} [1]" if passage else NOT_FOUND

    return FakeLLM(respond), calls


LEADER_SQL = "SELECT name, team, value FROM stat_leaders WHERE category='Rushing touchdowns' ORDER BY rank LIMIT 1"


def test_parse_plan_handles_fences_nulls_and_garbage():
    assert parse_plan('```json\n{"sql": "SELECT 1", "search": null}\n```') == {"sql": "SELECT 1", "search": None}
    assert parse_plan('{"sql": "null", "search": "  "}') == {"sql": None, "search": None}
    assert parse_plan("I don't know") == {"sql": None, "search": None}


def test_stat_question_runs_sql_and_answers_from_rows(loaded):
    settings = loaded[0]
    llm, _ = scripted_llm({"Who leads in rushing TDs?": [{"sql": LEADER_SQL, "search": None}]})
    answer = Assistant(settings, llm).answer("Who leads in rushing TDs?")
    assert answer.sql_rows == [{"name": "Jordan Testplayer", "team": "Riverside", "value": 17.0}]
    assert "Jordan Testplayer" in answer.text and answer.sql_error is None
    assert answer.input_tokens and answer.output_tokens  # summed over both model calls


def test_bad_sql_gets_one_retry_with_the_error(loaded):
    settings = loaded[0]
    llm, calls = scripted_llm({"Leader?": [{"sql": "SELECT nme FROM stat_leaders", "search": None},
                                           {"sql": LEADER_SQL, "search": None}]})
    answer = Assistant(settings, llm).answer("Leader?")
    assert calls["Leader?"] == 2
    assert answer.sql == LEADER_SQL and answer.sql_error is None
    assert "Jordan Testplayer" in answer.text


def test_unsafe_sql_is_never_run(loaded):
    settings = loaded[0]
    llm, _ = scripted_llm({"Wipe it": [{"sql": "DELETE FROM games", "search": None}]})
    answer = Assistant(settings, llm).answer("Wipe it")
    assert answer.sql_error and answer.is_refusal
    from sports import db
    assert db.run_query(settings.sports_db, "SELECT COUNT(*) FROM games", 1)[1] == [(2,)]


def test_news_question_uses_search(loaded, tmp_path):
    settings = loaded[0]
    from dataclasses import replace
    settings = replace(settings, db_dir=str(tmp_path / "vec"), embedder="hashing", top_k=1)
    embedder = HashingEmbedder()
    ingest(settings, embedder=embedder)
    store = VectorStore(settings.db_dir, settings.collection, embedder.name)
    question = "What happened when Northwestern played Riverside?"
    llm, _ = scripted_llm({question: [{"sql": None, "search": "Northwestern Riverside game"}]})
    answer = Assistant(settings, llm, store, embedder).answer(question)
    assert answer.sources[0]["source"].startswith("football-northwestern")
    assert "Northwestern holds off Riverside" in answer.text


def test_nothing_found_means_refusal(loaded):
    llm, _ = scripted_llm({"Mascot?": [{"sql": None, "search": None}]})
    assert Assistant(loaded[0], llm).answer("Mascot?").is_refusal


def test_planner_sees_catalog_and_follows(loaded, tmp_path):
    settings = loaded[0]
    from dataclasses import replace
    follows = tmp_path / "follows.json"
    follows.write_text(json.dumps({"players": ["Morgan Fixture"], "teams": ["Northwestern"]}))
    settings = replace(settings, follows_file=str(follows))
    seen = {}

    def respond(system, prompt):
        seen.setdefault("system", system)
        return '{"sql": null, "search": null}' if system.startswith("You plan") else NOT_FOUND

    Assistant(settings, FakeLLM(respond)).answer("How did my players do?")
    assert "Morgan Fixture" in seen["system"] and "category='Goals per game'" in seen["system"]
