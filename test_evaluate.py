import json

from eval.evaluate import format_table, keyword_match, load_questions, retrieval_hit, score_question, summarize
from eval.make_questions import make_questions
from sports.assistant import NOT_FOUND, Answer
from tests.test_sports_data import loaded  # noqa: F401  (pytest fixture)


def make_answer(text, sql=None, sql_error=None, sources=(), seconds=0.5, tokens=(100, 20)):
    return Answer("q", text, sql=sql, sql_error=sql_error, sources=list(sources),
                  provider="fake", model="fake-1", seconds=seconds,
                  input_tokens=tokens[0], output_tokens=tokens[1])


def test_keyword_match_needs_every_keyword_and_accepts_alternatives():
    assert keyword_match("Testplayer leads with 17.", ["Testplayer", "17"])
    assert not keyword_match("Testplayer leads with 17.", ["Testplayer", "18"])
    assert keyword_match("Northwestern won", [["NU", "Northwestern"]])
    assert keyword_match("Anything", [])


def test_retrieval_hit_checks_expected_article():
    sources = [{"source": "football-a.md", "page": None, "score": 0.8}]
    assert retrieval_hit(sources, "football-a.md")
    assert not retrieval_hit(sources, "football-b.md")
    assert retrieval_hit(sources, None) is None


def test_scores_answer_and_sql_success():
    item = {"id": "q1", "question": "q", "answerable": True, "answer_keywords": ["17"]}
    row = score_question(item, make_answer("17 rushing TDs [S]", sql="SELECT 1"))
    assert row["correct"] and row["sql_ok"]
    failed = score_question(item, make_answer(NOT_FOUND, sql="SELECT x", sql_error="no such column"))
    assert not failed["correct"] and failed["sql_ok"] is False
    assert score_question(item, make_answer("17"))["sql_ok"] is None  # no query attempted


def test_refusal_is_wrong_for_answerable_and_right_for_unanswerable():
    answerable = {"question": "q", "answerable": True, "answer_keywords": []}
    unanswerable = {"question": "q", "answerable": False, "answer_keywords": []}
    assert not score_question(answerable, make_answer(NOT_FOUND))["correct"]
    assert score_question(unanswerable, make_answer(NOT_FOUND))["correct"]
    assert not score_question(unanswerable, make_answer("Riverside, probably."))["correct"]


def test_summary_rates_latency_and_cost():
    base = {"input_tokens": 1000, "output_tokens": 100, "retrieval_hit": None}
    rows = [
        {**base, "answerable": True, "correct": True, "sql_ok": True, "latency_s": 1.0},
        {**base, "answerable": True, "correct": False, "sql_ok": False, "latency_s": 3.0},
        {**base, "answerable": False, "correct": True, "sql_ok": None, "latency_s": 2.0},
    ]
    summary = summarize(rows, price=(1.0, 2.0))
    assert summary["answer_accuracy"] == 0.5
    assert summary["refusal_accuracy"] == 1.0
    assert summary["sql_success_rate"] == 0.5
    assert summary["latency_p50_s"] == 2.0 and summary["latency_p95_s"] == 3.0
    assert summary["cost_per_1000_questions_usd"] == 1.2  # (1000*$1 + 100*$2) per million, x1000
    assert "SQL ok" in format_table({"fake/fake-1": summary})


def test_load_questions_skips_comments_and_sets_defaults(tmp_path):
    path = tmp_path / "q.jsonl"
    path.write_text('// comment\n\n' + json.dumps({"question": "Why?"}) + "\n")
    assert load_questions(str(path)) == [{"question": "Why?", "answerable": True, "answer_keywords": []}]
    assert len(load_questions("eval/questions.example.jsonl")) == 3


def test_make_questions_builds_ground_truth_from_database(loaded):
    questions = make_questions(loaded[0].sports_db, per_category=3)
    by_kind = {}
    for q in questions:
        by_kind.setdefault(q["kind"], []).append(q)
    leader_keywords = [q["answer_keywords"] for q in by_kind["leader"]]
    assert ["Testplayer"] in leader_keywords and ["Fixture"] in leader_keywords
    assert by_kind["game"][0]["answer_keywords"] == ["Northwestern"]  # 24-17 home win; pre-game skipped
    assert all(not q["answerable"] for q in by_kind["unanswerable"])
    assert len({q["id"] for q in questions}) == len(questions)
