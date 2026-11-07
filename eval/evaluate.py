"""Measure how well Sideline answers, and compare model providers.

Usage:
  python -m eval.make_questions                  # build a question set from the stats database
  python -m eval.evaluate --providers gemini claude ollama

For each question it records:
  - answer correct: does the answer contain every expected keyword (or an accepted alternative)?
  - refusal: for questions the data can't answer, did the model say so instead of guessing?
  - SQL success: when the model wrote a stats query, did it run without errors?
  - retrieval hit: for news questions with an expected source, was that article retrieved?
  - latency and tokens, plus cost if prices are set in rag/config.py
Results are printed as a table and saved to eval/results/.
"""
import argparse
import json
import statistics
from datetime import datetime
from pathlib import Path


def load_questions(path: str) -> list:
    questions = []
    for line_number, line in enumerate(Path(path).read_text().splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        item = json.loads(line)
        if "question" not in item:
            raise ValueError(f"{path} line {line_number}: missing 'question'")
        item.setdefault("answerable", True)
        item.setdefault("answer_keywords", [])
        questions.append(item)
    return questions


def keyword_match(answer: str, keywords: list) -> bool:
    """Every keyword must appear (case-insensitive). A keyword may be a list of alternatives."""
    text = answer.lower()
    for keyword in keywords:
        options = keyword if isinstance(keyword, list) else [keyword]
        if not any(str(option).lower() in text for option in options):
            return False
    return True


def retrieval_hit(sources: list, expected_source: str | None) -> bool | None:
    if not expected_source:
        return None
    return any(source["source"] == expected_source for source in sources)


def score_question(item: dict, answer) -> dict:
    """Score one answer. `answer` is a sports.assistant.Answer."""
    answerable = item["answerable"]
    if answerable:
        correct = keyword_match(answer.text, item["answer_keywords"]) and not answer.is_refusal
    else:
        correct = answer.is_refusal  # the right behaviour is to say the data doesn't cover it
    return {
        "id": item.get("id"),
        "kind": item.get("kind"),
        "question": item["question"],
        "answerable": answerable,
        "correct": correct,
        "sql": answer.sql,
        "sql_ok": (answer.sql_error is None) if answer.sql else None,
        "retrieval_hit": retrieval_hit(answer.sources, item.get("source")) if answerable else None,
        "answer": answer.text,
        "latency_s": round(answer.latency_s, 3),
        "input_tokens": answer.input_tokens,
        "output_tokens": answer.output_tokens,
    }


def _rate(values: list) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def _percentile(values: list, fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(fraction * (len(ordered) - 1)))]


def summarize(rows: list, price: tuple | None = None) -> dict:
    answerable = [r for r in rows if r["answerable"]]
    unanswerable = [r for r in rows if not r["answerable"]]
    latencies = [r["latency_s"] for r in rows]
    tokens_in = [r["input_tokens"] for r in rows if r["input_tokens"] is not None]
    tokens_out = [r["output_tokens"] for r in rows if r["output_tokens"] is not None]
    summary = {
        "questions": len(rows),
        "answer_accuracy": _rate([r["correct"] for r in answerable]),
        "refusal_accuracy": _rate([r["correct"] for r in unanswerable]),
        "sql_success_rate": _rate([r["sql_ok"] for r in rows]),
        "retrieval_hit_rate": _rate([r["retrieval_hit"] for r in answerable]),
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_p95_s": _percentile(latencies, 0.95),
        "mean_input_tokens": statistics.mean(tokens_in) if tokens_in else None,
        "mean_output_tokens": statistics.mean(tokens_out) if tokens_out else None,
    }
    if price and tokens_in and tokens_out:
        per_question = (statistics.mean(tokens_in) * price[0] + statistics.mean(tokens_out) * price[1]) / 1e6
        summary["cost_per_1000_questions_usd"] = round(per_question * 1000, 4)
    return summary


def run(assistant, questions: list) -> list:
    return [score_question(item, assistant.answer(item["question"])) for item in questions]


def format_table(results: dict) -> str:
    columns = [
        ("answer_accuracy", "answer acc", "{:.0%}"),
        ("refusal_accuracy", "refusal acc", "{:.0%}"),
        ("sql_success_rate", "SQL ok", "{:.0%}"),
        ("retrieval_hit_rate", "news hit", "{:.0%}"),
        ("latency_p50_s", "p50 latency", "{:.2f}s"),
        ("latency_p95_s", "p95 latency", "{:.2f}s"),
        ("cost_per_1000_questions_usd", "$ / 1k q", "${:.3f}"),
    ]
    header = f"{'model':<30}" + "".join(f"{label:>13}" for _, label, _ in columns)
    lines = [header, "-" * len(header)]
    for name, summary in results.items():
        cells = []
        for key, _, fmt in columns:
            value = summary.get(key)
            cells.append(f"{fmt.format(value) if value is not None else '-':>13}")
        lines.append(f"{name:<30}" + "".join(cells))
    return "\n".join(lines)


def main() -> None:
    from rag.config import load_settings
    from rag.embeddings import make_embedder
    from sports.assistant import build_assistant

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--providers", nargs="+", default=None, help="default: RAG_PROVIDER or gemini")
    parser.add_argument("--questions", default="eval/questions.jsonl")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    base = load_settings()
    embedder = make_embedder(base)  # load once, share across providers
    results, details = {}, {}
    for provider in args.providers or [base.provider]:
        settings = load_settings(provider=provider)
        assistant = build_assistant(settings, embedder=embedder)
        name = f"{provider}/{settings.model_name}"
        print(f"Running {len(questions)} questions with {name} ...")
        rows = run(assistant, questions)
        results[name] = summarize(rows, settings.prices_per_million.get(provider))
        details[name] = rows

    print()
    print(format_table(results))
    out_dir = Path("eval/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{datetime.now():%Y%m%d-%H%M%S}.json"
    out_path.write_text(json.dumps(
        {"questions": args.questions, "summary": results, "details": details}, indent=2, default=str,
    ))
    wrong = [(name, r) for name, rows in details.items() for r in rows if not r["correct"]]
    if wrong:
        print(f"\n{len(wrong)} wrong answers (full text and SQL in {out_path}):")
        for name, row in wrong[:10]:
            print(f"  [{name}] {row['id']}: {row['question']}")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
