"""Ask one question from the command line.

Usage: python -m rag.ask "Who leads FBS in rushing touchdowns?" [--provider claude]
"""
import argparse

from sports.assistant import build_assistant

from .config import load_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question")
    parser.add_argument("--provider", help="gemini, claude or ollama")
    parser.add_argument("--model")
    parser.add_argument("--show-work", action="store_true", help="print the SQL query and news search")
    args = parser.parse_args()

    settings = load_settings(provider=args.provider, model=args.model)
    answer = build_assistant(settings).answer(args.question)
    print(answer.text)
    if args.show_work:
        print(f"\nSQL: {answer.sql}\nRows: {len(answer.sql_rows)}" + (f"\nSQL error: {answer.sql_error}" if answer.sql_error else ""))
        print(f"News search: {answer.search}")
        for number, source in enumerate(answer.sources, start=1):
            print(f"  [{number}] {source['source']} (similarity {source['score']})")
    print(f"\n{answer.provider}/{answer.model} · {answer.latency_s:.2f}s")


if __name__ == "__main__":
    main()
