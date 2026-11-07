import pytest

from rag.documents import Chunk, chunk_pages, chunk_text, load_pages

FIXTURES = "tests/fixtures/docs"


def test_short_text_is_one_chunk():
    assert chunk_text("Short rule.", 100, 20) == ["Short rule."]


def test_chunks_respect_max_length_and_overlap():
    text = " ".join(f"Sentence number {i} about lacrosse." for i in range(200))
    chunks = chunk_text(text, 300, 60)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)
    # Each chunk after the first starts inside the previous one (the overlap).
    for previous, current in zip(chunks, chunks[1:]):
        assert current[:20] in previous


def test_prefers_sentence_breaks():
    text = "First sentence is here. " * 30
    for chunk in chunk_text(text, 200, 30):
        assert chunk.endswith(".")


def test_whole_text_is_covered():
    words = [f"w{i}" for i in range(500)]
    chunks = chunk_text(" ".join(words), 250, 40)
    covered = set(" ".join(chunks).split())
    assert covered == set(words)


def test_overlap_must_be_smaller_than_chunk():
    with pytest.raises(ValueError):
        chunk_text("text", 50, 50)


def test_load_pages_reads_text_and_markdown():
    pages = load_pages(FIXTURES)
    assert sorted(p.source for p in pages) == ["game_day.txt", "practice_notes.md"]
    assert all(p.page is None for p in pages)


def test_chunk_ids_are_unique():
    chunks = chunk_pages(load_pages(FIXTURES), 200, 40)
    assert len({c.id for c in chunks}) == len(chunks)
    assert Chunk("x", "a.pdf", 3, 1).id == "a.pdf::p3::c1"
