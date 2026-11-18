"""Gemini falls back to another model when the first one is busy."""
import types as pytypes

import pytest

genai_errors = pytest.importorskip("google.genai.errors")
from rag.llm import GeminiLLM


class FakeModels:
    def __init__(self, busy):
        self.busy, self.calls = busy, []

    def generate_content(self, model, contents, config):
        self.calls.append(model)
        self.last_config, self.last_contents = config, contents
        if model in self.busy:
            raise genai_errors.ServerError(503, {"error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}})
        usage = pytypes.SimpleNamespace(prompt_token_count=3, candidates_token_count=2)
        return pytypes.SimpleNamespace(text=f"hi from {model}", usage_metadata=usage)

    def list(self):
        return [pytypes.SimpleNamespace(name="models/flash-b", supported_actions=["generateContent"]),
                pytypes.SimpleNamespace(name="models/pro-x", supported_actions=["generateContent"])]


def make(busy, monkeypatch, fallbacks=""):
    monkeypatch.setenv("RAG_FALLBACK_MODELS", fallbacks)
    llm = GeminiLLM.__new__(GeminiLLM)
    llm.model, llm._max = "flash-a", 50
    llm._client = pytypes.SimpleNamespace(models=FakeModels(busy))
    return llm


def test_busy_model_falls_back_to_default_list(monkeypatch):
    llm = make({"flash-a", "gemini-3.1-flash-lite"}, monkeypatch)
    assert llm.complete("sys", "q").text == "hi from gemma-4-31b-it"
    assert llm.model == "gemma-4-31b-it"
    sent = llm._client.models.last_config
    assert sent.system_instruction is None and "sys" in llm._client.models.last_contents


def test_configured_fallbacks_are_used(monkeypatch):
    llm = make({"flash-a"}, monkeypatch, "my-backup")
    assert llm.complete("sys", "q").text == "hi from my-backup"


def test_all_busy_raises(monkeypatch):
    llm = make({"flash-a", "x"}, monkeypatch, "x")
    with pytest.raises(genai_errors.ServerError):
        llm.complete("sys", "q")
