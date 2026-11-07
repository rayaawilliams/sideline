"""Swappable language models: Gemini, Claude, or a local open-source model via Ollama.

Every provider returns the same Completion, so the pipeline and the evaluation
never need to know which one is running.
"""
import json
import os
import urllib.request
from dataclasses import dataclass
from typing import Callable, Protocol


@dataclass(frozen=True)
class Completion:
    text: str
    input_tokens: int | None = None
    output_tokens: int | None = None


class LLM(Protocol):
    provider: str
    model: str

    def complete(self, system: str, prompt: str) -> Completion: ...


class GeminiLLM:
    """Google Gemini through the google-genai SDK. Needs GEMINI_API_KEY."""

    provider = "gemini"

    def __init__(self, model: str, max_output_tokens: int):
        from google import genai

        self.model = model
        self._max = max_output_tokens
        self._client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def complete(self, system: str, prompt: str) -> Completion:
        from google.genai import types

        response = self._client.models.generate_content(
            model=self.model,
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system, temperature=0, max_output_tokens=self._max,
            ),
        )
        usage = response.usage_metadata
        return Completion(
            response.text or "",
            getattr(usage, "prompt_token_count", None),
            getattr(usage, "candidates_token_count", None),
        )


class ClaudeLLM:
    """Anthropic Claude through the anthropic SDK. Needs ANTHROPIC_API_KEY."""

    provider = "claude"

    def __init__(self, model: str, max_output_tokens: int):
        import anthropic

        self.model = model
        self._max = max_output_tokens
        self._client = anthropic.Anthropic()

    def complete(self, system: str, prompt: str) -> Completion:
        message = self._client.messages.create(
            model=self.model,
            max_tokens=self._max,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(block.text for block in message.content if block.type == "text")
        return Completion(text, message.usage.input_tokens, message.usage.output_tokens)


class OllamaLLM:
    """A free open-source model running on your own computer (https://ollama.com)."""

    provider = "ollama"

    def __init__(self, model: str, max_output_tokens: int, host: str | None = None):
        self.model = model
        self._max = max_output_tokens
        self._host = host or os.getenv("OLLAMA_HOST", "http://localhost:11434")

    def complete(self, system: str, prompt: str) -> Completion:
        body = json.dumps({
            "model": self.model,
            "stream": False,
            "options": {"temperature": 0, "num_predict": self._max},
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        }).encode()
        request = urllib.request.Request(
            f"{self._host}/api/chat", data=body, headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            data = json.load(response)
        return Completion(
            data["message"]["content"], data.get("prompt_eval_count"), data.get("eval_count"),
        )


class FakeLLM:
    """For tests: answers with a function of the prompt, no network."""

    provider = "fake"

    def __init__(self, respond: Callable[[str, str], str], model: str = "fake-1"):
        self.model = model
        self._respond = respond

    def complete(self, system: str, prompt: str) -> Completion:
        text = self._respond(system, prompt)
        return Completion(text, len(prompt.split()), len(text.split()))


def make_llm(settings) -> LLM:
    makers = {"gemini": GeminiLLM, "claude": ClaudeLLM, "ollama": OllamaLLM}
    if settings.provider not in makers:
        raise ValueError(f"Unknown provider '{settings.provider}'. Choose from {sorted(makers)}.")
    return makers[settings.provider](settings.model_name, settings.max_output_tokens)
