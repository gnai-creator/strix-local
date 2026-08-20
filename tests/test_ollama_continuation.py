"""Compatibility for Ollama templates that require a trailing user query."""

from __future__ import annotations

from typing import Any

import pytest
from agents.models.interface import Model

from strix.config import loader
from strix.config.loader import load_settings
from strix.config.models import (
    StrixProvider,
    _ollama_continuation_input,
    _OllamaContinuationModel,
    _TurnGuardModel,
)


def _user(text: str) -> dict[str, Any]:
    return {
        "type": "message",
        "role": "user",
        "content": [{"type": "input_text", "text": text}],
    }


def _tool_output() -> dict[str, Any]:
    return {"type": "function_call_output", "call_id": "call_1", "output": "ok"}


def test_adds_user_continuation_after_tool_output() -> None:
    original = [_user("start"), _tool_output()]

    normalized = _ollama_continuation_input(original)

    assert normalized is not original
    assert normalized[:-1] == original
    assert normalized[-1] == _user("Continue from the tool results above.")


@pytest.mark.parametrize(
    "original",
    [
        "start",
        [_user("start")],
        [_user("start"), _tool_output(), _user("next task")],
    ],
)
def test_leaves_ordinary_inputs_unchanged(original: Any) -> None:
    assert _ollama_continuation_input(original) is original


class _DummyModel(Model):
    async def get_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def stream_response(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError


@pytest.fixture
def _reset_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(loader, "_cached", None)
    monkeypatch.setattr(loader, "_override", None)
    monkeypatch.delenv("STRIX_LLM", raising=False)
    load_settings()


def test_provider_wraps_only_ollama_models(
    monkeypatch: pytest.MonkeyPatch, _reset_settings: None
) -> None:
    monkeypatch.setattr("strix.config.models.MultiProvider.get_model", lambda *_: _DummyModel())
    provider = StrixProvider()

    ollama = provider.get_model("ollama/qwen3.8:27b")
    openai = provider.get_model("openai/gpt-5.4")

    assert isinstance(ollama, _TurnGuardModel)
    assert isinstance(ollama._inner, _OllamaContinuationModel)
    assert isinstance(openai, _TurnGuardModel)
    assert not isinstance(openai._inner, _OllamaContinuationModel)
