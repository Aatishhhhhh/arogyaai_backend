"""Provider routing for the rule-based engine and a local Ollama model."""

from typing import Any

import httpx
from fastapi.concurrency import run_in_threadpool

from api.settings import get_settings
from engine.pipeline import pipeline


class ProviderUnavailableError(RuntimeError):
    """The selected AI provider could not be reached or did not respond correctly."""


SYSTEM_PROMPT = """You are Arogya AI, a health-information assistant.

Safety rules:
- You provide general health information, not a definitive medical diagnosis.
- Do not claim certainty or invent clinical evidence, citations, test results, or user history.
- Do not tell users to start, stop, or change prescription medicines or provide individualized dosing.
- Ask concise follow-up questions when important information is missing.
- For possible emergencies, advise the user to seek emergency medical care immediately.
- Do not claim to calculate a validated health score unless a validated scoring method and required inputs are explicitly provided.
- Protect privacy: ask only for health details needed to respond.
- Reply in the language used by the user (Hindi, English, or Hinglish).
- Encourage qualified medical assessment when symptoms are persistent, severe, worsening, or unclear.
"""


def _initial_state() -> dict[str, Any]:
    return {
        "history": [],
        "scores": {},
        "asked_questions": [],
        "ask_count": 0,
        "current_focus": None,
        "clarification_mode": False,
        "clarification_data": None,
        "clarification_answer": None,
        "clarification_complete": False,
        "ai_messages": [],
    }


async def run_chat(message: str, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], str]:
    """Route each chat request to the configured provider.

    The legacy provider is the existing rule-based pipeline and remains the default.
    Ollama sends a non-streaming request to the local Ollama HTTP API.
    """
    settings = get_settings()
    state.setdefault("ai_messages", [])

    if settings.ai_provider == "legacy":
        state.setdefault("history", []).append(message)
        result, updated_state = await run_in_threadpool(pipeline, message, state)
        updated_state.setdefault("ai_messages", [])
        return result, updated_state, "legacy"

    if settings.ai_provider == "ollama":
        result, updated_state = await _run_ollama(message, state)
        return result, updated_state, "ollama"

    raise ProviderUnavailableError("The configured AI provider is not supported.")


async def _run_ollama(message: str, state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    settings = get_settings()
    base_url = settings.ollama_base_url.rstrip("/")
    history: list[dict[str, str]] = state.setdefault("ai_messages", [])
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history[-10:], {"role": "user", "content": message}]
    payload = {
        "model": settings.ollama_model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": 0.2},
    }

    try:
        async with httpx.AsyncClient(timeout=settings.ollama_timeout_seconds) as client:
            response = await client.post(f"{base_url}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
    except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as exc:
        raise ProviderUnavailableError("The Ollama provider is unreachable or timed out.") from exc
    except httpx.HTTPStatusError as exc:
        raise ProviderUnavailableError(f"Ollama returned HTTP {exc.response.status_code}.") from exc
    except (ValueError, httpx.InvalidURL) as exc:
        raise ProviderUnavailableError("Ollama returned an invalid response.") from exc

    answer = data.get("message", {}).get("content") if isinstance(data, dict) else None
    if not isinstance(answer, str) or not answer.strip():
        raise ProviderUnavailableError("Ollama returned an empty or malformed assistant response.")

    history.extend([
        {"role": "user", "content": message},
        {"role": "assistant", "content": answer.strip()},
    ])
    # Bound in-memory conversation context. This is a prototype store, not durable history.
    if len(history) > 12:
        del history[:-12]

    state["ai_messages"] = history
    result = {
        "stage": "assistant_response",
        # No clinically calibrated confidence is available from this provider.
        "confidence_score": 0.0,
        "questions": None,
        "clarification_data": None,
        "final_output": {
            "message": answer.strip(),
            "provider": "ollama",
            "model": settings.ollama_model,
        },
    }
    return result, state


def new_session_state() -> dict[str, Any]:
    """Create a fresh session state compatible with the current pipeline."""
    return _initial_state()
