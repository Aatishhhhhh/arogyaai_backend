"""Arogya AI API entry point with backward-compatible and versioned routes."""

import logging
import os
import sys
import uuid
from typing import Any

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from api.providers import ProviderUnavailableError, new_session_state, run_chat
from api.safety import EMERGENCY_GUIDANCE, has_urgent_signal
from api.settings import get_settings

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("arogya_ai")
settings = get_settings()

app = FastAPI(
    title="Arogya AI API",
    description="API layer for Arogya AI. Medical outputs are informational and require appropriate clinical validation.",
    version="0.2.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "X-Request-ID"],
)

# Prototype session memory only. Replace with an authenticated database/cache before production.
SESSION_STORE: dict[str, dict[str, Any]] = {}
SAFETY_NOTICE = (
    "Arogya AI provides health information, not a diagnosis or a substitute for a qualified clinician. "
    "Do not delay emergency care based on this service."
)


class ChatRequest(BaseModel):
    session_id: str | None = Field(default=None, max_length=100)
    message: str = Field(..., min_length=1, max_length=settings.max_message_chars)


class ChatResponse(BaseModel):
    session_id: str
    stage: str
    confidence_score: float = 0.0
    questions: list[str] | None = None
    clarification_data: dict[str, Any] | None = None
    final_output: dict[str, Any] | None = None
    provider: str
    request_id: str
    safety_notice: str = SAFETY_NOTICE


def _request_id(request: Request) -> str:
    return getattr(request.state, "request_id", "unknown")


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    supplied = request.headers.get("X-Request-ID", "").strip()
    request_id = supplied[:80] if supplied and all(ch.isalnum() or ch in "-_." for ch in supplied[:80]) else str(uuid.uuid4())
    request.state.request_id = request_id
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    # Do not echo submitted health text or Pydantic's raw input values back to the client.
    return JSONResponse(
        status_code=422,
        content={"error": {"code": "validation_error", "message": "Request payload is invalid.", "request_id": _request_id(request)}},
    )


@app.exception_handler(ProviderUnavailableError)
async def provider_exception_handler(request: Request, exc: ProviderUnavailableError):
    logger.warning("AI provider unavailable; request_id=%s reason=%s", _request_id(request), str(exc))
    return JSONResponse(
        status_code=503,
        content={"error": {"code": "ai_provider_unavailable", "message": "The AI service is temporarily unavailable. Please try again shortly.", "request_id": _request_id(request)}},
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    message = exc.detail if isinstance(exc.detail, str) else "The request could not be completed."
    return JSONResponse(
        status_code=exc.status_code,
        headers=exc.headers,
        content={"error": {"code": "http_error", "message": message, "request_id": _request_id(request)}},
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception):
    # Never log the user's message or session payload. request_id correlates server logs.
    logger.exception("Unhandled API error; request_id=%s", _request_id(request))
    return JSONResponse(
        status_code=500,
        content={"error": {"code": "internal_server_error", "message": "An unexpected server error occurred.", "request_id": _request_id(request)}},
    )


@app.get("/health")
@app.get("/v1/health")
def health_check():
    """Liveness endpoint; does not imply the configured model is ready."""
    return {"status": "ok", "service": "arogya-ai-api", "version": app.version}


@app.get("/ready")
@app.get("/v1/ready")
async def readiness_check():
    """Readiness endpoint checks Ollama only when it is the selected provider."""
    if settings.ai_provider == "legacy":
        return {"ready": True, "provider": "legacy"}
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            response = await client.get(f"{settings.ollama_base_url.rstrip('/')}/api/tags")
            response.raise_for_status()
    except (httpx.HTTPError, ValueError):
        raise HTTPException(status_code=503, detail="Configured AI provider is not ready.")
    return {"ready": True, "provider": "ollama", "model": settings.ollama_model}


@app.post("/chat", response_model=ChatResponse)
@app.post("/v1/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, request: Request):
    """Main chat endpoint. /chat remains available for the existing Flutter client."""
    message = req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be empty.")

    session_id = (req.session_id or "").strip() or str(uuid.uuid4())
    if session_id not in SESSION_STORE:
        SESSION_STORE[session_id] = new_session_state()
    state = SESSION_STORE[session_id]

    # A lightweight precautionary guard, not a validated or exhaustive emergency detector.
    if has_urgent_signal(message):
        result = {
            "stage": "urgent_guidance",
            "confidence_score": 0.0,
            "questions": None,
            "clarification_data": None,
            "final_output": {"message": EMERGENCY_GUIDANCE, "type": "precautionary_safety_guidance"},
        }
        provider = "safety_gate"
        updated_state = state
    else:
        try:
            result, updated_state, provider = await run_chat(message, state)
        except ProviderUnavailableError:
            raise
        except Exception:
            logger.exception("Chat processing failed; request_id=%s", _request_id(request))
            raise HTTPException(status_code=500, detail="Unable to process this chat request.")

    SESSION_STORE[session_id] = updated_state
    return ChatResponse(
        session_id=session_id,
        stage=str(result.get("stage", "questions")),
        confidence_score=float(result.get("confidence_score", 0.0) or 0.0),
        questions=result.get("questions"),
        clarification_data=result.get("clarification_data"),
        final_output=result.get("final_output"),
        provider=provider,
        request_id=_request_id(request),
    )
