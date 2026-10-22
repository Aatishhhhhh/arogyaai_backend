import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional
# PROJECT ROOT
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )
# THIRD-PARTY IMPORTS
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
# PROJECT IMPORTS
from engine.pipeline import pipeline
# PROJECT ROOT
PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        ".."
    )
)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(
        0,
        PROJECT_ROOT
    )
# LOGGING
logging.basicConfig(
    level=logging.INFO
)
logger = logging.getLogger(
    "arogya_ai"
)
# FASTAPI APP
app = FastAPI(
    title="Arogya AI Backend",
    description="Prototype backend for Arogya AI",
    version="0.1.0"
)

# Allow the deployed Arogya AI frontend to call the legacy
# keyword-matching API. Override/extend origins through FRONTEND_ORIGINS.
configured_origins = os.getenv("FRONTEND_ORIGINS", "")
origins = [
    origin.strip().rstrip("/")
    for origin in configured_origins.split(",")
    if origin.strip()
]
origins.extend([
    "https://arogya-ai-frontend-nine.vercel.app",
    "http://localhost:5500",
    "http://127.0.0.1:5500",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
])
origins = list(dict.fromkeys(origins))

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
)
# IN-MEMORY SESSION STORE
#
# Prototype only.
#
# Later this will move to a real database / cache.

SESSION_STORE: Dict[
    str,
    Dict[str, Any]
] = {}
# REQUEST MODEL
class ChatRequest(BaseModel):
    session_id: Optional[str] = Field(
        default=None,
        max_length=100
    )
    message: str = Field(
        ...,
        min_length=1,
        max_length=500
    )
# RESPONSE MODEL
class ChatResponse(BaseModel):
    session_id: str
    stage: str
    confidence_score: float
    questions: Optional[
        List[str]
    ] = None
    clarification_data: Optional[
        Dict[str, Any]
    ] = None
    final_output: Optional[
        Dict[str, Any]
    ] = None
# HEALTH CHECK
@app.get("/health")
def health_check():
    """
    Basic health endpoint.
    Used to check whether the FastAPI server is running.
    """
    return {
        "status": "ok",
        "service": "arogya-ai-backend",
        "version": "0.1.0"
    }
# CHAT ENDPOINT
@app.post(
    "/chat",
    response_model=ChatResponse
)
def chat(req: ChatRequest):
    """
    Main chat endpoint.
    Flow:
        Flutter
            ↓
        POST /chat
            ↓
        session lookup
            ↓
        pipeline()
            ↓
        response
    """
    # CLEAN USER MESSAGE
    message = req.message.strip()
    if not message:
        raise HTTPException(
            status_code=400,
            detail="Message cannot be empty."
        )
    # SESSION ID
    session_id = (
        req.session_id.strip()
        if req.session_id
        else ""
    )
    if not session_id:
        session_id = str(
            uuid.uuid4()
        )
    # GET OR CREATE SESSION
    if session_id not in SESSION_STORE:
        SESSION_STORE[
            session_id
        ] = {
            "history": [],
            "scores": {},
            "asked_questions": [],
            "ask_count": 0,
            "current_focus": None,
            # Clarification state
            "clarification_mode": False,
            "clarification_data": None,
            "clarification_answer": None,
            "clarification_complete": False
        }
    state = SESSION_STORE[
        session_id
    ]
    # ADD MESSAGE TO HISTORY
    #
    # pipeline.py already contains a duplicate guard,
    # so this remains compatible with direct pipeline usage.
    state["history"].append(
        message
    )
    # RUN ENGINE
    try:
        result, updated_state = pipeline(
            message,
            state
        )
    except HTTPException:
        # Preserve intentional HTTP errors.
        raise
    except Exception:
        logger.exception(
            "Unexpected error while processing chat request. session_id=%s",
            session_id
        )
        raise HTTPException(
            status_code=500,
            detail="Internal server error."
        )
    # SAVE UPDATED STATE
    SESSION_STORE[
        session_id
    ] = updated_state
    # BUILD RESPONSE
    return ChatResponse(
        session_id=session_id,
        stage=result.get(
            "stage",
            "questions"
        ),
        confidence_score=float(
            result.get(
                "confidence_score",
                0.0
            )
        ),
        questions=result.get(
            "questions"
        ),
        clarification_data=result.get(
            "clarification_data"
        ),
        final_output=result.get(
            "final_output"
        )
    )