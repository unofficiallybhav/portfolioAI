"""
Run:
    uvicorn api:app --reload --port 8000
"""

import json
import logging
import os
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from portfolioAI import JD, AgentState, Match, extract_profile, profile, run_agent

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("portfolio-api")

BASE_DIR = Path(__file__).parent
RESUME_PATH = BASE_DIR / os.getenv("RESUME_FILE", "resume.pdf")
PROFILE_CACHE = BASE_DIR / "profile.json"

CANDIDATE: Optional[profile] = None


def load_candidate() -> profile:
    """Read the cached profile, or build it from the resume and cache it.

    The cache is reused unless the resume is newer, so a restart costs nothing
    and the profile stays stable between runs. Delete profile.json to rebuild.
    """
    if not RESUME_PATH.exists():
        raise FileNotFoundError(
            f"Resume not found at {RESUME_PATH}. This bot needs it to start."
        )

    if PROFILE_CACHE.exists() and PROFILE_CACHE.stat().st_mtime >= RESUME_PATH.stat().st_mtime:
        log.info("Using cached profile from %s", PROFILE_CACHE.name)
        return profile(**json.loads(PROFILE_CACHE.read_text(encoding="utf-8")))

    log.info("Building profile from %s", RESUME_PATH.name)
    candidate = extract_profile(str(RESUME_PATH))
    PROFILE_CACHE.write_text(candidate.model_dump_json(indent=2), encoding="utf-8")
    log.info("Cached profile to %s - edit it directly to correct anything", PROFILE_CACHE.name)
    return candidate


class Session:
    def __init__(self, state: AgentState):
        self.state = state
        self.lock = threading.Lock()


SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()


@asynccontextmanager
async def lifespan(app: FastAPI):
    global CANDIDATE
    # Deliberately not caught: a bot with no knowledge base should not start.
    CANDIDATE = load_candidate()
    log.info("Portfolio bot ready for %s", CANDIDATE.name)
    yield
    SESSIONS.clear()


app = FastAPI(title="Portfolio Bot", version="1.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000"
    ).split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    session_id: Optional[str] = None


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    jd: Optional[JD] = None
    match_result: Optional[Match] = None


class SessionInfo(BaseModel):
    session_id: str
    candidate: profile
    jd: Optional[JD] = None
    match_result: Optional[Match] = None
    turns: int = 0


def new_session() -> str:
    session_id = uuid.uuid4().hex
    state = AgentState(candidate=CANDIDATE, prompt="", history=[])
    with SESSIONS_LOCK:
        SESSIONS[session_id] = Session(state)
    return session_id


def get_session(session_id: str) -> Session:
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, f"Unknown session {session_id!r}")
    return session


@app.get("/health")
def health():
    return {"status": "ok", "candidate": CANDIDATE.name, "active_sessions": len(SESSIONS)}


@app.post("/sessions", response_model=SessionInfo)
def create_session():
    return SessionInfo(session_id=new_session(), candidate=CANDIDATE)


# Sync def on purpose: the Groq SDK blocks, so FastAPI runs this in a worker
# thread instead of stalling the event loop for the whole agent run.
@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    session_id = req.session_id or new_session()
    session = get_session(session_id)

    if not session.lock.acquire(timeout=120):
        raise HTTPException(429, "This session is already handling a request")

    try:
        state = session.state
        state.prompt = req.message
        try:
            answer = run_agent(state)
        except Exception as exc:
            log.exception("Agent run failed")
            raise HTTPException(500, f"Agent error: {exc}")
    finally:
        session.lock.release()

    return ChatResponse(
        session_id=session_id,
        answer=answer or "The agent returned nothing.",
        jd=state.jd,
        match_result=state.match_result,
    )


@app.get("/sessions/{session_id}", response_model=SessionInfo)
def read_session(session_id: str):
    state = get_session(session_id).state
    return SessionInfo(
        session_id=session_id,
        candidate=state.candidate,
        jd=state.jd,
        match_result=state.match_result,
        turns=len(state.history or []) // 2,
    )


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    with SESSIONS_LOCK:
        if SESSIONS.pop(session_id, None) is None:
            raise HTTPException(404, f"Unknown session {session_id!r}")