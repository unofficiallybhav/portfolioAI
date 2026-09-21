"""
HTTP layer for the portfolio chatbot.

Run:
    uv run uvicorn api:app --reload --port 8000

POST /chat takes multipart form data (message, session_id, optional file) and
answers with a Server-Sent Events stream:
    event: status  {"text": "Parsing the job description"}
    event: token   {"text": "..."}
    event: done    {"jd": ..., "match": ..., "retries": ...}
    event: error   {"message": "..."}
"""

import json
import logging
import threading
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse

from agent import GRAPH, MAX_RETRIES, initial_state
from documents import SUPPORTED_SUFFIXES, extract_text
from tools import build_profile

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("portfolio-api")

BASE_DIR = Path(__file__).parent
MAX_UPLOAD_BYTES = 5 * 1024 * 1024
MAX_MESSAGE_CHARS = 8000
SESSION_TTL_SECONDS = 2 * 60 * 60
MAX_SESSIONS = 500


class Session:
    def __init__(self):
        self.lock = threading.Lock()
        self.last_used = time.monotonic()
        self.fresh = True  # the graph thread has no state yet


SESSIONS: dict[str, Session] = {}
SESSIONS_LOCK = threading.Lock()


def _drop(session_id: str) -> None:
    SESSIONS.pop(session_id, None)
    GRAPH.checkpointer.delete_thread(session_id)


def prune_sessions() -> None:
    cutoff = time.monotonic() - SESSION_TTL_SECONDS
    with SESSIONS_LOCK:
        for sid in [s for s, sess in SESSIONS.items() if sess.last_used < cutoff]:
            _drop(sid)


def new_session() -> str:
    prune_sessions()
    with SESSIONS_LOCK:
        if len(SESSIONS) >= MAX_SESSIONS:
            raise HTTPException(503, "Too many active chats right now. Please try again later.")
        session_id = uuid.uuid4().hex
        SESSIONS[session_id] = Session()
    return session_id


def get_session(session_id: str) -> Session:
    with SESSIONS_LOCK:
        session = SESSIONS.get(session_id)
    if session is None:
        raise HTTPException(404, "This chat has expired. Start a new one.")
    session.last_used = time.monotonic()
    return session


def config(session_id: str) -> dict:
    return {"configurable": {"thread_id": session_id}, "recursion_limit": 30}


def profile_summary() -> dict:
    profile, version, _ = build_profile()
    return {
        "name": profile["name"],
        "links": profile.get("links", []),
        "skills": sorted({s for group in profile.get("skills", {}).values() for s in group}),
        "projects": [{"name": p["name"], "status": p["status"]} for p in profile.get("projects", [])],
        "version": version,
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Build (or load) the profile before taking traffic. Deliberately not caught:
    # a portfolio bot with no knowledge base should not start.
    profile, _, rebuilt = build_profile()
    log.info("Profile %s for %s", "rebuilt" if rebuilt else "loaded from cache", profile["name"])
    yield
    with SESSIONS_LOCK:
        for sid in list(SESSIONS):
            _drop(sid)


app = FastAPI(title="Portfolio Bot", version="2.0", lifespan=lifespan)


@app.get("/", include_in_schema=False)
def frontend():
    return FileResponse(BASE_DIR / "index.html")


@app.get("/health")
def health():
    return {"status": "ok", "active_sessions": len(SESSIONS)}


@app.post("/sessions")
def create_session():
    return {"session_id": new_session(), "candidate": profile_summary()}


@app.get("/sessions/{session_id}")
def read_session(session_id: str):
    get_session(session_id)
    values = GRAPH.get_state(config(session_id)).values
    return {
        "session_id": session_id,
        "jd": values.get("jd"),
        "match": values.get("match"),
        "retries": values.get("retries"),
        "turns": sum(1 for m in values.get("messages", []) if m["role"] == "user"),
    }


@app.delete("/sessions/{session_id}", status_code=204)
def delete_session(session_id: str):
    with SESSIONS_LOCK:
        if session_id not in SESSIONS:
            raise HTTPException(404, "Unknown session")
        _drop(session_id)


def sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


# Sync handler on purpose: the Groq SDK blocks, so FastAPI runs this and the
# streaming generator in worker threads instead of stalling the event loop.
@app.post("/chat")
def chat(
    session_id: str = Form(...),
    message: str = Form(""),
    file: UploadFile | None = File(None),
):
    session = get_session(session_id)
    message = message.strip()[:MAX_MESSAGE_CHARS]

    inputs: dict = {}
    if file is not None and file.filename:
        if Path(file.filename).suffix.lower() not in SUPPORTED_SUFFIXES:
            raise HTTPException(400, f"Only {', '.join(SUPPORTED_SUFFIXES)} files are supported.")
        data = file.file.read(MAX_UPLOAD_BYTES + 1)
        if len(data) > MAX_UPLOAD_BYTES:
            raise HTTPException(413, "That file is larger than 5 MB.")
        try:
            text = extract_text(file.filename, data)
        except Exception as exc:
            log.warning("Could not read upload %s: %s", file.filename, exc)
            raise HTTPException(400, f"Could not read {file.filename}. Is it a valid, text-based file?")
        inputs["upload"] = {"filename": file.filename, "text": text}
        message = f"{message or 'Here is a job description.'}\n\n[Attached file: {file.filename}]"

    if not message:
        raise HTTPException(400, "Type a message or attach a file.")
    inputs["messages"] = [{"role": "user", "content": message}]

    def events():
        # Locked inside the generator so an abandoned request can never hold it.
        if not session.lock.acquire(blocking=False):
            yield sse("error", {"message": "This chat is still answering the previous message."})
            return
        try:
            run_inputs = {**initial_state(), **inputs} if session.fresh else inputs
            session.fresh = False
            for event in GRAPH.stream(run_inputs, config(session_id), stream_mode="custom"):
                yield sse(event["type"], event)
            values = GRAPH.get_state(config(session_id)).values
            yield sse("done", {
                "jd": values.get("jd"),
                "match": values.get("match"),
                "retries_left": {
                    tool: max(0, MAX_RETRIES + 1 - used)
                    for tool, used in (values.get("retries") or {}).items()
                },
            })
        except Exception:
            log.exception("Agent run failed")
            yield sse("error", {"message": "Something went wrong on the server. Please try again."})
        finally:
            session.lock.release()
            session.last_used = time.monotonic()

    return StreamingResponse(events(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
