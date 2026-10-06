"""Shared Groq client and helpers for model calls.

Both helpers are wrapped in @traceable so the raw Groq SDK calls show up as LLM
runs in LangSmith, with prompts, tool calls and token counts. traceable is a
no-op unless LANGSMITH_TRACING is true, so nothing is sent when it is off.
"""

import json
import os

from dotenv import load_dotenv
from groq import Groq
from langsmith import traceable
from pydantic import BaseModel, ValidationError

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. Add it to .env.")

# The SDK waits for the retry-after header on 429s. The free tier allows only
# 8k tokens per minute on gpt-oss-120b, so waiting is the normal case, not an error.
client = Groq(max_retries=6)

# Tells LangSmith which model these runs used, so it can price them.
LS_METADATA = {"ls_provider": "groq", "ls_model_name": MODEL}


class JSONCallError(Exception):
    """The model answered, but not with JSON that fits the schema."""


@traceable(run_type="llm", name="groq.chat", metadata=LS_METADATA)
def _chat(messages: list[dict], **kwargs) -> dict:
    """One blocking completion, returned as a plain dict for the trace."""
    return client.chat.completions.create(model=MODEL, messages=messages, **kwargs).model_dump()


def reduce_stream(chunks: list[dict]) -> dict:
    """Rebuild one completion from streamed chunks, for the trace output."""
    content, calls, usage, finish = "", {}, None, None
    for chunk in chunks:
        usage = (chunk.get("x_groq") or {}).get("usage") or chunk.get("usage") or usage
        for choice in chunk.get("choices") or []:
            finish = choice.get("finish_reason") or finish
            delta = choice.get("delta") or {}
            content += delta.get("content") or ""
            for tc in delta.get("tool_calls") or []:
                slot = calls.setdefault(tc.get("index", 0), {"id": "", "type": "function",
                                                             "function": {"name": "", "arguments": ""}})
                slot["id"] = tc.get("id") or slot["id"]
                fn = tc.get("function") or {}
                slot["function"]["name"] += fn.get("name") or ""
                slot["function"]["arguments"] += fn.get("arguments") or ""
    message = {"role": "assistant", "content": content}
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return {"choices": [{"message": message, "finish_reason": finish}], "usage": usage}


@traceable(run_type="llm", name="groq.chat.stream", metadata=LS_METADATA, reduce_fn=reduce_stream)
def stream_chat(messages: list[dict], **kwargs):
    """Yield raw streaming chunks; the trace shows the assembled completion."""
    for chunk in client.chat.completions.create(model=MODEL, messages=messages, stream=True, **kwargs):
        yield chunk.model_dump()


def json_call[T: BaseModel](system: str, user: str, schema: type[T]) -> T:
    """Ask for a JSON object and validate it against a pydantic schema."""
    response = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = response["choices"][0]["message"]["content"] or ""
    try:
        return schema.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise JSONCallError(f"Model output did not match the {schema.__name__} schema: {exc}") from exc


def schema_text(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema())
