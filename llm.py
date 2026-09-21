"""Shared Groq client and a helper for JSON-only model calls."""

import json
import os

from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError

load_dotenv()

MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")

if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("GROQ_API_KEY is not set. Add it to .env.")

# The SDK waits for the retry-after header on 429s. The free tier allows only
# 8k tokens per minute on gpt-oss-120b, so waiting is the normal case, not an error.
client = Groq(max_retries=6)


class JSONCallError(Exception):
    """The model answered, but not with JSON that fits the schema."""


def json_call[T: BaseModel](system: str, user: str, schema: type[T]) -> T:
    """Ask for a JSON object and validate it against a pydantic schema."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        response_format={"type": "json_object"},
        temperature=0.2,
    )
    raw = response.choices[0].message.content or ""
    try:
        return schema.model_validate(json.loads(raw))
    except (json.JSONDecodeError, ValidationError) as exc:
        raise JSONCallError(f"Model output did not match the {schema.__name__} schema: {exc}") from exc


def schema_text(schema: type[BaseModel]) -> str:
    return json.dumps(schema.model_json_schema())
