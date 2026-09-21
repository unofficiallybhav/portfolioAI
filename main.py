"""Terminal chat with the portfolio agent, streaming answers as they arrive.

    uv run python main.py
    > /attach path/to/job_description.pdf   optional message
    > /state                                show jd, match and retries
"""

import json
import shlex
import sys
import uuid
from pathlib import Path

from agent import GRAPH, initial_state
from documents import extract_text


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    config = {"configurable": {"thread_id": uuid.uuid4().hex}, "recursion_limit": 30}
    first = True
    print("Portfolio agent. /attach <file> [message], /state, Ctrl-C to quit.\n")

    while True:
        try:
            line = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not line:
            continue

        if line == "/state":
            values = GRAPH.get_state(config).values
            print(json.dumps({k: values.get(k) for k in ("jd", "match", "retries", "last_error")}, indent=2))
            continue

        inputs: dict = {}
        if line.startswith("/attach "):
            parts = shlex.split(line[len("/attach "):], posix=False)
            path = Path(parts[0].strip('"'))
            try:
                inputs["upload"] = {"filename": path.name, "text": extract_text(path.name, path.read_bytes())}
            except (OSError, ValueError) as exc:
                print(f"Could not read {path}: {exc}")
                continue
            line = f"{' '.join(parts[1:]) or 'Here is a job description.'}\n\n[Attached file: {path.name}]"

        inputs["messages"] = [{"role": "user", "content": line}]
        if first:
            inputs = {**initial_state(), **inputs}
            first = False

        for event in GRAPH.stream(inputs, config, stream_mode="custom"):
            if event["type"] == "token":
                print(event["text"], end="", flush=True)
            elif event["type"] == "status":
                print(f"[{event['text']}]", flush=True)
        print("\n")


if __name__ == "__main__":
    main()
