# Portfolio Assistant

A chatbot recruiters and hiring managers can talk to about Bhavyy Khurana's technical
background. Every answer is grounded in two documents: the resume and the project
write-ups in `knowledge/`. Attach or paste a job description and it scores the fit
out of 100 and writes a cover-letter style verdict.

## Run

```
uv sync
uv run python -m uvicorn api:app --reload --port 8000     # web app on http://localhost:8000
uv run python main.py                                     # terminal chat, streams too
```

`.env` needs `GROQ_API_KEY`. Set `GROQ_MODEL` to use a model other than `openai/gpt-oss-120b`.
Use `python -m uvicorn` rather than the `uvicorn` launcher: the launcher fails when the
project path contains a space.

## Ground truth

| File | What it holds |
| --- | --- |
| `knowledge/resume.pdf` (or `.docx`, `.md`) | Required. Education, experience, skills, links |
| `knowledge/projects.md` (or `.docx`, `.pdf`) | Detailed project write-ups. Wins over the resume for project detail |
| `profile.json` | Generated profile the bot answers from, plus a content hash of each source |

Edit either source file and the profile rebuilds on the next message, and also at server
start. Nothing rebuilds while the hashes match. The build parses the projects file one
`##` section at a time, and sections that only summarise several projects are skipped.

## Agent graph (LangGraph)

```
START ──(sources changed?)──► build_profile ──► llm
  └──────────────────────────────────────────► llm
llm ──(tool call?)──► build_profile | parse_documents | match_profile | END
parse_documents, match_profile ──(transient failure, retries left)──► same node
                               └──────────────────────────────────► llm
```

| Node | Does |
| --- | --- |
| `llm` | Answers from the profile and streams tokens. Picks one tool at a time |
| `build_profile` | Parses resume and projects into `profile.json` when they change |
| `parse_documents` | Parses a `.pdf`, `.docx` or `.md` upload, or pasted text, into the schema for its type: `job_description`, `resume` or `projects` |
| `match_profile` | Scores the last JD against the profile (0 to 100), returns matched and missing skills and a cover-letter verdict with no resume metrics in it |

**Retries.** Each tool allows 2 retries after its first failure. A transient failure,
such as malformed JSON or a cover letter that quotes resume metrics, loops straight back
into the tool with the reason as feedback. An unclear input, such as a JD with no role or
no criteria, goes back to the model, which tells the user what is missing. After the third
failure the tool is removed from the model's tool list for that chat. The model then says
it can't process the input and points the user to a new chat or to email. A success
resets the counter.

**State** holds the messages, the profile and its version, the latest upload, the
parsed JD, the match result, the retry count per tool, and the last error.

## Files

| File | Role |
| --- | --- |
| `agent.py` | Graph, nodes, routing, system prompt |
| `tools.py` | build_profile and match_profile logic |
| `documents.py` | Text extraction and the three document schemas |
| `llm.py` | Groq client and JSON-mode helper |
| `api.py` | FastAPI: sessions, uploads, Server-Sent Events streaming |
| `index.html` | Chat UI with attachments, streamed answers and the verdict card |
| `legacy/` | The previous ReAct version, kept for reference |

## Groq free tier

`openai/gpt-oss-120b` on the free tier allows 8,000 tokens per minute and 200,000 per day.
A chat turn uses about 5,000 tokens, because the full profile goes into every call. A job
description match uses 3 to 4 calls, and a full profile rebuild uses about 10. The client
waits and retries on per-minute limits. The daily limit is reached after roughly 40 chat
turns.
