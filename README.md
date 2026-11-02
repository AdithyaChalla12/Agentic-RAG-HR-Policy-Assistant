# Agentic RAG Copilot (People Ops)

An internal assistant that answers from private company documents first, falls back to web search only when they are not enough, retries with a rewritten query, and states its limits when nothing reliable is found. Every answer shows its source, confidence, and the full decision path.

**Stack:** LangGraph · FastAPI · Pinecone · HuggingFace `all-MiniLM-L6-v2` (384-dim) · Groq or OpenAI · Tavily · SQLite audit · vanilla HTML/CSS/JS · Docker

## Flow

```
User -> (2) Router --direct--> Direct answer
            |kb
            v
        (3) Retrieve Pinecone -> (4) Grade KB --good--> (7) Answer from KB
                 ^                    |weak
                 |                    v
                 |              (5) Tavily search -> (6) Grade web --good--> (8) Answer from web
                 |                                        |weak
                 +------ Rewrite query (retries left) ----+
                                                          +-- no retries --> (9) Fallback (states limitation)
```

## Quick start

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements-dev.txt
cp .env.example .env                                  # add GROQ_API_KEY, PINECONE_API_KEY, TAVILY_API_KEY
pytest -q                                             # offline tests, no keys needed
python -m scripts.ingest_kb                           # index data/sample_kb into Pinecone
python run.py                                         # http://127.0.0.1:8000  (API docs at /docs)
python -m scripts.eval                                # golden-set check against the live graph
```

## Project layout

```
app/
  core/config.py        settings from .env
  services/ingestion.py load + structure-aware chunking + deterministic ids
  rag/vectorstore.py    embeddings, Pinecone index, search, upsert
  rag/state.py          graph state + structured LLM outputs
  rag/prompts.py        every prompt in one file
  rag/llm.py            LLM + Tavily clients
  rag/workflow.py       the LangGraph agent (nodes numbered like the diagram)
  services/audit.py     SQLite audit log + stats
  api/routes.py         /api/chat, /api/ingest, /api/audit, /api/health, /api/graph
  main.py               FastAPI app
templates/ static/      UI
scripts/                ingest_kb.py, eval.py
tests/                  offline tests with a scripted fake LLM
```

## API

| Method | Path | Auth | Purpose |
|---|---|---|---|
| POST | `/api/chat` | none | `{"question": "..."}` -> answer, source_used, confidence, trace, citations |
| POST | `/api/ingest` | `X-Admin-Key` | upload .pdf/.md/.txt/.docx; replaces older chunks of the same file |
| GET | `/api/audit` | `X-Admin-Key` | answer counts by source + recent traces |
| GET | `/api/graph` | none | Mermaid source of the compiled graph |
| GET | `/api/health` | none | config + which keys are present |

## Adapt to another domain

1. Replace the files in `data/sample_kb/`.
2. Set `ASSISTANT_NAME` and `DOMAIN_DESCRIPTION` in `.env` (the prompts read them).
3. Update `CASES` in `scripts/eval.py` and the suggestion chips in `templates/index.html`.
