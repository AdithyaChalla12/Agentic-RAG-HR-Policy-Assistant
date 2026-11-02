"""REST API: chat, ingest, audit, health, and the graph diagram."""
import logging
import time
from pathlib import Path

from fastapi import APIRouter, Depends, File, Header, HTTPException, UploadFile
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.rag import workflow
from app.services import audit
from app.services.ingestion import SUPPORTED, chunk_documents, load_file

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api")
MAX_UPLOAD_MB = 15


class ChatRequest(BaseModel):
    question: str = Field(min_length=2, max_length=3000)


class ChatResponse(BaseModel):
    answer: str
    source_used: str
    confidence: str
    final_query: str
    retries: int
    latency_ms: int
    trace: list[dict]
    citations: list[dict]


def require_admin(x_admin_key: str = Header(default="")):
    if x_admin_key != get_settings().admin_api_key:
        raise HTTPException(status_code=401, detail="Invalid admin key")


@router.get("/health")
def health():
    s = get_settings()
    return {
        "status": "ok",
        "assistant": s.assistant_name,
        "llm": f"{s.llm_provider}:{s.llm_model}",
        "embeddings": f"{s.embedding_provider}:{s.embedding_model}",
        "keys_present": {
            "llm": bool(s.groq_api_key if s.llm_provider == "groq" else s.openai_api_key),
            "pinecone": bool(s.pinecone_api_key),
            "tavily": bool(s.tavily_api_key),
        },
    }


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    start = time.perf_counter()
    try:
        result = workflow.ask(payload.question)
    except Exception as exc:
        logger.exception("Graph failed")
        raise HTTPException(status_code=500, detail=f"Agent error: {exc}") from exc
    latency_ms = int((time.perf_counter() - start) * 1000)
    audit.write_audit(payload.question, result, latency_ms)
    return ChatResponse(
        answer=result["answer"],
        source_used=result["source_used"],
        confidence=result.get("confidence", ""),
        final_query=result.get("current_query", payload.question),
        retries=result.get("retry_count", 0),
        latency_ms=latency_ms,
        trace=result.get("trace", []),
        citations=result.get("citations", []),
    )


@router.post("/ingest", dependencies=[Depends(require_admin)])
async def ingest(file: UploadFile = File(...)):
    from app.rag.vectorstore import upsert_chunks

    name = Path(file.filename or "").name  # strip any path the client sent
    if Path(name).suffix.lower() not in SUPPORTED:
        raise HTTPException(status_code=400, detail=f"Supported types: {', '.join(sorted(SUPPORTED))}")
    data = await file.read()
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(status_code=413, detail=f"Max {MAX_UPLOAD_MB} MB")

    dest = Path(get_settings().upload_dir) / name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(data)

    chunks = chunk_documents(load_file(dest))
    if not chunks:
        raise HTTPException(status_code=422, detail="No extractable text (scanned PDF?)")
    count = upsert_chunks(chunks)
    return {"message": "Indexed", "file": name, "chunks": count}


@router.get("/audit", dependencies=[Depends(require_admin)])
def audit_log(limit: int = 20):
    return {"stats": audit.stats(), "recent": audit.recent(min(limit, 200))}


@router.get("/graph")
def graph_mermaid():
    """Mermaid source of the compiled graph (paste into mermaid.live)."""
    return {"mermaid": workflow.get_graph().get_graph().draw_mermaid()}
