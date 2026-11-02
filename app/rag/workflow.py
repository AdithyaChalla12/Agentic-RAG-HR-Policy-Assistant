"""The agentic RAG graph. Node numbers match the architecture diagram.

  (2) router ──kb──▶ (3) retrieve_kb ─▶ (4) grade_kb ──good──▶ (7) generate_from_kb ─▶ END
     │                      ▲                 │weak
     │direct                │                 ▼
     ▼                      │           (5) web_search ─▶ (6) grade_web ──good──▶ (8) generate_from_web ─▶ END
  direct_answer ─▶ END      │                                    │weak
                            └──── rewrite_query ◀── retries left ┤
                                                                 └── no retries ─▶ (9) fallback_answer ─▶ END

Dependencies (llm, retriever, web search) are injected so tests can run the
whole graph offline with fakes.
"""
import logging
from dataclasses import dataclass
from functools import lru_cache
from typing import Callable, Literal

from langchain_core.documents import Document
from langchain_core.language_models import BaseChatModel
from langgraph.graph import END, START, StateGraph

from app.core.config import Settings, get_settings
from app.rag import prompts
from app.rag.state import AgentState, EvidenceGrade, RewrittenQuery, RouteDecision

logger = logging.getLogger(__name__)


@dataclass
class Deps:
    llm: BaseChatModel
    retrieve: Callable[[str], list[Document]]
    web_search: Callable[[str], list[dict]]
    settings: Settings


def step(node: str, detail: str) -> dict:
    """One trace entry. The reducer in AgentState appends it."""
    logger.info("%s | %s", node, detail)
    return {"trace": [{"node": node, "detail": detail}]}


def format_kb(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[{i}] {d.metadata.get('source', 'unknown')}"
        f"{' > ' + d.metadata['section'] if d.metadata.get('section') else ''}\n{d.page_content}"
        for i, d in enumerate(docs, 1)
    )


def format_web(results: list[dict]) -> str:
    return "\n\n".join(
        f"[{i}] {r['title']} ({r['url']})\n{r['content']}" for i, r in enumerate(results, 1)
    )


def build_graph(deps: Deps):
    s = deps.settings
    ctx = {"assistant": s.assistant_name, "domain": s.domain_description}

    def structured(schema, prompt: str):
        return deps.llm.with_structured_output(schema).invoke(prompt)

    def text(prompt: str) -> str:
        return deps.llm.invoke(prompt).content.strip()

    # ---------- (2) Router ----------
    def router(state: AgentState):
        d: RouteDecision = structured(
            RouteDecision, prompts.ROUTER.format(**ctx, question=state["question"])
        )
        return {"route": d.route, **step("router", f"{d.route.upper()}: {d.reasoning}")}

    # ---------- (3) Retrieve from Pinecone ----------
    def retrieve_kb(state: AgentState):
        docs = deps.retrieve(state["current_query"])
        best = max((d.metadata.get("score", 0) for d in docs), default=0)
        return {"kb_docs": docs, **step("retrieve_kb", f"{len(docs)} chunks, best score {best:.2f}")}

    # ---------- (4) Grade KB evidence ----------
    def grade_kb(state: AgentState):
        docs = state.get("kb_docs", [])
        if not docs:
            return {"kb_grade": "weak", **step("grade_kb", "WEAK: no chunks retrieved")}
        best = max(d.metadata.get("score", 1.0) for d in docs)
        if s.min_relevance_score and best < s.min_relevance_score:
            # Cheap shortcut: skip an LLM call when vectors are clearly off-topic
            return {"kb_grade": "weak", **step("grade_kb", f"WEAK: best score {best:.2f} below threshold")}
        g: EvidenceGrade = structured(
            EvidenceGrade,
            prompts.GRADE_KB.format(question=state["question"], context=format_kb(docs)),
        )
        return {"kb_grade": g.grade, **step("grade_kb", f"{g.grade.upper()}: {g.reasoning}")}

    # ---------- (5) Web search ----------
    def web_search(state: AgentState):
        try:
            results = deps.web_search(state["current_query"])
        except Exception as exc:  # web is a fallback; never crash the request
            logger.warning("Web search failed: %s", exc)
            results = []
        return {"web_results": results, **step("web_search", f"{len(results)} results")}

    # ---------- (6) Grade web evidence ----------
    def grade_web(state: AgentState):
        results = state.get("web_results", [])
        if not results:
            return {"web_grade": "weak", **step("grade_web", "WEAK: no web results")}
        g: EvidenceGrade = structured(
            EvidenceGrade,
            prompts.GRADE_WEB.format(question=state["question"], context=format_web(results)),
        )
        return {"web_grade": g.grade, **step("grade_web", f"{g.grade.upper()}: {g.reasoning}")}

    # ---------- Retry loop: rewrite the query ----------
    def rewrite_query(state: AgentState):
        r: RewrittenQuery = structured(
            RewrittenQuery,
            prompts.REWRITE.format(question=state["question"], current_query=state["current_query"]),
        )
        return {
            "current_query": r.query,
            "retry_count": state.get("retry_count", 0) + 1,
            **step("rewrite_query", f"retry {state.get('retry_count', 0) + 1}: {r.query}"),
        }

    # ---------- (7) Answer from KB ----------
    def generate_from_kb(state: AgentState):
        docs = state["kb_docs"]
        answer = text(prompts.GENERATE_KB.format(**ctx, question=state["question"], context=format_kb(docs)))
        citations = [
            {
                "n": i,
                "title": d.metadata.get("source", "Company KB"),
                "detail": d.metadata.get("section") or (f"page {d.metadata['page']}" if d.metadata.get("page") else ""),
                "url": "",
                "type": "private_kb",
                "score": d.metadata.get("score"),
            }
            for i, d in enumerate(docs, 1)
        ]
        return {
            "answer": answer, "source_used": "private_kb", "confidence": "high",
            "citations": citations, **step("generate_from_kb", "answered from company KB"),
        }

    # ---------- (8) Answer from web ----------
    def generate_from_web(state: AgentState):
        results = state["web_results"]
        answer = text(prompts.GENERATE_WEB.format(**ctx, question=state["question"], context=format_web(results)))
        citations = [
            {"n": i, "title": r["title"], "detail": "", "url": r["url"], "type": "web"}
            for i, r in enumerate(results, 1)
        ]
        return {
            "answer": answer, "source_used": "web", "confidence": "medium",
            "citations": citations, **step("generate_from_web", "answered from web search"),
        }

    # ---------- (9) Fallback ----------
    def fallback_answer(state: AgentState):
        partial = format_kb(state.get("kb_docs", [])[:2]) + "\n\n" + format_web(state.get("web_results", [])[:2])
        answer = text(prompts.FALLBACK.format(**ctx, question=state["question"], context=partial.strip() or "none"))
        return {
            "answer": answer, "source_used": "fallback", "confidence": "low",
            "citations": [], **step("fallback_answer", "no reliable evidence; answered with stated limitation"),
        }

    # ---------- Direct (small talk) ----------
    def direct_answer(state: AgentState):
        answer = text(prompts.DIRECT.format(**ctx, question=state["question"]))
        return {
            "answer": answer, "source_used": "direct", "confidence": "n/a",
            "citations": [], **step("direct_answer", "no retrieval needed"),
        }

    # ---------- Conditional edges (pure functions of state) ----------
    def after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
        return "retrieve_kb" if state["route"] == "kb" else "direct_answer"

    def after_grade_kb(state: AgentState) -> Literal["generate_from_kb", "web_search"]:
        return "generate_from_kb" if state["kb_grade"] == "good" else "web_search"

    def after_grade_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "fallback_answer"]:
        if state["web_grade"] == "good":
            return "generate_from_web"
        if state.get("retry_count", 0) < s.max_retries:
            return "rewrite_query"
        return "fallback_answer"

    g = StateGraph(AgentState)
    for fn in (router, retrieve_kb, grade_kb, web_search, grade_web, rewrite_query,
               generate_from_kb, generate_from_web, fallback_answer, direct_answer):
        g.add_node(fn.__name__, fn)

    g.add_edge(START, "router")
    g.add_conditional_edges("router", after_router)
    g.add_edge("retrieve_kb", "grade_kb")
    g.add_conditional_edges("grade_kb", after_grade_kb)
    g.add_edge("web_search", "grade_web")
    g.add_conditional_edges("grade_web", after_grade_web)
    g.add_edge("rewrite_query", "retrieve_kb")
    for terminal in ("generate_from_kb", "generate_from_web", "fallback_answer", "direct_answer"):
        g.add_edge(terminal, END)
    return g.compile()


def initial_state(question: str) -> AgentState:
    return {
        "question": question, "current_query": question, "retry_count": 0,
        "kb_docs": [], "web_results": [], "citations": [], "trace": [],
    }


@lru_cache
def get_graph():
    """The production graph with real clients (built once, on first request)."""
    from app.rag.llm import get_llm, web_search
    from app.rag.vectorstore import search

    return build_graph(Deps(llm=get_llm(), retrieve=search, web_search=web_search, settings=get_settings()))


def ask(question: str, graph=None) -> AgentState:
    return (graph or get_graph()).invoke(initial_state(question))
