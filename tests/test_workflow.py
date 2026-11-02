"""Every branch of the diagram, run offline with fakes."""
from langchain_core.documents import Document

from app.core.config import Settings
from app.rag.state import EvidenceGrade, RewrittenQuery, RouteDecision
from app.rag.workflow import Deps, ask, build_graph
from tests.fakes import FakeLLM

KB_DOC = Document(
    page_content="Unused PTO carries over up to a maximum of 5 days.",
    metadata={"source": "leave_and_time_off_policy.md", "section": "Paid time off (PTO)", "score": 0.71},
)
WEB = [{"title": "IRS limits", "url": "https://example.com/irs", "content": "The limit is ..."}]


def route(r):
    return RouteDecision(route=r, reasoning="test")


def grade(g):
    return EvidenceGrade(grade=g, reasoning="test")


def run(llm, retrieve=lambda q: [KB_DOC], web=lambda q: WEB, **settings):
    deps = Deps(llm=llm, retrieve=retrieve, web_search=web, settings=Settings(_env_file=None, **settings))
    result = ask("question?", graph=build_graph(deps))
    return result, [s["node"] for s in result["trace"]]


def test_kb_good_answers_from_kb_and_skips_web():
    web_calls = []
    result, path = run(FakeLLM([route("kb"), grade("good")]), web=lambda q: web_calls.append(q) or WEB)
    assert path == ["router", "retrieve_kb", "grade_kb", "generate_from_kb"]
    assert result["source_used"] == "private_kb" and result["confidence"] == "high"
    assert result["citations"][0]["title"] == "leave_and_time_off_policy.md"
    assert web_calls == []


def test_kb_weak_falls_back_to_web():
    result, path = run(FakeLLM([route("kb"), grade("weak"), grade("good")]))
    assert path == ["router", "retrieve_kb", "grade_kb", "web_search", "grade_web", "generate_from_web"]
    assert result["source_used"] == "web"
    assert result["citations"][0]["url"] == "https://example.com/irs"


def test_both_weak_rewrites_once_then_fallback():
    llm = FakeLLM([
        route("kb"), grade("weak"), grade("weak"),          # first pass
        RewrittenQuery(query="better query"),               # retry
        grade("weak"), grade("weak"),                       # second pass
    ])
    queries = []
    result, path = run(llm, retrieve=lambda q: queries.append(q) or [KB_DOC], max_retries=1)
    assert path == [
        "router", "retrieve_kb", "grade_kb", "web_search", "grade_web",
        "rewrite_query", "retrieve_kb", "grade_kb", "web_search", "grade_web", "fallback_answer",
    ]
    assert queries == ["question?", "better query"]
    assert result["source_used"] == "fallback" and result["retry_count"] == 1


def test_rewrite_can_rescue_from_kb():
    llm = FakeLLM([route("kb"), grade("weak"), grade("weak"), RewrittenQuery(query="pto carryover"), grade("good")])
    result, path = run(llm)
    assert path[-1] == "generate_from_kb" and result["current_query"] == "pto carryover"


def test_direct_route_skips_retrieval():
    calls = []
    result, path = run(FakeLLM([route("direct")]), retrieve=lambda q: calls.append(q) or [])
    assert path == ["router", "direct_answer"] and calls == []
    assert result["source_used"] == "direct"


def test_empty_retrieval_and_low_score_skip_the_llm_grader():
    # empty KB -> weak without an LLM call
    llm = FakeLLM([route("kb"), grade("good")])
    result, _ = run(llm, retrieve=lambda q: [])
    assert result["source_used"] == "web"
    # score below threshold -> weak without an LLM call
    low = Document(page_content="x", metadata={"source": "a.md", "score": 0.05})
    llm = FakeLLM([route("kb"), grade("good")])
    result, _ = run(llm, retrieve=lambda q: [low], min_relevance_score=0.3)
    assert result["source_used"] == "web"


def test_web_failure_does_not_crash():
    def boom(q):
        raise TimeoutError("tavily down")

    result, path = run(FakeLLM([route("kb"), grade("weak")]), web=boom, max_retries=0)
    assert path[-1] == "fallback_answer"
