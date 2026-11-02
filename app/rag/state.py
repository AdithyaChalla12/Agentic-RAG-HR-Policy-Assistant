"""Shared graph state + the structured outputs the LLM must return."""
import operator
from typing import Annotated, Literal

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ---------- Structured LLM outputs (the "decisions") ----------
class RouteDecision(BaseModel):
    """Step 2: should this question hit the knowledge base?"""

    reasoning: str = Field(description="One short sentence explaining the choice")
    route: Literal["kb", "direct"] = Field(
        description="kb = needs company knowledge; direct = greeting, thanks, or small talk"
    )


class EvidenceGrade(BaseModel):
    """Steps 4 and 6: is the retrieved evidence enough to answer?"""

    reasoning: str = Field(description="One short sentence: what the evidence covers or misses")
    grade: Literal["good", "weak"] = Field(
        description="good = sufficient to answer specifically; weak = missing, partial, or off-topic"
    )


class RewrittenQuery(BaseModel):
    """Retry loop: a better search query."""

    query: str = Field(description="A keyword-rich search query that preserves the user's intent")


# ---------- Graph state ----------
class TraceStep(TypedDict):
    node: str
    detail: str


class AgentState(TypedDict, total=False):
    question: str            # original user question (never changes)
    current_query: str       # what we actually search with (may be rewritten)
    route: str               # "kb" | "direct"
    kb_docs: list[Document]
    kb_grade: str            # "good" | "weak"
    web_results: list[dict]  # [{title, url, content}]
    web_grade: str
    retry_count: int
    answer: str
    source_used: str         # private_kb | web | fallback | direct
    confidence: str          # high | medium | low | n/a
    citations: list[dict]
    # Reducer: every node APPENDS its step instead of overwriting the list
    trace: Annotated[list[TraceStep], operator.add]
