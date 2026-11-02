"""LLM + web-search clients. Swapping providers is a .env change, not a code change."""
from functools import lru_cache

from app.core.config import get_settings


@lru_cache
def get_llm():
    s = get_settings()
    if s.llm_provider == "groq":
        from langchain_groq import ChatGroq

        if not s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        return ChatGroq(model=s.llm_model, temperature=0, api_key=s.groq_api_key)

    from langchain_openai import ChatOpenAI

    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return ChatOpenAI(model=s.llm_model, temperature=0, api_key=s.openai_api_key)


@lru_cache
def _tavily():
    from langchain_tavily import TavilySearch

    s = get_settings()
    if not s.tavily_api_key:
        raise RuntimeError("TAVILY_API_KEY is missing")
    return TavilySearch(tavily_api_key=s.tavily_api_key, max_results=5, topic="general")


def web_search(query: str) -> list[dict]:
    """Return a clean list of {title, url, content} from Tavily."""
    raw = _tavily().invoke({"query": query})
    results = raw.get("results", []) if isinstance(raw, dict) else []
    return [
        {"title": r.get("title") or r.get("url", ""), "url": r.get("url", ""), "content": r.get("content", "")}
        for r in results
        if r.get("content")
    ]
