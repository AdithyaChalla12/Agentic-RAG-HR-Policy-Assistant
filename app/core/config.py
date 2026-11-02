"""Central settings: every tunable lives here and is read from .env."""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=str(BASE_DIR / ".env"), extra="ignore")

    # App
    app_env: str = "development"
    assistant_name: str = "People Ops Copilot"
    domain_description: str = (
        "HR policies, benefits, leave, payroll, onboarding, expenses, travel, "
        "and workplace conduct at Lumen Labs"
    )

    # LLM
    llm_provider: Literal["groq", "openai"] = "groq"
    llm_model: str = "openai/gpt-oss-20b"
    groq_api_key: str = ""
    openai_api_key: str = ""

    # Embeddings
    embedding_provider: Literal["huggingface", "openai"] = "huggingface"
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    # Pinecone
    pinecone_api_key: str = ""
    pinecone_index_name: str = "agentic-rag-copilot"
    pinecone_namespace: str = "company-kb"
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"

    # Web search
    tavily_api_key: str = ""

    # Agent behavior
    top_k: int = 4
    min_relevance_score: float = 0.0
    max_retries: int = 1

    # Security + storage
    admin_api_key: str = "change-me"
    audit_db_path: str = str(BASE_DIR / "data" / "audit.db")
    upload_dir: str = str(BASE_DIR / "uploads")
    sample_kb_dir: str = str(BASE_DIR / "data" / "sample_kb")


@lru_cache
def get_settings() -> Settings:
    return Settings()
