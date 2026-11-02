"""Embeddings + Pinecone. Lazily created so the app starts even before keys are set."""
import logging
import time
from functools import lru_cache

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec

from app.core.config import get_settings
from app.services.ingestion import doc_slug

logger = logging.getLogger(__name__)

EMBEDDING_DIMENSIONS = {
    "sentence-transformers/all-minilm-l6-v2": 384,
    "all-minilm-l6-v2": 384,
    "text-embedding-3-small": 1536,
    "text-embedding-3-large": 3072,
}


def embedding_dimension() -> int:
    name = get_settings().embedding_model.lower()
    if name not in EMBEDDING_DIMENSIONS:
        raise ValueError(f"Add the dimension for '{name}' to EMBEDDING_DIMENSIONS")
    return EMBEDDING_DIMENSIONS[name]


@lru_cache
def get_embeddings():
    s = get_settings()
    if s.embedding_provider == "huggingface":
        from langchain_huggingface import HuggingFaceEmbeddings

        # normalize so cosine scores are comparable across queries
        return HuggingFaceEmbeddings(
            model_name=s.embedding_model, encode_kwargs={"normalize_embeddings": True}
        )
    from langchain_openai import OpenAIEmbeddings

    if not s.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is missing")
    return OpenAIEmbeddings(model=s.embedding_model, api_key=s.openai_api_key)


@lru_cache
def get_index():
    """Create the serverless index on first use. Refuses to touch a mismatched one."""
    s = get_settings()
    if not s.pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is missing")
    pc = Pinecone(api_key=s.pinecone_api_key)
    dim = embedding_dimension()

    if not pc.has_index(s.pinecone_index_name):
        logger.info("Creating Pinecone index %s (dim=%s)", s.pinecone_index_name, dim)
        pc.create_index(
            name=s.pinecone_index_name,
            dimension=dim,
            metric="cosine",
            spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
        )
        while not pc.describe_index(s.pinecone_index_name).status["ready"]:
            time.sleep(1)
    else:
        existing = pc.describe_index(s.pinecone_index_name).dimension
        if existing != dim:
            # Never auto-delete user data. Fail loudly with the fix.
            raise RuntimeError(
                f"Index '{s.pinecone_index_name}' has dimension {existing} but "
                f"'{s.embedding_model}' produces {dim}. Use a new PINECONE_INDEX_NAME."
            )
    return pc.Index(s.pinecone_index_name)


@lru_cache
def get_vectorstore():
    from langchain_pinecone import PineconeVectorStore

    return PineconeVectorStore(
        index=get_index(),
        embedding=get_embeddings(),
        namespace=get_settings().pinecone_namespace,
    )


def search(query: str, k: int | None = None) -> list[Document]:
    """Top-k semantic search. The similarity score is copied into metadata['score']."""
    k = k or get_settings().top_k
    hits = get_vectorstore().similarity_search_with_score(query, k=k)
    docs = []
    for doc, score in hits:
        doc.metadata["score"] = round(float(score), 4)
        docs.append(doc)
    return docs


def delete_source(source: str) -> int:
    """Remove every chunk of one file (ids share the '<slug>#' prefix)."""
    index, ns = get_index(), get_settings().pinecone_namespace
    removed = 0
    for id_batch in index.list(prefix=f"{doc_slug(source)}#", namespace=ns):
        if id_batch:
            index.delete(ids=list(id_batch), namespace=ns)
            removed += len(id_batch)
    return removed


def upsert_chunks(chunks: list[Document]) -> int:
    """Replace each file's old chunks, then add the new ones under deterministic ids."""
    for source in {c.metadata["source"] for c in chunks}:
        delete_source(source)
    ids = get_vectorstore().add_documents(chunks, ids=[c.id for c in chunks])
    return len(ids)
