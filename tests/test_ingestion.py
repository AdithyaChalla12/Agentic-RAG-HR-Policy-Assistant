from pathlib import Path

from app.services.ingestion import chunk_documents, doc_slug, load_file

KB = Path(__file__).resolve().parents[1] / "data" / "sample_kb"


def test_markdown_chunks_carry_section_and_stable_ids():
    path = KB / "leave_and_time_off_policy.md"
    first = chunk_documents(load_file(path))
    second = chunk_documents(load_file(path))
    assert len(first) > 3
    assert [c.id for c in first] == [c.id for c in second]          # deterministic
    assert first[0].id == "leave-and-time-off-policy-md#0"
    sections = {c.metadata["section"] for c in first}
    assert any("Parental leave" in s for s in sections)
    assert all(len(c.page_content) <= 900 for c in first)


def test_ids_are_numbered_per_file():
    docs = [d for p in sorted(KB.glob("*.md")) for d in load_file(p)]
    chunks = chunk_documents(docs)
    assert len({c.id for c in chunks}) == len(chunks)
    assert doc_slug("Leave Policy v2.PDF") == "leave-policy-v2-pdf"
