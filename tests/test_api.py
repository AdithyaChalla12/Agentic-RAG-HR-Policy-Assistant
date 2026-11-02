import pytest
from fastapi.testclient import TestClient

from app.core.config import get_settings


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(get_settings(), "audit_db_path", str(tmp_path / "audit.db"))
    from app.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    assert client.get("/api/health").json()["status"] == "ok"


def test_chat_returns_trace_and_writes_audit(client, monkeypatch):
    fake = {
        "answer": "5 days", "source_used": "private_kb", "confidence": "high",
        "current_query": "q", "retry_count": 0, "citations": [],
        "trace": [{"node": "router", "detail": "KB"}],
    }
    monkeypatch.setattr("app.rag.workflow.ask", lambda q: fake)
    r = client.post("/api/chat", json={"question": "PTO carryover?"})
    assert r.status_code == 200 and r.json()["answer"] == "5 days"

    key = get_settings().admin_api_key
    log = client.get("/api/audit", headers={"X-Admin-Key": key}).json()
    assert log["stats"]["private_kb"]["count"] == 1


def test_admin_routes_need_key(client):
    assert client.get("/api/audit").status_code == 401
    files = {"file": ("x.exe", b"hi")}
    assert client.post("/api/ingest", files=files).status_code == 401
