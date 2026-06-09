import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app import config
from app.generator import DemoAnswerGenerator
from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "DATA_DIR", tmp_path / "tenants")
    import app.storage as storage
    import app.main as main_module

    monkeypatch.setattr(storage, "DATA_DIR", tmp_path / "tenants")
    monkeypatch.setattr(main_module, "generator", DemoAnswerGenerator())
    yield TestClient(app)


def upload_text(client: TestClient, tenant_id: str, filename: str, text: str):
    return client.post(
        f"/tenants/{tenant_id}/files",
        params={"chunk_size": 120, "chunk_overlap": 20},
        files={"file": (filename, text.encode("utf-8"), "text/plain")},
    )


def test_index_page_contains_qa_ui(client: TestClient):
    response = client.get("/")

    assert response.status_code == 200
    assert "多租户 AI 问答演示" in response.text
    assert "/qa/stream" in response.text


def test_ingestion_status_and_source_lines(client: TestClient):
    response = upload_text(
        client,
        "tenant_a",
        "policy.txt",
        "Alpha SLA is 99.9 percent monthly uptime.\nEscalation is required within 15 minutes.",
    )
    assert response.status_code == 200
    job = response.json()
    assert job["status"] == "done"
    assert job["chunks_created"] >= 1

    status = client.get(f"/tenants/tenant_a/ingestions/{job['id']}")
    assert status.status_code == 200
    assert status.json()["status"] == "done"

    search = client.post("/tenants/tenant_a/search", json={"query": "99.9 uptime SLA", "top_k": 2})
    assert search.status_code == 200
    hit = search.json()["hits"][0]
    assert hit["source"]["filename"] == "policy.txt"
    assert hit["source"]["start_line"] == 1
    assert "99.9" in hit["source"]["text"]


def test_tenant_indexes_are_isolated(client: TestClient):
    upload_text(client, "tenant_a", "a.txt", "Tenant A secret launch codename is ORCHID.")
    upload_text(client, "tenant_b", "b.txt", "Tenant B retention policy is seven years.")

    own = client.post("/tenants/tenant_a/search", json={"query": "ORCHID", "top_k": 3})
    assert own.status_code == 200
    assert any("ORCHID" in hit["source"]["text"] for hit in own.json()["hits"])

    other = client.post("/tenants/tenant_b/search", json={"query": "ORCHID", "top_k": 3})
    assert other.status_code == 200
    assert all("ORCHID" not in hit["source"]["text"] for hit in other.json()["hits"])


def test_streaming_qa_returns_tokens_and_metrics(client: TestClient):
    upload_text(client, "tenant_a", "sla.txt", "Tenant A SLA is 99.9 percent. Incidents escalate in 15 minutes.")

    with client.stream(
        "POST",
        "/tenants/tenant_a/qa/stream",
        json={"question": "What is Tenant A SLA?", "top_k": 2},
    ) as response:
        assert response.status_code == 200
        body = "".join(response.iter_text())

    assert "event: token" in body
    assert "event: metrics" in body
    metrics_payload = body.split("event: metrics\ndata: ", 1)[1].strip()
    metrics = json.loads(metrics_payload)
    assert metrics["token_usage"] > 0
    assert metrics["chunks_used"] >= 1
    assert metrics["retrieval_ms"] >= 0
    assert metrics["generation_ms"] >= 0
    assert metrics["model"] == "local-demo-generator"


def test_generator_uses_deepseek_model_when_key_is_present(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test-key")
    monkeypatch.setenv("DEEPSEEK_MODEL", "deepseek-v4-flash")

    from app.generator import FallbackAnswerGenerator

    generator = FallbackAnswerGenerator()

    assert generator.model_name == "deepseek-v4-flash"


def test_generator_ignores_placeholder_deepseek_key(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "sk-your-deepseek-api-key")
    monkeypatch.setenv("QA_GENERATOR", "deepseek")

    from app.generator import FallbackAnswerGenerator

    generator = FallbackAnswerGenerator()

    assert generator.model_name == "local-demo-generator"


def test_quota_error_is_clear(client: TestClient, monkeypatch):
    monkeypatch.setitem(config.TENANT_CONFIGS, "tenant_c", config.TenantConfig(model="tiny", daily_token_quota=1))
    with client.stream(
        "POST",
        "/tenants/tenant_c/qa/stream",
        json={"question": "This request will exceed quota", "top_k": 1},
    ) as response:
        body = "".join(response.iter_text())

    assert "event: error" in body
    assert "daily token quota exceeded" in body


def test_rejects_unknown_tenant(client: TestClient):
    response = client.post("/tenants/tenant_x/search", json={"query": "hello", "top_k": 1})
    assert response.status_code == 404


def test_hybrid_search_uses_keyword_and_vector_paths(client: TestClient):
    upload_text(
        client,
        "tenant_a",
        "hybrid.txt",
        "The enterprise answer mentions zero trust networking and private vector indexes.",
    )
    response = client.post(
        "/tenants/tenant_a/search",
        json={"query": "private vector indexes", "top_k": 1},
    )
    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert hit["keyword_rank"] is not None
    assert hit["vector_rank"] is not None


def test_chinese_quota_question_prefers_answer_chunk(client: TestClient):
    upload_text(
        client,
        "tenant_a",
        "quota.txt",
        (
            "四、访问配额\n"
            "Tenant A 每日默认 token 配额为 2000。超过配额后，系统必须返回清晰错误信息。\n\n"
            "七、推荐测试问题\n"
            "Tenant A 的 token 配额是多少？"
        ),
    )

    response = client.post(
        "/tenants/tenant_a/search",
        json={"query": "Tenant A 的 token 配额是多少？", "top_k": 1},
    )

    assert response.status_code == 200
    hit = response.json()["hits"][0]
    assert "2000" in hit["source"]["text"]
