import os

os.environ["EMBEDDING_PROVIDER"] = "local"
os.environ["ENABLE_OPENAI_ROUTER"] = "false"

from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200


def test_health():

    response = client.get("/api/health")

    assert response.status_code == 200

    assert response.json()["status"] == "healthy"


def test_order_chat():

    response = client.post(
        "/api/chat",
        json={
            "session_id": "day5-order",
            "message": "Where is order 4521?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["action"] == "tool"

    assert data["tool_result"]["success"] is True


def test_account_chat():

    response = client.post(
        "/api/chat",
        json={
            "session_id": "day5-account",
            "message": "What plan is account 1001 on?",
        },
    )

    assert response.status_code == 200

    assert response.json()["action"] == "tool"


def test_rag_chat():

    response = client.post(
        "/api/chat",
        json={
            "session_id": "day5-rag",
            "message": "What is your refund policy?",
        },
    )

    assert response.status_code == 200

    data = response.json()

    assert data["action"] == "rag"

    assert len(data["retrieved_documents"]) > 0


def test_escalation():

    response = client.post(
        "/api/chat",
        json={
            "session_id": "day5-escalation",
            "message": "I want to speak with a human.",
        },
    )

    assert response.status_code == 200

    assert response.json()["action"] == "escalate"


def test_memory():

    session_id = "day5-memory"

    client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "Where is order 4521?",
        },
    )

    response = client.get(
        f"/api/sessions/{session_id}"
    )

    assert response.status_code == 200

    assert len(response.json()["messages"]) >= 2


def test_clear_memory():

    session_id = "day5-clear"

    client.post(
        "/api/chat",
        json={
            "session_id": session_id,
            "message": "Where is order 4521?",
        },
    )

    response = client.delete(
        f"/api/sessions/{session_id}"
    )

    assert response.status_code == 200


def run_all_day5_tests():

    tests = [
        ("Root endpoint", test_root),
        ("Health endpoint", test_health),
        ("Order chat API", test_order_chat),
        ("Account chat API", test_account_chat),
        ("RAG chat API", test_rag_chat),
        ("Escalation API", test_escalation),
        ("Conversation memory", test_memory),
        ("Memory clearing", test_clear_memory),
    ]

    for name, test in tests:

        try:

            test()

            print(f"PASS: {name}")

        except Exception as exc:

            print(f"FAIL: {name}")

            raise exc

    print("========================================")
    print("ALL DAY 5 API & MEMORY TESTS PASSED!")
    print("========================================")