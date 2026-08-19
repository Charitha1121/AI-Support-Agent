import sys
from pathlib import Path

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.database.db import create_tables, seed_data
from app.main import app
from app.schemas.chat import ChatRequest
from app.services.chat_service import process_chat_message


def setup_module():
    create_tables()
    seed_data()


def test_health_check():
    from app.main import health_check
    res = health_check()
    assert res.status == "healthy"
    assert "NovaTech" in res.service
    print("PASS: API /health endpoint")


def test_chat_rag_endpoint():
    req = ChatRequest(message="What is your refund policy?", conversation_id="test-api-rag")
    res = process_chat_message(req)
    assert res.action_taken == "rag"
    assert res.conversation_id == "test-api-rag"
    assert "14" in res.response or "refund" in res.response.lower()
    print(f"PASS: API RAG request (Action: {res.action_taken})")


def test_chat_order_tool_endpoint():
    req = ChatRequest(message="Where is order 4521?", conversation_id="test-api-order")
    res = process_chat_message(req)
    assert res.action_taken == "tool"
    assert res.tool_name == "check_order_status"
    assert "Rahul" in res.response or "Shipped" in res.response or "4521" in res.response
    print(f"PASS: API Order Tool request (Action: {res.action_taken}, Tool: {res.tool_name})")


def test_chat_account_tool_endpoint():
    req = ChatRequest(message="Is account 1001 active?", conversation_id="test-api-account")
    res = process_chat_message(req)
    assert res.action_taken == "tool"
    assert res.tool_name == "check_account_status"
    assert "Pro" in res.response or "Active" in res.response or "1001" in res.response
    print(f"PASS: API Account Tool request (Action: {res.action_taken}, Tool: {res.tool_name})")


def test_chat_escalation_endpoint():
    req = ChatRequest(message="I want to speak with a human representative immediately.", conversation_id="test-api-esc")
    res = process_chat_message(req)
    assert res.action_taken == "escalate"
    assert res.escalation_id is not None
    assert res.escalation_id.startswith("ESC-")
    print(f"PASS: API Escalation request (Action: {res.action_taken}, Ticket: {res.escalation_id})")


def test_chat_multiturn_continuity():
    conv_id = "test-api-multiturn"
    req1 = ChatRequest(message="Where is order 4521?", conversation_id=conv_id)
    res1 = process_chat_message(req1)
    assert res1.action_taken == "tool"

    req2 = ChatRequest(message="What about 4522?", conversation_id=conv_id)
    res2 = process_chat_message(req2)
    assert res2.action_taken == "tool"
    assert "4522" in res2.response or "Priya" in res2.response or "Processing" in res2.response
    print("PASS: API Multi-turn conversation continuity")


def run_all_api_tests():
    print("========================================")
    print("RUNNING DAY 7 FASTAPI INTEGRATION TESTS")
    print("========================================")
    setup_module()
    test_health_check()
    test_chat_rag_endpoint()
    test_chat_order_tool_endpoint()
    test_chat_account_tool_endpoint()
    test_chat_escalation_endpoint()
    test_chat_multiturn_continuity()
    print("========================================")
    print("ALL DAY 7 FASTAPI TESTS PASSED SUCCESSFULLY!")
    print("========================================\n")


if __name__ == "__main__":
    run_all_api_tests()
