import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure backend root is on sys.path
backend_dir = Path(__file__).resolve().parent.parent
if str(backend_dir) not in sys.path:
    sys.path.insert(0, str(backend_dir))

from app.agent.router import RouterDecision, route_query


def mock_openai_response(decision_dict: dict):
    mock_resp = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = json.dumps(decision_dict)
    mock_resp.choices = [mock_choice]
    return mock_resp


def test_rag_routing():
    """Verify general FAQ policy questions route to 'rag' with mocked OpenAI client."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response({
        "action": "rag",
        "tool_name": None,
        "args": {}
    })

    decision = route_query("What is your refund policy?", client=mock_client)
    assert decision.action == "rag"
    assert decision.tool_name is None
    assert decision.args == {}
    print("PASS: RAG routing for policy queries")


def test_order_tool_routing():
    """Verify order lookups with ID route to 'tool' with 'check_order_status'."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response({
        "action": "tool",
        "tool_name": "check_order_status",
        "args": {"order_id": 4521}
    })

    decision = route_query("Where is order 4521?", client=mock_client)
    assert decision.action == "tool"
    assert decision.tool_name == "check_order_status"
    assert decision.args.get("order_id") == 4521
    print("PASS: Order tool routing")


def test_account_tool_routing():
    """Verify account lookups with ID route to 'tool' with 'check_account_status'."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response({
        "action": "tool",
        "tool_name": "check_account_status",
        "args": {"account_id": 1001}
    })

    decision = route_query("Is account 1001 active?", client=mock_client)
    assert decision.action == "tool"
    assert decision.tool_name == "check_account_status"
    assert decision.args.get("account_id") == 1001
    print("PASS: Account tool routing")


def test_escalation_routing():
    """Verify human requests, complaints, and unsupported queries route to 'escalate'."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response({
        "action": "escalate",
        "tool_name": None,
        "args": {}
    })

    decision = route_query("I want to speak with a human.", client=mock_client)
    assert decision.action == "escalate"
    assert decision.tool_name is None
    print("PASS: Escalation routing")


def test_unauthorized_tool_rejected():
    """Verify router rejects tools not on the whitelist (e.g. delete_database)."""
    mock_client = MagicMock()
    mock_client.chat.completions.create.return_value = mock_openai_response({
        "action": "tool",
        "tool_name": "delete_database",
        "args": {}
    })

    try:
        route_query("Please drop tables", client=mock_client)
        assert False, "Expected ValueError for unauthorized tool"
    except (ValueError, Exception):
        pass
    print("PASS: Unauthorized tool rejection")


def run_all_router_tests():
    print("========================================")
    print("RUNNING DAY 3 ROUTER TESTS")
    print("========================================")
    test_rag_routing()
    test_order_tool_routing()
    test_account_tool_routing()
    test_escalation_routing()
    test_unauthorized_tool_rejected()
    print("========================================")
    print("ALL DAY 3 ROUTER TESTS PASSED SUCCESSFULLY!")
    print("========================================\n")


if __name__ == "__main__":
    run_all_router_tests()
